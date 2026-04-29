# sae_models.py
# PyTorch SAE variants for UNI / UNI2-h pathology features:
# 1) Baseline ReLU SAE (+ optional L1 sparsity penalty)
# 2) k-sparse TopK SAE (Makhzani & Frey style hard L0 control)
#
# Usage (minimal):
#   from sae_models import ReLUSparseSAE, TopKSAE, train_sae
#   model = TopKSAE(d_in=1024, d_latent=256, k=16)
#   train_sae(model, dataloader, device="cuda")
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Utilities
# --------------------------

def topk_activation(
    a: torch.Tensor,
    k: int,
    mode: Literal["value", "magnitude"] = "value",
    nonneg: bool = True,
    return_topk: bool = False,
):
    """
    TopK activation: keep only k entries per sample, zero the rest.

    Args:
        a: (B, d_latent) pre-activations
        k: number of active latents per sample
        mode:
          - "value": keep k largest values
          - "magnitude": keep k largest |a| (preserve sign unless nonneg=True)
        nonneg: if True, apply ReLU before selection and output is nonnegative
        return_topk: if True, also return topk values and indices

    Returns:
        If return_topk=False:
            z: (B, d_latent) sparse activations with exactly k non-zeros per row
        If return_topk=True:
            (z, topk_val, topk_idx) where topk_val and topk_idx are (B, k)
    """
    

    if nonneg:
        a = F.relu(a)

    if mode == "magnitude" and not nonneg:
        scores = a.abs()
    else:
        scores = a

    _, topk_idx = torch.topk(scores, k, dim=-1, largest=True, sorted=False)

    # gather the actual values to be placed into z
    # (signed if magnitude+not nonneg)
    topk_val = a.gather(dim=-1, index=topk_idx)

    # build dense z without allocating a dense mask
    z = torch.zeros_like(a)
    z.scatter_(dim=-1, index=topk_idx, src=topk_val)

    if return_topk:
        return z, topk_val, topk_idx
    return z


def batch_topk_activation(
    a: torch.Tensor,
    k_per_sample: int,
    mode: Literal["value", "magnitude"] = "value",
    nonneg: bool = True,
    return_topk: bool = False,
):
    """
    Batch-TopK activation.

    Keeps exactly (batch_size * k_per_sample) activations across the entire batch tensor,
    rather than k_per_sample per row. This matches the "batch top-k" sparsity style where
    activity budget is allocated globally per minibatch.

    Args:
        a: (B, d_latent) pre-activations
        k_per_sample: target average active latents per sample
        mode:
          - "value": keep largest values
          - "magnitude": keep largest |a|
        nonneg: if True, apply ReLU before selection
        return_topk: if True, return selected flat indices and values
    """
    if a.ndim != 2:
        raise ValueError(f"batch_topk_activation expects 2D tensor, got shape {tuple(a.shape)}")
    bsz, d_latent = a.shape
    if k_per_sample <= 0:
        raise ValueError("k_per_sample must be > 0")
    total_budget = int(k_per_sample) * int(bsz)
    total_dim = int(bsz * d_latent)
    total_budget = max(1, min(total_budget, total_dim))

    if nonneg:
        a = F.relu(a)

    scores = a.abs() if (mode == "magnitude" and not nonneg) else a
    af = a.reshape(-1)
    sf = scores.reshape(-1)

    topk_val, topk_idx = torch.topk(sf, k=total_budget, dim=0, largest=True, sorted=False)
    kept_val = af.gather(dim=0, index=topk_idx)

    zf = torch.zeros_like(af)
    zf.scatter_(dim=0, index=topk_idx, src=kept_val)
    z = zf.view_as(a)

    if return_topk:
        return z, kept_val, topk_idx
    return z


def l2_recon_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean squared error (per-batch)."""
    return F.mse_loss(x_hat, x, reduction="mean")


def l1_sparsity(z: torch.Tensor) -> torch.Tensor:
    """Mean L1 of activations (per-batch)."""
    return z.abs().mean()


def activation_stats(z: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    """
    Basic stats for monitoring sparsity / distribution.
    z: (B, d_latent)
    """
    with torch.no_grad():
        nz = (z.abs() > 0).float()
        frac_nz = nz.mean().item()
        per_row_nz = nz.sum(dim=-1).float().mean().item()
        mean = z.mean().item()
        std = z.std(unbiased=False).item()
        # margin between top1 and top2 (for interpretability)
        if z.shape[1] >= 2:
            top2 = torch.topk(z, k=2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1]).mean().item()
        else:
            margin = 0.0
        return {
            "frac_nonzero": frac_nz,
            "mean_nonzero_per_sample": per_row_nz,
            "z_mean": mean,
            "z_std": std,
            "top1_top2_margin": margin,
        }


# --------------------------
# Models
# --------------------------

class SAEBase(nn.Module):
    """
    Base SAE interface.
    Implementations should return:
      x_hat: reconstructed input
      z: latent activations (post-nonlinearity)
      a: pre-activations (optional for debugging/TopK)
    """
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class ReLUSparseSAE(SAEBase):
    """
    Baseline SAE: Linear encoder -> ReLU -> Linear decoder.
    Optional L1 sparsity is applied in the training loss (not inside the module).
    Includes a learned pre-bias b_pre as in many SAE implementations: encode(x - b_pre).
    """
    def __init__(
        self,
        d_in: int,
        d_latent: int,
        tied: bool = False,
        use_pre_bias: bool = True,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.tied = tied
        self.use_pre_bias = use_pre_bias

        self.b_pre = nn.Parameter(torch.zeros(d_in)) if use_pre_bias else None
        self.enc = nn.Linear(d_in, d_latent, bias=True)

        if tied:
            # Decoder weight is tied to encoder weight (transpose). Use a separate bias.
            self.dec_bias = nn.Parameter(torch.zeros(d_in))
            self.dec = None
        else:
            self.dec = nn.Linear(d_latent, d_in, bias=True)
            self.dec_bias = None

        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float = 0.02):
        nn.init.kaiming_normal_(self.enc.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)
        if self.dec is not None:
            nn.init.xavier_normal_(self.dec.weight)
            nn.init.zeros_(self.dec.bias)
        if self.b_pre is not None:
            nn.init.zeros_(self.b_pre)
        if self.dec_bias is not None:
            nn.init.zeros_(self.dec_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.tied:
            # x_hat = z @ W_enc + b_dec, where W_dec = W_enc^T
            x_hat = F.linear(z, self.enc.weight.t(), self.dec_bias)
        else:
            x_hat = self.dec(z)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = x - self.b_pre if self.b_pre is not None else x
        a = self.enc(x0)                 # pre-activation
        z = F.relu(a)                    # activation
        x_hat = self.decode(z)
        return x_hat, z, a


class TopKSAE(SAEBase):
    """
    k-sparse TopK SAE:
      z = TopK(W_enc(x - b_pre))
      x_hat = W_dec z (+ bias)

    Notes:
    - If nonneg=True, applies ReLU before TopK selection.
    - mode="value" keeps largest values; mode="magnitude" keeps largest |a|.
    - No L1 penalty needed; reconstruction loss only (per paper excerpt).
    """
    def __init__(
        self,
        d_in: int,
        d_latent: int,
        k: int,
        tied: bool = False,
        use_pre_bias: bool = True,
        topk_mode: Literal["value", "magnitude"] = "value",
        nonneg: bool = True,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.k = k
        self.tied = tied
        self.use_pre_bias = use_pre_bias
        self.topk_mode = topk_mode
        self.nonneg = nonneg

        self.b_pre = nn.Parameter(torch.zeros(d_in)) if use_pre_bias else None
        self.enc = nn.Linear(d_in, d_latent, bias=True)

        if k <= 0:
            raise ValueError("TopKSAE requires k > 0")
        if k >= d_latent:
            raise ValueError("TopKSAE requires k < d_latent")
        
        if tied:
            self.dec_bias = nn.Parameter(torch.zeros(d_in))
            self.dec = None
        else:
            self.dec = nn.Linear(d_latent, d_in, bias=True)
            self.dec_bias = None

        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float = 0.02):
        nn.init.kaiming_normal_(self.enc.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)
        if self.dec is not None:
            nn.init.xavier_normal_(self.dec.weight)
            nn.init.zeros_(self.dec.bias)
        if self.b_pre is not None:
            nn.init.zeros_(self.b_pre)
        if self.dec_bias is not None:
            nn.init.zeros_(self.dec_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.tied:
            x_hat = F.linear(z, self.enc.weight.t(), self.dec_bias)
        else:
            x_hat = self.dec(z)
        return x_hat

    def forward(self, x: torch.Tensor, return_topk: bool = False):
        topk_val, topk_idx = None, None
        x0 = x - self.b_pre if self.b_pre is not None else x
        a = self.enc(x0)
        if return_topk:
            z, topk_val, topk_idx = topk_activation(a, k=self.k, mode=self.topk_mode, nonneg=self.nonneg, return_topk=True)
            x_hat = self.decode(z)
            return x_hat, z, a, topk_val, topk_idx
        else:
            z = topk_activation(a, k=self.k, mode=self.topk_mode, nonneg=self.nonneg, return_topk=False)
        
        x_hat = self.decode(z)
        return x_hat, z, a


class BatchTopKSAE(SAEBase):
    """
    Batch-TopK SAE:
      z = BatchTopK(W_enc(x - b_pre))
      x_hat = W_dec z (+ bias)

    Difference vs TopKSAE:
    - TopKSAE enforces exactly k non-zeros per sample.
    - BatchTopKSAE enforces exactly (B * k) non-zeros across the full minibatch.
      This gives a global sparsity budget and can reallocate capacity across samples.
    """
    def __init__(
        self,
        d_in: int,
        d_latent: int,
        k: int,
        tied: bool = False,
        use_pre_bias: bool = True,
        topk_mode: Literal["value", "magnitude"] = "value",
        nonneg: bool = True,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.k = k
        self.tied = tied
        self.use_pre_bias = use_pre_bias
        self.topk_mode = topk_mode
        self.nonneg = nonneg

        self.b_pre = nn.Parameter(torch.zeros(d_in)) if use_pre_bias else None
        self.enc = nn.Linear(d_in, d_latent, bias=True)

        if k <= 0:
            raise ValueError("BatchTopKSAE requires k > 0")
        if k >= d_latent:
            raise ValueError("BatchTopKSAE requires k < d_latent")

        if tied:
            self.dec_bias = nn.Parameter(torch.zeros(d_in))
            self.dec = None
        else:
            self.dec = nn.Linear(d_latent, d_in, bias=True)
            self.dec_bias = None

        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float = 0.02):
        nn.init.kaiming_normal_(self.enc.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)
        if self.dec is not None:
            nn.init.xavier_normal_(self.dec.weight)
            nn.init.zeros_(self.dec.bias)
        if self.b_pre is not None:
            nn.init.zeros_(self.b_pre)
        if self.dec_bias is not None:
            nn.init.zeros_(self.dec_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.tied:
            x_hat = F.linear(z, self.enc.weight.t(), self.dec_bias)
        else:
            x_hat = self.dec(z)
        return x_hat

    def forward(self, x: torch.Tensor, return_topk: bool = False):
        topk_val, topk_idx = None, None
        x0 = x - self.b_pre if self.b_pre is not None else x
        a = self.enc(x0)
        if return_topk:
            z, topk_val, topk_idx = batch_topk_activation(
                a, k_per_sample=self.k, mode=self.topk_mode, nonneg=self.nonneg, return_topk=True
            )
            x_hat = self.decode(z)
            return x_hat, z, a, topk_val, topk_idx
        z = batch_topk_activation(
            a, k_per_sample=self.k, mode=self.topk_mode, nonneg=self.nonneg, return_topk=False
        )
        x_hat = self.decode(z)
        return x_hat, z, a


class SDFSAE2Level(SAEBase):
    """
    Two-level Sparse Dictionary Factorization SAE (SDF-SAE), aligned with Section 1.1:
    - Level-1: standard sparse autoencoder (Linear -> ReLU -> decode).
    - Level-2: factorize Level-1 decoder dictionary columns:
        w_j ~= U a_j
      where U are high-level prototypes and a_j are sparse coefficient vectors.

    Forward returns Level-1 SAE outputs only; factorization regularization is computed by
    `sdf_factorization_loss(...)` and should be added in the training loop.
    """
    def __init__(
        self,
        d_in: int,
        d_latent: int,
        d_level2: int,
        tied: bool = False,
        use_pre_bias: bool = True,
        coeff_nonneg: bool = True,
        coeff_simplex: bool = False,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.d_level2 = d_level2
        self.tied = tied
        self.use_pre_bias = use_pre_bias
        self.coeff_nonneg = coeff_nonneg
        self.coeff_simplex = coeff_simplex

        if d_level2 <= 0:
            raise ValueError("SDFSAE2Level requires d_level2 > 0")
        if d_level2 >= d_latent:
            raise ValueError("SDFSAE2Level expects d_level2 < d_latent for hierarchy compression")

        self.b_pre = nn.Parameter(torch.zeros(d_in)) if use_pre_bias else None
        self.enc = nn.Linear(d_in, d_latent, bias=True)

        if tied:
            self.dec_bias = nn.Parameter(torch.zeros(d_in))
            self.dec = None
        else:
            self.dec = nn.Linear(d_latent, d_in, bias=True)
            self.dec_bias = None

        # U in R^{d_in x d_level2}
        self.U = nn.Parameter(torch.empty(d_in, d_level2))
        # A_raw in R^{d_latent x d_level2}, row j corresponds to a_j
        self.A_raw = nn.Parameter(torch.empty(d_latent, d_level2))

        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float = 0.02):
        nn.init.kaiming_normal_(self.enc.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)
        if self.dec is not None:
            nn.init.xavier_normal_(self.dec.weight)
            nn.init.zeros_(self.dec.bias)
        if self.b_pre is not None:
            nn.init.zeros_(self.b_pre)
        if self.dec_bias is not None:
            nn.init.zeros_(self.dec_bias)
        nn.init.normal_(self.U, mean=0.0, std=init_scale)
        nn.init.normal_(self.A_raw, mean=0.0, std=init_scale)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.tied:
            return F.linear(z, self.enc.weight.t(), self.dec_bias)
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        x0 = x - self.b_pre if self.b_pre is not None else x
        a = self.enc(x0)
        z = F.relu(a)
        x_hat = self.decode(z)
        return x_hat, z, a

    def decoder_dictionary(self) -> torch.Tensor:
        # Returns W_dec^{(1)} with shape (d_in, d_latent), columns are Level-1 atoms.
        if self.tied:
            return self.enc.weight.t()
        return self.dec.weight

    def coeff_matrix(self) -> torch.Tensor:
        A = self.A_raw
        if self.coeff_simplex:
            return torch.softmax(A, dim=-1)
        if self.coeff_nonneg:
            return F.softplus(A)
        return A

    def parent_assignment(self) -> torch.Tensor:
        # pi(j) = argmax_k A_{j,k}
        return torch.argmax(self.coeff_matrix(), dim=-1)

    def sdf_factorization_loss(
        self,
        z: Optional[torch.Tensor] = None,
        active_only: bool = True,
        active_eps: float = 1e-6,
        lambda_a: float = 1e-4,
        lambda_u: float = 1e-4,
    ) -> Dict[str, torch.Tensor]:
        """
        Implements Eq.(5)-style Level-2 objective:
            mean_j ||w_j - U a_j||^2 + lambda_a * ||a_j||_1 + lambda_u * ||U||_F^2

        If active_only=True and z is provided, applies dynamic masking over active Level-1 atoms
        in current batch (similar to Eq.(8)).
        """
        W = self.decoder_dictionary()  # (d_in, n1)
        A = self.coeff_matrix()        # (n1, n2)
        U = self.U                     # (d_in, n2)

        if active_only and z is not None:
            active_mask = (z.abs() > active_eps).any(dim=0)
            idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                zero = W.sum() * 0.0
                return {
                    "sdf_total": zero,
                    "sdf_recon": zero,
                    "sdf_l1_a": zero,
                    "sdf_l2_u": zero,
                    "sdf_active_atoms": torch.tensor(0.0, device=W.device),
                }
            W_sel = W[:, idx]
            A_sel = A[idx, :]
        else:
            W_sel = W
            A_sel = A
            idx = None

        W_hat = U @ A_sel.t()
        loss_recon = F.mse_loss(W_hat, W_sel, reduction="mean")
        loss_l1_a = A_sel.abs().mean()
        loss_l2_u = U.pow(2).mean()
        total = loss_recon + float(lambda_a) * loss_l1_a + float(lambda_u) * loss_l2_u

        active_count = float(A_sel.shape[0])
        return {
            "sdf_total": total,
            "sdf_recon": loss_recon,
            "sdf_l1_a": loss_l1_a,
            "sdf_l2_u": loss_l2_u,
            "sdf_active_atoms": torch.tensor(active_count, device=W.device),
        }


# --------------------------
# Training
# --------------------------

class InputNormWrapper(torch.nn.Module):
    def __init__(self, d_in: int, base: torch.nn.Module, eps: float = 1e-5):
        super().__init__()
        self.base = base
        self.ln = torch.nn.LayerNorm(d_in, eps=eps)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.ln(x)
        return self.base(x, **kwargs)


# --------------------------
# Inference helpers
# --------------------------

@torch.no_grad()
def encode_only(model: SAEBase, x: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    model.eval().to(device)
    x = x.to(device).float()
    _, z, _ = model(x)
    return z


@torch.no_grad()
def top_tiles_for_concept(
    z: torch.Tensor,
    concept_idx: int,
    top_k: int = 50,
    margin: bool = True,
) -> torch.Tensor:
    """
    Given latent activations z (N, d_latent), return indices of top tiles for a concept.
    If margin=True, uses z_j - max_{k!=j} z_k to prefer monosemantic tiles.
    """
    if margin and z.shape[1] >= 2:
        zj = z[:, concept_idx]
        z_other = z.clone()
        z_other[:, concept_idx] = torch.finfo(z.dtype).min
        m = zj - z_other.max(dim=1).values
        return torch.topk(m, k=min(top_k, z.shape[0]), largest=True).indices
    else:
        return torch.topk(z[:, concept_idx], k=min(top_k, z.shape[0]), largest=True).indices
