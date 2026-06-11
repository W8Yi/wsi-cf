from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttentionScorer(nn.Module):
    """
    Standard (non-gated) attention scorer for MIL.
    Computes one scalar score per instance.
    """
    def __init__(self, d_in: int, d_attn: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_attn, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # (N, 1)


class _GatedAttentionScorer(nn.Module):
    """
    Gated attention scorer for MIL.
    score_i = w^T (tanh(Vh_i) * sigmoid(Uh_i))
    """
    def __init__(self, d_in: int, d_attn: int, dropout: float = 0.0):
        super().__init__()
        self.attn_a = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attn_b = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attn_c = nn.Linear(d_attn, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = self.attn_a(h)
        b = self.attn_b(h)
        return self.attn_c(a * b)  # (N, 1)


class AttentionMIL(nn.Module):
    """
    Attention-based MIL classifier (single-head, non-gated).

    Forward signature is CLAM-compatible:
      returns (logits, y_prob, y_hat, A_raw, results_dict)

    Inputs:
      h: (N, embed_dim) instance features for one bag.

    Optional:
      attention_only=True -> returns A_raw (1, N) before softmax.
      return_features=True -> results_dict["features"] contains pooled bag embedding (1, d_hidden).
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn = _AttentionScorer(hidden_dim, attn_dim, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        h: torch.Tensor,
        label=None,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        h_mid = self.proj(h)                # (N, hidden_dim)
        A_raw = self.attn(h_mid).transpose(1, 0)  # (1, N)
        if attention_only:
            return A_raw

        A = F.softmax(A_raw, dim=1)
        M = torch.mm(A, h_mid)              # (1, hidden_dim)
        logits = self.classifier(M)         # (1, n_classes)
        y_hat = torch.topk(logits, 1, dim=1)[1]
        y_prob = F.softmax(logits, dim=1)

        results: Dict[str, torch.Tensor] = {}
        if return_features:
            results["features"] = M
        return logits, y_prob, y_hat, A_raw, results


class GatedAttentionMIL(nn.Module):
    """
    Attention-based MIL classifier (single-head, gated attention).

    Differences vs AttentionMIL:
    - Uses tanh-sigmoid gated attention scorer.
    - Optional learnable attention temperature for sharper/flatter bag pooling.
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.0,
        learnable_temperature: bool = True,
        init_temperature: float = 1.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn = _GatedAttentionScorer(hidden_dim, attn_dim, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

        init_tau = max(float(init_temperature), 1e-4)
        if learnable_temperature:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))
        else:
            self.register_buffer("log_tau", torch.log(torch.tensor(init_tau)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(min=1e-4)

    def forward(
        self,
        h: torch.Tensor,
        label=None,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        h_mid = self.proj(h)                              # (N, hidden_dim)
        A_raw = self.attn(h_mid).transpose(1, 0)         # (1, N)
        if attention_only:
            return A_raw

        A = F.softmax(A_raw / self.tau, dim=1)
        M = torch.mm(A, h_mid)                            # (1, hidden_dim)
        logits = self.classifier(M)                       # (1, n_classes)
        y_hat = torch.topk(logits, 1, dim=1)[1]
        y_prob = F.softmax(logits, dim=1)

        results: Dict[str, torch.Tensor] = {"attn_temperature": self.tau.detach()}
        if return_features:
            results["features"] = M
        return logits, y_prob, y_hat, A_raw, results


class AttentionMILRegressor(nn.Module):
    """Attention MIL regressor with a bounded slide-level risk score."""
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn = _AttentionScorer(hidden_dim, attn_dim, dropout=dropout)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h: torch.Tensor,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        h_mid = self.proj(h)
        A_raw = self.attn(h_mid).transpose(1, 0)
        if attention_only:
            return A_raw

        A = F.softmax(A_raw, dim=1)
        M = torch.mm(A, h_mid)
        risk_logit = self.regressor(M)
        risk_score = torch.sigmoid(risk_logit)
        results: Dict[str, torch.Tensor] = {"risk_logit": risk_logit}
        if return_features:
            results["features"] = M
        return risk_score, A_raw, results


class GatedAttentionMILRegressor(nn.Module):
    """Gated-attention MIL regressor with a bounded slide-level risk score."""
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.0,
        learnable_temperature: bool = True,
        init_temperature: float = 1.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn = _GatedAttentionScorer(hidden_dim, attn_dim, dropout=dropout)
        self.regressor = nn.Linear(hidden_dim, 1)

        init_tau = max(float(init_temperature), 1e-4)
        if learnable_temperature:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))
        else:
            self.register_buffer("log_tau", torch.log(torch.tensor(init_tau)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(min=1e-4)

    def forward(
        self,
        h: torch.Tensor,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        h_mid = self.proj(h)
        A_raw = self.attn(h_mid).transpose(1, 0)
        if attention_only:
            return A_raw

        A = F.softmax(A_raw / self.tau, dim=1)
        M = torch.mm(A, h_mid)
        risk_logit = self.regressor(M)
        risk_score = torch.sigmoid(risk_logit)
        results: Dict[str, torch.Tensor] = {
            "risk_logit": risk_logit,
            "attn_temperature": self.tau.detach(),
        }
        if return_features:
            results["features"] = M
        return risk_score, A_raw, results


class AttentionMILOrdinalRegressor(nn.Module):
    """Attention MIL model that predicts cumulative ordinal thresholds."""
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        n_thresholds: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn = _AttentionScorer(hidden_dim, attn_dim, dropout=dropout)
        self.ordinal_head = nn.Linear(hidden_dim, n_thresholds)

    def forward(
        self,
        h: torch.Tensor,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        h_mid = self.proj(h)
        A_raw = self.attn(h_mid).transpose(1, 0)
        if attention_only:
            return A_raw

        A = F.softmax(A_raw, dim=1)
        M = torch.mm(A, h_mid)
        ordinal_logits = self.ordinal_head(M)
        ordinal_probs = torch.sigmoid(ordinal_logits)
        risk_score = ordinal_probs.mean(dim=1, keepdim=True)
        results: Dict[str, torch.Tensor] = {
            "ordinal_logits": ordinal_logits,
            "ordinal_probs": ordinal_probs,
        }
        if return_features:
            results["features"] = M
        return risk_score, A_raw, results


class GatedAttentionMILOrdinalRegressor(nn.Module):
    """Gated-attention MIL model that predicts cumulative ordinal thresholds."""
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        n_thresholds: int = 3,
        dropout: float = 0.0,
        learnable_temperature: bool = True,
        init_temperature: float = 1.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn = _GatedAttentionScorer(hidden_dim, attn_dim, dropout=dropout)
        self.ordinal_head = nn.Linear(hidden_dim, n_thresholds)

        init_tau = max(float(init_temperature), 1e-4)
        if learnable_temperature:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))
        else:
            self.register_buffer("log_tau", torch.log(torch.tensor(init_tau)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(min=1e-4)

    def forward(
        self,
        h: torch.Tensor,
        return_features: bool = False,
        attention_only: bool = False,
    ):
        h_mid = self.proj(h)
        A_raw = self.attn(h_mid).transpose(1, 0)
        if attention_only:
            return A_raw

        A = F.softmax(A_raw / self.tau, dim=1)
        M = torch.mm(A, h_mid)
        ordinal_logits = self.ordinal_head(M)
        ordinal_probs = torch.sigmoid(ordinal_logits)
        risk_score = ordinal_probs.mean(dim=1, keepdim=True)
        results: Dict[str, torch.Tensor] = {
            "ordinal_logits": ordinal_logits,
            "ordinal_probs": ordinal_probs,
            "attn_temperature": self.tau.detach(),
        }
        if return_features:
            results["features"] = M
        return risk_score, A_raw, results
