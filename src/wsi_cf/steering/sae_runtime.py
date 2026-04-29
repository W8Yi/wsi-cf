from __future__ import annotations

import json
from pathlib import Path

import torch


def load_sae_from_config(
    sae_ckpt: Path | str,
    sae_cfg: Path | str | dict,
    device: str = "cuda",
):
    """
    Load SAE using config/checkpoint layout from models/sae.py training scripts.

    Config supports:
      stage in {"relu", "topk", "batch_topk", "sdf2"}
      d_in, latent_dim, tied
      topk_k/topk_mode/topk_nonneg for topk variants
      sdf_n_level2/sdf_coeff_nonneg/sdf_coeff_simplex for sdf2
    """
    from wsi_cf.models.sae import ReLUSparseSAE, TopKSAE, BatchTopKSAE, SDFSAE2Level, InputNormWrapper

    ckpt = torch.load(str(sae_ckpt), map_location="cpu")
    if isinstance(sae_cfg, dict):
        cfg = sae_cfg
    else:
        with open(str(sae_cfg), "r") as f:
            cfg = json.load(f)

    stage = str(cfg.get("stage", "relu"))
    d_in = int(cfg["d_in"])
    d_latent = int(cfg["latent_dim"])
    tied = bool(cfg.get("tied", False))
    model_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    wrapper_ln = bool(ckpt.get("wrapper_layernorm", False)) if isinstance(ckpt, dict) else False

    def _infer_topk_k(for_stage: str) -> int:
        # For batch_topk runs, prefer batch_topk_k when present.
        # Older configs may still only provide topk_k or checkpoint k.
        if for_stage == "batch_topk":
            if "batch_topk_k" in cfg and cfg["batch_topk_k"] is not None:
                return int(cfg["batch_topk_k"])
            if "topk_k" in cfg:
                return int(cfg["topk_k"])
        else:
            if "topk_k" in cfg:
                return int(cfg["topk_k"])
        if "k" in cfg:
            return int(cfg["k"])
        if isinstance(ckpt, dict) and "k" in ckpt:
            return int(ckpt["k"])
        raise KeyError("Could not infer top-k value. Add topk_k (or batch_topk_k for batch_topk) in SAE config.")

    if stage == "relu":
        base = ReLUSparseSAE(d_in=d_in, d_latent=d_latent, tied=tied, use_pre_bias=True)
    elif stage == "topk":
        base = TopKSAE(
            d_in=d_in,
            d_latent=d_latent,
            k=_infer_topk_k("topk"),
            tied=tied,
            use_pre_bias=True,
            topk_mode=str(cfg.get("topk_mode", "value")),
            nonneg=bool(cfg.get("topk_nonneg", False)),
        )
    elif stage == "batch_topk":
        base = BatchTopKSAE(
            d_in=d_in,
            d_latent=d_latent,
            k=_infer_topk_k("batch_topk"),
            tied=tied,
            use_pre_bias=True,
            topk_mode=str(cfg.get("topk_mode", "value")),
            nonneg=bool(cfg.get("topk_nonneg", False)),
        )
    elif stage == "sdf2":
        d_level2 = cfg.get("sdf_n_level2", None)
        if d_level2 is None:
            U = model_sd.get("U") if isinstance(model_sd, dict) else None
            if U is None:
                raise KeyError("sdf2 config missing 'sdf_n_level2' and checkpoint missing 'U'.")
            d_level2 = int(U.shape[1])
        base = SDFSAE2Level(
            d_in=d_in,
            d_latent=d_latent,
            d_level2=int(d_level2),
            tied=tied,
            use_pre_bias=True,
            coeff_nonneg=bool(cfg.get("sdf_coeff_nonneg", True)),
            coeff_simplex=bool(cfg.get("sdf_coeff_simplex", False)),
        )
    else:
        raise ValueError(f"Unsupported stage '{stage}'")

    base.load_state_dict(model_sd, strict=True)
    model = InputNormWrapper(d_in, base) if wrapper_ln else base
    model.eval().to(device)
    return model, d_in, d_latent


@torch.no_grad()
def sae_encode_features(sae_model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Encode input features -> SAE latent z.
    Works with models/sae.py output signatures.
    """
    if hasattr(sae_model, "encode") and callable(getattr(sae_model, "encode")):
        z = sae_model.encode(x)
        if not torch.is_tensor(z):
            raise RuntimeError("sae_model.encode(x) did not return a tensor.")
        return z

    out = sae_model(x)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError(f"Unexpected SAE forward output type={type(out)}")
    z = out[1]
    if not torch.is_tensor(z):
        raise RuntimeError("SAE forward output[1] is not a tensor.")
    return z


def sae_decode_latents(sae_model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    if hasattr(sae_model, "decode") and callable(getattr(sae_model, "decode")):
        return sae_model.decode(z)
    if hasattr(sae_model, "base") and hasattr(sae_model.base, "decode"):
        return sae_model.base.decode(z)
    raise RuntimeError("Could not find decode(z) on SAE model.")


# Backward-compatible alias used by older callers.
_sae_decode_latents = sae_decode_latents
