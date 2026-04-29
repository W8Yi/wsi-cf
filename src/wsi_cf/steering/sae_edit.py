from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from wsi_cf.steering.sae_runtime import sae_decode_latents, sae_encode_features


@torch.no_grad()
def edit_uni_z_grid_with_sae(
    sae_model: torch.nn.Module,
    z_grid: torch.Tensor,
    latent_idx: Optional[int] = None,
    target_latent_vector: Optional[torch.Tensor | np.ndarray] = None,
    target_latent_vector_strength: float = 1.0,
    delta_latent_vector: Optional[torch.Tensor | np.ndarray] = None,
    delta_latent_vector_scale: float = 1.0,
    target_value: Optional[float] = None,
    delta: Optional[float] = None,
    scale: Optional[float] = None,
    clamp_value: Optional[float] = None,
    tile_indices: Optional[np.ndarray] = None,
    tile_mask: Optional[np.ndarray] = None,
    blend: float = 1.0,
    latent_strength: float = 1.0,
    soft_mask_sigma: float = 0.0,
    max_feature_delta_norm: Optional[float] = None,
    keep_non_selected: bool = True,
    return_debug: bool = True,
):
    """
    Edit UNI z_grid in SAE latent space and decode back to UNI feature space.

    Returns:
      z_grid_new, debug_dict
    """
    if not (0.0 <= blend <= 1.0):
        raise ValueError("blend must be in [0,1].")
    if not (0.0 <= latent_strength <= 1.0):
        raise ValueError("latent_strength must be in [0,1].")
    if not (0.0 <= target_latent_vector_strength <= 1.0):
        raise ValueError("target_latent_vector_strength must be in [0,1].")
    # Full-code delta mode is allowed to use negative / >1 scales.
    if soft_mask_sigma < 0.0:
        raise ValueError("soft_mask_sigma must be >= 0.")
    if max_feature_delta_norm is not None and max_feature_delta_norm <= 0.0:
        raise ValueError("max_feature_delta_norm must be > 0 when provided.")
    if clamp_value is not None and (target_value is not None or delta is not None or scale is not None):
        raise ValueError("clamp_value cannot be combined with target_value, delta, or scale.")
    if scale is not None and (target_value is not None or delta is not None):
        raise ValueError("scale cannot be combined with target_value or delta.")
    if target_latent_vector is not None and delta_latent_vector is not None:
        raise ValueError("target_latent_vector and delta_latent_vector are mutually exclusive.")
    if target_latent_vector is not None:
        # Full-code target interpolation is a separate edit mode.
        if latent_idx is not None or target_value is not None or delta is not None or scale is not None or clamp_value is not None:
            raise ValueError(
                "target_latent_vector cannot be combined with latent_idx/target_value/delta/scale/clamp_value edits."
            )
    if delta_latent_vector is not None:
        if latent_idx is not None or target_value is not None or delta is not None or scale is not None or clamp_value is not None:
            raise ValueError(
                "delta_latent_vector cannot be combined with latent_idx/target_value/delta/scale/clamp_value edits."
            )

    if z_grid.dim() == 4:
        if z_grid.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for z_grid.")
        z3 = z_grid[0]
        add_batch_back = True
    elif z_grid.dim() == 3:
        z3 = z_grid
        add_batch_back = False
    else:
        raise ValueError(f"Expected z_grid [Gh,Gw,D] or [1,Gh,Gw,D], got {tuple(z_grid.shape)}")

    Gh, Gw, D = z3.shape
    N = Gh * Gw
    device = next(sae_model.parameters()).device

    x = z3.reshape(N, D).to(device=device, dtype=torch.float32)
    z_lat = sae_encode_features(sae_model, x)

    if tile_mask is not None:
        tile_mask = np.asarray(tile_mask)
        if tile_mask.shape != (Gh, Gw):
            raise ValueError(f"tile_mask must be shape {(Gh, Gw)}, got {tile_mask.shape}")
        if tile_mask.dtype == np.bool_:
            weight_np = tile_mask.astype(np.float32)
        else:
            weight_np = np.clip(tile_mask.astype(np.float32), 0.0, 1.0)
        sel = np.flatnonzero(weight_np.reshape(-1) > 0.0)
    elif tile_indices is not None:
        sel = np.asarray(tile_indices).reshape(-1).astype(np.int64)
        if sel.size and (sel.min() < 0 or sel.max() >= N):
            raise ValueError(f"tile_indices must be in [0, {N-1}]")
        weight_np = np.zeros((Gh, Gw), dtype=np.float32)
        if sel.size:
            weight_np.reshape(-1)[sel] = 1.0
    else:
        sel = np.arange(N, dtype=np.int64)
        weight_np = np.ones((Gh, Gw), dtype=np.float32)

    if soft_mask_sigma > 0.0:
        radius = max(1, int(round(3.0 * soft_mask_sigma)))
        k = 2 * radius + 1
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
        ker = np.exp(-0.5 * (xx * xx + yy * yy) / float(soft_mask_sigma * soft_mask_sigma))
        ker = ker / np.maximum(ker.sum(), 1e-8)
        ker_t = torch.from_numpy(ker).to(device=device, dtype=torch.float32).view(1, 1, k, k)
        w_t = torch.from_numpy(weight_np).to(device=device, dtype=torch.float32).view(1, 1, Gh, Gw)
        w_t = F.conv2d(w_t, ker_t, padding=radius)
        w_max = w_t.max().clamp(min=1e-8)
        w_t = (w_t / w_max).clamp(0.0, 1.0)
        weight_flat = w_t.view(N, 1)
    else:
        weight_flat = torch.from_numpy(weight_np.reshape(N, 1)).to(device=device, dtype=torch.float32)

    z_edit = z_lat.clone()
    if target_latent_vector is not None:
        tgt_vec = target_latent_vector
        if isinstance(tgt_vec, np.ndarray):
            tgt_vec = torch.from_numpy(tgt_vec)
        if not torch.is_tensor(tgt_vec):
            raise TypeError("target_latent_vector must be a torch.Tensor or numpy.ndarray")
        tgt_vec = tgt_vec.to(device=z_edit.device, dtype=z_edit.dtype).reshape(-1)
        if tgt_vec.numel() != z_edit.shape[1]:
            raise ValueError(
                f"target_latent_vector size {tgt_vec.numel()} does not match latent dim {z_edit.shape[1]}"
            )
        sel_t = torch.as_tensor(sel, device=z_edit.device, dtype=torch.long)
        cur = z_edit[sel_t]
        z_edit[sel_t] = (1.0 - float(target_latent_vector_strength)) * cur + float(target_latent_vector_strength) * tgt_vec.view(1, -1)
    elif delta_latent_vector is not None:
        dvec = delta_latent_vector
        if isinstance(dvec, np.ndarray):
            dvec = torch.from_numpy(dvec)
        if not torch.is_tensor(dvec):
            raise TypeError("delta_latent_vector must be a torch.Tensor or numpy.ndarray")
        dvec = dvec.to(device=z_edit.device, dtype=z_edit.dtype).reshape(-1)
        if dvec.numel() != z_edit.shape[1]:
            raise ValueError(
                f"delta_latent_vector size {dvec.numel()} does not match latent dim {z_edit.shape[1]}"
            )
        sel_t = torch.as_tensor(sel, device=z_edit.device, dtype=torch.long)
        z_edit[sel_t] = z_edit[sel_t] + float(delta_latent_vector_scale) * dvec.view(1, -1)
    elif latent_idx is not None:
        if latent_idx < 0 or latent_idx >= z_edit.shape[1]:
            raise ValueError(f"latent_idx must be in [0, {z_edit.shape[1]-1}]")
        sel_t = torch.as_tensor(sel, device=z_edit.device, dtype=torch.long)
        if target_value is not None:
            cur = z_edit[sel_t, latent_idx]
            tgt = torch.full_like(cur, float(target_value))
            z_edit[sel_t, latent_idx] = (1.0 - latent_strength) * cur + latent_strength * tgt
        if delta is not None:
            z_edit[sel_t, latent_idx] = z_edit[sel_t, latent_idx] + float(delta)
        if scale is not None:
            z_edit[sel_t, latent_idx] = z_edit[sel_t, latent_idx] * float(scale)
        if clamp_value is not None:
            z_edit[sel_t, latent_idx] = torch.clamp(z_edit[sel_t, latent_idx], max=float(clamp_value))
    elif target_value is not None or delta is not None:
        raise ValueError("latent_idx is required when target_value or delta is provided.")

    x_rec = sae_decode_latents(sae_model, z_edit)
    if max_feature_delta_norm is not None:
        d = x_rec - x
        dn = d.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scl = torch.clamp(float(max_feature_delta_norm) / dn, max=1.0)
        x_rec = x + d * scl

    if keep_non_selected:
        blend_w = blend * weight_flat
        x_new = x * (1.0 - blend_w) + x_rec * blend_w
    else:
        x_new = (1.0 - blend) * x + blend * x_rec

    z_grid_new = x_new.reshape(Gh, Gw, D).to(device=z3.device, dtype=z3.dtype)
    if add_batch_back:
        z_grid_new = z_grid_new.unsqueeze(0)

    if return_debug:
        dbg = {
            "Gh": int(Gh),
            "Gw": int(Gw),
            "D": int(D),
            "num_tiles_total": int(N),
            "num_tiles_selected": int(sel.size),
            "latent_dim": int(z_edit.shape[1]),
            "latent_idx": None if latent_idx is None else int(latent_idx),
            "target_latent_vector_shape": None if target_latent_vector is None else [int(z_edit.shape[1])],
            "target_latent_vector_strength": float(target_latent_vector_strength),
            "delta_latent_vector_shape": None if delta_latent_vector is None else [int(z_edit.shape[1])],
            "delta_latent_vector_scale": float(delta_latent_vector_scale),
            "target_value": target_value,
            "delta": delta,
            "scale": scale,
            "clamp_value": clamp_value,
            "blend": float(blend),
            "latent_strength": float(latent_strength),
            "soft_mask_sigma": float(soft_mask_sigma),
            "max_feature_delta_norm": max_feature_delta_norm,
            "selected_indices": sel,
            "blend_weight_min": float(weight_flat.min().item()),
            "blend_weight_mean": float(weight_flat.mean().item()),
            "blend_weight_max": float(weight_flat.max().item()),
            "z_lat_before": z_lat.detach(),
            "z_lat_after": z_edit.detach(),
        }
        return z_grid_new, dbg
    return z_grid_new, None
