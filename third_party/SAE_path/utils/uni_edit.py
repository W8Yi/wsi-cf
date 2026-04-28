from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def edit_uni_z_grid_single_tile_with_vector(
    z_grid: torch.Tensor,
    *,
    mode: str,
    vector: np.ndarray,
    vector_strength: float,
    blend: float,
    normalize_vector: bool,
) -> tuple[torch.Tensor, dict]:
    """
    Apply a UNI-space edit to exactly one tile in the z-grid.

    Supported modes:
    - ``delta``: additive edit ``x <- x + alpha * v``
    - ``target``: blend toward a target vector ``x <- (1-b)x + b*(alpha*v)``
    """
    if not (0.0 <= blend <= 1.0):
        raise ValueError("blend must be in [0, 1].")

    if z_grid.dim() == 4:
        if z_grid.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for z_grid.")
        z3 = z_grid[0]
        add_batch = True
    elif z_grid.dim() == 3:
        z3 = z_grid
        add_batch = False
    else:
        raise ValueError(f"Expected z_grid [Gh,Gw,D] or [1,Gh,Gw,D], got {tuple(z_grid.shape)}")

    gh, gw, d = z3.shape
    if (gh, gw) != (1, 1):
        raise ValueError(f"Expected a single-tile grid [1,1,D], got {(gh, gw)}")

    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] != d:
        raise ValueError(f"vector must be [D={d}], got {vector.shape}")

    z_new = z3.clone()
    x_old = z3[0, 0].to(dtype=torch.float32)

    v = torch.from_numpy(vector).to(device=x_old.device, dtype=x_old.dtype)
    if normalize_vector:
        v = v / v.norm().clamp(min=1e-8)

    if mode == "delta":
        x_edit = x_old + (float(blend) * float(vector_strength)) * v
    elif mode == "target":
        target = float(vector_strength) * v
        x_edit = (1.0 - float(blend)) * x_old + float(blend) * target
    else:
        raise ValueError(f"Unsupported vector edit mode '{mode}'")

    z_new[0, 0] = x_edit.to(device=z3.device, dtype=z3.dtype)
    if add_batch:
        z_out = z_new.unsqueeze(0)
    else:
        z_out = z_new

    dbg = {
        "mode": mode,
        "vector_strength": float(vector_strength),
        "blend": float(blend),
        "normalize_vector": bool(normalize_vector),
        "tile_feature_l2_before": float(x_old.norm().item()),
        "tile_feature_l2_after": float(x_edit.norm().item()),
        "tile_feature_delta_l2": float((x_edit - x_old).norm().item()),
    }
    return z_out, dbg
