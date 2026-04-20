from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from wsi_cf.steering.manifest import parse_steer_spec


def load_feature_vector(path: Path, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Steer feature file not found: {path}")
    if path.suffix.lower() == ".npy":
        vec = torch.from_numpy(np.load(path))
    elif path.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            raise ValueError(f"Steer file {path} is a dict; expected a plain tensor or array")
        vec = torch.as_tensor(obj)
    else:
        raise ValueError(f"Unsupported steer feature format: {path.suffix}")
    return vec.flatten().to(device=device, dtype=dtype)


def apply_tile_steering(z_grid: torch.Tensor, *, steer_specs: Sequence[str], steer_blend: float) -> torch.Tensor:
    if len(steer_specs) == 0:
        return z_grid
    if not (0.0 <= float(steer_blend) <= 1.0):
        raise ValueError("steer_blend must be in [0,1]")
    gh, gw, dim = z_grid.shape
    out = z_grid.clone()
    alpha = float(steer_blend)
    for spec in steer_specs:
        gx, gy, path = parse_steer_spec(spec)
        if gx < 0 or gx >= gw or gy < 0 or gy >= gh:
            raise ValueError(f"Steer tile ({gx},{gy}) out of range for grid [Gh={gh}, Gw={gw}]")
        vec = load_feature_vector(path, device=out.device, dtype=out.dtype)
        if vec.numel() != dim:
            raise ValueError(f"Steer feature dim mismatch for {path}: got {vec.numel()}, expected {dim}")
        if alpha >= 1.0:
            out[gy, gx] = vec
        elif alpha > 0.0:
            out[gy, gx] = (1.0 - alpha) * out[gy, gx] + alpha * vec
    return out
