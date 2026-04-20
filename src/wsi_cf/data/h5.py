from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_h5_features_coords(h5_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{h5_path}: missing dataset 'features'")
        feats = handle["features"]
        if feats.ndim == 2:
            x = feats[:]
        elif feats.ndim == 3 and int(feats.shape[0]) == 1:
            x = feats[0]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")

        coords = None
        if "coords" in handle:
            c = handle["coords"]
            if c.ndim == 2 and int(c.shape[1]) == 2:
                coords = c[:]
            elif c.ndim == 3 and int(c.shape[0]) == 1 and int(c.shape[2]) == 2:
                coords = c[0]
            else:
                raise RuntimeError(f"{h5_path}: unsupported coords shape {tuple(c.shape)}")

    x_np = np.asarray(x, dtype=np.float32)
    coords_np = None if coords is None else np.asarray(coords, dtype=np.int64)
    if coords_np is not None and x_np.shape[0] != coords_np.shape[0]:
        raise RuntimeError(f"{h5_path}: features/coords row mismatch {x_np.shape} vs {coords_np.shape}")
    return x_np, coords_np


def resolve_tile_index(*, feats: np.ndarray, coords: np.ndarray | None, tile_index: int | None, coord: tuple[int, int] | None) -> int:
    if tile_index is None and coord is None:
        raise ValueError("Provide either tile_index or coord")
    if tile_index is not None:
        idx = int(tile_index)
        if idx < 0 or idx >= int(feats.shape[0]):
            raise IndexError(f"tile_index {idx} out of range for feature table with {feats.shape[0]} rows")
        return idx
    if coords is None:
        raise RuntimeError("coords are required when selecting by coord")
    target_x, target_y = int(coord[0]), int(coord[1])
    matches = np.where((coords[:, 0] == target_x) & (coords[:, 1] == target_y))[0]
    if matches.size == 0:
        raise RuntimeError(f"No tile found at coord=({target_x}, {target_y})")
    return int(matches[0])
