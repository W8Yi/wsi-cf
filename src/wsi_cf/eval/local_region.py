from __future__ import annotations

from typing import Any

import numpy as np


def replace_selected_cells_in_local_bag(
    *,
    features_local: np.ndarray,
    tile_rows: list[dict[str, Any]],
    replacement_grid: np.ndarray,
    selected_cells: list[tuple[int, int]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = np.asarray(features_local, dtype=np.float32).copy()
    z_grid = np.asarray(replacement_grid, dtype=np.float32)
    if z_grid.ndim != 3:
        raise ValueError(f"replacement_grid must have shape [Gh,Gw,D], got {tuple(z_grid.shape)}")
    selected = {(int(gx), int(gy)) for gx, gy in selected_cells}
    manifest: list[dict[str, Any]] = []
    for local_idx, row in enumerate(tile_rows):
        gx = int(row["cell_gx"])
        gy = int(row["cell_gy"])
        replace = (gx, gy) in selected
        if replace:
            out[local_idx] = z_grid[gy, gx]
        manifest.append(
            {
                "local_tile_index": int(local_idx),
                "global_tile_index": int(row["tile_index"]),
                "coord_x": int(row["coord_x"]),
                "coord_y": int(row["coord_y"]),
                "cell_gx": int(gx),
                "cell_gy": int(gy),
                "replaced": bool(replace),
            }
        )
    return out, manifest


def build_local_attention_rows(
    *,
    tile_rows: list[dict[str, Any]],
    attention_before: np.ndarray,
    attention_after: np.ndarray,
    selected_cells: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    attn_before = np.asarray(attention_before, dtype=np.float32).reshape(-1)
    attn_after = np.asarray(attention_after, dtype=np.float32).reshape(-1)
    if attn_before.shape != attn_after.shape:
        raise ValueError("attention_before and attention_after must have the same shape")
    if len(tile_rows) != int(attn_before.shape[0]):
        raise ValueError("tile_rows length must match local attention length")
    selected = {(int(gx), int(gy)) for gx, gy in selected_cells}
    order_before = np.argsort(-attn_before)
    order_after = np.argsort(-attn_after)
    rank_before = {int(tile_i): rank + 1 for rank, tile_i in enumerate(order_before.tolist())}
    rank_after = {int(tile_i): rank + 1 for rank, tile_i in enumerate(order_after.tolist())}
    rows: list[dict[str, Any]] = []
    for local_idx, row in enumerate(tile_rows):
        gx = int(row["cell_gx"])
        gy = int(row["cell_gy"])
        rows.append(
            {
                "local_tile_index": int(local_idx),
                "global_tile_index": int(row["tile_index"]),
                "coord_x": int(row["coord_x"]),
                "coord_y": int(row["coord_y"]),
                "cell_gx": int(gx),
                "cell_gy": int(gy),
                "is_selected_cell": bool((gx, gy) in selected),
                "is_high_attention": bool(row.get("is_high_attention", False)),
                "attention_before": float(attn_before[local_idx]),
                "attention_after": float(attn_after[local_idx]),
                "delta_attention": float(attn_after[local_idx] - attn_before[local_idx]),
                "attention_rank_before": int(rank_before[local_idx]),
                "attention_rank_after": int(rank_after[local_idx]),
            }
        )
    return rows
