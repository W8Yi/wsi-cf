from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def compute_attention_threshold(attention: np.ndarray, *, percentile: float) -> float:
    attn = np.asarray(attention, dtype=np.float32).reshape(-1)
    if attn.size == 0:
        raise ValueError("attention must be non-empty")
    pct = float(percentile)
    if not (0.0 <= pct <= 100.0):
        raise ValueError("percentile must be in [0,100]")
    return float(np.percentile(attn, pct))


def clamp_region_origin(
    *,
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
    slide_w: int,
    slide_h: int,
) -> tuple[int, int]:
    x0 = int(round(float(center_x) - float(crop_w) / 2.0))
    y0 = int(round(float(center_y) - float(crop_h) / 2.0))
    x0 = max(0, min(x0, max(0, int(slide_w) - int(crop_w))))
    y0 = max(0, min(y0, max(0, int(slide_h) - int(crop_h))))
    return x0, y0


def tile_centers(coords: np.ndarray, *, tile_size_level0: int) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"coords must have shape [N,2], got {tuple(arr.shape)}")
    return arr + float(tile_size_level0) / 2.0


def map_center_to_region_cell(
    *,
    center_x: float,
    center_y: float,
    region_x: int,
    region_y: int,
    crop_w_level0: int,
    crop_h_level0: int,
    grid_side: int,
) -> tuple[int, int]:
    rel_x = (float(center_x) - float(region_x)) / float(max(1, crop_w_level0))
    rel_y = (float(center_y) - float(region_y)) / float(max(1, crop_h_level0))
    gx = min(int(grid_side) - 1, max(0, int(math.floor(rel_x * int(grid_side)))))
    gy = min(int(grid_side) - 1, max(0, int(math.floor(rel_y * int(grid_side)))))
    return gx, gy


def build_region_candidate(
    *,
    slide_key: str,
    case_id: str,
    label: int,
    coords: np.ndarray,
    attention: np.ndarray,
    anchor_tile_index: int,
    anchor_rank: int,
    high_attention_threshold: float,
    slide_w: int,
    slide_h: int,
    crop_w_level0: int,
    crop_h_level0: int,
    tile_size_level0: int,
    grid_side: int,
) -> dict[str, Any]:
    coords_arr = np.asarray(coords, dtype=np.int64)
    attn_arr = np.asarray(attention, dtype=np.float32).reshape(-1)
    centers = tile_centers(coords_arr, tile_size_level0=int(tile_size_level0))
    anchor_center = centers[int(anchor_tile_index)]
    region_x, region_y = clamp_region_origin(
        center_x=float(anchor_center[0]),
        center_y=float(anchor_center[1]),
        crop_w=int(crop_w_level0),
        crop_h=int(crop_h_level0),
        slide_w=int(slide_w),
        slide_h=int(slide_h),
    )
    x1 = int(region_x) + int(crop_w_level0)
    y1 = int(region_y) + int(crop_h_level0)

    inside = (
        (centers[:, 0] >= float(region_x))
        & (centers[:, 0] < float(x1))
        & (centers[:, 1] >= float(region_y))
        & (centers[:, 1] < float(y1))
    )
    inside_idx = np.where(inside)[0].tolist()
    tile_rows: list[dict[str, Any]] = []
    selected_cells: set[tuple[int, int]] = set()
    anchor_cell: tuple[int, int] | None = None
    for tile_idx in inside_idx:
        gx, gy = map_center_to_region_cell(
            center_x=float(centers[tile_idx, 0]),
            center_y=float(centers[tile_idx, 1]),
            region_x=int(region_x),
            region_y=int(region_y),
            crop_w_level0=int(crop_w_level0),
            crop_h_level0=int(crop_h_level0),
            grid_side=int(grid_side),
        )
        is_high = bool(float(attn_arr[tile_idx]) >= float(high_attention_threshold))
        if is_high:
            selected_cells.add((int(gx), int(gy)))
        if int(tile_idx) == int(anchor_tile_index):
            anchor_cell = (int(gx), int(gy))
        tile_rows.append(
            {
                "tile_index": int(tile_idx),
                "coord_x": int(coords_arr[tile_idx, 0]),
                "coord_y": int(coords_arr[tile_idx, 1]),
                "center_x": float(centers[tile_idx, 0]),
                "center_y": float(centers[tile_idx, 1]),
                "attention": float(attn_arr[tile_idx]),
                "cell_gx": int(gx),
                "cell_gy": int(gy),
                "is_high_attention": bool(is_high),
            }
        )
    tile_rows.sort(key=lambda row: int(row["tile_index"]))
    if not selected_cells and anchor_cell is not None:
        selected_cells.add(anchor_cell)
        for row in tile_rows:
            if int(row["tile_index"]) == int(anchor_tile_index):
                row["is_high_attention"] = True
                break
    selected_cells_sorted = sorted(selected_cells, key=lambda item: (item[1], item[0]))
    return {
        "slide_key": str(slide_key),
        "case_id": str(case_id),
        "label": int(label),
        "anchor_tile_index": int(anchor_tile_index),
        "anchor_rank": int(anchor_rank),
        "anchor_attention": float(attn_arr[int(anchor_tile_index)]),
        "anchor_coord_x": int(coords_arr[int(anchor_tile_index), 0]),
        "anchor_coord_y": int(coords_arr[int(anchor_tile_index), 1]),
        "region_x": int(region_x),
        "region_y": int(region_y),
        "crop_w_level0": int(crop_w_level0),
        "crop_h_level0": int(crop_h_level0),
        "grid_side": int(grid_side),
        "tile_rows": tile_rows,
        "tile_indices_in_region": [int(row["tile_index"]) for row in tile_rows],
        "high_attention_threshold": float(high_attention_threshold),
        "selected_cells": selected_cells_sorted,
        "selected_cell_count": int(len(selected_cells_sorted)),
        "selected_cell_fraction": float(len(selected_cells_sorted) / float(max(1, int(grid_side) * int(grid_side)))),
        "high_attention_tile_count": int(sum(1 for row in tile_rows if bool(row["is_high_attention"]))),
        "local_tile_count": int(len(tile_rows)),
        "selection_fallback": False,
    }


def iter_attention_region_candidates(
    *,
    slide_key: str,
    case_id: str,
    label: int,
    coords: np.ndarray,
    attention: np.ndarray,
    slide_w: int,
    slide_h: int,
    crop_w_level0: int,
    crop_h_level0: int,
    tile_size_level0: int,
    grid_side: int,
    attention_percentile: float,
    candidate_anchor_limit: int,
) -> list[dict[str, Any]]:
    attn_arr = np.asarray(attention, dtype=np.float32).reshape(-1)
    if attn_arr.size == 0:
        raise ValueError("attention must be non-empty")
    threshold = compute_attention_threshold(attn_arr, percentile=float(attention_percentile))
    ranked = np.argsort(-attn_arr)
    candidates: list[dict[str, Any]] = []
    seen_regions: set[tuple[int, int]] = set()
    for anchor_rank, anchor_tile_index in enumerate(ranked.tolist(), start=1):
        candidate = build_region_candidate(
            slide_key=str(slide_key),
            case_id=str(case_id),
            label=int(label),
            coords=coords,
            attention=attn_arr,
            anchor_tile_index=int(anchor_tile_index),
            anchor_rank=int(anchor_rank),
            high_attention_threshold=float(threshold),
            slide_w=int(slide_w),
            slide_h=int(slide_h),
            crop_w_level0=int(crop_w_level0),
            crop_h_level0=int(crop_h_level0),
            tile_size_level0=int(tile_size_level0),
            grid_side=int(grid_side),
        )
        region_key = (int(candidate["region_x"]), int(candidate["region_y"]))
        if region_key in seen_regions:
            continue
        seen_regions.add(region_key)
        candidates.append(candidate)
        if len(candidates) >= max(1, int(candidate_anchor_limit)):
            break
    return candidates


def select_attention_region(
    *,
    slide_key: str,
    case_id: str,
    label: int,
    coords: np.ndarray,
    attention: np.ndarray,
    slide_w: int,
    slide_h: int,
    crop_w_level0: int,
    crop_h_level0: int,
    tile_size_level0: int,
    grid_side: int,
    attention_percentile: float,
    min_high_attention_cells: int,
    max_high_attention_cells: int,
    candidate_anchor_limit: int,
) -> dict[str, Any] | None:
    fallback: dict[str, Any] | None = None
    for candidate in iter_attention_region_candidates(
        slide_key=str(slide_key),
        case_id=str(case_id),
        label=int(label),
        coords=coords,
        attention=attention,
        slide_w=int(slide_w),
        slide_h=int(slide_h),
        crop_w_level0=int(crop_w_level0),
        crop_h_level0=int(crop_h_level0),
        tile_size_level0=int(tile_size_level0),
        grid_side=int(grid_side),
        attention_percentile=float(attention_percentile),
        candidate_anchor_limit=int(candidate_anchor_limit),
    ):
        if fallback is None:
            fallback = dict(candidate)
            fallback["selection_fallback"] = True
        if int(min_high_attention_cells) <= int(candidate["selected_cell_count"]) <= int(max_high_attention_cells):
            candidate["selection_fallback"] = False
            return candidate
    if fallback is None:
        raise RuntimeError("Could not build any region proposal candidates.")
    if int(min_high_attention_cells) > 0:
        fallback["selection_fallback"] = True
    return fallback
