#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


AREA_ORDER = [
    "full_region",
    "selected_cells",
    "unselected_cells",
    "committed_pixels",
    "outside_committed_pixels",
]

SEAM_AREA_ORDER = [
    "all_tile_boundaries",
    "selected_cell_perimeter",
    "edited_cell_perimeter",
    "visited_cell_perimeter",
    "committed_window_edges",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare visual perturbation for two edit methods by measuring RGB difference "
            "between each generated after-image and the same source region."
        )
    )
    parser.add_argument("--ours-root", type=Path, required=True, help="Run root for the paper edit method.")
    parser.add_argument("--naive-root", type=Path, required=True, help="Run root for the naive edit baseline.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional combined edit manifest used to annotate run ids.")
    parser.add_argument("--ours-name", type=str, default="ours")
    parser.add_argument("--naive-name", type=str, default="bad_naive")
    parser.add_argument("--source-image-name", type=str, default="source_region_actual.png")
    parser.add_argument("--generated-image-name", type=str, default="generated.png")
    parser.add_argument("--run-manifest-name", type=str, default="run_manifest.json")
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--max-runs", type=int, default=0, help="Debug limit; 0 means all paired runs.")
    parser.add_argument(
        "--run-ids-file",
        type=Path,
        default=None,
        help="Optional text file with one run_id per line. Only these paired runs are scored.",
    )
    parser.add_argument(
        "--no-per-cell",
        action="store_true",
        help="Skip visual_perturbation_per_cell.csv to keep metric outputs smaller.",
    )
    parser.add_argument(
        "--no-seam",
        action="store_true",
        help="Skip tile-border seam inconsistency metrics.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing pairs instead of failing.")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--formats", type=str, default="png,pdf,svg")
    return parser


def load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def load_manifest_requests(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("edit_requests") or payload.get("requests") or []
    else:
        raise ValueError(f"Unsupported manifest payload in {path}: {type(payload)}")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id", "")).strip()
        if run_id:
            out[run_id] = dict(row)
    return out


def load_run_ids_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    run_ids = {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")}
    return run_ids


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def finite_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else math.nan


def cell_key(cell: Any) -> tuple[int, int] | None:
    if isinstance(cell, dict):
        if "gx" in cell and "gy" in cell:
            return int(cell["gx"]), int(cell["gy"])
        if "cell_gx" in cell and "cell_gy" in cell:
            return int(cell["cell_gx"]), int(cell["cell_gy"])
    if isinstance(cell, (list, tuple)) and len(cell) >= 2:
        return int(cell[0]), int(cell[1])
    return None


def cells_from_payload(*payloads: dict[str, Any]) -> list[dict[str, int]]:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in ("target_cells", "selected_cells", "edit_cells", "cells"):
            raw = payload.get(key)
            if not raw:
                continue
            cells: list[dict[str, int]] = []
            seen: set[tuple[int, int]] = set()
            for item in raw:
                key_xy = cell_key(item)
                if key_xy is None or key_xy in seen:
                    continue
                seen.add(key_xy)
                cells.append({"gx": int(key_xy[0]), "gy": int(key_xy[1])})
            if cells:
                return cells
    return []


def image_cell_shape(image_shape: tuple[int, int, int], grid_step_px: int) -> tuple[int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    return int(math.ceil(height / int(grid_step_px))), int(math.ceil(width / int(grid_step_px)))


def cell_mask(image_shape: tuple[int, int, int], cells: list[dict[str, int]], grid_step_px: int) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=bool)
    for cell in cells:
        gx, gy = int(cell["gx"]), int(cell["gy"])
        x0 = max(0, min(width, gx * int(grid_step_px)))
        y0 = max(0, min(height, gy * int(grid_step_px)))
        x1 = max(0, min(width, (gx + 1) * int(grid_step_px)))
        y1 = max(0, min(height, (gy + 1) * int(grid_step_px)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def commit_mask_from_steps(image_shape: tuple[int, int, int], steps: list[dict[str, Any]]) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=bool)
    for step in steps:
        if not isinstance(step, dict):
            continue
        bounds = step.get("commit_bounds_global")
        if isinstance(bounds, dict):
            x0 = int(bounds.get("x0", 0))
            y0 = int(bounds.get("y0", 0))
            x1 = int(bounds.get("x1", x0))
            y1 = int(bounds.get("y1", y0))
        else:
            x0 = int(step.get("left", step.get("x0", 0)))
            y0 = int(step.get("top", step.get("y0", 0)))
            x1 = int(step.get("right", step.get("x1", x0)))
            y1 = int(step.get("bottom", step.get("y1", y0)))
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def cell_set_from_payload_key(payload: dict[str, Any], keys: tuple[str, ...]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    if not isinstance(payload, dict):
        return out
    for key in keys:
        raw = payload.get(key)
        if not raw:
            continue
        for item in raw:
            key_xy = cell_key(item)
            if key_xy is not None:
                out.add((int(key_xy[0]), int(key_xy[1])))
    return out


def window_history_from_manifest(run_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("window_history", "step_records", "progressive_steps", "steps"):
        raw = run_manifest.get(key)
        if isinstance(raw, list):
            return [dict(item) for item in raw if isinstance(item, dict)]
    return []


def selected_cell_mask_grid(grid_shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for gx, gy in cells:
        if 0 <= int(gx) < grid_w and 0 <= int(gy) < grid_h:
            mask[int(gy), int(gx)] = True
    return mask


def tile_boundary_values(
    image: np.ndarray,
    *,
    grid_step_px: int,
    selector: str,
    cell_mask_grid: np.ndarray | None = None,
) -> np.ndarray:
    values: list[np.ndarray] = []
    height, width = image.shape[:2]
    grid_h, grid_w = image_cell_shape(image.shape, grid_step_px)
    step = int(grid_step_px)
    for gy in range(grid_h):
        y0 = gy * step
        y1 = min((gy + 1) * step, height)
        for gx in range(1, grid_w):
            if selector == "perimeter":
                assert cell_mask_grid is not None
                if bool(cell_mask_grid[gy, gx - 1]) == bool(cell_mask_grid[gy, gx]):
                    continue
            x = gx * step
            if not (0 < x < width and y1 > y0):
                continue
            values.append(np.abs(image[y0:y1, x, :] - image[y0:y1, x - 1, :]).mean(axis=1))
    for gy in range(1, grid_h):
        y = gy * step
        if not (0 < y < height):
            continue
        y_left = gy - 1
        y_right = gy
        for gx in range(grid_w):
            if selector == "perimeter":
                assert cell_mask_grid is not None
                if bool(cell_mask_grid[y_left, gx]) == bool(cell_mask_grid[y_right, gx]):
                    continue
            x0 = gx * step
            x1 = min((gx + 1) * step, width)
            if x1 <= x0:
                continue
            values.append(np.abs(image[y, x0:x1, :] - image[y - 1, x0:x1, :]).mean(axis=1))
    if not values:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(values).astype(np.float32, copy=False)


def window_edge_values(image: np.ndarray, steps: list[dict[str, Any]]) -> np.ndarray:
    values: list[np.ndarray] = []
    height, width = image.shape[:2]
    for step in steps:
        bounds = step.get("commit_bounds_global")
        if not isinstance(bounds, dict):
            continue
        x0 = max(0, min(width, int(bounds.get("x0", 0))))
        y0 = max(0, min(height, int(bounds.get("y0", 0))))
        x1 = max(0, min(width, int(bounds.get("x1", x0))))
        y1 = max(0, min(height, int(bounds.get("y1", y0))))
        if x1 <= x0 or y1 <= y0:
            continue
        if 0 < x0 < width:
            values.append(np.abs(image[y0:y1, x0, :] - image[y0:y1, x0 - 1, :]).mean(axis=1))
        if 0 < x1 < width:
            values.append(np.abs(image[y0:y1, x1, :] - image[y0:y1, x1 - 1, :]).mean(axis=1))
        if 0 < y0 < height:
            values.append(np.abs(image[y0, x0:x1, :] - image[y0 - 1, x0:x1, :]).mean(axis=1))
        if 0 < y1 < height:
            values.append(np.abs(image[y1, x0:x1, :] - image[y1 - 1, x0:x1, :]).mean(axis=1))
    if not values:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(values).astype(np.float32, copy=False)


def summarize_seam_values(source_values: np.ndarray, generated_values: np.ndarray) -> dict[str, float | int]:
    if source_values.shape != generated_values.shape:
        raise ValueError(f"Seam value shape mismatch: {source_values.shape} vs {generated_values.shape}")
    if generated_values.size == 0:
        return {
            "boundary_sample_count": 0,
            "source_seam_mean_abs_rgb": math.nan,
            "after_seam_mean_abs_rgb": math.nan,
            "seam_excess_mean_abs_rgb": math.nan,
            "seam_excess_median_abs_rgb": math.nan,
            "seam_excess_p95_abs_rgb": math.nan,
            "seam_excess_positive_rate": math.nan,
            "hard_seam_excess_gt10_rate": math.nan,
            "hard_seam_excess_gt25_rate": math.nan,
        }
    excess = generated_values.astype(np.float32) - source_values.astype(np.float32)
    return {
        "boundary_sample_count": int(generated_values.size),
        "source_seam_mean_abs_rgb": float(source_values.mean()),
        "after_seam_mean_abs_rgb": float(generated_values.mean()),
        "seam_excess_mean_abs_rgb": float(excess.mean()),
        "seam_excess_median_abs_rgb": float(np.median(excess)),
        "seam_excess_p95_abs_rgb": float(np.percentile(excess, 95.0)),
        "seam_excess_positive_rate": float(np.mean(excess > 0.0)),
        "hard_seam_excess_gt10_rate": float(np.mean(excess > 10.0)),
        "hard_seam_excess_gt25_rate": float(np.mean(excess > 25.0)),
    }


def seam_rows_for_method(
    *,
    run_id: str,
    method: str,
    source: np.ndarray,
    generated: np.ndarray,
    grid_step_px: int,
    selected_cells: set[tuple[int, int]],
    edited_cells: set[tuple[int, int]],
    visited_cells: set[tuple[int, int]],
    steps: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    grid_shape = image_cell_shape(source.shape, grid_step_px)
    cell_masks = {
        "selected_cell_perimeter": selected_cell_mask_grid(grid_shape, selected_cells),
        "edited_cell_perimeter": selected_cell_mask_grid(grid_shape, edited_cells),
        "visited_cell_perimeter": selected_cell_mask_grid(grid_shape, visited_cells),
    }
    specs: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "all_tile_boundaries",
            tile_boundary_values(source, grid_step_px=grid_step_px, selector="all"),
            tile_boundary_values(generated, grid_step_px=grid_step_px, selector="all"),
        ),
        (
            "committed_window_edges",
            window_edge_values(source, steps),
            window_edge_values(generated, steps),
        ),
    ]
    for area, mask in cell_masks.items():
        specs.append(
            (
                area,
                tile_boundary_values(source, grid_step_px=grid_step_px, selector="perimeter", cell_mask_grid=mask),
                tile_boundary_values(generated, grid_step_px=grid_step_px, selector="perimeter", cell_mask_grid=mask),
            )
        )
    rows: list[dict[str, Any]] = []
    for area, source_values, generated_values in specs:
        rows.append(
            {
                **run_meta,
                "run_id": run_id,
                "method": method,
                "seam_area": area,
                **summarize_seam_values(source_values, generated_values),
            }
        )
    return rows


def summarize_diff(source: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    if source.shape != generated.shape:
        raise ValueError(f"Image shape mismatch: {source.shape} vs {generated.shape}")
    if not bool(mask.any()):
        return {
            "pixel_count": 0,
            "mean_abs_rgb": math.nan,
            "median_abs_rgb": math.nan,
            "rmse_rgb": math.nan,
            "changed_fraction_gt10": math.nan,
            "changed_fraction_gt25": math.nan,
        }
    diff = np.abs(generated.astype(np.float32) - source.astype(np.float32))
    selected = diff[mask]
    mean_per_pixel = selected.mean(axis=1)
    return {
        "pixel_count": int(selected.shape[0]),
        "mean_abs_rgb": float(mean_per_pixel.mean()),
        "median_abs_rgb": float(np.median(mean_per_pixel)),
        "rmse_rgb": float(np.sqrt(np.mean(np.square(selected)))),
        "changed_fraction_gt10": float(np.mean(mean_per_pixel > 10.0)),
        "changed_fraction_gt25": float(np.mean(mean_per_pixel > 25.0)),
    }


def per_cell_diff_rows(
    *,
    run_id: str,
    method: str,
    source: np.ndarray,
    generated: np.ndarray,
    selected_cells: list[dict[str, int]],
    grid_step_px: int,
    run_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    grid_h, grid_w = image_cell_shape(source.shape, grid_step_px)
    selected_set = {(int(cell["gx"]), int(cell["gy"])) for cell in selected_cells}
    diff = np.abs(generated.astype(np.float32) - source.astype(np.float32)).mean(axis=2)
    rows: list[dict[str, Any]] = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            x0 = gx * int(grid_step_px)
            y0 = gy * int(grid_step_px)
            x1 = min((gx + 1) * int(grid_step_px), diff.shape[1])
            y1 = min((gy + 1) * int(grid_step_px), diff.shape[0])
            values = diff[y0:y1, x0:x1]
            rows.append(
                {
                    **run_meta,
                    "run_id": run_id,
                    "method": method,
                    "cell_gx": int(gx),
                    "cell_gy": int(gy),
                    "is_selected_cell": int((gx, gy) in selected_set),
                    "mean_abs_rgb": float(values.mean()) if values.size else math.nan,
                    "median_abs_rgb": float(np.median(values)) if values.size else math.nan,
                }
            )
    return rows


def run_metadata(request: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "task_name",
        "direction",
        "source_label",
        "target_label",
        "selector",
        "repeat_id",
        "budget",
        "budget_is_full",
        "valid_cell_count",
        "region_id",
        "slide_id",
    ]
    return {key: request.get(key, "") for key in keep}


def discover_run_ids(
    ours_root: Path,
    naive_root: Path,
    manifest: dict[str, dict[str, Any]],
    run_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    ours_ids = {path.name for path in ours_root.iterdir() if path.is_dir()} if ours_root.exists() else set()
    naive_ids = {path.name for path in naive_root.iterdir() if path.is_dir()} if naive_root.exists() else set()
    if manifest:
        wanted = set(manifest)
        ours_ids &= wanted
        naive_ids &= wanted
    if run_ids is not None:
        ours_ids &= run_ids
        naive_ids &= run_ids
    paired = sorted(ours_ids & naive_ids)
    wanted_missing = set(manifest)
    if run_ids is not None:
        wanted_missing &= run_ids
    missing = sorted((ours_ids | naive_ids | wanted_missing) - set(paired))
    return paired, missing


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["area"]))].append(row)
    out: list[dict[str, Any]] = []
    for (method, area), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        values = np.asarray([float(row["mean_abs_rgb"]) for row in group], dtype=np.float32)
        values = values[np.isfinite(values)]
        out.append(
            {
                "method": method,
                "area": area,
                "n": int(values.size),
                "mean_abs_rgb_mean": finite_mean(values),
                "mean_abs_rgb_median": finite_median(values),
                "mean_abs_rgb_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "mean_abs_rgb_sem": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
            }
        )
    return out


def pairwise_rows(summary_rows: list[dict[str, Any]], ours_name: str, naive_name: str) -> list[dict[str, Any]]:
    by_key = {(str(row["run_id"]), str(row["area"]), str(row["method"])): row for row in summary_rows}
    keys = sorted({(run_id, area) for run_id, area, _method in by_key})
    rows: list[dict[str, Any]] = []
    for run_id, area in keys:
        ours = by_key.get((run_id, area, ours_name))
        naive = by_key.get((run_id, area, naive_name))
        if ours is None or naive is None:
            continue
        ours_value = float(ours["mean_abs_rgb"])
        naive_value = float(naive["mean_abs_rgb"])
        if not np.isfinite(ours_value) or not np.isfinite(naive_value):
            continue
        rows.append(
            {
                "run_id": run_id,
                "area": area,
                "ours_method": ours_name,
                "naive_method": naive_name,
                "ours_mean_abs_rgb": ours_value,
                "naive_mean_abs_rgb": naive_value,
                "naive_minus_ours_mean_abs_rgb": float(naive_value - ours_value),
                "naive_over_ours_mean_abs_rgb": float(naive_value / ours_value) if ours_value > 0 else math.nan,
            }
        )
    return rows


def aggregate_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["area"])].append(row)
    out: list[dict[str, Any]] = []
    for area, group in sorted(grouped.items()):
        delta = np.asarray([float(row["naive_minus_ours_mean_abs_rgb"]) for row in group], dtype=np.float32)
        ratio = np.asarray([float(row["naive_over_ours_mean_abs_rgb"]) for row in group], dtype=np.float32)
        delta = delta[np.isfinite(delta)]
        ratio = ratio[np.isfinite(ratio)]
        out.append(
            {
                "area": area,
                "n_paired_runs": int(len(group)),
                "mean_naive_minus_ours_mean_abs_rgb": finite_mean(delta),
                "median_naive_minus_ours_mean_abs_rgb": finite_median(delta),
                "mean_naive_over_ours_mean_abs_rgb": finite_mean(ratio),
                "median_naive_over_ours_mean_abs_rgb": finite_median(ratio),
                "naive_more_perturbed_rate": float(np.mean(delta > 0.0)) if delta.size else math.nan,
            }
        )
    return out


def aggregate_seam_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["seam_area"]))].append(row)
    out: list[dict[str, Any]] = []
    for (method, area), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        excess = np.asarray([float(row["seam_excess_mean_abs_rgb"]) for row in group], dtype=np.float32)
        after = np.asarray([float(row["after_seam_mean_abs_rgb"]) for row in group], dtype=np.float32)
        source = np.asarray([float(row["source_seam_mean_abs_rgb"]) for row in group], dtype=np.float32)
        positive = np.asarray([float(row["seam_excess_positive_rate"]) for row in group], dtype=np.float32)
        excess = excess[np.isfinite(excess)]
        after = after[np.isfinite(after)]
        source = source[np.isfinite(source)]
        positive = positive[np.isfinite(positive)]
        out.append(
            {
                "method": method,
                "seam_area": area,
                "n": int(excess.size),
                "mean_source_seam_abs_rgb": finite_mean(source),
                "mean_after_seam_abs_rgb": finite_mean(after),
                "mean_seam_excess_abs_rgb": finite_mean(excess),
                "median_seam_excess_abs_rgb": finite_median(excess),
                "mean_seam_excess_positive_rate": finite_mean(positive),
            }
        )
    return out


def pairwise_seam_rows(rows: list[dict[str, Any]], ours_name: str, naive_name: str) -> list[dict[str, Any]]:
    by_key = {(str(row["run_id"]), str(row["seam_area"]), str(row["method"])): row for row in rows}
    keys = sorted({(run_id, area) for run_id, area, _method in by_key})
    out: list[dict[str, Any]] = []
    for run_id, area in keys:
        ours = by_key.get((run_id, area, ours_name))
        naive = by_key.get((run_id, area, naive_name))
        if ours is None or naive is None:
            continue
        ours_excess = float(ours["seam_excess_mean_abs_rgb"])
        naive_excess = float(naive["seam_excess_mean_abs_rgb"])
        if not np.isfinite(ours_excess) or not np.isfinite(naive_excess):
            continue
        out.append(
            {
                "run_id": run_id,
                "seam_area": area,
                "ours_method": ours_name,
                "naive_method": naive_name,
                "ours_seam_excess_mean_abs_rgb": ours_excess,
                "naive_seam_excess_mean_abs_rgb": naive_excess,
                "naive_minus_ours_seam_excess_mean_abs_rgb": float(naive_excess - ours_excess),
            }
        )
    return out


def aggregate_pairwise_seam(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["seam_area"])].append(row)
    out: list[dict[str, Any]] = []
    for area, group in sorted(grouped.items()):
        delta = np.asarray([float(row["naive_minus_ours_seam_excess_mean_abs_rgb"]) for row in group], dtype=np.float32)
        delta = delta[np.isfinite(delta)]
        out.append(
            {
                "seam_area": area,
                "n_paired_runs": int(delta.size),
                "mean_naive_minus_ours_seam_excess_abs_rgb": finite_mean(delta),
                "median_naive_minus_ours_seam_excess_abs_rgb": finite_median(delta),
                "naive_has_more_excess_seam_rate": float(np.mean(delta > 0.0)) if delta.size else math.nan,
            }
        )
    return out


def save_bar_plot(aggregate: list[dict[str, Any]], out_dir: Path, title: str, formats: list[str]) -> None:
    if not formats:
        return
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsi_cf_matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    wanted = ["full_region", "selected_cells", "unselected_cells"]
    rows = [row for row in aggregate if row["area"] in wanted and int(row["n"]) > 0]
    if not rows:
        return
    methods = sorted({str(row["method"]) for row in rows})
    x = np.arange(len(wanted), dtype=np.float32)
    width = 0.34 if len(methods) <= 2 else 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, method in enumerate(methods):
        values = []
        errors = []
        for area in wanted:
            row = next((candidate for candidate in rows if candidate["method"] == method and candidate["area"] == area), None)
            values.append(float(row["mean_abs_rgb_mean"]) if row is not None else math.nan)
            errors.append(float(row["mean_abs_rgb_sem"]) if row is not None else 0.0)
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, yerr=errors, label=method, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Full region", "Selected cells", "Unselected cells"])
    ax.set_ylabel("Mean absolute RGB difference vs source")
    ax.set_title(title or "Visual Perturbation")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"visual_perturbation_summary.{fmt}", dpi=300)
    plt.close(fig)


def collect_visual_rows(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest_requests(args.manifest)
    run_ids = load_run_ids_file(getattr(args, "run_ids_file", None))
    paired, missing = discover_run_ids(args.ours_root, args.naive_root, manifest, run_ids)
    if int(args.max_runs) > 0:
        paired = paired[: int(args.max_runs)]
    if missing and not args.allow_missing:
        preview = ", ".join(missing[:8])
        raise FileNotFoundError(f"Missing unpaired generated run directories ({len(missing)} total): {preview}")

    method_roots = [(str(args.ours_name), args.ours_root), (str(args.naive_name), args.naive_root)]
    summary_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    seam_rows: list[dict[str, Any]] = []

    for run_id in paired:
        request = manifest.get(run_id, {})
        ours_dir = args.ours_root / run_id
        run_manifest = load_json(ours_dir / args.run_manifest_name) if (ours_dir / args.run_manifest_name).exists() else {}
        selected_cells = cells_from_payload(request, run_manifest)
        selected_cell_set = {(int(cell["gx"]), int(cell["gy"])) for cell in selected_cells}
        grid_step_px = int(request.get("grid_step_px") or run_manifest.get("grid_step_px") or args.grid_step_px)
        meta = run_metadata(request)

        source_path = ours_dir / args.source_image_name
        if not source_path.exists():
            source_path = args.naive_root / run_id / args.source_image_name
        if not source_path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing source image for {run_id}: {source_path}")
        source = load_rgb(source_path)
        selected_mask = cell_mask(source.shape, selected_cells, grid_step_px)
        full_mask = np.ones(source.shape[:2], dtype=bool)
        unselected_mask = ~selected_mask if bool(selected_mask.any()) else np.zeros(source.shape[:2], dtype=bool)
        for method, root in method_roots:
            run_dir = root / run_id
            generated_path = run_dir / args.generated_image_name
            if not generated_path.exists():
                if args.allow_missing:
                    continue
                raise FileNotFoundError(f"Missing generated image for {method}/{run_id}: {generated_path}")
            generated = load_rgb(generated_path)
            if generated.shape != source.shape:
                raise ValueError(f"Image shape mismatch for {method}/{run_id}: {source.shape} vs {generated.shape}")
            method_manifest = load_json(run_dir / args.run_manifest_name) if (run_dir / args.run_manifest_name).exists() else run_manifest
            method_window_history = window_history_from_manifest(method_manifest)
            method_commit_mask = commit_mask_from_steps(source.shape, method_window_history)
            method_outside_commit_mask = (
                ~method_commit_mask if bool(method_commit_mask.any()) else np.zeros(source.shape[:2], dtype=bool)
            )
            area_masks = {
                "full_region": full_mask,
                "selected_cells": selected_mask,
                "unselected_cells": unselected_mask,
                "committed_pixels": method_commit_mask,
                "outside_committed_pixels": method_outside_commit_mask,
            }
            for area, mask in area_masks.items():
                summary_rows.append(
                    {
                        **meta,
                        "run_id": run_id,
                        "method": method,
                        "area": area,
                        "source_image": str(source_path),
                        "generated_image": str(generated_path),
                        **summarize_diff(source, generated, mask),
                    }
                )
            if not bool(getattr(args, "no_per_cell", False)):
                cell_rows.extend(
                    per_cell_diff_rows(
                        run_id=run_id,
                        method=method,
                        source=source,
                        generated=generated,
                        selected_cells=selected_cells,
                        grid_step_px=grid_step_px,
                        run_meta=meta,
                    )
                )
            if not bool(getattr(args, "no_seam", False)):
                edited_cell_set = cell_set_from_payload_key(method_manifest, ("edited_cells",)) or cell_set_from_payload_key(
                    method_manifest, ("runtime_edited_cells",)
                )
                visited_cell_set = cell_set_from_payload_key(method_manifest, ("visited_cells",)) or cell_set_from_payload_key(
                    method_manifest, ("runtime_visited_cells",)
                )
                seam_rows.extend(
                    seam_rows_for_method(
                        run_id=run_id,
                        method=method,
                        source=source,
                        generated=generated,
                        grid_step_px=grid_step_px,
                        selected_cells=selected_cell_set,
                        edited_cells=edited_cell_set,
                        visited_cells=visited_cell_set,
                        steps=method_window_history,
                        run_meta=meta,
                    )
                )

    aggregate = aggregate_rows(summary_rows)
    paired_rows = pairwise_rows(summary_rows, str(args.ours_name), str(args.naive_name))
    paired_aggregate = aggregate_pairwise(paired_rows)
    seam_aggregate = aggregate_seam_rows(seam_rows)
    seam_pairwise = pairwise_seam_rows(seam_rows, str(args.ours_name), str(args.naive_name))
    seam_pairwise_aggregate = aggregate_pairwise_seam(seam_pairwise)
    return {
        "manifest": manifest,
        "paired": paired,
        "missing": missing,
        "summary_rows": summary_rows,
        "cell_rows": cell_rows,
        "aggregate": aggregate,
        "paired_rows": paired_rows,
        "paired_aggregate": paired_aggregate,
        "seam_rows": seam_rows,
        "seam_aggregate": seam_aggregate,
        "seam_pairwise": seam_pairwise,
        "seam_pairwise_aggregate": seam_pairwise_aggregate,
    }


def write_visual_outputs(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "visual_perturbation_by_run.csv", list(result["summary_rows"]))
    if not bool(getattr(args, "no_per_cell", False)):
        write_csv(args.out_dir / "visual_perturbation_per_cell.csv", list(result["cell_rows"]))
    write_csv(args.out_dir / "visual_perturbation_summary.csv", list(result["aggregate"]))
    write_csv(args.out_dir / "visual_perturbation_paired_by_run.csv", list(result["paired_rows"]))
    write_csv(args.out_dir / "visual_perturbation_paired_summary.csv", list(result["paired_aggregate"]))
    if not bool(getattr(args, "no_seam", False)):
        write_csv(args.out_dir / "border_inconsistency_by_run.csv", list(result["seam_rows"]))
        write_csv(args.out_dir / "border_inconsistency_summary.csv", list(result["seam_aggregate"]))
        write_csv(args.out_dir / "border_inconsistency_paired_by_run.csv", list(result["seam_pairwise"]))
        write_csv(args.out_dir / "border_inconsistency_paired_summary.csv", list(result["seam_pairwise_aggregate"]))

    formats = [fmt.strip() for fmt in str(args.formats).split(",") if fmt.strip()]
    save_bar_plot(list(result["aggregate"]), args.out_dir, str(args.title), formats)

    payload = {
        "ours_root": str(args.ours_root),
        "naive_root": str(args.naive_root),
        "manifest": str(args.manifest) if args.manifest is not None else "",
        "out_dir": str(args.out_dir),
        "n_paired_runs": int(len(result["paired"])),
        "n_missing_or_unpaired": int(len(result["missing"])),
        "missing_or_unpaired_preview": list(result["missing"])[:20],
        "outputs": {
            "by_run": str(args.out_dir / "visual_perturbation_by_run.csv"),
            "per_cell": "" if bool(getattr(args, "no_per_cell", False)) else str(args.out_dir / "visual_perturbation_per_cell.csv"),
            "summary": str(args.out_dir / "visual_perturbation_summary.csv"),
            "paired_by_run": str(args.out_dir / "visual_perturbation_paired_by_run.csv"),
            "paired_summary": str(args.out_dir / "visual_perturbation_paired_summary.csv"),
            "border_by_run": "" if bool(getattr(args, "no_seam", False)) else str(args.out_dir / "border_inconsistency_by_run.csv"),
            "border_summary": "" if bool(getattr(args, "no_seam", False)) else str(args.out_dir / "border_inconsistency_summary.csv"),
            "border_paired_by_run": "" if bool(getattr(args, "no_seam", False)) else str(args.out_dir / "border_inconsistency_paired_by_run.csv"),
            "border_paired_summary": "" if bool(getattr(args, "no_seam", False)) else str(args.out_dir / "border_inconsistency_paired_summary.csv"),
            "plot_stem": str(args.out_dir / "visual_perturbation_summary"),
        },
    }
    (args.out_dir / "visual_perturbation_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def compare(args: argparse.Namespace) -> dict[str, Any]:
    return write_visual_outputs(args, collect_visual_rows(args))


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(compare(args), indent=2))


if __name__ == "__main__":
    main()
