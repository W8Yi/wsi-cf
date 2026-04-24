from __future__ import annotations

from typing import Any

import numpy as np


def flatten_region_zgrid(z_grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arr = np.asarray(z_grid, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"z_grid must have shape [Gh,Gw,D], got {tuple(arr.shape)}")
    grid_h, grid_w, dim = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    bag = arr.reshape(grid_h * grid_w, dim).astype(np.float32, copy=False)
    tile_rows: list[dict[str, Any]] = []
    tile_idx = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            tile_rows.append(
                {
                    "tile_index": int(tile_idx),
                    "cell_gx": int(gx),
                    "cell_gy": int(gy),
                    "coord_x": -1,
                    "coord_y": -1,
                    "is_high_attention": False,
                }
            )
            tile_idx += 1
    return bag, tile_rows


def select_attention_cells(
    *,
    attention: np.ndarray,
    grid_w: int,
    grid_h: int,
    mode: str,
    top_k: int,
    percentile: float,
    min_cells: int,
    max_cells: int,
    allowed_cells: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    attn = np.asarray(attention, dtype=np.float32).reshape(-1)
    expected = int(grid_w) * int(grid_h)
    if attn.shape[0] != expected:
        raise ValueError(f"attention length {attn.shape[0]} does not match grid size {expected}")
    allowed_idx: list[int] | None = None
    if allowed_cells is not None:
        allowed_norm = {
            (int(gx), int(gy))
            for gx, gy in allowed_cells
            if 0 <= int(gx) < int(grid_w) and 0 <= int(gy) < int(grid_h)
        }
        if not allowed_norm:
            raise ValueError("allowed_cells must contain at least one valid in-grid cell")
        allowed_idx = sorted(int(gy) * int(grid_w) + int(gx) for gx, gy in allowed_norm)
        attn_work = np.full_like(attn, fill_value=-np.inf, dtype=np.float32)
        attn_work[np.asarray(allowed_idx, dtype=np.int64)] = attn[np.asarray(allowed_idx, dtype=np.int64)]
    else:
        attn_work = attn
    selected_idx: list[int]
    mode_norm = str(mode).strip().lower()
    if mode_norm == "topk":
        k_total = int(len(allowed_idx)) if allowed_idx is not None else int(attn.shape[0])
        k = max(1, min(int(top_k), k_total))
        selected_idx = np.argsort(-attn_work)[:k].astype(np.int64).tolist()
    elif mode_norm == "percentile":
        pct = float(percentile)
        if not (0.0 <= pct <= 100.0):
            raise ValueError("percentile must be in [0,100]")
        base = attn[np.asarray(allowed_idx, dtype=np.int64)] if allowed_idx is not None else attn
        threshold = float(np.percentile(base, pct))
        ranked = np.argsort(-attn_work).astype(np.int64).tolist()
        selected_idx = [int(idx) for idx in ranked if np.isfinite(attn_work[int(idx)]) and float(attn_work[int(idx)]) >= threshold]
        if int(min_cells) > 0 and len(selected_idx) < int(min_cells):
            k_floor = min(int(min_cells), int(len(allowed_idx)) if allowed_idx is not None else int(attn.shape[0]))
            selected_idx = [int(idx) for idx in ranked if np.isfinite(attn_work[int(idx)])][:k_floor]
        if int(max_cells) > 0 and len(selected_idx) > int(max_cells):
            selected_idx = selected_idx[: int(max_cells)]
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for idx in selected_idx:
        gx = int(idx) % int(grid_w)
        gy = int(idx) // int(grid_w)
        cell = (gx, gy)
        if cell not in seen:
            cells.append(cell)
            seen.add(cell)
    cells.sort(key=lambda item: (item[1], item[0]))
    return cells


def true_label_confidence(*, label: int, prob_pos: float) -> float:
    if int(label) == 1:
        return float(prob_pos)
    return float(1.0 - float(prob_pos))


def passes_label_confidence(*, label: int, pred: int, prob_pos: float, min_confidence: float, require_label_match: bool) -> tuple[bool, str, float]:
    confidence = true_label_confidence(label=int(label), prob_pos=float(prob_pos))
    if bool(require_label_match) and int(pred) != int(label):
        return False, "pred_label_mismatch", float(confidence)
    if confidence < float(min_confidence):
        return False, "low_label_confidence", float(confidence)
    return True, "eligible_label_confidence", float(confidence)


def count_high_attention_cells(
    *,
    attention: np.ndarray,
    grid_w: int,
    grid_h: int,
    percentile: float,
    allowed_cells: set[tuple[int, int]] | None = None,
) -> tuple[int, float]:
    attn = np.asarray(attention, dtype=np.float32).reshape(-1)
    expected = int(grid_w) * int(grid_h)
    if attn.shape[0] != expected:
        raise ValueError(f"attention length {attn.shape[0]} does not match grid size {expected}")
    pct = float(percentile)
    if not (0.0 <= pct <= 100.0):
        raise ValueError("percentile must be in [0,100]")
    threshold = float(np.percentile(attn, pct))
    cells = allowed_cells
    if cells is None:
        cells = {(gx, gy) for gy in range(int(grid_h)) for gx in range(int(grid_w))}
    count = 0
    for gx, gy in cells:
        if not (0 <= int(gx) < int(grid_w) and 0 <= int(gy) < int(grid_h)):
            continue
        idx = int(gy) * int(grid_w) + int(gx)
        if float(attn[idx]) >= threshold:
            count += 1
    return int(count), float(threshold)


def select_attention_mass_cells(
    *,
    attention: np.ndarray,
    grid_w: int,
    grid_h: int,
    target_mass: float,
    min_cells: int,
    max_cells: int,
    allowed_cells: set[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], float, bool, str]:
    attn = np.asarray(attention, dtype=np.float32).reshape(-1)
    expected = int(grid_w) * int(grid_h)
    if attn.shape[0] != expected:
        raise ValueError(f"attention length {attn.shape[0]} does not match grid size {expected}")
    target = float(target_mass)
    if not (0.0 <= target <= 1.0):
        raise ValueError("target_mass must be in [0,1]")
    min_n = max(1, int(min_cells))
    max_n = max(min_n, int(max_cells))
    if allowed_cells is None:
        allowed = {(gx, gy) for gy in range(int(grid_h)) for gx in range(int(grid_w))}
    else:
        allowed = {
            (int(gx), int(gy))
            for gx, gy in allowed_cells
            if 0 <= int(gx) < int(grid_w) and 0 <= int(gy) < int(grid_h)
        }
    if not allowed:
        return [], 0.0, False, "no_allowed_cells"

    ranked = sorted(
        allowed,
        key=lambda cell: (-float(attn[int(cell[1]) * int(grid_w) + int(cell[0])]), int(cell[1]), int(cell[0])),
    )
    selected: list[tuple[int, int]] = []
    mass = 0.0
    for cell in ranked[:max_n]:
        selected.append((int(cell[0]), int(cell[1])))
        mass += float(attn[int(cell[1]) * int(grid_w) + int(cell[0])])
        if len(selected) >= min_n and mass >= target:
            break
    selected.sort(key=lambda item: (item[1], item[0]))
    if len(selected) < min_n:
        return selected, float(mass), False, "too_few_selected_cells"
    if mass < target:
        return selected, float(mass), False, "insufficient_attention_mass"
    return selected, float(mass), True, "eligible_attention_mass"


def select_balanced_region_rows(
    rows: list[dict[str, Any]],
    *,
    per_label: int,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    selected: list[dict[str, Any]] = []
    eligible_counts: dict[int, int] = {0: 0, 1: 0}
    for label in (0, 1):
        label_rows = [
            row
            for row in rows
            if int(row["label"]) == int(label) and bool(row.get("eligible", False))
        ]
        label_rows.sort(key=lambda row: (str(row.get("slide_key", "")), str(row.get("region_id", ""))))
        eligible_counts[label] = int(len(label_rows))
        selected.extend(label_rows[: int(per_label)])
    return selected, eligible_counts


def summarize_region_classifier_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = int(len(rows))
    if total == 0:
        return {
            "n_regions": 0,
            "accuracy_before": 0.0,
            "accuracy_after": 0.0,
            "target_pred_rate_after": 0.0,
            "mean_target_prob_before": 0.0,
            "mean_target_prob_after": 0.0,
            "mean_delta_target_prob": 0.0,
            "target_shift_success_rate": 0.0,
        }
    acc_before = np.mean([float(bool(int(row["pred_before"]) == int(row["source_label"]))) for row in rows])
    acc_after = np.mean([float(bool(int(row["pred_after"]) == int(row["source_label"]))) for row in rows])
    target_pred_rate_after = np.mean([float(bool(int(row["pred_after"]) == int(row["target_label"]))) for row in rows])
    target_prob_before = np.asarray([float(row["target_prob_before"]) for row in rows], dtype=np.float32)
    target_prob_after = np.asarray([float(row["target_prob_after"]) for row in rows], dtype=np.float32)
    return {
        "n_regions": total,
        "accuracy_before": float(acc_before),
        "accuracy_after": float(acc_after),
        "target_pred_rate_after": float(target_pred_rate_after),
        "mean_target_prob_before": float(target_prob_before.mean()),
        "mean_target_prob_after": float(target_prob_after.mean()),
        "mean_delta_target_prob": float((target_prob_after - target_prob_before).mean()),
        "target_shift_success_rate": float(np.mean((target_prob_after > target_prob_before).astype(np.float32))),
    }


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
