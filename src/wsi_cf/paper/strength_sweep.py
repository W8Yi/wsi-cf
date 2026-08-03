from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_STRENGTHS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def parse_strengths(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        values = [float(token.strip()) for token in value.split(",") if token.strip()]
    else:
        values = [float(item) for item in value]
    if not values:
        raise ValueError("At least one steering strength is required")
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in values):
        raise ValueError("Steering strengths must be finite values in [0, 1]")
    return tuple(sorted(set(values)))


def strength_slug(strength: float) -> str:
    return f"strength_{int(round(float(strength) * 100)):03d}"


def read_json_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        requests = payload
    elif isinstance(payload, dict):
        requests = payload.get("edit_requests") or payload.get("requests") or []
    else:
        raise ValueError(f"Unsupported edit manifest payload in {path}: {type(payload)}")
    if not all(isinstance(row, dict) for row in requests):
        raise ValueError(f"Every edit request in {path} must be an object")
    return [dict(row) for row in requests]


def read_region_bank(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="") as handle:
        return {str(row["region_id"]): dict(row) for row in csv.DictReader(handle)}


def _normalized_label(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "hpv+": "hpv_pos",
        "hpv-positive": "hpv_pos",
        "positive": "hpv_pos",
        "hpv-": "hpv_neg",
        "hpv-negative": "hpv_neg",
        "negative": "hpv_neg",
    }
    return aliases.get(text, text)


def _matches_source_label(row: dict[str, Any], source_label: str) -> bool:
    if not source_label:
        return True
    expected = _normalized_label(source_label)
    fields = (
        "source_label",
        "label_name",
        "hpv_status",
        "grade_group",
        "morphology_group",
        "label",
    )
    return any(_normalized_label(row.get(field)) == expected for field in fields if str(row.get(field, "")).strip())


def select_random_requests(
    requests: Sequence[dict[str, Any]],
    region_by_id: dict[str, dict[str, str]],
    *,
    seed: int,
    n_regions: int,
    region_size: int = 2048,
    source_label: str = "",
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for request in requests:
        region_id = str(request.get("region_id", ""))
        region = region_by_id.get(region_id)
        if region is None:
            continue
        width = int(float(region.get("region_w") or region_size))
        height = int(float(region.get("region_h") or region_size))
        if width != int(region_size) or height != int(region_size):
            continue
        if source_label and not (_matches_source_label(request, source_label) or _matches_source_label(region, source_label)):
            continue
        if not request.get("target_cells"):
            continue
        eligible.append(dict(request))
    eligible.sort(key=lambda row: (str(row.get("region_id", "")), str(row.get("run_id", ""))))
    if len(eligible) < int(n_regions):
        raise ValueError(
            f"Requested {n_regions} random {region_size}x{region_size} region(s), "
            f"but only {len(eligible)} eligible manifest request(s) were found"
        )
    rng = random.Random(int(seed))
    indices = sorted(rng.sample(range(len(eligible)), k=int(n_regions)))
    return [eligible[index] for index in indices]


def expand_requests_to_all_region_cells(
    requests: Sequence[dict[str, Any]],
    region_by_id: dict[str, dict[str, str]],
    *,
    region_size: int = 2048,
    default_grid_step_px: int = 256,
) -> list[dict[str, Any]]:
    """Replace each request mask with every grid cell in its region."""
    output: list[dict[str, Any]] = []
    for request in requests:
        region_id = str(request.get("region_id", ""))
        region = region_by_id.get(region_id)
        if region is None:
            raise ValueError(f"Region {region_id!r} is missing from the region bank")
        width = int(float(region.get("region_w") or region_size))
        height = int(float(region.get("region_h") or region_size))
        grid_step_px = int(float(region.get("grid_step_px") or default_grid_step_px))
        if width <= 0 or height <= 0 or grid_step_px <= 0:
            raise ValueError(
                f"Region {region_id!r} has invalid geometry: "
                f"{width}x{height} with grid_step_px={grid_step_px}"
            )
        if width % grid_step_px != 0 or height % grid_step_px != 0:
            raise ValueError(
                f"Region {region_id!r} geometry is not divisible by grid_step_px: "
                f"{width}x{height} with grid_step_px={grid_step_px}"
            )
        target_cells = [
            {"gx": gx, "gy": gy}
            for gy in range(height // grid_step_px)
            for gx in range(width // grid_step_px)
        ]
        row = dict(request)
        row["base_selector"] = str(row.get("selector", ""))
        row["selector"] = "all_region_cells"
        row["target_cell_mode"] = "all_region_cells"
        row["target_cells"] = target_cells
        row["valid_cell_count"] = len(target_cells)
        output.append(row)
    return output


def expand_requests_to_valid_feature_cells(
    requests: Sequence[dict[str, Any]],
    region_by_id: dict[str, dict[str, str]],
    *,
    root: Path | None = None,
    center_margin_cells: int = 0,
) -> list[dict[str, Any]]:
    """Replace each request mask with valid feature cells, optionally inset from every edge."""
    margin = int(center_margin_cells)
    if margin < 0:
        raise ValueError("center_margin_cells must be >= 0")
    output: list[dict[str, Any]] = []
    for request in requests:
        region_id = str(request.get("region_id", ""))
        region = region_by_id.get(region_id)
        if region is None:
            raise ValueError(f"Region {region_id!r} is missing from the region bank")
        feature_grid_path = Path(str(region.get("feature_grid_path", "")))
        if root is not None and not feature_grid_path.is_absolute():
            feature_grid_path = Path(root) / feature_grid_path
        valid_mask_path = feature_grid_path.with_name("valid_feature_mask.npy")
        if not valid_mask_path.is_file():
            raise FileNotFoundError(
                f"Region {region_id!r} is missing its valid feature mask: {valid_mask_path}"
            )
        valid_mask = np.asarray(np.load(valid_mask_path), dtype=bool)
        if valid_mask.ndim != 2:
            raise ValueError(
                f"Region {region_id!r} valid feature mask must be 2D, got {valid_mask.shape}"
            )
        grid_h, grid_w = (int(valid_mask.shape[0]), int(valid_mask.shape[1]))
        if 2 * margin >= grid_w or 2 * margin >= grid_h:
            raise ValueError(
                f"center_margin_cells={margin} leaves no cells in region {region_id!r} "
                f"with grid shape {valid_mask.shape}"
            )
        target_cells = [
            {"gx": gx, "gy": gy}
            for gy in range(margin, grid_h - margin)
            for gx in range(margin, grid_w - margin)
            if bool(valid_mask[gy, gx])
        ]
        if not target_cells:
            raise ValueError(
                f"Region {region_id!r} has no valid cells after center_margin_cells={margin}"
            )
        mode = "all_valid_feature_cells" if margin == 0 else "center_valid_feature_cells"
        row = dict(request)
        row["base_selector"] = str(row.get("selector", ""))
        row["selector"] = mode
        row["target_cell_mode"] = mode
        row["target_cells"] = target_cells
        row["valid_cell_count"] = len(target_cells)
        row["valid_feature_mask_path"] = str(valid_mask_path)
        row["center_margin_cells"] = margin
        output.append(row)
    return output


def expand_requests_to_random_valid_blocks(
    requests: Sequence[dict[str, Any]],
    region_by_id: dict[str, dict[str, str]],
    *,
    block_side_cells: int,
    seed: int,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Select one deterministic random square block containing only valid feature cells."""
    side = int(block_side_cells)
    if side <= 0:
        raise ValueError("block_side_cells must be > 0")
    output: list[dict[str, Any]] = []
    for request in requests:
        region_id = str(request.get("region_id", ""))
        region = region_by_id.get(region_id)
        if region is None:
            raise ValueError(f"Region {region_id!r} is missing from the region bank")
        feature_grid_path = Path(str(region.get("feature_grid_path", "")))
        if root is not None and not feature_grid_path.is_absolute():
            feature_grid_path = Path(root) / feature_grid_path
        valid_mask_path = feature_grid_path.with_name("valid_feature_mask.npy")
        if not valid_mask_path.is_file():
            raise FileNotFoundError(
                f"Region {region_id!r} is missing its valid feature mask: {valid_mask_path}"
            )
        valid_mask = np.asarray(np.load(valid_mask_path), dtype=bool)
        if valid_mask.ndim != 2:
            raise ValueError(
                f"Region {region_id!r} valid feature mask must be 2D, got {valid_mask.shape}"
            )
        grid_h, grid_w = (int(valid_mask.shape[0]), int(valid_mask.shape[1]))
        if side > grid_w or side > grid_h:
            raise ValueError(
                f"block_side_cells={side} exceeds region {region_id!r} grid shape {valid_mask.shape}"
            )
        candidates = [
            (gx0, gy0)
            for gy0 in range(grid_h - side + 1)
            for gx0 in range(grid_w - side + 1)
            if bool(valid_mask[gy0 : gy0 + side, gx0 : gx0 + side].all())
        ]
        if not candidates:
            raise ValueError(
                f"Region {region_id!r} has no fully valid {side}x{side} feature-cell block"
            )
        digest = hashlib.sha256(f"{int(seed)}:{region_id}:{side}".encode("utf-8")).digest()
        gx0, gy0 = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
        target_cells = [
            {"gx": gx, "gy": gy}
            for gy in range(int(gy0), int(gy0) + side)
            for gx in range(int(gx0), int(gx0) + side)
        ]
        row = dict(request)
        row["base_selector"] = str(row.get("selector", ""))
        row["selector"] = "random_valid_block"
        row["target_cell_mode"] = "random_valid_block"
        row["target_cells"] = target_cells
        row["valid_cell_count"] = len(target_cells)
        row["valid_feature_mask_path"] = str(valid_mask_path)
        row["random_valid_block_side_cells"] = side
        row["random_valid_block_top_left"] = {"gx": int(gx0), "gy": int(gy0)}
        row["random_valid_block_candidate_count"] = len(candidates)
        row["random_valid_block_seed"] = int(seed)
        output.append(row)
    return output


def prepare_selected_requests(
    requests: Sequence[dict[str, Any]],
    *,
    task_name: str,
    direction: str,
    source_label: str,
    target_label: str,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for request in requests:
        row = dict(request)
        target_cells = list(row["target_cells"])
        row.update(
            {
                "task_name": str(task_name),
                "direction": str(direction),
                "source_label": str(source_label),
                "target_label": str(target_label),
                "valid_cell_count": int(row.get("valid_cell_count") or max(len(target_cells), 1)),
                "sweep_seed": int(seed),
            }
        )
        output.append(row)
    return output


def manifest_digest(requests: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(list(requests), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and arr[order[end]] == arr[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    if x_rank.size < 2 or np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def monotonicity_record(
    strengths: Sequence[float],
    values: Sequence[float],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    x = np.asarray(strengths, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    diffs = np.diff(y)
    comparable = int(diffs.size)
    violations = int(np.sum(diffs < -abs(float(tolerance))))
    return {
        "n_points": int(y.size),
        "n_adjacent_pairs": comparable,
        "n_decreases": violations,
        "adjacent_nondecreasing_fraction": float((comparable - violations) / comparable) if comparable else float("nan"),
        "strictly_monotonic_nondecreasing": bool(violations == 0 and comparable > 0),
        "spearman_rho": spearman_correlation(x, y),
        "start_value": float(y[0]) if y.size else float("nan"),
        "end_value": float(y[-1]) if y.size else float("nan"),
        "total_change": float(y[-1] - y[0]) if y.size else float("nan"),
    }


def summarize_monotonicity(
    rows: Sequence[dict[str, Any]],
    *,
    value_columns: Sequence[str] = ("target_probability", "concept_activation"),
    group_columns: Sequence[str] = ("task_name", "direction", "region_id", "seed"),
    sweep_column: str = "steering_strength",
    tolerance: float = 1e-8,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(column, "")) for column in group_columns)].append(dict(row))
    output: list[dict[str, Any]] = []
    for group_key, values in sorted(grouped.items()):
        values.sort(key=lambda row: float(row[sweep_column]))
        base = {column: value for column, value in zip(group_columns, group_key)}
        for value_column in value_columns:
            pairs = [
                (float(row[sweep_column]), float(row[value_column]))
                for row in values
                if row.get(value_column, "") not in ("", None)
            ]
            if not pairs:
                continue
            record = monotonicity_record(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                tolerance=tolerance,
            )
            output.append({**base, "metric": value_column, **record})
    return output


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields: list[str] = []
    seen: set[str] = set()
    for row in values:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(values)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
