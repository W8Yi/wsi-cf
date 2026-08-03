#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score prediction-transition generated edits either on the local region bag "
            "or by replacing selected region cells inside the original whole-slide feature bag."
        )
    )
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--edit-manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--classifier-run-dir", type=Path, default=None)
    parser.add_argument("--classifier-ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--direction", default="")
    parser.add_argument("--target-label", default="")
    parser.add_argument("--label-order", default="", help="Comma-separated label order fallback when label_mapping.json is absent.")
    parser.add_argument("--grade-label-order", default="GG1,GG2,GG3,GG4,GG5")
    parser.add_argument(
        "--score-scope",
        choices=["slide_bag", "local_region"],
        default="slide_bag",
        help=(
            "slide_bag replaces generated cells into the full WSI feature bag; "
            "local_region scores only the local region grid."
        ),
    )
    parser.add_argument(
        "--local-source",
        choices=["source_image", "feature_grid"],
        default="feature_grid",
        help=(
            "For --score-scope local_region, score the source from the rendered "
            "source_region_actual.png or from the region-bank feature_grid_path."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--skip-source-grid-check", action="store_true")
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def read_region_bank(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="") as handle:
        return {str(row["region_id"]): dict(row) for row in csv.DictReader(handle)}


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WSI_CF_ROOT / path


def infer_coord_step(coords: np.ndarray, fallback: int = 256) -> int:
    candidates: list[int] = []
    arr = np.asarray(coords, dtype=np.int64)
    for axis in (0, 1):
        values = np.unique(arr[:, axis])
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    return int(min(candidates)) if candidates else int(fallback)


def aggregate_pt_features_to_supercells(
    features: np.ndarray,
    coords: np.ndarray,
    *,
    tile_size_level0: int,
    block_ratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = int(block_ratio)
    if ratio <= 1:
        return np.asarray(features, dtype=np.float32), np.asarray(coords, dtype=np.int64)
    super_members: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(np.asarray(coords, dtype=np.int64).tolist()):
        raw_gx = int(round(int(x) / float(tile_size_level0)))
        raw_gy = int(round(int(y) / float(tile_size_level0)))
        cell = (raw_gx // ratio, raw_gy // ratio)
        super_members.setdefault(cell, []).append(int(idx))
    ordered_cells = sorted(super_members.keys(), key=lambda cell: (int(cell[1]), int(cell[0])))
    agg_features = np.zeros((len(ordered_cells), int(features.shape[1])), dtype=np.float32)
    agg_coords = np.zeros((len(ordered_cells), 2), dtype=np.int64)
    agg_tile = int(tile_size_level0) * ratio
    for out_idx, cell in enumerate(ordered_cells):
        member_idx = super_members[cell]
        agg_features[out_idx] = np.asarray(features[member_idx], dtype=np.float32).mean(axis=0)
        agg_coords[out_idx] = np.asarray([int(cell[0]) * agg_tile, int(cell[1]) * agg_tile], dtype=np.int64)
    return agg_features, agg_coords


def load_feature_bag(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        features = np.asarray(handle["features"], dtype=np.float32)
        coords = np.asarray(handle["coords"], dtype=np.int64)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    return features, coords


def load_feature_bag_for_region(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    feature_path = resolve_repo_path(row.get("canonical_h5_path") or row.get("feature_path") or "")
    if feature_path.suffix.lower() in {".pt", ".pth"}:
        import torch

        coords_path = resolve_repo_path(row.get("coords_path", ""))
        if not coords_path.is_file():
            raise FileNotFoundError(f"{row.get('region_id', '<unknown>')}: missing coords_path for PT features: {coords_path}")
        feats = torch.load(feature_path, map_location="cpu")
        if not isinstance(feats, torch.Tensor):
            raise TypeError(f"Unexpected PT payload in {feature_path}: {type(feats)}")
        with h5py.File(coords_path, "r") as handle:
            coords = np.asarray(handle["coords"][:], dtype=np.int64)
        features = feats.detach().cpu().float().numpy().astype(np.float32, copy=False)
        raw_step = int(float(row.get("raw_coord_tile_size_level0") or 0)) or infer_coord_step(coords)
        ratio = int(float(row.get("clam_aggregate_ratio") or 1))
        if ratio > 1:
            features, coords = aggregate_pt_features_to_supercells(
                features,
                coords,
                tile_size_level0=int(raw_step),
                block_ratio=int(ratio),
            )
        if int(features.shape[0]) != int(coords.shape[0]):
            raise ValueError(f"{feature_path}: feature/coord count mismatch {features.shape[0]} != {coords.shape[0]}")
        return features, coords
    return load_feature_bag(feature_path)


def parse_label_order(value: str) -> tuple[dict[str, int], dict[int, str]]:
    labels = [token.strip() for token in str(value).split(",") if token.strip()]
    if not labels:
        raise ValueError("A label order is required when label_mapping.json is missing")
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return label_to_id, {idx: label for label, idx in label_to_id.items()}


def resolve_classifier_ckpt(classifier_run_dir: Path | None, classifier_ckpt: Path | None) -> Path:
    if classifier_ckpt is not None:
        return classifier_ckpt
    if classifier_run_dir is None:
        raise ValueError("Provide --classifier-run-dir or --classifier-ckpt")
    return classifier_run_dir / "best_model.pt"


def load_label_mapping(classifier_run_dir: Path | None, label_order: str) -> tuple[dict[str, int], dict[int, str]]:
    path = classifier_run_dir / "label_mapping.json" if classifier_run_dir is not None else None
    if path is not None and path.exists():
        payload = read_json(path)
        label_to_id = {str(k): int(v) for k, v in payload["label_to_id"].items()}
        return label_to_id, {idx: label for label, idx in label_to_id.items()}
    return parse_label_order(label_order)


def score_classifier(
    model: Any,
    features: np.ndarray,
    *,
    id_to_label: dict[int, str],
    device: Any,
) -> tuple[str, np.ndarray]:
    import torch

    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.inference_mode():
        _, y_prob, y_hat, _, _ = model(x)
        pred_id = int(y_hat.detach().cpu().reshape(-1)[0].item())
        probs = y_prob.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return id_to_label.get(pred_id, str(pred_id)), probs


def expected_grade(probs: np.ndarray, id_to_label: dict[int, str], grade_label_order: str) -> float | None:
    labels = [token.strip() for token in str(grade_label_order).split(",") if token.strip()]
    grade_value_by_label = {label: float(index + 1) for index, label in enumerate(labels)}
    total = 0.0
    used = 0
    for idx, prob in enumerate(np.asarray(probs, dtype=np.float64).reshape(-1)):
        label = id_to_label.get(int(idx), str(idx))
        if label not in grade_value_by_label:
            return None
        total += float(prob) * grade_value_by_label[label]
        used += 1
    return float(total) if used else None


def target_cells_from_request(request: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(cell["gx"]), int(cell["gy"]))
        for cell in request.get("target_cells", [])
    }


def replace_local_grid_cells(
    source_grid: np.ndarray,
    generated_grid: np.ndarray,
    cells: set[tuple[int, int]],
) -> tuple[np.ndarray, int]:
    edited = np.asarray(source_grid, dtype=np.float32).copy()
    generated = np.asarray(generated_grid, dtype=np.float32)
    height = min(int(edited.shape[0]), int(generated.shape[0]))
    width = min(int(edited.shape[1]), int(generated.shape[1]))
    replaced = 0
    for gx, gy in sorted(cells, key=lambda cell: (cell[1], cell[0])):
        if 0 <= int(gx) < width and 0 <= int(gy) < height:
            edited[int(gy), int(gx)] = generated[int(gy), int(gx)]
            replaced += 1
    return edited, int(replaced)


def replace_local_grid_all(source_grid: np.ndarray, generated_grid: np.ndarray) -> tuple[np.ndarray, int]:
    edited = np.asarray(source_grid, dtype=np.float32).copy()
    generated = np.asarray(generated_grid, dtype=np.float32)
    height = min(int(edited.shape[0]), int(generated.shape[0]))
    width = min(int(edited.shape[1]), int(generated.shape[1]))
    edited[:height, :width] = generated[:height, :width]
    return edited, int(height * width)


def flatten_grid_bag(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected region feature grid with shape (H, W, D), got {arr.shape}")
    return arr.reshape(-1, arr.shape[-1])


def safe_cache_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def summarize_group(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)
    summaries: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        deltas = np.asarray([float(row["target_probability_delta"]) for row in values], dtype=np.float64)
        out = {field: value for field, value in zip(group_fields, key)}
        out.update(
            {
                "n": int(len(values)),
                "mean_target_probability_delta": float(deltas.mean()),
                "median_target_probability_delta": float(np.median(deltas)),
                "positive_target_delta_rate": float(np.mean(deltas > 0.0)),
                "flip_to_target_rate": float(np.mean([int(row["flip_to_target"]) for row in values])),
                "target_prediction_rate_after_edit": float(np.mean([int(row["edited_pred_is_target"]) for row in values])),
                "mean_source_target_probability": float(np.mean([float(row["source_target_probability"]) for row in values])),
                "mean_edited_target_probability": float(np.mean([float(row["edited_target_probability"]) for row in values])),
            }
        )
        grade_deltas = [
            float(row["expected_grade_delta"])
            for row in values
            if row.get("expected_grade_delta", "") not in ("", None)
        ]
        if grade_deltas:
            out["mean_expected_grade_delta"] = float(np.mean(grade_deltas))
            out["median_expected_grade_delta"] = float(np.median(grade_deltas))
        summaries.append(out)
    return summaries


def transition_auc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["task_name"]),
                str(row["direction"]),
                str(row["selector"]),
                int(row["repeat_id"]),
                str(row["region_id"]),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (task_name, direction, selector, repeat_id, region_id), values in grouped.items():
        values = sorted(values, key=lambda row: float(row["budget_fraction"]))
        if len(values) < 2:
            continue
        x = np.asarray([float(row["budget_fraction"]) for row in values], dtype=np.float64)
        y = np.asarray([float(row["target_probability_delta"]) for row in values], dtype=np.float64)
        out.append(
            {
                "task_name": task_name,
                "direction": direction,
                "selector": selector,
                "repeat_id": int(repeat_id),
                "region_id": region_id,
                "n_points": int(len(values)),
                "transition_auc": float(np.trapz(y, x)),
            }
        )
    return out


def random_vs_attention_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(str(row["task_name"]), str(row["direction"]), int(row["budget"]))][str(row["selector"])].append(row)
    out: list[dict[str, Any]] = []
    for (task_name, direction, budget), by_selector in sorted(grouped.items()):
        attn = by_selector.get("attention", [])
        rand = by_selector.get("random", [])
        if not attn or not rand:
            continue
        attn_mean = float(np.mean([float(row["target_probability_delta"]) for row in attn]))
        rand_mean = float(np.mean([float(row["target_probability_delta"]) for row in rand]))
        out.append(
            {
                "task_name": task_name,
                "direction": direction,
                "budget": int(budget),
                "n_attention": int(len(attn)),
                "n_random": int(len(rand)),
                "mean_attention_target_probability_delta": attn_mean,
                "mean_random_target_probability_delta": rand_mean,
                "attention_minus_random_delta": float(attn_mean - rand_mean),
            }
        )
    return out


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from PIL import Image

    from wsi_cf.common.runtime import resolve_device
    from wsi_cf.eval.grade_risk import map_region_cells_to_bag, replace_region_features
    from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint
    from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2

    device = resolve_device(args.device)
    region_by_id = read_region_bank(args.region_bank_csv)
    requests = read_json(args.edit_manifest)
    label_to_id, id_to_label = load_label_mapping(args.classifier_run_dir, str(args.label_order))
    model = build_mil_from_checkpoint(resolve_classifier_ckpt(args.classifier_run_dir, args.classifier_ckpt), device=device)
    uni_model = None
    uni_transform = None
    feature_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    result_rows: list[dict[str, Any]] = []
    missing_generated: list[str] = []

    for request in requests:
        region = region_by_id.get(str(request["region_id"]))
        if region is None:
            raise KeyError(f"Unknown region_id in manifest: {request['region_id']}")
        run_id = str(request["run_id"])
        generated_path = args.generated_root / run_id / "generated.png"
        if not generated_path.exists():
            missing_generated.append(str(generated_path))
            if bool(args.allow_missing):
                continue
            raise FileNotFoundError(f"Generated image not found: {generated_path}")
        cache_path = args.out_dir / "encoded_generated_grids" / f"{run_id}.npy"
        if cache_path.exists() and not bool(args.force_reencode):
            generated_grid = np.load(cache_path)
        else:
            if uni_model is None:
                uni_model, uni_transform = load_uni2(device)
            generated_grid = (
                build_uni_grid_from_image(
                    Image.open(generated_path),
                    uni_model=uni_model,
                    uni_transform=uni_transform,
                    grid_step_px=int(region["grid_step_px"]),
                    device=device,
                    out_dtype=torch.float32,
                )
                .detach()
                .cpu()
                .numpy()
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, generated_grid)

        source_grid_path = resolve_repo_path(region["feature_grid_path"])
        source_grid = np.load(source_grid_path)
        target_label = str(request.get("target_label") or args.target_label)
        target_id = int(label_to_id[target_label])
        target_cells = target_cells_from_request(request)

        if str(args.score_scope) == "local_region":
            if str(args.local_source) == "source_image":
                source_image_path = args.generated_root / run_id / "source_region_actual.png"
                if not source_image_path.exists():
                    raise FileNotFoundError(f"Source region image not found for local scoring: {source_image_path}")
                source_cache_path = args.out_dir / "encoded_source_grids" / f"{safe_cache_stem(str(request['region_id']))}.npy"
                if source_cache_path.exists() and not bool(args.force_reencode):
                    source_grid_for_scoring = np.load(source_cache_path)
                else:
                    if uni_model is None:
                        uni_model, uni_transform = load_uni2(device)
                    source_grid_for_scoring = (
                        build_uni_grid_from_image(
                            Image.open(source_image_path),
                            uni_model=uni_model,
                            uni_transform=uni_transform,
                            grid_step_px=int(region["grid_step_px"]),
                            device=device,
                            out_dtype=torch.float32,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    source_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(source_cache_path, source_grid_for_scoring)
            else:
                source_grid_for_scoring = source_grid
            edited_grid, replaced = replace_local_grid_cells(source_grid, generated_grid, target_cells)
            if str(args.local_source) == "source_image":
                edited_grid, replaced = replace_local_grid_cells(source_grid_for_scoring, generated_grid, target_cells)
            full_grid, full_replaced = replace_local_grid_all(source_grid_for_scoring, generated_grid)
            source_features = flatten_grid_bag(source_grid_for_scoring)
            edited_features = flatten_grid_bag(edited_grid)
            full_features = flatten_grid_bag(full_grid)
        else:
            feature_path = resolve_repo_path(region.get("canonical_h5_path") or region.get("feature_path") or "")
            coords_path = resolve_repo_path(region.get("coords_path", ""))
            cache_key = f"{feature_path.resolve()}::{coords_path.resolve() if coords_path.exists() else ''}"
            if cache_key not in feature_cache:
                feature_cache[cache_key] = load_feature_bag_for_region(region)
            source_features, coords = feature_cache[cache_key]
            cell_to_index = map_region_cells_to_bag(
                coords,
                region_gx0=int(region["region_gx0"]),
                region_gy0=int(region["region_gy0"]),
                grid_shape=(int(generated_grid.shape[0]), int(generated_grid.shape[1])),
            )
            if not bool(args.skip_source_grid_check):
                for (gx, gy), bag_index in cell_to_index.items():
                    if not np.allclose(source_grid[gy, gx], source_features[bag_index], atol=1e-5, rtol=1e-5):
                        raise ValueError(f"Source grid does not align with bag for {run_id} cell {(gx, gy)}")
            edited_features, replaced = replace_region_features(source_features, generated_grid, cell_to_index, cells=target_cells)
            full_features, full_replaced = replace_region_features(source_features, generated_grid, cell_to_index, cells=None)

        source_pred, source_probs = score_classifier(model, source_features, id_to_label=id_to_label, device=device)
        edited_pred, edited_probs = score_classifier(model, edited_features, id_to_label=id_to_label, device=device)
        full_pred, full_probs = score_classifier(model, full_features, id_to_label=id_to_label, device=device)
        source_expected = expected_grade(source_probs, id_to_label, str(args.grade_label_order))
        edited_expected = expected_grade(edited_probs, id_to_label, str(args.grade_label_order))
        full_expected = expected_grade(full_probs, id_to_label, str(args.grade_label_order))
        valid_count = int(request.get("valid_cell_count") or max(len(cell_to_index), 1))
        budget = int(request.get("budget") or len(target_cells))
        result_rows.append(
            {
                "task_name": str(request.get("task_name") or args.task_name),
                "direction": str(request.get("direction") or args.direction),
                "score_scope": str(args.score_scope),
                "local_source": str(args.local_source) if str(args.score_scope) == "local_region" else "",
                "selector": str(request.get("selector", "")),
                "repeat_id": int(request.get("repeat_id", -1)),
                "budget": int(budget),
                "budget_fraction": float(budget / max(valid_count, 1)),
                "budget_is_full": int(bool(request.get("budget_is_full", budget >= valid_count))),
                "valid_cell_count": int(valid_count),
                "run_id": run_id,
                "region_id": str(request["region_id"]),
                "slide_key": region.get("slide_key", ""),
                "source_label": str(request.get("source_label", "")),
                "target_label": target_label,
                "target_cells_requested": int(len(target_cells)),
                "target_cells_replaced": int(replaced),
                "full_region_cells_replaced": int(full_replaced),
                "source_pred_label": source_pred,
                "edited_pred_label": edited_pred,
                "full_region_pred_label": full_pred,
                "source_target_probability": float(source_probs[target_id]),
                "edited_target_probability": float(edited_probs[target_id]),
                "full_region_target_probability": float(full_probs[target_id]),
                "target_probability_delta": float(edited_probs[target_id] - source_probs[target_id]),
                "full_region_target_probability_delta": float(full_probs[target_id] - source_probs[target_id]),
                "edited_pred_is_target": int(edited_pred == target_label),
                "flip_to_target": int(source_pred != target_label and edited_pred == target_label),
                "source_expected_grade": "" if source_expected is None else float(source_expected),
                "edited_expected_grade": "" if edited_expected is None else float(edited_expected),
                "full_region_expected_grade": "" if full_expected is None else float(full_expected),
                "expected_grade_delta": "" if source_expected is None or edited_expected is None else float(edited_expected - source_expected),
                "full_region_expected_grade_delta": "" if source_expected is None or full_expected is None else float(full_expected - source_expected),
                "generated_path": str(generated_path),
                "encoded_grid_path": str(cache_path),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_budget = summarize_group(result_rows, ["task_name", "direction", "selector", "budget"])
    by_direction = summarize_group(result_rows, ["task_name", "direction", "selector"])
    random_vs = random_vs_attention_summary(result_rows)
    auc_rows = transition_auc_rows(result_rows)
    write_csv(args.out_dir / "prediction_transition_by_run.csv", result_rows)
    write_csv(args.out_dir / "prediction_transition_summary_by_budget.csv", by_budget)
    write_csv(args.out_dir / "prediction_transition_summary_by_direction.csv", by_direction)
    write_csv(args.out_dir / "random_vs_attention_summary.csv", random_vs)
    write_csv(args.out_dir / "prediction_transition_auc_by_region.csv", auc_rows)
    summary = {
        "task_name": str(args.task_name),
        "direction": str(args.direction),
        "region_bank_csv": str(args.region_bank_csv),
        "edit_manifest": str(args.edit_manifest),
        "generated_root": str(args.generated_root),
        "score_scope": str(args.score_scope),
        "local_source": str(args.local_source) if str(args.score_scope) == "local_region" else "",
        "classifier_run_dir": "" if args.classifier_run_dir is None else str(args.classifier_run_dir),
        "classifier_ckpt": "" if args.classifier_ckpt is None else str(args.classifier_ckpt),
        "n_requests": int(len(requests)),
        "n_scored": int(len(result_rows)),
        "n_missing_generated": int(len(missing_generated)),
        "missing_generated": missing_generated,
        "metrics": {
            "by_run": str(args.out_dir / "prediction_transition_by_run.csv"),
            "summary_by_budget": str(args.out_dir / "prediction_transition_summary_by_budget.csv"),
            "summary_by_direction": str(args.out_dir / "prediction_transition_summary_by_direction.csv"),
            "random_vs_attention": str(args.out_dir / "random_vs_attention_summary.csv"),
            "auc_by_region": str(args.out_dir / "prediction_transition_auc_by_region.csv"),
        },
    }
    write_json(args.out_dir / "benchmark_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(evaluate(build_arg_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
