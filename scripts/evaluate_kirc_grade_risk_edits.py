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
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json  # noqa: E402
from wsi_cf.common.runtime import resolve_device  # noqa: E402
from wsi_cf.eval.grade_risk import (  # noqa: E402
    load_grade_risk_model,
    map_region_cells_to_bag,
    replace_region_features,
    score_feature_bag,
)
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2  # noqa: E402


DEFAULT_EDIT_ROOT = WSI_CF_ROOT / "artifacts/morphology_label_concept_review_top1_showcase_best_10slides"
DEFAULT_LABEL_SOURCE = WSI_CF_ROOT / "resources/labels/master/slide_labels_master.tsv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate KIRC generated edits by continuous full-slide grade-risk shift.")
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        default=WSI_CF_ROOT / "artifacts/grade_risk_training/kirc_continuous_grade_risk/best_model.pt",
    )
    parser.add_argument("--edit-root", type=Path, default=DEFAULT_EDIT_ROOT)
    parser.add_argument("--label-source", type=Path, default=DEFAULT_LABEL_SOURCE)
    parser.add_argument(
        "--directions",
        type=str,
        default="03_kirc_low_to_high,03_kirc_high_to_low",
        help="Comma-separated edit direction subdirectories.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-runs-per-direction", type=int, default=10)
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--skip-source-grid-check", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else WSI_CF_ROOT / path


def load_slide_grades(path: Path) -> dict[str, str]:
    with path.open("r", newline="") as handle:
        return {
            str(row["slide_key"]): str(row.get("tumor_grade", ""))
            for row in csv.DictReader(handle, delimiter="\t")
            if str(row.get("project_dir", "")) == "TCGA-KIRC"
        }


def load_feature_bag(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        features = np.asarray(handle["features"], dtype=np.float32)
        coords = np.asarray(handle["coords"], dtype=np.int64)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if features.ndim != 2 or coords.ndim != 2:
        raise ValueError(f"Unexpected feature/coordinate shapes at {path}: {features.shape}, {coords.shape}")
    return features, coords


def load_direction_requests(edit_root: Path, direction: str, limit: int) -> list[tuple[dict[str, str], dict[str, Any]]]:
    region_dir = edit_root / "_regions" / direction
    manifest_path = region_dir / "progressive_edit_manifest_showcase_best.json"
    if not manifest_path.exists():
        manifest_path = region_dir / "progressive_edit_manifest.json"
    requests = list(read_json(manifest_path))
    with (region_dir / "region_bank.csv").open("r", newline="") as handle:
        region_rows = {str(row["region_id"]): row for row in csv.DictReader(handle)}
    out: list[tuple[dict[str, str], dict[str, Any]]] = []
    for request in requests[: int(limit)]:
        region = region_rows.get(str(request["region_id"]))
        if region is None:
            raise KeyError(f"Region {request['region_id']} in {manifest_path} is absent from region_bank.csv")
        out.append((region, request))
    return out


def desired_risk_sign(request: dict[str, Any]) -> int:
    target = str(request.get("target_label", "")).lower()
    if target == "high":
        return 1
    if target == "low":
        return -1
    raise ValueError(f"Unsupported grade-risk target label '{target}'")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["direction"])].append(row)
    summaries: list[dict[str, Any]] = []
    for direction, direction_rows in sorted(grouped.items()):
        full_delta = np.asarray([float(row["full_region_risk_delta"]) for row in direction_rows])
        target_delta = np.asarray([float(row["target_cells_risk_delta"]) for row in direction_rows])
        summaries.append(
            {
                "direction": direction,
                "n": len(direction_rows),
                "full_region_mean_delta": float(full_delta.mean()),
                "full_region_median_delta": float(np.median(full_delta)),
                "full_region_direction_success_rate": float(np.mean([int(row["full_region_direction_success"]) for row in direction_rows])),
                "target_cells_mean_delta": float(target_delta.mean()),
                "target_cells_median_delta": float(np.median(target_delta)),
                "target_cells_direction_success_rate": float(np.mean([int(row["target_cells_direction_success"]) for row in direction_rows])),
            }
        )
    return summaries


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    out_dir = args.out_dir or args.edit_root / "grade_risk_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.model_ckpt.exists():
        raise FileNotFoundError(
            f"Grade-risk checkpoint does not exist: {args.model_ckpt}. "
            "Train it first with scripts/train_grade_risk_regressor.py."
        )
    model, checkpoint = load_grade_risk_model(args.model_ckpt, device=device)
    slide_grades = load_slide_grades(args.label_source)
    uni_model = None
    uni_transform = None
    result_rows: list[dict[str, Any]] = []

    for direction in [value.strip() for value in str(args.directions).split(",") if value.strip()]:
        for region, request in load_direction_requests(args.edit_root, direction, int(args.max_runs_per_direction)):
            run_id = str(request["run_id"])
            run_dir = args.edit_root / direction / run_id
            generated_path = run_dir / "generated.png"
            if not generated_path.exists():
                raise FileNotFoundError(f"Generated edit image not found: {generated_path}")
            cache_path = out_dir / "encoded_generated_grids" / direction / f"{run_id}.npy"
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

            original_features, coords = load_feature_bag(Path(region["canonical_h5_path"]))
            source_grid = np.load(resolve_repo_path(region["feature_grid_path"]))
            cell_to_index = map_region_cells_to_bag(
                coords,
                region_gx0=int(region["region_gx0"]),
                region_gy0=int(region["region_gy0"]),
                grid_shape=(int(generated_grid.shape[0]), int(generated_grid.shape[1])),
            )
            if not args.skip_source_grid_check:
                for (gx, gy), bag_index in cell_to_index.items():
                    if not np.allclose(source_grid[gy, gx], original_features[bag_index], atol=1e-5, rtol=1e-5):
                        raise ValueError(f"Source grid does not align with full-slide feature bag for {run_id} cell {(gx, gy)}")
            target_cells = {(int(cell["gx"]), int(cell["gy"])) for cell in request.get("target_cells", [])}
            target_features, n_target_replaced = replace_region_features(
                original_features, generated_grid, cell_to_index, cells=target_cells
            )
            full_features, n_full_replaced = replace_region_features(original_features, generated_grid, cell_to_index)
            source_risk = score_feature_bag(model, original_features, device=device)
            target_risk = score_feature_bag(model, target_features, device=device)
            full_risk = score_feature_bag(model, full_features, device=device)
            sign = desired_risk_sign(request)
            result_rows.append(
                {
                    "direction": direction,
                    "run_id": run_id,
                    "slide_key": region["slide_key"],
                    "raw_grade": slide_grades.get(str(region["slide_key"]), ""),
                    "source_label": request.get("source_label", ""),
                    "target_label": request.get("target_label", ""),
                    "desired_risk_sign": sign,
                    "mapped_region_cells": len(cell_to_index),
                    "target_cells_requested": len(target_cells),
                    "target_cells_replaced": n_target_replaced,
                    "full_region_cells_replaced": n_full_replaced,
                    "source_risk": source_risk,
                    "target_cells_risk": target_risk,
                    "target_cells_risk_delta": target_risk - source_risk,
                    "target_cells_direction_success": int(sign * (target_risk - source_risk) > 0),
                    "full_region_risk": full_risk,
                    "full_region_risk_delta": full_risk - source_risk,
                    "full_region_direction_success": int(sign * (full_risk - source_risk) > 0),
                    "generated_path": str(generated_path),
                    "encoded_generated_grid_path": str(cache_path),
                }
            )
            print(f"[score] {direction} {run_id}: {source_risk:.4f} -> {full_risk:.4f}")

    summary_rows = summarize(result_rows)
    write_csv(out_dir / "grade_risk_shift_by_run.csv", result_rows)
    write_csv(out_dir / "grade_risk_shift_summary.csv", summary_rows)
    write_json(
        out_dir / "summary.json",
        {
            "checkpoint": str(args.model_ckpt),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "primary_metric": "full_region_risk_delta",
            "interpretation": "Positive is desired for low_to_high; negative is desired for high_to_low.",
            "summaries": summary_rows,
            "n_runs": len(result_rows),
        },
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
