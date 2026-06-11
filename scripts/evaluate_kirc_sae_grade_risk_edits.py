#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
for search_path in (SCRIPT_DIR, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_kirc_grade_risk_edits import (  # noqa: E402
    DEFAULT_EDIT_ROOT,
    DEFAULT_LABEL_SOURCE,
    load_direction_requests,
    load_feature_bag,
    load_slide_grades,
    resolve_repo_path,
    write_csv,
)
from wsi_cf.common.io import write_json  # noqa: E402
from wsi_cf.common.runtime import resolve_device  # noqa: E402
from wsi_cf.eval.grade_risk import map_region_cells_to_bag, replace_region_features  # noqa: E402
from wsi_cf.eval.sae_grade_risk import AGGREGATE_NAMES, apply_feature_scaler, load_sae_grade_risk_model, summarize_feature_bag_with_sae  # noqa: E402
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2  # noqa: E402
from wsi_cf.steering.sae_runtime import load_sae_from_config  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score generated KIRC regions with a concept-only SAE ordinal grade-risk model.")
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        default=WSI_CF_ROOT / "artifacts/sae_grade_risk_training/kirc_batch_topk_ordinal/best_model.pt",
    )
    parser.add_argument("--edit-root", type=Path, default=DEFAULT_EDIT_ROOT)
    parser.add_argument("--label-source", type=Path, default=DEFAULT_LABEL_SOURCE)
    parser.add_argument("--directions", default="03_kirc_low_to_high,03_kirc_high_to_low")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-runs-per-direction", type=int, default=10)
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--skip-source-grid-check", action="store_true")
    parser.add_argument("--edit-origin", default="binary_classifier_selected_existing_edits")
    parser.add_argument("--device", default="cuda:0")
    return parser


def desired_sign(target_label: str) -> int:
    if str(target_label).lower() == "high":
        return 1
    if str(target_label).lower() == "low":
        return -1
    raise ValueError(f"Unsupported target label: {target_label}")


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["direction"])].append(row)
    out: list[dict[str, Any]] = []
    for direction, values in sorted(grouped.items()):
        full = np.asarray([float(row["full_region_risk_delta"]) for row in values])
        target = np.asarray([float(row["target_cells_risk_delta"]) for row in values])
        out.append(
            {
                "direction": direction,
                "n": len(values),
                "full_region_mean_delta": float(full.mean()),
                "full_region_median_delta": float(np.median(full)),
                "full_region_direction_success_rate": float(np.mean([int(row["full_region_direction_success"]) for row in values])),
                "target_cells_mean_delta": float(target.mean()),
                "target_cells_median_delta": float(np.median(target)),
                "target_cells_direction_success_rate": float(np.mean([int(row["target_cells_direction_success"]) for row in values])),
            }
        )
    return out


@torch.no_grad()
def score_bag(
    features: np.ndarray,
    *,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    sae_model: torch.nn.Module,
    d_latent: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    selected_latents = np.asarray(checkpoint["selected_latents"], dtype=np.int64)
    summary = summarize_feature_bag_with_sae(
        features,
        sae_model=sae_model,
        device=device,
        batch_size=int(checkpoint["args"]["sae_batch_size"]),
        d_latent=int(d_latent),
        active_threshold=float(checkpoint["active_threshold"]),
        top_fraction=float(checkpoint["top_fraction"]),
        selected_latents=selected_latents,
    )
    vector = np.concatenate([summary[name] for name in AGGREGATE_NAMES], axis=0).reshape(1, -1)
    scaled = apply_feature_scaler(vector, np.asarray(checkpoint["scaler_mean"]), np.asarray(checkpoint["scaler_scale"]))
    risk, probabilities, _ = model(torch.as_tensor(scaled, dtype=torch.float32, device=device))
    return float(risk.detach().cpu().item()), probabilities.detach().cpu().numpy().reshape(-1).astype(float).tolist()


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(str(args.device))
    out_dir = args.out_dir or args.edit_root / "sae_grade_risk_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.model_ckpt.exists():
        raise FileNotFoundError(f"SAE grade-risk checkpoint does not exist: {args.model_ckpt}")
    model, checkpoint = load_sae_grade_risk_model(args.model_ckpt, device=device)
    if int(checkpoint["args"].get("max_tiles_per_slide", 0)) != 0:
        raise ValueError("Edit scoring requires a concept-risk checkpoint trained on full slide bags (--max-tiles-per-slide 0).")
    sae_model, _, d_latent = load_sae_from_config(
        checkpoint["sae"]["checkpoint"], checkpoint["sae"]["config"], device=str(device)
    )
    slide_grades = load_slide_grades(args.label_source)
    uni_model = None
    uni_transform = None
    result_rows: list[dict[str, Any]] = []

    for direction in [value.strip() for value in str(args.directions).split(",") if value.strip()]:
        for region, request in load_direction_requests(args.edit_root, direction, int(args.max_runs_per_direction)):
            run_id = str(request["run_id"])
            generated_path = args.edit_root / direction / run_id / "generated.png"
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
            cell_to_index = map_region_cells_to_bag(
                coords,
                region_gx0=int(region["region_gx0"]),
                region_gy0=int(region["region_gy0"]),
                grid_shape=(int(generated_grid.shape[0]), int(generated_grid.shape[1])),
            )
            if not args.skip_source_grid_check:
                source_grid = np.load(resolve_repo_path(region["feature_grid_path"]))
                for (gx, gy), feature_index in cell_to_index.items():
                    if not np.allclose(source_grid[gy, gx], original_features[feature_index], atol=1e-5, rtol=1e-5):
                        raise ValueError(f"Source grid does not align with whole-slide bag for {run_id}, cell {(gx, gy)}")
            target_cells = {(int(cell["gx"]), int(cell["gy"])) for cell in request.get("target_cells", [])}
            target_features, n_target = replace_region_features(original_features, generated_grid, cell_to_index, cells=target_cells)
            full_features, n_full = replace_region_features(original_features, generated_grid, cell_to_index)
            source_risk, source_probs = score_bag(
                original_features, model=model, checkpoint=checkpoint, sae_model=sae_model, d_latent=d_latent, device=device
            )
            target_risk, target_probs = score_bag(
                target_features, model=model, checkpoint=checkpoint, sae_model=sae_model, d_latent=d_latent, device=device
            )
            full_risk, full_probs = score_bag(
                full_features, model=model, checkpoint=checkpoint, sae_model=sae_model, d_latent=d_latent, device=device
            )
            sign = desired_sign(str(request.get("target_label", "")))
            row = {
                "edit_origin": str(args.edit_origin),
                "direction": direction,
                "run_id": run_id,
                "slide_key": region["slide_key"],
                "raw_grade": slide_grades.get(str(region["slide_key"]), ""),
                "source_label": request.get("source_label", ""),
                "target_label": request.get("target_label", ""),
                "mapped_region_cells": len(cell_to_index),
                "target_cells_replaced": n_target,
                "full_region_cells_replaced": n_full,
                "source_risk": source_risk,
                "target_cells_risk": target_risk,
                "target_cells_risk_delta": target_risk - source_risk,
                "target_cells_direction_success": int(sign * (target_risk - source_risk) > 0),
                "full_region_risk": full_risk,
                "full_region_risk_delta": full_risk - source_risk,
                "full_region_direction_success": int(sign * (full_risk - source_risk) > 0),
            }
            for prefix, probs in (("source", source_probs), ("target_cells", target_probs), ("full_region", full_probs)):
                for threshold, probability in zip(("g2", "g3", "g4"), probs):
                    row[f"{prefix}_prob_ge_{threshold}"] = probability
            result_rows.append(row)
            print(f"[score] {direction} {run_id}: {source_risk:.4f} -> {full_risk:.4f}", flush=True)

    summary_rows = summarize_results(result_rows)
    write_csv(out_dir / "sae_grade_risk_shift_by_run.csv", result_rows)
    write_csv(out_dir / "sae_grade_risk_shift_summary.csv", summary_rows)
    write_json(
        out_dir / "summary.json",
        {
            "checkpoint": str(args.model_ckpt),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "edit_origin": str(args.edit_origin),
            "primary_metric": "full_region_risk_delta",
            "interpretation": "Positive is desired for low_to_high; negative is desired for high_to_low.",
            "summaries": summary_rows,
            "n_runs": len(result_rows),
        },
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
