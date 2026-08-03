#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
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
from wsi_cf.eval.grade_risk import map_region_cells_to_bag, replace_region_features  # noqa: E402
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, run_mil_attention  # noqa: E402
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2  # noqa: E402
from wsi_cf.steering.edit_policy import DEFAULT_EDIT_POLICY  # noqa: E402
from wsi_cf.steering.progressive import split_cells_by_edit_support  # noqa: E402


DEFAULT_SOURCE_ROOT = WSI_CF_ROOT / "artifacts/morphology_label_concept_review_top1_showcase_best_10slides"
DEFAULT_OUT_DIR = WSI_CF_ROOT / "artifacts/luad_to_lusc_cell_budget_showcase_best_10slides"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate LUAD-to-LUSC edits at multiple cell budgets and score whole-slide P(LUSC) change."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--direction", default="01_luad_to_lusc")
    parser.add_argument("--classifier-run-dir", type=Path, default=WSI_CF_ROOT / "artifacts/classifier_training/luad_lusc")
    parser.add_argument(
        "--concept-dir",
        type=Path,
        default=WSI_CF_ROOT / "artifacts/morphology_label_concept_review/selected/01_luad_lusc__LUSC",
    )
    parser.add_argument("--edit-policy", type=Path, default=WSI_CF_ROOT / DEFAULT_EDIT_POLICY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cell-counts", default="1,4,16,32,64")
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--concept-target-top-k", type=int, default=5)
    parser.add_argument("--sae-variant", default="relu_sae_base")
    parser.add_argument("--output-mode", default="minimal", choices=["minimal", "debug"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-edits", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-reencode", action="store_true")
    return parser


def parse_cell_counts(value: str) -> list[int]:
    counts = [int(token.strip()) for token in str(value).split(",") if token.strip()]
    if not counts or any(count <= 0 or count > 64 for count in counts):
        raise ValueError("--cell-counts must contain integers in [1, 64]")
    return list(dict.fromkeys(counts))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WSI_CF_ROOT / path


def load_feature_bag(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        features = np.asarray(handle["features"], dtype=np.float32)
        coords = np.asarray(handle["coords"], dtype=np.int64)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    return features, coords


def rank_cells_by_attention(
    attention: np.ndarray,
    cell_to_index: dict[tuple[int, int], int],
    eligible_cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    return sorted(
        [(int(gx), int(gy)) for gx, gy in eligible_cells if (int(gx), int(gy)) in cell_to_index],
        key=lambda cell: (-float(attention[cell_to_index[cell]]), int(cell[1]), int(cell[0])),
    )


@torch.no_grad()
def score_target_probability(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    target_label_id: int,
    device: torch.device,
) -> tuple[int, float]:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    _, y_prob, y_hat, _, _ = model(x)
    return int(y_hat.detach().cpu().reshape(-1)[0].item()), float(y_prob.detach().cpu().numpy().reshape(-1)[target_label_id])


def load_source_regions(source_root: Path, direction: str, max_runs: int) -> tuple[Path, list[tuple[dict[str, str], dict[str, Any]]]]:
    region_dir = source_root / "_regions" / direction
    manifest_path = region_dir / "progressive_edit_manifest_showcase_best.json"
    if not manifest_path.exists():
        manifest_path = region_dir / "progressive_edit_manifest.json"
    with (region_dir / "region_bank.csv").open("r", newline="") as handle:
        regions = {str(row["region_id"]): row for row in csv.DictReader(handle)}
    selected: list[tuple[dict[str, str], dict[str, Any]]] = []
    for request in read_json(manifest_path)[: int(max_runs)]:
        selected.append((regions[str(request["region_id"])], request))
    return region_dir, selected


def build_generation_command(
    args: argparse.Namespace,
    *,
    region_bank: Path,
    manifest: Path,
    generated_dir: Path,
    edit_support: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_progressive_region_edit.py"),
        "--task",
        "luad_lusc",
        "--region-bank-csv",
        str(region_bank),
        "--edit-manifest",
        str(manifest),
        "--out-dir",
        str(generated_dir),
        "--edit-policy",
        str(args.edit_policy),
        "--concepts-json",
        str(args.concept_dir / "selected_concepts.json"),
        "--representative-tiles-csv",
        str(args.concept_dir / "representative_tiles.csv"),
        "--concept-class-label",
        "LUSC",
        "--concept-ranking-method",
        "attention_weighted",
        "--concept-target-stat",
        "median",
        "--concept-target-top-k",
        str(args.concept_target_top_k),
        "--concept-steering-mode",
        "prototype_vector",
        "--max-concepts",
        "1",
        "--max-runs",
        str(args.max_runs),
        "--target-magnification",
        "20",
        "--sae-variant",
        str(args.sae_variant),
        "--edit-support",
        str(edit_support),
        "--output-mode",
        str(args.output_mode),
        "--device",
        str(args.device),
    ]
    if bool(args.skip_existing):
        cmd.append("--skip-existing")
    return cmd


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["cell_count"])].append(row)
    summary: list[dict[str, Any]] = []
    for count in sorted(grouped):
        values = grouped[count]
        deltas = np.asarray([float(row["lusc_probability_delta"]) for row in values], dtype=np.float64)
        summary.append(
            {
                "cell_count": count,
                "region_fraction_percent": float(100.0 * count / 64.0),
                "n": len(values),
                "mean_cells_replaced": float(np.mean([int(row.get("cells_replaced", count)) for row in values])),
                "mean_source_lusc_probability": float(np.mean([float(row["source_lusc_probability"]) for row in values])),
                "mean_edited_lusc_probability": float(np.mean([float(row["edited_lusc_probability"]) for row in values])),
                "mean_lusc_probability_delta": float(deltas.mean()),
                "median_lusc_probability_delta": float(np.median(deltas)),
                "positive_delta_rate": float(np.mean(deltas > 0.0)),
                "prediction_lusc_rate_after_edit": float(np.mean([int(row["edited_pred_is_lusc"]) for row in values])),
                "flip_luad_to_lusc_rate": float(np.mean([int(row["flip_luad_to_lusc"]) for row in values])),
            }
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    counts = parse_cell_counts(args.cell_counts)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    region_dir, selected_regions = load_source_regions(args.source_root, str(args.direction), int(args.max_runs))
    if not selected_regions:
        raise ValueError(f"No source regions found for {args.direction}")
    label_mapping = read_json(args.classifier_run_dir / "label_mapping.json")
    label_to_id = {str(key): int(value) for key, value in label_mapping["label_to_id"].items()}
    target_label_id = int(label_to_id["LUSC"])
    model = build_mil_from_checkpoint(args.classifier_run_dir / "best_model.pt", device=device)

    requests_by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    selected_cell_rows: list[dict[str, Any]] = []
    source_cache: dict[str, tuple[dict[tuple[int, int], int], int, float]] = {}
    for region, source_request in selected_regions:
        source_features, coords = load_feature_bag(Path(region["canonical_h5_path"]))
        source_grid = np.load(resolve_repo_path(region["feature_grid_path"]))
        cell_to_index = map_region_cells_to_bag(
            coords,
            region_gx0=int(region["region_gx0"]),
            region_gy0=int(region["region_gy0"]),
            grid_shape=(int(source_grid.shape[0]), int(source_grid.shape[1])),
        )
        eligible, _ = split_cells_by_edit_support(
            target_cells=list(cell_to_index),
            grid_w=int(source_grid.shape[1]),
            grid_h=int(source_grid.shape[0]),
            window_grid_side=4,
            stride_cells=2,
            grid_step_px=int(region["grid_step_px"]),
            edit_support="center_2x2",
        )
        attention, _, _ = run_mil_attention(model, source_features, device=device)
        ranked = rank_cells_by_attention(attention, cell_to_index, eligible)
        interior_counts = [count for count in counts if count < int(source_grid.shape[0] * source_grid.shape[1])]
        if interior_counts and max(interior_counts) > len(ranked):
            raise ValueError(f"{region['slide_key']} has only {len(ranked)} eligible feature cells; need {max(interior_counts)}")
        source_pred, source_probability = score_target_probability(
            model, source_features, target_label_id=target_label_id, device=device
        )
        source_cache[str(region["region_id"])] = (cell_to_index, source_pred, source_probability)
        for rank, cell in enumerate(ranked, start=1):
            selected_cell_rows.append(
                {
                    "region_id": region["region_id"],
                    "slide_key": region["slide_key"],
                    "cell_rank": rank,
                    "gx": cell[0],
                    "gy": cell[1],
                    "attention": float(attention[cell_to_index[cell]]),
                    "used_by_attention_ranked_budgets": ",".join(
                        str(count) for count in counts if count < int(source_grid.shape[0] * source_grid.shape[1]) and rank <= count
                    ),
                }
            )
        for count in counts:
            if count == int(source_grid.shape[0] * source_grid.shape[1]):
                chosen_cells = [
                    (gx, gy)
                    for gy in range(int(source_grid.shape[0]))
                    for gx in range(int(source_grid.shape[1]))
                ]
                edit_support = "border_relaxed"
                selector = "whole_8x8_region__border_relaxed"
            else:
                chosen_cells = ranked[:count]
                edit_support = "center_2x2"
                selector = "luad_lusc_classifier_attention_topk__center_2x2_policy"
            request = dict(source_request)
            request["run_id"] = f"{source_request['run_id']}__cells_{count:02d}"
            request["target_cells"] = [{"gx": gx, "gy": gy} for gx, gy in chosen_cells]
            request["cell_count"] = count
            request["region_fraction_percent"] = float(100.0 * count / 64.0)
            request["edit_support"] = edit_support
            request["selector"] = selector
            requests_by_count[count].append(request)

    write_csv(args.out_dir / "selected_cells_by_attention.csv", selected_cell_rows)
    commands: list[str] = []
    generated_roots: dict[int, Path] = {}
    for count in counts:
        manifest_path = args.out_dir / "manifests" / f"cells_{count:02d}.json"
        write_json(manifest_path, requests_by_count[count])
        generated_dir = args.out_dir / "generated" / f"cells_{count:02d}"
        generated_roots[count] = generated_dir
        edit_support = str(requests_by_count[count][0]["edit_support"])
        cmd = build_generation_command(
            args,
            region_bank=region_dir / "region_bank.csv",
            manifest=manifest_path,
            generated_dir=generated_dir,
            edit_support=edit_support,
        )
        commands.append(" ".join(shlex.quote(part) for part in cmd))
        if bool(args.run_edits):
            expected_images = [generated_dir / str(request["run_id"]) / "generated.png" for request in requests_by_count[count]]
            if bool(args.skip_existing) and all(path.exists() for path in expected_images):
                print(f"[reuse] {count} cells: all {len(expected_images)} generated images already exist", flush=True)
            else:
                print(f"[generate] {count} cells ({100.0 * count / 64.0:.4g}% of 8x8 region)", flush=True)
                subprocess.run(cmd, cwd=str(WSI_CF_ROOT), check=True)
    commands_path = args.out_dir / "run_generation_commands.sh"
    commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n")

    uni_model = None
    uni_transform = None
    results: list[dict[str, Any]] = []
    missing_generated: list[str] = []
    for count in counts:
        requests_by_region = {str(request["region_id"]): request for request in requests_by_count[count]}
        for region, source_request in selected_regions:
            run_id = f"{source_request['run_id']}__cells_{count:02d}"
            generated_path = generated_roots[count] / run_id / "generated.png"
            if not generated_path.exists():
                missing_generated.append(str(generated_path))
                continue
            cache_path = args.out_dir / "encoded_generated_grids" / f"cells_{count:02d}" / f"{run_id}.npy"
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
            cell_to_index, source_pred, source_probability = source_cache[str(region["region_id"])]
            source_features, _ = load_feature_bag(Path(region["canonical_h5_path"]))
            target_cells = {
                (int(cell["gx"]), int(cell["gy"]))
                for cell in requests_by_region[str(region["region_id"])]["target_cells"]
            }
            edited_features, replaced = replace_region_features(source_features, generated_grid, cell_to_index, cells=target_cells)
            edited_pred, edited_probability = score_target_probability(
                model, edited_features, target_label_id=target_label_id, device=device
            )
            results.append(
                {
                    "cell_count": count,
                    "region_fraction_percent": float(100.0 * count / 64.0),
                    "edit_support": requests_by_region[str(region["region_id"])]["edit_support"],
                    "run_id": run_id,
                    "region_id": region["region_id"],
                    "slide_key": region["slide_key"],
                    "cells_replaced": replaced,
                    "source_pred_label": "LUSC" if source_pred == target_label_id else "LUAD",
                    "source_lusc_probability": source_probability,
                    "edited_pred_label": "LUSC" if edited_pred == target_label_id else "LUAD",
                    "edited_pred_is_lusc": int(edited_pred == target_label_id),
                    "edited_lusc_probability": edited_probability,
                    "lusc_probability_delta": edited_probability - source_probability,
                    "flip_luad_to_lusc": int(source_pred != target_label_id and edited_pred == target_label_id),
                    "generated_path": str(generated_path),
                    "encoded_grid_path": str(cache_path),
                }
            )
    summary = summarize_results(results)
    write_csv(args.out_dir / "prediction_shift_by_run.csv", results)
    write_csv(args.out_dir / "prediction_shift_summary.csv", summary)
    write_json(
        args.out_dir / "summary.json",
        {
            "source_root": str(args.source_root),
            "direction": str(args.direction),
            "classifier_run_dir": str(args.classifier_run_dir),
            "concept_dir": str(args.concept_dir),
            "edit_policy": str(args.edit_policy),
            "cell_budgets": [{"cell_count": count, "region_fraction_percent": float(100.0 * count / 64.0)} for count in counts],
            "cell_selector": (
                "For 1/4/16/32 cells: highest LUAD/LUSC MIL attention among cells compatible with showcase_best "
                "center_2x2 edit support. For 64 cells: the entire 8x8 region with border_relaxed edit support."
            ),
            "prediction_endpoint": (
                "Whole-slide P(LUSC) after replacing generated target-cell UNI2 embeddings that are present in "
                "the source feature bag; cells_replaced records feature-backed coverage."
            ),
            "n_regions": len(selected_regions),
            "n_generated_results_scored": len(results),
            "n_missing_generated_images": len(missing_generated),
            "missing_generated_images": missing_generated,
            "summary": summary,
            "generation_commands": str(commands_path),
        },
    )
    if bool(args.run_edits) and missing_generated:
        raise FileNotFoundError(f"{len(missing_generated)} generated images are missing after generation; first={missing_generated[0]}")
    print(
        json.dumps(
            {"n_regions": len(selected_regions), "n_results_scored": len(results), "n_missing_generated_images": len(missing_generated), "summary": summary},
            indent=2,
        )
    )
    if not args.run_edits and not results:
        print(f"[prepared] generated manifests and commands; run bash {commands_path} or pass --run-edits to generate and score.")


if __name__ == "__main__":
    main()
