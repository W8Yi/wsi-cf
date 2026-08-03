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
from wsi_cf.eval.grade_risk import map_region_cells_to_bag, replace_region_features  # noqa: E402
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint  # noqa: E402
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2  # noqa: E402


DEFAULT_EDIT_ROOT = WSI_CF_ROOT / "paper_outputs/prad_gleason_low_high_steering"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PRAD low/high generated edits by classifier probability shift.")
    parser.add_argument("--edit-root", type=Path, default=DEFAULT_EDIT_ROOT)
    parser.add_argument("--classifier-run-dir", type=Path, default=WSI_CF_ROOT / "artifacts/classifier_training/prad_low_vs_high_grade")
    parser.add_argument(
        "--directions",
        default="auto",
        help="Comma-separated direction dirs, or 'auto' to discover edit_root/_regions/*.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-runs-per-direction", type=int, default=10)
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--skip-source-grid-check", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
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


def load_label_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = read_json(path)
    label_to_id = {str(k): int(v) for k, v in payload["label_to_id"].items()}
    id_to_label = {int(v): str(k) for k, v in label_to_id.items()}
    return label_to_id, id_to_label


@torch.no_grad()
def score_label_probability(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    label_id: int,
    id_to_label: dict[int, str],
    device: torch.device,
) -> tuple[str, float]:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    _, y_prob, y_hat, _, _ = model(x)
    pred_id = int(y_hat.detach().cpu().reshape(-1)[0].item())
    probs = y_prob.detach().cpu().numpy().reshape(-1)
    return id_to_label.get(pred_id, str(pred_id)), float(probs[int(label_id)])


def load_direction_requests(edit_root: Path, direction: str, limit: int) -> list[tuple[dict[str, str], dict[str, Any]]]:
    region_dir = edit_root / "_regions" / direction
    manifest_path = region_dir / "progressive_edit_manifest_showcase_best.json"
    if not manifest_path.exists():
        manifest_path = region_dir / "progressive_edit_manifest.json"
    with (region_dir / "region_bank.csv").open("r", newline="") as handle:
        regions = {str(row["region_id"]): row for row in csv.DictReader(handle)}
    out: list[tuple[dict[str, str], dict[str, Any]]] = []
    for request in list(read_json(manifest_path))[: int(limit)]:
        region = regions.get(str(request["region_id"]))
        if region is None:
            raise KeyError(f"Region {request['region_id']} from {manifest_path} is absent from region_bank.csv")
        out.append((region, request))
    return out


def resolve_directions(edit_root: Path, value: str) -> list[str]:
    if str(value).strip().lower() != "auto":
        return [token.strip() for token in str(value).split(",") if token.strip()]
    region_root = edit_root / "_regions"
    if not region_root.exists():
        raise FileNotFoundError(f"Cannot auto-discover directions; missing {region_root}")
    directions: list[str] = []
    for path in sorted(region_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "region_bank.csv").exists() and (
            (path / "progressive_edit_manifest_showcase_best.json").exists()
            or (path / "progressive_edit_manifest.json").exists()
        ):
            directions.append(path.name)
    if not directions:
        raise ValueError(f"No direction manifests found under {region_root}")
    return directions


def parse_grade_direction(direction: str) -> tuple[str, str]:
    tokens = str(direction).lower().split("_to_", maxsplit=1)
    if len(tokens) == 2 and tokens[0].startswith("gg") and tokens[1].startswith("gg"):
        return tokens[0].upper(), tokens[1].upper()
    if str(direction).lower().startswith("gg"):
        return str(direction).split("_", maxsplit=1)[0].upper(), ""
    return "", ""


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["direction"])].append(row)
    summary: list[dict[str, Any]] = []
    for direction, values in sorted(grouped.items()):
        deltas = np.asarray([float(row["target_probability_delta"]) for row in values], dtype=np.float64)
        full_deltas = np.asarray([float(row["full_region_target_probability_delta"]) for row in values], dtype=np.float64)
        summary.append(
            {
                "direction": direction,
                "n": int(len(values)),
                "mean_target_probability_delta": float(deltas.mean()),
                "median_target_probability_delta": float(np.median(deltas)),
                "positive_target_delta_rate": float(np.mean(deltas > 0.0)),
                "mean_full_region_target_probability_delta": float(full_deltas.mean()),
                "median_full_region_target_probability_delta": float(np.median(full_deltas)),
                "positive_full_region_target_delta_rate": float(np.mean(full_deltas > 0.0)),
                "target_prediction_rate_after_edit": float(np.mean([int(row["edited_pred_is_target"]) for row in values])),
                "flip_to_target_rate": float(np.mean([int(row["flip_to_target"]) for row in values])),
            }
        )
    return summary


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    out_dir = args.out_dir or args.edit_root / "classifier_shift_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_to_id, id_to_label = load_label_mapping(args.classifier_run_dir / "label_mapping.json")
    model = build_mil_from_checkpoint(args.classifier_run_dir / "best_model.pt", device=device)
    uni_model = None
    uni_transform = None
    result_rows: list[dict[str, Any]] = []

    directions = resolve_directions(args.edit_root, str(args.directions))
    for direction in directions:
        for region, request in load_direction_requests(args.edit_root, direction, int(args.max_runs_per_direction)):
            run_id = str(request["run_id"])
            generated_path = args.edit_root / direction / run_id / "generated.png"
            if not generated_path.exists():
                raise FileNotFoundError(f"Generated image not found: {generated_path}")
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

            original_features, coords = load_feature_bag(resolve_repo_path(region["canonical_h5_path"]))
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
            target_label = str(request.get("target_label", ""))
            source_grade_group, target_grade_group = parse_grade_direction(direction)
            target_id = int(label_to_id[target_label])
            target_cells = {(int(cell["gx"]), int(cell["gy"])) for cell in request.get("target_cells", [])}
            target_features, n_target = replace_region_features(original_features, generated_grid, cell_to_index, cells=target_cells)
            full_features, n_full = replace_region_features(original_features, generated_grid, cell_to_index)
            source_pred, source_prob = score_label_probability(model, original_features, label_id=target_id, id_to_label=id_to_label, device=device)
            edited_pred, edited_prob = score_label_probability(model, target_features, label_id=target_id, id_to_label=id_to_label, device=device)
            full_pred, full_prob = score_label_probability(model, full_features, label_id=target_id, id_to_label=id_to_label, device=device)
            result_rows.append(
                {
                    "direction": direction,
                    "source_grade_group": source_grade_group,
                    "target_grade_group": target_grade_group,
                    "run_id": run_id,
                    "region_id": region["region_id"],
                    "slide_key": region["slide_key"],
                    "source_label": request.get("source_label", ""),
                    "target_label": target_label,
                    "target_cells_requested": int(len(target_cells)),
                    "target_cells_replaced": int(n_target),
                    "full_region_cells_replaced": int(n_full),
                    "source_pred_label": source_pred,
                    "source_target_probability": source_prob,
                    "edited_pred_label": edited_pred,
                    "edited_pred_is_target": int(edited_pred == target_label),
                    "edited_target_probability": edited_prob,
                    "target_probability_delta": edited_prob - source_prob,
                    "full_region_pred_label": full_pred,
                    "full_region_target_probability": full_prob,
                    "full_region_target_probability_delta": full_prob - source_prob,
                    "flip_to_target": int(source_pred != target_label and edited_pred == target_label),
                    "generated_path": str(generated_path),
                    "encoded_grid_path": str(cache_path),
                }
            )
            print(f"[score] {direction} {run_id}: P({target_label}) {source_prob:.4f} -> {edited_prob:.4f}")

    summary_rows = summarize(result_rows)
    write_csv(out_dir / "prediction_shift_by_run.csv", result_rows)
    write_csv(out_dir / "prediction_shift_summary.csv", summary_rows)
    write_json(
        out_dir / "summary.json",
        {
            "edit_root": str(args.edit_root),
            "classifier_run_dir": str(args.classifier_run_dir),
            "directions": directions,
            "n_runs": int(len(result_rows)),
            "summary": summary_rows,
        },
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
