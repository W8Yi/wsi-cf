#!/usr/bin/env python
"""Compute tutorial metrics for the HNSCC HPV smoothed-28 showcase.

The script compares a progressive counterfactual run against a naive baseline
run on the same selected cells. It writes only data artifacts: classifier
probabilities, progressive stage trajectory, RGB perturbation summaries, and a
verification JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_RUN_ID = "case_clean_sae_expand_attn_seed_p90__attention_only_top23_smooth4_prune28"
DEFAULT_OUT_ROOT = REPO_ROOT / "artifacts/hnscc_hpv_showcase_smoothed28_tutorial"
DEFAULT_PROGRESSIVE_RUN_DIR = DEFAULT_OUT_ROOT / "progressive" / DEFAULT_RUN_ID
DEFAULT_NAIVE_RUN_DIR = DEFAULT_OUT_ROOT / "naive_previous_settings" / DEFAULT_RUN_ID
DEFAULT_METRICS_DIR = DEFAULT_OUT_ROOT / "metrics"
DEFAULT_MIL_CKPT = REPO_ROOT / "resources/models/classifiers/hnscc_hpv/mil_split0.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progressive-run-dir", type=Path, default=DEFAULT_PROGRESSIVE_RUN_DIR)
    parser.add_argument("--naive-run-dir", type=Path, default=DEFAULT_NAIVE_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--mil-ckpt", type=Path, default=DEFAULT_MIL_CKPT)
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:N', or 'cpu'.")
    parser.add_argument(
        "--stage-mode",
        choices=["none", "existing", "reencode"],
        default="existing",
        help=(
            "none skips the progressive stage trajectory; existing uses stage grids already in --out-dir; "
            "reencode rebuilds stage canvases from steered windows and encodes them with UNI2."
        ),
    )
    parser.add_argument("--grid-step-px", type=int, default=256)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def mean_abs_rgb_diff(source: np.ndarray, edited: np.ndarray) -> np.ndarray:
    if source.shape != edited.shape:
        raise ValueError(f"Image shape mismatch: {source.shape} vs {edited.shape}")
    return np.abs(edited.astype(np.float32) - source.astype(np.float32)).mean(axis=2)


def tukey_box_stats(values: np.ndarray, method: str) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    q1, med, q3 = np.percentile(flat, [25, 50, 75])
    iqr = float(q3 - q1)
    lower = float(q1 - 1.5 * iqr)
    upper = float(q3 + 1.5 * iqr)
    inlier = flat[(flat >= lower) & (flat <= upper)]
    return {
        "method": method,
        "mean": float(flat.mean()),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "whisker_low": float(inlier.min()) if inlier.size else float(flat.min()),
        "whisker_high": float(inlier.max()) if inlier.size else float(flat.max()),
    }


def load_mil(device_name: str, ckpt: Path):
    import torch

    from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    return build_mil_from_checkpoint(ckpt, device=device), device


def eval_grid(mil_model: Any, device: Any, grid_path: Path) -> dict[str, Any]:
    from wsi_cf.eval.hnsc_hpv import run_mil_attention

    grid = np.asarray(np.load(grid_path), dtype=np.float32)
    bag = grid.reshape(-1, grid.shape[-1]).astype(np.float32, copy=False)
    _, pred, prob_pos = run_mil_attention(mil_model, bag, device=device)
    return {
        "pred_label": "HPV+" if int(pred) == 1 else "HPV-",
        "prob_hpv_pos": float(prob_pos),
        "prob_hpv_neg": float(1.0 - prob_pos),
    }


def find_generated_grid(run_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in run_dir.glob("*generated_reencoded_uni2_grid.npy")
        if path.name != "source_reencoded_uni2_grid.npy"
    )
    if not candidates:
        raise FileNotFoundError(f"No generated re-encoded UNI2 grid found in {run_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple generated re-encoded UNI2 grids found in {run_dir}: {candidates}")
    return candidates[0]


def load_run_paths(progressive_run_dir: Path, naive_run_dir: Path) -> dict[str, Path]:
    paths = {
        "source_image": progressive_run_dir / "source_region_actual.png",
        "progressive_image": progressive_run_dir / "generated.png",
        "naive_image": naive_run_dir / "generated.png",
        "source_grid": progressive_run_dir / "source_reencoded_uni2_grid.npy",
        "progressive_grid": find_generated_grid(progressive_run_dir),
        "naive_grid": find_generated_grid(naive_run_dir),
        "progressive_manifest": progressive_run_dir / "run_manifest.json",
        "naive_manifest": naive_run_dir / "run_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required run artifacts:\n" + "\n".join(missing))
    return paths


def encode_image_to_grid(
    image_path: Path,
    grid_path: Path,
    *,
    grid_step_px: int,
    device: Any,
    uni_model: Any,
    uni_transform: Any,
) -> None:
    import torch

    from wsi_cf.generation.pixcell import build_uni_grid_from_image

    image = Image.open(image_path).convert("RGB")
    z_grid = build_uni_grid_from_image(
        image,
        uni_model=uni_model,
        uni_transform=uni_transform,
        grid_step_px=int(grid_step_px),
        device=device,
        out_dtype=torch.float32,
    )
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(grid_path, z_grid.detach().cpu().numpy().astype(np.float32))


def rebuild_stage_canvases(progressive_run_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    run_manifest = json.loads((progressive_run_dir / "run_manifest.json").read_text())
    stage_dir = out_dir / "progressive_stage_canvases"
    stage_dir.mkdir(parents=True, exist_ok=True)

    canvas = Image.open(progressive_run_dir / "source_region_actual.png").convert("RGB")
    stage0_image = stage_dir / "stage_00_original.png"
    canvas.save(stage0_image)
    rows = [
        {
            "stage": 0,
            "num_windows_committed": 0,
            "window_id": "",
            "image_path": str(stage0_image),
            "uni2_grid_path": str(stage_dir / "stage_00_reencoded_uni2_grid.npy"),
        }
    ]

    for idx, step in enumerate(run_manifest.get("window_history", []), start=1):
        steered_path = REPO_ROOT / str(step["steered_window_path"])
        if not steered_path.exists():
            steered_path = progressive_run_dir / "steps" / f"step_{idx:02d}" / "steered_window.png"
        steered = Image.open(steered_path).convert("RGB")
        bounds = step["commit_bounds_global"]
        x0 = int(bounds["x0"])
        y0 = int(bounds["y0"])
        x1 = int(bounds["x1"])
        y1 = int(bounds["y1"])
        canvas.paste(steered.crop((0, 0, x1 - x0, y1 - y0)), (x0, y0))
        image_path = stage_dir / f"stage_{idx:02d}_after_{step['window_id']}.png"
        grid_path = stage_dir / f"stage_{idx:02d}_reencoded_uni2_grid.npy"
        canvas.save(image_path)
        rows.append(
            {
                "stage": idx,
                "num_windows_committed": idx,
                "window_id": str(step["window_id"]),
                "image_path": str(image_path),
                "uni2_grid_path": str(grid_path),
            }
        )
    return rows


def load_existing_stage_rows(out_dir: Path) -> list[dict[str, Any]]:
    trajectory_json = out_dir / "progressive_stage_prediction_trajectory.json"
    if not trajectory_json.exists():
        raise FileNotFoundError(
            f"{trajectory_json} does not exist. Use --stage-mode reencode to build the stage trajectory."
        )
    rows = json.loads(trajectory_json.read_text()).get("rows", [])
    return [
        {
            "stage": int(row["stage"]),
            "num_windows_committed": int(row["num_windows_committed"]),
            "window_id": str(row["window_id"]),
            "image_path": str(row["image_path"]),
            "uni2_grid_path": str(row["uni2_grid_path"]),
        }
        for row in rows
    ]


def compute_stage_trajectory(
    *,
    progressive_run_dir: Path,
    out_dir: Path,
    stage_mode: str,
    grid_step_px: int,
    mil_model: Any,
    device: Any,
) -> list[dict[str, Any]]:
    if stage_mode == "none":
        return []
    if stage_mode == "existing":
        stage_rows = load_existing_stage_rows(out_dir)
    else:
        stage_rows = rebuild_stage_canvases(progressive_run_dir, out_dir)

    rows_to_encode = [
        row
        for row in stage_rows
        if stage_mode == "reencode" or not Path(row["uni2_grid_path"]).exists()
    ]
    if rows_to_encode:
        import torch

        from wsi_cf.generation.pixcell import load_uni2

        uni_model, uni_transform = load_uni2(device)
        for row in rows_to_encode:
            encode_image_to_grid(
                Path(row["image_path"]),
                Path(row["uni2_grid_path"]),
                grid_step_px=grid_step_px,
                device=device,
                uni_model=uni_model,
                uni_transform=uni_transform,
            )
        del uni_model
        if getattr(device, "type", "") == "cuda":
            torch.cuda.empty_cache()

    trajectory_rows: list[dict[str, Any]] = []
    for row in stage_rows:
        trajectory_rows.append({**row, **eval_grid(mil_model, device, Path(row["uni2_grid_path"]))})
    return trajectory_rows


def max_abs_array_delta(path_a: Path, path_b: Path) -> float:
    if not path_a.exists() or not path_b.exists():
        return math.nan
    a = np.asarray(np.load(path_a), dtype=np.float32)
    b = np.asarray(np.load(path_b), dtype=np.float32)
    if a.shape != b.shape:
        return math.inf
    return float(np.max(np.abs(a - b)))


def main() -> None:
    args = parse_args()
    progressive_run_dir = args.progressive_run_dir.resolve()
    naive_run_dir = args.naive_run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = load_run_paths(progressive_run_dir, naive_run_dir)
    source = image_array(paths["source_image"])
    progressive = image_array(paths["progressive_image"])
    naive = image_array(paths["naive_image"])
    progressive_diff = mean_abs_rgb_diff(source, progressive)
    naive_diff = mean_abs_rgb_diff(source, naive)

    perturbation_summary_rows = [
        {"method": "Progressive", "mean_abs_rgb_diff_0_255": float(progressive_diff.mean())},
        {"method": "Naive", "mean_abs_rgb_diff_0_255": float(naive_diff.mean())},
    ]
    perturbation_box_rows = [
        tukey_box_stats(progressive_diff, "Progressive"),
        tukey_box_stats(naive_diff, "Naive"),
    ]

    mil_model, device = load_mil(str(args.device), args.mil_ckpt.resolve())
    source_eval = eval_grid(mil_model, device, paths["source_grid"])
    progressive_eval = eval_grid(mil_model, device, paths["progressive_grid"])
    naive_eval = eval_grid(mil_model, device, paths["naive_grid"])
    classifier_rows = [
        {"sample": "source", **source_eval, "image_path": str(paths["source_image"]), "uni2_grid_path": str(paths["source_grid"])},
        {
            "sample": "progressive",
            **progressive_eval,
            "image_path": str(paths["progressive_image"]),
            "uni2_grid_path": str(paths["progressive_grid"]),
        },
        {"sample": "naive", **naive_eval, "image_path": str(paths["naive_image"]), "uni2_grid_path": str(paths["naive_grid"])},
    ]

    trajectory_rows = compute_stage_trajectory(
        progressive_run_dir=progressive_run_dir,
        out_dir=out_dir,
        stage_mode=str(args.stage_mode),
        grid_step_px=int(args.grid_step_px),
        mil_model=mil_model,
        device=device,
    )

    write_csv(
        out_dir / "classifier_predictions.csv",
        classifier_rows,
        ["sample", "pred_label", "prob_hpv_pos", "prob_hpv_neg", "image_path", "uni2_grid_path"],
    )
    write_json(
        out_dir / "classifier_predictions.json",
        {
            "prediction_method": "image_reencoded_uni2_region_mil",
            "mil_ckpt": str(args.mil_ckpt.resolve()),
            "rows": classifier_rows,
        },
    )
    write_csv(
        out_dir / "perturbation_summary.csv",
        perturbation_summary_rows,
        ["method", "mean_abs_rgb_diff_0_255"],
    )
    write_json(
        out_dir / "perturbation_summary.json",
        {
            "metric": "per-pixel mean absolute RGB difference from source image on 0-255 RGB scale",
            "rows": perturbation_summary_rows,
        },
    )
    write_csv(
        out_dir / "perturbation_boxplot_stats.csv",
        perturbation_box_rows,
        ["method", "mean", "median", "q1", "q3", "whisker_low", "whisker_high"],
    )
    write_json(
        out_dir / "perturbation_boxplot_stats.json",
        {
            "metric": "per-pixel mean absolute RGB difference from source image on 0-255 RGB scale",
            "rows": perturbation_box_rows,
        },
    )
    if trajectory_rows:
        write_csv(
            out_dir / "progressive_stage_prediction_trajectory.csv",
            trajectory_rows,
            [
                "stage",
                "num_windows_committed",
                "window_id",
                "pred_label",
                "prob_hpv_pos",
                "prob_hpv_neg",
                "image_path",
                "uni2_grid_path",
            ],
        )
        write_json(out_dir / "progressive_stage_prediction_trajectory.json", {"rows": trajectory_rows})

    verification = {
        "inputs": {key: str(value) for key, value in paths.items()},
        "metrics_dir": str(out_dir),
        "stage_mode": str(args.stage_mode),
        "checks": {
            "source_is_hpv_pos": source_eval["pred_label"] == "HPV+",
            "progressive_counterfactual_is_hpv_neg": progressive_eval["pred_label"] == "HPV-",
            "naive_counterfactual_is_hpv_neg": naive_eval["pred_label"] == "HPV-",
            "progressive_rgb_diff_lower_than_naive": perturbation_summary_rows[0]["mean_abs_rgb_diff_0_255"]
            < perturbation_summary_rows[1]["mean_abs_rgb_diff_0_255"],
        },
        "classifier": {
            "source": source_eval,
            "progressive": progressive_eval,
            "naive": naive_eval,
        },
        "perturbation_summary": perturbation_summary_rows,
        "perturbation_boxplot_stats": perturbation_box_rows,
    }
    if trajectory_rows:
        verification["checks"].update(
            {
                "trajectory_stage0_matches_source_prob": abs(trajectory_rows[0]["prob_hpv_neg"] - source_eval["prob_hpv_neg"])
                < 1e-8,
                "trajectory_final_matches_progressive_prob": abs(
                    trajectory_rows[-1]["prob_hpv_neg"] - progressive_eval["prob_hpv_neg"]
                )
                < 1e-8,
                "stage0_grid_matches_progressive_run_source_grid_max_abs_delta": max_abs_array_delta(
                    Path(trajectory_rows[0]["uni2_grid_path"]), paths["source_grid"]
                ),
                "final_stage_grid_matches_progressive_run_generated_grid_max_abs_delta": max_abs_array_delta(
                    Path(trajectory_rows[-1]["uni2_grid_path"]), paths["progressive_grid"]
                ),
            }
        )
        verification["trajectory"] = trajectory_rows

    write_json(out_dir / "verification.json", verification)
    print(json.dumps(verification["checks"], indent=2))
    print(f"Wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
