#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
for search_path in (SCRIPT_DIR, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from train_attention_classifier import DEFAULT_FEATURES_ROOT, DEFAULT_LABEL_SOURCE, DEFAULT_SPLIT_MANIFEST, prepare_bag, write_csv  # noqa: E402
from train_grade_risk_regressor import assign_validation_split, load_grade_rows, parse_target_map  # noqa: E402
from wsi_cf.common.io import write_json  # noqa: E402
from wsi_cf.common.paths import resolve_sae_paths  # noqa: E402
from wsi_cf.common.runtime import resolve_device, set_seed  # noqa: E402
from wsi_cf.eval.grade_risk import grade_risk_metrics  # noqa: E402
from wsi_cf.eval.sae_grade_risk import (  # noqa: E402
    AGGREGATE_NAMES,
    aggregate_summaries_by_case,
    apply_feature_scaler,
    build_feature_matrix,
    fit_feature_scaler,
    select_top_latents,
    select_stable_top_latents,
    summarize_feature_bag_with_sae,
)
from wsi_cf.models.concept_risk import LinearProportionalOddsRisk  # noqa: E402
from wsi_cf.steering.sae_runtime import load_sae_from_config  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a concept-only ordinal KIRC grade-risk predictor from SAE slide summaries.")
    parser.add_argument("--task-name", default="kirc_batch_topk_ordinal")
    parser.add_argument("--label-source", type=Path, default=DEFAULT_LABEL_SOURCE)
    parser.add_argument("--projects", default="TCGA-KIRC")
    parser.add_argument("--label-column", default="tumor_grade")
    parser.add_argument("--target-map", default="G1:0.0,G2:0.33,G3:0.66,G4:1.0")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/sae_grade_risk_training")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--validate-h5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embed-dim", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--sae-variant", default="tcga_sae_batch_topk_20x_interp")
    parser.add_argument("--sae-ckpt", type=Path, default=None)
    parser.add_argument("--sae-cfg", type=Path, default=None)
    parser.add_argument("--sae-batch-size", type=int, default=4096)
    parser.add_argument("--max-tiles-per-slide", type=int, default=0)
    parser.add_argument("--active-threshold", type=float, default=1e-6)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--top-concepts", type=int, default=100)
    parser.add_argument(
        "--case-level",
        action="store_true",
        help="Average slide SAE summaries within each patient before concept selection, fitting, and evaluation.",
    )
    parser.add_argument(
        "--selection-folds",
        type=int,
        default=1,
        help="Number of stratified training-only folds for stable concept selection; 1 uses a single full-training ranking.",
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.10)
    parser.add_argument(
        "--low-high-loss-weight",
        type=float,
        default=0.0,
        help="Extra loss weight for the G1/G2 versus G3/G4 cumulative boundary.",
    )
    parser.add_argument(
        "--val-monotonicity-penalty",
        type=float,
        default=0.0,
        help="Penalty per validation mean-risk grade-order violation during checkpoint selection.",
    )
    parser.add_argument("--raw-ordinal-run-dir", type=Path, default=WSI_CF_ROOT / "artifacts/grade_risk_training/kirc_ordinal_grade_risk")
    parser.add_argument("--device", default="cuda:0")
    return parser


def grade_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["raw_grade"]) for row in rows)
    total = float(sum(counts.values()))
    return np.asarray([total / (len(counts) * counts[str(row["raw_grade"])]) for row in rows], dtype=np.float32)


def ordinal_targets(rows: list[dict[str, Any]], target_map: dict[str, float]) -> np.ndarray:
    grades = [grade for grade, _ in sorted(target_map.items(), key=lambda item: item[1])]
    grade_to_index = {grade: index for index, grade in enumerate(grades)}
    return np.asarray(
        [[1.0 if grade_to_index[str(row["raw_grade"])] >= threshold else 0.0 for threshold in range(1, len(grades))] for row in rows],
        dtype=np.float32,
    )


def binary_auroc(y_true: list[int], scores: list[float]) -> float | None:
    n_pos = int(sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(np.asarray(scores), kind="mergesort")
    ranks = np.empty((len(scores),), dtype=np.float64)
    cursor = 0
    values = np.asarray(scores)
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    rank_sum = float(sum(ranks[index] for index, value in enumerate(y_true) if value == 1))
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def metrics_with_grade_detail(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(grade_risk_metrics([float(row["risk_target"]) for row in rows], scores))
    by_grade: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_grade[str(row["raw_grade"])].append(float(score))
    metrics["mean_risk_by_grade"] = {grade: float(np.mean(values)) for grade, values in sorted(by_grade.items())}
    extremes = [(row, score) for row, score in zip(rows, scores) if str(row["raw_grade"]) in {"G1", "G4"}]
    metrics["g1_vs_g4_auroc"] = binary_auroc(
        [int(str(row["raw_grade"]) == "G4") for row, _ in extremes],
        [float(score) for _, score in extremes],
    )
    return metrics


def grade_mean_violations(metrics: dict[str, Any], target_map: dict[str, float]) -> int:
    grade_means = metrics.get("mean_risk_by_grade", {})
    ordered = [grade for grade, _ in sorted(target_map.items(), key=lambda item: item[1]) if grade in grade_means]
    return sum(float(grade_means[left]) >= float(grade_means[right]) for left, right in zip(ordered, ordered[1:]))


def extract_slide_summaries(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    cache_path: Path,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], int]:
    if bool(args.reuse_feature_cache) and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        expected = [str(row["slide_key"]) for row in rows]
        if payload["slide_keys"].astype(str).tolist() != expected:
            raise ValueError(f"Cached slide keys do not match current manifest: {cache_path}")
        expected_metadata = {
            "sae_variant": str(args.sae_variant),
            "sae_batch_size": int(args.sae_batch_size),
            "active_threshold": float(args.active_threshold),
            "top_fraction": float(args.top_fraction),
            "max_tiles_per_slide": int(args.max_tiles_per_slide),
        }
        cached_metadata = json.loads(str(payload["metadata_json"].item())) if "metadata_json" in payload.files else {}
        if cached_metadata != expected_metadata:
            raise ValueError(f"Cached feature extraction settings do not match current args: {cached_metadata} != {expected_metadata}")
        return {name: payload[name] for name in AGGREGATE_NAMES} | {"latent_ids": payload["latent_ids"]}, int(payload["latent_ids"].shape[0])

    sae_ckpt, sae_cfg = resolve_sae_paths(str(args.sae_variant), args.sae_ckpt, args.sae_cfg)
    sae_model, d_in, d_latent = load_sae_from_config(sae_ckpt, sae_cfg, device=str(device))
    if int(d_in) != int(args.embed_dim):
        raise ValueError(f"SAE feature dimension {d_in} does not match expected embed dimension {args.embed_dim}")
    accumulated: dict[str, list[np.ndarray]] = {name: [] for name in AGGREGATE_NAMES}
    for index, row in enumerate(rows, start=1):
        features = prepare_bag(
            row,
            max_tiles=int(args.max_tiles_per_slide),
            seed=int(args.seed),
            epoch=0,
            train=False,
        ).numpy()
        summary = summarize_feature_bag_with_sae(
            features,
            sae_model=sae_model,
            device=device,
            batch_size=int(args.sae_batch_size),
            d_latent=int(d_latent),
            active_threshold=float(args.active_threshold),
            top_fraction=float(args.top_fraction),
        )
        for name in AGGREGATE_NAMES:
            accumulated[name].append(summary[name])
        if index % 25 == 0 or index == len(rows):
            print(f"[features] {index}/{len(rows)} slides encoded", flush=True)
    arrays = {name: np.stack(values, axis=0).astype(np.float32) for name, values in accumulated.items()}
    arrays["latent_ids"] = np.arange(int(d_latent), dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        slide_keys=np.asarray([str(row["slide_key"]) for row in rows]),
        splits=np.asarray([str(row["split"]) for row in rows]),
        grades=np.asarray([str(row["raw_grade"]) for row in rows]),
        latent_ids=arrays["latent_ids"],
        metadata_json=np.asarray(
            json.dumps(
                {
                    "sae_variant": str(args.sae_variant),
                    "sae_batch_size": int(args.sae_batch_size),
                    "active_threshold": float(args.active_threshold),
                    "top_fraction": float(args.top_fraction),
                    "max_tiles_per_slide": int(args.max_tiles_per_slide),
                },
                sort_keys=True,
            )
        ),
        **{name: arrays[name] for name in AGGREGATE_NAMES},
    )
    return arrays, int(d_latent)


def prediction_rows(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, score, probs in zip(rows, scores.tolist(), probabilities.tolist()):
        item = dict(row)
        item.update(
            {
                "risk_score": float(score),
                "error": float(score - float(row["risk_target"])),
                "abs_error": abs(float(score - float(row["risk_target"]))),
                "prob_ge_g2": float(probs[0]),
                "prob_ge_g3": float(probs[1]),
                "prob_ge_g4": float(probs[2]),
            }
        )
        output.append(item)
    return output


@torch.no_grad()
def evaluate_model(
    model: LinearProportionalOddsRisk,
    features: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    scores, probabilities, _ = model(x)
    score_array = scores.detach().cpu().numpy()
    prob_array = probabilities.detach().cpu().numpy()
    metrics = metrics_with_grade_detail(rows, score_array.astype(float).tolist())
    metrics["low_high_auroc"] = binary_auroc(
        [int(str(row["raw_grade"]) in {"G3", "G4"}) for row in rows],
        prob_array[:, 1].astype(float).tolist(),
    )
    return metrics, prediction_rows(rows, score_array, prob_array)


def load_raw_benchmark(path: Path, *, expected_ids: set[str], case_level: bool) -> dict[str, Any]:
    predictions = path / "test_predictions.csv"
    if not predictions.exists():
        return {"available": False, "path": str(path), "reason": "missing test_predictions.csv"}
    with predictions.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "risk_score" not in rows[0] or "raw_grade" not in rows[0]:
        return {"available": False, "path": str(path), "reason": "predictions lack risk_score/raw_grade"}
    if case_level:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["case_id"])].append(row)
        if set(grouped) != expected_ids:
            return {"available": False, "path": str(path), "reason": "held-out case set does not match SAE model test split"}
        rows = [
            {
                "raw_grade": values[0]["raw_grade"],
                "risk_target": values[0]["risk_target"],
                "risk_score": float(np.mean([float(row["risk_score"]) for row in values])),
            }
            for values in grouped.values()
        ]
    elif {str(row["slide_key"]) for row in rows} != expected_ids:
        return {"available": False, "path": str(path), "reason": "held-out slide set does not match SAE model test split"}
    detail_rows = [{"raw_grade": row["raw_grade"], "risk_target": float(row["risk_target"])} for row in rows]
    return {"available": True, "path": str(path), "metrics": metrics_with_grade_detail(detail_rows, [float(row["risk_score"]) for row in rows])}


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not 0.0 < float(args.top_fraction) <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1]")
    if int(args.selection_folds) < 1:
        raise ValueError("--selection-folds must be positive")
    if float(args.low_high_loss_weight) < 0.0 or float(args.val_monotonicity_penalty) < 0.0:
        raise ValueError("Loss weights and validation penalties must be non-negative")
    set_seed(int(args.seed))
    target_map = parse_target_map(str(args.target_map))
    out_dir = args.out_dir / str(args.task_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = args.feature_cache or out_dir / "slide_sae_features.npz"
    sae_ckpt, sae_cfg = resolve_sae_paths(str(args.sae_variant), args.sae_ckpt, args.sae_cfg)

    rows, skipped = load_grade_rows(args, target_map)
    rows = assign_validation_split(rows, fraction=float(args.val_fraction), seed=int(args.seed))
    if not all(any(row["split"] == split for row in rows) for split in ("train", "val", "test")):
        raise RuntimeError("Need non-empty train, validation, and test splits")
    manifest_fields = ["case_id", "slide_key", "sample_id", "sample_code", "project_dir", "raw_grade", "risk_target", "source_split", "split", "h5_path"]
    write_csv(out_dir / "task_manifest.csv", rows, manifest_fields)
    write_csv(out_dir / "skipped_feature_files.csv", skipped)
    args_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args_payload.update(
        {
            "target_map": target_map,
            "sae_ckpt": str(sae_ckpt),
            "sae_cfg": str(sae_cfg),
            "feature_cache": str(feature_cache),
            "aggregate_names": list(AGGREGATE_NAMES),
        }
    )
    write_json(out_dir / "args.json", args_payload)
    data_summary = {
        "slides": {split: int(sum(row["split"] == split for row in rows)) for split in ("train", "val", "test")},
        "grade_counts": {split: dict(Counter(row["raw_grade"] for row in rows if row["split"] == split)) for split in ("train", "val", "test")},
        "cases": {split: len({str(row["case_id"]) for row in rows if row["split"] == split}) for split in ("train", "val", "test")},
        "case_grade_counts": {
            split: dict(Counter({str(row["case_id"]): str(row["raw_grade"]) for row in rows if row["split"] == split}.values()))
            for split in ("train", "val", "test")
        },
        "skipped_feature_files": len(skipped),
    }
    if bool(args.dry_run):
        write_json(out_dir / "summary.json", {"args": args_payload, "data": data_summary, "dry_run": True})
        print(json.dumps(data_summary, indent=2))
        return

    device = resolve_device(str(args.device))
    arrays, d_latent = extract_slide_summaries(rows, args=args, cache_path=feature_cache, device=device)
    if bool(args.extract_only):
        write_json(out_dir / "summary.json", {"args": args_payload, "data": data_summary, "d_latent": d_latent, "extract_only": True})
        return

    model_rows = rows
    model_arrays = arrays
    model_manifest_path = out_dir / "task_manifest.csv"
    if bool(args.case_level):
        model_rows, model_arrays = aggregate_summaries_by_case(rows, arrays)
        model_manifest_path = out_dir / "case_task_manifest.csv"
        write_csv(model_manifest_path, model_rows)
    train_indices = np.asarray([index for index, row in enumerate(model_rows) if row["split"] == "train"], dtype=np.int64)
    val_indices = np.asarray([index for index, row in enumerate(model_rows) if row["split"] == "val"], dtype=np.int64)
    test_indices = np.asarray([index for index, row in enumerate(model_rows) if row["split"] == "test"], dtype=np.int64)
    train_rows = [model_rows[index] for index in train_indices.tolist()]
    val_rows = [model_rows[index] for index in val_indices.tolist()]
    test_rows = [model_rows[index] for index in test_indices.tolist()]
    selection_fn = select_stable_top_latents if int(args.selection_folds) > 1 else select_top_latents
    selection_kwargs: dict[str, Any] = {}
    if int(args.selection_folds) > 1:
        selection_kwargs = {
            "strata": [str(row["raw_grade"]) for row in train_rows],
            "n_folds": int(args.selection_folds),
            "seed": int(args.seed),
        }
    selected_latents, selected_rows = selection_fn(
        {name: model_arrays[name][train_indices] for name in AGGREGATE_NAMES},
        [float(row["risk_target"]) for row in train_rows],
        latent_ids=model_arrays["latent_ids"],
        top_n=int(args.top_concepts),
        **selection_kwargs,
    )
    write_csv(out_dir / "selected_concepts.csv", selected_rows)
    feature_matrix, feature_columns = build_feature_matrix(
        {name: model_arrays[name] for name in AGGREGATE_NAMES},
        selected_latents=selected_latents,
        latent_ids=model_arrays["latent_ids"],
    )
    mean, scale = fit_feature_scaler(feature_matrix[train_indices])
    scaled = apply_feature_scaler(feature_matrix, mean, scale)
    write_json(
        out_dir / "feature_scaler.json",
        {
            "feature_columns": feature_columns,
            "mean": mean.astype(float).tolist(),
            "scale": scale.astype(float).tolist(),
        },
    )

    model = LinearProportionalOddsRisk(n_features=int(scaled.shape[1]), n_thresholds=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    x_train = torch.as_tensor(scaled[train_indices], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(ordinal_targets(train_rows, target_map), dtype=torch.float32, device=device)
    weight_train = torch.as_tensor(grade_weights(train_rows), dtype=torch.float32, device=device)
    train_targets = np.asarray([float(row["risk_target"]) for row in train_rows], dtype=np.float32)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_spearman = -math.inf
    best_mae = math.inf
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk, _, logits = model(x_train)
        bce = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none").mean(dim=1)
        primary_loss = (bce * weight_train).mean()
        low_high_bce = F.binary_cross_entropy_with_logits(logits[:, 1], y_train[:, 1], reduction="none")
        low_high_loss = (low_high_bce * weight_train).mean()
        pair_rng = random.Random(int(args.seed) + epoch)
        pairs: list[tuple[int, int, float]] = []
        for index, target in enumerate(train_targets.tolist()):
            candidates = [j for j, other in enumerate(train_targets.tolist()) if other != target]
            comparison = pair_rng.choice(candidates)
            sign = 1.0 if target > float(train_targets[comparison]) else -1.0
            pairs.append((index, comparison, sign))
        left = torch.as_tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
        right = torch.as_tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
        signs = torch.as_tensor([pair[2] for pair in pairs], dtype=torch.float32, device=device)
        ranking_loss = F.relu(float(args.ranking_margin) - signs * (risk[left] - risk[right])).mean()
        loss = (
            primary_loss
            + float(args.ranking_loss_weight) * ranking_loss
            + float(args.low_high_loss_weight) * low_high_loss
        )
        loss.backward()
        optimizer.step()
        val_metrics, _ = evaluate_model(model, scaled[val_indices], val_rows, device=device)
        val_spearman = -math.inf if val_metrics["spearman"] is None else float(val_metrics["spearman"])
        val_violations = grade_mean_violations(val_metrics, target_map)
        val_selection_score = val_spearman - float(args.val_monotonicity_penalty) * val_violations
        metric_row = {
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu().item()),
            "train_bce_loss": float(primary_loss.detach().cpu().item()),
            "train_ranking_loss": float(ranking_loss.detach().cpu().item()),
            "train_low_high_loss": float(low_high_loss.detach().cpu().item()),
            "val_monotonicity_violations": int(val_violations),
            "val_selection_score": float(val_selection_score),
            **{f"val_{key}": value for key, value in val_metrics.items() if key != "mean_risk_by_grade"},
        }
        history.append(metric_row)
        improved = val_selection_score > best_spearman + 1e-12 or (
            abs(val_selection_score - best_spearman) <= 1e-12 and float(val_metrics["mae"]) < best_mae
        )
        if improved:
            best_spearman = val_selection_score
            best_mae = float(val_metrics["mae"])
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                "epoch": epoch,
                "prediction_task": "sae_concept_ordinal_grade_risk",
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_config": {"n_features": int(scaled.shape[1]), "n_thresholds": 3},
                "metrics": val_metrics,
                "args": args_payload,
                "target_map": target_map,
                "sae": {"variant": str(args.sae_variant), "checkpoint": str(sae_ckpt), "config": str(sae_cfg)},
                "selected_latents": selected_latents.tolist(),
                "aggregate_names": list(AGGREGATE_NAMES),
                "active_threshold": float(args.active_threshold),
                "top_fraction": float(args.top_fraction),
                "scaler_mean": mean.tolist(),
                "scaler_scale": scale.tolist(),
                "feature_columns": feature_columns,
                "ordered_thresholds": model.thresholds.detach().cpu().tolist(),
                "split_name": "sae_patient_train_val_test_90_10",
                "model_unit": "case" if bool(args.case_level) else "slide",
                "case_aggregation": "mean_of_slide_sae_aggregates" if bool(args.case_level) else None,
                "validation_selection_score": float(val_selection_score),
            }
            torch.save(best_state, out_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
        print(json.dumps(metric_row), flush=True)
        if epochs_without_improvement >= int(args.early_stopping_patience):
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a selected checkpoint")
    torch.save(
        {
            **best_state,
            "epoch": int(history[-1]["epoch"]),
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metrics": history[-1],
            "ordered_thresholds": model.thresholds.detach().cpu().tolist(),
        },
        out_dir / "final_model.pt",
    )
    model.load_state_dict(best_state["model_state_dict"])
    model.to(device)
    validation_metrics, validation_predictions = evaluate_model(model, scaled[val_indices], val_rows, device=device)
    test_metrics, test_predictions = evaluate_model(model, scaled[test_indices], test_rows, device=device)
    write_csv(out_dir / "train_metrics.csv", history)
    write_csv(out_dir / "validation_predictions.csv", validation_predictions)
    write_csv(out_dir / "test_predictions.csv", test_predictions)
    standard_weights = model.linear.weight.detach().cpu().numpy().reshape(-1)
    coefficient_rows = [
        {
            **column,
            "standardized_coefficient": float(standard_weights[index]),
            "original_scale_coefficient": float(standard_weights[index] / scale[index]),
        }
        for index, column in enumerate(feature_columns)
    ]
    coefficient_rows.sort(key=lambda row: -abs(float(row["standardized_coefficient"])))
    write_csv(out_dir / "model_coefficients.csv", coefficient_rows)
    expected_ids = {str(row["case_id"] if bool(args.case_level) else row["slide_key"]) for row in test_rows}
    raw_benchmark = load_raw_benchmark(args.raw_ordinal_run_dir, expected_ids=expected_ids, case_level=bool(args.case_level))
    comparison_rows = [
        {"model": "sae_concept_ordinal", **{key: value for key, value in test_metrics.items() if key != "mean_risk_by_grade"}}
    ]
    if raw_benchmark.get("available"):
        comparison_rows.append({"model": "raw_embedding_ordinal", **{key: value for key, value in raw_benchmark["metrics"].items() if key != "mean_risk_by_grade"}})
    write_csv(out_dir / "raw_vs_sae_test_comparison.csv", comparison_rows)
    write_json(
        out_dir / "summary.json",
        {
            "args": args_payload,
            "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:]))),
            "data": data_summary,
            "sae_d_latent": d_latent,
            "selected_concepts": int(selected_latents.shape[0]),
            "model_input_features": int(scaled.shape[1]),
            "model_unit": "case" if bool(args.case_level) else "slide",
            "case_aggregation": "mean_of_slide_sae_aggregates" if bool(args.case_level) else None,
            "best_epoch": best_epoch,
            "best_validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "ordered_thresholds": model.thresholds.detach().cpu().tolist(),
            "raw_embedding_benchmark": raw_benchmark,
            "outputs": {
                "feature_cache": str(feature_cache),
                "model_manifest": str(model_manifest_path),
                "selected_concepts": str(out_dir / "selected_concepts.csv"),
                "feature_scaler": str(out_dir / "feature_scaler.json"),
                "model_coefficients": str(out_dir / "model_coefficients.csv"),
                "best_model": str(out_dir / "best_model.pt"),
                "test_predictions": str(out_dir / "test_predictions.csv"),
                "raw_vs_sae_test_comparison": str(out_dir / "raw_vs_sae_test_comparison.csv"),
            },
        },
    )
    print(json.dumps({"best_epoch": best_epoch, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
