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

from train_attention_classifier import (  # noqa: E402
    DEFAULT_FEATURES_ROOT,
    DEFAULT_LABEL_SOURCE,
    DEFAULT_SPLIT_MANIFEST,
    UNKNOWN_VALUES,
    case_id_from_slide_key,
    load_split_cases,
    parse_csv_list,
    prepare_bag,
    project_has_features,
    sample_code_from_slide_key,
    validate_feature_h5,
    write_csv,
)
from wsi_cf.common.io import write_json  # noqa: E402
from wsi_cf.common.runtime import resolve_device, set_seed  # noqa: E402
from wsi_cf.eval.grade_risk import build_grade_risk_model, grade_risk_metrics  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a continuous MIL grade-risk predictor from ordinal tumor grades.")
    parser.add_argument("--task-name", type=str, default="kirc_continuous_grade_risk")
    parser.add_argument("--label-source", type=Path, default=DEFAULT_LABEL_SOURCE)
    parser.add_argument("--projects", type=str, default="TCGA-KIRC")
    parser.add_argument("--label-column", type=str, default="tumor_grade")
    parser.add_argument("--target-map", type=str, default="G1:0.0,G2:0.33,G3:0.66,G4:1.0")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/grade_risk_training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-h5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-tiles-per-slide", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--model", choices=["gated", "attention"], default="gated")
    parser.add_argument("--embed-dim", type=int, default=1536)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--attn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--fixed-temperature", action="store_true")
    parser.add_argument("--init-temperature", type=float, default=1.0)
    parser.add_argument("--objective", choices=["regression", "ordinal"], default="regression")
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--grade-balanced-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ranking-loss-weight",
        type=float,
        default=0.0,
        help="Weight for pairwise higher-grade-above-lower-grade margin loss.",
    )
    parser.add_argument("--ranking-margin", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def parse_target_map(value: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for item in parse_csv_list(value):
        if ":" not in item:
            raise ValueError(f"Invalid --target-map item '{item}'. Expected grade:score")
        grade, score = item.split(":", 1)
        score_float = float(score)
        if not 0.0 <= score_float <= 1.0:
            raise ValueError(f"Grade-risk target must be within [0, 1], got {grade}:{score_float}")
        mapping[grade.strip()] = score_float
    if len(mapping) < 2:
        raise ValueError("Need at least two ordinal grades in --target-map")
    return mapping


def load_grade_rows(args: argparse.Namespace, target_map: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    projects = set(parse_csv_list(args.projects))
    use_all_projects = projects == {"all"}
    train_cases, test_cases = load_split_cases(args.split_manifest)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with args.label_source.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if args.label_column not in (reader.fieldnames or []):
            raise ValueError(f"Label column '{args.label_column}' not found in {args.label_source}")
        for source_row in reader:
            project = str(source_row.get("project_dir", ""))
            if not use_all_projects and project not in projects:
                continue
            if use_all_projects and not project_has_features(args.features_root, project):
                continue
            grade = str(source_row.get(args.label_column, "")).strip()
            if grade.upper() in UNKNOWN_VALUES or grade not in target_map:
                continue
            slide_key = str(source_row.get("slide_key", ""))
            case_id = str(source_row.get("case_id", "")) or case_id_from_slide_key(slide_key)
            source_split = "train" if case_id in train_cases else "test" if case_id in test_cases else ""
            if not source_split:
                continue
            h5_path = args.features_root / project / "features_uni2" / f"{slide_key}.h5"
            reason = ""
            if not h5_path.exists():
                reason = "missing_h5"
            elif bool(args.validate_h5):
                reason = validate_feature_h5(h5_path, embed_dim=int(args.embed_dim))
            if reason:
                skipped.append(
                    {
                        "case_id": case_id,
                        "slide_key": slide_key,
                        "project_dir": project,
                        "raw_grade": grade,
                        "source_split": source_split,
                        "h5_path": str(h5_path),
                        "reason": reason,
                    }
                )
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "slide_key": slide_key,
                    "sample_id": str(source_row.get("sample_id", "")),
                    "sample_code": sample_code_from_slide_key(slide_key),
                    "project_dir": project,
                    "raw_grade": grade,
                    "risk_target": float(target_map[grade]),
                    "source_split": source_split,
                    "split": source_split,
                    "h5_path": str(h5_path),
                }
            )
    return rows, skipped


def assign_validation_split(rows: list[dict[str, Any]], *, fraction: float, seed: int) -> list[dict[str, Any]]:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    by_grade_cases: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["source_split"] == "train":
            by_grade_cases[str(row["raw_grade"])].add(str(row["case_id"]))
    val_cases: set[str] = set()
    for grade, case_ids in sorted(by_grade_cases.items()):
        ordered = sorted(case_ids)
        random.Random(int(seed) + sum(ord(ch) for ch in grade)).shuffle(ordered)
        n_val = max(1, int(round(len(ordered) * float(fraction))))
        n_val = min(n_val, max(len(ordered) - 1, 1))
        val_cases.update(ordered[:n_val])
    final_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item["source_split"] == "train" and item["case_id"] in val_cases:
            item["split"] = "val"
        final_rows.append(item)
    return final_rows


def grade_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(str(row["raw_grade"]) for row in rows)
    total = float(sum(counts.values()))
    return {grade: total / (len(counts) * float(count)) for grade, count in counts.items()}


def regression_loss(prediction: torch.Tensor, target: torch.Tensor, *, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target, reduction="none")
    return F.smooth_l1_loss(prediction, target, reduction="none")


def ordinal_target(raw_grade: str, target_map: dict[str, float], *, device: torch.device) -> torch.Tensor:
    ordered_grades = [grade for grade, _ in sorted(target_map.items(), key=lambda item: item[1])]
    grade_index = ordered_grades.index(str(raw_grade))
    values = [1.0 if grade_index >= threshold else 0.0 for threshold in range(1, len(ordered_grades))]
    return torch.as_tensor([values], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    max_tiles_per_slide: int,
    seed: int,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    model.eval()
    targets: list[float] = []
    scores: list[float] = []
    predictions: list[dict[str, Any]] = []
    for row in rows:
        x = prepare_bag(row, max_tiles=max_tiles_per_slide, seed=seed, epoch=0, train=False).to(device)
        risk_score, _, results = model(x)
        score = float(risk_score.detach().cpu().reshape(-1)[0].item())
        target = float(row["risk_target"])
        item = dict(row)
        item.update({"risk_score": score, "error": score - target, "abs_error": abs(score - target)})
        if "ordinal_probs" in results:
            probabilities = results["ordinal_probs"].detach().cpu().numpy().reshape(-1).astype(float).tolist()
            for index, probability in enumerate(probabilities, start=2):
                item[f"prob_ge_g{index}"] = probability
        predictions.append(item)
        targets.append(target)
        scores.append(score)
    if not predictions:
        raise RuntimeError("No evaluation feature bags were available")
    return grade_risk_metrics(targets, scores), predictions


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args_payload: dict[str, Any],
    target_map: dict[str, float],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "prediction_task": f"{args_payload.get('objective', 'regression')}_grade_risk",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": args_payload,
            "target_map": target_map,
            "metrics": metrics,
            "split_name": "sae_patient_train_val_test_90_10",
        },
        path,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(int(args.seed))
    target_map = parse_target_map(args.target_map)
    out_dir = args.out_dir / str(args.task_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, skipped = load_grade_rows(args, target_map)
    rows = assign_validation_split(rows, fraction=float(args.val_fraction), seed=int(args.seed))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    case_sets = {split: {row["case_id"] for row in subset} for split, subset in (("train", train_rows), ("val", val_rows), ("test", test_rows))}
    if any(case_sets[a] & case_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Case leakage detected among train, validation, and test sets")
    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError(f"Need non-empty train/val/test rows; got {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

    args_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args_payload["target_map"] = target_map
    args_payload["n_thresholds"] = max(len(target_map) - 1, 1)
    manifest_fields = [
        "case_id", "slide_key", "sample_id", "sample_code", "project_dir", "raw_grade",
        "risk_target", "source_split", "split", "h5_path",
    ]
    write_csv(out_dir / "task_manifest.csv", rows, manifest_fields)
    write_csv(out_dir / "skipped_feature_files.csv", skipped)
    write_json(out_dir / "target_mapping.json", target_map)
    write_json(out_dir / "args.json", args_payload)
    data_summary = {
        "target_map": target_map,
        "slides": {split: len(subset) for split, subset in (("train", train_rows), ("val", val_rows), ("test", test_rows))},
        "cases": {split: len(case_sets[split]) for split in ("train", "val", "test")},
        "grade_counts": {
            split: dict(Counter(row["raw_grade"] for row in subset))
            for split, subset in (("train", train_rows), ("val", val_rows), ("test", test_rows))
        },
        "skipped_feature_files": len(skipped),
    }
    if args.dry_run:
        write_json(out_dir / "summary.json", {"args": args_payload, "data": data_summary, "dry_run": True})
        print(json.dumps(data_summary, indent=2))
        return

    device = resolve_device(args.device)
    model = build_grade_risk_model(args_payload).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    weights = grade_weights(train_rows) if bool(args.grade_balanced_loss) else defaultdict(lambda: 1.0)
    history: list[dict[str, Any]] = []
    best_val_mae = math.inf
    best_epoch = -1
    best_val_metrics: dict[str, Any] = {}

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_rows = list(train_rows)
        random.Random(int(args.seed) + epoch).shuffle(epoch_rows)
        losses: list[float] = []
        ranking_losses: list[float] = []
        different_grade_rows = {
            grade: [row for row in train_rows if str(row["raw_grade"]) != grade]
            for grade in {str(row["raw_grade"]) for row in train_rows}
        }
        pair_rng = random.Random(int(args.seed) + 1000 + epoch)
        for row in epoch_rows:
            x = prepare_bag(row, max_tiles=int(args.max_tiles_per_slide), seed=int(args.seed), epoch=epoch, train=True).to(device)
            target = torch.as_tensor([[float(row["risk_target"])]], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            score, _, results = model(x)
            if str(args.objective) == "ordinal":
                grade_target = ordinal_target(str(row["raw_grade"]), target_map, device=device)
                primary_loss = F.binary_cross_entropy_with_logits(results["ordinal_logits"], grade_target)
            else:
                primary_loss = regression_loss(score, target, loss_name=str(args.loss)).mean()
            loss = primary_loss * float(weights[str(row["raw_grade"])])
            if float(args.ranking_loss_weight) > 0.0:
                comparison = pair_rng.choice(different_grade_rows[str(row["raw_grade"])])
                x_comparison = prepare_bag(
                    comparison,
                    max_tiles=int(args.max_tiles_per_slide),
                    seed=int(args.seed),
                    epoch=epoch,
                    train=True,
                ).to(device)
                comparison_score, _, _ = model(x_comparison)
                sign = 1.0 if float(row["risk_target"]) > float(comparison["risk_target"]) else -1.0
                ranking_loss = F.relu(
                    torch.as_tensor(float(args.ranking_margin), device=device)
                    - sign * (score - comparison_score)
                ).mean()
                loss = loss + float(args.ranking_loss_weight) * ranking_loss
                ranking_losses.append(float(ranking_loss.detach().cpu().item()))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics, _ = evaluate(
            model, val_rows, device=device, max_tiles_per_slide=int(args.max_tiles_per_slide), seed=int(args.seed)
        )
        metric_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_ranking_loss": float(np.mean(ranking_losses)) if ranking_losses else "",
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(metric_row)
        if float(val_metrics["mae"]) < best_val_mae:
            best_val_mae = float(val_metrics["mae"])
            best_epoch = epoch
            best_val_metrics = val_metrics
            save_checkpoint(
                out_dir / "best_model.pt", model=model, optimizer=optimizer, epoch=epoch,
                args_payload=args_payload, target_map=target_map, metrics=val_metrics,
            )
        print(json.dumps(metric_row))

    save_checkpoint(
        out_dir / "final_model.pt", model=model, optimizer=optimizer, epoch=int(args.epochs),
        args_payload=args_payload, target_map=target_map, metrics=history[-1],
    )
    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_predictions = evaluate(
        model, test_rows, device=device, max_tiles_per_slide=int(args.max_tiles_per_slide), seed=int(args.seed)
    )
    _, val_predictions = evaluate(
        model, val_rows, device=device, max_tiles_per_slide=int(args.max_tiles_per_slide), seed=int(args.seed)
    )
    write_csv(out_dir / "train_metrics.csv", history)
    write_csv(out_dir / "validation_predictions.csv", val_predictions)
    write_csv(out_dir / "test_predictions.csv", test_predictions)
    write_json(
        out_dir / "summary.json",
        {
            "args": args_payload,
            "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:]))),
            "data": data_summary,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_val_metrics,
            "test_metrics_at_best_validation_epoch": test_metrics,
            "outputs": {
                "best_model": str(out_dir / "best_model.pt"),
                "task_manifest": str(out_dir / "task_manifest.csv"),
                "train_metrics": str(out_dir / "train_metrics.csv"),
                "test_predictions": str(out_dir / "test_predictions.csv"),
            },
        },
    )
    print(json.dumps({"best_epoch": best_epoch, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
