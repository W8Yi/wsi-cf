#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.runtime import resolve_device
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint


GROUPS = [
    "pattern_1_3_well_formed",
    "pattern_4_cribriform_poorly_formed_fused",
    "pattern_5_solid_single_necrosis",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare direct PRAD morphology classifier against collapsed 5-class grade-group classifier."
    )
    parser.add_argument("--slide-labels", type=Path, default=Path("artifacts/prad_gleason_inputs/slide_labels.csv"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/prad_gleason_inputs/splits/prad_morphology_group_patient_stratified_80_20.json"),
    )
    parser.add_argument("--morphology-run-dir", type=Path, default=Path("artifacts/classifier_training/prad_morphology_group"))
    parser.add_argument("--grade-run-dir", type=Path, default=Path("artifacts/classifier_training/prad_grade_group"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/classifier_training/prad_morphology_group_comparison"))
    parser.add_argument("--max-tiles-per-slide", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def slide_key_from_path(path_value: str) -> str:
    return Path(str(path_value)).name.split(".")[0]


def load_test_slide_keys(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    return {slide_key_from_path(item) for item in payload.get("test", [])}


def load_label_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text())
    label_to_id = {str(k): int(v) for k, v in payload["label_to_id"].items()}
    return label_to_id, {idx: label for label, idx in label_to_id.items()}


def read_features(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        features = np.asarray(handle["features"], dtype=np.float32)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 2:
        raise ValueError(f"{path}: expected features shape [N, D], got {features.shape}")
    return features


def stable_seed(seed: int, *parts: Any) -> int:
    import hashlib

    digest = hashlib.md5("::".join([str(seed), *map(str, parts)]).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def maybe_sample_tiles(features: np.ndarray, *, max_tiles: int, seed: int, slide_key: str) -> np.ndarray:
    if int(max_tiles) <= 0 or int(features.shape[0]) <= int(max_tiles):
        return features
    rng = np.random.default_rng(stable_seed(seed, "eval", slide_key))
    idx = rng.choice(features.shape[0], size=int(max_tiles), replace=False)
    idx.sort()
    return np.asarray(features[idx], dtype=np.float32)


@torch.no_grad()
def score_rows(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    *,
    id_to_label: dict[int, str],
    device: torch.device,
    max_tiles: int,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    model.eval()
    for row in rows:
        features = maybe_sample_tiles(
            read_features(Path(row["h5_path"])),
            max_tiles=int(max_tiles),
            seed=int(seed),
            slide_key=str(row["slide_key"]),
        )
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        _, y_prob, y_hat, _, _ = model(x)
        probs = y_prob.detach().cpu().numpy().reshape(-1).astype(float)
        pred_id = int(y_hat.detach().cpu().reshape(-1)[0].item())
        item: dict[str, Any] = {
            "case_id": row["case_id"],
            "slide_key": row["slide_key"],
            "h5_path": row["h5_path"],
            "true_morphology_group": row["morphology_group"],
            "true_grade_group": row["grade_group"],
            "gleason_pattern": row["gleason_pattern"],
            "pred_label": id_to_label[pred_id],
        }
        for index, prob in enumerate(probs.tolist()):
            item[f"prob_{index}"] = float(prob)
        out.append(item)
    return out


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    cm = [[sum(t == a and p == b for t, p in zip(y_true, y_pred)) for b in GROUPS] for a in GROUPS]
    per_class: list[dict[str, Any]] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index, label in enumerate(GROUPS):
        tp = float(cm[index][index])
        fp = float(sum(row[index] for row in cm) - cm[index][index])
        fn = float(sum(cm[index]) - cm[index][index])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        support = int(sum(cm[index]))
        per_class.append(
            {
                "label": label,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
        if support:
            recalls.append(float(recall))
            f1s.append(float(f1))
    total = max(len(y_true), 1)
    return {
        "n": int(len(y_true)),
        "accuracy": float(sum(t == p for t, p in zip(y_true, y_pred)) / total),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "confusion_matrix": cm,
        "per_class": per_class,
        "true_counts": dict(sorted(Counter(y_true).items())),
        "pred_counts": dict(sorted(Counter(y_pred).items())),
    }


def collapsed_label_from_grade(label: str, *, gg4_to: str) -> str:
    if label == "GG1":
        return GROUPS[0]
    if label in {"GG2", "GG3"}:
        return GROUPS[1]
    if label == "GG4":
        return gg4_to
    if label == "GG5":
        return GROUPS[2]
    raise ValueError(f"Unexpected grade-group label: {label}")


def gg4_pattern4_fraction(rows: list[dict[str, str]]) -> float:
    gg4 = [row for row in rows if row["grade_group"] == "GG4"]
    if not gg4:
        return 0.5
    return float(sum(row["morphology_group"] == GROUPS[1] for row in gg4) / len(gg4))


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    test_keys = load_test_slide_keys(args.split_manifest)
    label_rows = [row for row in read_csv(args.slide_labels) if row["slide_key"] in test_keys]
    label_rows.sort(key=lambda row: row["slide_key"])
    if not label_rows:
        raise ValueError(f"No test rows found from {args.split_manifest}")

    _, morph_id_to_label = load_label_mapping(args.morphology_run_dir / "label_mapping.json")
    _, grade_id_to_label = load_label_mapping(args.grade_run_dir / "label_mapping.json")
    morph_model = build_mil_from_checkpoint(args.morphology_run_dir / "best_model.pt", device=device)
    grade_model = build_mil_from_checkpoint(args.grade_run_dir / "best_model.pt", device=device)

    morph_scores = score_rows(
        morph_model,
        label_rows,
        id_to_label=morph_id_to_label,
        device=device,
        max_tiles=int(args.max_tiles_per_slide),
        seed=int(args.seed),
    )
    grade_scores = score_rows(
        grade_model,
        label_rows,
        id_to_label=grade_id_to_label,
        device=device,
        max_tiles=int(args.max_tiles_per_slide),
        seed=int(args.seed),
    )
    true = [row["morphology_group"] for row in label_rows]

    alpha = gg4_pattern4_fraction(read_csv(args.slide_labels))
    result_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    direct_pred = [row["pred_label"] for row in morph_scores]
    summaries["direct_3_class"] = metrics(true, direct_pred)

    for mode, gg4_to in [
        ("collapsed_5_class_gg4_to_pattern4", GROUPS[1]),
        ("collapsed_5_class_gg4_to_pattern5", GROUPS[2]),
    ]:
        pred = [collapsed_label_from_grade(row["pred_label"], gg4_to=gg4_to) for row in grade_scores]
        summaries[mode] = metrics(true, pred)

    split_pred: list[str] = []
    for row in grade_scores:
        probs = [float(row[f"prob_{i}"]) for i in range(5)]
        group_probs = [
            probs[0],
            probs[1] + probs[2] + float(alpha) * probs[3],
            (1.0 - float(alpha)) * probs[3] + probs[4],
        ]
        split_pred.append(GROUPS[int(np.argmax(group_probs))])
    summaries["collapsed_5_class_probability_split_gg4"] = metrics(true, split_pred)

    for idx, row in enumerate(label_rows):
        out = {
            "case_id": row["case_id"],
            "slide_key": row["slide_key"],
            "gleason_pattern": row["gleason_pattern"],
            "grade_group": row["grade_group"],
            "true_morphology_group": row["morphology_group"],
            "direct_3_class_pred": direct_pred[idx],
            "collapsed_5_class_gg4_to_pattern4_pred": collapsed_label_from_grade(
                grade_scores[idx]["pred_label"], gg4_to=GROUPS[1]
            ),
            "collapsed_5_class_gg4_to_pattern5_pred": collapsed_label_from_grade(
                grade_scores[idx]["pred_label"], gg4_to=GROUPS[2]
            ),
            "collapsed_5_class_probability_split_gg4_pred": split_pred[idx],
            "grade_group_pred": grade_scores[idx]["pred_label"],
        }
        result_rows.append(out)

    payload = {
        "split_manifest": str(args.split_manifest),
        "slide_labels": str(args.slide_labels),
        "morphology_run_dir": str(args.morphology_run_dir),
        "grade_run_dir": str(args.grade_run_dir),
        "n_test_slides": int(len(label_rows)),
        "gg4_pattern4_fraction": float(alpha),
        "labels": GROUPS,
        "summaries": summaries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "comparison_summary.json", payload)
    write_csv(args.out_dir / "comparison_by_slide.csv", result_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
