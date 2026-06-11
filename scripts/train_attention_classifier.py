#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.common.paths import WSI_CF_ROOT
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.models.mil import AttentionMIL, GatedAttentionMIL


DEFAULT_LABEL_SOURCE = WSI_CF_ROOT / "resources/labels/master/slide_labels_master.tsv"
DEFAULT_SPLIT_MANIFEST = WSI_CF_ROOT / "resources/manifests/sae_manifests_tcga_patient_train_test_90_10.json"
DEFAULT_FEATURES_ROOT = Path("/research/projects/mllab/WSI/TCGA_features")
UNKNOWN_VALUES = {"", "UNKNOWN", "UNK", "NA", "N/A", "NONE", "NULL"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train repo-native attention MIL classifiers on TCGA UNI2 feature bags.")
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--label-source", type=Path, default=DEFAULT_LABEL_SOURCE)
    parser.add_argument("--projects", type=str, default="all")
    parser.add_argument("--label-column", type=str, required=True)
    parser.add_argument("--include-labels", type=str, default="")
    parser.add_argument("--label-map", type=str, default="", help="Optional comma list like raw:new,raw2:new2.")
    parser.add_argument("--label-order", type=str, default="", help="Optional comma list fixing class id order, e.g. low,high.")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/classifier_training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-h5",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open each candidate H5 during manifest construction and skip unreadable/malformed feature files.",
    )

    parser.add_argument("--max-train-slides", type=int, default=0)
    parser.add_argument("--max-test-slides", type=int, default=0)
    parser.add_argument("--max-slides-per-class", type=int, default=0)
    parser.add_argument("--min-slides-per-class", type=int, default=1)
    parser.add_argument("--max-tiles-per-slide", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--model", type=str, default="gated", choices=["gated", "attention"])
    parser.add_argument("--embed-dim", type=int, default=1536)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--attn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--fixed-temperature", action="store_true")
    parser.add_argument("--init-temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-balanced-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def parse_csv_list(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def parse_label_map(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in parse_csv_list(value):
        if ":" not in item:
            raise ValueError(f"Invalid --label-map item '{item}'. Expected raw:new")
        raw, new = item.split(":", 1)
        out[raw.strip()] = new.strip()
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(str(key))
                    seen.add(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sample_code_from_slide_key(slide_key: str) -> str:
    parts = str(slide_key).split("-")
    if len(parts) >= 4 and len(parts[3]) >= 2:
        return parts[3][:2]
    return ""


def slide_key_from_manifest_path(path_value: str) -> str:
    return Path(str(path_value)).name.split(".")[0]


def case_id_from_slide_key(slide_key: str) -> str:
    parts = str(slide_key).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(slide_key)


def load_split_cases(path: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(path.read_text())
    train_cases = {case_id_from_slide_key(slide_key_from_manifest_path(p)) for p in payload.get("train", [])}
    test_cases = {case_id_from_slide_key(slide_key_from_manifest_path(p)) for p in payload.get("test", [])}
    overlap = train_cases & test_cases
    if overlap:
        raise ValueError(f"Split manifest has {len(overlap)} overlapping case IDs; example={sorted(overlap)[:5]}")
    return train_cases, test_cases


def project_has_features(features_root: Path, project: str) -> bool:
    return (features_root / project / "features_uni2").is_dir()


def validate_feature_h5(path: Path, *, embed_dim: int) -> str:
    try:
        with h5py.File(path, "r") as handle:
            if "features" not in handle:
                return "missing_features_dataset"
            feats = handle["features"]
            if feats.ndim == 2:
                shape = tuple(feats.shape)
            elif feats.ndim == 3 and feats.shape[0] == 1:
                shape = tuple(feats.shape[1:])
            else:
                return f"unsupported_features_shape:{tuple(feats.shape)}"
            if len(shape) != 2 or int(shape[1]) != int(embed_dim):
                return f"unexpected_feature_dim:{shape}"
            if int(shape[0]) <= 0:
                return "empty_features"
    except Exception as exc:
        return f"read_error:{type(exc).__name__}:{exc}"
    return ""


def load_label_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    projects_arg = parse_csv_list(args.projects)
    all_projects = len(projects_arg) == 1 and projects_arg[0].lower() == "all"
    include_labels = set(parse_csv_list(args.include_labels))
    label_map = parse_label_map(args.label_map)
    train_cases, test_cases = load_split_cases(args.split_manifest)

    rows: list[dict[str, Any]] = []
    skipped_features: list[dict[str, Any]] = []
    delimiter = "\t" if args.label_source.suffix.lower() in {".tsv", ".tab"} else ","
    with args.label_source.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if args.label_column not in (reader.fieldnames or []):
            raise ValueError(f"Label column '{args.label_column}' not found in {args.label_source}")
        for row in reader:
            project = str(row.get("project_dir", ""))
            if not all_projects and project not in set(projects_arg):
                continue
            slide_key = str(row.get("slide_key", ""))
            case_id = str(row.get("case_id", "")) or case_id_from_slide_key(slide_key)
            split = "train" if case_id in train_cases else "test" if case_id in test_cases else ""
            if not split:
                continue
            raw_label = project if str(args.label_column) == "project_dir" else str(row.get(args.label_column, ""))
            label_name = label_map.get(raw_label, raw_label)
            if label_name.strip().upper() in UNKNOWN_VALUES:
                continue
            if include_labels and label_name not in include_labels:
                continue
            fallback_h5 = args.features_root / project / "features_uni2" / f"{slide_key}.h5"
            explicit_h5 = str(row.get("h5_path", "")).strip()
            h5_path = Path(explicit_h5) if explicit_h5 else fallback_h5
            if not h5_path.exists() and fallback_h5.exists():
                h5_path = fallback_h5
            if all_projects and not explicit_h5 and not project_has_features(args.features_root, project):
                continue
            if not h5_path.exists():
                skipped_features.append(
                    {
                        "case_id": case_id,
                        "slide_key": slide_key,
                        "project_dir": project,
                        "label_name": label_name,
                        "split": split,
                        "h5_path": str(h5_path),
                        "reason": "missing_h5",
                    }
                )
                continue
            if bool(args.validate_h5):
                reason = validate_feature_h5(h5_path, embed_dim=int(args.embed_dim))
                if reason:
                    skipped_features.append(
                        {
                            "case_id": case_id,
                            "slide_key": slide_key,
                            "project_dir": project,
                            "label_name": label_name,
                            "split": split,
                            "h5_path": str(h5_path),
                            "reason": reason,
                        }
                    )
                    continue
            rows.append(
                {
                    "case_id": case_id,
                    "slide_key": slide_key,
                    "sample_id": str(row.get("sample_id", "")),
                    "sample_code": sample_code_from_slide_key(slide_key),
                    "project_dir": project,
                    "label_name": label_name,
                    "raw_label": raw_label,
                    "split": split,
                    "h5_path": str(h5_path),
                }
            )
    return rows, skipped_features


def filter_and_encode_labels(
    rows: list[dict[str, Any]],
    *,
    min_slides_per_class: int,
    max_slides_per_class: int,
    max_train_slides: int,
    max_test_slides: int,
    seed: int,
    label_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    class_counts = Counter(str(row["label_name"]) for row in rows)
    keep_labels = {label for label, count in class_counts.items() if int(count) >= int(min_slides_per_class)}
    rows = [row for row in rows if str(row["label_name"]) in keep_labels]
    discovered_labels = {str(row["label_name"]) for row in rows}
    if label_order:
        ordered = [str(label) for label in label_order]
        missing = sorted(discovered_labels - set(ordered))
        unknown = sorted(set(ordered) - discovered_labels)
        if missing or unknown:
            raise ValueError(f"--label-order mismatch: missing={missing}, unknown={unknown}")
        label_names = ordered
    else:
        label_names = sorted(discovered_labels)
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    for row in rows:
        row["label_id"] = int(label_to_id[str(row["label_name"])])

    rng = random.Random(int(seed))
    final_rows: list[dict[str, Any]] = []
    def _balanced_total_limit(class_rows: dict[int, list[dict[str, Any]]], total_limit: int) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        label_ids = sorted(class_rows)
        cursor = 0
        while len(ordered) < int(total_limit):
            added = False
            for label_id in label_ids:
                rows_i = class_rows[label_id]
                if cursor < len(rows_i):
                    ordered.append(rows_i[cursor])
                    added = True
                    if len(ordered) >= int(total_limit):
                        break
            if not added:
                break
            cursor += 1
        return ordered

    for split in ("train", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_rows.sort(key=lambda r: (int(r["label_id"]), str(r["case_id"]), str(r["slide_key"])))
        by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in split_rows:
            by_class[int(row["label_id"])].append(row)
        for label_id, class_rows in sorted(by_class.items()):
            rng.shuffle(class_rows)
            if int(max_slides_per_class) > 0:
                by_class[label_id] = class_rows[: int(max_slides_per_class)]
        limit = int(max_train_slides) if split == "train" else int(max_test_slides)
        if limit > 0:
            capped = _balanced_total_limit(by_class, limit)
        else:
            capped = [row for label_id in sorted(by_class) for row in by_class[label_id]]
        capped.sort(key=lambda r: (str(r["case_id"]), str(r["slide_key"])))
        final_rows.extend(capped)
    final_rows.sort(key=lambda r: (str(r["split"]), int(r["label_id"]), str(r["case_id"]), str(r["slide_key"])))
    return final_rows, label_to_id


def read_features(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{path}: missing dataset 'features'")
        feats = handle["features"]
        if feats.ndim == 2:
            arr = feats[:]
        elif feats.ndim == 3 and feats.shape[0] == 1:
            arr = feats[0]
        else:
            raise ValueError(f"{path}: unsupported features shape {tuple(feats.shape)}")
    return np.asarray(arr, dtype=np.float32)


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def prepare_bag(row: dict[str, Any], *, max_tiles: int, seed: int, epoch: int, train: bool) -> torch.Tensor:
    feats = read_features(Path(str(row["h5_path"])))
    if feats.ndim != 2:
        raise ValueError(f"{row['h5_path']}: expected [N,D], got {feats.shape}")
    if int(max_tiles) > 0 and feats.shape[0] > int(max_tiles):
        rng = np.random.default_rng(stable_seed(seed, epoch if train else "eval", row["slide_key"]))
        idx = rng.choice(feats.shape[0], size=int(max_tiles), replace=False)
        idx.sort()
        feats = feats[idx]
    return torch.as_tensor(feats, dtype=torch.float32)


def build_model(args: argparse.Namespace, n_classes: int) -> torch.nn.Module:
    common = {
        "embed_dim": int(args.embed_dim),
        "hidden_dim": int(args.hidden_dim),
        "attn_dim": int(args.attn_dim),
        "n_classes": int(n_classes),
        "dropout": float(args.dropout),
    }
    if str(args.model) == "gated":
        return GatedAttentionMIL(
            **common,
            learnable_temperature=not bool(args.fixed_temperature),
            init_temperature=float(args.init_temperature),
        )
    return AttentionMIL(**common)


def binary_auroc(y_true: list[int], y_score: list[float]) -> float | None:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = sum(1 for y in y_true if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(1 for _, y in pairs[i:j] if y == 1)
        i = j
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(y_true: list[int], y_pred: list[int], probs: list[list[float]], n_classes: int) -> dict[str, Any]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    per_class: list[dict[str, float]] = []
    recalls = []
    f1s = []
    for c in range(n_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        per_class.append({"class_id": c, "precision": precision, "recall": recall, "f1": f1, "support": float(cm[c, :].sum())})
        if cm[c, :].sum() > 0:
            recalls.append(recall)
            f1s.append(f1)
    accuracy = float(np.trace(cm) / max(cm.sum(), 1))
    out: dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    if n_classes == 2 and probs:
        auroc = binary_auroc(y_true, [float(p[1]) for p in probs])
        out["auroc"] = auroc
    else:
        out["auroc"] = None
    return out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    n_classes: int,
    max_tiles_per_slide: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probs_all: list[list[float]] = []
    pred_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            x = prepare_bag(row, max_tiles=max_tiles_per_slide, seed=seed, epoch=0, train=False).to(device)
        except Exception as exc:
            print(f"[warn] skipping eval bag {row.get('slide_key', '')}: {exc}", file=sys.stderr)
            continue
        logits, y_prob, y_hat, _, _ = model(x)
        probs = y_prob.detach().cpu().numpy().reshape(-1).astype(float).tolist()
        pred = int(y_hat.detach().cpu().reshape(-1)[0].item())
        true = int(row["label_id"])
        y_true.append(true)
        y_pred.append(pred)
        probs_all.append(probs)
        pred_item = dict(row)
        pred_item["pred_label_id"] = pred
        pred_item["correct"] = int(pred == true)
        for i, p in enumerate(probs):
            pred_item[f"prob_{i}"] = float(p)
        pred_rows.append(pred_item)
    if not y_true:
        raise RuntimeError("No readable evaluation bags remained after H5 read validation/skipping.")
    return compute_metrics(y_true, y_pred, probs_all, n_classes), pred_rows


def make_loss_weights(rows: list[dict[str, Any]], n_classes: int, device: torch.device) -> torch.Tensor | None:
    counts = Counter(int(row["label_id"]) for row in rows)
    if len(counts) != n_classes:
        return None
    total = float(sum(counts.values()))
    weights = [total / max(float(n_classes * counts[c]), 1.0) for c in range(n_classes)]
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args_payload: dict[str, Any],
    label_to_id: dict[str, int],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": args_payload,
            "label_to_id": label_to_id,
            "metrics": metrics,
            "split_name": "sae_patient_train_test_90_10",
        },
        path,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = args.out_dir / str(args.task_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, skipped_feature_rows = load_label_rows(args)
    rows, label_to_id = filter_and_encode_labels(
        raw_rows,
        min_slides_per_class=int(args.min_slides_per_class),
        max_slides_per_class=int(args.max_slides_per_class),
        max_train_slides=int(args.max_train_slides),
        max_test_slides=int(args.max_test_slides),
        seed=int(args.seed),
        label_order=parse_csv_list(args.label_order),
    )
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    train_cases = {row["case_id"] for row in train_rows}
    test_cases = {row["case_id"] for row in test_rows}
    overlap = train_cases & test_cases
    if overlap:
        raise RuntimeError(f"Train/test case leakage detected: {sorted(overlap)[:5]}")
    if len(label_to_id) < 2:
        raise RuntimeError(f"Need at least two classes after filtering; got {label_to_id}")
    if not train_rows or not test_rows:
        raise RuntimeError(f"Need non-empty train and test rows; got train={len(train_rows)}, test={len(test_rows)}")

    label_mapping = {
        "label_to_id": label_to_id,
        "id_to_label": {str(v): k for k, v in label_to_id.items()},
    }
    manifest_fields = ["case_id", "slide_key", "sample_id", "sample_code", "project_dir", "label_name", "label_id", "raw_label", "split", "h5_path"]
    write_csv(out_dir / "task_manifest.csv", rows, manifest_fields)
    skipped_feature_fields = ["case_id", "slide_key", "project_dir", "label_name", "split", "h5_path", "reason"]
    write_csv(out_dir / "skipped_feature_files.csv", skipped_feature_rows, skipped_feature_fields)
    write_json(out_dir / "label_mapping.json", label_mapping)
    args_payload = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args_payload.update({"n_classes": int(len(label_to_id)), "label_to_id": label_to_id})
    write_json(out_dir / "args.json", args_payload)

    data_summary = {
        "task_name": str(args.task_name),
        "n_classes": int(len(label_to_id)),
        "label_to_id": label_to_id,
        "train_slides": int(len(train_rows)),
        "test_slides": int(len(test_rows)),
        "train_cases": int(len(train_cases)),
        "test_cases": int(len(test_cases)),
        "train_counts": dict(Counter(row["label_name"] for row in train_rows)),
        "test_counts": dict(Counter(row["label_name"] for row in test_rows)),
        "dropped_rows_after_filtering": int(len(raw_rows) - len(rows)),
        "skipped_feature_files": int(len(skipped_feature_rows)),
        "skipped_feature_files_csv": str(out_dir / "skipped_feature_files.csv"),
        "case_overlap": int(len(overlap)),
    }
    if bool(args.dry_run):
        write_json(out_dir / "summary.json", {"args": args_payload, "data": data_summary, "dry_run": True})
        print(json.dumps(data_summary, indent=2))
        return

    model = build_model(args, n_classes=len(label_to_id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_weight = make_loss_weights(train_rows, len(label_to_id), device) if bool(args.class_balanced_loss) else None

    train_metrics_rows: list[dict[str, Any]] = []
    best_bal_acc = -math.inf
    best_metrics: dict[str, Any] = {}
    best_epoch = -1
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_rows = list(train_rows)
        random.Random(int(args.seed) + epoch).shuffle(epoch_rows)
        losses: list[float] = []
        correct = 0
        seen_train_bags = 0
        for row in epoch_rows:
            try:
                x = prepare_bag(row, max_tiles=int(args.max_tiles_per_slide), seed=int(args.seed), epoch=epoch, train=True).to(device)
            except Exception as exc:
                print(f"[warn] skipping train bag {row.get('slide_key', '')}: {exc}", file=sys.stderr)
                continue
            y = torch.as_tensor([int(row["label_id"])], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, y_hat, _, _ = model(x)
            loss = F.cross_entropy(logits, y, weight=loss_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            correct += int(int(y_hat.detach().cpu().reshape(-1)[0].item()) == int(row["label_id"]))
            seen_train_bags += 1
        if seen_train_bags == 0:
            raise RuntimeError("No readable training bags remained after H5 read validation/skipping.")
        test_metrics, _ = evaluate(
            model,
            test_rows,
            device=device,
            n_classes=len(label_to_id),
            max_tiles_per_slide=int(args.max_tiles_per_slide),
            seed=int(args.seed),
        )
        train_acc = float(correct / max(seen_train_bags, 1))
        metric_row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_accuracy": train_acc,
            "train_bags_seen": int(seen_train_bags),
            "train_bags_skipped": int(len(epoch_rows) - seen_train_bags),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "test_auroc": "" if test_metrics["auroc"] is None else float(test_metrics["auroc"]),
        }
        train_metrics_rows.append(metric_row)
        if float(test_metrics["balanced_accuracy"]) > best_bal_acc:
            best_bal_acc = float(test_metrics["balanced_accuracy"])
            best_metrics = test_metrics
            best_epoch = int(epoch)
            save_checkpoint(
                out_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args_payload=args_payload,
                label_to_id=label_to_id,
                metrics=test_metrics,
            )
        print(json.dumps(metric_row))

    final_metrics, test_predictions = evaluate(
        model,
        test_rows,
        device=device,
        n_classes=len(label_to_id),
        max_tiles_per_slide=int(args.max_tiles_per_slide),
        seed=int(args.seed),
    )
    save_checkpoint(
        out_dir / "final_model.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(args.epochs),
        args_payload=args_payload,
        label_to_id=label_to_id,
        metrics=final_metrics,
    )
    write_csv(out_dir / "train_metrics.csv", train_metrics_rows)
    pred_fields = manifest_fields + ["pred_label_id", "correct"] + [f"prob_{i}" for i in range(len(label_to_id))]
    write_csv(out_dir / "test_predictions.csv", test_predictions, pred_fields)
    write_json(
        out_dir / "summary.json",
        {
            "args": args_payload,
            "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:]))),
            "data": data_summary,
            "best_epoch": int(best_epoch),
            "best_test_metrics": best_metrics,
            "final_test_metrics": final_metrics,
            "outputs": {
                "final_model": str(out_dir / "final_model.pt"),
                "best_model": str(out_dir / "best_model.pt"),
                "task_manifest": str(out_dir / "task_manifest.csv"),
                "train_metrics": str(out_dir / "train_metrics.csv"),
                "test_predictions": str(out_dir / "test_predictions.csv"),
                "label_mapping": str(out_dir / "label_mapping.json"),
            },
        },
    )


if __name__ == "__main__":
    main()
