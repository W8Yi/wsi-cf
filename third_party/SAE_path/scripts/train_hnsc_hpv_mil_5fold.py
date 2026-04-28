#!/usr/bin/env python3
"""
Train attention-based MIL classifiers on the HNSC HPV 90/10 split set.

This consumes the repeated 90/10 patient-level split folder:
  metadata/manifests/hnsc_hpv_5fold/

For each outer split:
- trains one MIL model on outer `train`
- evaluates once on outer `test`
- saves checkpoint, metrics, predictions, and history

By default this uses models.classifier.AttentionMIL.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.classifier import AttentionMIL, GatedAttentionMIL


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_h5_features(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{h5_path}: missing dataset 'features'")
        feats = handle["features"]
        if feats.ndim == 3 and feats.shape[0] == 1:
            arr = feats[0]
        elif feats.ndim == 2:
            arr = feats[:]
        else:
            raise ValueError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")
    return np.asarray(arr, dtype=np.float32)


def normalize_feats(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "l2":
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / denom
    if mode == "layernorm":
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-8
        return (x - mu) / sd
    raise ValueError(f"Unknown normalize mode: {mode}")


def maybe_subsample_tiles(
    x: np.ndarray,
    max_tiles: int,
    seed: int,
) -> np.ndarray:
    if max_tiles <= 0 or x.shape[0] <= max_tiles:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_tiles, replace=False)
    idx.sort()
    return x[idx]


def load_bag_tensor(
    sample: dict,
    *,
    max_tiles: int,
    normalize: str,
    bag_seed: int,
    device: torch.device,
) -> torch.Tensor:
    x = read_h5_features(sample["h5_path"])
    x = maybe_subsample_tiles(x, max_tiles=max_tiles, seed=bag_seed)
    x = normalize_feats(x, normalize)
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)


def load_split_rows(
    tsv_path: Path,
    *,
    h5_dir: Path | None = None,
) -> Dict[str, List[dict]]:
    rows = read_tsv(tsv_path)
    groups: Dict[str, List[dict]] = defaultdict(list)
    missing_h5: List[str] = []
    for row in rows:
        row["label"] = int(row["label"])
        if h5_dir is not None:
            resolved = h5_dir / f"{row['slide_key']}.h5"
            row["h5_path"] = str(resolved)
            if not resolved.exists():
                missing_h5.append(str(resolved))
        groups[row["split"]].append(row)
    if missing_h5:
        shown = "\n".join(missing_h5[:10])
        raise FileNotFoundError(
            f"{tsv_path}: missing {len(missing_h5)} H5 files under override dir {h5_dir}. "
            f"First missing paths:\n{shown}"
        )
    if "train" not in groups or "test" not in groups:
        raise ValueError(f"{tsv_path} must contain train and test rows")
    return groups


def resolve_split_tsvs(splits_dir: Path) -> List[Tuple[str, Path]]:
    index_path = splits_dir / "index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        out = []
        for item in payload.get("splits", []):
            out.append((item["split_name"], Path(item["tsv_path"])))
        if out:
            return out

    paths = sorted(splits_dir.glob("split_*.tsv"))
    return [(p.stem, p) for p in paths]


def build_model(args: argparse.Namespace) -> nn.Module:
    kwargs = {
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "attn_dim": args.attn_dim,
        "n_classes": 2,
        "dropout": args.dropout,
    }
    if args.model == "attention":
        return AttentionMIL(**kwargs)
    if args.model == "gated":
        return GatedAttentionMIL(
            **kwargs,
            learnable_temperature=(not args.fixed_temperature),
            init_temperature=args.init_temperature,
        )
    raise ValueError(f"Unknown model type: {args.model}")


def compute_class_weights(rows: List[dict], device: torch.device) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in rows)
    total = sum(counts.values())
    weights = []
    for cls in range(2):
        cls_count = max(1, counts.get(cls, 0))
        weights.append(total / (2.0 * cls_count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def safe_roc_auc(y_true: List[int], y_score: List[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def classification_metrics(y_true: List[int], y_pred: List[int], y_prob_pos: List[float]) -> Dict[str, object]:
    auc = safe_roc_auc(y_true, y_prob_pos)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def aggregate_case_metrics(pred_rows: List[dict]) -> Dict[str, object]:
    case_to_probs: Dict[str, List[float]] = defaultdict(list)
    case_to_label: Dict[str, int] = {}

    for row in pred_rows:
        case_id = row["case_id"]
        case_to_probs[case_id].append(float(row["prob_pos"]))
        label = int(row["label"])
        old = case_to_label.get(case_id)
        if old is not None and old != label:
            raise ValueError(f"Inconsistent labels for case {case_id}")
        case_to_label[case_id] = label

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    for case_id in sorted(case_to_probs):
        prob = float(np.mean(case_to_probs[case_id]))
        label = case_to_label[case_id]
        pred = 1 if prob >= 0.5 else 0
        y_true.append(label)
        y_pred.append(pred)
        y_prob.append(prob)

    return classification_metrics(y_true, y_pred, y_prob)


def run_epoch(
    *,
    model: nn.Module,
    rows: List[dict],
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    train: bool,
    epoch: int,
    max_tiles: int,
    normalize: str,
    seed: int,
) -> Tuple[Dict[str, object], List[dict]]:
    if train:
        model.train()
        ordered_rows = list(rows)
        rng = random.Random(seed + epoch)
        rng.shuffle(ordered_rows)
    else:
        model.eval()
        ordered_rows = list(rows)

    loss_sum = 0.0
    pred_rows: List[dict] = []

    for idx, row in enumerate(ordered_rows):
        bag_seed = seed + epoch * 1000003 + idx
        x = load_bag_tensor(
            row,
            max_tiles=max_tiles,
            normalize=normalize,
            bag_seed=bag_seed,
            device=device,
        )
        target = torch.tensor([int(row["label"])], dtype=torch.long, device=device)

        with torch.set_grad_enabled(train):
            logits, y_prob, y_hat, _, _ = model(x)
            loss = criterion(logits, target)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        loss_sum += float(loss.detach().cpu().item())
        prob_pos = float(y_prob.detach().cpu()[0, 1].item())
        pred_label = int(y_hat.detach().cpu()[0, 0].item())
        pred_rows.append(
            {
                "case_id": row["case_id"],
                "slide_key": row["slide_key"],
                "label": int(row["label"]),
                "pred": pred_label,
                "prob_pos": prob_pos,
                "h5_path": row["h5_path"],
            }
        )

    y_true = [int(row["label"]) for row in pred_rows]
    y_pred = [int(row["pred"]) for row in pred_rows]
    y_prob_pos = [float(row["prob_pos"]) for row in pred_rows]

    metrics = {
        "loss": loss_sum / max(1, len(pred_rows)),
        "slide": classification_metrics(y_true, y_pred, y_prob_pos),
        "case": aggregate_case_metrics(pred_rows),
    }
    return metrics, pred_rows


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def json_safe_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def save_predictions(path: Path, rows: List[dict]) -> None:
    fieldnames = ["case_id", "slide_key", "label", "pred", "prob_pos", "h5_path"]
    write_csv(path, fieldnames, rows)


def summarize_metric_dicts(values: List[Dict[str, object]]) -> Dict[str, object]:
    numeric_keys = ["accuracy", "balanced_accuracy", "f1", "precision", "recall"]
    summary: Dict[str, object] = {}
    for key in numeric_keys:
        items = [float(v[key]) for v in values]
        summary[key] = {"mean": mean(items), "std": (pstdev(items) if len(items) > 1 else 0.0)}

    auc_items = [v["roc_auc"] for v in values if v["roc_auc"] is not None]
    summary["roc_auc"] = {
        "mean": (mean([float(v) for v in auc_items]) if auc_items else None),
        "std": (pstdev([float(v) for v in auc_items]) if len(auc_items) > 1 else 0.0 if auc_items else None),
        "n_defined": len(auc_items),
    }
    return summary


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    split_name: str,
) -> None:
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "args": json_safe_args(args),
            "split_name": split_name,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("metadata/manifests/hnsc_hpv_5fold"),
        help="Directory containing split_*.tsv files and optional index.json.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold"),
        help="Output directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--h5_dir",
        type=Path,
        default=None,
        help="Optional feature override directory. If set, each slide uses h5_dir/<slide_key>.h5.",
    )
    parser.add_argument(
        "--model",
        choices=["attention", "gated"],
        default="attention",
        help="MIL model variant from models/classifier.py",
    )
    parser.add_argument("--embed_dim", type=int, default=1536)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--attn_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--fixed_temperature", action="store_true", help="For gated model only.")
    parser.add_argument("--init_temperature", type=float, default=1.0, help="For gated model only.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_tiles_train", type=int, default=2048, help="0 means use all tiles.")
    parser.add_argument("--max_tiles_eval", type=int, default=4096, help="0 means use all tiles.")
    parser.add_argument(
        "--normalize",
        choices=["none", "l2", "layernorm"],
        default="none",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string, e.g. cuda:0, cpu, or auto.",
    )
    parser.add_argument(
        "--limit_splits",
        type=int,
        default=0,
        help="For debugging: limit the number of outer splits to run (0 = all).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    split_items = resolve_split_tsvs(args.splits_dir)
    if not split_items:
        raise SystemExit(f"No split TSVs found under {args.splits_dir}")
    if args.limit_splits > 0:
        split_items = split_items[: args.limit_splits]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.out_dir / "config.json", json_safe_args(args))

    split_summaries: List[dict] = []

    for split_idx, (split_name, split_tsv) in enumerate(split_items):
        print(f"[split] {split_name} <- {split_tsv}", flush=True)
        split_rows = load_split_rows(split_tsv, h5_dir=args.h5_dir)
        train_rows = list(split_rows["train"])
        test_rows = list(split_rows["test"])

        split_out = args.out_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        save_json(
            split_out / "split_counts.json",
            {
                "train_slides": len(train_rows),
                "test_slides": len(test_rows),
                "train_cases": len({row["case_id"] for row in train_rows}),
                "test_cases": len({row["case_id"] for row in test_rows}),
                "train_slide_labels": dict(Counter(int(row["label"]) for row in train_rows)),
                "test_slide_labels": dict(Counter(int(row["label"]) for row in test_rows)),
            },
        )

        model = build_model(args).to(device)
        class_weights = compute_class_weights(train_rows, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        history: List[dict] = []

        for epoch in range(1, args.epochs + 1):
            train_metrics, _ = run_epoch(
                model=model,
                rows=train_rows,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                train=True,
                epoch=epoch,
                max_tiles=args.max_tiles_train,
                normalize=args.normalize,
                seed=args.seed + split_idx * 10000,
            )
            history.append({"epoch": epoch, "train": train_metrics})
            print(
                f"  epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f}",
                flush=True,
            )

        final_epoch = args.epochs
        save_checkpoint(
            split_out / "best.pt",
            epoch=final_epoch,
            model=model,
            optimizer=optimizer,
            args=args,
            split_name=split_name,
        )
        save_checkpoint(
            split_out / "final.pt",
            epoch=final_epoch,
            model=model,
            optimizer=optimizer,
            args=args,
            split_name=split_name,
        )
        test_metrics, test_preds = run_epoch(
            model=model,
            rows=test_rows,
            device=device,
            criterion=criterion,
            optimizer=None,
            train=False,
            epoch=final_epoch,
            max_tiles=args.max_tiles_eval,
            normalize=args.normalize,
            seed=args.seed + split_idx * 10000 + 1777,
        )

        save_predictions(split_out / "test_predictions.csv", test_preds)
        save_json(split_out / "history.json", history)

        split_summary = {
            "split_name": split_name,
            "final_epoch": final_epoch,
            "test": test_metrics,
        }
        save_json(split_out / "metrics.json", split_summary)
        split_summaries.append(split_summary)

    summary = {
        "num_splits": len(split_summaries),
        "splits": split_summaries,
        "aggregate": {
            "test_slide": summarize_metric_dicts([item["test"]["slide"] for item in split_summaries]),
            "test_case": summarize_metric_dicts([item["test"]["case"] for item in split_summaries]),
        },
    }
    save_json(args.out_dir / "summary.json", summary)
    print(f"[ok] wrote {args.out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
