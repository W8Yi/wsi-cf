#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
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

from wsi_cf.common.io import write_json
from wsi_cf.common.paths import DEFAULT_SAE_CFG, DEFAULT_SAE_CKPT, DEFAULT_SAE_VARIANT, SAE_VARIANTS, resolve_sae_paths
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SAE concept-label association artifacts from a trained classifier task_manifest.csv. "
            "This makes classifier-trained tasks directly usable by find_label_concepts.py."
        )
    )
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--classifier-run-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=WSI_CF_ROOT / "artifacts/concept_label_associations_classifier")
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--label-column", type=str, default="label_name")
    parser.add_argument("--include-labels", type=str, default="")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "test"])
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--max-slides-per-class", type=int, default=0)
    parser.add_argument("--max-tiles-per-slide", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--sae-ckpt", type=Path, default=None, help=f"Explicit SAE checkpoint override. Defaults to --sae-variant ({DEFAULT_SAE_CKPT}).")
    parser.add_argument("--sae-cfg", type=Path, default=None, help=f"Explicit SAE config override. Defaults to --sae-variant ({DEFAULT_SAE_CFG}).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def parse_csv_list(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def load_manifest_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    manifest = args.manifest_csv or args.classifier_run_dir / "task_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing task manifest: {manifest}")
    include_labels = set(parse_csv_list(args.include_labels))
    rows: list[dict[str, str]] = []
    by_label_seen: Counter[str] = Counter()
    for row in read_csv_rows(manifest):
        if str(args.split) != "all" and str(row.get("split", "")) != str(args.split):
            continue
        label = str(row.get(args.label_column, row.get("label_name", row.get("label", ""))))
        if include_labels and label not in include_labels:
            continue
        if int(args.max_slides_per_class) > 0 and by_label_seen[label] >= int(args.max_slides_per_class):
            continue
        h5_path = Path(str(row.get("h5_path", "")))
        if not h5_path.exists():
            continue
        item = {
            "case_id": str(row.get("case_id", "")),
            "slide_key": str(row.get("slide_key", "")),
            "project_dir": str(row.get("project_dir", "")),
            "label": label,
            "h5_path": str(h5_path),
            "split": str(row.get("split", "")),
        }
        rows.append(item)
        by_label_seen[label] += 1
        if int(args.max_slides) > 0 and len(rows) >= int(args.max_slides):
            break
    rows.sort(key=lambda r: (str(r["label"]), str(r["project_dir"]), str(r["case_id"]), str(r["slide_key"])))
    return rows


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


def maybe_subsample_tiles(features: np.ndarray, *, max_tiles: int, seed: int, slide_key: str) -> np.ndarray:
    if int(max_tiles) <= 0 or features.shape[0] <= int(max_tiles):
        return features
    digest = hashlib.md5(f"{int(seed)}::{slide_key}".encode("utf-8")).hexdigest()
    local_seed = int(digest[:8], 16)
    rng = np.random.default_rng(local_seed)
    idx = rng.choice(features.shape[0], size=int(max_tiles), replace=False)
    idx.sort()
    return features[idx]


@torch.no_grad()
def summarize_slide(
    *,
    features: np.ndarray,
    sae_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    d_latent: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tiles = int(features.shape[0])
    sum_z = np.zeros((d_latent,), dtype=np.float64)
    sum_active = np.zeros((d_latent,), dtype=np.float64)
    max_z = np.full((d_latent,), -np.inf, dtype=np.float32)
    for start in range(0, n_tiles, int(batch_size)):
        end = min(n_tiles, start + int(batch_size))
        x = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
        z = sae_encode_features(sae_model, x).detach().cpu().numpy().astype(np.float32, copy=False)
        sum_z += z.sum(axis=0, dtype=np.float64)
        sum_active += (z > 0).sum(axis=0, dtype=np.float64)
        max_z = np.maximum(max_z, z.max(axis=0))
    mean_activation = (sum_z / max(n_tiles, 1)).astype(np.float32)
    fraction_active = (sum_active / max(n_tiles, 1)).astype(np.float32)
    max_z[~np.isfinite(max_z)] = 0.0
    return mean_activation, fraction_active, max_z.astype(np.float32)


def cohen_d(class_values: np.ndarray, rest_values: np.ndarray) -> np.ndarray:
    n1 = int(class_values.shape[0])
    n2 = int(rest_values.shape[0])
    mean1 = class_values.mean(axis=0)
    mean2 = rest_values.mean(axis=0)
    var1 = class_values.var(axis=0, ddof=1) if n1 > 1 else np.zeros_like(mean1)
    var2 = rest_values.var(axis=0, ddof=1) if n2 > 1 else np.zeros_like(mean2)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    return ((mean1 - mean2) / np.maximum(pooled, 1e-8)).astype(np.float32)


def build_association_rows(
    *,
    metric_name: str,
    values: np.ndarray,
    labels: list[str],
    latent_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels_arr = np.asarray(labels)
    for class_label in sorted(set(labels)):
        mask = labels_arr == class_label
        rest = ~mask
        if int(mask.sum()) == 0 or int(rest.sum()) == 0:
            continue
        class_values = values[mask]
        rest_values = values[rest]
        mean_class = class_values.mean(axis=0)
        mean_rest = rest_values.mean(axis=0)
        diff = mean_class - mean_rest
        d = cohen_d(class_values, rest_values)
        for latent_idx, mc, mr, delta, dz in zip(latent_ids.tolist(), mean_class.tolist(), mean_rest.tolist(), diff.tolist(), d.tolist()):
            rows.append(
                {
                    "latent_idx": int(latent_idx),
                    "metric": metric_name,
                    "class_label": class_label,
                    "n_class": int(mask.sum()),
                    "n_rest": int(rest.sum()),
                    "mean_class": float(mc),
                    "mean_rest": float(mr),
                    "diff_class_minus_rest": float(delta),
                    "abs_diff": float(abs(delta)),
                    "cohen_d": float(dz),
                }
            )
    rows.sort(
        key=lambda r: (
            str(r["metric"]),
            str(r["class_label"]),
            -float(r["cohen_d"]),
            -float(r["abs_diff"]),
            int(r["latent_idx"]),
        )
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.sae_ckpt, args.sae_cfg = resolve_sae_paths(args.sae_variant, args.sae_ckpt, args.sae_cfg)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = args.out_root / str(args.task_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / "latent_label_associations.csv", out_dir / "cohort_slides.csv", out_dir / "task_summary.json"]
    if bool(args.skip_existing) and all(path.exists() for path in expected):
        print(f"[skip] association artifacts already exist in {out_dir}")
        return

    slides = load_manifest_rows(args)
    if len({row["label"] for row in slides}) < 2:
        raise RuntimeError(f"Need at least two labels for association; got {Counter(row['label'] for row in slides)}")

    sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    sae_model.eval()
    mean_rows: list[np.ndarray] = []
    fraction_rows: list[np.ndarray] = []
    max_rows: list[np.ndarray] = []
    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, slide in enumerate(slides, start=1):
        h5_path = Path(slide["h5_path"])
        try:
            features = read_features(h5_path)
            if int(features.shape[1]) != int(d_in):
                raise ValueError(f"feature_dim_{features.shape[1]}_expected_{d_in}")
            features = maybe_subsample_tiles(
                features,
                max_tiles=int(args.max_tiles_per_slide),
                seed=int(args.seed),
                slide_key=str(slide["slide_key"]),
            )
            mean_z, frac_z, max_z = summarize_slide(
                features=features,
                sae_model=sae_model,
                device=device,
                batch_size=int(args.batch_size),
                d_latent=int(d_latent),
            )
        except Exception as exc:
            skipped.append({**slide, "reason": str(exc)})
            continue
        mean_rows.append(mean_z)
        fraction_rows.append(frac_z)
        max_rows.append(max_z)
        processed.append({**slide, "n_tiles_used": int(features.shape[0])})
        if idx % 100 == 0:
            print(f"[progress] {idx}/{len(slides)} slides scanned for {args.task_name}", file=sys.stderr)

    if len(processed) < 2 or len({row["label"] for row in processed}) < 2:
        raise RuntimeError(f"Need at least two processed labels; processed={Counter(row['label'] for row in processed)}, skipped={len(skipped)}")

    labels = [row["label"] for row in processed]
    latent_ids = np.arange(int(d_latent), dtype=np.int64)
    mean_arr = np.stack(mean_rows, axis=0).astype(np.float32)
    frac_arr = np.stack(fraction_rows, axis=0).astype(np.float32)
    max_arr = np.stack(max_rows, axis=0).astype(np.float32)

    assoc_rows: list[dict[str, Any]] = []
    assoc_rows.extend(build_association_rows(metric_name="mean_activation", values=mean_arr, labels=labels, latent_ids=latent_ids))
    assoc_rows.extend(build_association_rows(metric_name="fraction", values=frac_arr, labels=labels, latent_ids=latent_ids))
    assoc_rows.extend(build_association_rows(metric_name="prevalence", values=frac_arr, labels=labels, latent_ids=latent_ids))
    assoc_rows.extend(build_association_rows(metric_name="max_activation", values=max_arr, labels=labels, latent_ids=latent_ids))

    cohort_fields = ["case_id", "slide_key", "project_dir", "label", "h5_path"]
    write_csv(out_dir / "cohort_slides.csv", processed, cohort_fields)
    write_csv(out_dir / "skipped_slides.csv", skipped, cohort_fields + ["split", "reason"])
    write_csv(
        out_dir / "latent_label_associations.csv",
        assoc_rows,
        [
            "latent_idx",
            "metric",
            "class_label",
            "n_class",
            "n_rest",
            "mean_class",
            "mean_rest",
            "diff_class_minus_rest",
            "abs_diff",
            "cohen_d",
        ],
    )
    np.savez_compressed(
        out_dir / "slide_sae_summary.npz",
        latent_ids=latent_ids,
        slide_keys=np.asarray([row["slide_key"] for row in processed]),
        labels=np.asarray(labels),
        mean_activation=mean_arr,
        fraction=frac_arr,
        max_activation=max_arr,
    )
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:]))),
        "task_name": str(args.task_name),
        "n_input_slides": int(len(slides)),
        "n_processed_slides": int(len(processed)),
        "n_skipped_slides": int(len(skipped)),
        "label_counts": dict(Counter(labels)),
        "sae_d_in": int(d_in),
        "sae_d_latent": int(d_latent),
        "outputs": {
            "latent_label_associations": str(out_dir / "latent_label_associations.csv"),
            "cohort_slides": str(out_dir / "cohort_slides.csv"),
            "skipped_slides": str(out_dir / "skipped_slides.csv"),
            "slide_sae_summary_npz": str(out_dir / "slide_sae_summary.npz"),
            "task_summary": str(out_dir / "task_summary.json"),
        },
    }
    write_json(out_dir / "task_summary.json", summary)
    write_json(out_dir / "coverage_summary.json", summary)
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
