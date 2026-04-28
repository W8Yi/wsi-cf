#!/usr/bin/env python3
"""
Run SAE on exported high-attention tiles and summarize latent stats for HPV+ vs HPV-.

Input:
- top_attention_tiles.csv from export_hnsc_hpv_mil_attention.py
- SAE checkpoint + config

Output:
- latent_stats.csv
- summary.json

This is intentionally simple and fast:
- reads only the selected tile rows from each H5
- encodes in minibatches
- accumulates label-wise latent means and activity rates
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sae import load_sae_from_config, sae_encode_features


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_h5_features_indexed(h5_path: str, idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    order = np.argsort(idx)
    idx_sorted = idx[order]

    with h5py.File(h5_path, "r") as handle:
        feats = handle["features"]
        if feats.ndim == 3 and feats.shape[0] == 1:
            x_sorted = feats[0, idx_sorted, :]
        elif feats.ndim == 2:
            x_sorted = feats[idx_sorted, :]
        else:
            raise ValueError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    x = np.asarray(x_sorted, dtype=np.float32)[inv]
    return x


def build_batches(features: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    n = features.shape[0]
    for start in range(0, n, batch_size):
        yield features[start:start + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles_csv",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold/attention_exports/split_0/top_attention_tiles.csv"),
        help="top_attention_tiles.csv path",
    )
    parser.add_argument(
        "--sae_ckpt",
        type=Path,
        required=True,
        help="SAE checkpoint path",
    )
    parser.add_argument(
        "--sae_cfg",
        type=Path,
        required=True,
        help="SAE config JSON path",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold/sae_stats/split_0"),
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="SAE encoding batch size",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional debug cap on rows from the tiles CSV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda:0, cpu, or auto",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    rows = read_csv(args.tiles_csv)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit(f"No rows found in {args.tiles_csv}")

    sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=device)

    rows_by_h5: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        row["label"] = int(row["label"])
        row["tile_index"] = int(row["tile_index"])
        row["attention"] = float(row["attention"])
        rows_by_h5[row["h5_path"]].append(row)

    sum_z = {
        0: np.zeros(d_latent, dtype=np.float64),
        1: np.zeros(d_latent, dtype=np.float64),
    }
    sum_z_weighted = {
        0: np.zeros(d_latent, dtype=np.float64),
        1: np.zeros(d_latent, dtype=np.float64),
    }
    count_active = {
        0: np.zeros(d_latent, dtype=np.int64),
        1: np.zeros(d_latent, dtype=np.int64),
    }
    n_tiles = {0: 0, 1: 0}
    attn_weight_sum = {0: 0.0, 1: 0.0}

    total_rows = 0
    for group_idx, (h5_path, group_rows) in enumerate(rows_by_h5.items(), start=1):
        idx = np.asarray([row["tile_index"] for row in group_rows], dtype=np.int64)
        x = read_h5_features_indexed(h5_path, idx)
        if x.shape[1] != d_in:
            raise RuntimeError(f"{h5_path}: feature dim {x.shape[1]} != SAE d_in {d_in}")

        z_chunks = []
        for xb_np in build_batches(x, args.batch_size):
            xb = torch.from_numpy(xb_np).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                z = sae_encode_features(sae_model, xb)
            z_chunks.append(z.detach().cpu().numpy().astype(np.float32, copy=False))

        z_all = np.concatenate(z_chunks, axis=0)

        for row, z in zip(group_rows, z_all):
            label = int(row["label"])
            attn = float(row["attention"])
            sum_z[label] += z
            sum_z_weighted[label] += z * attn
            count_active[label] += (np.abs(z) > 0).astype(np.int64)
            n_tiles[label] += 1
            attn_weight_sum[label] += attn
            total_rows += 1

        if group_idx % 50 == 0:
            print(f"[progress] h5_groups={group_idx} tile_rows={total_rows}", flush=True)

    if n_tiles[0] == 0 or n_tiles[1] == 0:
        raise SystemExit(f"Need both HPV- and HPV+ rows. Got counts: {n_tiles}")

    mean_z_neg = sum_z[0] / max(1, n_tiles[0])
    mean_z_pos = sum_z[1] / max(1, n_tiles[1])
    mean_z_w_neg = sum_z_weighted[0] / max(1e-12, attn_weight_sum[0])
    mean_z_w_pos = sum_z_weighted[1] / max(1e-12, attn_weight_sum[1])
    active_rate_neg = count_active[0] / max(1, n_tiles[0])
    active_rate_pos = count_active[1] / max(1, n_tiles[1])

    latent_rows = []
    for j in range(d_latent):
        latent_rows.append(
            {
                "latent_idx": j,
                "mean_neg": float(mean_z_neg[j]),
                "mean_pos": float(mean_z_pos[j]),
                "mean_diff_pos_minus_neg": float(mean_z_pos[j] - mean_z_neg[j]),
                "weighted_mean_neg": float(mean_z_w_neg[j]),
                "weighted_mean_pos": float(mean_z_w_pos[j]),
                "weighted_mean_diff_pos_minus_neg": float(mean_z_w_pos[j] - mean_z_w_neg[j]),
                "active_rate_neg": float(active_rate_neg[j]),
                "active_rate_pos": float(active_rate_pos[j]),
                "active_rate_diff_pos_minus_neg": float(active_rate_pos[j] - active_rate_neg[j]),
                "abs_mean_diff": float(abs(mean_z_pos[j] - mean_z_neg[j])),
                "abs_weighted_mean_diff": float(abs(mean_z_w_pos[j] - mean_z_w_neg[j])),
            }
        )

    latent_rows.sort(key=lambda r: r["abs_weighted_mean_diff"], reverse=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "latent_stats.csv",
        [
            "latent_idx",
            "mean_neg",
            "mean_pos",
            "mean_diff_pos_minus_neg",
            "weighted_mean_neg",
            "weighted_mean_pos",
            "weighted_mean_diff_pos_minus_neg",
            "active_rate_neg",
            "active_rate_pos",
            "active_rate_diff_pos_minus_neg",
            "abs_mean_diff",
            "abs_weighted_mean_diff",
        ],
        latent_rows,
    )

    top_pos = [int(row["latent_idx"]) for row in sorted(latent_rows, key=lambda r: r["weighted_mean_diff_pos_minus_neg"], reverse=True)[:20]]
    top_neg = [int(row["latent_idx"]) for row in sorted(latent_rows, key=lambda r: r["weighted_mean_diff_pos_minus_neg"])[:20]]

    summary = {
        "tiles_csv": str(args.tiles_csv),
        "sae_ckpt": str(args.sae_ckpt),
        "sae_cfg": str(args.sae_cfg),
        "device": device,
        "d_in": d_in,
        "d_latent": d_latent,
        "tile_counts": {
            "hpv_neg": int(n_tiles[0]),
            "hpv_pos": int(n_tiles[1]),
            "total": int(total_rows),
        },
        "attention_weight_sums": {
            "hpv_neg": float(attn_weight_sum[0]),
            "hpv_pos": float(attn_weight_sum[1]),
        },
        "top_latents_hpv_pos_by_weighted_mean_diff": top_pos,
        "top_latents_hpv_neg_by_weighted_mean_diff": top_neg,
        "latent_stats_csv": str(args.out_dir / "latent_stats.csv"),
    }

    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[ok] wrote {args.out_dir / 'latent_stats.csv'}", flush=True)
    print(f"[ok] wrote {args.out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
