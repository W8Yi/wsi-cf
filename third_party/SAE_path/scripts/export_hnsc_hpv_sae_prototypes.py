#!/usr/bin/env python3
"""
Export top prototype tile references for selected SAE latents from the high-attention tile pool.

Typical use:
- read split_0 top_attention_tiles.csv
- read split_0 SAE stats summary.json
- take top N HPV+ latents and top N HPV- latents
- score all selected tiles with the SAE
- keep top activated tiles per latent

Outputs:
- prototype_tiles.csv
- selection.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    return np.asarray(x_sorted, dtype=np.float32)[inv]


def encode_batches(
    sae_model: torch.nn.Module,
    x: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device=device, dtype=torch.float32)
            zb = sae_encode_features(sae_model, xb)
            chunks.append(zb.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def parse_latent_ids(spec: str) -> List[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles_csv",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold/attention_exports/split_0/top_attention_tiles.csv"),
        help="Input top_attention_tiles.csv",
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
        help="SAE config path",
    )
    parser.add_argument(
        "--stats_summary",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold/sae_stats/split_0/summary.json"),
        help="SAE stats summary.json used to auto-select top HPV+ / HPV- latents",
    )
    parser.add_argument(
        "--latent_ids",
        type=str,
        default="",
        help="Optional explicit comma-separated latent IDs. If set, overrides auto-selection.",
    )
    parser.add_argument(
        "--top_pos_n",
        type=int,
        default=5,
        help="How many top HPV+ latents to use from stats_summary when latent_ids is not set.",
    )
    parser.add_argument(
        "--top_neg_n",
        type=int,
        default=5,
        help="How many top HPV- latents to use from stats_summary when latent_ids is not set.",
    )
    parser.add_argument(
        "--top_tiles_per_latent",
        type=int,
        default=25,
        help="How many prototype tile references to keep per latent.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold/sae_prototypes/split_0"),
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
        help="Optional debug cap on rows from tiles_csv",
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


def select_latents(args: argparse.Namespace) -> Tuple[List[int], Dict[int, str], dict]:
    if args.latent_ids:
        latent_ids = parse_latent_ids(args.latent_ids)
        polarity = {lid: "manual" for lid in latent_ids}
        meta = {"mode": "manual", "latent_ids": latent_ids}
        return latent_ids, polarity, meta

    payload = json.loads(args.stats_summary.read_text())
    pos = [int(x) for x in payload.get("top_latents_hpv_pos_by_weighted_mean_diff", [])[: args.top_pos_n]]
    neg = [int(x) for x in payload.get("top_latents_hpv_neg_by_weighted_mean_diff", [])[: args.top_neg_n]]
    latent_ids = pos + [x for x in neg if x not in set(pos)]
    polarity: Dict[int, str] = {}
    for lid in pos:
        polarity[lid] = "hpv_pos"
    for lid in neg:
        polarity[lid] = "hpv_neg"
    meta = {
        "mode": "from_stats_summary",
        "stats_summary": str(args.stats_summary),
        "top_pos_n": args.top_pos_n,
        "top_neg_n": args.top_neg_n,
        "pos_latents": pos,
        "neg_latents": neg,
    }
    return latent_ids, polarity, meta


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rows = read_csv(args.tiles_csv)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit(f"No rows in {args.tiles_csv}")

    latent_ids, polarity_map, selection_meta = select_latents(args)
    if not latent_ids:
        raise SystemExit("No latent IDs selected.")

    sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=device)
    for lid in latent_ids:
        if lid < 0 or lid >= d_latent:
            raise SystemExit(f"Latent {lid} out of range for d_latent={d_latent}")

    rows_by_h5: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        row["label"] = int(row["label"])
        row["tile_index"] = int(row["tile_index"])
        row["attention"] = float(row["attention"])
        rows_by_h5[row["h5_path"]].append(row)

    collected: Dict[int, List[dict]] = {lid: [] for lid in latent_ids}
    total_rows = 0

    latent_ids_np = np.asarray(latent_ids, dtype=np.int64)
    for group_idx, (h5_path, group_rows) in enumerate(rows_by_h5.items(), start=1):
        idx = np.asarray([row["tile_index"] for row in group_rows], dtype=np.int64)
        x = read_h5_features_indexed(h5_path, idx)
        if x.shape[1] != d_in:
            raise RuntimeError(f"{h5_path}: feature dim {x.shape[1]} != SAE d_in {d_in}")

        z = encode_batches(sae_model, x, batch_size=args.batch_size, device=device)
        z_sel = z[:, latent_ids_np]  # [N, L]

        for local_i, row in enumerate(group_rows):
            for j, lid in enumerate(latent_ids):
                activation = float(z_sel[local_i, j])
                collected[lid].append(
                    {
                        "latent_idx": lid,
                        "latent_group": polarity_map.get(lid, "manual"),
                        "activation": activation,
                        "attention": float(row["attention"]),
                        "label": int(row["label"]),
                        "pred": int(row["pred"]),
                        "prob_pos": float(row["prob_pos"]),
                        "case_id": row["case_id"],
                        "slide_key": row["slide_key"],
                        "tile_index": int(row["tile_index"]),
                        "tile_rank_from_attention": int(row["tile_rank"]),
                        "coord_x": row["coord_x"],
                        "coord_y": row["coord_y"],
                        "h5_path": row["h5_path"],
                    }
                )
            total_rows += 1

        if group_idx % 50 == 0:
            print(f"[progress] h5_groups={group_idx} tile_rows={total_rows}", flush=True)

    output_rows: List[dict] = []
    latent_summary: Dict[str, dict] = {}
    for lid in latent_ids:
        ranked = sorted(
            collected[lid],
            key=lambda r: (r["activation"], r["attention"]),
            reverse=True,
        )[: args.top_tiles_per_latent]
        for rank, row in enumerate(ranked, start=1):
            row = dict(row)
            row["prototype_rank"] = rank
            output_rows.append(row)

        top_acts = [float(r["activation"]) for r in ranked]
        latent_summary[str(lid)] = {
            "latent_idx": lid,
            "latent_group": polarity_map.get(lid, "manual"),
            "n_candidates_scored": len(collected[lid]),
            "top_tiles_kept": len(ranked),
            "top_activation_max": (max(top_acts) if top_acts else None),
            "top_activation_mean": (float(np.mean(np.asarray(top_acts, dtype=np.float32))) if top_acts else None),
        }

    output_rows.sort(key=lambda r: (r["latent_group"], r["latent_idx"], r["prototype_rank"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "prototype_tiles.csv",
        [
            "latent_idx",
            "latent_group",
            "prototype_rank",
            "activation",
            "attention",
            "label",
            "pred",
            "prob_pos",
            "case_id",
            "slide_key",
            "tile_index",
            "tile_rank_from_attention",
            "coord_x",
            "coord_y",
            "h5_path",
        ],
        output_rows,
    )

    summary = {
        "tiles_csv": str(args.tiles_csv),
        "sae_ckpt": str(args.sae_ckpt),
        "sae_cfg": str(args.sae_cfg),
        "device": device,
        "d_in": d_in,
        "d_latent": d_latent,
        "tile_rows_scored": total_rows,
        "selection": selection_meta,
        "top_tiles_per_latent": args.top_tiles_per_latent,
        "prototype_tiles_csv": str(args.out_dir / "prototype_tiles.csv"),
        "per_latent": latent_summary,
    }
    with (args.out_dir / "selection.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[ok] wrote {args.out_dir / 'prototype_tiles.csv'}", flush=True)
    print(f"[ok] wrote {args.out_dir / 'selection.json'}", flush=True)


if __name__ == "__main__":
    main()
