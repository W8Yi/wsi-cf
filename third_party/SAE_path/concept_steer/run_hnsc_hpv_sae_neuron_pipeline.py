#!/usr/bin/env python3
"""
Run split-level MIL+SAE neuron analysis for HPV on HNSC.

Pipeline:
1) Load split rows and MIL checkpoint.
2) For each slide, compute MIL attention across all tiles.
3) Encode all tiles with SAE, combine with attention, and compute per-slide neuron fractions:
     fraction_j = sum_i(attn_i * relu(z_ij)) / sum_k sum_i(attn_i * relu(z_ik))
4) Average slide-level fractions by HPV label (HPV- / HPV+).
5) Rank prominent neurons by (mean_fraction_pos - mean_fraction_neg).
6) Re-scan selected top neurons and export top attention-weighted tiles for visualization.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.classifier import AttentionMIL, GatedAttentionMIL
from utils.sae import load_sae_from_config, sae_encode_features


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def load_split_index(splits_dir: Path) -> List[Tuple[str, Path]]:
    index_path = splits_dir / "index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        items = []
        for item in payload.get("splits", []):
            items.append((item["split_name"], Path(item["tsv_path"])))
        if items:
            return items
    paths = sorted(splits_dir.glob("split_*.tsv"))
    return [(p.stem, p) for p in paths]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_mil_from_checkpoint(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})
    model_type = saved_args.get("model", "attention")
    common = {
        "embed_dim": int(saved_args.get("embed_dim", 1536)),
        "hidden_dim": int(saved_args.get("hidden_dim", 512)),
        "attn_dim": int(saved_args.get("attn_dim", 256)),
        "n_classes": 2,
        "dropout": float(saved_args.get("dropout", 0.25)),
    }
    if model_type == "gated":
        model = GatedAttentionMIL(
            **common,
            learnable_temperature=(not bool(saved_args.get("fixed_temperature", False))),
            init_temperature=float(saved_args.get("init_temperature", 1.0)),
        )
    else:
        model = AttentionMIL(**common)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def read_h5_features_coords(h5_path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{h5_path}: missing dataset 'features'")
        feats = handle["features"]
        if feats.ndim == 3 and feats.shape[0] == 1:
            x = feats[0]
        elif feats.ndim == 2:
            x = feats[:]
        else:
            raise ValueError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")

        coords = None
        if "coords" in handle:
            c = handle["coords"]
            if c.ndim == 3 and c.shape[0] == 1:
                coords = c[0]
            elif c.ndim == 2 and c.shape[1] == 2:
                coords = c[:]
    return np.asarray(x, dtype=np.float32), (np.asarray(coords) if coords is not None else None)


def encode_sae_batched(
    sae_model: torch.nn.Module,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device=device, dtype=torch.float32)
            zb = sae_encode_features(sae_model, xb)
            chunks.append(zb.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def run_mil_attention(
    mil_model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
) -> Tuple[np.ndarray, int, float]:
    xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits, y_prob, y_hat, a_raw, _ = mil_model(xt)
        attn = F.softmax(a_raw, dim=1).detach().cpu().numpy().reshape(-1)
        pred = int(y_hat.detach().cpu()[0, 0].item())
        prob_pos = float(y_prob.detach().cpu()[0, 1].item())
    return attn, pred, prob_pos


def resolve_rows(
    *,
    split_tsv: Path | None,
    rows_csv: Path | None,
    data_split: str,
    h5_dir: Path | None,
    max_slides: int,
) -> List[dict]:
    if rows_csv is not None:
        rows = read_csv(rows_csv)
    elif split_tsv is not None:
        rows = read_tsv(split_tsv)
    else:
        raise RuntimeError("Need either split_tsv or rows_csv")

    out = []
    for row in rows:
        row_split = row.get("split", row.get("data_split", ""))
        if rows_csv is None:
            if data_split != "both" and row_split != data_split:
                continue
        else:
            # When a direct rows CSV is provided (for example test_predictions.csv),
            # assume it already represents the target partition.
            row_split = "test" if data_split == "both" else data_split

        if "label" not in row or "slide_key" not in row:
            raise KeyError(f"rows CSV must contain at least label and slide_key columns: {rows_csv}")
        row["label"] = int(row["label"])
        row["split"] = row_split
        if "case_id" not in row or row["case_id"] == "":
            row["case_id"] = row["slide_key"]
        if h5_dir is not None:
            row["h5_path"] = str(h5_dir / f"{row['slide_key']}.h5")
        elif "h5_path" not in row:
            raise KeyError(f"rows CSV missing h5_path and --h5_dir was not provided: {rows_csv}")
        if not Path(row["h5_path"]).exists():
            raise FileNotFoundError(f"Missing H5: {row['h5_path']}")
        out.append(row)
    out.sort(key=lambda r: (r["split"], r["case_id"], r["slide_key"]))
    if max_slides > 0:
        out = out[:max_slides]
    if not out:
        raise RuntimeError("No rows selected after filters.")
    return out


def make_plots(
    out_dir: Path,
    *,
    top_pos_idx: np.ndarray,
    top_neg_idx: np.ndarray,
    diff: np.ndarray,
    mean_frac_neg: np.ndarray,
    mean_frac_pos: np.ndarray,
) -> None:
    # Top neurons by positive/negative HPV enrichment.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    pos_vals = diff[top_pos_idx]
    neg_vals = diff[top_neg_idx]

    axes[0].barh(np.arange(len(top_pos_idx)), pos_vals, color="#d62728")
    axes[0].set_yticks(np.arange(len(top_pos_idx)))
    axes[0].set_yticklabels([str(int(i)) for i in top_pos_idx])
    axes[0].invert_yaxis()
    axes[0].set_title("Top HPV+ Enriched Neurons")
    axes[0].set_xlabel("mean_fraction_pos - mean_fraction_neg")
    axes[0].set_ylabel("latent_idx")

    axes[1].barh(np.arange(len(top_neg_idx)), neg_vals, color="#1f77b4")
    axes[1].set_yticks(np.arange(len(top_neg_idx)))
    axes[1].set_yticklabels([str(int(i)) for i in top_neg_idx])
    axes[1].invert_yaxis()
    axes[1].set_title("Top HPV- Enriched Neurons")
    axes[1].set_xlabel("mean_fraction_pos - mean_fraction_neg")
    axes[1].set_ylabel("latent_idx")

    fig.savefig(out_dir / "top_neuron_diffs.png", dpi=200)
    plt.close(fig)

    # Heatmap of mean fractions for selected neurons.
    selected = np.unique(np.concatenate([top_pos_idx, top_neg_idx]))
    data = np.stack([mean_frac_neg[selected], mean_frac_pos[selected]], axis=0)
    fig2, ax2 = plt.subplots(figsize=(max(10, 0.28 * selected.size), 3.6), constrained_layout=True)
    im = ax2.imshow(data, aspect="auto", cmap="viridis")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["HPV-", "HPV+"])
    ax2.set_xticks(np.arange(selected.size))
    ax2.set_xticklabels([str(int(i)) for i in selected], rotation=90, fontsize=8)
    ax2.set_title("Mean Attention-Weighted Neuron Fractions (Selected)")
    fig2.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
    fig2.savefig(out_dir / "selected_neuron_fraction_heatmap.png", dpi=220)
    plt.close(fig2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=Path("runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h"))
    parser.add_argument("--splits_dir", type=Path, default=Path("metadata/manifests/hnsc_hpv_5fold"))
    parser.add_argument("--split_name", type=str, default="split_0")
    parser.add_argument(
        "--rows_csv",
        type=Path,
        default=None,
        help="Optional direct rows CSV (e.g., split_0/test_predictions.csv) to match a trained run exactly.",
    )
    parser.add_argument("--checkpoint_name", type=str, default="final.pt")
    parser.add_argument("--data_split", type=str, choices=["train", "test", "both"], default="test")
    parser.add_argument("--h5_dir", type=Path, default=None, help="Optional override: h5_dir/<slide_key>.h5")
    parser.add_argument("--sae_ckpt", type=Path, required=True)
    parser.add_argument("--sae_cfg", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h/sae_neuron_pipeline/split_0"))
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--top_k_neurons", type=int, default=20)
    parser.add_argument("--top_tiles_per_neuron", type=int, default=50)
    parser.add_argument("--local_top_tiles_per_slide", type=int, default=128)
    parser.add_argument("--max_slides", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    split_tsv: Path | None = None
    if args.rows_csv is None:
        split_items = dict(load_split_index(args.splits_dir))
        if args.split_name not in split_items:
            raise SystemExit(f"Split not found: {args.split_name}")
        split_tsv = split_items[args.split_name]
    else:
        if not args.rows_csv.exists():
            raise SystemExit(f"rows_csv not found: {args.rows_csv}")

    ckpt_path = args.run_dir / args.split_name / args.checkpoint_name
    if not ckpt_path.exists():
        raise SystemExit(f"Missing MIL checkpoint: {ckpt_path}")

    mil_model = build_mil_from_checkpoint(ckpt_path, device=device)
    sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    rows = resolve_rows(
        split_tsv=split_tsv,
        rows_csv=args.rows_csv,
        data_split=args.data_split,
        h5_dir=args.h5_dir,
        max_slides=args.max_slides,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: per-slide fractions then label-level average.
    label_sum_fraction = {0: np.zeros(d_latent, dtype=np.float64), 1: np.zeros(d_latent, dtype=np.float64)}
    label_sum_weighted = {0: np.zeros(d_latent, dtype=np.float64), 1: np.zeros(d_latent, dtype=np.float64)}
    label_n_slides = {0: 0, 1: 0}
    slide_rows = []

    print(f"[pass1] slides={len(rows)} split={args.split_name} data_split={args.data_split}", flush=True)
    for idx, row in enumerate(rows, start=1):
        x, _ = read_h5_features_coords(row["h5_path"])
        if x.shape[1] != d_in:
            raise RuntimeError(f"{row['h5_path']}: feature dim {x.shape[1]} != SAE d_in {d_in}")

        attn, pred, prob_pos = run_mil_attention(mil_model, x, device=device)
        z = encode_sae_batched(sae_model, x, batch_size=args.batch_size, device=device)
        z_pos = np.maximum(z, 0.0)

        # attention-weighted neuron mass per slide
        neuron_mass = (z_pos * attn[:, None]).sum(axis=0).astype(np.float64, copy=False)
        total_mass = float(neuron_mass.sum())
        if total_mass > 0:
            fraction = neuron_mass / total_mass
        else:
            fraction = np.zeros(d_latent, dtype=np.float64)

        label = int(row["label"])
        label_sum_fraction[label] += fraction
        label_sum_weighted[label] += neuron_mass
        label_n_slides[label] += 1

        top_idx = int(np.argmax(fraction)) if total_mass > 0 else -1
        top_frac = float(fraction[top_idx]) if top_idx >= 0 else 0.0
        slide_rows.append(
            {
                "split_name": args.split_name,
                "data_split": row["split"],
                "case_id": row["case_id"],
                "slide_key": row["slide_key"],
                "label": label,
                "pred": pred,
                "prob_pos": prob_pos,
                "n_tiles": int(x.shape[0]),
                "total_neuron_mass": total_mass,
                "top_neuron_idx": top_idx,
                "top_neuron_fraction": top_frac,
                "h5_path": row["h5_path"],
            }
        )
        if idx % 10 == 0 or idx == len(rows):
            print(f"  processed {idx}/{len(rows)} slides", flush=True)

    if label_n_slides[0] == 0 or label_n_slides[1] == 0:
        raise RuntimeError(f"Need both labels in selected rows. Got slide counts: {label_n_slides}")

    mean_frac_neg = label_sum_fraction[0] / float(label_n_slides[0])
    mean_frac_pos = label_sum_fraction[1] / float(label_n_slides[1])
    mean_weighted_neg = label_sum_weighted[0] / float(label_n_slides[0])
    mean_weighted_pos = label_sum_weighted[1] / float(label_n_slides[1])
    diff = mean_frac_pos - mean_frac_neg
    abs_diff = np.abs(diff)

    k = max(1, int(args.top_k_neurons))
    top_pos_idx = np.argsort(-diff)[:k]
    top_neg_idx = np.argsort(diff)[:k]
    top_pos_set = set(int(x) for x in top_pos_idx.tolist())
    top_neg_set = set(int(x) for x in top_neg_idx.tolist())

    rank_pos = np.empty(d_latent, dtype=np.int64)
    rank_neg = np.empty(d_latent, dtype=np.int64)
    rank_pos[np.argsort(-diff)] = np.arange(1, d_latent + 1, dtype=np.int64)
    rank_neg[np.argsort(diff)] = np.arange(1, d_latent + 1, dtype=np.int64)

    neuron_rows = []
    for j in range(d_latent):
        direction = "neutral"
        if j in top_pos_set:
            direction = "hpv_pos"
        elif j in top_neg_set:
            direction = "hpv_neg"
        neuron_rows.append(
            {
                "latent_idx": j,
                "mean_fraction_neg": float(mean_frac_neg[j]),
                "mean_fraction_pos": float(mean_frac_pos[j]),
                "diff_pos_minus_neg": float(diff[j]),
                "abs_diff": float(abs_diff[j]),
                "mean_weighted_activation_neg": float(mean_weighted_neg[j]),
                "mean_weighted_activation_pos": float(mean_weighted_pos[j]),
                "diff_weighted_activation_pos_minus_neg": float(mean_weighted_pos[j] - mean_weighted_neg[j]),
                "rank_pos_enriched": int(rank_pos[j]),
                "rank_neg_enriched": int(rank_neg[j]),
                "selected_direction": direction,
            }
        )

    neuron_rows.sort(key=lambda r: r["abs_diff"], reverse=True)
    write_csv(
        args.out_dir / "neuron_fraction_summary.csv",
        [
            "latent_idx",
            "mean_fraction_neg",
            "mean_fraction_pos",
            "diff_pos_minus_neg",
            "abs_diff",
            "mean_weighted_activation_neg",
            "mean_weighted_activation_pos",
            "diff_weighted_activation_pos_minus_neg",
            "rank_pos_enriched",
            "rank_neg_enriched",
            "selected_direction",
        ],
        neuron_rows,
    )
    write_csv(
        args.out_dir / "slide_level_summary.csv",
        [
            "split_name",
            "data_split",
            "case_id",
            "slide_key",
            "label",
            "pred",
            "prob_pos",
            "n_tiles",
            "total_neuron_mass",
            "top_neuron_idx",
            "top_neuron_fraction",
            "h5_path",
        ],
        slide_rows,
    )

    selected_idx = np.unique(np.concatenate([top_pos_idx, top_neg_idx])).astype(np.int64)
    selected_set = set(int(x) for x in selected_idx)

    # Pass 2: export top tiles for selected neurons.
    print(f"[pass2] selected_neurons={selected_idx.size} top_tiles_per_neuron={args.top_tiles_per_neuron}", flush=True)
    heaps: Dict[int, List[Tuple[float, dict]]] = {int(j): [] for j in selected_idx}

    for idx, row in enumerate(rows, start=1):
        x, coords = read_h5_features_coords(row["h5_path"])
        attn, pred, prob_pos = run_mil_attention(mil_model, x, device=device)
        z = encode_sae_batched(sae_model, x, batch_size=args.batch_size, device=device)
        z_pos = np.maximum(z, 0.0)

        z_sel = z_pos[:, selected_idx]  # [N, S]
        score_sel = z_sel * attn[:, None]  # attention-weighted activation
        n_tiles = score_sel.shape[0]
        k_local = min(max(1, int(args.local_top_tiles_per_slide)), n_tiles)

        for col, latent_idx in enumerate(selected_idx.tolist()):
            scores = score_sel[:, col]
            if k_local < n_tiles:
                local_pick = np.argpartition(scores, -k_local)[-k_local:]
            else:
                local_pick = np.arange(n_tiles, dtype=np.int64)

            for tile_i in local_pick.tolist():
                score = float(scores[tile_i])
                if score <= 0:
                    continue
                record = {
                    "latent_idx": int(latent_idx),
                    "label": int(row["label"]),
                    "pred": int(pred),
                    "prob_pos": float(prob_pos),
                    "case_id": row["case_id"],
                    "slide_key": row["slide_key"],
                    "tile_index": int(tile_i),
                    "attention": float(attn[tile_i]),
                    "sae_activation": float(z_sel[tile_i, col]),
                    "attention_weighted_activation": score,
                    "coord_x": (int(coords[tile_i, 0]) if coords is not None else ""),
                    "coord_y": (int(coords[tile_i, 1]) if coords is not None else ""),
                    "h5_path": row["h5_path"],
                }
                heap = heaps[int(latent_idx)]
                item = (score, record)
                if len(heap) < args.top_tiles_per_neuron:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)

        if idx % 10 == 0 or idx == len(rows):
            print(f"  rescanned {idx}/{len(rows)} slides", flush=True)

    tile_rows = []
    for latent_idx in sorted(selected_set):
        ranked = sorted(heaps[latent_idx], key=lambda x: x[0], reverse=True)
        direction = "hpv_pos" if latent_idx in top_pos_set else "hpv_neg"
        for rnk, (_, rec) in enumerate(ranked, start=1):
            rec_out = dict(rec)
            rec_out["prototype_rank"] = rnk
            rec_out["selected_direction"] = direction
            tile_rows.append(rec_out)

    tile_rows.sort(key=lambda r: (r["selected_direction"], r["latent_idx"], r["prototype_rank"]))
    write_csv(
        args.out_dir / "top_neuron_tiles.csv",
        [
            "latent_idx",
            "selected_direction",
            "prototype_rank",
            "label",
            "pred",
            "prob_pos",
            "case_id",
            "slide_key",
            "tile_index",
            "attention",
            "sae_activation",
            "attention_weighted_activation",
            "coord_x",
            "coord_y",
            "h5_path",
        ],
        tile_rows,
    )

    make_plots(
        args.out_dir,
        top_pos_idx=top_pos_idx,
        top_neg_idx=top_neg_idx,
        diff=diff,
        mean_frac_neg=mean_frac_neg,
        mean_frac_pos=mean_frac_pos,
    )

    summary = {
        "split_name": args.split_name,
        "split_tsv": (str(split_tsv) if split_tsv is not None else None),
        "rows_csv": (str(args.rows_csv) if args.rows_csv is not None else None),
        "run_dir": str(args.run_dir),
        "mil_checkpoint": str(ckpt_path),
        "data_split": args.data_split,
        "sae_ckpt": str(args.sae_ckpt),
        "sae_cfg": str(args.sae_cfg),
        "device": str(device),
        "d_in": int(d_in),
        "d_latent": int(d_latent),
        "n_slides": int(len(rows)),
        "n_slides_by_label": {
            "hpv_neg": int(label_n_slides[0]),
            "hpv_pos": int(label_n_slides[1]),
        },
        "top_k_neurons": int(args.top_k_neurons),
        "top_pos_latents": [int(x) for x in top_pos_idx.tolist()],
        "top_neg_latents": [int(x) for x in top_neg_idx.tolist()],
        "outputs": {
            "neuron_fraction_summary_csv": str(args.out_dir / "neuron_fraction_summary.csv"),
            "slide_level_summary_csv": str(args.out_dir / "slide_level_summary.csv"),
            "top_neuron_tiles_csv": str(args.out_dir / "top_neuron_tiles.csv"),
            "plot_top_neuron_diffs_png": str(args.out_dir / "top_neuron_diffs.png"),
            "plot_fraction_heatmap_png": str(args.out_dir / "selected_neuron_fraction_heatmap.png"),
        },
    }
    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[ok] wrote {args.out_dir / 'summary.json'}", flush=True)
    print(f"[ok] wrote {args.out_dir / 'neuron_fraction_summary.csv'}", flush=True)
    print(f"[ok] wrote {args.out_dir / 'top_neuron_tiles.csv'}", flush=True)


if __name__ == "__main__":
    main()
