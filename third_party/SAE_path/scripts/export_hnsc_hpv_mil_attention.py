#!/usr/bin/env python3
"""
Fast, simple attention export for HNSC HPV MIL runs.

For each selected split:
- load the trained MIL checkpoint
- run one forward pass per slide
- save slide-level predictions
- save high-attention tiles per slide

Outputs:
- <out_dir>/<split_name>/slide_scores.csv
- <out_dir>/<split_name>/top_attention_tiles.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

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


def read_h5(h5_path: str) -> Tuple[np.ndarray, np.ndarray | None]:
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


def maybe_subsample(
    x: np.ndarray,
    coords: np.ndarray | None,
    max_tiles: int,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    n = x.shape[0]
    source_idx = np.arange(n, dtype=np.int64)
    if max_tiles <= 0 or n <= max_tiles:
        return x, coords, source_idx

    # Fast deterministic downsample: evenly spaced indices.
    idx = np.linspace(0, n - 1, num=max_tiles, dtype=np.int64)
    x = x[idx]
    coords = coords[idx] if coords is not None else None
    source_idx = source_idx[idx]
    return x, coords, source_idx


def build_model_from_checkpoint(ckpt_path: Path, device: torch.device):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold"),
        help="Training run directory containing split_*/final.pt",
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("metadata/manifests/hnsc_hpv_5fold"),
        help="Directory containing split_*.tsv and optional index.json",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold/attention_exports"),
        help="Where to write exported attention CSVs",
    )
    parser.add_argument(
        "--data_split",
        choices=["train", "test", "both"],
        default="train",
        help="Which rows from the split TSV to process.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="split_0",
        help="Split name to export. Use empty string to export all splits.",
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=1.0,
        help="Percent of tiles to keep per slide by attention rank when threshold is not set.",
    )
    parser.add_argument(
        "--attn_threshold",
        type=float,
        default=-1.0,
        help="If >= 0, keep all tiles with attention >= this value instead of percentile selection.",
    )
    parser.add_argument("--max_slides", type=int, default=0, help="Limit slides per split for quick runs.")
    parser.add_argument("--max_tiles", type=int, default=0, help="Optional downsample of tiles per slide.")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="final.pt",
        help="Checkpoint filename inside each split folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda:0, cpu, or auto",
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
    if args.top_percent <= 0 or args.top_percent > 100:
        raise SystemExit("--top_percent must be in (0, 100].")

    split_items = load_split_index(args.splits_dir)
    if args.split_name:
        split_items = [item for item in split_items if item[0] == args.split_name]
    if not split_items:
        raise SystemExit("No matching split TSVs found.")

    for split_name, split_tsv in split_items:
        ckpt_path = args.run_dir / split_name / args.checkpoint_name
        if not ckpt_path.exists():
            raise SystemExit(f"Missing checkpoint: {ckpt_path}")

        model = build_model_from_checkpoint(ckpt_path, device=device)
        rows = read_tsv(split_tsv)

        chosen_rows: List[dict] = []
        for row in rows:
            row_split = row["split"]
            if args.data_split == "both" or row_split == args.data_split:
                chosen_rows.append(row)

        if args.max_slides > 0:
            chosen_rows = chosen_rows[: args.max_slides]

        slide_rows = []
        tile_rows = []

        print(f"[split] {split_name} rows={len(chosen_rows)}", flush=True)

        for row_idx, row in enumerate(chosen_rows):
            h5_path = row["h5_path"]
            x, coords = read_h5(h5_path)
            x, coords, source_idx = maybe_subsample(x, coords, max_tiles=args.max_tiles)

            xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                logits, y_prob, y_hat, a_raw, _ = model(xt)
                attn = F.softmax(a_raw, dim=1).detach().cpu().numpy().reshape(-1)

            pred = int(y_hat.detach().cpu()[0, 0].item())
            prob_pos = float(y_prob.detach().cpu()[0, 1].item())
            logit_neg = float(logits.detach().cpu()[0, 0].item())
            logit_pos = float(logits.detach().cpu()[0, 1].item())

            slide_rows.append(
                {
                    "split_name": split_name,
                    "data_split": row["split"],
                    "case_id": row["case_id"],
                    "slide_key": row["slide_key"],
                    "label": int(row["label"]),
                    "pred": pred,
                    "prob_pos": prob_pos,
                    "logit_neg": logit_neg,
                    "logit_pos": logit_pos,
                    "n_tiles_used": int(x.shape[0]),
                    "h5_path": h5_path,
                }
            )

            order = np.argsort(-attn)
            if args.attn_threshold >= 0:
                selected = np.where(attn >= args.attn_threshold)[0]
                if selected.size == 0:
                    selected = order[:1]
                else:
                    selected = selected[np.argsort(-attn[selected])]
            else:
                k = max(1, int(np.ceil(attn.shape[0] * (args.top_percent / 100.0))))
                k = min(k, attn.shape[0])
                selected = order[:k]

            top_idx = selected
            for rank, local_idx in enumerate(top_idx, start=1):
                orig_idx = int(source_idx[local_idx])
                coord_x = ""
                coord_y = ""
                if coords is not None:
                    coord_x = int(coords[local_idx, 0])
                    coord_y = int(coords[local_idx, 1])
                tile_rows.append(
                    {
                        "split_name": split_name,
                        "data_split": row["split"],
                        "case_id": row["case_id"],
                        "slide_key": row["slide_key"],
                        "label": int(row["label"]),
                        "pred": pred,
                        "prob_pos": prob_pos,
                        "tile_rank": rank,
                        "tile_index": orig_idx,
                        "attention": float(attn[local_idx]),
                        "coord_x": coord_x,
                        "coord_y": coord_y,
                        "h5_path": h5_path,
                    }
                )

            if (row_idx + 1) % 25 == 0:
                print(f"  processed {row_idx + 1}/{len(chosen_rows)}", flush=True)

        split_out = args.out_dir / split_name
        write_csv(
            split_out / "slide_scores.csv",
            [
                "split_name",
                "data_split",
                "case_id",
                "slide_key",
                "label",
                "pred",
                "prob_pos",
                "logit_neg",
                "logit_pos",
                "n_tiles_used",
                "h5_path",
            ],
            slide_rows,
        )
        write_csv(
            split_out / "top_attention_tiles.csv",
            [
                "split_name",
                "data_split",
                "case_id",
                "slide_key",
                "label",
                "pred",
                "prob_pos",
                "tile_rank",
                "tile_index",
                "attention",
                "coord_x",
                "coord_y",
                "h5_path",
            ],
            tile_rows,
        )
        with (split_out / "export_config.json").open("w") as handle:
            json.dump(
                {
                    "split_name": split_name,
                    "split_tsv": str(split_tsv),
                    "checkpoint": str(ckpt_path),
                    "data_split": args.data_split,
                    "top_percent": args.top_percent,
                    "attn_threshold": args.attn_threshold,
                    "max_slides": args.max_slides,
                    "max_tiles": args.max_tiles,
                    "n_slides_exported": len(slide_rows),
                    "n_tile_rows": len(tile_rows),
                },
                handle,
                indent=2,
            )
        print(f"[ok] wrote {split_out}", flush=True)


if __name__ == "__main__":
    main()
