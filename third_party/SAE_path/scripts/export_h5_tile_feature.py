#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export one tile feature vector from an H5 feature file to .npy so it can be used "
            "for PixCell or UNI steering experiments."
        )
    )
    ap.add_argument("--h5", type=Path, required=True, help="Input H5 with datasets `features` and optional `coords`.")
    ap.add_argument("--out", type=Path, required=True, help="Output .npy path for the exported feature vector.")
    ap.add_argument("--tile-index", type=int, default=None, help="Direct row index into the feature table.")
    ap.add_argument(
        "--coord",
        type=str,
        default=None,
        help="Exact tile coordinate as 'x,y'. Used only if --tile-index is omitted.",
    )
    ap.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="Optional JSON metadata output. Defaults to <out>.json",
    )
    return ap.parse_args()


def _load_features_and_coords(h5_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]
        if feats.ndim == 2:
            x = feats[:]
        elif feats.ndim == 3 and int(feats.shape[0]) == 1:
            x = feats[0]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")

        coords = None
        if "coords" in f:
            c = f["coords"]
            if c.ndim == 2 and int(c.shape[1]) == 2:
                coords = c[:]
            elif c.ndim == 3 and int(c.shape[0]) == 1 and int(c.shape[2]) == 2:
                coords = c[0]
    return np.asarray(x, dtype=np.float32), (None if coords is None else np.asarray(coords, dtype=np.int64))


def _parse_coord(coord_arg: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(coord_arg).split(",")]
    if len(parts) != 2:
        raise ValueError("--coord must be formatted as 'x,y'")
    return int(parts[0]), int(parts[1])


def main() -> None:
    args = _parse_args()
    if not args.h5.exists():
        raise FileNotFoundError(f"H5 not found: {args.h5}")
    if args.tile_index is None and args.coord is None:
        raise ValueError("Provide either --tile-index or --coord")

    feats, coords = _load_features_and_coords(args.h5)
    if feats.ndim != 2 or feats.shape[0] == 0:
        raise RuntimeError(f"{args.h5}: expected non-empty 2D features, got {feats.shape}")

    if args.tile_index is not None:
        idx = int(args.tile_index)
        if idx < 0 or idx >= int(feats.shape[0]):
            raise IndexError(f"--tile-index {idx} out of range for {args.h5} with {feats.shape[0]} rows")
    else:
        if coords is None:
            raise RuntimeError(f"{args.h5}: coords missing, so --coord cannot be used")
        target_xy = _parse_coord(args.coord)
        matches = np.where((coords[:, 0] == target_xy[0]) & (coords[:, 1] == target_xy[1]))[0]
        if matches.size == 0:
            raise RuntimeError(f"{args.h5}: no tile found at coord={target_xy}")
        idx = int(matches[0])

    vec = np.asarray(feats[idx], dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vec)

    meta_out = args.meta_out if args.meta_out is not None else args.out.with_suffix(args.out.suffix + ".json")
    meta = {
        "h5": str(args.h5),
        "tile_index": int(idx),
        "feature_dim": int(vec.shape[0]),
        "coord_x": (int(coords[idx, 0]) if coords is not None else None),
        "coord_y": (int(coords[idx, 1]) if coords is not None else None),
        "out": str(args.out),
    }
    meta_out.write_text(json.dumps(meta, indent=2))
    print(f"[ok] wrote {args.out}")
    print(f"[ok] wrote {meta_out}")


if __name__ == "__main__":
    main()
