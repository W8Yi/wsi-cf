#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.data.h5 import load_h5_features_coords, resolve_tile_index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one tile feature vector from an H5 feature file to .npy so it can be used "
            "for PixCell or UNI steering experiments."
        )
    )
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-index", type=int, default=None)
    parser.add_argument("--coord", type=str, default=None, help="Exact tile coordinate as 'x,y'")
    parser.add_argument("--meta-out", type=Path, default=None)
    return parser


def parse_coord(coord_arg: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(coord_arg).split(",")]
    if len(parts) != 2:
        raise ValueError("--coord must be formatted as 'x,y'")
    return int(parts[0]), int(parts[1])


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not args.h5.exists():
        raise FileNotFoundError(f"H5 not found: {args.h5}")
    feats, coords = load_h5_features_coords(args.h5)
    idx = resolve_tile_index(
        feats=feats,
        coords=coords,
        tile_index=args.tile_index,
        coord=(parse_coord(args.coord) if args.coord is not None else None),
    )
    vec = np.asarray(feats[idx], dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vec)
    meta_out = args.meta_out if args.meta_out is not None else args.out.with_suffix(args.out.suffix + ".json")
    write_json(
        meta_out,
        {
            "h5": str(args.h5),
            "tile_index": int(idx),
            "feature_dim": int(vec.shape[0]),
            "coord_x": (int(coords[idx, 0]) if coords is not None else None),
            "coord_y": (int(coords[idx, 1]) if coords is not None else None),
            "out": str(args.out),
        },
    )
    print(f"[ok] wrote {args.out}")
    print(f"[ok] wrote {meta_out}")


if __name__ == "__main__":
    main()
