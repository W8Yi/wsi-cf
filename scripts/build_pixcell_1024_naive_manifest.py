#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.data.donor_pool import parse_donor_pool_csv
from wsi_cf.steering.manifest import build_naive_manifest_preview, write_manifest_csv, write_manifest_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a naive PixCell-1024 4x4 steering manifest from 16 donor feature vectors. "
            "The output manifest can be passed to multidiff_img2img_wsi.py via --steer_manifest."
        )
    )
    parser.add_argument("--tile-pool-csv", type=Path, default=WSI_CF_ROOT / "artifacts/hnscc_tile_feature_pool/tile_pool.csv")
    parser.add_argument("--label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--select", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-json", type=Path, default=WSI_CF_ROOT / "artifacts/naive_pixcell1024_manifest.json")
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rows = parse_donor_pool_csv(args.tile_pool_csv, label=args.label)
    if len(rows) < int(args.count):
        raise SystemExit(f"Need {args.count} donor rows for label={args.label}, found {len(rows)}")
    if args.select == "random":
        rng = random.Random(int(args.seed))
        rows = list(rows)
        rng.shuffle(rows)
    manifest, preview = build_naive_manifest_preview(rows=rows, count=int(args.count), grid_side=4)
    write_manifest_json(args.out_json, manifest)
    out_csv = args.out_csv if args.out_csv is not None else args.out_json.with_suffix(".csv")
    write_manifest_csv(out_csv, preview)
    print(f"[ok] wrote {args.out_json}")
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
