#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.data.h5 import load_h5_features_coords
from wsi_cf.data.slides import crop_tile_rgb, open_slide
from wsi_cf.steering.manifest import build_real_grid_rows, write_manifest_csv, write_manifest_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a contiguous real donor grid from one slide/H5 pair for PixCell steering. "
            "Writes tile images, feature vectors, a steering manifest, and a donor mosaic."
        )
    )
    parser.add_argument("--slide", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--tile-index", type=int, required=True)
    parser.add_argument("--grid-side", type=int, default=4)
    parser.add_argument("--anchor-gx", type=int, default=1)
    parser.add_argument("--anchor-gy", type=int, default=1)
    parser.add_argument("--tile-size-20x", type=int, default=256)
    parser.add_argument("--out-tile-size", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def make_mosaic(paths: list[Path], grid_side: int, tile_px: int) -> Image.Image:
    canvas = Image.new("RGB", (grid_side * tile_px, grid_side * tile_px), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(paths):
        gx = idx % grid_side
        gy = idx // grid_side
        img = Image.open(path).convert("RGB")
        x0 = gx * tile_px
        y0 = gy * tile_px
        canvas.paste(img, (x0, y0))
        draw.rectangle([x0, y0, x0 + tile_px - 1, y0 + tile_px - 1], outline=(180, 180, 180), width=1)
        draw.text((x0 + 6, y0 + 6), f"{gx},{gy}", fill=(255, 255, 0))
    return canvas


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not args.slide.exists():
        raise FileNotFoundError(f"Slide not found: {args.slide}")
    if not args.h5.exists():
        raise FileNotFoundError(f"H5 not found: {args.h5}")

    feats, coords = load_h5_features_coords(args.h5)
    if coords is None:
        raise RuntimeError(f"{args.h5}: coords are required for real-grid export")
    rows = build_real_grid_rows(
        coords=coords,
        tile_index=int(args.tile_index),
        grid_side=int(args.grid_side),
        anchor_gx=int(args.anchor_gx),
        anchor_gy=int(args.anchor_gy),
    )

    export_dir = args.out_dir
    feature_dir = export_dir / "features"
    image_dir = export_dir / "tiles"
    feature_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    preview_rows: list[dict[str, object]] = []
    tile_paths: list[Path] = []
    crop_px_level0: int | None = None

    slide = open_slide(args.slide)
    try:
        for row in rows:
            gx = int(row["gx"])
            gy = int(row["gy"])
            idx = int(row["tile_index"])
            coord_x = int(row["coord_x"])
            coord_y = int(row["coord_y"])
            stem = f"gx_{gx}_gy_{gy}__tile_{idx:05d}__x_{coord_x}__y_{coord_y}"
            feat_path = feature_dir / f"{stem}.npy"
            img_path = image_dir / f"{stem}.png"
            np.save(feat_path, np.asarray(feats[idx], dtype=np.float32))
            tile_img, crop_px = crop_tile_rgb(
                slide,
                x=coord_x,
                y=coord_y,
                tile_size_20x=int(args.tile_size_20x),
                out_tile_size=int(args.out_tile_size),
            )
            crop_px_level0 = crop_px if crop_px_level0 is None else crop_px_level0
            tile_img.save(img_path)
            tile_paths.append(img_path)
            manifest.append({"gx": gx, "gy": gy, "path": str(feat_path)})
            preview_rows.append(
                {
                    "gx": gx,
                    "gy": gy,
                    "tile_index": idx,
                    "coord_x": coord_x,
                    "coord_y": coord_y,
                    "feature_path": str(feat_path),
                    "image_path": str(img_path),
                }
            )
    finally:
        slide.close()

    manifest_path = export_dir / "steer_manifest.json"
    csv_path = export_dir / "steer_manifest.csv"
    donor_grid_path = export_dir / "donor_grid.png"
    summary_path = export_dir / "summary.json"
    write_manifest_json(manifest_path, manifest)
    write_manifest_csv(csv_path, preview_rows)
    mosaic = make_mosaic(tile_paths, int(args.grid_side), int(args.out_tile_size))
    mosaic.save(donor_grid_path)
    write_json(
        summary_path,
        {
            "slide": str(args.slide),
            "h5": str(args.h5),
            "tile_index": int(args.tile_index),
            "grid_side": int(args.grid_side),
            "anchor_gx": int(args.anchor_gx),
            "anchor_gy": int(args.anchor_gy),
            "tile_size_20x": int(args.tile_size_20x),
            "out_tile_size": int(args.out_tile_size),
            "crop_px_level0": crop_px_level0,
            "manifest_path": str(manifest_path),
            "preview_path": str(donor_grid_path),
        },
    )
    print(f"[ok] wrote {manifest_path}")
    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {donor_grid_path}")
    print(f"[ok] wrote {summary_path}")


if __name__ == "__main__":
    main()
