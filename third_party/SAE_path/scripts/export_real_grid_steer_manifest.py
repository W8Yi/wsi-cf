#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from e


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export a contiguous real donor grid from one slide/H5 pair for PixCell steering. "
            "Writes 16 tile images, 16 feature vectors, a steering manifest, and a donor mosaic."
        )
    )
    ap.add_argument("--slide", type=Path, required=True, help="Path to donor slide .svs")
    ap.add_argument("--h5", type=Path, required=True, help="Path to donor UNI feature H5")
    ap.add_argument("--tile-index", type=int, required=True, help="Anchor tile index in donor H5")
    ap.add_argument("--grid-side", type=int, default=4, help="Grid side length. For PixCell-1024 use 4.")
    ap.add_argument(
        "--anchor-gx",
        type=int,
        default=1,
        help="Grid x-position where the chosen anchor tile should appear.",
    )
    ap.add_argument(
        "--anchor-gy",
        type=int,
        default=1,
        help="Grid y-position where the chosen anchor tile should appear.",
    )
    ap.add_argument("--tile-size-20x", type=int, default=256)
    ap.add_argument("--out-tile-size", type=int, default=256)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def _read_h5_features_coords(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]
        coords = f["coords"]
        if feats.ndim == 2:
            x = feats[:]
        elif feats.ndim == 3 and int(feats.shape[0]) == 1:
            x = feats[0]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")
        if coords.ndim == 2 and int(coords.shape[1]) == 2:
            c = coords[:]
        elif coords.ndim == 3 and int(coords.shape[0]) == 1 and int(coords.shape[2]) == 2:
            c = coords[0]
        else:
            raise RuntimeError(f"{h5_path}: unsupported coords shape {tuple(coords.shape)}")
    x = np.asarray(x, dtype=np.float32)
    c = np.asarray(c, dtype=np.int64)
    if x.shape[0] != c.shape[0]:
        raise RuntimeError(f"{h5_path}: features/coords row mismatch {x.shape} vs {c.shape}")
    return x, c


def _infer_coord_step(coords: np.ndarray) -> int:
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    dx = np.diff(xs)
    dy = np.diff(ys)
    vals = np.concatenate([dx[dx > 0], dy[dy > 0]])
    if vals.size == 0:
        raise RuntimeError("Could not infer coordinate step from coords")
    return int(np.min(vals))


def _infer_objective_power(slide: "openslide.OpenSlide") -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            try:
                v = float(props.get(key))
            except Exception:
                v = -1.0
            if v > 0:
                return v
    try:
        mpp_x = float(props.get("openslide.mpp-x", -1.0))
    except Exception:
        mpp_x = -1.0
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def _level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(float(tile_size_20x) * (float(objective_power) / 20.0))))


def _crop_tile_rgb(
    slide: "openslide.OpenSlide",
    *,
    x: int,
    y: int,
    tile_size_20x: int,
    out_tile_size: int,
) -> tuple[Image.Image, int]:
    crop_px = _level0_tile_size(tile_size_20x, _infer_objective_power(slide))
    rgba = slide.read_region((int(x), int(y)), 0, (int(crop_px), int(crop_px))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    if crop_px != int(out_tile_size):
        rgb = rgb.resize((int(out_tile_size), int(out_tile_size)), resample=Image.BILINEAR)
    return rgb, int(crop_px)


def _make_mosaic(paths: list[Path], grid_side: int, tile_px: int) -> Image.Image:
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


def main() -> None:
    args = _parse_args()
    if not args.slide.exists():
        raise FileNotFoundError(f"Slide not found: {args.slide}")
    if not args.h5.exists():
        raise FileNotFoundError(f"H5 not found: {args.h5}")
    if args.grid_side <= 0:
        raise SystemExit("--grid-side must be > 0")
    if not (0 <= args.anchor_gx < args.grid_side and 0 <= args.anchor_gy < args.grid_side):
        raise SystemExit("anchor-gx/anchor-gy must be within the grid")

    feats, coords = _read_h5_features_coords(args.h5)
    n = int(coords.shape[0])
    if args.tile_index < 0 or args.tile_index >= n:
        raise SystemExit(f"--tile-index {args.tile_index} out of range for {args.h5} with {n} tiles")

    step = _infer_coord_step(coords)
    anchor_x = int(coords[int(args.tile_index), 0])
    anchor_y = int(coords[int(args.tile_index), 1])
    top_left_x = int(anchor_x - args.anchor_gx * step)
    top_left_y = int(anchor_y - args.anchor_gy * step)
    coord_to_idx = {tuple(map(int, c)): i for i, c in enumerate(coords.tolist())}

    export_dir = args.out_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = export_dir / "features"
    image_dir = export_dir / "tiles"
    feature_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(args.slide))
    try:
        manifest: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        tile_paths: list[Path] = []
        crop_px_level0: int | None = None
        for gy in range(int(args.grid_side)):
            for gx in range(int(args.grid_side)):
                coord_x = int(top_left_x + gx * step)
                coord_y = int(top_left_y + gy * step)
                idx = coord_to_idx.get((coord_x, coord_y))
                if idx is None:
                    raise RuntimeError(
                        f"Requested real grid is incomplete. Missing coord {(coord_x, coord_y)} "
                        f"for gx={gx}, gy={gy}."
                    )
                feat = np.asarray(feats[int(idx)], dtype=np.float32)
                stem = f"gx_{gx}_gy_{gy}__tile_{int(idx):05d}__x_{coord_x}__y_{coord_y}"
                feat_path = feature_dir / f"{stem}.npy"
                img_path = image_dir / f"{stem}.png"
                np.save(feat_path, feat)
                tile_img, crop_px = _crop_tile_rgb(
                    slide,
                    x=coord_x,
                    y=coord_y,
                    tile_size_20x=int(args.tile_size_20x),
                    out_tile_size=int(args.out_tile_size),
                )
                crop_px_level0 = crop_px if crop_px_level0 is None else crop_px_level0
                tile_img.save(img_path)
                tile_paths.append(img_path)
                manifest.append({"gx": int(gx), "gy": int(gy), "path": str(feat_path)})
                rows.append(
                    {
                        "gx": int(gx),
                        "gy": int(gy),
                        "tile_index": int(idx),
                        "coord_x": int(coord_x),
                        "coord_y": int(coord_y),
                        "feature_path": str(feat_path),
                        "image_path": str(img_path),
                    }
                )
    finally:
        slide.close()

    manifest_path = export_dir / "steer_manifest.json"
    csv_path = export_dir / "steer_manifest.csv"
    preview_path = export_dir / "donor_grid.png"
    summary_path = export_dir / "summary.json"

    manifest_path.write_text(json.dumps(manifest, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    mosaic = _make_mosaic(tile_paths, grid_side=int(args.grid_side), tile_px=int(args.out_tile_size))
    mosaic.save(preview_path)

    summary = {
        "slide": str(args.slide),
        "h5": str(args.h5),
        "tile_index": int(args.tile_index),
        "anchor_coord_x": int(anchor_x),
        "anchor_coord_y": int(anchor_y),
        "grid_side": int(args.grid_side),
        "anchor_gx": int(args.anchor_gx),
        "anchor_gy": int(args.anchor_gy),
        "coord_step": int(step),
        "top_left_x": int(top_left_x),
        "top_left_y": int(top_left_y),
        "tile_size_20x": int(args.tile_size_20x),
        "out_tile_size": int(args.out_tile_size),
        "crop_px_level0": int(crop_px_level0) if crop_px_level0 is not None else None,
        "manifest_json": str(manifest_path),
        "manifest_csv": str(csv_path),
        "donor_grid_png": str(preview_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[ok] wrote {manifest_path}")
    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {preview_path}")
    print(f"[ok] wrote {summary_path}")


if __name__ == "__main__":
    main()
