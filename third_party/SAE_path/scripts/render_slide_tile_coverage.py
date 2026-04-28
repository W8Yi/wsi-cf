#!/usr/bin/env python3
"""
Render extracted tile coverage for one slide.

This script reads all tile coordinates from an H5 feature file, maps them onto a
real OpenSlide pyramid level, and writes:
- <out_prefix>__original.png
- <out_prefix>__tile_mask.png
- <out_prefix>__tile_overlay.png
- <out_prefix>__summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from e


def safe_float(x: object, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def infer_objective_power(slide: "openslide.OpenSlide") -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            val = safe_float(props.get(key), -1.0)
            if val > 0:
                return val
    mpp_x = safe_float(props.get("openslide.mpp-x"), -1.0)
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(tile_size_20x * (objective_power / 20.0))))


def choose_render_level(slide: "openslide.OpenSlide", max_dim: int) -> int:
    best_level = 0
    best_error = float("inf")
    target = float(max_dim)
    for level, (w, h) in enumerate(slide.level_dimensions):
        current = float(max(w, h))
        error = abs(current - target)
        if current <= target:
            return level
        if error < best_error:
            best_error = error
            best_level = level
    return best_level


def read_h5_coords(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        coords = handle["coords"]
        if coords.ndim == 3 and coords.shape[0] == 1:
            out = coords[0]
        elif coords.ndim == 2 and coords.shape[1] == 2:
            out = coords[:]
        else:
            raise ValueError(f"{h5_path}: unsupported coords shape {tuple(coords.shape)}")
    return np.asarray(out)


def build_bbox_density(coords: np.ndarray, stride: int) -> dict:
    xs = coords[:, 0].astype(np.int64)
    ys = coords[:, 1].astype(np.int64)
    width_slots = int((xs.max() - xs.min()) // stride + 1)
    height_slots = int((ys.max() - ys.min()) // stride + 1)
    bbox_slots = width_slots * height_slots
    return {
        "grid_width_slots": width_slots,
        "grid_height_slots": height_slots,
        "bbox_slots": bbox_slots,
        "bbox_density": float(coords.shape[0] / max(1, bbox_slots)),
    }


def save_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5_path", type=Path, required=True, help="Path to .h5 feature file.")
    ap.add_argument("--wsi_path", type=Path, required=True, help="Path to .svs slide.")
    ap.add_argument("--out_prefix", type=Path, required=True, help="Output prefix.")
    ap.add_argument("--thumb_max_dim", type=int, default=2048, help="Target render max dimension.")
    ap.add_argument("--tile_size_20x", type=int, default=256, help="Tile size at 20x extraction.")
    ap.add_argument("--overlay_alpha", type=int, default=140, help="Tile overlay alpha 0-255.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    coords = read_h5_coords(args.h5_path)
    if coords.shape[0] == 0:
        raise SystemExit(f"No coords in {args.h5_path}")

    slide = openslide.OpenSlide(str(args.wsi_path))
    try:
        w0, h0 = slide.dimensions
        objective_power = infer_objective_power(slide)
        tile_px = level0_tile_size(args.tile_size_20x, objective_power)
        render_level = choose_render_level(slide, args.thumb_max_dim)
        render_w, render_h = slide.level_dimensions[render_level]
        render_downsample = float(slide.level_downsamples[render_level])
        scale = 1.0 / max(render_downsample, 1e-12)

        original = slide.read_region((0, 0), render_level, (render_w, render_h)).convert("RGB")
        original_np = np.asarray(original)

        mask = np.zeros((render_h, render_w), dtype=np.uint8)
        for x0, y0 in coords:
            x1 = int(round(float(x0) * scale))
            y1 = int(round(float(y0) * scale))
            x2 = int(round((float(x0) + tile_px) * scale))
            y2 = int(round((float(y0) + tile_px) * scale))
            x1 = max(0, min(x1, render_w - 1))
            y1 = max(0, min(y1, render_h - 1))
            x2 = max(x1 + 1, min(x2, render_w))
            y2 = max(y1 + 1, min(y2, render_h))
            mask[y1:y2, x1:x2] = 255

        mask_rgb = np.zeros((render_h, render_w, 3), dtype=np.uint8)
        mask_rgb[..., 1] = mask

        overlay_rgba = np.zeros((render_h, render_w, 4), dtype=np.uint8)
        overlay_rgba[..., 1] = mask
        overlay_rgba[..., 3] = np.where(mask > 0, np.uint8(max(0, min(args.overlay_alpha, 255))), 0)
        overlay = Image.alpha_composite(
            original.convert("RGBA"),
            Image.fromarray(overlay_rgba, mode="RGBA"),
        ).convert("RGB")

        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        original.save(args.out_prefix.with_name(args.out_prefix.name + "__original.png"))
        Image.fromarray(mask_rgb).save(args.out_prefix.with_name(args.out_prefix.name + "__tile_mask.png"))
        overlay.save(args.out_prefix.with_name(args.out_prefix.name + "__tile_overlay.png"))

        covered_pixels = int((mask > 0).sum())
        summary = {
            "h5_path": str(args.h5_path),
            "wsi_path": str(args.wsi_path),
            "n_tiles": int(coords.shape[0]),
            "objective_power": float(objective_power),
            "tile_size_level0_px": int(tile_px),
            "slide_width_level0": int(w0),
            "slide_height_level0": int(h0),
            "render_level": int(render_level),
            "render_width": int(render_w),
            "render_height": int(render_h),
            "render_downsample": float(render_downsample),
            "covered_pixels_render": covered_pixels,
            "covered_fraction_render": float(covered_pixels / max(1, render_w * render_h)),
        }
        summary.update(build_bbox_density(coords, stride=tile_px))
        save_summary(args.out_prefix.with_name(args.out_prefix.name + "__summary.json"), summary)
        print(json.dumps(summary, indent=2))
    finally:
        slide.close()


if __name__ == "__main__":
    main()
