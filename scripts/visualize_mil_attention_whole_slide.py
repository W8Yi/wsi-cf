#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.data.slides import find_slide_path, infer_objective_power, level0_tile_size, open_slide
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, read_h5_features_coords, run_mil_attention


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize whole-slide MIL attention from UNI2 H5 feature bags. "
            "Writes a coordinate-space attention map, an optional SVS thumbnail overlay, "
            "top-tile CSVs, and cohort-level summaries."
        )
    )
    parser.add_argument("--split-tsv", type=Path, default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv"))
    parser.add_argument("--features-root", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"))
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/HNSCC_slides"))
    parser.add_argument("--mil-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("/common/users/wq50/wsi_cf/artifacts/whole_slide_attention_vis"))
    parser.add_argument("--split", type=str, default="test", help="Use a split name from split_0.tsv, or 'all'.")
    parser.add_argument("--slide-key", action="append", default=[], help="Optional slide key filter. Can be passed multiple times.")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None, help="Optional label filter.")
    parser.add_argument("--max-slides", type=int, default=8, help="0 means no limit.")
    parser.add_argument("--max-slides-per-label", type=int, default=0, help="Optional balanced cap per label. Applied before --max-slides.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--attention-percentile", type=float, default=95.0)
    parser.add_argument("--thumbnail-max-side", type=int, default=3072)
    parser.add_argument("--tile-size-20x", type=int, default=256)
    parser.add_argument("--heatmap-blur-px", type=float, default=5.0)
    parser.add_argument("--heatmap-alpha", type=int, default=170)
    parser.add_argument("--heatmap-gamma", type=float, default=0.75)
    parser.add_argument("--heatmap-render-mode", type=str, default="clam_like", choices=["clam_like", "tile_overlay"])
    parser.add_argument("--heatmap-cmap", type=str, default="coolwarm")
    parser.add_argument("--heatmap-convert-to-percentiles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-mismatched-slide-overlay",
        action="store_true",
        help="Debug only: draw overlay even if H5 coords exceed local SVS dimensions. Usually this is misleading.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def read_split_rows(path: Path, *, split: str, features_root: Path, slides_dir: Path, slide_keys: set[str], label: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            slide_key = str(row.get("slide_key", "")).strip()
            if not slide_key:
                continue
            if slide_keys and slide_key not in slide_keys:
                continue
            row_split = str(row.get("split", "")).strip()
            if str(split).lower() != "all" and row_split != str(split):
                continue
            row_label = int(row.get("label", -1))
            if row_label not in (0, 1):
                continue
            if label is not None and row_label != int(label):
                continue
            h5_path = features_root / f"{slide_key}.h5"
            if not h5_path.exists():
                candidate = Path(str(row.get("h5_path", "")))
                if candidate.exists():
                    h5_path = candidate
                else:
                    continue
            slide_path = find_slide_path(slides_dir, slide_key)
            rows.append(
                {
                    "slide_key": slide_key,
                    "case_id": str(row.get("case_id", slide_key)),
                    "split": row_split,
                    "label": row_label,
                    "h5_path": str(h5_path),
                    "slide_path": str(slide_path) if slide_path is not None else "",
                }
            )
    rows.sort(key=lambda item: (int(item["label"]), str(item["slide_key"])))
    return rows


def attention_color(value: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, float(value)))
    # Blue -> cyan -> yellow -> red -> black gives visible low-attention
    # context while making the strongest hotspots unmistakable.
    anchors = [
        (0.0, (20, 45, 235)),
        (0.35, (30, 210, 245)),
        (0.65, (255, 230, 70)),
        (0.88, (230, 35, 25)),
        (1.0, (20, 0, 0)),
    ]
    for (left_v, left_c), (right_v, right_c) in zip(anchors[:-1], anchors[1:]):
        if v <= right_v:
            t = (v - left_v) / max(1e-6, right_v - left_v)
            return tuple(int(round(left_c[i] * (1.0 - t) + right_c[i] * t)) for i in range(3))
    return anchors[-1][1]


def normalize_attention(attention: np.ndarray, *, percentile: float) -> np.ndarray:
    attn = np.asarray(attention, dtype=np.float32).reshape(-1)
    lo = float(np.percentile(attn, 5.0))
    hi = float(np.percentile(attn, max(5.0, min(100.0, float(percentile)))))
    if hi <= lo:
        hi = float(attn.max())
    if hi <= lo:
        return np.zeros_like(attn, dtype=np.float32)
    return np.clip((attn - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def attention_percentiles(attention: np.ndarray) -> np.ndarray:
    attn = np.asarray(attention, dtype=np.float32).reshape(-1)
    if attn.size == 0:
        return attn
    order = np.argsort(attn, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=attn.size, endpoint=True, dtype=np.float32)
    return ranks


def infer_coord_tile_size(coords: np.ndarray, *, fallback: int) -> int:
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return int(fallback)
    candidates: list[int] = []
    for axis in (0, 1):
        vals = np.unique(arr[:, axis])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    if not candidates:
        return int(fallback)
    return max(1, int(min(candidates)))


def coords_fit_slide_frame(*, coords: np.ndarray, tile_size: int, slide_w: int, slide_h: int, tolerance: float = 1.05) -> tuple[bool, str]:
    arr = np.asarray(coords, dtype=np.int64)
    max_x = int(arr[:, 0].max()) + int(tile_size)
    max_y = int(arr[:, 1].max()) + int(tile_size)
    if max_x > int(slide_w) * float(tolerance) or max_y > int(slide_h) * float(tolerance):
        return (
            False,
            f"h5_coord_extent_{max_x}x{max_y}_exceeds_svs_{int(slide_w)}x{int(slide_h)}",
        )
    return True, "coords_match_slide_frame"


def scaled_canvas_size(width: int, height: int, max_side: int) -> tuple[int, int, float]:
    scale = min(float(max_side) / float(max(1, width)), float(max_side) / float(max(1, height)), 1.0)
    out_w = max(1, int(round(float(width) * scale)))
    out_h = max(1, int(round(float(height) * scale)))
    return out_w, out_h, float(scale)


def draw_attention_rectangles(
    base: Image.Image,
    *,
    coords: np.ndarray,
    attention_norm: np.ndarray,
    tile_size_level0: int,
    scale_x: float,
    scale_y: float,
    top_indices: set[int],
    high_threshold: float,
    alpha: int,
) -> Image.Image:
    out = base.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    coords_i = np.asarray(coords, dtype=np.int64)
    for idx, (x, y) in enumerate(coords_i.tolist()):
        score = float(attention_norm[idx])
        if score <= 0.0 and idx not in top_indices:
            continue
        x0 = int(round(float(x) * float(scale_x)))
        y0 = int(round(float(y) * float(scale_y)))
        x1 = int(round(float(x + int(tile_size_level0)) * float(scale_x)))
        y1 = int(round(float(y + int(tile_size_level0)) * float(scale_y)))
        color = attention_color(score)
        fill_alpha = max(18, int(round(float(alpha) * score)))
        draw.rectangle([x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)], fill=(*color, fill_alpha))
        if idx in top_indices:
            draw.rectangle([x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)], outline=(255, 255, 0, 230), width=2)
        elif score >= float(high_threshold):
            draw.rectangle([x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)], outline=(255, 80, 40, 170), width=1)
    return Image.alpha_composite(out, overlay).convert("RGB")


def make_attention_heat_rgba(
    *,
    canvas_size: tuple[int, int],
    coords: np.ndarray,
    attention_norm: np.ndarray,
    tile_size_level0: int,
    scale_x: float,
    scale_y: float,
    blur_px: float,
    alpha: int,
) -> Image.Image:
    width, height = int(canvas_size[0]), int(canvas_size[1])
    heat = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=np.float32)
    coords_i = np.asarray(coords, dtype=np.int64)
    attn = np.asarray(attention_norm, dtype=np.float32).reshape(-1)
    for idx, (x, y) in enumerate(coords_i.tolist()):
        score = float(attn[idx])
        x0 = max(0, min(width, int(round(float(x) * float(scale_x)))))
        y0 = max(0, min(height, int(round(float(y) * float(scale_y)))))
        x1 = max(0, min(width, int(round(float(x + int(tile_size_level0)) * float(scale_x)))))
        y1 = max(0, min(height, int(round(float(y + int(tile_size_level0)) * float(scale_y)))))
        if x1 <= x0 or y1 <= y0:
            continue
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], score)
        valid[y0:y1, x0:x1] = 1.0

    heat_img = Image.fromarray(np.clip(heat * 255.0, 0, 255).astype(np.uint8), mode="L")
    if float(blur_px) > 0:
        heat_img = heat_img.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    h = np.asarray(heat_img, dtype=np.float32) / 255.0
    valid_mask = valid > 0.0

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    anchors = [
        (0.0, np.asarray([20, 45, 235], dtype=np.float32)),
        (0.35, np.asarray([30, 210, 245], dtype=np.float32)),
        (0.65, np.asarray([255, 230, 70], dtype=np.float32)),
        (0.88, np.asarray([230, 35, 25], dtype=np.float32)),
        (1.0, np.asarray([20, 0, 0], dtype=np.float32)),
    ]
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for (left_v, left_c), (right_v, right_c) in zip(anchors[:-1], anchors[1:]):
        mask = valid_mask & (h >= float(left_v)) & (h <= float(right_v))
        if not np.any(mask):
            continue
        t = (h[mask] - float(left_v)) / max(1e-6, float(right_v) - float(left_v))
        rgb[mask] = left_c[None, :] * (1.0 - t[:, None]) + right_c[None, :] * t[:, None]
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    tile_alpha = np.zeros_like(h, dtype=np.float32)
    tile_alpha[valid_mask] = 0.45 + 0.55 * h[valid_mask]
    rgba[..., 3] = np.clip(tile_alpha * float(alpha), 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def make_clam_like_heatmap_overlay(
    base: Image.Image,
    *,
    coords: np.ndarray,
    attention_scores: np.ndarray,
    tile_size_level0: int,
    scale_x: float,
    scale_y: float,
    blur_px: float,
    alpha: float,
    cmap_name: str,
    convert_to_percentiles: bool,
    gamma: float,
) -> Image.Image:
    base_rgb = np.asarray(base.convert("RGB"), dtype=np.uint8)
    height, width = base_rgb.shape[:2]
    overlay = np.zeros((height, width), dtype=np.float32)
    counter = np.zeros((height, width), dtype=np.float32)
    coords_i = np.asarray(coords, dtype=np.int64)
    scores = np.asarray(attention_scores, dtype=np.float32).reshape(-1)
    if bool(convert_to_percentiles):
        scores = attention_percentiles(scores)
    else:
        smin = float(scores.min()) if scores.size else 0.0
        smax = float(scores.max()) if scores.size else 1.0
        scores = (scores - smin) / (smax - smin) if smax > smin else np.zeros_like(scores, dtype=np.float32)
    scores = np.power(np.clip(scores, 0.0, 1.0), float(gamma)).astype(np.float32, copy=False)

    for idx, (x, y) in enumerate(coords_i.tolist()):
        score = float(scores[idx])
        x0 = max(0, min(width, int(round(float(x) * float(scale_x)))))
        y0 = max(0, min(height, int(round(float(y) * float(scale_y)))))
        x1 = max(0, min(width, int(round(float(x + int(tile_size_level0)) * float(scale_x)))))
        y1 = max(0, min(height, int(round(float(y + int(tile_size_level0)) * float(scale_y)))))
        if x1 <= x0 or y1 <= y0:
            continue
        overlay[y0:y1, x0:x1] += score
        counter[y0:y1, x0:x1] += 1.0

    valid_mask = counter > 0
    overlay[valid_mask] /= np.maximum(counter[valid_mask], 1e-6)
    overlay_img = Image.fromarray(np.clip(overlay * 255.0, 0, 255).astype(np.uint8))
    if float(blur_px) > 0:
        overlay_img = overlay_img.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    overlay = np.asarray(overlay_img, dtype=np.float32) / 255.0
    cmap = plt.get_cmap(str(cmap_name))
    color_rgb = (cmap(np.clip(overlay, 0.0, 1.0))[..., :3] * 255.0).astype(np.uint8)
    patch_w = max(1.0, float(tile_size_level0) * float(scale_x))
    patch_h = max(1.0, float(tile_size_level0) * float(scale_y))
    post_blur = max(float(blur_px) * 0.75, 0.12 * max(patch_w, patch_h))
    if post_blur > 0:
        color_rgb = np.asarray(
            Image.fromarray(color_rgb, mode="RGB").filter(ImageFilter.GaussianBlur(radius=float(post_blur))),
            dtype=np.uint8,
        )
    soft_valid = np.asarray(
        Image.fromarray((valid_mask.astype(np.uint8) * 255), mode="L").filter(
            ImageFilter.GaussianBlur(radius=max(1.0, float(post_blur) * 0.75))
        ),
        dtype=np.float32,
    ) / 255.0
    out = base_rgb.astype(np.float32).copy()
    a = float(max(0.0, min(1.0, alpha)))
    alpha_map = np.clip(soft_valid * a, 0.0, 1.0)[..., None]
    out = color_rgb.astype(np.float32) * alpha_map + out * (1.0 - alpha_map)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def draw_top_tile_outlines(
    img: Image.Image,
    *,
    coords: np.ndarray,
    tile_size_level0: int,
    scale_x: float,
    scale_y: float,
    top_indices: set[int],
) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    coords_i = np.asarray(coords, dtype=np.int64)
    for idx in sorted(top_indices):
        x, y = coords_i[int(idx)].tolist()
        x0 = int(round(float(x) * float(scale_x)))
        y0 = int(round(float(y) * float(scale_y)))
        x1 = int(round(float(x + int(tile_size_level0)) * float(scale_x)))
        y1 = int(round(float(y + int(tile_size_level0)) * float(scale_y)))
        draw.rectangle([x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)], outline=(255, 255, 0), width=2)
    return out


def make_heatmap_overlay(
    base: Image.Image,
    *,
    coords: np.ndarray,
    attention_norm: np.ndarray,
    tile_size_level0: int,
    scale_x: float,
    scale_y: float,
    top_indices: set[int],
    blur_px: float,
    alpha: int,
) -> tuple[Image.Image, Image.Image]:
    raise NotImplementedError("make_heatmap_overlay is replaced by explicit render-mode paths in main().")


def make_coordinate_attention_map(
    *,
    coords: np.ndarray,
    attention_norm: np.ndarray,
    tile_size_level0: int,
    max_side: int,
    top_indices: set[int],
    high_threshold: float,
) -> Image.Image:
    coords_i = np.asarray(coords, dtype=np.int64)
    max_x = int(coords_i[:, 0].max()) + int(tile_size_level0)
    max_y = int(coords_i[:, 1].max()) + int(tile_size_level0)
    out_w, out_h, scale = scaled_canvas_size(max_x, max_y, int(max_side))
    base = Image.new("RGB", (out_w, out_h), (245, 245, 245))
    return draw_attention_rectangles(
        base,
        coords=coords_i,
        attention_norm=attention_norm,
        tile_size_level0=int(tile_size_level0),
        scale_x=float(scale),
        scale_y=float(scale),
        top_indices=top_indices,
        high_threshold=float(high_threshold),
        alpha=235,
    )


def make_coordinate_heatmap(
    *,
    coords: np.ndarray,
    attention_scores: np.ndarray,
    attention_norm: np.ndarray,
    tile_size_level0: int,
    max_side: int,
    top_indices: set[int],
    blur_px: float,
    alpha: int,
    render_mode: str,
    cmap_name: str,
    convert_to_percentiles: bool,
    gamma: float,
) -> tuple[Image.Image, Image.Image]:
    coords_i = np.asarray(coords, dtype=np.int64)
    max_x = int(coords_i[:, 0].max()) + int(tile_size_level0)
    max_y = int(coords_i[:, 1].max()) + int(tile_size_level0)
    out_w, out_h, scale = scaled_canvas_size(max_x, max_y, int(max_side))
    base = Image.new("RGB", (out_w, out_h), (245, 245, 245))
    if str(render_mode) == "clam_like":
        overlay = make_clam_like_heatmap_overlay(
            base,
            coords=coords_i,
            attention_scores=attention_scores,
            tile_size_level0=int(tile_size_level0),
            scale_x=float(scale),
            scale_y=float(scale),
            blur_px=float(blur_px),
            alpha=float(alpha) / 255.0,
            cmap_name=str(cmap_name),
            convert_to_percentiles=bool(convert_to_percentiles),
            gamma=float(gamma),
        )
        return overlay, overlay.copy()
    heat_rgba = make_attention_heat_rgba(
        canvas_size=base.size,
        coords=coords_i,
        attention_norm=attention_norm,
        tile_size_level0=int(tile_size_level0),
        scale_x=float(scale),
        scale_y=float(scale),
        blur_px=float(blur_px),
        alpha=int(alpha),
    )
    heat_standalone = Image.alpha_composite(Image.new("RGBA", base.size, (245, 245, 245, 255)), heat_rgba).convert("RGB")
    overlay = Image.alpha_composite(base.convert("RGBA"), heat_rgba).convert("RGB")
    overlay = draw_top_tile_outlines(
        overlay,
        coords=coords_i,
        tile_size_level0=int(tile_size_level0),
        scale_x=float(scale),
        scale_y=float(scale),
        top_indices=top_indices,
    )
    return overlay, heat_standalone


def make_slide_thumbnail_overlay(
    *,
    slide_path: Path,
    coords: np.ndarray,
    attention_norm: np.ndarray,
    tile_size_level0: int,
    max_side: int,
    top_indices: set[int],
    high_threshold: float,
) -> tuple[Image.Image, dict[str, Any]]:
    slide = open_slide(slide_path)
    width, height = slide.dimensions
    thumb = slide.get_thumbnail((int(max_side), int(max_side))).convert("RGB")
    scale_x = float(thumb.size[0]) / float(max(1, width))
    scale_y = float(thumb.size[1]) / float(max(1, height))
    overlay = draw_attention_rectangles(
        thumb,
        coords=coords,
        attention_norm=attention_norm,
        tile_size_level0=int(tile_size_level0),
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        top_indices=top_indices,
        high_threshold=float(high_threshold),
        alpha=210,
    )
    meta = {
        "slide_width_level0": int(width),
        "slide_height_level0": int(height),
        "thumbnail_width": int(thumb.size[0]),
        "thumbnail_height": int(thumb.size[1]),
        "objective_power": float(infer_objective_power(slide)),
    }
    slide.close()
    return overlay, meta


def make_slide_thumbnail_heatmap_overlay(
    *,
    slide_path: Path,
    coords: np.ndarray,
    attention_scores: np.ndarray,
    attention_norm: np.ndarray,
    tile_size_level0: int,
    max_side: int,
    top_indices: set[int],
    blur_px: float,
    alpha: int,
    render_mode: str,
    cmap_name: str,
    convert_to_percentiles: bool,
    gamma: float,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    slide = open_slide(slide_path)
    width, height = slide.dimensions
    thumb = slide.get_thumbnail((int(max_side), int(max_side))).convert("RGB")
    scale_x = float(thumb.size[0]) / float(max(1, width))
    scale_y = float(thumb.size[1]) / float(max(1, height))
    if str(render_mode) == "clam_like":
        overlay = make_clam_like_heatmap_overlay(
            thumb,
            coords=coords,
            attention_scores=attention_scores,
            tile_size_level0=int(tile_size_level0),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            blur_px=float(blur_px),
            alpha=float(alpha) / 255.0,
            cmap_name=str(cmap_name),
            convert_to_percentiles=bool(convert_to_percentiles),
            gamma=float(gamma),
        )
        heat = overlay.copy()
    else:
        heat_rgba = make_attention_heat_rgba(
            canvas_size=thumb.size,
            coords=coords,
            attention_norm=attention_norm,
            tile_size_level0=int(tile_size_level0),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            blur_px=float(blur_px),
            alpha=int(alpha),
        )
        heat = Image.alpha_composite(Image.new("RGBA", thumb.size, (245, 245, 245, 255)), heat_rgba).convert("RGB")
        overlay = Image.alpha_composite(thumb.convert("RGBA"), heat_rgba).convert("RGB")
        overlay = draw_top_tile_outlines(
            overlay,
            coords=coords,
            tile_size_level0=int(tile_size_level0),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            top_indices=top_indices,
        )
    meta = {
        "slide_width_level0": int(width),
        "slide_height_level0": int(height),
        "thumbnail_width": int(thumb.size[0]),
        "thumbnail_height": int(thumb.size[1]),
        "objective_power": float(infer_objective_power(slide)),
    }
    slide.close()
    return overlay, heat, meta


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "experiment_args.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

    requested_slide_keys = {str(item) for item in args.slide_key}
    rows = read_split_rows(
        args.split_tsv,
        split=str(args.split),
        features_root=args.features_root,
        slides_dir=args.slides_dir,
        slide_keys=requested_slide_keys,
        label=args.label,
    )
    if int(args.max_slides_per_label) > 0 and args.label is None:
        balanced_rows: list[dict[str, Any]] = []
        for label in (0, 1):
            label_rows = [row for row in rows if int(row["label"]) == label]
            balanced_rows.extend(label_rows[: int(args.max_slides_per_label)])
        rows = sorted(balanced_rows, key=lambda item: (int(item["label"]), str(item["slide_key"])))
    if int(args.max_slides) > 0:
        rows = rows[: int(args.max_slides)]
    if not rows:
        raise RuntimeError("No slide rows matched the requested filters.")

    device = torch.device(args.device if str(args.device) != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    mil_model = build_mil_from_checkpoint(args.mil_ckpt, device=device)

    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        slide_key = str(row["slide_key"])
        slide_dir = out_dir / slide_key
        slide_dir.mkdir(parents=True, exist_ok=True)
        features, coords = read_h5_features_coords(str(row["h5_path"]))
        if coords is None:
            raise RuntimeError(f"{row['h5_path']} does not contain coords, needed for whole-slide attention visualization.")
        attention, pred, prob_pos = run_mil_attention(mil_model, features, device=device)
        order = np.argsort(-np.asarray(attention, dtype=np.float32))
        top_k = min(int(args.top_k), int(order.shape[0]))
        top_indices = {int(idx) for idx in order[:top_k].tolist()}
        attn_norm = normalize_attention(attention, percentile=float(args.attention_percentile))
        high_threshold_norm = float(np.percentile(attn_norm, float(args.attention_percentile)))

        tile_size_level0 = infer_coord_tile_size(coords, fallback=int(args.tile_size_20x))
        slide_meta: dict[str, Any] = {}
        slide_path = Path(str(row["slide_path"])) if str(row["slide_path"]) else None
        if slide_path is not None and slide_path.exists():
            try:
                slide_tmp = open_slide(slide_path)
                objective_tile_size = level0_tile_size(int(args.tile_size_20x), infer_objective_power(slide_tmp))
                slide_w, slide_h = slide_tmp.dimensions
                frame_ok, frame_reason = coords_fit_slide_frame(
                    coords=coords,
                    tile_size=int(tile_size_level0),
                    slide_w=int(slide_w),
                    slide_h=int(slide_h),
                )
                slide_meta.update(
                    {
                        "objective_tile_size_level0": int(objective_tile_size),
                        "h5_coord_tile_size": int(tile_size_level0),
                        "coord_frame_matches_local_svs": bool(frame_ok),
                        "coord_frame_check": str(frame_reason),
                    }
                )
                slide_tmp.close()
            except Exception:
                slide_meta.update(
                    {
                        "h5_coord_tile_size": int(tile_size_level0),
                        "coord_frame_matches_local_svs": False,
                        "coord_frame_check": "could_not_open_slide_for_frame_check",
                    }
                )

        coord_map = make_coordinate_attention_map(
            coords=coords,
            attention_norm=attn_norm,
            tile_size_level0=int(tile_size_level0),
            max_side=int(args.thumbnail_max_side),
            top_indices=top_indices,
            high_threshold=high_threshold_norm,
        )
        coord_map_path = slide_dir / "attention_coordinate_map.png"
        coord_map.save(coord_map_path)
        coord_heat_overlay, coord_heat = make_coordinate_heatmap(
            coords=coords,
            attention_scores=attention,
            attention_norm=attn_norm,
            tile_size_level0=int(tile_size_level0),
            max_side=int(args.thumbnail_max_side),
            top_indices=top_indices,
            blur_px=float(args.heatmap_blur_px),
            alpha=int(args.heatmap_alpha),
            render_mode=str(args.heatmap_render_mode),
            cmap_name=str(args.heatmap_cmap),
            convert_to_percentiles=bool(args.heatmap_convert_to_percentiles),
            gamma=float(args.heatmap_gamma),
        )
        coord_heatmap_path = slide_dir / "attention_coordinate_heatmap.png"
        coord_heat_overlay_path = slide_dir / "attention_coordinate_heatmap_overlay.png"
        coord_heat.save(coord_heatmap_path)
        coord_heat_overlay.save(coord_heat_overlay_path)

        overlay_path = ""
        heatmap_path = ""
        heatmap_overlay_path = ""
        if slide_path is not None and slide_path.exists():
            should_draw_overlay = bool(slide_meta.get("coord_frame_matches_local_svs", False)) or bool(args.allow_mismatched_slide_overlay)
            if should_draw_overlay:
                try:
                    overlay, slide_meta_extra = make_slide_thumbnail_overlay(
                        slide_path=slide_path,
                        coords=coords,
                        attention_norm=attn_norm,
                        tile_size_level0=int(tile_size_level0),
                        max_side=int(args.thumbnail_max_side),
                        top_indices=top_indices,
                        high_threshold=high_threshold_norm,
                    )
                    slide_meta.update(slide_meta_extra)
                    overlay_out = slide_dir / "attention_slide_overlay.png"
                    overlay.save(overlay_out)
                    overlay_path = str(overlay_out)
                    heat_overlay, heat, slide_heat_meta = make_slide_thumbnail_heatmap_overlay(
                        slide_path=slide_path,
                        coords=coords,
                        attention_scores=attention,
                        attention_norm=attn_norm,
                        tile_size_level0=int(tile_size_level0),
                        max_side=int(args.thumbnail_max_side),
                        top_indices=top_indices,
                        blur_px=float(args.heatmap_blur_px),
                        alpha=int(args.heatmap_alpha),
                        render_mode=str(args.heatmap_render_mode),
                        cmap_name=str(args.heatmap_cmap),
                        convert_to_percentiles=bool(args.heatmap_convert_to_percentiles),
                        gamma=float(args.heatmap_gamma),
                    )
                    slide_meta.update(slide_heat_meta)
                    heatmap_out = slide_dir / "attention_heatmap.png"
                    heatmap_overlay_out = slide_dir / "attention_heatmap_overlay.png"
                    heat.save(heatmap_out)
                    heat_overlay.save(heatmap_overlay_out)
                    heatmap_path = str(heatmap_out)
                    heatmap_overlay_path = str(heatmap_overlay_out)
                except Exception as exc:
                    slide_meta["overlay_error"] = str(exc)
            else:
                slide_meta["overlay_skipped_reason"] = str(slide_meta.get("coord_frame_check", "coord_frame_mismatch"))

        top_rows: list[dict[str, Any]] = []
        for rank, idx in enumerate(order[:top_k].tolist(), start=1):
            top_rows.append(
                {
                    "rank": int(rank),
                    "tile_index": int(idx),
                    "attention": float(attention[int(idx)]),
                    "coord_x": int(coords[int(idx), 0]),
                    "coord_y": int(coords[int(idx), 1]),
                }
            )
        write_csv(slide_dir / "top_attention_tiles.csv", top_rows)

        meta = {
            "slide_key": slide_key,
            "case_id": str(row["case_id"]),
            "split": str(row["split"]),
            "label": int(row["label"]),
            "pred": int(pred),
            "prob_pos": float(prob_pos),
            "label_match": bool(int(pred) == int(row["label"])),
            "n_tiles": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "tile_size_level0": int(tile_size_level0),
            "h5_coord_tile_size": int(tile_size_level0),
            "attention_percentile": float(args.attention_percentile),
            "attention_threshold": float(np.percentile(attention, float(args.attention_percentile))),
            "top_k": int(top_k),
            "heatmap_render_mode": str(args.heatmap_render_mode),
            "heatmap_cmap": str(args.heatmap_cmap),
            "h5_path": str(row["h5_path"]),
            "slide_path": str(row["slide_path"]),
            "coordinate_map": str(coord_map_path),
            "coordinate_heatmap": str(coord_heatmap_path),
            "coordinate_heatmap_overlay": str(coord_heat_overlay_path),
            "slide_overlay": overlay_path,
            "slide_heatmap": heatmap_path,
            "slide_heatmap_overlay": heatmap_overlay_path,
            **slide_meta,
        }
        write_json(slide_dir / "attention_summary.json", meta)
        summary_rows.append(meta)

    write_csv(out_dir / "slide_attention_summary.csv", summary_rows)
    print(f"[ok] wrote attention visualizations for {len(summary_rows)} slides to {out_dir}")


if __name__ == "__main__":
    main()
