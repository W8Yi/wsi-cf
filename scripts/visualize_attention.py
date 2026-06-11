#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.common.paths import (
    DEFAULT_HNSCC_CLAM_CKPT,
    DEFAULT_HNSCC_CLAM_COORDS_H5_DIR,
    DEFAULT_HNSCC_CLAM_DATASET_CSV,
    DEFAULT_HNSCC_CLAM_FEATURES_PT_DIR,
    DEFAULT_HNSCC_CLAM_SLIDES_DIR,
    DEFAULT_HNSCC_CLAM_SPLITS_CSV,
    DEFAULT_HNSCC_FEATURES_ROOT,
    DEFAULT_HNSCC_MIL_CKPT,
    DEFAULT_HNSCC_SLIDES_DIR,
    DEFAULT_HNSCC_SPLIT_TSV,
)
from wsi_cf.data.donor_pool import load_split_rows
from wsi_cf.data.slides import find_slide_path, infer_objective_power, open_slide
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, run_mil_attention


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize whole-slide CLAM attention using CLAM UNI2 feature pt-files, "
            "patch coordinate H5s, and the CLAM slide directory."
        )
    )
    parser.add_argument("--backend", type=str, default="clam", choices=["clam", "mil"])
    parser.add_argument("--task", type=str, default="hnscc_hpv")
    parser.add_argument("--splits-csv", type=Path, default=DEFAULT_HNSCC_CLAM_SPLITS_CSV)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_HNSCC_CLAM_DATASET_CSV)
    parser.add_argument("--features-pt-dir", type=Path, default=DEFAULT_HNSCC_CLAM_FEATURES_PT_DIR)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_HNSCC_FEATURES_ROOT)
    parser.add_argument("--coords-h5-dir", type=Path, default=DEFAULT_HNSCC_CLAM_COORDS_H5_DIR)
    parser.add_argument("--slides-dir", type=Path, default=DEFAULT_HNSCC_CLAM_SLIDES_DIR)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_HNSCC_CLAM_CKPT)
    parser.add_argument("--mil-ckpt", type=Path, default=DEFAULT_HNSCC_MIL_CKPT)
    parser.add_argument("--split-tsv", type=Path, default=DEFAULT_HNSCC_SPLIT_TSV)
    parser.add_argument("--out-dir", type=Path, default=Path("/common/users/wq50/wsi_cf/artifacts/clam_attention_vis"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--slide-id", action="append", default=[], help="Optional exact CLAM slide_id filter.")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--max-slides", type=int, default=8)
    parser.add_argument("--max-slides-per-label", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--attention-percentile", type=float, default=90.0)
    parser.add_argument("--thumbnail-max-side", type=int, default=3072)
    parser.add_argument("--heatmap-blur-px", type=float, default=5.0)
    parser.add_argument("--heatmap-alpha", type=int, default=170)
    parser.add_argument("--heatmap-gamma", type=float, default=0.75, help="Values < 1 brighten low/mid attention for smoother transitions.")
    parser.add_argument("--heatmap-min-alpha", type=int, default=95)
    parser.add_argument("--heatmap-render-mode", type=str, default="clam_like", choices=["clam_like", "tile_overlay"])
    parser.add_argument("--heatmap-cmap", type=str, default="coolwarm")
    parser.add_argument("--heatmap-convert-to-percentiles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-class", type=str, default="pred", choices=["pred", "pos", "neg"])
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def hpv_label_from_status(status: str) -> int:
    s = str(status).strip().upper()
    if s == "HPV+":
        return 1
    if s == "HPV-":
        return 0
    raise ValueError(f"Unexpected hpv_status: {status}")


def load_dataset_labels(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slide_id = str(row.get("slide_id", "")).strip()
            if not slide_id:
                continue
            out[slide_id] = {
                "case_id": str(row.get("case_id", slide_id)),
                "slide_id": slide_id,
                "hpv_status": str(row.get("hpv_status", "")),
                "label": hpv_label_from_status(str(row.get("hpv_status", ""))),
            }
    return out


def read_split_slide_ids(path: Path, split_name: str) -> list[str]:
    out: list[str] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = str(row.get(split_name, "")).strip()
            if sid:
                out.append(sid)
    return out


def load_clam_mb(ckpt_path: Path, device: torch.device):
    from wsi_cf.models.clam import CLAM_MB

    model = CLAM_MB(gate=True, size_arg="small", n_classes=2, embed_dim=1536)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_clam_attention(model, features: torch.Tensor, *, attn_class: str) -> tuple[np.ndarray, int, float]:
    logits, y_prob, y_hat, a_raw, _ = model(features)
    a = F.softmax(a_raw, dim=1)
    pred = int(y_hat.item())
    prob_pos = float(y_prob[0, 1].item())
    if str(attn_class) == "pred":
        row = pred
    elif str(attn_class) == "pos":
        row = 1
    else:
        row = 0
    attn = a[row].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    return attn, pred, prob_pos


def attention_color(value: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, float(value)))
    anchors = [
        (0.0, (38, 64, 230)),
        (0.25, (62, 196, 245)),
        (0.55, (236, 246, 112)),
        (0.82, (241, 132, 68)),
        (1.0, (120, 18, 26)),
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
    candidates: list[int] = []
    for axis in (0, 1):
        vals = np.unique(arr[:, axis])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    return int(min(candidates)) if candidates else int(fallback)


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
    gamma: float,
    min_alpha: int,
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
    valid_img = Image.fromarray(np.clip(valid * 255.0, 0, 255).astype(np.uint8), mode="L")
    if float(blur_px) > 0:
        heat_img = heat_img.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    heat_arr = np.asarray(heat_img, dtype=np.uint8)
    valid_arr = np.asarray(valid_img, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if valid_arr[y, x] == 0:
                continue
            score = float(heat_arr[y, x]) / 255.0
            score = float(np.power(max(0.0, min(1.0, score)), float(gamma)))
            color = attention_color(score)
            rgba[y, x, :3] = color
            rgba[y, x, 3] = max(int(min_alpha), int(round(float(alpha) * score)))
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
        if smax > smin:
            scores = (scores - smin) / (smax - smin)
        else:
            scores = np.zeros_like(scores, dtype=np.float32)
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
    overlay = np.clip(overlay, 0.0, 1.0)

    cmap = plt.get_cmap(str(cmap_name))
    color_rgb = (cmap(overlay)[..., :3] * 255.0).astype(np.uint8)
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


def read_pt_features(path: Path) -> np.ndarray:
    feats = torch.load(path, map_location="cpu")
    if isinstance(feats, torch.Tensor):
        return feats.detach().cpu().float().numpy().astype(np.float32, copy=False)
    raise TypeError(f"Unexpected pt feature payload in {path}: {type(feats)}")


def read_patch_coords(path: Path) -> tuple[np.ndarray, int]:
    with h5py.File(path, "r") as f:
        coords = np.asarray(f["coords"][:], dtype=np.int64)
        patch_size = int(f["coords"].attrs.get("patch_size", 256))
    return coords, patch_size


def read_h5_features_coords(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(path, "r") as f:
        feats = np.asarray(f["features"][:], dtype=np.float32)
        coords = np.asarray(f["coords"][:], dtype=np.int64)
        patch_size = int(f["coords"].attrs.get("patch_size", 256)) if "coords" in f else 256
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    return feats.astype(np.float32, copy=False), coords, patch_size


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device if str(args.device) != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    slide_id_filter = {str(x) for x in args.slide_id}
    rows: list[dict[str, Any]] = []
    per_label_seen = {0: 0, 1: 0}
    if str(args.backend) == "clam":
        labels_by_slide = load_dataset_labels(args.dataset_csv)
        split_ids = read_split_slide_ids(args.splits_csv, str(args.split))
        for slide_id in split_ids:
            if slide_id_filter and slide_id not in slide_id_filter:
                continue
            meta = labels_by_slide.get(slide_id)
            if meta is None:
                continue
            label = int(meta["label"])
            if args.label is not None and label != int(args.label):
                continue
            if int(args.max_slides_per_label) > 0 and per_label_seen[label] >= int(args.max_slides_per_label):
                continue
            feat_path = args.features_pt_dir / f"{slide_id}.pt"
            coord_path = args.coords_h5_dir / f"{slide_id}.h5"
            slide_path = find_slide_path(args.slides_dir, slide_id)
            if not feat_path.exists() or not coord_path.exists() or slide_path is None:
                continue
            per_label_seen[label] += 1
            rows.append({**meta, "feat_path": str(feat_path), "coord_path": str(coord_path), "slide_path": str(slide_path)})
            if int(args.max_slides) > 0 and len(rows) >= int(args.max_slides):
                break
        model = load_clam_mb(args.ckpt, device=device)
    else:
        args.slides_dir = DEFAULT_HNSCC_SLIDES_DIR if args.slides_dir == DEFAULT_HNSCC_CLAM_SLIDES_DIR else args.slides_dir
        for item in load_split_rows(args.split_tsv, split_filter=str(args.split)):
            slide_id = str(item["slide_key"])
            if slide_id_filter and slide_id not in slide_id_filter:
                continue
            label = int(item["label"])
            if args.label is not None and label != int(args.label):
                continue
            if int(args.max_slides_per_label) > 0 and per_label_seen[label] >= int(args.max_slides_per_label):
                continue
            feat_path = args.features_root / f"{slide_id}.h5"
            slide_path = find_slide_path(args.slides_dir, slide_id)
            if not feat_path.exists() or slide_path is None:
                continue
            per_label_seen[label] += 1
            rows.append(
                {
                    "case_id": str(item.get("case_id", slide_id)),
                    "slide_id": slide_id,
                    "hpv_status": "HPV+" if label == 1 else "HPV-",
                    "label": label,
                    "feat_path": str(feat_path),
                    "coord_path": str(feat_path),
                    "slide_path": str(slide_path),
                }
            )
            if int(args.max_slides) > 0 and len(rows) >= int(args.max_slides):
                break
        model = build_mil_from_checkpoint(args.mil_ckpt, device=device)
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        slide_id = str(row["slide_id"])
        slide_dir = args.out_dir / slide_id
        slide_dir.mkdir(parents=True, exist_ok=True)
        if str(args.backend) == "clam":
            features = read_pt_features(Path(row["feat_path"]))
            coords, patch_size = read_patch_coords(Path(row["coord_path"]))
        else:
            features, coords, patch_size = read_h5_features_coords(Path(row["feat_path"]))
        if int(features.shape[0]) != int(coords.shape[0]):
            raise RuntimeError(f"{slide_id}: features n={features.shape[0]} but coords n={coords.shape[0]}")
        if str(args.backend) == "clam":
            feats_t = torch.from_numpy(features).to(device=device, dtype=torch.float32)
            attention, pred, prob_pos = run_clam_attention(model, feats_t, attn_class=str(args.attn_class))
        else:
            attention, pred, prob_pos = run_mil_attention(model, features, device=device)
        order = np.argsort(-np.asarray(attention, dtype=np.float32))
        top_indices = set(int(i) for i in order[: int(args.top_k)].tolist())
        attn_norm = normalize_attention(attention, percentile=float(args.attention_percentile))
        high_threshold_norm = float(np.percentile(attn_norm, float(args.attention_percentile)))

        slide = open_slide(Path(row["slide_path"]))
        try:
            slide_w, slide_h = slide.dimensions
            req_w, req_h, scale = scaled_canvas_size(slide_w, slide_h, int(args.thumbnail_max_side))
            thumbnail = slide.get_thumbnail((req_w, req_h)).convert("RGB")
            thumb_w, thumb_h = thumbnail.size
            objective_power = infer_objective_power(slide)
        finally:
            slide.close()
        tile_size_level0 = infer_coord_tile_size(coords, fallback=int(patch_size))
        scale_x = float(thumb_w) / float(slide_w)
        scale_y = float(thumb_h) / float(slide_h)

        overlay = draw_attention_rectangles(
            thumbnail,
            coords=coords,
            attention_norm=attn_norm,
            tile_size_level0=int(tile_size_level0),
            scale_x=scale_x,
            scale_y=scale_y,
            top_indices=top_indices,
            high_threshold=float(high_threshold_norm),
            alpha=int(args.heatmap_alpha),
        )
        overlay_out = slide_dir / "attention_slide_overlay.png"
        overlay.save(overlay_out)

        heatmap_out = slide_dir / "attention_heatmap.png"
        heatmap_overlay_out = slide_dir / "attention_heatmap_overlay.png"
        if str(args.heatmap_render_mode) == "clam_like":
            heat_overlay = make_clam_like_heatmap_overlay(
                thumbnail,
                coords=coords,
                attention_scores=attention,
                tile_size_level0=int(tile_size_level0),
                scale_x=scale_x,
                scale_y=scale_y,
                blur_px=float(args.heatmap_blur_px),
                alpha=float(args.heatmap_alpha) / 255.0,
                cmap_name=str(args.heatmap_cmap),
                convert_to_percentiles=bool(args.heatmap_convert_to_percentiles),
                gamma=float(args.heatmap_gamma),
            )
            heat_overlay.save(heatmap_overlay_out)
            heat_overlay.save(heatmap_out)
        else:
            heat_rgba = make_attention_heat_rgba(
                canvas_size=(thumb_w, thumb_h),
                coords=coords,
                attention_norm=attn_norm,
                tile_size_level0=int(tile_size_level0),
                scale_x=scale_x,
                scale_y=scale_y,
                blur_px=float(args.heatmap_blur_px),
                alpha=int(args.heatmap_alpha),
                gamma=float(args.heatmap_gamma),
                min_alpha=int(args.heatmap_min_alpha),
            )
            heat_rgba.save(heatmap_out)
            Image.alpha_composite(thumbnail.convert("RGBA"), heat_rgba).convert("RGB").save(heatmap_overlay_out)

        top_rows = []
        for rank, idx in enumerate(order[: int(args.top_k)].tolist(), start=1):
            top_rows.append(
                {
                    "rank": int(rank),
                    "tile_index": int(idx),
                    "attention": float(attention[int(idx)]),
                    "coord_x": int(coords[int(idx), 0]),
                    "coord_y": int(coords[int(idx), 1]),
                }
            )
        with (slide_dir / "top_attention_tiles.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(top_rows[0].keys()) if top_rows else [])
            if top_rows:
                writer.writeheader()
                writer.writerows(top_rows)

        meta = {
            "slide_id": slide_id,
            "case_id": str(row["case_id"]),
            "label": int(row["label"]),
            "hpv_status": str(row["hpv_status"]),
            "pred": int(pred),
            "prob_pos": float(prob_pos),
            "label_match": bool(int(pred) == int(row["label"])),
            "n_tiles": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "coord_tile_size_level0": int(tile_size_level0),
            "patch_h5_patch_size": int(patch_size),
            "attention_percentile": float(args.attention_percentile),
            "attention_threshold": float(np.percentile(attention, float(args.attention_percentile))),
            "top_k": int(args.top_k),
            "feat_path": str(row["feat_path"]),
            "coord_path": str(row["coord_path"]),
            "slide_path": str(row["slide_path"]),
            "slide_overlay": str(overlay_out),
            "slide_heatmap": str(heatmap_out),
            "slide_heatmap_overlay": str(heatmap_overlay_out),
            "slide_width_level0": int(slide_w),
            "slide_height_level0": int(slide_h),
            "thumbnail_width": int(thumb_w),
            "thumbnail_height": int(thumb_h),
            "objective_power": float(objective_power),
            "attn_class": str(args.attn_class),
            "heatmap_render_mode": str(args.heatmap_render_mode),
            "heatmap_cmap": str(args.heatmap_cmap),
        }
        write_json(slide_dir / "attention_summary.json", meta)
        summary_rows.append(meta)

    with (args.out_dir / "slide_attention_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    write_json(
        args.out_dir / "run_summary.json",
        {
            "n_slides": int(len(summary_rows)),
            "split": str(args.split),
            "attn_class": str(args.attn_class),
            "slides": summary_rows,
        },
    )
    print(f"[ok] wrote CLAM attention visualizations for {len(summary_rows)} slides to {args.out_dir}")


if __name__ == "__main__":
    main()
