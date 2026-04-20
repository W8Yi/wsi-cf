#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shlex
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png, write_json
from wsi_cf.common.paths import ensure_legacy_repo_root_on_path
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.slides import find_slide_path, open_slide, read_region_rgb_at_magnification
from wsi_cf.eval.hnsc_hpv import load_prototypes, pick_prototype_latent
from wsi_cf.generation.pixcell import (
    build_uni_grid_from_image,
    load_uni2,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
    vae_encode_auto,
)
from wsi_cf.steering.cell_selection import encode_cells

ensure_legacy_repo_root_on_path()

from utils.sae import load_sae_from_config  # type: ignore
from utils.sae_edit import edit_uni_z_grid_with_sae  # type: ignore


PATTERN_NAMES = ["edge_1cell", "edge_2cell", "corner_L", "full_edge_band"]


def _jsonify(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _serialize_args(args: argparse.Namespace) -> dict[str, object]:
    return {str(k): _jsonify(v) for k, v in vars(args).items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 4096x4096 seam stress test with overlapping 1024x1024 PixCell windows. "
            "Each window is steered using edge-adjacent cell patterns, then the generated windows "
            "are stitched with multiple policies to evaluate seam robustness."
        )
    )
    parser.add_argument("--input-svs", type=Path, default=None, help="Optional direct slide path")
    parser.add_argument("--slide-key", type=str, default="TCGA-BB-4225-01Z-00-DX1")
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/HNSCC_slides"))
    parser.add_argument("--x", type=int, default=0, help="Top-left region x at level 0")
    parser.add_argument("--y", type=int, default=0, help="Top-left region y at level 0")
    parser.add_argument("--target-magnification", type=float, default=10.0)
    parser.add_argument("--canvas-size", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--edit-layout", type=str, default="seam_patterns", choices=["seam_patterns", "center_area"])
    parser.add_argument("--patterns", type=str, default="all", help="Comma-separated seam stress patterns or 'all'")
    parser.add_argument("--center-area-size", type=int, default=2048, help="Central square size in pixels when --edit-layout center_area is used")
    parser.add_argument("--direction", type=str, default="hpv_pos", choices=["hpv_pos", "hpv_neg"])
    parser.add_argument("--sae-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt"))
    parser.add_argument("--sae-cfg", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json"))
    parser.add_argument("--prototype-npz", type=Path, default=Path("/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"))
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--pix_model_id", type=str, default="StonyBrook-CVLab/PixCell-1024")
    parser.add_argument("--pix_pipeline_id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae_model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--preserve-outside-latents", action="store_true", default=True)
    parser.add_argument("--preserve-outside-strength", type=float, default=0.85)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.5)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=0.0)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--trusted-margin-px", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/seam_stress_4096")
    return parser


def parse_patterns(value: str) -> list[str]:
    if str(value).strip().lower() == "all":
        return list(PATTERN_NAMES)
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    bad = [item for item in items if item not in PATTERN_NAMES]
    if bad:
        raise ValueError(f"Unsupported patterns: {bad}")
    return items


def resolve_slide_path(args: argparse.Namespace) -> Path:
    if args.input_svs is not None:
        return args.input_svs
    slide_path = find_slide_path(args.slides_dir, str(args.slide_key))
    if slide_path is None:
        raise FileNotFoundError(f"Could not resolve slide for slide-key={args.slide_key}")
    return slide_path


def make_window_starts(total: int, window: int, stride: int) -> list[int]:
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be > 0")
    if total <= window:
        return [0]
    starts = list(range(0, total - window + 1, stride))
    if starts[-1] != total - window:
        starts.append(total - window)
    return starts


def enumerate_windows(canvas_size: int, window_size: int, stride: int) -> list[dict[str, int]]:
    xs = make_window_starts(int(canvas_size), int(window_size), int(stride))
    ys = make_window_starts(int(canvas_size), int(window_size), int(stride))
    windows: list[dict[str, int]] = []
    idx = 0
    for row_idx, top in enumerate(ys):
        for col_idx, left in enumerate(xs):
            windows.append(
                {
                    "window_index": idx,
                    "row_index": row_idx,
                    "col_index": col_idx,
                    "left": int(left),
                    "top": int(top),
                }
            )
            idx += 1
    return windows


def selected_cells_for_pattern(pattern: str, *, variant: int) -> list[tuple[int, int]]:
    v = int(variant) % 4
    if pattern == "edge_1cell":
        options = [[(0, 1)], [(3, 1)], [(1, 0)], [(1, 3)]]
        return options[v]
    if pattern == "edge_2cell":
        options = [
            [(0, 1), (0, 2)],
            [(3, 1), (3, 2)],
            [(1, 0), (2, 0)],
            [(1, 3), (2, 3)],
        ]
        return options[v]
    if pattern == "corner_L":
        options = [
            [(0, 0), (1, 0), (0, 1)],
            [(3, 0), (2, 0), (3, 1)],
            [(0, 3), (0, 2), (1, 3)],
            [(3, 3), (2, 3), (3, 2)],
        ]
        return options[v]
    if pattern == "full_edge_band":
        options = [
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(3, 0), (3, 1), (3, 2), (3, 3)],
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 3), (1, 3), (2, 3), (3, 3)],
        ]
        return options[v]
    raise ValueError(f"Unsupported pattern: {pattern}")


def assign_window_patterns(windows: list[dict[str, int]], patterns: list[str]) -> list[dict[str, object]]:
    assignments: list[dict[str, object]] = []
    for window in windows:
        idx = int(window["window_index"])
        pattern = patterns[idx % len(patterns)]
        variant = idx // max(1, len(patterns))
        cells = selected_cells_for_pattern(pattern, variant=variant)
        assignments.append({**window, "pattern": pattern, "selected_cells": cells})
    return assignments


def center_area_bounds(*, canvas_size: int, area_size: int) -> tuple[int, int, int, int]:
    area = min(int(area_size), int(canvas_size))
    left = max(0, (int(canvas_size) - area) // 2)
    top = max(0, (int(canvas_size) - area) // 2)
    return left, top, left + area, top + area


def assign_center_area_cells(
    windows: list[dict[str, int]],
    *,
    canvas_size: int,
    area_size: int,
    grid_step_px: int,
) -> tuple[list[dict[str, object]], tuple[int, int, int, int]]:
    x0, y0, x1, y1 = center_area_bounds(canvas_size=int(canvas_size), area_size=int(area_size))
    assignments: list[dict[str, object]] = []
    for window in windows:
        left = int(window["left"])
        top = int(window["top"])
        cells: list[tuple[int, int]] = []
        for gy in range(4):
            for gx in range(4):
                cx = left + int((gx + 0.5) * int(grid_step_px))
                cy = top + int((gy + 0.5) * int(grid_step_px))
                if x0 <= cx < x1 and y0 <= cy < y1:
                    cells.append((gx, gy))
        assignments.append({**window, "pattern": "center_area", "selected_cells": cells})
    return assignments, (x0, y0, x1, y1)


def draw_selected_cells_overlay(img: Image.Image, *, cells: list[tuple[int, int]], grid_step_px: int) -> Image.Image:
    canvas = img.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for idx, (gx, gy) in enumerate(cells, start=1):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(canvas.size[0] - 1, x0 + int(grid_step_px) - 1)
        y1 = min(canvas.size[1] - 1, y0 + int(grid_step_px) - 1)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=6)
        draw.text((x0 + 8, y0 + 8), str(idx), fill=(255, 255, 0))
    return canvas


def draw_global_edit_overlay(
    img: Image.Image,
    *,
    window_rows: list[dict[str, object]],
    grid_step_px: int,
) -> Image.Image:
    canvas = img.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for row in window_rows:
        left = int(row["left"])
        top = int(row["top"])
        cells = row.get("selected_cells_decoded", [])
        for idx, (gx, gy) in enumerate(cells, start=1):
            x0 = left + int(gx) * int(grid_step_px)
            y0 = top + int(gy) * int(grid_step_px)
            x1 = x0 + int(grid_step_px) - 1
            y1 = y0 + int(grid_step_px) - 1
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
            draw.text((x0 + 6, y0 + 6), str(idx), fill=(255, 255, 0))
    return canvas


def draw_global_area_box(img: Image.Image, *, bounds: tuple[int, int, int, int]) -> Image.Image:
    canvas = img.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = bounds
    draw.rectangle([int(x0), int(y0), int(x1) - 1, int(y1) - 1], outline=(0, 255, 255), width=6)
    draw.text((int(x0) + 8, int(y0) + 8), "edit area", fill=(255, 255, 0))
    return canvas


def make_edit_region_mask(*, width: int, height: int, cells: list[tuple[int, int]], grid_step_px: int) -> torch.Tensor:
    mask = torch.zeros((1, 1, int(height), int(width)), dtype=torch.float32)
    for gx, gy in cells:
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(int(width), x0 + int(grid_step_px))
        y1 = min(int(height), y0 + int(grid_step_px))
        mask[:, :, y0:y1, x0:x1] = 1.0
    return mask


def build_center_weight_mask(size: int, *, mode: str, trusted_margin_px: int) -> np.ndarray:
    if mode == "hard_stitch":
        return np.ones((int(size), int(size)), dtype=np.float32)
    if mode == "overlap_average":
        return np.ones((int(size), int(size)), dtype=np.float32)
    if mode == "center_weighted_blend":
        yy = np.linspace(-1.0, 1.0, int(size), dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, int(size), dtype=np.float32)[None, :]
        win = np.exp(-2.0 * (yy**2 + xx**2))
        win = win / max(float(win.max()), 1e-8)
        return win.astype(np.float32)
    if mode == "trusted_center_only":
        margin = max(0, int(trusted_margin_px))
        mask = np.zeros((int(size), int(size)), dtype=np.float32)
        if margin * 2 >= int(size):
            mask[:, :] = 1.0
        else:
            mask[margin : int(size) - margin, margin : int(size) - margin] = 1.0
        return mask
    raise ValueError(f"Unsupported stitch mode: {mode}")


def stitch_windows(
    *,
    window_outputs: list[dict[str, object]],
    canvas_size: int,
    window_size: int,
    mode: str,
    trusted_margin_px: int,
) -> np.ndarray:
    out = np.zeros((int(canvas_size), int(canvas_size), 3), dtype=np.float32)
    wgt = np.zeros((int(canvas_size), int(canvas_size), 1), dtype=np.float32)
    mask2d = build_center_weight_mask(int(window_size), mode=mode, trusted_margin_px=int(trusted_margin_px))
    mask3d = mask2d[..., None]
    for row in window_outputs:
        left = int(row["left"])
        top = int(row["top"])
        img = np.asarray(row["steered_img"], dtype=np.float32) / 255.0
        out[top : top + int(window_size), left : left + int(window_size)] += img * mask3d
        wgt[top : top + int(window_size), left : left + int(window_size)] += mask3d
    return np.clip(out / np.clip(wgt, 1e-8, None), 0.0, 1.0)


def compute_seam_metrics(*, stitched: np.ndarray, stride: int, window_size: int, canvas_size: int, seam_band_px: int = 16) -> dict[str, float]:
    arr = np.asarray(stitched, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("stitched must have shape [H,W,3]")
    lines = make_window_starts(int(canvas_size), int(window_size), int(stride))[1:]
    vertical_jumps = []
    horizontal_jumps = []
    seam_band_means = []
    band = max(1, int(seam_band_px))
    for x in lines:
        if 1 <= int(x) < arr.shape[1]:
            left = arr[:, int(x) - 1, :]
            right = arr[:, int(x), :]
            vertical_jumps.append(float(np.mean(np.abs(left - right))))
            x0 = max(0, int(x) - band)
            x1 = min(arr.shape[1], int(x) + band)
            seam_band_means.append(float(np.mean(arr[:, x0:x1, :])))
    for y in lines:
        if 1 <= int(y) < arr.shape[0]:
            up = arr[int(y) - 1, :, :]
            down = arr[int(y), :, :]
            horizontal_jumps.append(float(np.mean(np.abs(up - down))))
            y0 = max(0, int(y) - band)
            y1 = min(arr.shape[0], int(y) + band)
            seam_band_means.append(float(np.mean(arr[y0:y1, :, :])))
    return {
        "mean_vertical_jump_l1": float(np.mean(vertical_jumps)) if vertical_jumps else 0.0,
        "mean_horizontal_jump_l1": float(np.mean(horizontal_jumps)) if horizontal_jumps else 0.0,
        "mean_seam_jump_l1": float(np.mean(vertical_jumps + horizontal_jumps)) if (vertical_jumps or horizontal_jumps) else 0.0,
        "mean_seam_band_intensity": float(np.mean(seam_band_means)) if seam_band_means else 0.0,
    }


def save_rows_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "cli_args": _serialize_args(args),
        "argv": list(argv) if argv is not None else list(sys.argv[1:]),
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else list(sys.argv[1:])))),
        "cwd": str(Path.cwd()),
    }
    write_json(args.out_dir / "experiment_args.json", args_payload)

    slide_path = resolve_slide_path(args)
    slide = open_slide(slide_path)
    try:
        canvas_img, _, _ = read_region_rgb_at_magnification(
            slide,
            x0=int(args.x),
            y0=int(args.y),
            out_w=int(args.canvas_size),
            out_h=int(args.canvas_size),
            target_magnification=float(args.target_magnification),
        )
    finally:
        slide.close()
    save_png(canvas_img, args.out_dir / "source_canvas_actual.png")

    windows_base = enumerate_windows(int(args.canvas_size), int(args.window_size), int(args.window_stride))
    edit_area_bounds = None
    if str(args.edit_layout) == "center_area":
        patterns = ["center_area"]
        windows, edit_area_bounds = assign_center_area_cells(
            windows_base,
            canvas_size=int(args.canvas_size),
            area_size=int(args.center_area_size),
            grid_step_px=int(args.grid_step_px),
        )
    else:
        patterns = parse_patterns(args.patterns)
        windows = assign_window_patterns(windows_base, patterns)

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    chosen_latent = int(pos_latent if str(args.direction) == "hpv_pos" else neg_latent)
    prototype_vec = proto_by_latent[chosen_latent]
    uni_model, uni_transform = load_uni2(device=device)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=AutoencoderKL.from_pretrained(args.vae_model_id, subfolder=args.vae_subfolder, torch_dtype=dtype),
        custom_pipeline=args.pix_pipeline_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_px, stride_px, cond_grid_side = resolve_pixcell_window_config(pix_model_id=args.pix_model_id, patch_px=0, stride_px=0)

    window_rows: list[dict[str, object]] = []
    generated_windows: list[dict[str, object]] = []
    for row in windows:
        left = int(row["left"])
        top = int(row["top"])
        pattern = str(row["pattern"])
        selected_cells = [(int(gx), int(gy)) for gx, gy in row["selected_cells"]]
        window_img = canvas_img.crop((left, top, left + int(args.window_size), top + int(args.window_size))).convert("RGB")
        if not selected_cells and str(args.edit_layout) == "center_area":
            steered_img = window_img.copy()
            pattern = "center_area_outside"
            source_path = (args.out_dir / "windows" / f"window_{int(row['window_index']):03d}" / "source_window.png")
            overlay_path = (args.out_dir / "windows" / f"window_{int(row['window_index']):03d}" / "selected_cells_overlay.png")
            steered_path = (args.out_dir / "windows" / f"window_{int(row['window_index']):03d}" / "steered_window.png")
            source_path.parent.mkdir(parents=True, exist_ok=True)
            save_png(window_img, source_path)
            save_png(draw_selected_cells_overlay(window_img, cells=selected_cells, grid_step_px=int(args.grid_step_px)), overlay_path)
            save_png(steered_img, steered_path)
            window_rows.append(
                {
                    "window_index": int(row["window_index"]),
                    "row_index": int(row["row_index"]),
                    "col_index": int(row["col_index"]),
                    "left": int(left),
                    "top": int(top),
                    "pattern": pattern,
                    "selected_cells": encode_cells(selected_cells),
                    "selected_cells_decoded": selected_cells,
                    "selected_cell_count": 0,
                    "source_path": str(source_path),
                    "overlay_path": str(overlay_path),
                    "steered_path": str(steered_path),
                }
            )
            generated_windows.append(
                {
                    "window_index": int(row["window_index"]),
                    "left": int(left),
                    "top": int(top),
                    "steered_img": np.asarray(steered_img, dtype=np.uint8),
                }
            )
            print(f"[ok] window {int(row['window_index']):03d} pattern={pattern} cells=")
            continue
        source_zgrid = build_uni_grid_from_image(
            window_img,
            uni_model=uni_model,
            uni_transform=uni_transform,
            grid_step_px=int(args.grid_step_px),
            device=device,
            out_dtype=dtype,
        )
        tile_mask = np.zeros(tuple(source_zgrid.shape[:2]), dtype=np.float32)
        for gx, gy in selected_cells:
            tile_mask[int(gy), int(gx)] = 1.0
        steered_zgrid_t, _ = edit_uni_z_grid_with_sae(
            sae_model=sae_model,
            z_grid=source_zgrid.to(device=device, dtype=torch.float32),
            target_latent_vector=prototype_vec,
            target_latent_vector_strength=float(args.prototype_strength),
            tile_mask=tile_mask,
            blend=float(args.steer_blend),
            keep_non_selected=True,
            return_debug=False,
        )
        z_grid_base_pix = source_zgrid.to(device=device, dtype=dtype)
        z_grid_edit_pix = steered_zgrid_t.to(device=device, dtype=dtype)

        preserve_source_latents = None
        edit_region_mask = None
        if bool(args.preserve_outside_latents) and selected_cells:
            source_np = np.asarray(window_img, dtype=np.float32) / 255.0
            source_img_t = torch.from_numpy(source_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
            preserve_source_latents = vae_encode_auto(
                pipeline.vae,
                source_img_t,
                use_tiled=False,
                tile_img=0,
                overlap_img=0,
            )
            edit_region_mask = make_edit_region_mask(
                width=int(window_img.size[0]),
                height=int(window_img.size[1]),
                cells=selected_cells,
                grid_step_px=int(args.grid_step_px),
            ).to(device=device)

        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed) + int(row["window_index"]))
        use_autocast = device.type == "cuda" and dtype == torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
        with torch.inference_mode(), ctx:
            steered_img_t = sample_large_pixcell_multidiffusion(
                pipeline=pipeline,
                z_grid=z_grid_base_pix,
                scheduled_z_grid=z_grid_edit_pix,
                condition_start_ratio=float(args.mid_steer_start_ratio),
                condition_end_ratio=float(args.mid_steer_end_ratio),
                condition_alpha_start=float(args.mid_steer_alpha_start),
                condition_alpha_end=float(args.mid_steer_alpha_end),
                condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
                out_h=int(args.window_size),
                out_w=int(args.window_size),
                patch_px=patch_px,
                stride_px=stride_px,
                cond_grid_side=cond_grid_side,
                guidance_scale=float(args.guidance),
                num_steps=int(args.steps),
                patch_batch=int(args.patch_batch),
                strength=0.0,
                init_latents=None,
                preserve_source_latents=preserve_source_latents,
                edit_region_mask=edit_region_mask,
                preserve_outside_strength=float(args.preserve_outside_strength),
                use_tiled_vae_decode=False,
                decode_tile_lat=128,
                decode_overlap_lat=16,
                generator=generator,
            )
        steered_img = Image.fromarray((steered_img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))

        window_dir = args.out_dir / "windows" / f"window_{int(row['window_index']):03d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        source_path = window_dir / "source_window.png"
        overlay_path = window_dir / "selected_cells_overlay.png"
        steered_path = window_dir / "steered_window.png"
        save_png(window_img, source_path)
        save_png(draw_selected_cells_overlay(window_img, cells=selected_cells, grid_step_px=int(args.grid_step_px)), overlay_path)
        save_png(steered_img, steered_path)

        window_rows.append(
            {
                "window_index": int(row["window_index"]),
                "row_index": int(row["row_index"]),
                "col_index": int(row["col_index"]),
                "left": int(left),
                "top": int(top),
                "pattern": pattern,
                "selected_cells": encode_cells(selected_cells),
                "selected_cells_decoded": selected_cells,
                "selected_cell_count": len(selected_cells),
                "source_path": str(source_path),
                "overlay_path": str(overlay_path),
                "steered_path": str(steered_path),
            }
        )
        generated_windows.append(
            {
                "window_index": int(row["window_index"]),
                "left": int(left),
                "top": int(top),
                "steered_img": np.asarray(steered_img, dtype=np.uint8),
            }
        )
        print(f"[ok] window {int(row['window_index']):03d} pattern={pattern} cells={encode_cells(selected_cells)}")

    save_rows_csv(args.out_dir / "window_manifest.csv", window_rows)
    source_overlay = draw_global_edit_overlay(
        canvas_img,
        window_rows=window_rows,
        grid_step_px=int(args.grid_step_px),
    )
    if edit_area_bounds is not None:
        source_overlay = draw_global_area_box(source_overlay, bounds=edit_area_bounds)
    save_png(source_overlay, args.out_dir / "source_canvas_edit_overlay.png")

    stitch_modes = ["hard_stitch", "overlap_average", "center_weighted_blend", "trusted_center_only"]
    stitch_rows: list[dict[str, object]] = []
    stitched_contact_items: list[tuple[str, Image.Image]] = [("source_actual", canvas_img)]
    for mode in stitch_modes:
        stitched = stitch_windows(
            window_outputs=generated_windows,
            canvas_size=int(args.canvas_size),
            window_size=int(args.window_size),
            mode=mode,
            trusted_margin_px=int(args.trusted_margin_px),
        )
        metrics = compute_seam_metrics(
            stitched=stitched,
            stride=int(args.window_stride),
            window_size=int(args.window_size),
            canvas_size=int(args.canvas_size),
        )
        stitched_img = Image.fromarray((stitched * 255.0).astype(np.uint8))
        out_path = args.out_dir / f"stitched_{mode}.png"
        overlay_path = args.out_dir / f"stitched_{mode}_edit_overlay.png"
        save_png(stitched_img, out_path)
        save_png(
            draw_global_area_box(
                draw_global_edit_overlay(
                    stitched_img,
                    window_rows=window_rows,
                    grid_step_px=int(args.grid_step_px),
                ),
                bounds=edit_area_bounds,
            ) if edit_area_bounds is not None else draw_global_edit_overlay(
                stitched_img,
                window_rows=window_rows,
                grid_step_px=int(args.grid_step_px),
            ),
            overlay_path,
        )
        write_json(args.out_dir / f"stitched_{mode}_metrics.json", metrics)
        stitched_contact_items.append((mode, stitched_img))
        stitch_rows.append({"mode": mode, "stitched_path": str(out_path), "stitched_overlay_path": str(overlay_path), **metrics})

    save_rows_csv(args.out_dir / "stitch_metrics.csv", stitch_rows)

    sheet = Image.new("RGB", (256 * len(stitched_contact_items), 256), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(stitched_contact_items):
        thumb = img.convert("RGB").resize((256, 256), resample=Image.BILINEAR)
        x0 = idx * 256
        sheet.paste(thumb, (x0, 0))
        draw.text((x0 + 8, 8), label, fill=(255, 255, 0))
    save_png(sheet, args.out_dir / "stitched_contact_sheet.png")

    summary = {
        "slide_path": str(slide_path),
        "patterns": patterns,
        "edit_layout": str(args.edit_layout),
        "center_area_bounds": list(edit_area_bounds) if edit_area_bounds is not None else None,
        "n_windows": len(window_rows),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "canvas_size": int(args.canvas_size),
        "trusted_margin_px": int(args.trusted_margin_px),
        "prototype_latent": int(chosen_latent),
        "stitch_metrics_csv": str(args.out_dir / "stitch_metrics.csv"),
        "window_manifest_csv": str(args.out_dir / "window_manifest.csv"),
        "experiment_args_path": str(args.out_dir / "experiment_args.json"),
        "cli_args": args_payload["cli_args"],
        "command": args_payload["command"],
    }
    write_json(args.out_dir / "summary.json", summary)
    print(f"[ok] wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
