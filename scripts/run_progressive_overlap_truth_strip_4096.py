#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


CENTER_2X2 = [(1, 1), (2, 1), (1, 2), (2, 2)]


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
            "Progressive overlap-truth strip exploration. The first 1024 window is generated and "
            "accepted as truth. Each later overlapping window is generated from the evolving canvas, "
            "then only its newly exposed right-side band is committed back."
        )
    )
    parser.add_argument("--input-svs", type=Path, default=None, help="Optional direct slide path")
    parser.add_argument("--slide-key", type=str, default="TCGA-BA-6872-01Z-00-DX1")
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/HNSCC_slides"))
    parser.add_argument("--x", type=int, default=0, help="Top-left region x at level 0")
    parser.add_argument("--y", type=int, default=0, help="Top-left region y at level 0")
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument("--canvas-size", type=int, default=4096)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--target-row-index", type=int, default=1)
    parser.add_argument("--start-col-index", type=int, default=0)
    parser.add_argument("--max-span", type=int, default=3, help="Number of consecutive windows in the rightward strip.")
    parser.add_argument("--direction", type=str, default="hpv_pos", choices=["hpv_pos", "hpv_neg"])
    parser.add_argument("--sae-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt"))
    parser.add_argument("--sae-cfg", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json"))
    parser.add_argument("--prototype-npz", type=Path, default=Path("/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"))
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--pix-model-id", type=str, default="StonyBrook-CVLab/PixCell-1024")
    parser.add_argument("--pix-pipeline-id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae-model-id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae-subfolder", type=str, default="vae")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--preserve-outside-latents", action="store_true", default=True)
    parser.add_argument("--preserve-outside-strength", type=float, default=0.2)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.5)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument(
        "--composite-outside-source",
        action="store_true",
        default=True,
        help="After generation, copy the source window back outside the selected cells with a soft feather.",
    )
    parser.add_argument("--composite-feather-px", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WSI_CF_ROOT / "artifacts/progressive_overlap_truth_strip_4096",
    )
    return parser


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


def draw_canvas_overlay(
    img: Image.Image,
    *,
    active_windows: list[dict[str, int]],
    committed_boxes: list[tuple[int, int, int, int]],
) -> Image.Image:
    canvas = img.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for row in active_windows:
        left = int(row["left"])
        top = int(row["top"])
        draw.rectangle([left, top, left + 1024 - 1, top + 1024 - 1], outline=(0, 255, 255), width=5)
    for x0, y0, x1, y1 in committed_boxes:
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0), width=4)
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


def make_soft_edit_mask_2d(*, width: int, height: int, cells: list[tuple[int, int]], grid_step_px: int, feather_px: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=np.float32)
    for gx, gy in cells:
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(int(width), x0 + int(grid_step_px))
        y1 = min(int(height), y0 + int(grid_step_px))
        mask[y0:y1, x0:x1] = 1.0
    feather = max(0, int(feather_px))
    if feather <= 0:
        return mask
    soft = mask.copy()
    for radius in range(1, feather + 1):
        alpha = 1.0 - (float(radius) / float(feather + 1))
        shifted = [
            np.pad(mask[:-radius, :], ((radius, 0), (0, 0))),
            np.pad(mask[radius:, :], ((0, radius), (0, 0))),
            np.pad(mask[:, :-radius], ((0, 0), (radius, 0))),
            np.pad(mask[:, radius:], ((0, 0), (0, radius))),
        ]
        for arr in shifted:
            soft = np.maximum(soft, np.minimum(arr, alpha))
    return np.clip(soft, 0.0, 1.0)


def save_rows_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def steer_window(
    *,
    window_img: Image.Image,
    selected_cells: list[tuple[int, int]],
    uni_model,
    uni_transform,
    sae_model,
    prototype_vec: torch.Tensor,
    pipeline,
    device,
    dtype,
    patch_px: int,
    stride_px: int,
    cond_grid_side: int,
    args: argparse.Namespace,
    generator_seed: int,
) -> Image.Image:
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
    if bool(args.preserve_outside_latents):
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
    generator.manual_seed(int(generator_seed))
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
    if bool(args.composite_outside_source):
        src_arr = np.asarray(window_img, dtype=np.float32) / 255.0
        gen_arr = np.asarray(steered_img, dtype=np.float32) / 255.0
        soft_mask = make_soft_edit_mask_2d(
            width=int(window_img.size[0]),
            height=int(window_img.size[1]),
            cells=selected_cells,
            grid_step_px=int(args.grid_step_px),
            feather_px=int(args.composite_feather_px),
        )[..., None]
        comp_arr = src_arr * (1.0 - soft_mask) + gen_arr * soft_mask
        steered_img = Image.fromarray((np.clip(comp_arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    return steered_img


def commit_window_update(
    *,
    current_canvas: np.ndarray,
    steered_img: Image.Image,
    left: int,
    top: int,
    window_size: int,
    window_stride: int,
    step_index: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = current_canvas.copy()
    img = np.asarray(steered_img, dtype=np.float32) / 255.0
    if step_index == 0:
        x0_local, x1_local = 0, int(window_size)
    else:
        x0_local = int(window_size - window_stride)
        x1_local = int(window_size)
    y0_local, y1_local = 0, int(window_size)

    dst_x0 = int(left + x0_local)
    dst_x1 = int(left + x1_local)
    dst_y0 = int(top + y0_local)
    dst_y1 = int(top + y1_local)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[y0_local:y1_local, x0_local:x1_local]
    return out, (dst_x0, dst_y0, dst_x1, dst_y1)


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

    windows = enumerate_windows(int(args.canvas_size), int(args.window_size), int(args.window_stride))
    windows_by_rc = {(int(row["row_index"]), int(row["col_index"])): row for row in windows}
    max_col_index = max(int(row["col_index"]) for row in windows)
    max_row_index = max(int(row["row_index"]) for row in windows)
    if not (0 <= int(args.target_row_index) <= max_row_index):
        raise ValueError(f"target-row-index must be in [0,{max_row_index}]")
    if not (0 <= int(args.start_col_index) <= max_col_index):
        raise ValueError(f"start-col-index must be in [0,{max_col_index}]")
    span_limit = min(int(args.max_span), max_col_index - int(args.start_col_index) + 1)
    if span_limit <= 0:
        raise ValueError("No valid progressive span is possible with the requested start column.")

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

    span_rows: list[dict[str, object]] = []
    contact_items: list[tuple[str, Image.Image]] = [("source_actual", canvas_img)]

    for span in range(1, span_limit + 1):
        current_canvas = np.asarray(canvas_img, dtype=np.float32) / 255.0
        span_dir = args.out_dir / f"span_{span}"
        span_dir.mkdir(parents=True, exist_ok=True)
        step_rows: list[dict[str, object]] = []
        committed_boxes: list[tuple[int, int, int, int]] = []
        active_windows: list[dict[str, int]] = []

        for step_idx in range(int(span)):
            row_idx = int(args.target_row_index)
            col_idx = int(args.start_col_index) + step_idx
            row = windows_by_rc[(row_idx, col_idx)]
            active_windows.append(row)
            left = int(row["left"])
            top = int(row["top"])

            current_canvas_img = Image.fromarray((np.clip(current_canvas, 0.0, 1.0) * 255.0).astype(np.uint8))
            window_img = current_canvas_img.crop((left, top, left + int(args.window_size), top + int(args.window_size))).convert("RGB")
            steered_img = steer_window(
                window_img=window_img,
                selected_cells=list(CENTER_2X2),
                uni_model=uni_model,
                uni_transform=uni_transform,
                sae_model=sae_model,
                prototype_vec=prototype_vec,
                pipeline=pipeline,
                device=device,
                dtype=dtype,
                patch_px=patch_px,
                stride_px=stride_px,
                cond_grid_side=cond_grid_side,
                args=args,
                generator_seed=int(args.seed) + int(span) * 1000 + int(step_idx),
            )
            current_canvas, commit_box = commit_window_update(
                current_canvas=current_canvas,
                steered_img=steered_img,
                left=left,
                top=top,
                window_size=int(args.window_size),
                window_stride=int(args.window_stride),
                step_index=int(step_idx),
            )
            committed_boxes.append(commit_box)

            step_dir = span_dir / "steps" / f"step_{step_idx + 1:02d}_col_{col_idx:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            save_png(window_img, step_dir / "source_window_from_current_canvas.png")
            save_png(draw_selected_cells_overlay(window_img, cells=list(CENTER_2X2), grid_step_px=int(args.grid_step_px)), step_dir / "selected_cells_overlay.png")
            save_png(steered_img, step_dir / "steered_window.png")
            step_rows.append(
                {
                    "span": int(span),
                    "step_index": int(step_idx),
                    "row_index": int(row_idx),
                    "col_index": int(col_idx),
                    "window_index": int(row["window_index"]),
                    "left": int(left),
                    "top": int(top),
                    "selected_cells": encode_cells(CENTER_2X2),
                    "source_window_path": str(step_dir / "source_window_from_current_canvas.png"),
                    "overlay_path": str(step_dir / "selected_cells_overlay.png"),
                    "steered_window_path": str(step_dir / "steered_window.png"),
                    "commit_mode": "full_window" if step_idx == 0 else "new_right_band_only",
                    "commit_bounds_global": f"{commit_box[0]},{commit_box[1]},{commit_box[2]},{commit_box[3]}",
                }
            )

        save_rows_csv(span_dir / "step_manifest.csv", step_rows)
        final_img = Image.fromarray((np.clip(current_canvas, 0.0, 1.0) * 255.0).astype(np.uint8))
        save_png(final_img, span_dir / "progressive_overlap_truth.png")
        save_png(
            draw_canvas_overlay(final_img, active_windows=active_windows, committed_boxes=committed_boxes),
            span_dir / "progressive_overlap_truth_edit_overlay.png",
        )
        save_png(
            draw_canvas_overlay(canvas_img, active_windows=active_windows, committed_boxes=committed_boxes),
            span_dir / "source_canvas_edit_overlay.png",
        )
        write_json(
            span_dir / "summary.json",
            {
                "span": int(span),
                "mode": "overlap_truth_progressive_right_strip",
                "target_row_index": int(args.target_row_index),
                "start_col_index": int(args.start_col_index),
                "window_size": int(args.window_size),
                "window_stride": int(args.window_stride),
                "first_step_commit": "full_window",
                "later_step_commit": "new_right_band_only",
                "selected_cells_per_window": [{"gx": int(gx), "gy": int(gy)} for gx, gy in CENTER_2X2],
                "active_windows": [
                    {"row_index": int(row["row_index"]), "col_index": int(row["col_index"]), "left": int(row["left"]), "top": int(row["top"])}
                    for row in active_windows
                ],
                "committed_boxes": [
                    {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}
                    for x0, y0, x1, y1 in committed_boxes
                ],
                "step_manifest_csv": str(span_dir / "step_manifest.csv"),
                "final_image_path": str(span_dir / "progressive_overlap_truth.png"),
                "overlay_path": str(span_dir / "progressive_overlap_truth_edit_overlay.png"),
                "prototype_latent": int(chosen_latent),
                "experiment_args_path": str(args.out_dir / "experiment_args.json"),
                "cli_args": args_payload["cli_args"],
                "command": args_payload["command"],
            },
        )
        span_rows.append(
            {
                "span": int(span),
                "mode": "overlap_truth_progressive_right_strip",
                "active_window_count": int(span),
                "step_manifest_csv": str(span_dir / "step_manifest.csv"),
                "final_image_path": str(span_dir / "progressive_overlap_truth.png"),
                "overlay_path": str(span_dir / "progressive_overlap_truth_edit_overlay.png"),
            }
        )
        contact_items.append((f"span_{span}", final_img))
        print(f"[ok] span {span} complete")

    save_rows_csv(args.out_dir / "progressive_summary.csv", span_rows)
    thumb_w = 256
    thumb_h = 256
    sheet = Image.new("RGB", (thumb_w * len(contact_items), thumb_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(contact_items):
        thumb = img.convert("RGB").resize((thumb_w, thumb_h), resample=Image.BILINEAR)
        x0 = idx * thumb_w
        sheet.paste(thumb, (x0, 0))
        draw.text((x0 + 8, 8), label, fill=(255, 255, 0))
    save_png(sheet, args.out_dir / "progressive_contact_sheet.png")

    write_json(
        args.out_dir / "summary.json",
        {
            "mode": "overlap_truth_progressive_right_strip",
            "slide_path": str(slide_path),
            "canvas_size": int(args.canvas_size),
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "grid_step_px": int(args.grid_step_px),
            "first_step_commit": "full_window",
            "later_step_commit": "new_right_band_only",
            "target_row_index": int(args.target_row_index),
            "start_col_index": int(args.start_col_index),
            "max_span": int(span_limit),
            "center_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in CENTER_2X2],
            "prototype_latent": int(chosen_latent),
            "progressive_summary_csv": str(args.out_dir / "progressive_summary.csv"),
            "contact_sheet_path": str(args.out_dir / "progressive_contact_sheet.png"),
            "experiment_args_path": str(args.out_dir / "experiment_args.json"),
            "cli_args": args_payload["cli_args"],
            "command": args_payload["command"],
        },
    )
    print(f"[ok] wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
