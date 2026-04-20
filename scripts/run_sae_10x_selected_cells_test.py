#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
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
from wsi_cf.data.slides import (
    infer_objective_power,
    open_slide,
    quick_tissue_score,
    read_region_rgb_at_magnification,
)
from wsi_cf.eval.hnsc_hpv import load_prototypes, pick_prototype_latent
from wsi_cf.generation.pixcell import (
    build_uni_grid_from_image,
    load_uni2,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
)
from wsi_cf.steering.cell_selection import (
    block_cells,
    parse_cell_specs,
    random_cells,
    random_connected_cells,
    validate_cells,
)

ensure_legacy_repo_root_on_path()

from utils.sae import load_sae_from_config  # type: ignore
from utils.sae_edit import edit_uni_z_grid_with_sae  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-off SAE selected-cells steering test on a freshly extracted 10x 1024x1024 region. "
            "The source image is cropped from the slide at target magnification, encoded into a 4x4 UNI grid, "
            "and then steered on custom or random selected cells."
        )
    )
    parser.add_argument("--input-svs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--x", type=int, default=None, help="Optional level-0 top-left x for the 10x crop")
    parser.add_argument("--y", type=int, default=None, help="Optional level-0 top-left y for the 10x crop")
    parser.add_argument("--target-magnification", type=float, default=10.0)
    parser.add_argument("--region-size", type=int, default=1024)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--max-region-tries", type=int, default=64)
    parser.add_argument("--min-tissue", type=float, default=0.35)
    parser.add_argument("--selection-mode", type=str, default="manual", choices=["manual", "random_k", "random_connected_k", "block"])
    parser.add_argument("--steer-cell", action="append", default=[], help="Repeatable gx,gy spec used in manual mode")
    parser.add_argument("--steer-count", type=int, default=2, help="Used by random_k and random_connected_k")
    parser.add_argument("--anchor-gx", type=int, default=None, help="Optional anchor gx for connected/block patterns")
    parser.add_argument("--anchor-gy", type=int, default=None, help="Optional anchor gy for connected/block patterns")
    parser.add_argument("--block-w", type=int, default=2)
    parser.add_argument("--block-h", type=int, default=2)
    parser.add_argument("--direction", type=str, default="hpv_pos", choices=["hpv_pos", "hpv_neg"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pix_model_id", type=str, default="StonyBrook-CVLab/PixCell-1024")
    parser.add_argument("--pix_pipeline_id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae_model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.0)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--sae-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt"))
    parser.add_argument("--sae-cfg", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json"))
    parser.add_argument(
        "--prototype-npz",
        type=Path,
        default=Path("/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"),
    )
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    return parser


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


def make_region_cells_preview(img: Image.Image, *, grid_step_px: int, cells: list[tuple[int, int]]) -> Image.Image:
    preview = img.copy().convert("RGB")
    draw = ImageDraw.Draw(preview)
    idx = 0
    selected = {(gx, gy): rank + 1 for rank, (gx, gy) in enumerate(cells)}
    grid_h = preview.size[1] // int(grid_step_px)
    grid_w = preview.size[0] // int(grid_step_px)
    for gy in range(grid_h):
        for gx in range(grid_w):
            x0 = gx * int(grid_step_px)
            y0 = gy * int(grid_step_px)
            x1 = min(preview.size[0] - 1, x0 + int(grid_step_px) - 1)
            y1 = min(preview.size[1] - 1, y0 + int(grid_step_px) - 1)
            color = (255, 0, 0) if (gx, gy) in selected else (180, 180, 180)
            width = 5 if (gx, gy) in selected else 1
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
            label = str(selected[(gx, gy)]) if (gx, gy) in selected else str(idx)
            draw.text((x0 + 6, y0 + 6), label, fill=(255, 255, 0))
            idx += 1
    return preview


def pick_random_region_at_target_magnification(
    slide,
    *,
    out_size: int,
    target_magnification: float,
    n_tries: int,
    min_tissue: float,
    rng: random.Random,
) -> tuple[int, int, float, int, int]:
    objective = infer_objective_power(slide)
    crop_w = max(1, int(round(float(out_size) * (float(objective) / float(target_magnification)))))
    crop_h = crop_w
    width, height = slide.dimensions
    if width <= crop_w or height <= crop_h:
        img, used_w, used_h = read_region_rgb_at_magnification(
            slide,
            x0=0,
            y0=0,
            out_w=int(out_size),
            out_h=int(out_size),
            target_magnification=float(target_magnification),
        )
        return 0, 0, quick_tissue_score(img), used_w, used_h

    best = (0, 0, -1.0, crop_w, crop_h)
    for _ in range(max(1, int(n_tries))):
        x0 = rng.randint(0, max(0, width - crop_w))
        y0 = rng.randint(0, max(0, height - crop_h))
        img, used_w, used_h = read_region_rgb_at_magnification(
            slide,
            x0=int(x0),
            y0=int(y0),
            out_w=int(out_size),
            out_h=int(out_size),
            target_magnification=float(target_magnification),
        )
        score = quick_tissue_score(img)
        if score > best[2]:
            best = (x0, y0, score, used_w, used_h)
        if score >= float(min_tissue):
            return x0, y0, score, used_w, used_h
    return best


def resolve_selected_cells(args, *, grid_w: int, grid_h: int, rng: random.Random) -> list[tuple[int, int]]:
    if args.selection_mode == "manual":
        cells = parse_cell_specs(list(args.steer_cell))
        if not cells:
            raise ValueError("manual selection_mode requires at least one --steer-cell gx,gy")
        return validate_cells(cells, grid_w=int(grid_w), grid_h=int(grid_h))
    if args.selection_mode == "random_k":
        return random_cells(grid_w=int(grid_w), grid_h=int(grid_h), count=int(args.steer_count), rng=rng)
    if args.selection_mode == "random_connected_k":
        start = None
        if args.anchor_gx is not None and args.anchor_gy is not None:
            start = (int(args.anchor_gx), int(args.anchor_gy))
        cells = random_connected_cells(
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            count=int(args.steer_count),
            rng=rng,
            start=start,
        )
        return validate_cells(cells, grid_w=int(grid_w), grid_h=int(grid_h))
    if args.selection_mode == "block":
        if args.anchor_gx is None or args.anchor_gy is None:
            raise ValueError("block selection_mode requires --anchor-gx and --anchor-gy")
        return block_cells(
            origin_gx=int(args.anchor_gx),
            origin_gy=int(args.anchor_gy),
            width=int(args.block_w),
            height=int(args.block_h),
            grid_w=int(grid_w),
            grid_h=int(grid_h),
        )
    raise ValueError(f"Unsupported selection_mode {args.selection_mode}")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    set_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    slide = open_slide(args.input_svs)
    try:
        if args.x is None or args.y is None:
            region_x, region_y, tissue_score, crop_w_level0, crop_h_level0 = pick_random_region_at_target_magnification(
                slide,
                out_size=int(args.region_size),
                target_magnification=float(args.target_magnification),
                n_tries=int(args.max_region_tries),
                min_tissue=float(args.min_tissue),
                rng=rng,
            )
            source_img, _, _ = read_region_rgb_at_magnification(
                slide,
                x0=int(region_x),
                y0=int(region_y),
                out_w=int(args.region_size),
                out_h=int(args.region_size),
                target_magnification=float(args.target_magnification),
            )
        else:
            region_x = int(args.x)
            region_y = int(args.y)
            source_img, crop_w_level0, crop_h_level0 = read_region_rgb_at_magnification(
                slide,
                x0=int(region_x),
                y0=int(region_y),
                out_w=int(args.region_size),
                out_h=int(args.region_size),
                target_magnification=float(args.target_magnification),
            )
            tissue_score = quick_tissue_score(source_img)
    finally:
        slide.close()

    grid_side = int(args.region_size) // int(args.grid_step_px)
    if grid_side * int(args.grid_step_px) != int(args.region_size):
        raise ValueError("region-size must be divisible by grid-step-px for the 4x4 test runner")

    selected_cells = resolve_selected_cells(args, grid_w=grid_side, grid_h=grid_side, rng=rng)

    save_png(source_img, args.out_dir / "source_10x.png")
    save_png(draw_selected_cells_overlay(source_img, cells=selected_cells, grid_step_px=int(args.grid_step_px)), args.out_dir / "source_selected_overlay.png")
    save_png(make_region_cells_preview(source_img, grid_step_px=int(args.grid_step_px), cells=selected_cells), args.out_dir / "source_cells.png")

    uni_model, uni_transform = load_uni2(device=device)
    z_grid_base = build_uni_grid_from_image(
        source_img,
        uni_model=uni_model,
        uni_transform=uni_transform,
        grid_step_px=int(args.grid_step_px),
        device=device,
        out_dtype=dtype,
    ).detach().float()
    np.save(args.out_dir / "source_zgrid.npy", z_grid_base.cpu().numpy().astype(np.float32))

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    chosen_latent = int(pos_latent if str(args.direction) == "hpv_pos" else neg_latent)

    tile_mask = np.zeros((grid_side, grid_side), dtype=np.float32)
    for gx, gy in selected_cells:
        tile_mask[int(gy), int(gx)] = 1.0

    z_grid_edit, _ = edit_uni_z_grid_with_sae(
        sae_model=sae_model,
        z_grid=z_grid_base.to(device=device, dtype=torch.float32),
        target_latent_vector=proto_by_latent[chosen_latent],
        target_latent_vector_strength=float(args.prototype_strength),
        tile_mask=tile_mask,
        blend=float(args.steer_blend),
        keep_non_selected=True,
        return_debug=False,
    )
    np.save(args.out_dir / "edited_zgrid.npy", z_grid_edit.detach().cpu().numpy().astype(np.float32))

    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=AutoencoderKL.from_pretrained(args.vae_model_id, subfolder=args.vae_subfolder, torch_dtype=dtype),
        custom_pipeline=args.pix_pipeline_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_px, stride_px, cond_grid_side = resolve_pixcell_window_config(
        pix_model_id=args.pix_model_id,
        patch_px=0,
        stride_px=0,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    use_autocast = device.type == "cuda" and dtype == torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
    with torch.inference_mode(), ctx:
        img_t = sample_large_pixcell_multidiffusion(
            pipeline=pipeline,
            z_grid=z_grid_base.to(device=device, dtype=dtype),
            scheduled_z_grid=z_grid_edit.to(device=device, dtype=dtype),
            condition_start_ratio=float(args.mid_steer_start_ratio),
            condition_end_ratio=float(args.mid_steer_end_ratio),
            condition_alpha_start=float(args.mid_steer_alpha_start),
            condition_alpha_end=float(args.mid_steer_alpha_end),
            condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
            out_h=int(args.region_size),
            out_w=int(args.region_size),
            patch_px=patch_px,
            stride_px=stride_px,
            cond_grid_side=cond_grid_side,
            guidance_scale=float(args.guidance),
            num_steps=int(args.steps),
            patch_batch=int(args.patch_batch),
            strength=0.0,
            init_latents=None,
            use_tiled_vae_decode=False,
            decode_tile_lat=128,
            decode_overlap_lat=16,
            generator=generator,
        )

    img_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    save_png(Image.fromarray(img_np), args.out_dir / "generated.png")

    write_json(
        args.out_dir / "run_meta.json",
        {
            "input_svs": str(args.input_svs),
            "region_x_level0": int(region_x),
            "region_y_level0": int(region_y),
            "crop_w_level0": int(crop_w_level0),
            "crop_h_level0": int(crop_h_level0),
            "target_magnification": float(args.target_magnification),
            "region_size": int(args.region_size),
            "grid_step_px": int(args.grid_step_px),
            "selection_mode": str(args.selection_mode),
            "selected_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in selected_cells],
            "selected_cell_count": len(selected_cells),
            "direction": str(args.direction),
            "prototype_key": str(args.prototype_key),
            "prototype_latent": int(chosen_latent),
            "prototype_strength": float(args.prototype_strength),
            "steer_blend": float(args.steer_blend),
            "mid_steer_start_ratio": float(args.mid_steer_start_ratio),
            "mid_steer_end_ratio": float(args.mid_steer_end_ratio),
            "mid_steer_alpha_start": float(args.mid_steer_alpha_start),
            "mid_steer_alpha_end": float(args.mid_steer_alpha_end),
            "mid_steer_alpha_schedule": str(args.mid_steer_alpha_schedule),
            "pix_model_id": str(args.pix_model_id),
            "seed": int(args.seed),
            "steps": int(args.steps),
            "guidance": float(args.guidance),
            "tissue_score": float(tissue_score),
        },
    )
    print(f"[ok] wrote {args.out_dir / 'generated.png'}")


if __name__ == "__main__":
    main()
