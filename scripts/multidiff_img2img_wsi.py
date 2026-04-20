#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
import sys

import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.slides import open_slide, quick_tissue_score, read_region_rgb
from wsi_cf.generation.pixcell import (
    build_uni_grid_from_image,
    load_uni2,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
    vae_encode_auto,
)
from wsi_cf.steering.manifest import load_steer_manifest
from wsi_cf.steering.tile_features import apply_tile_steering


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MultiDiff img2img pipeline for WSI counterfactual research.")
    parser.add_argument("--input_image", type=Path, default=None)
    parser.add_argument("--input_svs", type=Path, default=None)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--region_w", type=int, default=1024)
    parser.add_argument("--region_h", type=int, default=1024)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--save_real", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--patch_px", type=int, default=0)
    parser.add_argument("--stride_px", type=int, default=0)
    parser.add_argument("--grid_step_px", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch_batch", type=int, default=256)
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--steer_tile", action="append", default=[])
    parser.add_argument("--steer_manifest", type=Path, default=None)
    parser.add_argument("--steer_blend", type=float, default=1.0)
    parser.add_argument("--mid_steer_start_ratio", type=float, default=0.0)
    parser.add_argument("--mid_steer_end_ratio", type=float, default=1.0)
    parser.add_argument("--mid_steer_alpha_start", type=float, default=1.0)
    parser.add_argument("--mid_steer_alpha_end", type=float, default=1.0)
    parser.add_argument("--mid_steer_alpha_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--pix_model_id", type=str, default="StonyBrook-CVLab/PixCell-256")
    parser.add_argument("--pix_pipeline_id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae_model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--use_tiled_vae_encode", action="store_true")
    parser.add_argument("--use_tiled_vae_decode", action="store_true")
    parser.add_argument("--encode_tile_img", type=int, default=1024)
    parser.add_argument("--encode_overlap_img", type=int, default=128)
    parser.add_argument("--decode_tile_lat", type=int, default=128)
    parser.add_argument("--decode_overlap_lat", type=int, default=16)
    parser.add_argument("--max_region_tries", type=int, default=32)
    parser.add_argument("--min_tissue", type=float, default=0.2)
    return parser


def pick_random_region(slide, *, region_w: int, region_h: int, n_tries: int, min_tissue: float, rng: random.Random) -> tuple[int, int]:
    width, height = slide.dimensions
    if width <= region_w or height <= region_h:
        return 0, 0
    best = (0, 0, -1.0)
    for _ in range(max(1, n_tries)):
        x0 = rng.randint(0, width - region_w)
        y0 = rng.randint(0, height - region_h)
        thumb = read_region_rgb(slide, x0, y0, min(256, region_w), min(256, region_h))
        score = quick_tissue_score(thumb)
        if score > best[2]:
            best = (x0, y0, score)
        if score >= min_tissue:
            return x0, y0
    return best[0], best[1]


def load_source_region(args, rng: random.Random) -> tuple[Image.Image, str]:
    if args.input_image is not None and args.input_svs is not None:
        raise ValueError("Use either --input_image or --input_svs, not both")
    if args.input_svs is not None:
        slide = open_slide(args.input_svs)
        try:
            x = int(args.x) if args.x is not None else None
            y = int(args.y) if args.y is not None else None
            if x is None or y is None:
                x, y = pick_random_region(
                    slide,
                    region_w=int(args.region_w),
                    region_h=int(args.region_h),
                    n_tries=int(args.max_region_tries),
                    min_tissue=float(args.min_tissue),
                    rng=rng,
                )
            img = read_region_rgb(slide, x, y, int(args.region_w), int(args.region_h))
            tag = f"{Path(args.input_svs).stem}__x_{x}__y_{y}"
            return img, tag
        finally:
            slide.close()
    if args.input_image is not None:
        img = Image.open(args.input_image).convert("RGB")
        x = int(args.x or 0)
        y = int(args.y or 0)
        w = int(args.region_w or (img.size[0] - x))
        h = int(args.region_h or (img.size[1] - y))
        img = img.crop((x, y, x + w, y + h))
        return img, f"{Path(args.input_image).stem}__x_{x}__y_{y}"
    raise ValueError("Provide either --input_image or --input_svs")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    set_seed(int(args.seed))
    rng = random.Random(int(args.seed))

    source_img, source_tag = load_source_region(args, rng)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_real:
        save_png(source_img, out_dir / f"{source_tag}_real.png")

    uni_model, uni_transform = load_uni2(device=device)
    sd3_vae = AutoencoderKL.from_pretrained(args.vae_model_id, subfolder=args.vae_subfolder, torch_dtype=dtype)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=sd3_vae,
        custom_pipeline=args.pix_pipeline_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    patch_px, stride_px, cond_grid_side = resolve_pixcell_window_config(
        pix_model_id=args.pix_model_id,
        patch_px=args.patch_px,
        stride_px=args.stride_px,
    )
    print(
        f"Resolved PixCell window config: model={args.pix_model_id} "
        f"patch_px={patch_px} stride_px={stride_px} cond_grid_side={cond_grid_side}"
    )

    z_grid = build_uni_grid_from_image(
        source_img,
        uni_model=uni_model,
        uni_transform=uni_transform,
        grid_step_px=int(args.grid_step_px),
        device=device,
        out_dtype=dtype,
    )
    z_grid_base = z_grid
    steer_specs = list(args.steer_tile)
    if args.steer_manifest is not None:
        steer_specs.extend(load_steer_manifest(args.steer_manifest))
    z_grid_edit = apply_tile_steering(z_grid_base, steer_specs=steer_specs, steer_blend=float(args.steer_blend))
    scheduled_z_grid = None
    if steer_specs:
        scheduled_z_grid = z_grid_edit
    else:
        z_grid_edit = z_grid_base

    init_latents = None
    if float(args.strength) > 0.0:
        real_np = np.asarray(source_img, dtype=np.uint8)
        real_t = torch.from_numpy(real_np).to(device=device).permute(2, 0, 1).float() / 255.0
        real_t = real_t.unsqueeze(0).to(dtype=dtype)
        with torch.inference_mode():
            init_latents = vae_encode_auto(
                pipeline.vae,
                real_t,
                use_tiled=bool(args.use_tiled_vae_encode),
                tile_img=int(args.encode_tile_img),
                overlap_img=int(args.encode_overlap_img),
            )
        del real_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    h, w = source_img.size[1], source_img.size[0]
    use_autocast = device.type == "cuda" and dtype == torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
    with torch.inference_mode(), ctx:
            img_t = sample_large_pixcell_multidiffusion(
                pipeline=pipeline,
                z_grid=z_grid_base,
                scheduled_z_grid=scheduled_z_grid,
                condition_start_ratio=float(args.mid_steer_start_ratio),
                condition_end_ratio=float(args.mid_steer_end_ratio),
                condition_alpha_start=float(args.mid_steer_alpha_start),
                condition_alpha_end=float(args.mid_steer_alpha_end),
                condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
                out_h=h,
                out_w=w,
            patch_px=patch_px,
            stride_px=stride_px,
            cond_grid_side=cond_grid_side,
            guidance_scale=float(args.guidance),
            num_steps=int(args.steps),
            patch_batch=int(args.patch_batch),
            strength=float(args.strength),
            init_latents=init_latents,
            use_tiled_vae_decode=bool(args.use_tiled_vae_decode),
            decode_tile_lat=int(args.decode_tile_lat),
            decode_overlap_lat=int(args.decode_overlap_lat),
            generator=generator,
        )

    img_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    gen = Image.fromarray(img_np)
    save_png(gen, out_dir / f"{source_tag}_gen.png")
    print(f"[ok] wrote {out_dir / f'{source_tag}_gen.png'}")


if __name__ == "__main__":
    main()
