#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png, write_json
from wsi_cf.common.paths import (
    DEFAULT_HNSCC_PROTOTYPE_NPZ,
    DEFAULT_SAE_VARIANT,
    SAE_VARIANTS,
    resolve_sae_paths,
)
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import parse_region_bank_csv
from wsi_cf.eval.hnsc_hpv import load_prototypes, pick_prototype_latent
from wsi_cf.generation.pixcell import (
    build_uni_grid_from_image,
    load_uni2,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
    vae_encode_auto,
)
from wsi_cf.steering.sae_edit import edit_uni_z_grid_with_sae
from wsi_cf.steering.sae_runtime import load_sae_from_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run true 256x256 single-tile HNSCC HPV steering with PixCell-256."
    )
    parser.add_argument("--tile-image", type=Path, default=None, help="Optional standalone 256 tile image.")
    parser.add_argument("--tile-feature-npy", type=Path, default=None, help="Optional [1536] or [1,1,1536] UNI feature for --tile-image.")
    parser.add_argument(
        "--region-bank-csv",
        type=Path,
        default=WSI_CF_ROOT / "artifacts/hnscc_region_bank_20x_2048_sample20/region_bank.csv",
        help="Region bank used when --tile-image is not supplied.",
    )
    parser.add_argument("--region-index", type=int, default=0)
    parser.add_argument("--cell", type=str, default="auto", help="Tile cell as gx,gy, or auto for a center-ish tile.")
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/hnscc_single_tile_bidirectional_256")
    parser.add_argument("--directions", type=str, default="hpv_pos,hpv_neg")
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--preserve-strength", type=float, default=0.05)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.5)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=0.5)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--pix-model-id", type=str, default="StonyBrook-CVLab/PixCell-256")
    parser.add_argument("--pix-pipeline-id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae-model-id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae-subfolder", type=str, default="vae")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=64)
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--sae-ckpt", type=Path, default=None)
    parser.add_argument("--sae-cfg", type=Path, default=None)
    parser.add_argument("--prototype-npz", type=Path, default=DEFAULT_HNSCC_PROTOTYPE_NPZ)
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--save-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_dirs(value: str) -> list[str]:
    out = [token.strip() for token in str(value).split(",") if token.strip()]
    bad = [direction for direction in out if direction not in {"hpv_pos", "hpv_neg"}]
    if bad:
        raise ValueError(f"Unsupported directions: {bad}")
    if not out:
        raise ValueError("At least one direction is required")
    return out


def parse_cell(value: str, *, grid_w: int, grid_h: int) -> tuple[int, int]:
    if str(value).strip().lower() == "auto":
        return max(0, int(grid_w) // 2 - 1), max(0, int(grid_h) // 2 - 1)
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError("--cell must be 'auto' or 'gx,gy'")
    gx, gy = int(parts[0]), int(parts[1])
    if not (0 <= gx < int(grid_w) and 0 <= gy < int(grid_h)):
        raise ValueError(f"Cell {(gx, gy)} outside grid {grid_w}x{grid_h}")
    return gx, gy


def load_tile_from_region_bank(args: argparse.Namespace) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    rows = parse_region_bank_csv(args.region_bank_csv)
    if not rows:
        raise ValueError(f"No rows in region bank: {args.region_bank_csv}")
    if int(args.region_index) < 0 or int(args.region_index) >= len(rows):
        raise ValueError(f"--region-index must be in [0,{len(rows)-1}], got {args.region_index}")
    row = rows[int(args.region_index)]
    image = Image.open(row.image_path).convert("RGB")
    zgrid = np.asarray(np.load(row.feature_grid_path), dtype=np.float32)
    if zgrid.ndim != 3:
        raise ValueError(f"Expected feature grid [H,W,D], got {zgrid.shape}")
    grid_h, grid_w, feature_dim = zgrid.shape
    gx, gy = parse_cell(args.cell, grid_w=int(grid_w), grid_h=int(grid_h))
    step = int(row.grid_step_px)
    tile = image.crop((gx * step, gy * step, (gx + 1) * step, (gy + 1) * step)).convert("RGB")
    if tile.size != (int(args.grid_step_px), int(args.grid_step_px)):
        tile = tile.resize((int(args.grid_step_px), int(args.grid_step_px)), resample=Image.BILINEAR)
    feature = np.asarray(zgrid[gy, gx], dtype=np.float32).reshape(1, 1, int(feature_dim))
    meta = {
        "source": "region_bank",
        "region_bank_csv": str(args.region_bank_csv),
        "region_index": int(args.region_index),
        "region_id": str(row.region_id),
        "slide_key": str(row.slide_key),
        "case_id": str(row.case_id),
        "label": int(row.label),
        "hpv_status": str(row.hpv_status),
        "cell": {"gx": int(gx), "gy": int(gy)},
        "region_image_path": str(row.image_path),
        "region_feature_grid_path": str(row.feature_grid_path),
        "grid_shape": [int(grid_h), int(grid_w)],
        "grid_step_px": int(row.grid_step_px),
    }
    return tile, feature, meta


def load_tile_image_feature(args: argparse.Namespace, device: torch.device) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    if args.tile_image is None:
        return load_tile_from_region_bank(args)
    tile = Image.open(args.tile_image).convert("RGB")
    if tile.size != (int(args.grid_step_px), int(args.grid_step_px)):
        raise ValueError(f"--tile-image must be {args.grid_step_px}x{args.grid_step_px}, got {tile.size}")
    if args.tile_feature_npy is not None:
        arr = np.asarray(np.load(args.tile_feature_npy), dtype=np.float32)
        feature = arr.reshape(1, 1, -1)
    else:
        uni_model, uni_transform = load_uni2(device)
        with torch.inference_mode():
            z = build_uni_grid_from_image(
                tile,
                uni_model=uni_model,
                uni_transform=uni_transform,
                grid_step_px=int(args.grid_step_px),
                device=device,
                out_dtype=torch.float32,
            )
        feature = z.detach().cpu().numpy().astype(np.float32)
        del uni_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    meta = {
        "source": "tile_image",
        "tile_image": str(args.tile_image),
        "tile_feature_npy": str(args.tile_feature_npy) if args.tile_feature_npy else "",
        "grid_shape": [1, 1],
        "grid_step_px": int(args.grid_step_px),
    }
    return tile, feature, meta


def save_args(args: argparse.Namespace) -> None:
    payload = {
        "args": {str(k): str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "command": " ".join(shlex.quote(part) for part in [sys.executable, __file__, *sys.argv[1:]]),
    }
    write_json(args.out_dir / "experiment_args.json", payload)


def tensor_image_from_pil(img: Image.Image, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def run_sample(
    *,
    pipeline,
    base_z: torch.Tensor,
    edited_z: torch.Tensor | None,
    source_latents: torch.Tensor,
    preserve_map: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    seed_offset: int,
) -> Image.Image:
    patch_px, stride_px, cond_grid_side = resolve_pixcell_window_config(
        pix_model_id=str(args.pix_model_id),
        patch_px=int(args.grid_step_px),
        stride_px=int(args.grid_step_px),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(seed_offset))
    use_autocast = device.type == "cuda" and dtype == torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
    with torch.inference_mode(), ctx:
        img_t = sample_large_pixcell_multidiffusion(
            pipeline=pipeline,
            z_grid=base_z.to(device=device, dtype=dtype),
            scheduled_z_grid=None if edited_z is None else edited_z.to(device=device, dtype=dtype),
            condition_start_ratio=float(args.mid_steer_start_ratio),
            condition_end_ratio=float(args.mid_steer_end_ratio),
            condition_alpha_start=float(args.mid_steer_alpha_start),
            condition_alpha_end=float(args.mid_steer_alpha_end),
            condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
            out_h=int(args.grid_step_px),
            out_w=int(args.grid_step_px),
            patch_px=int(patch_px),
            stride_px=int(stride_px),
            cond_grid_side=int(cond_grid_side),
            guidance_scale=float(args.guidance),
            num_steps=int(args.steps),
            patch_batch=int(args.patch_batch),
            strength=0.0,
            init_latents=None,
            preserve_source_latents=source_latents,
            preserve_strength_map=preserve_map,
            preserve_outside_strength=float(args.preserve_strength),
            preserve_edit_strength=float(args.preserve_strength),
            use_tiled_vae_decode=False,
            decode_tile_lat=128,
            decode_overlap_lat=16,
            generator=generator,
        )
    arr = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_args(args)
    directions = parse_dirs(args.directions)
    device = resolve_device(args.device)
    dtype = torch.float16 if str(args.dtype) == "fp16" else torch.float32

    tile_img, feature_np, tile_meta = load_tile_image_feature(args, device=device)
    if feature_np.shape != (1, 1, 1536):
        raise ValueError(f"Expected tile feature shape [1,1,1536], got {feature_np.shape}")
    save_png(tile_img, args.out_dir / "source_tile_actual.png")
    np.save(args.out_dir / "source_tile_feature.npy", feature_np.reshape(1536).astype(np.float32))
    write_json(args.out_dir / "tile_meta.json", tile_meta)

    if bool(args.dry_run):
        write_json(
            args.out_dir / "run_summary.json",
            {
                "dry_run": True,
                "directions": directions,
                "tile_meta": tile_meta,
                "outputs": [],
            },
        )
        print(json.dumps(json.loads((args.out_dir / "run_summary.json").read_text()), indent=2))
        return

    sae_ckpt, sae_cfg = resolve_sae_paths(args.sae_variant, args.sae_ckpt, args.sae_cfg)
    sae_model, _, _ = load_sae_from_config(sae_ckpt, sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")

    base_z = torch.from_numpy(feature_np).to(device=device, dtype=torch.float32)
    edited_by_direction: dict[str, torch.Tensor] = {}
    for direction in directions:
        latent = int(pos_latent if direction == "hpv_pos" else neg_latent)
        edited, _ = edit_uni_z_grid_with_sae(
            sae_model=sae_model,
            z_grid=base_z,
            target_latent_vector=torch.from_numpy(np.asarray(proto_by_latent[latent], dtype=np.float32)),
            target_latent_vector_strength=float(args.prototype_strength),
            blend=float(args.steer_blend),
            keep_non_selected=True,
            return_debug=False,
        )
        edited_by_direction[direction] = edited.detach()

    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=AutoencoderKL.from_pretrained(args.vae_model_id, subfolder=args.vae_subfolder, torch_dtype=dtype),
        custom_pipeline=args.pix_pipeline_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    source_img_t = tensor_image_from_pil(tile_img, device=device, dtype=dtype)
    source_latents = vae_encode_auto(
        pipeline.vae,
        source_img_t,
        use_tiled=False,
        tile_img=0,
        overlap_img=0,
    )
    preserve_map = torch.full(
        (int(args.grid_step_px), int(args.grid_step_px)),
        float(args.preserve_strength),
        device=device,
        dtype=torch.float32,
    )

    outputs: list[dict[str, Any]] = []
    if bool(args.save_baseline):
        baseline = run_sample(
            pipeline=pipeline,
            base_z=base_z,
            edited_z=None,
            source_latents=source_latents,
            preserve_map=preserve_map,
            args=args,
            device=device,
            dtype=dtype,
            seed_offset=0,
        )
        save_png(baseline, args.out_dir / "baseline_regenerated.png")
        outputs.append({"kind": "baseline", "path": str(args.out_dir / "baseline_regenerated.png")})

    for idx, direction in enumerate(directions, start=1):
        latent = int(pos_latent if direction == "hpv_pos" else neg_latent)
        img = run_sample(
            pipeline=pipeline,
            base_z=base_z,
            edited_z=edited_by_direction[direction],
            source_latents=source_latents,
            preserve_map=preserve_map,
            args=args,
            device=device,
            dtype=dtype,
            seed_offset=idx,
        )
        direction_dir = args.out_dir / direction
        direction_dir.mkdir(parents=True, exist_ok=True)
        save_png(img, direction_dir / "generated.png")
        np.save(direction_dir / "edited_tile_feature.npy", edited_by_direction[direction].detach().cpu().numpy().reshape(1536).astype(np.float32))
        run_meta = {
            "direction": direction,
            "prototype_latent": int(latent),
            "prototype_key": str(args.prototype_key),
            "prototype_strength": float(args.prototype_strength),
            "steer_blend": float(args.steer_blend),
            "preserve_strength": float(args.preserve_strength),
            "mid_steer_start_ratio": float(args.mid_steer_start_ratio),
            "mid_steer_end_ratio": float(args.mid_steer_end_ratio),
            "mid_steer_alpha_start": float(args.mid_steer_alpha_start),
            "mid_steer_alpha_end": float(args.mid_steer_alpha_end),
            "steps": int(args.steps),
            "guidance": float(args.guidance),
            "pix_model_id": str(args.pix_model_id),
            "sae_variant": str(args.sae_variant),
            "source_tile_path": str(args.out_dir / "source_tile_actual.png"),
            "generated_path": str(direction_dir / "generated.png"),
        }
        write_json(direction_dir / "run_manifest.json", run_meta)
        outputs.append(run_meta)

    write_json(
        args.out_dir / "run_summary.json",
        {
            "dry_run": False,
            "directions": directions,
            "tile_meta": tile_meta,
            "outputs": outputs,
        },
    )
    print(json.dumps(json.loads((args.out_dir / "run_summary.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
