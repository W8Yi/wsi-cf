from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, DiffusionPipeline

from utils.diffusion import build_uni_grid_from_pil, sample_multidiffusion_from_zgrid_with_midref
from utils.image_io import pil_to_nchw_float01, save_image_tensor_01, validate_min_image_size, validate_min_output_size
from utils.image_inputs import gather_image_paths
from utils.uni_edit import edit_uni_z_grid_single_tile_with_vector
from utils.uni import get_uni


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Edit tile image(s) in UNI space (one input image = one tile) and regenerate with PixCell."
    )

    # Inputs
    ap.add_argument("--image", type=str, default=None, help="Input tile image path.")
    ap.add_argument("--image-dir", type=str, default=None, help="Directory of tile images (png/jpg/tif).")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)

    # UNI
    ap.add_argument("--tile-px", type=int, default=256)
    ap.add_argument("--grid-step-px", type=int, default=256)

    # Diffusion
    ap.add_argument("--pixcell-model", type=str, default="StonyBrook-CVLab/PixCell-256")
    ap.add_argument("--pixcell-custom-pipeline", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    ap.add_argument("--vae-model", type=str, default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--out-h", type=int, default=None)
    ap.add_argument("--out-w", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--patch-px", type=int, default=256)
    ap.add_argument("--stride-px", type=int, default=128)
    ap.add_argument("--patch-batch", type=int, default=256)
    ap.add_argument("--reference-start-ratio", type=float, default=0.6)
    ap.add_argument("--reference-mix", type=float, default=0.0)

    # Steering controls
    ap.add_argument("--mode", type=str, required=True, choices=["delta", "target"])
    ap.add_argument("--vector-path", type=str, required=True, help=".npy direction in UNI space (shape [D]).")
    ap.add_argument("--vector-strength", type=float, default=1.0)
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--normalize-vector", action="store_true")

    # Output extras
    ap.add_argument("--save-z", action="store_true", help="Save z grids as .npy for debugging.")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()

    # Prepare filesystem and input file list.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = gather_image_paths(args.image, args.image_dir)

    device = args.device

    print("[setup] Loading UNI model...")
    uni, uni_transform = get_uni(device)

    print("[setup] Loading diffusion pipeline...")
    sd3_vae = AutoencoderKL.from_pretrained(args.vae_model, subfolder="vae")
    pipe = DiffusionPipeline.from_pretrained(
        args.pixcell_model,
        vae=sd3_vae,
        custom_pipeline=args.pixcell_custom_pipeline,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    pipe.to(device)
    vec = np.load(args.vector_path)

    for i, img_path in enumerate(image_paths, start=1):
        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        validate_min_image_size(w, h, min_size=256)

        out_h = int(args.out_h) if args.out_h is not None else h
        out_w = int(args.out_w) if args.out_w is not None else w
        validate_min_output_size(out_h, out_w, min_size=256)

        item_out_dir = out_dir if len(image_paths) == 1 else (out_dir / img_path.stem)
        item_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[tile {i}/{len(image_paths)}] Building UNI z-grid for {img_path.name}...")
        z, coords_grid = build_uni_grid_from_pil(
            img_pil,
            uni,
            uni_transform,
            tile_px=args.tile_px,
            grid_step_px=args.grid_step_px,
            device=device,
            dtype=torch.float16,
            return_numpy=False,
        )
        gh, gw, d = z.shape
        if (gh, gw) != (1, 1):
            raise ValueError(
                "uni_steer now expects each input image to be a single tile. "
                f"Got z-grid shape {(gh, gw)} for image {img_path}. "
                "Use cropped tile images, or set tile/grid sizes so the grid is 1x1."
            )

        print(f"[tile {i}/{len(image_paths)}] Applying UNI edit mode: {args.mode}")
        z_edit, dbg = edit_uni_z_grid_single_tile_with_vector(
            z_grid=z,
            mode=args.mode,
            vector=vec,
            vector_strength=args.vector_strength,
            blend=args.blend,
            normalize_vector=args.normalize_vector,
        )

        print(f"[tile {i}/{len(image_paths)}] Regenerating image...")
        img_tensor_01 = pil_to_nchw_float01(img_pil)
        g = torch.Generator(device=device).manual_seed(args.seed)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            img_edit_t = sample_multidiffusion_from_zgrid_with_midref(
                pipeline=pipe,
                z_grid=z_edit,
                original_image=img_tensor_01,
                out_h=out_h,
                out_w=out_w,
                patch_px=args.patch_px,
                stride_px=args.stride_px,
                steps=args.steps,
                guidance=args.guidance,
                patch_batch=args.patch_batch,
                reference_start_ratio=args.reference_start_ratio,
                reference_mix=args.reference_mix,
                generator=g,
            )

        print(f"[tile {i}/{len(image_paths)}] Saving outputs...")
        out_img = item_out_dir / "uni_steer_edit.png"
        save_image_tensor_01(img_edit_t, out_img)

        if args.save_z:
            np.save(item_out_dir / "z_orig.npy", z.detach().float().cpu().numpy())
            np.save(item_out_dir / "z_edit.npy", z_edit.detach().float().cpu().numpy())
            np.save(item_out_dir / "coords_grid.npy", coords_grid)

        meta = {
            "image": str(img_path),
            "out_image": str(out_img),
            "input_size": [h, w],
            "output_size": [out_h, out_w],
            "z_shape": [int(gh), int(gw), int(d)],
            "mode": args.mode,
            "vector_path": args.vector_path,
            "vector_strength": args.vector_strength,
            "blend": args.blend,
            "normalize_vector": args.normalize_vector,
            "reference_start_ratio": args.reference_start_ratio,
            "reference_mix": args.reference_mix,
            "debug": dbg,
        }
        with open(item_out_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print("Saved:", out_img)
        print("Saved:", item_out_dir / "run_meta.json")


if __name__ == "__main__":
    main()
