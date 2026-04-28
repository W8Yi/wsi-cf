from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, DiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.diffusion import build_uni_grid_from_pil, sample_multidiffusion_from_zgrid_with_midref
from utils.image_inputs import gather_image_paths
from utils.image_io import pil_to_nchw_float01, save_image_tensor_01, validate_min_image_size, validate_min_output_size
from utils.sae import load_sae_from_config
from utils.sae_edit import edit_uni_z_grid_with_sae
from utils.uni import get_uni


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Edit tile image(s) via SAE latent edits on UNI features, then regenerate with PixCell."
    )

    # Inputs
    ap.add_argument("--image", type=str, default=None, help="Input tile image path.")
    ap.add_argument("--image-dir", type=str, default=None, help="Directory of tile images (png/jpg/tif).")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)

    # UNI + SAE
    ap.add_argument("--sae-ckpt", type=str, required=True)
    ap.add_argument("--sae-cfg", type=str, required=True)
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

    # SAE latent steering (single latent only; keep this script focused and easy to reason about)
    ap.add_argument(
        "--steer-mode",
        type=str,
        required=True,
        choices=["latent_delta", "latent_target", "latent_scale", "latent_clamp"],
    )
    ap.add_argument("--latent-idx", type=int, required=True, help="SAE latent index to edit.")
    ap.add_argument("--delta", type=float, default=None, help="Used by --steer-mode latent_delta.")
    ap.add_argument("--target-value", type=float, default=None, help="Used by --steer-mode latent_target.")
    ap.add_argument("--scale", type=float, default=None, help="Used by --steer-mode latent_scale.")
    ap.add_argument("--clamp-value", type=float, default=None, help="Used by --steer-mode latent_clamp.")
    ap.add_argument(
        "--clamp-values-path",
        type=str,
        default=None,
        help="Optional .npy/.json vector of per-latent clamp values; uses the entry at --latent-idx.",
    )

    # Stabilizers / edit strength controls
    ap.add_argument("--blend", type=float, default=0.35)
    ap.add_argument("--latent-strength", type=float, default=0.4)
    ap.add_argument("--max-feature-delta-norm", type=float, default=6.0)

    # Output extras
    ap.add_argument("--save-z", action="store_true", help="Save z grids as .npy for debugging.")
    return ap


def _load_clamp_values(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            if "values" in data:
                arr = data["values"]
            elif "pctl" in data:
                arr = data["pctl"]
            else:
                raise ValueError("Clamp values JSON must contain 'values' or 'pctl'.")
        else:
            arr = data
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Clamp values must be 1D, got {arr.shape}")
    return arr


def _sanitize_for_json(value: Any) -> Any:
    """Summarize tensors/arrays so metadata stays readable and JSON-serializable."""
    if torch.is_tensor(value):
        arr = value.detach().float().cpu().numpy()
        return _summarize_array(arr)
    if isinstance(value, np.ndarray):
        return _summarize_array(value)
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def _summarize_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "size": 0}
    out = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    # Numeric stats are the most useful for this debugging context.
    if np.issubdtype(arr.dtype, np.number):
        arrf = arr.astype(np.float32, copy=False)
        out.update(
            {
                "min": float(arrf.min()),
                "mean": float(arrf.mean()),
                "max": float(arrf.max()),
            }
        )
    return out


def _resolve_clamp_value(args: argparse.Namespace, clamp_arr: np.ndarray | None) -> float | None:
    if args.steer_mode != "latent_clamp":
        return None
    if clamp_arr is not None:
        if args.latent_idx < 0 or args.latent_idx >= clamp_arr.shape[0]:
            raise ValueError("--clamp-values-path does not cover --latent-idx")
        return float(clamp_arr[args.latent_idx])
    if args.clamp_value is None:
        raise ValueError("--clamp-value or --clamp-values-path is required for latent_clamp")
    return float(args.clamp_value)


def _validate_steer_args(args: argparse.Namespace) -> None:
    if args.steer_mode == "latent_delta" and args.delta is None:
        raise ValueError("--delta is required for latent_delta")
    if args.steer_mode == "latent_target" and args.target_value is None:
        raise ValueError("--target-value is required for latent_target")
    if args.steer_mode == "latent_scale" and args.scale is None:
        raise ValueError("--scale is required for latent_scale")
    if args.steer_mode == "latent_clamp" and args.clamp_value is None and args.clamp_values_path is None:
        raise ValueError("--clamp-value or --clamp-values-path is required for latent_clamp")


def main() -> None:
    args = _build_argparser().parse_args()
    _validate_steer_args(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = gather_image_paths(args.image, args.image_dir)
    device = args.device

    clamp_arr = _load_clamp_values(Path(args.clamp_values_path)) if args.clamp_values_path else None

    print("[setup] Loading UNI model...")
    uni, uni_transform = get_uni(device)

    print("[setup] Loading SAE...")
    sae_model, d_in, _d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=device)

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
                "sae_steer now expects each input image to be a single tile. "
                f"Got z-grid shape {(gh, gw)} for image {img_path}. "
                "Use cropped tile images, or set tile/grid sizes so the grid is 1x1."
            )
        if d_in != d:
            raise ValueError(f"SAE d_in={d_in} does not match UNI feature dim D={d}")

        print(f"[tile {i}/{len(image_paths)}] Applying SAE edit: {args.steer_mode} @ latent {args.latent_idx}")
        clamp_value = _resolve_clamp_value(args, clamp_arr)
        z_edit, dbg = edit_uni_z_grid_with_sae(
            sae_model=sae_model,
            z_grid=z,
            latent_idx=int(args.latent_idx),
            target_value=float(args.target_value) if args.steer_mode == "latent_target" else None,
            delta=float(args.delta) if args.steer_mode == "latent_delta" else None,
            scale=float(args.scale) if args.steer_mode == "latent_scale" else None,
            clamp_value=clamp_value if args.steer_mode == "latent_clamp" else None,
            tile_mask=None,  # For tile-only inputs, edit the single tile directly.
            blend=float(args.blend),
            latent_strength=float(args.latent_strength),
            soft_mask_sigma=0.0,  # No spatial smoothing needed for a single-tile grid.
            max_feature_delta_norm=float(args.max_feature_delta_norm) if args.max_feature_delta_norm is not None else None,
            keep_non_selected=True,
            return_debug=True,
        )

        with torch.no_grad():
            z_diff = (z_edit - z).float()
            diff_stats = {
                "mean_abs": float(z_diff.abs().mean().item()),
                "max_abs": float(z_diff.abs().max().item()),
                "mean_l2": float(z_diff.view(-1, z_diff.shape[-1]).norm(dim=1).mean().item()),
            }
            dbg["z_edit_diff"] = diff_stats

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
        out_img = item_out_dir / "sae_steer_edit.png"
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
            "steer_mode": args.steer_mode,
            "latent_idx": int(args.latent_idx),
            "delta": args.delta,
            "target_value": args.target_value,
            "scale": args.scale,
            "clamp_value": clamp_value if args.steer_mode == "latent_clamp" else None,
            "clamp_values_path": args.clamp_values_path,
            "blend": args.blend,
            "latent_strength": args.latent_strength,
            "max_feature_delta_norm": args.max_feature_delta_norm,
            "reference_start_ratio": args.reference_start_ratio,
            "reference_mix": args.reference_mix,
            "debug": _sanitize_for_json(dbg),
        }
        (item_out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

        print("Saved:", out_img)
        print("Saved:", item_out_dir / "run_meta.json")


if __name__ == "__main__":
    main()
