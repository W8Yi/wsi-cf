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
        description=(
            "Optimized SAE sweep for tile image(s): loads UNI/SAE/PixCell once and sweeps many latents. "
            "Supports single-latent delta edits and full-SAE prototype target/delta edits."
        )
    )
    ap.add_argument("--image", type=str, default=None, help="Input tile image path.")
    ap.add_argument("--image-dir", type=str, default=None, help="Directory of tile images (png/jpg/tif).")
    ap.add_argument("--out-dir", type=str, required=True, help="Root output directory for sweep runs.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--steer-mode",
        type=str,
        default="latent_delta",
        choices=["latent_delta", "prototype_target", "prototype_delta"],
        help="SAE edit mode for the sweep.",
    )

    # Latent source and sweep values
    ap.add_argument(
        "--latent-contact-dir",
        type=str,
        required=True,
        help="Directory containing contact sheet images named latent_*.png.",
    )
    ap.add_argument(
        "--latent-ids",
        type=str,
        default="",
        help="Optional comma-separated latent IDs to steer. Filters the set discovered from --latent-contact-dir.",
    )
    ap.add_argument(
        "--deltas",
        type=str,
        default="-2,-1,-0.5,0.5,1,2",
        help=(
            "Comma-separated sweep values. "
            "For latent_delta mode these are latent deltas; "
            "for prototype_target mode these are interpolation strengths in [0,1]; "
            "for prototype_delta mode these are alphas in z' = z + alpha * (proto - baseline)."
        ),
    )
    ap.add_argument("--latent-limit", type=int, default=0, help="Optional cap on number of latents (0 = all).")
    ap.add_argument("--tile-limit", type=int, default=0, help="Optional cap on number of tiles (0 = all).")
    ap.add_argument(
        "--prototype-npz",
        type=str,
        default="",
        help="NPZ from scripts.build_sae_prototypes_from_pass2.py (required for prototype_* modes).",
    )
    ap.add_argument(
        "--prototype-key",
        type=str,
        default="prototype_median",
        choices=["prototype_mean", "prototype_median"],
        help="Which prototype vectors to use from --prototype-npz.",
    )
    ap.add_argument(
        "--prototype-baseline",
        type=str,
        default="global_mean",
        choices=["global_mean", "zero"],
        help="Baseline vector used in prototype_delta: direction = prototype - baseline.",
    )

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

    # SAE edit stabilizers
    ap.add_argument("--blend", type=float, default=0.35)
    ap.add_argument("--latent-strength", type=float, default=0.4)
    ap.add_argument(
        "--max-feature-delta-norm",
        type=float,
        default=None,
        help="Optional clamp on decoded UNI feature delta L2 norm per tile. Omit to disable.",
    )

    # Output / summary
    ap.add_argument("--save-z", action="store_true", help="Save z grids for every run (can use a lot of storage).")
    ap.add_argument("--make-summary", action="store_true", help="Build summary grids after the sweep completes.")
    ap.add_argument("--summary-out-dir", type=str, default="", help="Override summary output dir (default <out-dir>_summary).")
    ap.add_argument("--summary-tile-limit", type=int, default=24)
    ap.add_argument("--summary-latent-limit", type=int, default=0)
    ap.add_argument("--summary-thumb-size", type=int, default=192)
    ap.add_argument("--summary-include-diff", action="store_true")
    return ap


def _parse_float_list_csv(val: str) -> list[float]:
    out = [float(x.strip()) for x in val.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one numeric value in --deltas")
    return out


def _parse_int_list_csv(val: str) -> list[int]:
    out = [int(x.strip()) for x in val.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one integer value in --latent-ids")
    return out


def _slug_float(x: float) -> str:
    return str(float(x)).replace("-", "m").replace(".", "p")


def _latents_from_contact_dir(path: Path) -> list[int]:
    latents: list[int] = []
    seen: set[int] = set()

    for p in sorted(path.glob("latent_*.png")):
        stem = p.stem
        if not stem.startswith("latent_"):
            continue
        s = stem.replace("latent_", "")
        s = s.lstrip("0") or "0"
        lid = int(s)
        if lid not in seen:
            latents.append(lid)
            seen.add(lid)

    # Also support visualizer output layout: by_neuron/latent_<idx>/sheet.png
    for p in sorted(path.glob("latent_*")):
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("latent_"):
            continue
        s = name.replace("latent_", "")
        s = s.lstrip("0") or "0"
        lid = int(s)
        if lid not in seen:
            latents.append(lid)
            seen.add(lid)

    if not latents:
        raise ValueError(f"No latent_*.png files found in {path}")
    return latents


def _load_prototype_table(npz_path: Path, key: str) -> tuple[dict[int, np.ndarray], np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Prototype NPZ not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        if "latent_ids" not in data:
            raise KeyError(f"{npz_path}: missing array 'latent_ids'")
        if key not in data:
            raise KeyError(f"{npz_path}: missing array '{key}'")
        latent_ids = np.asarray(data["latent_ids"], dtype=np.int64).reshape(-1)
        protos = np.asarray(data[key], dtype=np.float32)
    if protos.ndim != 2:
        raise ValueError(f"{npz_path}:{key} expected [M,L], got {protos.shape}")
    if latent_ids.shape[0] != protos.shape[0]:
        raise ValueError(
            f"{npz_path}: latent_ids len {latent_ids.shape[0]} does not match {key} rows {protos.shape[0]}"
        )
    return {int(lid): protos[i] for i, lid in enumerate(latent_ids.tolist())}, protos


def _summarize_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    out = {"shape": list(arr.shape), "dtype": str(arr.dtype), "size": int(arr.size)}
    if arr.size and np.issubdtype(arr.dtype, np.number):
        arrf = arr.astype(np.float32, copy=False)
        out.update({"min": float(arrf.min()), "mean": float(arrf.mean()), "max": float(arrf.max())})
    return out


def _sanitize_for_json(value: Any) -> Any:
    if torch.is_tensor(value):
        return _summarize_array(value.detach().float().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _summarize_array(value)
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def main() -> None:
    args = _build_argparser().parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    image_paths = gather_image_paths(args.image, args.image_dir)
    if args.tile_limit and args.tile_limit > 0:
        image_paths = image_paths[: int(args.tile_limit)]
    if not image_paths:
        raise SystemExit("No input tiles selected.")

    latents = _latents_from_contact_dir(Path(args.latent_contact_dir))
    if args.latent_ids:
        requested = _parse_int_list_csv(args.latent_ids)
        allowed = set(requested)
        latents = [lid for lid in latents if lid in allowed]
    if args.latent_limit and args.latent_limit > 0:
        latents = latents[: int(args.latent_limit)]
    if not latents:
        raise SystemExit("No latents selected from contact directory.")

    sweep_values = _parse_float_list_csv(args.deltas)
    device = args.device

    prototype_table: dict[int, np.ndarray] | None = None
    prototype_matrix: np.ndarray | None = None
    prototype_baseline_vec: np.ndarray | None = None
    if args.steer_mode in {"prototype_target", "prototype_delta"}:
        if not args.prototype_npz:
            raise SystemExit(f"--prototype-npz is required when --steer-mode={args.steer_mode}")
        prototype_table, prototype_matrix = _load_prototype_table(Path(args.prototype_npz), args.prototype_key)
        if args.steer_mode == "prototype_delta":
            assert prototype_matrix is not None
            if args.prototype_baseline == "global_mean":
                prototype_baseline_vec = prototype_matrix.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
            elif args.prototype_baseline == "zero":
                prototype_baseline_vec = np.zeros((prototype_matrix.shape[1],), dtype=np.float32)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported prototype baseline {args.prototype_baseline}")

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

    total_runs = len(image_paths) * len(latents) * len(sweep_values)
    run_idx = 0

    sweep_meta = {
        "image_count": len(image_paths),
        "latent_count": len(latents),
        "latent_ids": latents,
        "steer_mode": args.steer_mode,
        "sweep_values": sweep_values,
        "deltas": sweep_values if args.steer_mode == "latent_delta" else None,
        "total_runs": total_runs,
        "out_dir": str(out_root),
        "contact_dir": str(args.latent_contact_dir),
        "prototype_npz": args.prototype_npz if args.prototype_npz else None,
        "prototype_key": args.prototype_key if args.prototype_npz else None,
        "prototype_baseline": args.prototype_baseline if args.steer_mode == "prototype_delta" else None,
        "reference_mix": args.reference_mix,
        "blend": args.blend,
        "latent_strength": args.latent_strength,
        "max_feature_delta_norm": args.max_feature_delta_norm,
        "notes": [
            "Each input tile is expected to produce a 1x1 UNI grid.",
            "Outputs are organized as latent_<idx>/<steer_mode>_strength_<tag>/<tile_stem>/...",
        ],
    }
    (out_root / "sweep_meta.json").write_text(json.dumps(sweep_meta, indent=2))

    for i, img_path in enumerate(image_paths, start=1):
        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        validate_min_image_size(w, h, min_size=256)

        out_h = int(args.out_h) if args.out_h is not None else h
        out_w = int(args.out_w) if args.out_w is not None else w
        validate_min_output_size(out_h, out_w, min_size=256)

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
                "sae_steer_sweep expects tile inputs that produce a 1x1 UNI grid. "
                f"Got {(gh, gw)} for {img_path}"
            )
        if d != d_in:
            raise ValueError(f"SAE d_in={d_in} does not match UNI feature dim D={d}")

        img_tensor_01 = pil_to_nchw_float01(img_pil)

        for latent_idx in latents:
            if (
                args.steer_mode in {"prototype_target", "prototype_delta"}
                and prototype_table is not None
                and int(latent_idx) not in prototype_table
            ):
                print(f"[skip] latent {latent_idx}: no prototype vector found in {args.prototype_npz}")
                continue
            for sweep_val in sweep_values:
                run_idx += 1
                tag = _slug_float(sweep_val)
                run_dir = out_root / f"latent_{latent_idx}" / f"{args.steer_mode}_strength_{tag}" / img_path.stem
                run_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"[{run_idx}/{total_runs}] {img_path.name} | latent={latent_idx} | "
                    f"{args.steer_mode}={sweep_val} -> {run_dir}"
                )

                if args.steer_mode == "latent_delta":
                    z_edit, dbg = edit_uni_z_grid_with_sae(
                        sae_model=sae_model,
                        z_grid=z,
                        latent_idx=int(latent_idx),
                        target_value=None,
                        delta=float(sweep_val),
                        scale=None,
                        clamp_value=None,
                        tile_mask=None,
                        blend=float(args.blend),
                        latent_strength=float(args.latent_strength),
                        soft_mask_sigma=0.0,
                        max_feature_delta_norm=float(args.max_feature_delta_norm)
                        if args.max_feature_delta_norm is not None
                        else None,
                        keep_non_selected=True,
                        return_debug=True,
                    )
                elif args.steer_mode == "prototype_target":
                    assert prototype_table is not None
                    z_edit, dbg = edit_uni_z_grid_with_sae(
                        sae_model=sae_model,
                        z_grid=z,
                        latent_idx=None,
                        target_latent_vector=prototype_table[int(latent_idx)],
                        target_latent_vector_strength=float(sweep_val),
                        target_value=None,
                        delta=None,
                        scale=None,
                        clamp_value=None,
                        tile_mask=None,
                        blend=float(args.blend),
                        latent_strength=float(args.latent_strength),
                        soft_mask_sigma=0.0,
                        max_feature_delta_norm=float(args.max_feature_delta_norm)
                        if args.max_feature_delta_norm is not None
                        else None,
                        keep_non_selected=True,
                        return_debug=True,
                    )
                elif args.steer_mode == "prototype_delta":
                    assert prototype_table is not None
                    assert prototype_baseline_vec is not None
                    proto_vec = prototype_table[int(latent_idx)]
                    dir_vec = np.asarray(proto_vec, dtype=np.float32) - np.asarray(prototype_baseline_vec, dtype=np.float32)
                    z_edit, dbg = edit_uni_z_grid_with_sae(
                        sae_model=sae_model,
                        z_grid=z,
                        latent_idx=None,
                        target_value=None,
                        target_latent_vector=None,
                        target_latent_vector_strength=1.0,
                        delta_latent_vector=dir_vec,
                        delta_latent_vector_scale=float(sweep_val),
                        delta=None,
                        scale=None,
                        clamp_value=None,
                        tile_mask=None,
                        blend=float(args.blend),
                        latent_strength=float(args.latent_strength),
                        soft_mask_sigma=0.0,
                        max_feature_delta_norm=float(args.max_feature_delta_norm)
                        if args.max_feature_delta_norm is not None
                        else None,
                        keep_non_selected=True,
                        return_debug=True,
                    )
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported steer mode {args.steer_mode}")

                with torch.no_grad():
                    z_diff = (z_edit - z).float()
                    dbg["z_edit_diff"] = {
                        "mean_abs": float(z_diff.abs().mean().item()),
                        "max_abs": float(z_diff.abs().max().item()),
                        "mean_l2": float(z_diff.view(-1, z_diff.shape[-1]).norm(dim=1).mean().item()),
                    }

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

                save_image_tensor_01(img_edit_t, run_dir / "sae_steer_edit.png")
                if args.save_z:
                    np.save(run_dir / "z_orig.npy", z.detach().float().cpu().numpy())
                    np.save(run_dir / "z_edit.npy", z_edit.detach().float().cpu().numpy())
                    np.save(run_dir / "coords_grid.npy", coords_grid)

                run_meta = {
                    "image": str(img_path),
                    "latent_idx": int(latent_idx),
                    "steer_mode": args.steer_mode,
                    "sweep_value": float(sweep_val),
                    "delta": float(sweep_val) if args.steer_mode == "latent_delta" else None,
                    "prototype_strength": float(sweep_val) if args.steer_mode == "prototype_target" else None,
                    "prototype_delta_alpha": float(sweep_val) if args.steer_mode == "prototype_delta" else None,
                    "prototype_key": args.prototype_key if args.steer_mode in {"prototype_target", "prototype_delta"} else None,
                    "prototype_baseline": args.prototype_baseline if args.steer_mode == "prototype_delta" else None,
                    "out_image": str(run_dir / "sae_steer_edit.png"),
                    "input_size": [h, w],
                    "output_size": [out_h, out_w],
                    "z_shape": [int(gh), int(gw), int(d)],
                    "blend": args.blend,
                    "latent_strength": args.latent_strength,
                    "max_feature_delta_norm": args.max_feature_delta_norm,
                    "reference_start_ratio": args.reference_start_ratio,
                    "reference_mix": args.reference_mix,
                    "debug": _sanitize_for_json(dbg),
                }
                (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    print("Sweep complete:", out_root)

    if args.make_summary:
        from concept_steer.visualize_sae_sweep import build_sae_sweep_summaries

        summary_out = (
            Path(args.summary_out_dir)
            if args.summary_out_dir
            else out_root.parent / f"{out_root.name}_summary"
        )
        print("[summary] Building summary grids...")
        build_sae_sweep_summaries(
            orig_dir=Path(args.image_dir if args.image_dir else Path(args.image).parent),
            sweep_root=out_root,
            out_dir=summary_out,
            tile_limit=int(args.summary_tile_limit),
            latent_limit=int(args.summary_latent_limit),
            strengths=sweep_values,
            thumb_size=int(args.summary_thumb_size),
            include_diff=bool(args.summary_include_diff),
        )
        print("[summary] Saved:", summary_out)


if __name__ == "__main__":
    main()
