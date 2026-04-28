from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, DiffusionPipeline

from utils.diffusion import build_uni_grid_from_pil, sample_multidiffusion_from_zgrid_with_midref
from utils.sae import load_sae_from_config
from utils.sae_edit import edit_uni_z_grid_with_sae
from utils.uni import get_uni


def _parse_list_f(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _tile_coords_from_args(
    gh: int,
    gw: int,
    tile_index: int | None,
    gy: int | None,
    gx: int | None,
    tile_policy: str,
    seed: int,
) -> tuple[int, int]:
    if tile_index is not None:
        if tile_index < 0 or tile_index >= gh * gw:
            raise ValueError(f"--tile-index must be in [0, {gh*gw-1}]")
        return tile_index // gw, tile_index % gw
    if gy is None or gx is None:
        if tile_policy == "center":
            return gh // 2, gw // 2
        if tile_policy == "random":
            rng = np.random.default_rng(seed)
            idx = int(rng.integers(0, gh * gw))
            return idx // gw, idx % gw
        raise ValueError(f"Unsupported tile_policy '{tile_policy}'")
    if gy < 0 or gy >= gh or gx < 0 or gx >= gw:
        raise ValueError(f"tile (gy,gx)=({gy},{gx}) out of range for grid ({gh},{gw})")
    return gy, gx


def _save_img(img_t: torch.Tensor, path: Path) -> np.ndarray:
    arr = (img_t[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return arr


def _crop_pad_tile(img_pil: Image.Image, x1: int, y1: int, tile_px: int) -> Image.Image:
    w, h = img_pil.size
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = min(w, x1 + tile_px)
    y2 = min(h, y1 + tile_px)
    patch = img_pil.crop((x1, y1, x2, y2))
    if patch.size != (tile_px, tile_px):
        bg = Image.new("RGB", (tile_px, tile_px), (255, 255, 255))
        bg.paste(patch, (0, 0))
        patch = bg
    return patch


def _tile_rest_mse(a: np.ndarray, b: np.ndarray, gy: int, gx: int, tile_px: int) -> tuple[float, float, float]:
    h, w = a.shape[:2]
    y1, y2 = gy * tile_px, min((gy + 1) * tile_px, h)
    x1, x2 = gx * tile_px, min((gx + 1) * tile_px, w)
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    tile_mse = float(d[y1:y2, x1:x2].mean())
    mask = np.ones((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = False
    rest_mse = float(d[mask].mean())
    ratio = tile_mse / (rest_mse + 1e-8)
    return tile_mse, rest_mse, ratio


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-tile SAE steering sweep with diffusion regeneration.")
    ap.add_argument("--image", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--seed", default=0, type=int)

    ap.add_argument("--sae-ckpt", required=True, type=str)
    ap.add_argument("--sae-cfg", required=True, type=str)
    ap.add_argument("--latent-idx", required=True, type=int)
    ap.add_argument("--mode", default="delta", choices=["delta", "target"], type=str)

    ap.add_argument("--deltas", default="-1.0,-0.5,0.0,0.5,1.0", type=str)
    ap.add_argument("--targets", default="0.0,1.0,2.0,4.0", type=str)
    ap.add_argument("--blends", default="0.25,0.35", type=str)
    ap.add_argument("--sigmas", default="0.8,1.2", type=str)
    ap.add_argument("--latent-strengths", default="0.4", type=str)
    ap.add_argument("--max-feature-delta-norms", default="6.0", type=str)
    ap.add_argument("--max-runs", default=0, type=int, help="0 means run all combinations.")

    ap.add_argument("--tile-px", default=256, type=int)
    ap.add_argument("--grid-step-px", default=256, type=int)
    ap.add_argument("--reconstruct-scope", default="tile", choices=["tile", "full"], type=str)
    ap.add_argument("--tile-policy", default="random", choices=["random", "center"], type=str)
    ap.add_argument("--tile-index", default=None, type=int)
    ap.add_argument("--gy", default=None, type=int)
    ap.add_argument("--gx", default=None, type=int)

    ap.add_argument("--out-h", default=None, type=int)
    ap.add_argument("--out-w", default=None, type=int)
    ap.add_argument("--steps", default=30, type=int)
    ap.add_argument("--guidance", default=2.0, type=float)
    ap.add_argument("--patch-px", default=256, type=int)
    ap.add_argument("--stride-px", default=128, type=int)
    ap.add_argument("--patch-batch", default=256, type=int)
    ap.add_argument("--reference-start-ratio", default=0.1, type=float)
    ap.add_argument("--reference-mix", default=0.1, type=float)

    ap.add_argument("--pixcell-model", default="StonyBrook-CVLab/PixCell-256", type=str)
    ap.add_argument("--pixcell-custom-pipeline", default="StonyBrook-CVLab/PixCell-pipeline", type=str)
    ap.add_argument("--vae-model", default="stabilityai/stable-diffusion-3.5-large", type=str)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_pil = Image.open(args.image).convert("RGB")
    w, h = img_pil.size
    if min(w, h) < 256:
        raise ValueError(f"Input image too small: {(w, h)}; minimum is 256x256")
    out_h = args.out_h or h
    out_w = args.out_w or w
    if min(out_h, out_w) < 256:
        raise ValueError(f"Output size too small: {(out_w, out_h)}; minimum is 256x256")

    device = args.device
    print("[1/6] load UNI")
    uni, uni_transform = get_uni(device)

    print("[2/6] build z-grid")
    z, coords_grid = build_uni_grid_from_pil(
        img_pil,
        uni,
        uni_transform,
        tile_px=args.tile_px,
        grid_step_px=args.grid_step_px,
        device=device,
        dtype=torch.float16,
    )
    gh, gw, d = z.shape
    gy, gx = _tile_coords_from_args(
        gh,
        gw,
        args.tile_index,
        args.gy,
        args.gx,
        args.tile_policy,
        seed=args.seed,
    )
    print(f"grid=({gh},{gw},{d}) tile=(gy={gy},gx={gx}) scope={args.reconstruct_scope}")

    print("[3/6] load SAE")
    sae, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=device)
    if d_in != d:
        raise ValueError(f"SAE d_in={d_in} != UNI D={d}")
    if args.latent_idx < 0 or args.latent_idx >= d_latent:
        raise ValueError(f"--latent-idx must be in [0, {d_latent-1}]")

    print("[4/6] load diffusion")
    sd3_vae = AutoencoderKL.from_pretrained(args.vae_model, subfolder="vae")
    pipe = DiffusionPipeline.from_pretrained(
        args.pixcell_model,
        vae=sd3_vae,
        custom_pipeline=args.pixcell_custom_pipeline,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    if args.reconstruct_scope == "tile":
        tx1, ty1 = [int(v) for v in coords_grid[gy, gx]]
        img_work_pil = _crop_pad_tile(img_pil, tx1, ty1, args.tile_px)
        z_work = z[gy:gy + 1, gx:gx + 1, :]
        out_h_work = args.tile_px
        out_w_work = args.tile_px
        tile_mask = np.ones((1, 1), dtype=np.float32)
    else:
        img_work_pil = img_pil
        z_work = z
        out_h_work = out_h
        out_w_work = out_w
        tile_mask = np.zeros((gh, gw), dtype=np.float32)
        tile_mask[gy, gx] = 1.0

    img_tensor_01 = torch.from_numpy(np.asarray(img_work_pil).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    g0 = torch.Generator(device=device).manual_seed(args.seed)

    print("[5/6] baseline regen")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        base_t = sample_multidiffusion_from_zgrid_with_midref(
            pipeline=pipe,
            z_grid=z_work,
            original_image=img_tensor_01,
            out_h=out_h_work,
            out_w=out_w_work,
            patch_px=args.patch_px,
            stride_px=args.stride_px,
            steps=args.steps,
            guidance=args.guidance,
            patch_batch=args.patch_batch,
            reference_start_ratio=args.reference_start_ratio,
            reference_mix=args.reference_mix,
            generator=g0,
        )
    base_np = _save_img(base_t, out_dir / "baseline.png")

    values = _parse_list_f(args.deltas if args.mode == "delta" else args.targets)
    blends = _parse_list_f(args.blends)
    sigmas = _parse_list_f(args.sigmas)
    latent_strengths = _parse_list_f(args.latent_strengths)
    max_norms = _parse_list_f(args.max_feature_delta_norms)
    if not (values and blends and sigmas and latent_strengths and max_norms):
        raise ValueError("Sweep lists must be non-empty.")

    combos = list(itertools.product(values, blends, sigmas, latent_strengths, max_norms))
    planned = len(combos)
    if args.max_runs > 0:
        combos = combos[: args.max_runs]
    print(f"sweep combinations: planned={planned}, running={len(combos)}")

    rows = []
    print("[6/6] run sweeps")
    for value, blend, sigma, lstr, max_norm in combos:
        z_edit, dbg = edit_uni_z_grid_with_sae(
            sae_model=sae,
            z_grid=z_work,
            latent_idx=args.latent_idx,
            target_value=value if args.mode == "target" else None,
            delta=value if args.mode == "delta" else None,
            tile_mask=tile_mask,
            blend=blend,
            latent_strength=lstr,
            soft_mask_sigma=sigma,
            max_feature_delta_norm=max_norm,
            keep_non_selected=True,
            return_debug=True,
        )

        g = torch.Generator(device=device).manual_seed(args.seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            img_t = sample_multidiffusion_from_zgrid_with_midref(
                pipeline=pipe,
                z_grid=z_edit,
                original_image=img_tensor_01,
                out_h=out_h_work,
                out_w=out_w_work,
                patch_px=args.patch_px,
                stride_px=args.stride_px,
                steps=args.steps,
                guidance=args.guidance,
                patch_batch=args.patch_batch,
                reference_start_ratio=args.reference_start_ratio,
                reference_mix=args.reference_mix,
                generator=g,
            )

        name = (
            f"{args.mode}_v{value:+.3f}_b{blend:.3f}_s{sigma:.3f}_"
            f"ls{lstr:.3f}_mn{max_norm:.3f}.png"
        )
        out_path = out_dir / name
        img_np = _save_img(img_t, out_path)
        if args.reconstruct_scope == "tile":
            tile_mse = float(((base_np.astype(np.float32) - img_np.astype(np.float32)) ** 2).mean())
            rest_mse = float("nan")
            ratio = float("nan")
        else:
            tile_mse, rest_mse, ratio = _tile_rest_mse(base_np, img_np, gy, gx, args.tile_px)
        rows.append(
            {
                "mode": args.mode,
                "value": float(value),
                "blend": float(blend),
                "soft_mask_sigma": float(sigma),
                "latent_strength": float(lstr),
                "max_feature_delta_norm": float(max_norm),
                "tile_mse": tile_mse,
                "rest_mse": rest_mse,
                "tile_rest_ratio": ratio,
                "out_image": str(out_path),
                "dbg_blend_weight_mean": float(dbg.get("blend_weight_mean", 0.0)),
            }
        )
        if args.reconstruct_scope == "tile":
            print(f"done {name} | tile_only_mse={tile_mse:.3f}")
        else:
            print(f"done {name} | tile_mse={tile_mse:.3f} rest_mse={rest_mse:.3f} ratio={ratio:.3f}")

    csv_path = out_dir / "sweep_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    meta = {
        "args": vars(args),
        "grid_shape": [int(gh), int(gw), int(d)],
        "reconstruct_scope": args.reconstruct_scope,
        "tile": {"gy": int(gy), "gx": int(gx), "index": int(gy * gw + gx)},
        "num_runs": len(rows),
        "best_ratio": max(float(r["tile_rest_ratio"]) for r in rows) if rows else None,
    }
    with open(out_dir / "sweep_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("saved:", csv_path)
    print("saved:", out_dir / "sweep_meta.json")


if __name__ == "__main__":
    main()
