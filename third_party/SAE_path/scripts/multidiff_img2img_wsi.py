#!/usr/bin/env python3
"""
MultiDiff img2img pipeline for WSI research.

What this script supports:
1) Input from either a standard image (--input_image) or a WSI slide (--input_svs).
2) Region control using x/y/width/height, with random tissue-aware sampling for SVS when x/y are omitted.
3) PixCell MultiDiffusion generation conditioned by a UNI2 feature grid.
4) Optional img2img-style partial denoising via --strength.
5) Optional tile-level steering by replacing or blending tile UNI embeddings.
6) Optional tiled VAE encode/decode (default is OFF for both).

Argument reference:
- --input_image
  Path to a regular image file. Mutually exclusive with --input_svs.
- --input_svs
  Path to a whole-slide image (.svs/.tif/.tiff/.ndpi/.mrxs). Mutually exclusive with --input_image.
- --x, --y
  Top-left region coordinate in source pixels. For SVS:
  if both are omitted, the script samples a random tissue-rich region.
  For image input, omitted values default to 0,0.
- --region_w, --region_h
  Region width and height in pixels. Required for --input_svs.
  For --input_image, if omitted they default to full remaining image size from (x,y).
- --out_dir
  Output directory where generated image and optional real crop are saved.
- --save_real
  Save source region/crop image as *_real.png.
- --device
  PyTorch device string, e.g. cuda:0, cuda:1, cpu.
- --dtype
  Internal inference precision for model forward pass: fp16 or fp32.
- --seed
  Random seed for NumPy/Torch/random; controls region sampling and generation noise.
- --patch_px
  Diffusion window size in image pixels (converted to latent window using VAE scale).
- --stride_px
  Diffusion window stride in image pixels. Smaller stride increases overlap and smoothness.
- --grid_step_px
  Spacing of UNI grid sampling in image pixels. Lower values create denser conditioning grids.
- --steps
  Number of denoising steps.
- --guidance
  Classifier-free guidance scale; higher values push stronger conditional adherence.
- --patch_batch
  Number of windows processed per transformer call. Higher values are faster but use more VRAM.
- --strength
  Img2img denoising strength in [0,1]. 0.0 means pure generation from noise.
  Higher values start denoising later (more deviation from source appearance).
- --steer_tile gx,gy,path
  Tile steering override. Repeat this argument to steer multiple tiles.
  gx/gy are tile indices in the UNI grid, and path points to a feature vector file.
  Supported formats: .npy, .pt, .pth (plain tensor/array).
- --steer_blend
  Blend factor in [0,1] for steering. 1.0 fully replaces the tile embedding.
- --pix_model_id
  Hugging Face model id for PixCell model weights.
- --pix_pipeline_id
  Custom pipeline id for PixCell inference pipeline code.
- --vae_model_id, --vae_subfolder
  VAE model source used for latent encode/decode.
- --use_tiled_vae_encode
  Enable tiled VAE encode for init latents (useful for very large regions, slower).
  Default is OFF.
- --use_tiled_vae_decode
  Enable tiled VAE decode for final image decode (useful for very large outputs, slower).
  Default is OFF.
- --encode_tile_img, --encode_overlap_img
  Tile size/overlap for tiled VAE encode (image-space pixels).
- --decode_tile_lat, --decode_overlap_lat
  Tile size/overlap for tiled VAE decode (latent-space pixels).
- --max_region_tries
  Number of random SVS region attempts before falling back to best tissue score.
- --min_tissue
  Minimum simple tissue score threshold for early accept in random SVS sampling.
- --svs_root, --svs_exts
  Optional helper settings for slide indexing logic and extension filtering.

Tile steering concept:
- Conditioning grid shape is [Gh, Gw, D].
- Each --steer_tile targets one grid location (gx, gy).
- Feature dimension must match D (for UNI2-h this is typically 1536).
- Steering is applied before diffusion starts.

Possible upgrades for research:
1) Add multi-scale conditioning (coarse + fine UNI grids) and fuse them per denoise step.
2) Add region masks so steering only affects selected tissue classes or polygons.
3) Add schedule-aware steering where tile overrides vary by timestep.
4) Add neighborhood-consistent steering (smooth/regularize across nearby tiles).
5) Add per-tile strength maps instead of a single global --steer_blend.
6) Add stronger stain normalization/preprocessing before UNI embedding extraction.
7) Cache UNI features on disk for repeated experiments over same region.
8) Add deterministic experiment manifests (JSON config + exact model revisions + hash logging).
9) Add direct support for slide pyramid levels and micron-per-pixel normalization.
10) Add batched region jobs for large-scale sweeps with automatic retry/checkpointing.
11) Add quantitative evaluation hooks (FID/KID, morphology metrics, downstream probe score).
12) Add explicit seam metrics to tune patch/stride/gaussian blending settings.
13) Add optional control adapters (e.g., edge/seg maps) as extra conditional channels.
14) Add mixed steering sources (UNI feature + text prompt + class token).
15) Add optional post-hoc tile replacement/compositing for local edits only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import openslide
from PIL import Image

import torch
import torch.nn.functional as F
import timm
from diffusers import AutoencoderKL, DiffusionPipeline
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.diffusion import (
    _align_uncond_embedding,
    _lookup_window_cond,
    infer_pixcell_cond_grid_side,
    infer_pixcell_native_patch_px,
)


SVS_EXTS_DEFAULT = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}


def save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=True)


def read_region_rgb(slide: openslide.OpenSlide, x0: int, y0: int, w: int, h: int) -> Image.Image:
    rgba = slide.read_region((int(x0), int(y0)), 0, (int(w), int(h))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def index_svs_files(root: Path, exts: Sequence[str]) -> List[Path]:
    exts_l = {e.lower() for e in exts}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_l]
    files.sort()
    return files


def quick_tissue_score(img: Image.Image) -> float:
    arr = np.asarray(img, dtype=np.uint8)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
    return float(np.mean(gray < 230.0))


def pick_random_region(
    slide: openslide.OpenSlide,
    region_w: int,
    region_h: int,
    n_tries: int,
    min_tissue: float,
    rng: random.Random,
) -> Tuple[int, int]:
    W, H = slide.dimensions
    if W <= region_w or H <= region_h:
        return 0, 0

    best = (0, 0, -1.0)
    for _ in range(max(1, n_tries)):
        x0 = rng.randint(0, W - region_w)
        y0 = rng.randint(0, H - region_h)
        thumb = read_region_rgb(slide, x0, y0, min(256, region_w), min(256, region_h))
        score = quick_tissue_score(thumb)
        if score > best[2]:
            best = (x0, y0, score)
        if score >= min_tissue:
            return x0, y0
    return best[0], best[1]


def load_uni2(device: torch.device):
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    tfm = create_transform(**resolve_data_config(uni.pretrained_cfg, model=uni))
    uni.eval().to(device=device)
    return uni, tfm


@torch.no_grad()
def build_uni_grid_from_image(
    img: Image.Image,
    uni_model,
    uni_transform,
    grid_step_px: int,
    device: torch.device,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    H, W = arr.shape[:2]
    gh = math.ceil(H / grid_step_px)
    gw = math.ceil(W / grid_step_px)
    half = grid_step_px // 2

    rows = []
    for gy in range(gh):
        cols = []
        for gx in range(gw):
            cx = int((gx + 0.5) * grid_step_px)
            cy = int((gy + 0.5) * grid_step_px)
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(W, x0 + grid_step_px)
            y1 = min(H, y0 + grid_step_px)
            patch = arr[y0:y1, x0:x1]
            if patch.shape[0] != grid_step_px or patch.shape[1] != grid_step_px:
                canvas = np.full((grid_step_px, grid_step_px, 3), 255, dtype=np.uint8)
                canvas[: patch.shape[0], : patch.shape[1]] = patch
                patch = canvas
            patch_pil = Image.fromarray(patch)
            inp = uni_transform(patch_pil).unsqueeze(0).to(device=device)
            emb = uni_model(inp).squeeze(0).to(dtype=torch.float32)
            cols.append(emb)
        rows.append(torch.stack(cols, dim=0))
    z_grid = torch.stack(rows, dim=0)
    return z_grid.to(device=device, dtype=out_dtype)


def _parse_steer_spec(spec: str) -> Tuple[int, int, Path]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Bad --steer_tile '{spec}'. Expected format: gx,gy,path.npy")
    gx = int(parts[0])
    gy = int(parts[1])
    fp = Path(parts[2])
    return gx, gy, fp


def _load_steer_manifest(manifest_path: Path) -> List[str]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Steer manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Steer manifest must be a JSON list: {manifest_path}")
    specs: List[str] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Steer manifest item {i} must be an object")
        if "gx" not in item or "gy" not in item or "path" not in item:
            raise ValueError(f"Steer manifest item {i} must contain gx, gy, path")
        gx = int(item["gx"])
        gy = int(item["gy"])
        path = Path(str(item["path"]))
        specs.append(f"{gx},{gy},{path}")
    return specs


def _load_feature_vector(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Steer feature file not found: {path}")
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        vec = torch.from_numpy(arr)
    elif path.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            raise ValueError(f"Steer file {path} is a dict; expected a plain tensor or array.")
        vec = obj
    else:
        raise ValueError(f"Unsupported steer feature format: {path.suffix}. Use .npy or .pt")
    vec = torch.as_tensor(vec, dtype=torch.float32).flatten()
    return vec.to(device=device, dtype=dtype)


def apply_tile_steering(
    z_grid: torch.Tensor,  # [Gh, Gw, D]
    steer_specs: Sequence[str],
    steer_blend: float,
) -> torch.Tensor:
    if len(steer_specs) == 0:
        return z_grid
    gh, gw, d = z_grid.shape
    out = z_grid.clone()
    alpha = float(steer_blend)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("--steer_blend must be in [0,1]")

    for spec in steer_specs:
        gx, gy, path = _parse_steer_spec(spec)
        if gy < 0 or gy >= gh or gx < 0 or gx >= gw:
            raise ValueError(f"Steer tile ({gx},{gy}) out of range for grid [Gh={gh}, Gw={gw}]")
        vec = _load_feature_vector(path, device=out.device, dtype=out.dtype)
        if vec.numel() != d:
            raise ValueError(
                f"Steer feature dim mismatch for {path}: got {vec.numel()}, expected {d}"
            )
        if alpha >= 1.0:
            out[gy, gx] = vec
        elif alpha > 0.0:
            out[gy, gx] = (1.0 - alpha) * out[gy, gx] + alpha * vec
    return out


def _resolve_pixcell_window_config(
    pix_model_id: str,
    patch_px: int,
    stride_px: int,
) -> tuple[int, int, int]:
    resolved_patch_px = int(patch_px) if int(patch_px) > 0 else int(infer_pixcell_native_patch_px(pix_model_id))
    resolved_stride_px = int(stride_px) if int(stride_px) > 0 else max(1, resolved_patch_px // 2)
    resolved_cond_grid_side = int(infer_pixcell_cond_grid_side(pix_model_id))
    if min(resolved_patch_px, resolved_stride_px, resolved_cond_grid_side) <= 0:
        raise ValueError("Resolved PixCell window settings must all be > 0.")
    return resolved_patch_px, resolved_stride_px, resolved_cond_grid_side


@torch.no_grad()
def vae_decode_tiled(
    vae,
    latents: torch.Tensor,  # [1,C,H,W]
    tile_lat: int,
    overlap_lat: int,
) -> torch.Tensor:
    device = latents.device
    vae_dtype = next(vae.parameters()).dtype
    _, _, H, W = latents.shape

    if tile_lat <= 0:
        raise ValueError("tile_lat must be > 0")
    if overlap_lat < 0 or overlap_lat >= tile_lat:
        raise ValueError("overlap_lat must satisfy 0 <= overlap_lat < tile_lat")

    step = max(1, tile_lat - overlap_lat)
    yy = torch.linspace(-1.0, 1.0, tile_lat, device=device, dtype=vae_dtype)[:, None]
    xx = torch.linspace(-1.0, 1.0, tile_lat, device=device, dtype=vae_dtype)[None, :]
    win = torch.exp(-2.0 * (yy**2 + xx**2))
    win = (win / win.max().clamp(min=1e-8))[None, None]

    out = None
    wgt = None
    scale = None

    for top in range(0, H, step):
        for left in range(0, W, step):
            top0 = min(top, H - tile_lat)
            left0 = min(left, W - tile_lat)
            z = latents[:, :, top0 : top0 + tile_lat, left0 : left0 + tile_lat].to(dtype=vae_dtype)
            x = vae.decode(z, return_dict=True).sample

            if out is None:
                scale_h = x.shape[-2] // tile_lat
                scale_w = x.shape[-1] // tile_lat
                if scale_h != scale_w:
                    raise RuntimeError("Unexpected non-uniform VAE decode scale.")
                scale = scale_h
                out = torch.zeros((1, x.shape[1], H * scale, W * scale), device=device, dtype=x.dtype)
                wgt = torch.zeros((1, 1, H * scale, W * scale), device=device, dtype=x.dtype)

            win_img = F.interpolate(win.to(dtype=x.dtype), size=x.shape[-2:], mode="bilinear", align_corners=False)
            y_img = top0 * scale
            x_img = left0 * scale
            out[:, :, y_img : y_img + x.shape[-2], x_img : x_img + x.shape[-1]] += x * win_img
            wgt[:, :, y_img : y_img + x.shape[-2], x_img : x_img + x.shape[-1]] += win_img

    return out / wgt.clamp(min=1e-8)


@torch.no_grad()
def vae_encode_tiled(
    vae,
    img01: torch.Tensor,  # [1,3,H,W] in [0,1]
    tile_img: int,
    overlap_img: int,
) -> torch.Tensor:
    device = img01.device
    dtype = img01.dtype
    _, _, H, W = img01.shape

    if tile_img <= 0:
        raise ValueError("tile_img must be > 0")
    if overlap_img < 0 or overlap_img >= tile_img:
        raise ValueError("overlap_img must satisfy 0 <= overlap_img < tile_img")

    step = max(1, tile_img - overlap_img)
    yy = torch.linspace(-1.0, 1.0, tile_img, device=device, dtype=dtype)[:, None]
    xx = torch.linspace(-1.0, 1.0, tile_img, device=device, dtype=dtype)[None, :]
    win = torch.exp(-2.0 * (yy**2 + xx**2))
    win = (win / win.max().clamp(min=1e-8))[None, None]

    out = None
    wgt = None
    scale = None

    for top in range(0, H, step):
        for left in range(0, W, step):
            top0 = min(top, H - tile_img)
            left0 = min(left, W - tile_img)

            x = img01[:, :, top0 : top0 + tile_img, left0 : left0 + tile_img]
            x = (x * 2.0 - 1.0).to(dtype)
            z = vae.encode(x, return_dict=True).latent_dist.sample()

            if out is None:
                scale_h = tile_img // z.shape[-2]
                scale_w = tile_img // z.shape[-1]
                if scale_h != scale_w:
                    raise RuntimeError("Unexpected non-uniform VAE encode scale.")
                scale = scale_h
                h_lat = math.ceil(H / scale)
                w_lat = math.ceil(W / scale)
                out = torch.zeros((1, z.shape[1], h_lat, w_lat), device=device, dtype=z.dtype)
                wgt = torch.zeros((1, 1, h_lat, w_lat), device=device, dtype=z.dtype)

            win_lat = F.interpolate(win, size=z.shape[-2:], mode="bilinear", align_corners=False)
            y_lat = top0 // scale
            x_lat = left0 // scale
            out[:, :, y_lat : y_lat + z.shape[-2], x_lat : x_lat + z.shape[-1]] += z * win_lat
            wgt[:, :, y_lat : y_lat + z.shape[-2], x_lat : x_lat + z.shape[-1]] += win_lat

    return out / wgt.clamp(min=1e-8)


@torch.no_grad()
def vae_encode_auto(
    vae,
    img01: torch.Tensor,
    use_tiled: bool,
    tile_img: int,
    overlap_img: int,
) -> torch.Tensor:
    if use_tiled:
        return vae_encode_tiled(vae, img01, tile_img=tile_img, overlap_img=overlap_img)
    x = (img01 * 2.0 - 1.0)
    return vae.encode(x, return_dict=True).latent_dist.sample()


@torch.no_grad()
def vae_decode_auto(
    vae,
    latents: torch.Tensor,
    use_tiled: bool,
    tile_lat: int,
    overlap_lat: int,
) -> torch.Tensor:
    if use_tiled:
        return vae_decode_tiled(vae, latents, tile_lat=tile_lat, overlap_lat=overlap_lat)
    return vae.decode(latents, return_dict=True).sample


@torch.no_grad()
def sample_large_pixcell_multidiffusion(
    pipeline,
    z_grid: torch.Tensor,  # [Gh,Gw,D] or [1,Gh,Gw,D]
    out_h: int,
    out_w: int,
    patch_px: int,
    stride_px: int,
    cond_grid_side: int,
    guidance_scale: float,
    num_steps: int,
    patch_batch: int,
    strength: float,
    init_latents: Optional[torch.Tensor],  # unscaled latents
    use_tiled_vae_decode: bool,
    decode_tile_lat: int,
    decode_overlap_lat: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    device = pipeline.device
    dtype = next(pipeline.transformer.parameters()).dtype
    vae_scale = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    h_lat = math.ceil(out_h / vae_scale)
    w_lat = math.ceil(out_w / vae_scale)

    ph = patch_px // vae_scale
    pw = patch_px // vae_scale
    sh = stride_px // vae_scale
    sw = stride_px // vae_scale
    if min(ph, pw, sh, sw) <= 0:
        raise ValueError("patch_px/stride_px must be >= VAE scale.")

    if z_grid.dim() == 3:
        z_grid = z_grid.unsqueeze(0)
    if z_grid.shape[0] != 1:
        raise ValueError("This script currently supports batch size 1.")

    _, gh, gw, d = z_grid.shape
    c = pipeline.vae.config.latent_channels

    yy = torch.arange(ph, device=device, dtype=dtype)[:, None]
    xx = torch.arange(pw, device=device, dtype=dtype)[None, :]
    cy = (ph - 1) / 2.0
    cx = (pw - 1) / 2.0
    sigma_y = max(ph / 4.0, 1e-6)
    sigma_x = max(pw / 4.0, 1e-6)
    gauss = torch.exp(-0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2))
    gauss = (gauss / gauss.max().clamp(min=1e-8))[None, None]

    coords = [
        (top, left)
        for top in range(0, h_lat - ph + 1, sh)
        for left in range(0, w_lat - pw + 1, sw)
    ]
    if len(coords) == 0:
        raise ValueError("No valid windows. Check patch/stride and output size.")

    pipeline.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    has_scale = hasattr(pipeline.scheduler, "scale_model_input")
    patch_batch = max(1, int(patch_batch))

    if strength <= 0.0 or init_latents is None:
        latents = torch.randn((1, c, h_lat, w_lat), device=device, dtype=dtype, generator=generator)
        start_idx = 0
    else:
        init_lat = init_latents.to(device=device, dtype=dtype)
        init_lat = init_lat[:, :, :h_lat, :w_lat]
        if hasattr(pipeline.vae.config, "scaling_factor"):
            init_lat = init_lat * pipeline.vae.config.scaling_factor
        strength = float(max(0.0, min(strength, 1.0)))
        start_idx = int(strength * (len(timesteps) - 1))
        start_idx = max(0, min(start_idx, len(timesteps) - 1))
        t_start = timesteps[start_idx]
        if not torch.is_tensor(t_start):
            t_start = torch.tensor(t_start, device=device)
        t_start_b = t_start.to(device=device).expand(init_lat.shape[0])
        noise = torch.randn_like(init_lat)
        latents = pipeline.scheduler.add_noise(init_lat, noise, t_start_b)

    timesteps = timesteps[start_idx:]

    for t_idx, t in enumerate(tqdm(timesteps, desc="Diffusion steps", leave=False)):
        t_model = t if torch.is_tensor(t) else torch.tensor(t, device=device)
        t_model = t_model.to(device=device).long()
        if t_model.dim() == 0:
            t_model = t_model[None]

        latents_in = pipeline.scheduler.scale_model_input(latents, t) if has_scale else latents
        eps_accum = torch.zeros_like(latents)
        weight = torch.zeros((1, 1, h_lat, w_lat), device=device, dtype=dtype)

        for i0 in tqdm(
            range(0, len(coords), patch_batch),
            desc=f"Windows @ step {t_idx + 1}/{len(timesteps)}",
            leave=False,
        ):
            chunk = coords[i0 : i0 + patch_batch]
            n = len(chunk)
            patch_in = torch.stack(
                [latents_in[:, :, top : top + ph, left : left + pw].squeeze(0) for (top, left) in chunk],
                dim=0,
            )
            cond = torch.cat(
                [
                    _lookup_window_cond(
                        z_grid,
                        top_lat=top,
                        left_lat=left,
                        ph=ph,
                        pw=pw,
                        h_lat=h_lat,
                        w_lat=w_lat,
                        cond_grid_side=cond_grid_side,
                    )
                    for (top, left) in chunk
                ],
                dim=0,
            ).to(device=device, dtype=dtype)
            uncond = _align_uncond_embedding(
                pipeline.get_unconditional_embedding(n).to(device=device, dtype=dtype),
                cond,
            )

            hs = torch.cat([patch_in, patch_in], dim=0)
            es = torch.cat([uncond, cond], dim=0)
            tt = t_model.expand(hs.shape[0])
            out = pipeline.transformer(
                hidden_states=hs,
                encoder_hidden_states=es,
                timestep=tt,
                return_dict=True,
            )
            eps2 = out.sample if hasattr(out, "sample") else out[0]
            if eps2.shape[1] == 2 * c:
                eps2 = eps2[:, :c]
            elif eps2.shape[1] != c:
                raise RuntimeError(f"Model output channels={eps2.shape[1]}, expected {c} or {2*c}")
            eps_u, eps_c = eps2[:n], eps2[n:]
            eps = eps_u + guidance_scale * (eps_c - eps_u)

            for j, (top, left) in enumerate(chunk):
                eps_accum[:, :, top : top + ph, left : left + pw] += eps[j : j + 1] * gauss
                weight[:, :, top : top + ph, left : left + pw] += gauss

        eps_full = eps_accum / weight.clamp(min=1e-8)
        step = pipeline.scheduler.step(eps_full, t, latents, return_dict=True)
        latents = step.prev_sample if hasattr(step, "prev_sample") else step[0]

    if hasattr(pipeline.vae.config, "scaling_factor"):
        latents = latents / pipeline.vae.config.scaling_factor

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    img = vae_decode_auto(
        pipeline.vae,
        latents,
        use_tiled=use_tiled_vae_decode,
        tile_lat=decode_tile_lat,
        overlap_lat=decode_overlap_lat,
    )
    return (img / 2 + 0.5).clamp(0, 1)


def _load_source_region(
    args,
    rng: random.Random,
) -> Tuple[Image.Image, str]:
    if args.input_image is not None:
        img = Image.open(args.input_image).convert("RGB")
        W, H = img.size
        x = args.x if args.x is not None else 0
        y = args.y if args.y is not None else 0
        w = args.region_w if args.region_w is not None else (W - x)
        h = args.region_h if args.region_h is not None else (H - y)
        if w <= 0 or h <= 0:
            raise ValueError("Invalid image crop size.")
        if x < 0 or y < 0 or x + w > W or y + h > H:
            raise ValueError(f"Image crop ({x},{y},{w},{h}) is out of bounds for image size ({W},{H})")
        crop = img.crop((x, y, x + w, y + h))
        stem = Path(args.input_image).stem
        tag = f"{stem}_x{x}_y{y}_w{w}_h{h}"
        return crop, tag

    slide = openslide.OpenSlide(str(args.input_svs))
    W, H = slide.dimensions
    w = args.region_w
    h = args.region_h
    if w is None or h is None:
        raise ValueError("--region_w and --region_h are required for --input_svs")
    if args.x is not None and args.y is not None:
        x = args.x
        y = args.y
    else:
        x, y = pick_random_region(
            slide=slide,
            region_w=w,
            region_h=h,
            n_tries=args.max_region_tries,
            min_tissue=args.min_tissue,
            rng=rng,
        )
    if x < 0 or y < 0 or x + w > W or y + h > H:
        raise ValueError(f"SVS region ({x},{y},{w},{h}) is out of bounds for slide size ({W},{H})")
    crop = read_region_rgb(slide, x, y, w, h)
    slide.close()
    tag = f"{Path(args.input_svs).stem}_x{x}_y{y}_w{w}_h{h}"
    return crop, tag


def parse_args():
    p = argparse.ArgumentParser()

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_image", type=str, default=None, help="Path to input image (PNG/JPG/etc).")
    src.add_argument("--input_svs", type=str, default=None, help="Path to an SVS/WSI slide.")

    p.add_argument("--x", type=int, default=None, help="Top-left x for source crop/region.")
    p.add_argument("--y", type=int, default=None, help="Top-left y for source crop/region.")
    p.add_argument("--region_w", type=int, default=None, help="Region width in pixels.")
    p.add_argument("--region_h", type=int, default=None, help="Region height in pixels.")

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--save_real", action="store_true", help="Also save the source region image.")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--patch_px", type=int, default=0,
                   help="Diffusion window size in pixels. 0 = infer from PixCell model id.")
    p.add_argument("--stride_px", type=int, default=0,
                   help="Diffusion window stride in pixels. 0 = half of resolved patch size.")
    p.add_argument("--grid_step_px", type=int, default=256)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--patch_batch", type=int, default=256)
    p.add_argument("--strength", type=float, default=0.0, help="0.0 = pure generation, >0 = img2img-style.")

    p.add_argument("--steer_tile", action="append", default=[],
                   help="Tile steering spec: gx,gy,path_to_feature.npy (repeatable)")
    p.add_argument("--steer_manifest", type=str, default=None,
                   help="Optional JSON list of steering cells with keys gx, gy, path.")
    p.add_argument("--steer_blend", type=float, default=1.0,
                   help="Steer blending weight in [0,1]. 1.0 replaces tile feature.")

    p.add_argument("--pix_model_id", type=str, default="StonyBrook-CVLab/PixCell-256")
    p.add_argument("--pix_pipeline_id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    p.add_argument("--vae_model_id", type=str, default="stabilityai/stable-diffusion-3.5-large")
    p.add_argument("--vae_subfolder", type=str, default="vae")

    p.add_argument("--use_tiled_vae_encode", action="store_true",
                   help="Use tiled VAE encode for init latents. Default: off.")
    p.add_argument("--use_tiled_vae_decode", action="store_true",
                   help="Use tiled VAE decode for output. Default: off.")
    p.add_argument("--encode_tile_img", type=int, default=1024)
    p.add_argument("--encode_overlap_img", type=int, default=128)
    p.add_argument("--decode_tile_lat", type=int, default=96)
    p.add_argument("--decode_overlap_lat", type=int, default=16)

    p.add_argument("--max_region_tries", type=int, default=30)
    p.add_argument("--min_tissue", type=float, default=0.05)
    p.add_argument("--svs_root", type=str, default=None,
                   help="Optional helper path to search an SVS by stem name (not required).")
    p.add_argument("--svs_exts", type=str, default=",".join(sorted(SVS_EXTS_DEFAULT)))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if args.input_svs is None and args.svs_root:
        exts = [e.strip().lower() for e in args.svs_exts.split(",") if e.strip()]
        _ = index_svs_files(Path(args.svs_root), exts)

    source_img, source_tag = _load_source_region(args, rng=rng)
    if args.save_real:
        save_png(source_img, out_dir / f"{source_tag}_real.png")

    uni_model, uni_transform = load_uni2(device=device)
    sd3_vae = AutoencoderKL.from_pretrained(args.vae_model_id, subfolder=args.vae_subfolder)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=sd3_vae,
        custom_pipeline=args.pix_pipeline_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_px, stride_px, cond_grid_side = _resolve_pixcell_window_config(
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
        grid_step_px=args.grid_step_px,
        device=device,
        out_dtype=dtype,
    )
    steer_specs = list(args.steer_tile)
    if args.steer_manifest is not None:
        steer_specs.extend(_load_steer_manifest(Path(args.steer_manifest)))
    z_grid = apply_tile_steering(z_grid, steer_specs=steer_specs, steer_blend=args.steer_blend)

    init_latents = None
    if args.strength > 0.0:
        real_np = np.asarray(source_img, dtype=np.uint8)
        real_t = torch.from_numpy(real_np).to(device=device).permute(2, 0, 1).float() / 255.0
        real_t = real_t.unsqueeze(0).to(dtype=dtype)
        with torch.inference_mode():
            init_latents = vae_encode_auto(
                pipeline.vae,
                real_t,
                use_tiled=args.use_tiled_vae_encode,
                tile_img=args.encode_tile_img,
                overlap_img=args.encode_overlap_img,
            )
        del real_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    h, w = source_img.size[1], source_img.size[0]
    use_autocast = device.type == "cuda" and dtype == torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
    with torch.inference_mode(), ctx:
        img_t = sample_large_pixcell_multidiffusion(
            pipeline=pipeline,
            z_grid=z_grid,
            out_h=h,
            out_w=w,
            patch_px=patch_px,
            stride_px=stride_px,
            cond_grid_side=cond_grid_side,
            guidance_scale=args.guidance,
            num_steps=args.steps,
            patch_batch=args.patch_batch,
            strength=args.strength,
            init_latents=init_latents,
            use_tiled_vae_decode=args.use_tiled_vae_decode,
            decode_tile_lat=args.decode_tile_lat,
            decode_overlap_lat=args.decode_overlap_lat,
            generator=g,
        )

    img_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    gen = Image.fromarray(img_np)
    out_path = out_dir / f"{source_tag}_gen.png"
    save_png(gen, out_path)

    print("Done.")
    print(f"Source tag: {source_tag}")
    print(f"Output: {out_path}")
    if steer_specs:
        print(f"Applied {len(steer_specs)} steer tile(s), blend={args.steer_blend}")
    print(
        f"Tiled VAE encode={args.use_tiled_vae_encode}, "
        f"decode={args.use_tiled_vae_decode} (both default to OFF)"
    )


if __name__ == "__main__":
    main()
