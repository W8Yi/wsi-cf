from __future__ import annotations

import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def infer_pixcell_native_patch_px(pix_model_id: str) -> int:
    name = str(pix_model_id).lower()
    if "1024" in name:
        return 1024
    if "256" in name:
        return 256
    return 256


def infer_pixcell_cond_grid_side(pix_model_id: str) -> int:
    return 4 if infer_pixcell_native_patch_px(pix_model_id) >= 1024 else 1


def resolve_pixcell_window_config(*, pix_model_id: str, patch_px: int, stride_px: int) -> tuple[int, int, int]:
    resolved_patch_px = int(patch_px) if int(patch_px) > 0 else int(infer_pixcell_native_patch_px(pix_model_id))
    resolved_stride_px = int(stride_px) if int(stride_px) > 0 else max(1, resolved_patch_px // 2)
    cond_grid_side = int(infer_pixcell_cond_grid_side(pix_model_id))
    if min(resolved_patch_px, resolved_stride_px, cond_grid_side) <= 0:
        raise ValueError("Resolved PixCell window settings must all be > 0")
    return resolved_patch_px, resolved_stride_px, cond_grid_side


def _sample_z_grid_bilinear(
    z_grid: torch.Tensor,
    *,
    y_lat: float,
    x_lat: float,
    h_lat: int,
    w_lat: int,
) -> torch.Tensor:
    _, gh, gw, _ = z_grid.shape
    denom_y = max(h_lat - 1, 1)
    denom_x = max(w_lat - 1, 1)
    y = (float(y_lat) / denom_y) * (gh - 1) if gh > 1 else 0.0
    x = (float(x_lat) / denom_x) * (gw - 1) if gw > 1 else 0.0
    y = float(max(0.0, min(y, gh - 1)))
    x = float(max(0.0, min(x, gw - 1)))

    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    y1 = min(y0 + 1, gh - 1)
    x1 = min(x0 + 1, gw - 1)
    wy = y - y0
    wx = x - x0

    z00 = z_grid[:, y0, x0, :]
    z01 = z_grid[:, y0, x1, :]
    z10 = z_grid[:, y1, x0, :]
    z11 = z_grid[:, y1, x1, :]
    z0 = z00 * (1.0 - wx) + z01 * wx
    z1 = z10 * (1.0 - wx) + z11 * wx
    return z0 * (1.0 - wy) + z1 * wy


def lookup_window_cond(
    z_grid: torch.Tensor,
    *,
    top_lat: int,
    left_lat: int,
    ph: int,
    pw: int,
    h_lat: int,
    w_lat: int,
    cond_grid_side: int,
) -> torch.Tensor:
    side = max(1, int(cond_grid_side))
    cond_tokens = []
    for gy in range(side):
        for gx in range(side):
            center_y = float(top_lat) + ((gy + 0.5) / side) * float(ph)
            center_x = float(left_lat) + ((gx + 0.5) / side) * float(pw)
            cond_tokens.append(
                _sample_z_grid_bilinear(
                    z_grid,
                    y_lat=center_y,
                    x_lat=center_x,
                    h_lat=h_lat,
                    w_lat=w_lat,
                )
            )
    return torch.stack(cond_tokens, dim=1)


def align_uncond_embedding(uncond: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    if uncond.dim() == 2:
        uncond = uncond.unsqueeze(1)
    if uncond.shape[0] != cond.shape[0] or uncond.shape[-1] != cond.shape[-1]:
        raise RuntimeError(
            f"Unconditional embedding shape {tuple(uncond.shape)} incompatible with condition {tuple(cond.shape)}"
        )
    if uncond.shape[1] == cond.shape[1]:
        return uncond
    if uncond.shape[1] == 1:
        return uncond.expand(uncond.shape[0], cond.shape[1], uncond.shape[2])
    raise RuntimeError(
        f"Unconditional embedding token count {uncond.shape[1]} does not match condition token count {cond.shape[1]}"
    )


def validate_condition_schedule_ratios(*, start_ratio: float, end_ratio: float) -> tuple[float, float]:
    start = float(start_ratio)
    end = float(end_ratio)
    if not (0.0 <= start <= 1.0):
        raise ValueError("start_ratio must be in [0,1]")
    if not (0.0 <= end <= 1.0):
        raise ValueError("end_ratio must be in [0,1]")
    if start > end:
        raise ValueError("start_ratio must be <= end_ratio")
    return start, end


def compute_condition_blend_for_step(
    *,
    step_idx: int,
    num_steps: int,
    start_ratio: float,
    end_ratio: float,
    alpha_start: float,
    alpha_end: float,
    schedule: str,
) -> float:
    start, end = validate_condition_schedule_ratios(start_ratio=start_ratio, end_ratio=end_ratio)
    if num_steps <= 1:
        step_ratio = 1.0
    else:
        step_ratio = float(step_idx) / float(num_steps - 1)
    alpha0 = float(alpha_start)
    alpha1 = float(alpha_end)
    if not (0.0 <= alpha0 <= 1.0 and 0.0 <= alpha1 <= 1.0):
        raise ValueError("alpha_start and alpha_end must be in [0,1]")
    mode = str(schedule).strip().lower()
    if mode not in {"linear", "cosine"}:
        raise ValueError("schedule must be 'linear' or 'cosine'")
    if step_ratio <= start:
        return alpha0
    if step_ratio >= end:
        return alpha1
    width = max(end - start, 1e-8)
    frac = (step_ratio - start) / width
    if mode == "cosine":
        frac = 0.5 - 0.5 * math.cos(math.pi * frac)
    return (1.0 - frac) * alpha0 + frac * alpha1


def select_condition_grid_for_step(
    *,
    base_z_grid: torch.Tensor,
    scheduled_z_grid: torch.Tensor | None,
    step_idx: int,
    num_steps: int,
    start_ratio: float,
    end_ratio: float,
    alpha_start: float,
    alpha_end: float,
    schedule: str,
) -> torch.Tensor:
    if scheduled_z_grid is None:
        return base_z_grid
    alpha = compute_condition_blend_for_step(
        step_idx=step_idx,
        num_steps=num_steps,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        schedule=schedule,
    )
    alpha_t = base_z_grid.new_tensor(float(alpha))
    return base_z_grid * (1.0 - alpha_t) + scheduled_z_grid * alpha_t


def prepare_edit_region_mask(
    edit_region_mask: torch.Tensor | None,
    *,
    out_h: int,
    out_w: int,
    h_lat: int,
    w_lat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if edit_region_mask is None:
        return None
    mask = edit_region_mask.to(device=device, dtype=torch.float32)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError("edit_region_mask must have 2, 3, or 4 dimensions")
    if mask.shape[0] != 1:
        raise ValueError("edit_region_mask currently supports batch size 1")
    if mask.shape[-2:] == (int(out_h), int(out_w)):
        mask = F.interpolate(mask, size=(int(h_lat), int(w_lat)), mode="area")
    elif mask.shape[-2:] != (int(h_lat), int(w_lat)):
        raise ValueError(
            f"edit_region_mask spatial shape {tuple(mask.shape[-2:])} must match output {(out_h, out_w)} or latent {(h_lat, w_lat)}"
        )
    return mask.clamp(0.0, 1.0).to(dtype=dtype)


def load_uni2(device: torch.device):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

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
    *,
    uni_model,
    uni_transform,
    grid_step_px: int,
    device: torch.device,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    height, width = arr.shape[:2]
    gh = math.ceil(height / grid_step_px)
    gw = math.ceil(width / grid_step_px)
    half = grid_step_px // 2

    rows = []
    for gy in range(gh):
        cols = []
        for gx in range(gw):
            cx = int((gx + 0.5) * grid_step_px)
            cy = int((gy + 0.5) * grid_step_px)
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(width, x0 + grid_step_px)
            y1 = min(height, y0 + grid_step_px)
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
    return torch.stack(rows, dim=0).to(device=device, dtype=out_dtype)


@torch.no_grad()
def vae_decode_tiled(vae, latents: torch.Tensor, tile_lat: int, overlap_lat: int) -> torch.Tensor:
    device = latents.device
    vae_dtype = next(vae.parameters()).dtype
    _, _, height, width = latents.shape
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
    for top in range(0, height, step):
        for left in range(0, width, step):
            top0 = min(top, height - tile_lat)
            left0 = min(left, width - tile_lat)
            z = latents[:, :, top0 : top0 + tile_lat, left0 : left0 + tile_lat].to(dtype=vae_dtype)
            x = vae.decode(z, return_dict=True).sample
            if out is None:
                scale_h = x.shape[-2] // tile_lat
                scale_w = x.shape[-1] // tile_lat
                if scale_h != scale_w:
                    raise RuntimeError("Unexpected non-uniform VAE decode scale.")
                scale = scale_h
                out = torch.zeros((1, x.shape[1], height * scale, width * scale), device=device, dtype=x.dtype)
                wgt = torch.zeros((1, 1, height * scale, width * scale), device=device, dtype=x.dtype)
            win_img = F.interpolate(win.to(dtype=x.dtype), size=x.shape[-2:], mode="bilinear", align_corners=False)
            y_img = top0 * scale
            x_img = left0 * scale
            out[:, :, y_img : y_img + x.shape[-2], x_img : x_img + x.shape[-1]] += x * win_img
            wgt[:, :, y_img : y_img + x.shape[-2], x_img : x_img + x.shape[-1]] += win_img
    return out / wgt.clamp(min=1e-8)


@torch.no_grad()
def vae_encode_tiled(vae, img01: torch.Tensor, tile_img: int, overlap_img: int) -> torch.Tensor:
    device = img01.device
    vae_dtype = next(vae.parameters()).dtype
    _, _, height, width = img01.shape
    if tile_img <= 0:
        raise ValueError("tile_img must be > 0")
    if overlap_img < 0 or overlap_img >= tile_img:
        raise ValueError("overlap_img must satisfy 0 <= overlap_img < tile_img")

    step = max(1, tile_img - overlap_img)
    yy = torch.linspace(-1.0, 1.0, tile_img, device=device, dtype=vae_dtype)[:, None]
    xx = torch.linspace(-1.0, 1.0, tile_img, device=device, dtype=vae_dtype)[None, :]
    win = torch.exp(-2.0 * (yy**2 + xx**2))
    win = (win / win.max().clamp(min=1e-8))[None, None]

    out = None
    wgt = None
    scale = None
    for top in range(0, height, step):
        for left in range(0, width, step):
            top0 = min(top, height - tile_img)
            left0 = min(left, width - tile_img)
            x = img01[:, :, top0 : top0 + tile_img, left0 : left0 + tile_img]
            x = (x * 2.0 - 1.0).to(dtype=vae_dtype)
            z = vae.encode(x, return_dict=True).latent_dist.sample()
            if out is None:
                scale_h = tile_img // z.shape[-2]
                scale_w = tile_img // z.shape[-1]
                if scale_h != scale_w:
                    raise RuntimeError("Unexpected non-uniform VAE encode scale.")
                scale = scale_h
                h_lat = math.ceil(height / scale)
                w_lat = math.ceil(width / scale)
                out = torch.zeros((1, z.shape[1], h_lat, w_lat), device=device, dtype=z.dtype)
                wgt = torch.zeros((1, 1, h_lat, w_lat), device=device, dtype=z.dtype)
            win_lat = F.interpolate(win, size=z.shape[-2:], mode="bilinear", align_corners=False).to(dtype=z.dtype)
            y_lat = top0 // scale
            x_lat = left0 // scale
            out[:, :, y_lat : y_lat + z.shape[-2], x_lat : x_lat + z.shape[-1]] += z * win_lat
            wgt[:, :, y_lat : y_lat + z.shape[-2], x_lat : x_lat + z.shape[-1]] += win_lat
    return out / wgt.clamp(min=1e-8)


def vae_encode_auto(vae, img01: torch.Tensor, *, use_tiled: bool, tile_img: int, overlap_img: int) -> torch.Tensor:
    if use_tiled:
        return vae_encode_tiled(vae, img01, tile_img=tile_img, overlap_img=overlap_img)
    vae_dtype = next(vae.parameters()).dtype
    return vae.encode((img01 * 2.0 - 1.0).to(dtype=vae_dtype), return_dict=True).latent_dist.sample()


def vae_decode_auto(vae, latents: torch.Tensor, *, use_tiled: bool, tile_lat: int, overlap_lat: int) -> torch.Tensor:
    if use_tiled:
        return vae_decode_tiled(vae, latents, tile_lat=tile_lat, overlap_lat=overlap_lat)
    return vae.decode(latents, return_dict=True).sample


@torch.no_grad()
def sample_large_pixcell_multidiffusion(
    pipeline,
    *,
    z_grid: torch.Tensor,
    scheduled_z_grid: torch.Tensor | None = None,
    condition_start_ratio: float = 0.0,
    condition_end_ratio: float = 1.0,
    condition_alpha_start: float = 1.0,
    condition_alpha_end: float = 1.0,
    condition_alpha_schedule: str = "linear",
    out_h: int,
    out_w: int,
    patch_px: int,
    stride_px: int,
    cond_grid_side: int,
    guidance_scale: float,
    num_steps: int,
    patch_batch: int,
    strength: float,
    init_latents: torch.Tensor | None,
    preserve_source_latents: torch.Tensor | None = None,
    edit_region_mask: torch.Tensor | None = None,
    preserve_outside_strength: float = 1.0,
    use_tiled_vae_decode: bool,
    decode_tile_lat: int,
    decode_overlap_lat: int,
    generator: torch.Generator | None,
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
        raise ValueError("This function currently supports batch size 1.")
    if scheduled_z_grid is not None:
        if scheduled_z_grid.dim() == 3:
            scheduled_z_grid = scheduled_z_grid.unsqueeze(0)
        if scheduled_z_grid.shape != z_grid.shape:
            raise ValueError(
                f"scheduled_z_grid shape {tuple(scheduled_z_grid.shape)} must match z_grid {tuple(z_grid.shape)}"
            )
        compute_condition_blend_for_step(
            step_idx=0,
            num_steps=2,
            start_ratio=float(condition_start_ratio),
            end_ratio=float(condition_end_ratio),
            alpha_start=float(condition_alpha_start),
            alpha_end=float(condition_alpha_end),
            schedule=str(condition_alpha_schedule),
        )
    preserve_ref_lat = None
    preserve_noise = None
    latent_edit_mask = None
    if preserve_source_latents is not None:
        preserve_ref_lat = preserve_source_latents.to(device=device, dtype=dtype)[:, :, :h_lat, :w_lat]
        if hasattr(pipeline.vae.config, "scaling_factor"):
            preserve_ref_lat = preserve_ref_lat * pipeline.vae.config.scaling_factor
        latent_edit_mask = prepare_edit_region_mask(
            edit_region_mask,
            out_h=int(out_h),
            out_w=int(out_w),
            h_lat=int(h_lat),
            w_lat=int(w_lat),
            device=device,
            dtype=dtype,
        )
        if latent_edit_mask is None:
            raise ValueError("edit_region_mask is required when preserve_source_latents is provided")
        preserve_alpha = float(preserve_outside_strength)
        if not (0.0 <= preserve_alpha <= 1.0):
            raise ValueError("preserve_outside_strength must be in [0,1]")
        preserve_noise = torch.randn(
            preserve_ref_lat.shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    channels = pipeline.vae.config.latent_channels

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
        latents = torch.randn((1, channels, h_lat, w_lat), device=device, dtype=dtype, generator=generator)
        start_idx = 0
    else:
        init_lat = init_latents.to(device=device, dtype=dtype)[:, :, :h_lat, :w_lat]
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
    total_steps = len(timesteps)
    for t_idx, t in enumerate(tqdm(timesteps, desc="Diffusion steps", leave=False)):
        t_model = t if torch.is_tensor(t) else torch.tensor(t, device=device)
        t_model = t_model.to(device=device).long()
        if t_model.dim() == 0:
            t_model = t_model[None]
        latents_in = pipeline.scheduler.scale_model_input(latents, t) if has_scale else latents
        active_z_grid = select_condition_grid_for_step(
            base_z_grid=z_grid,
            scheduled_z_grid=scheduled_z_grid,
            step_idx=int(t_idx),
            num_steps=int(total_steps),
            start_ratio=float(condition_start_ratio),
            end_ratio=float(condition_end_ratio),
            alpha_start=float(condition_alpha_start),
            alpha_end=float(condition_alpha_end),
            schedule=str(condition_alpha_schedule),
        )
        eps_accum = torch.zeros_like(latents)
        weight = torch.zeros((1, 1, h_lat, w_lat), device=device, dtype=dtype)
        for i0 in tqdm(range(0, len(coords), patch_batch), desc=f"Windows @ step {t_idx + 1}/{len(timesteps)}", leave=False):
            chunk = coords[i0 : i0 + patch_batch]
            n = len(chunk)
            patch_in = torch.stack(
                [latents_in[:, :, top : top + ph, left : left + pw].squeeze(0) for (top, left) in chunk],
                dim=0,
            )
            cond = torch.cat(
                [
                    lookup_window_cond(
                        active_z_grid,
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
            uncond = align_uncond_embedding(
                pipeline.get_unconditional_embedding(n).to(device=device, dtype=dtype),
                cond,
            )
            hs = torch.cat([patch_in, patch_in], dim=0)
            es = torch.cat([uncond, cond], dim=0)
            tt = t_model.expand(hs.shape[0])
            out = pipeline.transformer(hidden_states=hs, encoder_hidden_states=es, timestep=tt, return_dict=True)
            eps2 = out.sample if hasattr(out, "sample") else out[0]
            if eps2.shape[1] == 2 * channels:
                eps2 = eps2[:, :channels]
            elif eps2.shape[1] != channels:
                raise RuntimeError(f"Model output channels={eps2.shape[1]}, expected {channels} or {2 * channels}")
            eps_u, eps_c = eps2[:n], eps2[n:]
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            for j, (top, left) in enumerate(chunk):
                eps_accum[:, :, top : top + ph, left : left + pw] += eps[j : j + 1] * gauss
                weight[:, :, top : top + ph, left : left + pw] += gauss
        eps_full = eps_accum / weight.clamp(min=1e-8)
        step = pipeline.scheduler.step(eps_full, t, latents, return_dict=True)
        latents = step.prev_sample if hasattr(step, "prev_sample") else step[0]
        if preserve_ref_lat is not None and latent_edit_mask is not None:
            preserve_weight = (1.0 - latent_edit_mask) * float(preserve_outside_strength)
            if t_idx + 1 < total_steps:
                next_t = timesteps[t_idx + 1]
                if not torch.is_tensor(next_t):
                    next_t = torch.tensor(next_t, device=device)
                next_t_b = next_t.to(device=device).expand(preserve_ref_lat.shape[0])
                ref_latents = pipeline.scheduler.add_noise(preserve_ref_lat, preserve_noise, next_t_b)
            else:
                ref_latents = preserve_ref_lat
            latents = latents * (1.0 - preserve_weight) + ref_latents * preserve_weight

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
