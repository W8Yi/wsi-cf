import math
import numpy as np
import torch
import torch.nn.functional as F

def infer_pixcell_native_patch_px(pix_model_id: str) -> int:
    """
    Best-effort native image size inference from the PixCell model id.

    Current official releases are PixCell-256 and PixCell-1024. Unknown ids
    fall back to 256 to preserve existing behavior.
    """
    name = str(pix_model_id).lower()
    if "1024" in name:
        return 1024
    if "256" in name:
        return 256
    return 256


def infer_pixcell_cond_grid_side(pix_model_id: str) -> int:
    """
    Infer the per-window UNI conditioning grid side length.

    PixCell-256 conditions on a single UNI embedding.
    PixCell-1024 conditions on a 4x4 grid of UNI embeddings.
    """
    native_px = infer_pixcell_native_patch_px(pix_model_id)
    if native_px >= 1024:
        return 4
    return 1


def _sample_z_grid_bilinear(
    z_grid: torch.Tensor,  # [1,Gh,Gw,D]
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


def _lookup_window_cond(
    z_grid: torch.Tensor,  # [1,Gh,Gw,D]
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
    return torch.stack(cond_tokens, dim=1)  # [1, side*side, D]


def _align_uncond_embedding(uncond: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
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


def build_uni_grid_from_pil(
    img,                       # PIL.Image (RGB)
    uni_model,                 # UNI model
    uni_transform,             # transform -> (3,224,224) tensor
    tile_px: int = 256,        # patch size to crop from img
    grid_step_px: int | None = None,  # defaults to tile_px
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    return_numpy: bool = False,
):
    """
    Build a [Gh,Gw,D] UNI embedding grid by sampling patches on a regular grid.

    - If grid_step_px == tile_px and img is divisible by tile_px, this matches
      a non-overlapping tiling.
    - Uses center sampling consistent with the slide-based version (cx, cy at cell centers).
    - Pads/crops coordinates safely at image borders (no assertion).
    """
    if grid_step_px is None:
        grid_step_px = tile_px

    W, H = img.size
    Gw = math.ceil(W / grid_step_px)
    Gh = math.ceil(H / grid_step_px)

    print("image size:", (W, H))
    print("grid Gw, Gh:", (Gw, Gh), "| total cells:", Gw * Gh)

    uni_model.eval()

    # Pre-allocate on CPU float32, then cast at the end
    z_cpu = []
    coords = []

    half = grid_step_px // 2
    with torch.no_grad():
        for gy in range(Gh):
            row = []
            for gx in range(Gw):
                cx = int((gx + 0.5) * grid_step_px)
                cy = int((gy + 0.5) * grid_step_px)

                # Convert center -> top-left, clamp to image bounds
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(W, x1 + grid_step_px)
                y2 = min(H, y1 + grid_step_px)

                # If we got clipped on the right/bottom, shift left/up to keep size if possible
                x1 = max(0, x2 - grid_step_px)
                y1 = max(0, y2 - grid_step_px)

                patch = img.crop((x1, y1, x1 + grid_step_px, y1 + grid_step_px))  # may be smaller at borders
                # If smaller, pad to grid_step_px with white background
                if patch.size != (grid_step_px, grid_step_px):
                    bg = patch.new("RGB", (grid_step_px, grid_step_px), (255, 255, 255))
                    bg.paste(patch, (0, 0))
                    patch = bg

                inp = uni_transform(patch).unsqueeze(0).to(device=device)
                emb = uni_model(inp)  # [1, D]
                emb = emb.squeeze(0).detach().float().cpu()  # float32 on CPU
                row.append(emb)

                # coords as TOP-LEFT in the original image coordinate system
                coords.append((x1, y1))

            z_cpu.append(torch.stack(row, dim=0))  # [Gw, D]

    z = torch.stack(z_cpu, dim=0)  # [Gh, Gw, D]
    z = z.to(device=device, dtype=dtype)

    coords = np.asarray(coords, dtype=np.int32).reshape(Gh, Gw, 2)

    print("z grid:", tuple(z.shape), z.dtype)
    print("coords grid:", coords.shape, coords.dtype)
    if return_numpy:
        z_np = z.detach().float().cpu().numpy()
        print("z stats min/mean/max:", float(z_np.min()), float(z_np.mean()), float(z_np.max()))
        print("first 5 coords:", coords.reshape(-1, 2)[:5].tolist())
        return z_np, coords
    else:
        # If you still want stats without converting all of z:
        z_flat = z.detach().float().view(-1)
        print("z stats min/mean/max:", float(z_flat.min()), float(z_flat.mean()), float(z_flat.max()))
        print("first 5 coords:", coords.reshape(-1, 2)[:5].tolist())
        return z, coords

@torch.no_grad()
def sample_multidiffusion_from_zgrid(
    pipeline,                   # loaded PixCell pipeline
    z_grid: torch.Tensor,        # [Gh,Gw,D] or [1,Gh,Gw,D]
    out_h: int,
    out_w: int,
    patch_px: int = 256,         # in IMAGE pixels (PixCell-256)
    stride_px: int = 128,        # in IMAGE pixels
    cond_grid_side: int = 1,     # 1 for PixCell-256, 4 for PixCell-1024
    steps: int = 30,
    guidance: float = 2.0,
    patch_batch: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Returns img tensor [1,3,out_h,out_w] in [0,1].
    Minimal MultiDiffusion over a latent canvas, conditioned by z_grid.
    No img2img, no tiled VAE.
    """
    device = pipeline.device
    dtype = next(pipeline.transformer.parameters()).dtype

    # VAE spatial downsample factor (usually 8)
    vae_scale = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)

    H_lat = math.ceil(out_h / vae_scale)
    W_lat = math.ceil(out_w / vae_scale)
    ph = patch_px // vae_scale
    pw = patch_px // vae_scale
    sh = stride_px // vae_scale
    sw = stride_px // vae_scale
    if ph <= 0 or pw <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("patch_px/stride_px must be >= vae_scale")

    C = pipeline.vae.config.latent_channels

    if z_grid.dim() == 3:
        z_grid = z_grid.unsqueeze(0)
    if z_grid.shape[0] != 1:
        raise ValueError("This minimal function assumes B=1 conditioning grid.")
    _, Gh, Gw, D = z_grid.shape

    # Feather weight (Gaussian) for blending window predictions
    yy = torch.arange(ph, device=device, dtype=dtype)[:, None]
    xx = torch.arange(pw, device=device, dtype=dtype)[None, :]
    cy = (ph - 1) / 2.0
    cx = (pw - 1) / 2.0
    yy = yy - cy
    xx = xx - cx
    sigma_y = max(ph / 4.0, 1e-6)
    sigma_x = max(pw / 4.0, 1e-6)
    gauss = torch.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
    gauss = gauss / gauss.max().clamp(min=1e-8)
    gauss = gauss[None, None, :, :]  # [1,1,ph,pw]

    # Scheduler timesteps
    pipeline.scheduler.set_timesteps(steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    has_scale = hasattr(pipeline.scheduler, "scale_model_input")

    # Window coordinates over latent canvas
    coords = [
        (top, left)
        for top in range(0, H_lat - ph + 1, sh)
        for left in range(0, W_lat - pw + 1, sw)
    ]
    if not coords:
        raise RuntimeError("No windows generated; check out_h/out_w vs patch/stride.")
    patch_batch = max(1, int(patch_batch))

    # Initial latents (pure noise)
    if generator is None:
        latents = torch.randn((1, C, H_lat, W_lat), device=device, dtype=dtype)
    else:
        latents = torch.randn((1, C, H_lat, W_lat), device=device, dtype=dtype, generator=generator)

    for t in timesteps:
        latents_in = pipeline.scheduler.scale_model_input(latents, t) if has_scale else latents

        eps_accum = torch.zeros_like(latents)
        weight = torch.zeros((1, 1, H_lat, W_lat), device=device, dtype=dtype)

        for i0 in range(0, len(coords), patch_batch):
            chunk = coords[i0:i0 + patch_batch]
            n = len(chunk)

            patch_in = torch.stack(
                [latents_in[:, :, top:top+ph, left:left+pw].squeeze(0) for (top, left) in chunk],
                dim=0,
            )  # [N,C,ph,pw]

            cond = torch.cat(
                [
                    _lookup_window_cond(
                        z_grid,
                        top_lat=top,
                        left_lat=left,
                        ph=ph,
                        pw=pw,
                        h_lat=H_lat,
                        w_lat=W_lat,
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

            # One forward for CFG
            hs = torch.cat([patch_in, patch_in], dim=0)  # [2N,C,ph,pw]
            es = torch.cat([uncond, cond], dim=0)
            tt = (t if torch.is_tensor(t) else torch.tensor(t, device=device)).long()
            if tt.dim() == 0:
                tt = tt[None]
            tt = tt.expand(hs.shape[0])

            out = pipeline.transformer(
                hidden_states=hs,
                encoder_hidden_states=es,
                timestep=tt,
                return_dict=True,
            )
            eps2 = out.sample if hasattr(out, "sample") else out[0]

            if eps2.shape[1] == 2 * C:
                eps2 = eps2[:, :C]
            elif eps2.shape[1] != C:
                raise RuntimeError(f"model output channels={eps2.shape[1]} != {C} (or {2*C})")

            eps_u, eps_c = eps2[:n], eps2[n:]
            eps = eps_u + guidance * (eps_c - eps_u)  # [N,C,ph,pw]

            for j, (top, left) in enumerate(chunk):
                eps_accum[:, :, top:top+ph, left:left+pw] += eps[j:j+1] * gauss
                weight[:, :, top:top+ph, left:left+pw] += gauss

        eps_full = eps_accum / weight.clamp(min=1e-8)
        step_out = pipeline.scheduler.step(eps_full, t, latents, return_dict=True)
        latents = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

    # Decode (no tiling)
    if hasattr(pipeline.vae.config, "scaling_factor"):
        latents = latents / pipeline.vae.config.scaling_factor
    # Keep VAE input dtype/device aligned with VAE parameters.
    vae_param = next(pipeline.vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    img = pipeline.vae.decode(latents, return_dict=True).sample  # [1,3,H,W] in [-1,1]
    img = (img / 2 + 0.5).clamp(0, 1)
    return img


@torch.no_grad()
def sample_multidiffusion_from_zgrid_with_midref(
    pipeline,                    # loaded PixCell pipeline
    z_grid: torch.Tensor,        # [Gh,Gw,D] or [1,Gh,Gw,D]
    original_image,              # [1,3,H,W] tensor in [0,1] or uint8 numpy/PIL-like
    out_h: int,
    out_w: int,
    patch_px: int = 256,         # in IMAGE pixels (PixCell-256)
    stride_px: int = 128,        # in IMAGE pixels
    cond_grid_side: int = 1,     # 1 for PixCell-256, 4 for PixCell-1024
    steps: int = 30,
    guidance: float = 2.0,
    patch_batch: int = 256,
    reference_start_ratio: float = 0.5,  # 0=start, 0.5=halfway, 1=end
    reference_start_step: int | None = None,  # overrides ratio if provided
    reference_mix: float = 1.0,  # 1.0=full anchor to image trajectory after switch
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Returns img tensor [1,3,out_h,out_w] in [0,1].

    Same as `sample_multidiffusion_from_zgrid`, plus an optional midpoint
    switch where denoising is anchored to the original-image noise trajectory.
    This can improve global coherence.
    """
    device = pipeline.device
    dtype = next(pipeline.transformer.parameters()).dtype

    if not (0.0 <= reference_start_ratio <= 1.0):
        raise ValueError("reference_start_ratio must be in [0, 1].")
    if not (0.0 <= reference_mix <= 1.0):
        raise ValueError("reference_mix must be in [0, 1].")

    # VAE spatial downsample factor (usually 8)
    vae_scale = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)

    H_lat = math.ceil(out_h / vae_scale)
    W_lat = math.ceil(out_w / vae_scale)
    ph = patch_px // vae_scale
    pw = patch_px // vae_scale
    sh = stride_px // vae_scale
    sw = stride_px // vae_scale
    if ph <= 0 or pw <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("patch_px/stride_px must be >= vae_scale")

    C = pipeline.vae.config.latent_channels

    if z_grid.dim() == 3:
        z_grid = z_grid.unsqueeze(0)
    if z_grid.shape[0] != 1:
        raise ValueError("This function assumes B=1 conditioning grid.")
    _, Gh, Gw, D = z_grid.shape

    # Feather weight (Gaussian) for blending window predictions
    yy = torch.arange(ph, device=device, dtype=dtype)[:, None]
    xx = torch.arange(pw, device=device, dtype=dtype)[None, :]
    cy = (ph - 1) / 2.0
    cx = (pw - 1) / 2.0
    yy = yy - cy
    xx = xx - cx
    sigma_y = max(ph / 4.0, 1e-6)
    sigma_x = max(pw / 4.0, 1e-6)
    gauss = torch.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
    gauss = gauss / gauss.max().clamp(min=1e-8)
    gauss = gauss[None, None, :, :]  # [1,1,ph,pw]

    # Scheduler timesteps
    pipeline.scheduler.set_timesteps(steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    has_scale = hasattr(pipeline.scheduler, "scale_model_input")

    # Window coordinates over latent canvas
    coords = [
        (top, left)
        for top in range(0, H_lat - ph + 1, sh)
        for left in range(0, W_lat - pw + 1, sw)
    ]
    if not coords:
        raise RuntimeError("No windows generated; check out_h/out_w vs patch/stride.")
    patch_batch = max(1, int(patch_batch))

    # Initial latents (pure noise)
    if generator is None:
        init_noise = torch.randn((1, C, H_lat, W_lat), device=device, dtype=dtype)
    else:
        init_noise = torch.randn((1, C, H_lat, W_lat), device=device, dtype=dtype, generator=generator)
    latents = init_noise.clone()

    # Prepare original image latent for midpoint anchoring.
    if torch.is_tensor(original_image):
        ref_img = original_image
    else:
        ref_np = np.asarray(original_image)
        if ref_np.ndim == 3:
            ref_np = ref_np[None]
        if ref_np.shape[-1] == 3 and ref_np.shape[1] != 3:
            ref_np = np.transpose(ref_np, (0, 3, 1, 2))
        ref_img = torch.from_numpy(ref_np)

    ref_img = ref_img.to(device=device)
    if ref_img.dtype == torch.uint8:
        ref_img = ref_img.float() / 255.0
    else:
        ref_img = ref_img.float()
    if ref_img.ndim != 4 or ref_img.shape[0] != 1 or ref_img.shape[1] != 3:
        raise ValueError("original_image must be [1,3,H,W] or convertible to that shape.")
    if ref_img.min() < 0.0 or ref_img.max() > 1.0:
        raise ValueError("original_image values must be in [0,1] (or uint8).")
    if ref_img.shape[-2:] != (out_h, out_w):
        ref_img = F.interpolate(ref_img, size=(out_h, out_w), mode="bilinear", align_corners=False)

    ref_img_vae = (ref_img * 2.0 - 1.0).to(dtype=dtype)
    ref_dist = pipeline.vae.encode(ref_img_vae).latent_dist
    ref_latents_clean = ref_dist.sample()
    if hasattr(pipeline.vae.config, "scaling_factor"):
        ref_latents_clean = ref_latents_clean * pipeline.vae.config.scaling_factor
    ref_latents_clean = ref_latents_clean.to(device=device, dtype=dtype)

    n_steps = len(timesteps)
    if reference_start_step is None:
        switch_idx = int(round(reference_start_ratio * max(n_steps - 1, 0)))
    else:
        switch_idx = int(reference_start_step)
    switch_idx = max(0, min(switch_idx, n_steps - 1))

    for step_idx, t in enumerate(timesteps):
        if step_idx >= switch_idx and reference_mix > 0.0:
            # DPMSolver add_noise expects timesteps as a 1D tensor.
            t_for_noise = timesteps[step_idx:step_idx + 1]
            ref_xt = pipeline.scheduler.add_noise(ref_latents_clean, init_noise, t_for_noise)
            latents_step = latents * (1.0 - reference_mix) + ref_xt * reference_mix
        else:
            latents_step = latents

        latents_in = pipeline.scheduler.scale_model_input(latents_step, t) if has_scale else latents_step

        eps_accum = torch.zeros_like(latents_step)
        weight = torch.zeros((1, 1, H_lat, W_lat), device=device, dtype=dtype)

        for i0 in range(0, len(coords), patch_batch):
            chunk = coords[i0:i0 + patch_batch]
            n = len(chunk)

            patch_in = torch.stack(
                [latents_in[:, :, top:top+ph, left:left+pw].squeeze(0) for (top, left) in chunk],
                dim=0,
            )  # [N,C,ph,pw]

            cond = torch.cat(
                [
                    _lookup_window_cond(
                        z_grid,
                        top_lat=top,
                        left_lat=left,
                        ph=ph,
                        pw=pw,
                        h_lat=H_lat,
                        w_lat=W_lat,
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

            # One forward for CFG
            hs = torch.cat([patch_in, patch_in], dim=0)  # [2N,C,ph,pw]
            es = torch.cat([uncond, cond], dim=0)
            tt = (t if torch.is_tensor(t) else torch.tensor(t, device=device)).long()
            if tt.dim() == 0:
                tt = tt[None]
            tt = tt.expand(hs.shape[0])

            out = pipeline.transformer(
                hidden_states=hs,
                encoder_hidden_states=es,
                timestep=tt,
                return_dict=True,
            )
            eps2 = out.sample if hasattr(out, "sample") else out[0]

            if eps2.shape[1] == 2 * C:
                eps2 = eps2[:, :C]
            elif eps2.shape[1] != C:
                raise RuntimeError(f"model output channels={eps2.shape[1]} != {C} (or {2*C})")

            eps_u, eps_c = eps2[:n], eps2[n:]
            eps = eps_u + guidance * (eps_c - eps_u)  # [N,C,ph,pw]

            for j, (top, left) in enumerate(chunk):
                eps_accum[:, :, top:top+ph, left:left+pw] += eps[j:j+1] * gauss
                weight[:, :, top:top+ph, left:left+pw] += gauss

        eps_full = eps_accum / weight.clamp(min=1e-8)
        step_out = pipeline.scheduler.step(eps_full, t, latents_step, return_dict=True)
        latents = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

    # Decode (no tiling)
    if hasattr(pipeline.vae.config, "scaling_factor"):
        latents = latents / pipeline.vae.config.scaling_factor
    # Keep VAE input dtype/device aligned with VAE parameters.
    vae_param = next(pipeline.vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    img = pipeline.vae.decode(latents, return_dict=True).sample  # [1,3,H,W] in [-1,1]
    img = (img / 2 + 0.5).clamp(0, 1)
    return img


@torch.no_grad()
def sample_multidiffusion_from_zgrid_with_stepmask(
    pipeline,                    # loaded PixCell pipeline
    z_grid: torch.Tensor,        # [Gh,Gw,D] or [1,Gh,Gw,D]
    original_image,              # [1,3,H,W] tensor in [0,1] or uint8 numpy/PIL-like
    editable_mask_grid,          # [Gh,Gw], 1=editable, 0=preserve
    out_h: int,
    out_w: int,
    patch_px: int = 256,         # in IMAGE pixels
    stride_px: int = 128,        # in IMAGE pixels
    cond_grid_side: int = 1,     # 1 for PixCell-256, 4 for PixCell-1024
    steps: int = 30,
    guidance: float = 2.0,
    patch_batch: int = 256,
    mask_blur_cells: float = 0.0,
    preserve_strength: float = 1.0,  # 1.0 = strict preserve on masked-out region
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Returns img tensor [1,3,out_h,out_w] in [0,1].

    Step-wise masked editing:
      x_t <- M * x_t_generated + (1 - M) * x_t_reference
    at every denoising step, where M is the editable mask.
    """
    device = pipeline.device
    dtype = next(pipeline.transformer.parameters()).dtype

    if not (0.0 <= preserve_strength <= 1.0):
        raise ValueError("preserve_strength must be in [0, 1].")
    if mask_blur_cells < 0.0:
        raise ValueError("mask_blur_cells must be >= 0.")

    # VAE spatial downsample factor (usually 8)
    vae_scale = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)

    H_lat = math.ceil(out_h / vae_scale)
    W_lat = math.ceil(out_w / vae_scale)
    ph = patch_px // vae_scale
    pw = patch_px // vae_scale
    sh = stride_px // vae_scale
    sw = stride_px // vae_scale
    if ph <= 0 or pw <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("patch_px/stride_px must be >= vae_scale")

    C = pipeline.vae.config.latent_channels

    if z_grid.dim() == 3:
        z_grid = z_grid.unsqueeze(0)
    if z_grid.shape[0] != 1:
        raise ValueError("This function assumes B=1 conditioning grid.")
    _, Gh, Gw, D = z_grid.shape

    # Prepare editable mask in latent canvas resolution.
    edit_np = np.asarray(editable_mask_grid, dtype=np.float32)
    if edit_np.shape != (Gh, Gw):
        raise ValueError(f"editable_mask_grid must be shape {(Gh, Gw)}, got {edit_np.shape}")
    edit_np = np.clip(edit_np, 0.0, 1.0)
    edit_t = torch.from_numpy(edit_np).to(device=device, dtype=torch.float32).view(1, 1, Gh, Gw)
    if mask_blur_cells > 0.0:
        radius = max(1, int(round(3.0 * mask_blur_cells)))
        k = 2 * radius + 1
        yyb, xxb = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
        ker = np.exp(-0.5 * (xxb * xxb + yyb * yyb) / float(mask_blur_cells * mask_blur_cells))
        ker = ker / np.maximum(ker.sum(), 1e-8)
        ker_t = torch.from_numpy(ker).to(device=device, dtype=torch.float32).view(1, 1, k, k)
        edit_t = F.conv2d(edit_t, ker_t, padding=radius).clamp(0.0, 1.0)
    edit_lat = F.interpolate(edit_t, size=(H_lat, W_lat), mode="bilinear", align_corners=False).to(device=device, dtype=dtype)
    preserve_lat = (1.0 - edit_lat) * float(preserve_strength)

    # Feather weight (Gaussian) for blending window predictions
    yy = torch.arange(ph, device=device, dtype=dtype)[:, None]
    xx = torch.arange(pw, device=device, dtype=dtype)[None, :]
    cy = (ph - 1) / 2.0
    cx = (pw - 1) / 2.0
    yy = yy - cy
    xx = xx - cx
    sigma_y = max(ph / 4.0, 1e-6)
    sigma_x = max(pw / 4.0, 1e-6)
    gauss = torch.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
    gauss = gauss / gauss.max().clamp(min=1e-8)
    gauss = gauss[None, None, :, :]  # [1,1,ph,pw]

    # Scheduler timesteps
    pipeline.scheduler.set_timesteps(steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    has_scale = hasattr(pipeline.scheduler, "scale_model_input")

    # Window coordinates over latent canvas
    coords = [
        (top, left)
        for top in range(0, H_lat - ph + 1, sh)
        for left in range(0, W_lat - pw + 1, sw)
    ]
    if not coords:
        raise RuntimeError("No windows generated; check out_h/out_w vs patch/stride.")
    patch_batch = max(1, int(patch_batch))

    # Initial latents (pure noise)
    if generator is None:
        init_noise = torch.randn((1, C, H_lat, W_lat), device=device, dtype=dtype)
    else:
        init_noise = torch.randn((1, C, H_lat, W_lat), device=device, dtype=dtype, generator=generator)
    latents = init_noise.clone()

    # Prepare original-image latent trajectory
    if torch.is_tensor(original_image):
        ref_img = original_image
    else:
        ref_np = np.asarray(original_image)
        if ref_np.ndim == 3:
            ref_np = ref_np[None]
        if ref_np.shape[-1] == 3 and ref_np.shape[1] != 3:
            ref_np = np.transpose(ref_np, (0, 3, 1, 2))
        ref_img = torch.from_numpy(ref_np)

    ref_img = ref_img.to(device=device)
    if ref_img.dtype == torch.uint8:
        ref_img = ref_img.float() / 255.0
    else:
        ref_img = ref_img.float()
    if ref_img.ndim != 4 or ref_img.shape[0] != 1 or ref_img.shape[1] != 3:
        raise ValueError("original_image must be [1,3,H,W] or convertible to that shape.")
    if ref_img.min() < 0.0 or ref_img.max() > 1.0:
        raise ValueError("original_image values must be in [0,1] (or uint8).")
    if ref_img.shape[-2:] != (out_h, out_w):
        ref_img = F.interpolate(ref_img, size=(out_h, out_w), mode="bilinear", align_corners=False)

    ref_img_vae = (ref_img * 2.0 - 1.0).to(dtype=dtype)
    ref_dist = pipeline.vae.encode(ref_img_vae).latent_dist
    ref_latents_clean = ref_dist.sample()
    if hasattr(pipeline.vae.config, "scaling_factor"):
        ref_latents_clean = ref_latents_clean * pipeline.vae.config.scaling_factor
    ref_latents_clean = ref_latents_clean.to(device=device, dtype=dtype)

    for step_idx, t in enumerate(timesteps):
        # Build reference latent at the same timestep and enforce mask.
        t_for_noise = timesteps[step_idx : step_idx + 1]
        ref_xt = pipeline.scheduler.add_noise(ref_latents_clean, init_noise, t_for_noise)
        latents_step = latents * edit_lat + ref_xt * preserve_lat

        latents_in = pipeline.scheduler.scale_model_input(latents_step, t) if has_scale else latents_step

        eps_accum = torch.zeros_like(latents_step)
        weight = torch.zeros((1, 1, H_lat, W_lat), device=device, dtype=dtype)

        for i0 in range(0, len(coords), patch_batch):
            chunk = coords[i0:i0 + patch_batch]
            n = len(chunk)

            patch_in = torch.stack(
                [latents_in[:, :, top:top+ph, left:left+pw].squeeze(0) for (top, left) in chunk],
                dim=0,
            )  # [N,C,ph,pw]

            cond = torch.cat(
                [
                    _lookup_window_cond(
                        z_grid,
                        top_lat=top,
                        left_lat=left,
                        ph=ph,
                        pw=pw,
                        h_lat=H_lat,
                        w_lat=W_lat,
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

            hs = torch.cat([patch_in, patch_in], dim=0)  # [2N,C,ph,pw]
            es = torch.cat([uncond, cond], dim=0)
            tt = (t if torch.is_tensor(t) else torch.tensor(t, device=device)).long()
            if tt.dim() == 0:
                tt = tt[None]
            tt = tt.expand(hs.shape[0])

            out = pipeline.transformer(
                hidden_states=hs,
                encoder_hidden_states=es,
                timestep=tt,
                return_dict=True,
            )
            eps2 = out.sample if hasattr(out, "sample") else out[0]

            if eps2.shape[1] == 2 * C:
                eps2 = eps2[:, :C]
            elif eps2.shape[1] != C:
                raise RuntimeError(f"model output channels={eps2.shape[1]} != {C} (or {2*C})")

            eps_u, eps_c = eps2[:n], eps2[n:]
            eps = eps_u + guidance * (eps_c - eps_u)

            for j, (top, left) in enumerate(chunk):
                eps_accum[:, :, top:top+ph, left:left+pw] += eps[j:j+1] * gauss
                weight[:, :, top:top+ph, left:left+pw] += gauss

        eps_full = eps_accum / weight.clamp(min=1e-8)
        step_out = pipeline.scheduler.step(eps_full, t, latents_step, return_dict=True)
        latents = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

    if hasattr(pipeline.vae.config, "scaling_factor"):
        latents = latents / pipeline.vae.config.scaling_factor
    vae_param = next(pipeline.vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    img = pipeline.vae.decode(latents, return_dict=True).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    return img
