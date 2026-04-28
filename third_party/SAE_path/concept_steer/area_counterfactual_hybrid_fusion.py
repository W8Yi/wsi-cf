#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, DiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_steer.area_counterfactual_spatial_anchor_sweep import (
    _choose_slides_balanced,
    _find_slide_path,
    _load_prototypes,
    _load_test_rows,
    _make_area_from_anchor,
    _pick_latent,
    _read_h5_features_coords,
    _steer_area_grid,
    _tile_mosaic_from_idx_grid,
)
from concept_steer.run_hnsc_hpv_sae_neuron_pipeline import build_mil_from_checkpoint, run_mil_attention
from utils.diffusion import (
    infer_pixcell_cond_grid_side,
    infer_pixcell_native_patch_px,
    sample_multidiffusion_from_zgrid,
    sample_multidiffusion_from_zgrid_with_midref,
    sample_multidiffusion_from_zgrid_with_stepmask,
)
from utils.sae import load_sae_from_config

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from e


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Hybrid area counterfactual generation: generate naive + global-anchor outputs, then fuse with a "
            "smooth high-attention mask (naive-heavy on high-attn, anchor-heavy on low-attn)."
        )
    )
    ap.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.json",
    )
    ap.add_argument(
        "--split-tsv",
        type=Path,
        default=REPO_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.tsv",
    )
    ap.add_argument(
        "--features-root",
        type=Path,
        default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features"),
    )
    ap.add_argument(
        "--slides-root",
        type=Path,
        default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/slides"),
    )
    ap.add_argument(
        "--mil-ckpt",
        type=Path,
        default=REPO_ROOT / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt",
    )
    ap.add_argument(
        "--sae-ckpt",
        type=Path,
        default=REPO_ROOT / "runs/relu_sae_base/relu_final.pt",
    )
    ap.add_argument(
        "--sae-cfg",
        type=Path,
        default=REPO_ROOT / "runs/relu_sae_base/run_config.json",
    )
    ap.add_argument(
        "--prototype-npz",
        type=Path,
        default=REPO_ROOT
        / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/notebook_counterfactual_single_tile_and_area/prototype_vectors_for_selected_sae.npz",
    )
    ap.add_argument(
        "--prototype-key",
        type=str,
        default="prototype_median",
        choices=["prototype_mean", "prototype_median"],
    )
    ap.add_argument("--pos-latent", type=int, default=2645)
    ap.add_argument("--neg-latent", type=int, default=7036)

    ap.add_argument(
        "--manual-slide-keys",
        type=str,
        default="TCGA-CV-A460-01Z-00-DX1",
        help="Comma-separated slide_key list. If empty, uses balanced auto selection.",
    )
    ap.add_argument("--n-slides-total", type=int, default=2)
    ap.add_argument("--slide-selection", type=str, default="borderline", choices=["borderline", "first"])

    ap.add_argument("--area-side", type=int, default=7)
    ap.add_argument("--high-attn-fraction", type=float, default=0.10)
    ap.add_argument("--anchor-rank", type=int, default=1)
    ap.add_argument("--coord-step", type=int, default=0)
    ap.add_argument("--tile-size-20x", type=int, default=256)
    ap.add_argument("--out-tile-size", type=int, default=256)

    ap.add_argument("--directions", type=str, default="to_hpv_pos,to_hpv_neg")
    ap.add_argument("--strengths", type=str, default="0.00,0.20,0.40,0.60")
    ap.add_argument("--blend", type=float, default=0.65)
    ap.add_argument("--identity-at-zero", action="store_true")
    ap.add_argument("--norm-match-to-orig", action="store_true")

    ap.add_argument("--naive-steps", type=int, default=40)
    ap.add_argument("--naive-guidance", type=float, default=1.5)
    ap.add_argument("--anchor-steps", type=int, default=40)
    ap.add_argument("--anchor-guidance", type=float, default=1.5)
    ap.add_argument("--anchor-start-ratio", type=float, default=0.85)
    ap.add_argument("--anchor-mix", type=float, default=0.20)
    ap.add_argument(
        "--anchor-mode",
        type=str,
        default="midref",
        choices=["midref", "stepmask"],
        help="Anchor generation mode. stepmask applies editable/preserve mask at every diffusion step.",
    )
    ap.add_argument(
        "--stepmask-blur-cells",
        type=float,
        default=1.2,
        help="Mask blur sigma (tile cells) for stepmask anchor mode.",
    )
    ap.add_argument(
        "--stepmask-preserve-strength",
        type=float,
        default=1.0,
        help="Preserve strength in [0,1] for stepmask anchor mode.",
    )
    ap.add_argument(
        "--early-stop-target-delta",
        type=float,
        default=0.0,
        help=(
            "If >0, stop increasing strength for a slide+direction once directional delta reaches this target. "
            "Directional delta is (cf-orig) for to_hpv_pos and (orig-cf) for to_hpv_neg."
        ),
    )
    ap.add_argument(
        "--early-stop-min-strength",
        type=float,
        default=0.0,
        help="Only allow early stop at strengths >= this value.",
    )
    ap.add_argument(
        "--best-select-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "naive", "anchor", "all"],
        help="Which mode(s) to consider when picking recommended best strength.",
    )
    ap.add_argument(
        "--best-low-mse-weight",
        type=float,
        default=0.0,
        help=(
            "Score = directional_delta - best_low_mse_weight * low_mse. "
            "Set >0 to favor natural low-attention background."
        ),
    )

    ap.add_argument(
        "--fuse-high-weight",
        type=float,
        default=0.90,
        help="Naive weight in high-attention areas.",
    )
    ap.add_argument(
        "--fuse-low-weight",
        type=float,
        default=0.15,
        help="Naive weight in low-attention areas.",
    )
    ap.add_argument(
        "--fuse-mask-blur-cells",
        type=float,
        default=1.8,
        help="Gaussian blur sigma in tile-cell units for attention mask.",
    )

    ap.add_argument("--patch-px", type=int, default=0, help="0 = auto from PixCell model id.")
    ap.add_argument("--stride-px", type=int, default=0, help="0 = auto (half of resolved patch size).")
    ap.add_argument(
        "--pixcell-cond-grid-side",
        type=int,
        default=0,
        help="0 = auto infer from PixCell model id (1 for PixCell-256, 4 for PixCell-1024).",
    )
    ap.add_argument("--patch-batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fix-seed-across-modes", action="store_true", help="Use same seed for naive/anchor/fused per slide+dir+strength.")

    ap.add_argument("--pixcell-model", type=str, default="StonyBrook-CVLab/PixCell-256")
    ap.add_argument("--pixcell-custom-pipeline", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    ap.add_argument("--vae-model", type=str, default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/area_cf_hybrid_fusion",
    )
    return ap.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _parse_float_csv(v: str) -> list[float]:
    out = [float(x.strip()) for x in str(v).split(",") if x.strip()]
    if not out:
        raise ValueError("No values parsed from CSV float list.")
    return out


def _parse_str_csv(v: str) -> list[str]:
    out = [str(x.strip()) for x in str(v).split(",") if x.strip()]
    if not out:
        raise ValueError("No values parsed from CSV string list.")
    return out


def _resolve_pixcell_settings(
    *,
    pixcell_model: str,
    patch_px: int,
    stride_px: int,
    cond_grid_side: int,
) -> tuple[int, int, int]:
    resolved_patch_px = int(patch_px) if int(patch_px) > 0 else int(infer_pixcell_native_patch_px(pixcell_model))
    resolved_stride_px = int(stride_px) if int(stride_px) > 0 else max(1, resolved_patch_px // 2)
    resolved_cond_side = int(cond_grid_side) if int(cond_grid_side) > 0 else int(
        infer_pixcell_cond_grid_side(pixcell_model)
    )
    if resolved_patch_px <= 0 or resolved_stride_px <= 0 or resolved_cond_side <= 0:
        raise ValueError("Resolved PixCell settings must all be > 0.")
    return resolved_patch_px, resolved_stride_px, resolved_cond_side


def _to_uint8_pil(img_t: torch.Tensor) -> Image.Image:
    arr = (img_t[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _mask_alpha_from_high_mask(
    high_mask_grid: np.ndarray,
    *,
    out_h: int,
    out_w: int,
    low_weight: float,
    high_weight: float,
    blur_cells: float,
) -> np.ndarray:
    m = np.clip(np.asarray(high_mask_grid, dtype=np.float32), 0.0, 1.0)
    m_t = torch.from_numpy(m).view(1, 1, m.shape[0], m.shape[1]).float()
    if blur_cells > 0.0:
        radius = max(1, int(round(3.0 * blur_cells)))
        k = 2 * radius + 1
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
        ker = np.exp(-0.5 * (xx * xx + yy * yy) / float(blur_cells * blur_cells))
        ker = ker / max(float(ker.sum()), 1e-8)
        ker_t = torch.from_numpy(ker).view(1, 1, k, k).float()
        m_t = F.conv2d(m_t, ker_t, padding=radius)
        m_t = m_t.clamp(0.0, 1.0)
    m_px = F.interpolate(m_t, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0].numpy()
    alpha = float(low_weight) + (float(high_weight) - float(low_weight)) * m_px
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return alpha


def _fuse_naive_anchor(
    naive_pil: Image.Image,
    anchor_pil: Image.Image,
    alpha_map: np.ndarray,
) -> Image.Image:
    n = np.asarray(naive_pil).astype(np.float32) / 255.0
    a = np.asarray(anchor_pil).astype(np.float32) / 255.0
    if n.shape != a.shape:
        raise ValueError(f"naive and anchor image shape mismatch: {n.shape} vs {a.shape}")
    if alpha_map.shape != n.shape[:2]:
        raise ValueError(f"alpha map shape mismatch: {alpha_map.shape} vs {n.shape[:2]}")
    alpha3 = alpha_map[:, :, None]
    out = np.clip(alpha3 * n + (1.0 - alpha3) * a, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def _compute_mse_high_low(
    img: Image.Image,
    ref: Image.Image,
    high_mask: np.ndarray,
    tile_px: int,
) -> tuple[float, float, float]:
    a = np.asarray(img).astype(np.float32)
    b = np.asarray(ref).astype(np.float32)
    d = ((a - b) ** 2).mean(axis=2)
    high = np.kron(np.asarray(high_mask, dtype=np.float32), np.ones((tile_px, tile_px), dtype=np.float32))
    high = high[: d.shape[0], : d.shape[1]]
    low = 1.0 - high
    high_mse = float(d[high > 0.5].mean()) if (high > 0.5).sum() > 0 else float("nan")
    low_mse = float(d[low > 0.5].mean()) if (low > 0.5).sum() > 0 else float("nan")
    ratio = float(high_mse / (low_mse + 1e-8)) if np.isfinite(high_mse) and np.isfinite(low_mse) else float("nan")
    return high_mse, low_mse, ratio


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _directional_delta(direction: str, bag_prob_orig: float, bag_prob_cf: float) -> float:
    if direction == "to_hpv_pos":
        return float(bag_prob_cf - bag_prob_orig)
    if direction == "to_hpv_neg":
        return float(bag_prob_orig - bag_prob_cf)
    raise ValueError(f"Unsupported direction: {direction}")


def _render_strength_sheet(
    *,
    ref_img: Image.Image,
    direction: str,
    strengths: list[float],
    records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    thumb = 256
    pad = 10
    header_h = 56
    caption_h = 42
    cols = 1 + 3  # reference + naive + anchor + fused
    rows_n = len(strengths)
    w = pad + cols * (thumb + pad)
    h = header_h + pad + rows_n * (caption_h + thumb + pad)
    canvas = Image.new("RGB", (w, h), (242, 242, 242))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, header_h], fill=(225, 225, 225))
    draw.text((8, 8), f"direction={direction} | columns: ref, naive, anchor, hybrid", fill=(0, 0, 0))
    draw.text((8, 30), "caption: bag P(HPV+) and delta vs original", fill=(0, 0, 0))

    rec_map: dict[tuple[float, str], dict[str, Any]] = {}
    for r in records:
        rec_map[(float(r["strength"]), str(r["mode"]))] = r

    ref_thumb = ref_img.resize((thumb, thumb), resample=Image.BILINEAR)
    for ri, s in enumerate(strengths):
        y0 = header_h + pad + ri * (caption_h + thumb + pad)
        x_ref = pad
        draw.rectangle([x_ref, y0, x_ref + thumb, y0 + caption_h], fill=(255, 255, 255))
        draw.text((x_ref + 4, y0 + 4), f"s={s:.2f}\nreference", fill=(0, 0, 0))
        canvas.paste(ref_thumb, (x_ref, y0 + caption_h))

        for ci, mode in enumerate(["naive", "anchor", "hybrid"], start=1):
            x0 = pad + ci * (thumb + pad)
            draw.rectangle([x0, y0, x0 + thumb, y0 + caption_h], fill=(255, 255, 255))
            rr = rec_map.get((float(s), mode))
            if rr is None:
                draw.text((x0 + 4, y0 + 4), f"{mode}\nmissing", fill=(0, 0, 0))
                continue
            im = Image.open(str(rr["image_path"])).convert("RGB").resize((thumb, thumb), resample=Image.BILINEAR)
            txt = (
                f"{mode}\n"
                f"p={float(rr['bag_prob_cf']):.3f} d={float(rr['bag_delta_prob_cf_minus_orig']):+.3f}"
            )
            draw.text((x0 + 4, y0 + 4), txt, fill=(0, 0, 0))
            canvas.paste(im, (x0, y0 + caption_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    strengths = _parse_float_csv(args.strengths)
    directions = _parse_str_csv(args.directions)

    for d in directions:
        if d not in {"to_hpv_pos", "to_hpv_neg"}:
            raise ValueError(f"Unsupported direction: {d}")
    if not (0.0 <= args.fuse_low_weight <= 1.0 and 0.0 <= args.fuse_high_weight <= 1.0):
        raise ValueError("fuse weights must be in [0,1].")

    required = [
        args.split_json,
        args.split_tsv,
        args.features_root,
        args.slides_root,
        args.mil_ckpt,
        args.sae_ckpt,
        args.sae_cfg,
        args.prototype_npz,
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))
    if device.type != "cuda":
        raise RuntimeError("This script currently expects CUDA for PixCell generation.")

    print(f"[setup] device={device}")
    rows = _load_test_rows(args.split_json, args.split_tsv, args.features_root)
    print(f"[setup] test rows={len(rows)}")
    if not rows:
        raise SystemExit("No test rows found.")

    print("[setup] loading MIL")
    mil_model = build_mil_from_checkpoint(args.mil_ckpt, device)
    print("[setup] selecting slides")
    manual = [s.strip() for s in str(args.manual_slide_keys).split(",") if s.strip()]
    if manual:
        keyset = set(manual)
        selected = [r for r in rows if str(r["slide_key"]) in keyset]
        found = set(str(r["slide_key"]) for r in selected)
        missing_keys = sorted([k for k in manual if k not in found])
        if missing_keys:
            raise RuntimeError(f"Manual slide_key(s) not found in test rows: {missing_keys}")
        selected.sort(key=lambda r: (int(r["label"]), str(r["slide_key"])))
    else:
        selected = _choose_slides_balanced(
            rows,
            mil_model=mil_model,
            device=device,
            n_total=int(args.n_slides_total),
            mode=str(args.slide_selection),
        )
    print(f"[setup] selected slides={len(selected)}")

    print("[setup] loading SAE")
    sae_model, d_in, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_table, dir_table = _load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = _pick_latent(dir_table, proto_table, int(args.pos_latent), "hpv_pos")
    neg_latent = _pick_latent(dir_table, proto_table, int(args.neg_latent), "hpv_neg")
    proto_pos = np.asarray(proto_table[pos_latent], dtype=np.float32)
    proto_neg = np.asarray(proto_table[neg_latent], dtype=np.float32)
    print(f"[setup] prototype latents: hpv_pos={pos_latent}, hpv_neg={neg_latent}")

    print("[setup] loading PixCell pipeline")
    sd3_vae = AutoencoderKL.from_pretrained(args.vae_model, subfolder="vae")
    pipe = DiffusionPipeline.from_pretrained(
        args.pixcell_model,
        vae=sd3_vae,
        custom_pipeline=args.pixcell_custom_pipeline,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    pipe.to(str(device))
    pipe_dtype = next(pipe.transformer.parameters()).dtype
    patch_px, stride_px, cond_grid_side = _resolve_pixcell_settings(
        pixcell_model=str(args.pixcell_model),
        patch_px=int(args.patch_px),
        stride_px=int(args.stride_px),
        cond_grid_side=int(args.pixcell_cond_grid_side),
    )
    print(
        f"[setup] PixCell window config: model={args.pixcell_model} patch_px={patch_px} "
        f"stride_px={stride_px} cond_grid_side={cond_grid_side}"
    )

    selection_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    failed_slides: list[dict[str, Any]] = []
    slide_path_cache: dict[str, Path | None] = {}

    for si, row in enumerate(selected, start=1):
        slide_key = str(row["slide_key"])
        label = int(row["label"])
        case_id = str(row["case_id"])
        h5_path = str(row["h5_path"])
        print(f"[slide {si}/{len(selected)}] {slide_key} label={label}")

        try:
            x, coords = _read_h5_features_coords(h5_path)
            if coords is None:
                raise RuntimeError("coords missing in H5.")
            if x.shape[1] != d_in:
                raise RuntimeError(f"Feature dim mismatch: {x.shape[1]} != SAE d_in {d_in}")

            attn, full_pred_orig, full_prob_orig = run_mil_attention(mil_model, x, device=device)
            area = _make_area_from_anchor(
                x=x,
                coords=coords,
                attn=attn,
                area_side=int(args.area_side),
                anchor_rank=int(args.anchor_rank),
                high_fraction=float(args.high_attn_fraction),
                coord_step=int(args.coord_step),
            )

            slide_path = _find_slide_path(args.slides_root, slide_key, slide_path_cache)
            if slide_path is None:
                raise RuntimeError(f"Slide file not found for {slide_key}")
            slide_obj = openslide.OpenSlide(str(slide_path))
            try:
                ref_area_pil, overlay_pil = _tile_mosaic_from_idx_grid(
                    slide_obj,
                    coords=coords,
                    idx_grid=area["idx_grid"],
                    high_mask=area["high_mask"],
                    coord_grid=area["coord_grid"],
                    tile_size_20x=int(args.tile_size_20x),
                    out_tile_size=int(args.out_tile_size),
                )
            finally:
                slide_obj.close()

            slide_dir = args.out_dir / slide_key
            slide_dir.mkdir(parents=True, exist_ok=True)
            ref_path = slide_dir / "area_reference.png"
            ov_path = slide_dir / "area_high_low_overlay.png"
            ref_area_pil.save(ref_path)
            overlay_pil.save(ov_path)

            real_indices = np.asarray(area["real_indices"], dtype=np.int64)
            x_area_orig = np.asarray(x[real_indices], dtype=np.float32)
            _, bag_pred_orig, bag_prob_orig = run_mil_attention(mil_model, x_area_orig, device=device)

            selection_rows.append(
                {
                    "slide_key": slide_key,
                    "case_id": case_id,
                    "label": int(label),
                    "h5_path": h5_path,
                    "slide_path": str(slide_path),
                    "full_pred_orig": int(full_pred_orig),
                    "full_prob_orig": float(full_prob_orig),
                    "bag_pred_orig": int(bag_pred_orig),
                    "bag_prob_orig": float(bag_prob_orig),
                    "area_side": int(args.area_side),
                    "high_attn_fraction": float(args.high_attn_fraction),
                    "n_real_tiles_in_area": int(real_indices.size),
                    "n_high_selected": int(area["n_high_selected"]),
                    "anchor_idx": int(area["anchor_idx"]),
                    "anchor_coord_x": int(area["anchor_coord"][0]),
                    "anchor_coord_y": int(area["anchor_coord"][1]),
                    "reference_image_path": str(ref_path),
                    "overlay_image_path": str(ov_path),
                }
            )

            idx_flat = area["idx_grid"].reshape(-1)
            real_flat = area["real_mask"].reshape(-1) > 0.5
            high_flat = area["high_mask"].reshape(-1) > 0.5
            high_real_tile_indices = sorted(set(int(idx_flat[i]) for i in np.where(real_flat & high_flat)[0].tolist()))

            ref_np = np.asarray(ref_area_pil).astype(np.float32) / 255.0
            ref_t = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            out_h, out_w = ref_area_pil.height, ref_area_pil.width
            alpha_map = _mask_alpha_from_high_mask(
                area["high_mask"],
                out_h=out_h,
                out_w=out_w,
                low_weight=float(args.fuse_low_weight),
                high_weight=float(args.fuse_high_weight),
                blur_cells=float(args.fuse_mask_blur_cells),
            )

            for di, direction in enumerate(directions):
                proto_vec = proto_pos if direction == "to_hpv_pos" else proto_neg
                dir_dir = slide_dir / direction
                dir_dir.mkdir(parents=True, exist_ok=True)
                dir_records: list[dict[str, Any]] = []
                tested_strengths: list[float] = []

                for sj, s in enumerate(strengths):
                    s = float(s)
                    if bool(args.identity_at_zero) and abs(s) < 1e-12:
                        z_cf = np.asarray(area["z_grid"], dtype=np.float32).copy()
                    else:
                        z_cf = _steer_area_grid(
                            z_grid=np.asarray(area["z_grid"], dtype=np.float32),
                            high_mask=np.asarray(area["high_mask"], dtype=np.float32),
                            proto_vec=proto_vec,
                            strength=s,
                            sae_model=sae_model,
                            blend=float(args.blend),
                            device=device,
                            norm_match_to_orig=bool(args.norm_match_to_orig),
                        )

                    x_cf_full = np.asarray(x, dtype=np.float32).copy()
                    z_flat = z_cf.reshape(-1, z_cf.shape[-1])
                    for k in high_real_tile_indices:
                        loc = np.where(idx_flat == int(k))[0]
                        if loc.size == 0:
                            continue
                        x_cf_full[int(k)] = z_flat[int(loc[0])]
                    x_area_cf = np.asarray(x_cf_full[real_indices], dtype=np.float32)
                    _, bag_pred_cf, bag_prob_cf = run_mil_attention(mil_model, x_area_cf, device=device)
                    dir_delta = _directional_delta(direction, float(bag_prob_orig), float(bag_prob_cf))

                    z_cf_t = torch.from_numpy(z_cf).to(device=device, dtype=pipe_dtype)
                    base_seed = int(args.seed + si * 100000 + di * 10000 + sj * 1000)
                    g_naive = torch.Generator(device=device).manual_seed(base_seed)
                    if bool(args.fix_seed_across_modes):
                        g_anchor = torch.Generator(device=device).manual_seed(base_seed)
                    else:
                        g_anchor = torch.Generator(device=device).manual_seed(base_seed + 17)

                    ac = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                    with torch.inference_mode(), ac:
                        naive_t = sample_multidiffusion_from_zgrid(
                            pipeline=pipe,
                            z_grid=z_cf_t,
                            out_h=out_h,
                            out_w=out_w,
                            patch_px=patch_px,
                            stride_px=stride_px,
                            cond_grid_side=cond_grid_side,
                            steps=int(args.naive_steps),
                            guidance=float(args.naive_guidance),
                            patch_batch=int(args.patch_batch),
                            generator=g_naive,
                        )
                        if str(args.anchor_mode) == "stepmask":
                            anchor_t = sample_multidiffusion_from_zgrid_with_stepmask(
                                pipeline=pipe,
                                z_grid=z_cf_t,
                                original_image=ref_t,
                                editable_mask_grid=np.asarray(area["high_mask"], dtype=np.float32),
                                out_h=out_h,
                                out_w=out_w,
                                patch_px=patch_px,
                                stride_px=stride_px,
                                cond_grid_side=cond_grid_side,
                                steps=int(args.anchor_steps),
                                guidance=float(args.anchor_guidance),
                                patch_batch=int(args.patch_batch),
                                mask_blur_cells=float(args.stepmask_blur_cells),
                                preserve_strength=float(args.stepmask_preserve_strength),
                                generator=g_anchor,
                            )
                        else:
                            anchor_t = sample_multidiffusion_from_zgrid_with_midref(
                                pipeline=pipe,
                                z_grid=z_cf_t,
                                original_image=ref_t,
                                out_h=out_h,
                                out_w=out_w,
                                patch_px=patch_px,
                                stride_px=stride_px,
                                cond_grid_side=cond_grid_side,
                                steps=int(args.anchor_steps),
                                guidance=float(args.anchor_guidance),
                                patch_batch=int(args.patch_batch),
                                reference_start_ratio=float(args.anchor_start_ratio),
                                reference_mix=float(args.anchor_mix),
                                generator=g_anchor,
                            )

                    naive_pil = _to_uint8_pil(naive_t)
                    anchor_pil = _to_uint8_pil(anchor_t)
                    hybrid_pil = _fuse_naive_anchor(naive_pil, anchor_pil, alpha_map=alpha_map)

                    s_tag = f"{s:.2f}".replace(".", "p")
                    p_tag = f"{bag_prob_cf:.4f}".replace(".", "p")
                    naive_path = dir_dir / f"naive__s_{s_tag}__bagpred_{int(bag_pred_cf)}__p_{p_tag}.png"
                    anchor_path = dir_dir / f"anchor__s_{s_tag}__bagpred_{int(bag_pred_cf)}__p_{p_tag}.png"
                    hybrid_path = dir_dir / f"hybrid__s_{s_tag}__bagpred_{int(bag_pred_cf)}__p_{p_tag}.png"
                    naive_pil.save(naive_path)
                    anchor_pil.save(anchor_path)
                    hybrid_pil.save(hybrid_path)

                    for mode, im, path in [
                        ("naive", naive_pil, naive_path),
                        ("anchor", anchor_pil, anchor_path),
                        ("hybrid", hybrid_pil, hybrid_path),
                    ]:
                        high_mse, low_mse, ratio = _compute_mse_high_low(
                            im,
                            ref_area_pil,
                            high_mask=np.asarray(area["high_mask"], dtype=np.float32),
                            tile_px=int(args.out_tile_size),
                        )
                        rec = {
                            "slide_key": slide_key,
                            "case_id": case_id,
                            "label": int(label),
                            "direction": direction,
                            "strength": float(s),
                            "mode": mode,
                            "full_pred_orig": int(full_pred_orig),
                            "full_prob_orig": float(full_prob_orig),
                            "bag_pred_orig": int(bag_pred_orig),
                            "bag_prob_orig": float(bag_prob_orig),
                            "bag_pred_cf": int(bag_pred_cf),
                            "bag_prob_cf": float(bag_prob_cf),
                            "bag_delta_prob_cf_minus_orig": float(bag_prob_cf - bag_prob_orig),
                            "directional_delta": float(dir_delta),
                            "area_side": int(args.area_side),
                            "high_attn_fraction": float(args.high_attn_fraction),
                            "n_real_tiles_in_area": int(real_indices.size),
                            "n_high_selected": int(area["n_high_selected"]),
                            "high_mse": float(high_mse),
                            "low_mse": float(low_mse),
                            "high_low_mse_ratio": float(ratio),
                            "fuse_high_weight": float(args.fuse_high_weight),
                            "fuse_low_weight": float(args.fuse_low_weight),
                            "fuse_mask_blur_cells": float(args.fuse_mask_blur_cells),
                            "image_path": str(path),
                            "reference_image_path": str(ref_path),
                            "overlay_image_path": str(ov_path),
                        }
                        records.append(rec)
                        dir_records.append(rec)
                    tested_strengths.append(float(s))

                    if float(args.early_stop_target_delta) > 0.0 and float(s) >= float(args.early_stop_min_strength):
                        if float(dir_delta) >= float(args.early_stop_target_delta):
                            print(
                                f"[early-stop] {slide_key} {direction} at s={s:.3f} "
                                f"(directional_delta={dir_delta:.6f} >= target={float(args.early_stop_target_delta):.6f})"
                            )
                            break

                _render_strength_sheet(
                    ref_img=ref_area_pil,
                    direction=direction,
                    strengths=strengths,
                    records=dir_records,
                    out_path=slide_dir / "plots" / f"hybrid_strength_sheet_{direction}.png",
                )

                # Auto-recommend best strength for this slide+direction.
                if dir_records:
                    ddf = pd.DataFrame(dir_records)
                    mode_sel = str(args.best_select_mode)
                    if mode_sel != "all":
                        ddf = ddf[ddf["mode"] == mode_sel].copy()
                    if not ddf.empty:
                        ddf["select_score"] = ddf["directional_delta"] - float(args.best_low_mse_weight) * ddf["low_mse"]
                        best_idx = int(ddf["select_score"].idxmax())
                        br = ddf.loc[best_idx]
                        meets_target = (
                            float(args.early_stop_target_delta) > 0.0
                            and float(br["directional_delta"]) >= float(args.early_stop_target_delta)
                        )
                        best_rows.append(
                            {
                                "slide_key": slide_key,
                                "case_id": case_id,
                                "label": int(label),
                                "direction": direction,
                                "best_mode": str(br["mode"]),
                                "best_strength": float(br["strength"]),
                                "best_directional_delta": float(br["directional_delta"]),
                                "best_bag_prob_orig": float(br["bag_prob_orig"]),
                                "best_bag_prob_cf": float(br["bag_prob_cf"]),
                                "best_bag_delta_cf_minus_orig": float(br["bag_delta_prob_cf_minus_orig"]),
                                "best_low_mse": float(br["low_mse"]),
                                "best_high_mse": float(br["high_mse"]),
                                "best_high_low_ratio": float(br["high_low_mse_ratio"]),
                                "best_select_score": float(br["select_score"]),
                                "best_image_path": str(br["image_path"]),
                                "strengths_requested": ",".join(f"{float(v):.3f}" for v in strengths),
                                "strengths_tested": ",".join(f"{float(v):.3f}" for v in tested_strengths),
                                "best_select_mode": mode_sel,
                                "best_low_mse_weight": float(args.best_low_mse_weight),
                                "early_stop_target_delta": float(args.early_stop_target_delta),
                                "early_stop_min_strength": float(args.early_stop_min_strength),
                                "meets_target": bool(meets_target),
                            }
                        )

        except Exception as e:
            failed_slides.append({"slide_key": slide_key, "error": str(e)})
            print(f"[warn] failed slide {slide_key}: {e}")
            continue

    rec_csv = args.out_dir / "hybrid_fusion_records.csv"
    sel_csv = args.out_dir / "selected_slides_and_areas.csv"
    best_csv = args.out_dir / "best_strength_recommendations.csv"
    _write_csv(rec_csv, records)
    _write_csv(sel_csv, selection_rows)
    _write_csv(best_csv, best_rows)

    summary = {
        "n_test_rows_total": int(len(rows)),
        "n_selected": int(len(selected)),
        "n_failed_slides": int(len(failed_slides)),
        "n_success_slides": int(len(selected) - len(failed_slides)),
        "n_records": int(len(records)),
        "directions": directions,
        "strengths": [float(s) for s in strengths],
        "prototype_key": str(args.prototype_key),
        "prototype_latent_pos": int(pos_latent),
        "prototype_latent_neg": int(neg_latent),
        "naive_steps": int(args.naive_steps),
        "naive_guidance": float(args.naive_guidance),
        "anchor_steps": int(args.anchor_steps),
        "anchor_guidance": float(args.anchor_guidance),
        "anchor_start_ratio": float(args.anchor_start_ratio),
        "anchor_mix": float(args.anchor_mix),
        "anchor_mode": str(args.anchor_mode),
        "stepmask_blur_cells": float(args.stepmask_blur_cells),
        "stepmask_preserve_strength": float(args.stepmask_preserve_strength),
        "fuse_high_weight": float(args.fuse_high_weight),
        "fuse_low_weight": float(args.fuse_low_weight),
        "fuse_mask_blur_cells": float(args.fuse_mask_blur_cells),
        "resolved_patch_px": int(patch_px),
        "resolved_stride_px": int(stride_px),
        "resolved_cond_grid_side": int(cond_grid_side),
        "records_csv": str(rec_csv),
        "selected_csv": str(sel_csv),
        "best_csv": str(best_csv),
        "early_stop_target_delta": float(args.early_stop_target_delta),
        "early_stop_min_strength": float(args.early_stop_min_strength),
        "best_select_mode": str(args.best_select_mode),
        "best_low_mse_weight": float(args.best_low_mse_weight),
        "failed_slides": failed_slides,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[done] wrote:")
    print(f"  {rec_csv}")
    print(f"  {sel_csv}")
    print(f"  {best_csv}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
