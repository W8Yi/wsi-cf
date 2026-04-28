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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, DiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_steer.run_hnsc_hpv_sae_neuron_pipeline import build_mil_from_checkpoint, run_mil_attention
from utils.diffusion import sample_multidiffusion_from_zgrid, sample_multidiffusion_from_zgrid_with_midref
from utils.sae import load_sae_from_config
from utils.sae_edit import edit_uni_z_grid_with_sae

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide are required.") from e


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Counterfactual area steering with spatial background-only anchoring: "
            "select balanced slides, build local high-attention area, edit only high-attention tiles, "
            "regenerate area with multiple settings, and compare predictions + naturalness."
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

    ap.add_argument("--n-slides-total", type=int, default=10, help="Total selected slides; balanced across labels.")
    ap.add_argument(
        "--manual-slide-keys",
        type=str,
        default="",
        help="Optional comma-separated slide_key list. If set, bypasses balanced auto-selection.",
    )
    ap.add_argument(
        "--slide-selection",
        type=str,
        default="borderline",
        choices=["borderline", "first"],
        help="borderline = slides with full-slide p close to 0.5, per label.",
    )

    ap.add_argument("--area-side", type=int, default=7, help="Square area side in tile units (odd recommended).")
    ap.add_argument(
        "--high-attn-fraction",
        type=float,
        default=0.30,
        help="Fraction of total area tiles marked high-attention and edited.",
    )
    ap.add_argument("--anchor-rank", type=int, default=1, help="Anchor tile attention rank for area center.")
    ap.add_argument("--tile-size-20x", type=int, default=256, help="Read size at 20x for each tile patch.")
    ap.add_argument("--out-tile-size", type=int, default=256, help="Rendered patch size in area mosaic.")
    ap.add_argument("--coord-step", type=int, default=0, help="Set >0 to force coordinate grid step; 0 auto-estimate.")

    ap.add_argument(
        "--strengths",
        type=str,
        default="0.00,0.35,0.70,1.00",
    )
    ap.add_argument(
        "--directions",
        type=str,
        default="to_hpv_pos,to_hpv_neg",
    )
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--identity-at-zero", action="store_true")
    ap.add_argument("--norm-match-to-orig", action="store_true")

    ap.add_argument("--settings-json", type=Path, default=None, help="Optional JSON file overriding generation settings.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--patch-px", type=int, default=256)
    ap.add_argument("--stride-px", type=int, default=128)
    ap.add_argument("--patch-batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-slides", type=int, default=0, help="Debug cap after selection; 0 disables.")

    ap.add_argument("--pixcell-model", type=str, default="StonyBrook-CVLab/PixCell-256")
    ap.add_argument("--pixcell-custom-pipeline", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    ap.add_argument("--vae-model", type=str, default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/area_cf_spatial_anchor_sweep",
    )
    return ap.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _parse_float_csv(v: str) -> list[float]:
    out = [float(x.strip()) for x in str(v).split(",") if x.strip()]
    if not out:
        raise ValueError("No values parsed.")
    return out


def _parse_str_csv(v: str) -> list[str]:
    out = [str(x.strip()) for x in str(v).split(",") if x.strip()]
    if not out:
        raise ValueError("No values parsed.")
    return out


def _parse_slide_key_from_h5_path(h5_path: str) -> str:
    return Path(h5_path).name.split(".")[0]


def _load_split_json_test_map(split_json: Path) -> dict[str, str]:
    payload = json.loads(split_json.read_text())
    out: dict[str, str] = {}
    for p in payload.get("test", []):
        k = _parse_slide_key_from_h5_path(str(p))
        out.setdefault(k, str(p))
    return out


def _resolve_h5_for_slide(slide_key: str, tsv_h5: str, json_h5: str | None, features_root: Path) -> Path | None:
    cands = [
        features_root / f"{slide_key}.h5",
        Path(tsv_h5) if tsv_h5 else None,
        Path(json_h5) if json_h5 else None,
    ]
    for c in cands:
        if c is not None and c.exists():
            return c
    return None


def _load_test_rows(split_json: Path, split_tsv: Path, features_root: Path) -> list[dict[str, Any]]:
    test_map = _load_split_json_test_map(split_json)
    rows: list[dict[str, Any]] = []
    with split_tsv.open("r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if str(row.get("split", "")) != "test":
                continue
            slide_key = str(row.get("slide_key", ""))
            if slide_key not in test_map:
                continue
            label = int(row.get("label", -1))
            if label not in (0, 1):
                continue
            h5_path = _resolve_h5_for_slide(
                slide_key=slide_key,
                tsv_h5=str(row.get("h5_path", "")),
                json_h5=test_map.get(slide_key),
                features_root=features_root,
            )
            if h5_path is None:
                continue
            rows.append(
                {
                    "slide_key": slide_key,
                    "case_id": str(row.get("case_id", slide_key)),
                    "label": int(label),
                    "h5_path": str(h5_path),
                }
            )
    rows.sort(key=lambda z: (int(z["label"]), str(z["slide_key"])))
    return rows


def _read_h5_features_coords(h5_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]
        if feats.ndim == 2:
            x = feats[:]
        elif feats.ndim == 3 and int(feats.shape[0]) == 1:
            x = feats[0]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")

        coords = None
        if "coords" in f:
            c = f["coords"]
            if c.ndim == 2 and c.shape[1] == 2:
                coords = c[:]
            elif c.ndim == 3 and int(c.shape[0]) == 1:
                coords = c[0]
    return np.asarray(x, dtype=np.float32), (np.asarray(coords) if coords is not None else None)


def _load_prototypes(npz_path: Path, key: str) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    with np.load(npz_path, allow_pickle=False) as d:
        latent_ids = np.asarray(d["latent_ids"], dtype=np.int64).reshape(-1)
        vecs = np.asarray(d[key], dtype=np.float32)
        dirs = np.asarray(d["selected_direction"]).astype(str)
    table = {int(latent_ids[i]): vecs[i] for i in range(latent_ids.shape[0])}
    dir_table = {int(latent_ids[i]): str(dirs[i]) for i in range(latent_ids.shape[0])}
    return table, dir_table


def _pick_latent(dir_table: dict[int, str], table: dict[int, np.ndarray], preferred: int, direction: str) -> int:
    if preferred in table:
        return int(preferred)
    cands = sorted([lid for lid, d in dir_table.items() if d == direction])
    if not cands:
        raise RuntimeError(f"No prototype latent for direction={direction}")
    return int(cands[0])


def _infer_objective_power(slide: "openslide.OpenSlide") -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            try:
                v = float(props.get(key))
            except Exception:
                v = -1.0
            if v > 0:
                return v
    try:
        mpp_x = float(props.get("openslide.mpp-x", -1.0))
    except Exception:
        mpp_x = -1.0
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def _level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(tile_size_20x * (objective_power / 20.0))))


def _crop_tile_rgb(
    slide: "openslide.OpenSlide",
    x: int,
    y: int,
    tile_size_20x: int,
    out_tile_size: int,
) -> Image.Image:
    objective = _infer_objective_power(slide)
    crop_px = _level0_tile_size(tile_size_20x, objective)
    rgba = slide.read_region((int(x), int(y)), 0, (crop_px, crop_px)).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    if crop_px != out_tile_size:
        rgb = rgb.resize((out_tile_size, out_tile_size), resample=Image.BILINEAR)
    return rgb


def _find_slide_path(slides_root: Path, slide_key: str, cache: dict[str, Path | None]) -> Path | None:
    if slide_key in cache:
        return cache[slide_key]
    for ext in (".svs", ".tif", ".tiff", ".ndpi", ".mrxs"):
        p = slides_root / f"{slide_key}{ext}"
        if p.exists():
            cache[slide_key] = p
            return p
    for ext in ("svs", "tif", "tiff", "ndpi", "mrxs"):
        hits = sorted(slides_root.glob(f"{slide_key}*.{ext}"))
        if hits:
            cache[slide_key] = hits[0]
            return hits[0]
    cache[slide_key] = None
    return None


def _estimate_step(coords: np.ndarray, forced: int = 0) -> int:
    if forced > 0:
        return int(forced)
    xs = np.unique(np.asarray(coords[:, 0], dtype=np.int64))
    ys = np.unique(np.asarray(coords[:, 1], dtype=np.int64))
    dx = np.diff(np.sort(xs))
    dy = np.diff(np.sort(ys))
    dx = dx[dx > 0]
    dy = dy[dy > 0]
    vals: list[float] = []
    if dx.size > 0:
        vals.append(float(np.median(dx)))
    if dy.size > 0:
        vals.append(float(np.median(dy)))
    if not vals:
        return 256
    return max(1, int(round(float(np.median(np.asarray(vals, dtype=np.float32))))))


def _choose_slides_balanced(
    rows: list[dict[str, Any]],
    mil_model: torch.nn.Module,
    device: torch.device,
    n_total: int,
    mode: str,
) -> list[dict[str, Any]]:
    if n_total <= 0:
        raise ValueError("--n-slides-total must be > 0")
    n0 = n_total // 2
    n1 = n_total - n0

    rows0 = [r for r in rows if int(r["label"]) == 0]
    rows1 = [r for r in rows if int(r["label"]) == 1]
    if len(rows0) < n0 or len(rows1) < n1:
        raise RuntimeError(f"Not enough slides for balanced selection: label0={len(rows0)} label1={len(rows1)}")

    if mode == "first":
        picked = rows0[:n0] + rows1[:n1]
        picked.sort(key=lambda r: (int(r["label"]), str(r["slide_key"])))
        return picked

    scored: list[dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        x, _ = _read_h5_features_coords(str(r["h5_path"]))
        _, pred, prob = run_mil_attention(mil_model, x, device=device)
        rr = dict(r)
        rr["full_pred_orig"] = int(pred)
        rr["full_prob_orig"] = float(prob)
        rr["margin_to_0p5"] = float(abs(prob - 0.5))
        scored.append(rr)
        if i % 20 == 0 or i == len(rows):
            print(f"[slide-select] scored {i}/{len(rows)}")

    s0 = sorted([r for r in scored if int(r["label"]) == 0], key=lambda z: z["margin_to_0p5"])
    s1 = sorted([r for r in scored if int(r["label"]) == 1], key=lambda z: z["margin_to_0p5"])
    picked = s0[:n0] + s1[:n1]
    picked.sort(key=lambda r: (int(r["label"]), str(r["slide_key"])))
    return picked


def _make_area_from_anchor(
    x: np.ndarray,
    coords: np.ndarray,
    attn: np.ndarray,
    *,
    area_side: int,
    anchor_rank: int,
    high_fraction: float,
    coord_step: int,
) -> dict[str, Any]:
    if coords is None:
        raise RuntimeError("coords are required for area selection.")
    if area_side <= 0:
        raise ValueError("area_side must be > 0")
    if not (0.0 < high_fraction <= 1.0):
        raise ValueError("high_fraction must be in (0,1].")

    step = _estimate_step(coords, forced=coord_step)
    gx = np.round(coords[:, 0].astype(np.float64) / float(step)).astype(np.int64)
    gy = np.round(coords[:, 1].astype(np.float64) / float(step)).astype(np.int64)

    order = np.argsort(attn)[::-1]
    rank = max(1, int(anchor_rank))
    if rank > len(order):
        rank = len(order)
    anchor_idx = int(order[rank - 1])
    ax = int(gx[anchor_idx])
    ay = int(gy[anchor_idx])

    half = area_side // 2
    x_range = [ax - half + c for c in range(area_side)]
    y_range = [ay - half + r for r in range(area_side)]
    anchor_coord_x = int(coords[anchor_idx, 0])
    anchor_coord_y = int(coords[anchor_idx, 1])
    x_coord_range = [int(anchor_coord_x + (gxv - ax) * step) for gxv in x_range]
    y_coord_range = [int(anchor_coord_y + (gyv - ay) * step) for gyv in y_range]

    map_exact: dict[tuple[int, int], int] = {}
    for i in range(x.shape[0]):
        key = (int(gy[i]), int(gx[i]))
        if key not in map_exact:
            map_exact[key] = int(i)
        else:
            if float(attn[i]) > float(attn[map_exact[key]]):
                map_exact[key] = int(i)

    g_all = np.stack([gy, gx], axis=1).astype(np.float32)
    z_grid = np.zeros((area_side, area_side, x.shape[1]), dtype=np.float32)
    idx_grid = -np.ones((area_side, area_side), dtype=np.int64)
    real_mask = np.zeros((area_side, area_side), dtype=np.float32)
    attn_grid = np.zeros((area_side, area_side), dtype=np.float32)
    coord_grid = np.zeros((area_side, area_side, 2), dtype=np.int64)

    for r, yy in enumerate(y_range):
        for c, xx in enumerate(x_range):
            coord_grid[r, c, 0] = int(x_coord_range[c])
            coord_grid[r, c, 1] = int(y_coord_range[r])
            key = (int(yy), int(xx))
            if key in map_exact:
                idx = int(map_exact[key])
                real = 1.0
            else:
                d = (g_all[:, 0] - float(yy)) ** 2 + (g_all[:, 1] - float(xx)) ** 2
                idx = int(np.argmin(d))
                real = 0.0
            z_grid[r, c] = x[idx]
            idx_grid[r, c] = idx
            real_mask[r, c] = real
            attn_grid[r, c] = float(attn[idx]) if real > 0 else float(np.min(attn))

    n_total = int(area_side * area_side)
    n_high_target = max(1, int(round(float(high_fraction) * float(n_total))))
    real_flat = real_mask.reshape(-1) > 0.5
    attn_flat = attn_grid.reshape(-1)
    cand = np.where(real_flat)[0]
    if cand.size == 0:
        raise RuntimeError("No real tiles in selected area.")
    ord_local = cand[np.argsort(attn_flat[cand])[::-1]]
    n_high = min(n_high_target, int(ord_local.size))
    high_flat = np.zeros(n_total, dtype=np.float32)
    high_flat[ord_local[:n_high]] = 1.0
    high_mask = high_flat.reshape(area_side, area_side)
    low_mask = 1.0 - high_mask

    real_indices = sorted(set(int(i) for i in idx_grid.reshape(-1)[real_flat].tolist()))
    return {
        "step": int(step),
        "anchor_idx": int(anchor_idx),
        "anchor_coord": [int(coords[anchor_idx, 0]), int(coords[anchor_idx, 1])],
        "anchor_grid": [int(ax), int(ay)],
        "x_range": [int(x_range[0]), int(x_range[-1])],
        "y_range": [int(y_range[0]), int(y_range[-1])],
        "z_grid": z_grid,
        "idx_grid": idx_grid,
        "coord_grid": coord_grid,
        "real_mask": real_mask,
        "high_mask": high_mask,
        "low_mask": low_mask,
        "attn_grid": attn_grid,
        "real_indices": np.asarray(real_indices, dtype=np.int64),
        "n_high_target": int(n_high_target),
        "n_high_selected": int(n_high),
    }


def _tile_mosaic_from_idx_grid(
    slide: "openslide.OpenSlide",
    coords: np.ndarray,
    idx_grid: np.ndarray,
    high_mask: np.ndarray,
    coord_grid: np.ndarray | None = None,
    *,
    tile_size_20x: int,
    out_tile_size: int,
) -> tuple[Image.Image, Image.Image]:
    gh, gw = idx_grid.shape
    mosaic = Image.new("RGB", (gw * out_tile_size, gh * out_tile_size), (255, 255, 255))
    overlay = Image.new("RGB", (gw * out_tile_size, gh * out_tile_size), (255, 255, 255))
    draw = ImageDraw.Draw(overlay)
    for r in range(gh):
        for c in range(gw):
            if coord_grid is not None:
                x = int(coord_grid[r, c, 0])
                y = int(coord_grid[r, c, 1])
            else:
                idx = int(idx_grid[r, c])
                x = int(coords[idx, 0])
                y = int(coords[idx, 1])
            tile = _crop_tile_rgb(slide, x, y, tile_size_20x=tile_size_20x, out_tile_size=out_tile_size)
            x0 = c * out_tile_size
            y0 = r * out_tile_size
            mosaic.paste(tile, (x0, y0))
            overlay.paste(tile, (x0, y0))
            if float(high_mask[r, c]) > 0.5:
                draw.rectangle([x0 + 2, y0 + 2, x0 + out_tile_size - 3, y0 + out_tile_size - 3], outline=(220, 20, 20), width=3)
            else:
                draw.rectangle([x0 + 2, y0 + 2, x0 + out_tile_size - 3, y0 + out_tile_size - 3], outline=(20, 140, 20), width=1)
    return mosaic, overlay


def _steer_area_grid(
    z_grid: np.ndarray,
    high_mask: np.ndarray,
    *,
    proto_vec: np.ndarray,
    strength: float,
    sae_model: torch.nn.Module,
    blend: float,
    device: torch.device,
    norm_match_to_orig: bool,
) -> np.ndarray:
    z_t = torch.from_numpy(np.asarray(z_grid, dtype=np.float32)).to(device=device)
    sel = np.where(high_mask.reshape(-1) > 0.5)[0].astype(np.int64)
    z_edit_t, _ = edit_uni_z_grid_with_sae(
        sae_model=sae_model,
        z_grid=z_t,
        latent_idx=None,
        target_latent_vector=proto_vec,
        target_latent_vector_strength=float(strength),
        tile_indices=sel,
        blend=float(blend),
        latent_strength=1.0,
        keep_non_selected=True,
        return_debug=True,
    )
    z_edit = z_edit_t.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if norm_match_to_orig and sel.size > 0:
        f0 = z_grid.reshape(-1, z_grid.shape[-1])
        f1 = z_edit.reshape(-1, z_edit.shape[-1])
        for i in sel.tolist():
            n0 = float(np.linalg.norm(f0[i]))
            n1 = float(np.linalg.norm(f1[i]))
            if n0 > 0.0 and n1 > 1e-8:
                f1[i] *= (n0 / n1)
        z_edit = f1.reshape(z_edit.shape).astype(np.float32, copy=False)
    return z_edit


@torch.no_grad()
def _sample_multidiffusion_with_spatial_midref(
    *,
    pipeline,
    z_grid: torch.Tensor,              # [Gh,Gw,D] or [1,Gh,Gw,D]
    original_image: torch.Tensor,      # [1,3,H,W] in [0,1]
    anchor_mask_grid: np.ndarray,      # [Gh,Gw] in [0,1], 1 = anchor to reference
    out_h: int,
    out_w: int,
    patch_px: int = 256,
    stride_px: int = 128,
    steps: int = 30,
    guidance: float = 2.0,
    patch_batch: int = 256,
    reference_start_ratio: float = 0.6,
    reference_mix: float = 0.6,
    mask_blur_cells: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    device = pipeline.device
    dtype = next(pipeline.transformer.parameters()).dtype

    if z_grid.dim() == 3:
        z_grid = z_grid.unsqueeze(0)
    if z_grid.shape[0] != 1:
        raise ValueError("B must be 1.")
    _, Gh, Gw, _ = z_grid.shape
    if anchor_mask_grid.shape != (Gh, Gw):
        raise ValueError(f"anchor_mask_grid shape must be {(Gh, Gw)}, got {anchor_mask_grid.shape}")

    vae_scale = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    H_lat = math.ceil(out_h / vae_scale)
    W_lat = math.ceil(out_w / vae_scale)
    ph = patch_px // vae_scale
    pw = patch_px // vae_scale
    sh = stride_px // vae_scale
    sw = stride_px // vae_scale
    C = pipeline.vae.config.latent_channels
    if min(ph, pw, sh, sw) <= 0:
        raise ValueError("patch/stride too small for VAE scale.")

    def lookup_cond(center_y_lat: float, center_x_lat: float) -> torch.Tensor:
        denom_y = max(H_lat - 1, 1)
        denom_x = max(W_lat - 1, 1)
        y = (center_y_lat / denom_y) * (Gh - 1) if Gh > 1 else 0.0
        x = (center_x_lat / denom_x) * (Gw - 1) if Gw > 1 else 0.0
        y = float(max(0.0, min(y, Gh - 1)))
        x = float(max(0.0, min(x, Gw - 1)))
        y0 = int(math.floor(y))
        x0 = int(math.floor(x))
        y1 = min(y0 + 1, Gh - 1)
        x1 = min(x0 + 1, Gw - 1)
        wy = y - y0
        wx = x - x0
        z00 = z_grid[:, y0, x0, :]
        z01 = z_grid[:, y0, x1, :]
        z10 = z_grid[:, y1, x0, :]
        z11 = z_grid[:, y1, x1, :]
        z0 = z00 * (1.0 - wx) + z01 * wx
        z1 = z10 * (1.0 - wx) + z11 * wx
        z = z0 * (1.0 - wy) + z1 * wy
        return z.unsqueeze(1)

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
    gauss = gauss[None, None, :, :]

    m = np.clip(np.asarray(anchor_mask_grid, dtype=np.float32), 0.0, 1.0)
    m_t = torch.from_numpy(m).to(device=device, dtype=torch.float32).view(1, 1, Gh, Gw)
    if mask_blur_cells > 0.0:
        radius = max(1, int(round(3.0 * mask_blur_cells)))
        k = 2 * radius + 1
        yy2, xx2 = np.mgrid[-radius: radius + 1, -radius: radius + 1].astype(np.float32)
        ker = np.exp(-0.5 * (xx2 * xx2 + yy2 * yy2) / float(mask_blur_cells * mask_blur_cells))
        ker = ker / np.maximum(ker.sum(), 1e-8)
        ker_t = torch.from_numpy(ker).to(device=device, dtype=torch.float32).view(1, 1, k, k)
        m_t = F.conv2d(m_t, ker_t, padding=radius)
        m_t = m_t.clamp(0.0, 1.0)
    mask_lat = F.interpolate(m_t, size=(H_lat, W_lat), mode="bilinear", align_corners=False).to(device=device, dtype=dtype)

    pipeline.scheduler.set_timesteps(steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    has_scale = hasattr(pipeline.scheduler, "scale_model_input")
    coords = [
        (top, left)
        for top in range(0, H_lat - ph + 1, sh)
        for left in range(0, W_lat - pw + 1, sw)
    ]
    patch_batch = max(1, int(patch_batch))
    if len(coords) == 0:
        raise RuntimeError("No windows generated.")

    if generator is None:
        init_noise = torch.randn((1, C, H_lat, W_lat), device=device, dtype=dtype)
    else:
        init_noise = torch.randn((1, C, H_lat, W_lat), device=device, dtype=dtype, generator=generator)
    latents = init_noise.clone()

    ref_img = original_image.to(device=device, dtype=torch.float32)
    if ref_img.ndim != 4 or ref_img.shape[0] != 1 or ref_img.shape[1] != 3:
        raise ValueError("original_image must be [1,3,H,W]")
    if ref_img.shape[-2:] != (out_h, out_w):
        ref_img = F.interpolate(ref_img, size=(out_h, out_w), mode="bilinear", align_corners=False)
    ref_img_vae = (ref_img * 2.0 - 1.0).to(dtype=dtype)
    ref_dist = pipeline.vae.encode(ref_img_vae).latent_dist
    ref_latents_clean = ref_dist.sample()
    if hasattr(pipeline.vae.config, "scaling_factor"):
        ref_latents_clean = ref_latents_clean * pipeline.vae.config.scaling_factor
    ref_latents_clean = ref_latents_clean.to(device=device, dtype=dtype)

    n_steps = len(timesteps)
    switch_idx = int(round(float(reference_start_ratio) * max(n_steps - 1, 0)))
    switch_idx = max(0, min(switch_idx, n_steps - 1))

    for step_idx, t in enumerate(timesteps):
        if step_idx >= switch_idx and reference_mix > 0.0:
            t_for_noise = timesteps[step_idx:step_idx + 1]
            ref_xt = pipeline.scheduler.add_noise(ref_latents_clean, init_noise, t_for_noise)
            alpha = (float(reference_mix) * mask_lat).clamp(0.0, 1.0)
            latents_step = latents * (1.0 - alpha) + ref_xt * alpha
        else:
            latents_step = latents

        latents_in = pipeline.scheduler.scale_model_input(latents_step, t) if has_scale else latents_step
        eps_accum = torch.zeros_like(latents_step)
        weight = torch.zeros((1, 1, H_lat, W_lat), device=device, dtype=dtype)

        for i0 in range(0, len(coords), patch_batch):
            chunk = coords[i0:i0 + patch_batch]
            n = len(chunk)
            patch_in = torch.stack(
                [latents_in[:, :, top:top + ph, left:left + pw].squeeze(0) for (top, left) in chunk],
                dim=0,
            )
            cond = torch.cat(
                [lookup_cond(top + ph / 2.0, left + pw / 2.0) for (top, left) in chunk],
                dim=0,
            ).to(device=device, dtype=dtype)
            uncond = pipeline.get_unconditional_embedding(n).to(device=device, dtype=dtype)
            hs = torch.cat([patch_in, patch_in], dim=0)
            es = torch.cat([uncond, cond], dim=0)
            tt = (t if torch.is_tensor(t) else torch.tensor(t, device=device)).long()
            if tt.dim() == 0:
                tt = tt[None]
            tt = tt.expand(hs.shape[0])
            out = pipeline.transformer(hidden_states=hs, encoder_hidden_states=es, timestep=tt, return_dict=True)
            eps2 = out.sample if hasattr(out, "sample") else out[0]
            if eps2.shape[1] == 2 * C:
                eps2 = eps2[:, :C]
            eps_u, eps_c = eps2[:n], eps2[n:]
            eps = eps_u + float(guidance) * (eps_c - eps_u)
            for j, (top, left) in enumerate(chunk):
                eps_accum[:, :, top:top + ph, left:left + pw] += eps[j:j + 1] * gauss
                weight[:, :, top:top + ph, left:left + pw] += gauss

        eps_full = eps_accum / weight.clamp(min=1e-8)
        step_out = pipeline.scheduler.step(eps_full, t, latents, return_dict=True)
        latents = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

    if hasattr(pipeline.vae.config, "scaling_factor"):
        latents = latents / pipeline.vae.config.scaling_factor
    vae_param = next(pipeline.vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    img = pipeline.vae.decode(latents, return_dict=True).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    return img


def _to_pil_uint8(img_t: torch.Tensor) -> Image.Image:
    arr = (img_t[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _default_settings() -> list[dict[str, Any]]:
    return [
        {
            "name": "naive",
            "mode": "naive",
            "steps": 30,
            "guidance": 2.0,
        },
        {
            "name": "spatial_bg_soft",
            "mode": "spatial_bg_anchor",
            "steps": 30,
            "guidance": 2.0,
            "reference_start_ratio": 0.70,
            "reference_mix": 0.40,
            "mask_blur_cells": 0.60,
        },
        {
            "name": "spatial_bg_mid",
            "mode": "spatial_bg_anchor",
            "steps": 30,
            "guidance": 2.0,
            "reference_start_ratio": 0.60,
            "reference_mix": 0.60,
            "mask_blur_cells": 0.80,
        },
        {
            "name": "spatial_bg_late_strong",
            "mode": "spatial_bg_anchor",
            "steps": 35,
            "guidance": 1.8,
            "reference_start_ratio": 0.82,
            "reference_mix": 0.78,
            "mask_blur_cells": 1.00,
        },
        {
            "name": "global_anchor",
            "mode": "global_anchor",
            "steps": 30,
            "guidance": 2.0,
            "reference_start_ratio": 0.60,
            "reference_mix": 0.60,
        },
    ]


def _load_settings(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return _default_settings()
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("settings JSON must be a non-empty list.")
    out: list[dict[str, Any]] = []
    for i, s in enumerate(payload):
        if not isinstance(s, dict):
            raise ValueError(f"settings[{i}] must be object.")
        name = str(s.get("name", f"setting_{i}"))
        mode = str(s.get("mode", "naive"))
        if mode not in {"naive", "global_anchor", "spatial_bg_anchor"}:
            raise ValueError(f"Unsupported mode {mode} in settings[{i}]")
        out.append(dict(s, name=name, mode=mode))
    return out


def _compute_mse_high_low(img: Image.Image, ref: Image.Image, high_mask: np.ndarray, tile_px: int) -> tuple[float, float, float]:
    a = np.asarray(img).astype(np.float32)
    b = np.asarray(ref).astype(np.float32)
    d = ((a - b) ** 2).mean(axis=2)
    high = np.kron(high_mask.astype(np.float32), np.ones((tile_px, tile_px), dtype=np.float32))
    high = high[: d.shape[0], : d.shape[1]]
    low = 1.0 - high
    high_mse = float(d[high > 0.5].mean()) if float((high > 0.5).sum()) > 0 else float("nan")
    low_mse = float(d[low > 0.5].mean()) if float((low > 0.5).sum()) > 0 else float("nan")
    ratio = float(high_mse / (low_mse + 1e-8)) if not (math.isnan(high_mse) or math.isnan(low_mse)) else float("nan")
    return high_mse, low_mse, ratio


def _render_compare_sheet(
    *,
    ref_img: Image.Image,
    rows: list[dict[str, Any]],
    settings: list[dict[str, Any]],
    strengths: list[float],
    out_path: Path,
    title: str,
) -> None:
    # rows are filtered to one slide + one direction.
    thumb = 256
    pad = 10
    header_h = 56
    caption_h = 42
    cols = 1 + len(settings)
    rows_n = len(strengths)
    w = pad + cols * (thumb + pad)
    h = header_h + pad + rows_n * (caption_h + thumb + pad)
    canvas = Image.new("RGB", (w, h), (242, 242, 242))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, header_h], fill=(225, 225, 225))
    draw.text((8, 8), title, fill=(0, 0, 0))
    draw.text((8, 30), "Columns: [reference] + settings", fill=(0, 0, 0))

    row_map: dict[tuple[float, str], dict[str, Any]] = {}
    for r in rows:
        row_map[(float(r["strength"]), str(r["setting_name"]))] = r

    ref_thumb = ref_img.resize((thumb, thumb), resample=Image.BILINEAR)
    for i, s in enumerate(strengths):
        y0 = header_h + pad + i * (caption_h + thumb + pad)
        x_ref = pad
        draw.rectangle([x_ref, y0, x_ref + thumb, y0 + caption_h], fill=(255, 255, 255))
        draw.text((x_ref + 4, y0 + 4), f"strength={s:.2f}\nreference", fill=(0, 0, 0))
        canvas.paste(ref_thumb, (x_ref, y0 + caption_h))
        for j, st in enumerate(settings, start=1):
            x0 = pad + j * (thumb + pad)
            key = (float(s), str(st["name"]))
            rr = row_map.get(key)
            draw.rectangle([x0, y0, x0 + thumb, y0 + caption_h], fill=(255, 255, 255))
            if rr is None:
                draw.text((x0 + 4, y0 + 4), f"{st['name']}\n(missing)", fill=(0, 0, 0))
                continue
            img = Image.open(str(rr["generated_image_path"])).convert("RGB").resize((thumb, thumb), resample=Image.BILINEAR)
            txt = (
                f"{st['name']}\n"
                f"bag p={float(rr['bag_prob_cf']):.3f} d={float(rr['bag_delta_prob_cf_minus_orig']):+.3f}"
            )
            draw.text((x0 + 4, y0 + 4), txt, fill=(0, 0, 0))
            canvas.paste(img, (x0, y0 + caption_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _plot_prob_curves(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    if len(rows) == 0:
        return
    by_dir: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_dir.setdefault(str(r["direction"]), []).append(r)
    dirs = sorted(by_dir.keys())
    fig, axes = plt.subplots(1, len(dirs), figsize=(6 * len(dirs), 4), constrained_layout=True)
    if len(dirs) == 1:
        axes = [axes]
    for ax, d in zip(axes, dirs):
        rr = by_dir[d]
        base = float(rr[0]["bag_prob_orig"])
        settings = sorted(set(str(x["setting_name"]) for x in rr))
        for sn in settings:
            r2 = [x for x in rr if str(x["setting_name"]) == sn]
            r2 = sorted(r2, key=lambda z: float(z["strength"]))
            xs = [float(x["strength"]) for x in r2]
            ys = [float(x["bag_prob_cf"]) for x in r2]
            ax.plot(xs, ys, marker="o", label=sn)
        ax.axhline(base, color="k", linestyle="--", linewidth=1, label="orig")
        ax.set_title(d)
        ax.set_xlabel("strength")
        ax.set_ylabel("bag prob HPV+")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_naturalness(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    if len(rows) == 0:
        return
    by_dir: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_dir.setdefault(str(r["direction"]), []).append(r)
    dirs = sorted(by_dir.keys())
    fig, axes = plt.subplots(2, len(dirs), figsize=(6 * len(dirs), 8), constrained_layout=True)
    if len(dirs) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for ci, d in enumerate(dirs):
        rr = by_dir[d]
        settings = sorted(set(str(x["setting_name"]) for x in rr))
        for sn in settings:
            r2 = [x for x in rr if str(x["setting_name"]) == sn]
            r2 = sorted(r2, key=lambda z: float(z["strength"]))
            xs = [float(x["strength"]) for x in r2]
            high = [float(x["high_mse"]) for x in r2]
            low = [float(x["low_mse"]) for x in r2]
            ratio = [float(x["high_low_mse_ratio"]) for x in r2]
            axes[0, ci].plot(xs, high, marker="o", label=f"{sn} high")
            axes[0, ci].plot(xs, low, marker="x", linestyle="--", label=f"{sn} low")
            axes[1, ci].plot(xs, ratio, marker="o", label=sn)
        axes[0, ci].set_title(f"{d} | MSE")
        axes[0, ci].set_xlabel("strength")
        axes[0, ci].set_ylabel("MSE to reference")
        axes[0, ci].grid(alpha=0.3)
        axes[0, ci].legend(fontsize=7)
        axes[1, ci].set_title(f"{d} | high/low ratio")
        axes[1, ci].set_xlabel("strength")
        axes[1, ci].set_ylabel("ratio")
        axes[1, ci].grid(alpha=0.3)
        axes[1, ci].legend(fontsize=7)
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    strengths = _parse_float_csv(args.strengths)
    directions = _parse_str_csv(args.directions)
    for d in directions:
        if d not in {"to_hpv_pos", "to_hpv_neg"}:
            raise ValueError(f"Unsupported direction: {d}")

    settings = _load_settings(args.settings_json)
    if len(settings) == 0:
        raise RuntimeError("No settings loaded.")
    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This script expects CUDA for PixCell generation.")

    print(f"[setup] device={device}")
    rows = _load_test_rows(args.split_json, args.split_tsv, args.features_root)
    print(f"[setup] test rows={len(rows)}")
    if len(rows) == 0:
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
    if int(args.max_slides) > 0:
        selected = selected[: int(args.max_slides)]
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

    slide_path_cache: dict[str, Path | None] = {}
    selection_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failed_slides: list[dict[str, Any]] = []

    for si, row in enumerate(selected, start=1):
        slide_key = str(row["slide_key"])
        slide_label = int(row["label"])
        h5_path = str(row["h5_path"])
        case_id = str(row["case_id"])
        print(f"[slide {si}/{len(selected)}] {slide_key} label={slide_label}")

        try:
            x, coords = _read_h5_features_coords(h5_path)
            if coords is None:
                raise RuntimeError("coords missing in H5.")
            if x.ndim != 2 or x.shape[1] != d_in:
                raise RuntimeError(f"Feature shape {x.shape} incompatible with SAE d_in={d_in}")

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
                ref_area_pil, mask_overlay_pil = _tile_mosaic_from_idx_grid(
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
            overlay_path = slide_dir / "area_high_low_overlay.png"
            ref_area_pil.save(ref_path)
            mask_overlay_pil.save(overlay_path)

            real_indices = np.asarray(area["real_indices"], dtype=np.int64)
            x_area_orig = np.asarray(x[real_indices], dtype=np.float32)
            _, bag_pred_orig, bag_prob_orig = run_mil_attention(mil_model, x_area_orig, device=device)

            selection_rows.append(
                {
                    "slide_key": slide_key,
                    "case_id": case_id,
                    "label": int(slide_label),
                    "h5_path": h5_path,
                    "slide_path": str(slide_path),
                    "full_pred_orig": int(full_pred_orig),
                    "full_prob_orig": float(full_prob_orig),
                    "bag_pred_orig": int(bag_pred_orig),
                    "bag_prob_orig": float(bag_prob_orig),
                    "area_side": int(args.area_side),
                    "high_attn_fraction_target": float(args.high_attn_fraction),
                    "n_high_selected": int(area["n_high_selected"]),
                    "n_high_target": int(area["n_high_target"]),
                    "n_real_tiles_in_area": int(real_indices.size),
                    "coord_step": int(area["step"]),
                    "anchor_idx": int(area["anchor_idx"]),
                    "anchor_coord_x": int(area["anchor_coord"][0]),
                    "anchor_coord_y": int(area["anchor_coord"][1]),
                    "reference_image_path": str(ref_path),
                    "overlay_image_path": str(overlay_path),
                }
            )

            high_flat = area["high_mask"].reshape(-1) > 0.5
            idx_flat = area["idx_grid"].reshape(-1)
            real_flat = area["real_mask"].reshape(-1) > 0.5
            high_real_tile_indices = sorted(set(int(idx_flat[i]) for i in np.where(high_flat & real_flat)[0].tolist()))

            ref_np = np.asarray(ref_area_pil).astype(np.float32) / 255.0
            ref_t = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            out_h, out_w = ref_area_pil.height, ref_area_pil.width

            for di, direction in enumerate(directions):
                proto_vec = proto_pos if direction == "to_hpv_pos" else proto_neg
                dir_dir = slide_dir / direction
                dir_dir.mkdir(parents=True, exist_ok=True)

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
                        cell_idx = np.where(idx_flat == int(k))[0]
                        if cell_idx.size == 0:
                            continue
                        cell = int(cell_idx[0])
                        x_cf_full[int(k)] = z_flat[cell]
                    x_area_cf = np.asarray(x_cf_full[real_indices], dtype=np.float32)
                    _, bag_pred_cf, bag_prob_cf = run_mil_attention(mil_model, x_area_cf, device=device)

                    z_cf_t = torch.from_numpy(z_cf).to(device=device, dtype=pipe_dtype)

                    for ti, st in enumerate(settings):
                        setting_name = str(st["name"])
                        mode = str(st.get("mode", "naive"))
                        steps = int(st.get("steps", args.steps))
                        guidance = float(st.get("guidance", args.guidance))
                        ref_start = float(st.get("reference_start_ratio", 0.6))
                        ref_mix = float(st.get("reference_mix", 0.6))
                        mask_blur = float(st.get("mask_blur_cells", 0.0))

                        seed = int(args.seed + si * 100000 + di * 10000 + sj * 1000 + ti)
                        g = torch.Generator(device=device).manual_seed(seed)
                        ac = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                        with torch.inference_mode(), ac:
                            if mode == "naive":
                                img_t = sample_multidiffusion_from_zgrid(
                                    pipeline=pipe,
                                    z_grid=z_cf_t,
                                    out_h=out_h,
                                    out_w=out_w,
                                    patch_px=int(args.patch_px),
                                    stride_px=int(args.stride_px),
                                    steps=steps,
                                    guidance=guidance,
                                    patch_batch=int(args.patch_batch),
                                    generator=g,
                                )
                            elif mode == "global_anchor":
                                img_t = sample_multidiffusion_from_zgrid_with_midref(
                                    pipeline=pipe,
                                    z_grid=z_cf_t,
                                    original_image=ref_t,
                                    out_h=out_h,
                                    out_w=out_w,
                                    patch_px=int(args.patch_px),
                                    stride_px=int(args.stride_px),
                                    steps=steps,
                                    guidance=guidance,
                                    patch_batch=int(args.patch_batch),
                                    reference_start_ratio=ref_start,
                                    reference_mix=ref_mix,
                                    generator=g,
                                )
                            elif mode == "spatial_bg_anchor":
                                img_t = _sample_multidiffusion_with_spatial_midref(
                                    pipeline=pipe,
                                    z_grid=z_cf_t,
                                    original_image=ref_t,
                                    anchor_mask_grid=np.asarray(area["low_mask"], dtype=np.float32),
                                    out_h=out_h,
                                    out_w=out_w,
                                    patch_px=int(args.patch_px),
                                    stride_px=int(args.stride_px),
                                    steps=steps,
                                    guidance=guidance,
                                    patch_batch=int(args.patch_batch),
                                    reference_start_ratio=ref_start,
                                    reference_mix=ref_mix,
                                    mask_blur_cells=mask_blur,
                                    generator=g,
                                )
                            else:
                                raise RuntimeError(f"Unsupported mode: {mode}")

                        img_pil = _to_pil_uint8(img_t)
                        s_tag = f"{s:.2f}".replace(".", "p")
                        p_tag = f"{bag_prob_cf:.4f}".replace(".", "p")
                        out_name = f"{setting_name}__s_{s_tag}__bagpred_{bag_pred_cf}__p_{p_tag}.png"
                        out_img = dir_dir / out_name
                        img_pil.save(out_img)

                        high_mse, low_mse, ratio = _compute_mse_high_low(
                            img_pil,
                            ref_area_pil,
                            np.asarray(area["high_mask"], dtype=np.float32),
                            tile_px=int(args.out_tile_size),
                        )

                        records.append(
                            {
                                "slide_key": slide_key,
                                "case_id": case_id,
                                "label": int(slide_label),
                                "direction": direction,
                                "strength": float(s),
                                "setting_name": setting_name,
                                "setting_mode": mode,
                                "steps": int(steps),
                                "guidance": float(guidance),
                                "reference_start_ratio": float(ref_start),
                                "reference_mix": float(ref_mix),
                                "mask_blur_cells": float(mask_blur),
                                "full_pred_orig": int(full_pred_orig),
                                "full_prob_orig": float(full_prob_orig),
                                "bag_pred_orig": int(bag_pred_orig),
                                "bag_prob_orig": float(bag_prob_orig),
                                "bag_pred_cf": int(bag_pred_cf),
                                "bag_prob_cf": float(bag_prob_cf),
                                "bag_delta_prob_cf_minus_orig": float(bag_prob_cf - bag_prob_orig),
                                "n_real_tiles_in_area": int(real_indices.size),
                                "n_high_selected": int(area["n_high_selected"]),
                                "high_fraction_target": float(args.high_attn_fraction),
                                "high_mse": float(high_mse),
                                "low_mse": float(low_mse),
                                "high_low_mse_ratio": float(ratio),
                                "generated_image_path": str(out_img),
                                "reference_image_path": str(ref_path),
                                "overlay_image_path": str(overlay_path),
                            }
                        )

                slide_rows = [r for r in records if str(r["slide_key"]) == slide_key]
                _plot_prob_curves(
                    slide_rows,
                    out_path=slide_dir / "plots" / "bag_prob_vs_strength.png",
                    title=f"{slide_key} | bag prob vs strength",
                )
                _plot_naturalness(
                    slide_rows,
                    out_path=slide_dir / "plots" / "naturalness_vs_strength.png",
                    title=f"{slide_key} | naturalness comparison",
                )
                for d in directions:
                    d_rows = [r for r in slide_rows if str(r["direction"]) == d]
                    _render_compare_sheet(
                        ref_img=ref_area_pil,
                        rows=d_rows,
                        settings=settings,
                        strengths=strengths,
                        out_path=slide_dir / "plots" / f"compare_sheet_{d}.png",
                        title=f"{slide_key} | label={slide_label} | direction={d}",
                    )

        except Exception as e:
            failed_slides.append({"slide_key": slide_key, "error": str(e)})
            print(f"[warn] failed {slide_key}: {e}")

    rec_csv = args.out_dir / "area_counterfactual_records.csv"
    sel_csv = args.out_dir / "selected_slides_and_areas.csv"
    _write_csv(rec_csv, records)
    _write_csv(sel_csv, selection_rows)

    summary = {
        "n_test_rows_total": int(len(rows)),
        "n_selected": int(len(selected)),
        "n_failed_slides": int(len(failed_slides)),
        "n_success_slides": int(len(selected) - len(failed_slides)),
        "n_records": int(len(records)),
        "strengths": [float(s) for s in strengths],
        "directions": directions,
        "settings": settings,
        "prototype_key": str(args.prototype_key),
        "prototype_latent_pos": int(pos_latent),
        "prototype_latent_neg": int(neg_latent),
        "area_side": int(args.area_side),
        "high_attn_fraction": float(args.high_attn_fraction),
        "records_csv": str(rec_csv),
        "selected_csv": str(sel_csv),
        "failed_slides": failed_slides,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("[done] wrote:")
    print(f"  {rec_csv}")
    print(f"  {sel_csv}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
