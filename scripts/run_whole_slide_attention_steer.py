#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DiffusionPipeline
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png, write_json
from wsi_cf.common.paths import (
    DEFAULT_HNSCC_PROTOTYPE_NPZ,
    DEFAULT_HNSCC_SPLIT_TSV,
    DEFAULT_SAE_CFG,
    DEFAULT_SAE_CKPT,
    ensure_legacy_repo_root_on_path,
)
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.slides import find_slide_path, open_slide, read_region_rgb_at_magnification

ensure_legacy_repo_root_on_path()

from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, load_prototypes, pick_prototype_latent, run_mil_attention
from wsi_cf.generation.pixcell import (
    load_uni2,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
    vae_encode_auto,
)
from wsi_cf.steering.progressive import (
    CENTER_2X2_LOCAL_CELLS,
    ProgressiveWindow,
    build_history_aware_preserve_map,
    center_support_global_cells,
    preserve_map_preview,
    window_local_cells,
)

CLAM_ROOT = Path("/common/users/wq50/CLAM")
if str(CLAM_ROOT) not in sys.path:
    sys.path.append(str(CLAM_ROOT))

from utils.sae import load_sae_from_config  # type: ignore
from utils.sae_edit import edit_uni_z_grid_with_sae  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-slide whole-slide attention steering pilot. Select high-attention MIL tiles from a full-slide "
            "H5 bag, generate only local PixCell-1024 windows covering those tiles, re-encode only edited "
            "256-cell crops with UNI2, replace only those rows in the original bag, and rerun MIL."
        )
    )
    parser.add_argument("--slide-key", type=str, default="TCGA-BB-4225-01Z-00-DX1")
    parser.add_argument("--model-backend", type=str, default="mil", choices=["mil", "clam"])
    parser.add_argument("--split-tsv", type=Path, default=DEFAULT_HNSCC_SPLIT_TSV)
    parser.add_argument("--features-root", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"))
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/test"))
    parser.add_argument("--out-dir", type=Path, default=Path("/common/users/wq50/wsi_cf/artifacts/whole_slide_attention_steer_pilot"))
    parser.add_argument("--mil-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt"))
    parser.add_argument("--clam-ckpt", type=Path, default=Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/s_0_checkpoint.pt"))
    parser.add_argument("--clam-features-pt-dir", type=Path, default=Path("/common/users/wq50/CLAM/features/HPV_UNI2_features/pt_files"))
    parser.add_argument("--clam-coords-h5-dir", type=Path, default=Path("/common/users/wq50/CLAM/HNSCC_cases/patches"))
    parser.add_argument("--clam-dataset-csv", type=Path, default=Path("/common/users/wq50/CLAM/dataset_csv/HNSCC.csv"))
    parser.add_argument("--clam-splits-csv", type=Path, default=Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/splits_0.csv"))
    parser.add_argument("--clam-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--clam-attn-class", type=str, default="pred", choices=["pred", "pos", "neg"])
    parser.add_argument("--attention-percentile", type=float, default=99.5)
    parser.add_argument("--max-edit-tiles", type=int, default=8, help="0 means use every tile above threshold.")
    parser.add_argument("--max-windows", type=int, default=0, help="0 means no limit after planning.")
    parser.add_argument("--direction-mode", type=str, default="opposite_label", choices=["opposite_label", "hpv_pos", "hpv_neg"])
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument(
        "--require-full-window-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require every 4x4 PixCell context cell to exist in the H5 bag. "
            "Default false: missing context cells are zero-conditioned, marked empty in metadata, "
            "and are never re-encoded/replaced."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--pix-model-id", type=str, default="StonyBrook-CVLab/PixCell-1024")
    parser.add_argument("--pix-pipeline-id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae-model-id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae-subfolder", type=str, default="vae")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--preserve-edit-strength", type=float, default=0.05)
    parser.add_argument("--preserve-visited-strength", type=float, default=0.95)
    parser.add_argument("--preserve-fresh-context-strength", type=float, default=0.35)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.5)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=0.5)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--sae-ckpt", type=Path, default=DEFAULT_SAE_CKPT)
    parser.add_argument("--sae-cfg", type=Path, default=DEFAULT_SAE_CFG)
    parser.add_argument(
        "--prototype-npz",
        type=Path,
        default=DEFAULT_HNSCC_PROTOTYPE_NPZ,
    )
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--save-debug-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-vis-size", type=int, default=2048, help="Local before/after context size around edited areas. Use 0 to disable.")
    parser.add_argument("--max-local-vis-areas", type=int, default=0, help="0 means save one local area per executed window.")
    return parser


def resolve_custom_pipeline_ref(custom_pipeline: str) -> str:
    candidate = Path(str(custom_pipeline))
    if candidate.exists():
        return str(candidate)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{str(custom_pipeline).replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        commit = refs_main.read_text().strip()
        snapshot = repo_dir / "snapshots" / commit
        if snapshot.exists():
            return str(snapshot)
    return str(custom_pipeline)


def stable_seed(base_seed: int, key: str) -> int:
    acc = int(base_seed)
    for byte in str(key).encode("utf-8"):
        acc = (acc * 131 + int(byte)) % (2**31 - 1)
    return int(acc)


def sae_strength_from_power(value: float) -> float:
    return float(max(0.0, min(float(value), 1.0)))


def condition_alpha_end_from_power(*, base_alpha_end: float, value: float) -> float:
    if float(value) <= 1.0:
        return float(base_alpha_end)
    return float(base_alpha_end) * float(value)


def read_split_row(split_tsv: Path, slide_key: str) -> dict[str, Any]:
    with split_tsv.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("slide_key", "")) == str(slide_key):
                out = dict(row)
                out["label"] = int(out["label"])
                return out
    raise ValueError(f"Slide key not found in split TSV: {slide_key}")


def hpv_label_from_status(status: str) -> int:
    s = str(status).strip().upper()
    if s == "HPV+":
        return 1
    if s == "HPV-":
        return 0
    raise ValueError(f"Unexpected hpv_status: {status}")


def load_clam_dataset_labels(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slide_id = str(row.get("slide_id", "")).strip()
            if not slide_id:
                continue
            out[slide_id] = {
                "case_id": str(row.get("case_id", slide_id)).strip(),
                "slide_id": slide_id,
                "hpv_status": str(row.get("hpv_status", "")).strip(),
                "label": int(hpv_label_from_status(str(row.get("hpv_status", "")).strip())),
            }
    return out


def read_clam_split_members(path: Path, split_name: str) -> set[str]:
    members: set[str] = set()
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slide_id = str(row.get(split_name, "")).strip()
            if slide_id:
                members.add(slide_id)
    return members


def read_h5_features_coords(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        feats = handle["features"][:]
        coords = handle["coords"][:]
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    return np.asarray(feats, dtype=np.float32), np.asarray(coords, dtype=np.int64)


def read_pt_features(pt_path: Path) -> np.ndarray:
    feats = torch.load(pt_path, map_location="cpu")
    if not isinstance(feats, torch.Tensor):
        raise TypeError(f"Unexpected pt payload in {pt_path}: {type(feats)}")
    return feats.detach().cpu().float().numpy().astype(np.float32, copy=False)


def read_coord_h5(h5_path: Path) -> tuple[np.ndarray, int]:
    with h5py.File(h5_path, "r") as handle:
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
        patch_size = int(handle["coords"].attrs.get("patch_size", 256))
    return coords, patch_size


def infer_coord_tile_size(coords: np.ndarray, fallback: int = 512) -> int:
    arr = np.asarray(coords, dtype=np.int64)
    candidates: list[int] = []
    for axis in (0, 1):
        vals = np.unique(arr[:, axis])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    if not candidates:
        return int(fallback)
    return max(1, int(min(candidates)))


def target_direction_for_label(label: int, mode: str) -> str:
    if mode == "hpv_pos":
        return "hpv_pos"
    if mode == "hpv_neg":
        return "hpv_neg"
    return "hpv_pos" if int(label) == 0 else "hpv_neg"


def target_prob(prob_pos: float, target_label: int) -> float:
    return float(prob_pos) if int(target_label) == 1 else float(1.0 - float(prob_pos))


def load_clam_model(ckpt_path: Path, *, device: torch.device):
    from models.model_clam import CLAM_MB  # type: ignore

    model = CLAM_MB(gate=True, size_arg="small", n_classes=2, embed_dim=1536)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_clam_attention(model: Any, features: np.ndarray, *, device: torch.device, attn_class: str) -> tuple[np.ndarray, int, float]:
    feats_t = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device=device, dtype=torch.float32)
    logits, y_prob, y_hat, a_raw, _ = model(feats_t)
    a = F.softmax(a_raw, dim=1)
    pred = int(y_hat.item())
    prob_pos = float(y_prob[0, 1].item())
    if str(attn_class) == "pred":
        row = pred
    elif str(attn_class) == "pos":
        row = 1
    else:
        row = 0
    attn = a[row].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    return attn, pred, prob_pos


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def draw_cells_overlay(img: Image.Image, cells: Sequence[tuple[int, int]], *, grid_step_px: int) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for idx, (gx, gy) in enumerate(cells, start=1):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = x0 + int(grid_step_px) - 1
        y1 = y0 + int(grid_step_px) - 1
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=5)
        draw.text((x0 + 8, y0 + 8), str(idx), fill=(255, 0, 0))
    return out


def draw_window_context_overlay(
    img: Image.Image,
    *,
    edit_cells: Sequence[tuple[int, int]],
    empty_cells: Sequence[tuple[int, int]],
    grid_step_px: int,
) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    for lx, ly in empty_cells:
        x0 = int(lx) * int(grid_step_px)
        y0 = int(ly) * int(grid_step_px)
        x1 = x0 + int(grid_step_px) - 1
        y1 = y0 + int(grid_step_px) - 1
        draw.rectangle([x0, y0, x1, y1], fill=(80, 80, 80, 95), outline=(40, 40, 40, 255), width=4)
        draw.text((x0 + 8, y0 + 8), "empty", fill=(255, 255, 255, 255))
    for idx, (lx, ly) in enumerate(edit_cells, start=1):
        x0 = int(lx) * int(grid_step_px)
        y0 = int(ly) * int(grid_step_px)
        x1 = x0 + int(grid_step_px) - 1
        y1 = y0 + int(grid_step_px) - 1
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0, 255), width=6)
        draw.text((x0 + 8, y0 + 28), f"edit {idx}", fill=(255, 0, 0, 255))
    return out


def encode_cells(cells: Sequence[tuple[int, int]]) -> str:
    return ";".join(f"{int(gx)},{int(gy)}" for gx, gy in cells)


def make_contact_sheet(images: Sequence[tuple[str, Image.Image]], *, thumb_w: int = 512) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), "white")
    font_h = 28
    thumbs: list[tuple[str, Image.Image]] = []
    for label, img in images:
        rgb = img.convert("RGB")
        scale = float(thumb_w) / float(max(1, rgb.width))
        thumb_h = max(1, int(round(float(rgb.height) * scale)))
        thumb = rgb.resize((int(thumb_w), int(thumb_h)), resample=Image.BILINEAR)
        thumbs.append((label, thumb))
    w = int(thumb_w) * len(thumbs)
    h = max(thumb.height for _, thumb in thumbs) + font_h
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, thumb) in enumerate(thumbs):
        x = int(i) * int(thumb_w)
        draw.text((x + 8, 6), str(label), fill=(0, 0, 0))
        sheet.paste(thumb, (x, font_h))
    return sheet


def clamp_area_origin_cell(*, center_gx: int, center_gy: int, area_cells: int, grid_w: int, grid_h: int) -> tuple[int, int]:
    half = int(area_cells) // 2
    max_gx0 = max(0, int(grid_w) - int(area_cells))
    max_gy0 = max(0, int(grid_h) - int(area_cells))
    gx0 = max(0, min(int(center_gx) - half, max_gx0))
    gy0 = max(0, min(int(center_gy) - half, max_gy0))
    return int(gx0), int(gy0)


def save_local_edit_area_visuals(
    *,
    out_dir: Path,
    slide: Any,
    planned_steps: Sequence[tuple[ProgressiveWindow, list[tuple[int, int]]]],
    edited_patch_by_cell: dict[tuple[int, int], Image.Image],
    attention_by_cell: dict[tuple[int, int], float],
    tile_size_level0: int,
    grid_w: int,
    grid_h: int,
    target_magnification: float,
    local_vis_size: int,
    grid_step_px: int,
    max_local_vis_areas: int,
) -> list[dict[str, Any]]:
    if int(local_vis_size) <= 0 or not edited_patch_by_cell:
        return []
    area_cells = max(1, int(round(float(local_vis_size) / float(grid_step_px))))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    local_dir = out_dir / f"local_edit_areas_{int(local_vis_size)}"
    local_dir.mkdir(parents=True, exist_ok=True)
    for area_idx, (window, edit_cells) in enumerate(planned_steps, start=1):
        if int(max_local_vis_areas) > 0 and len(rows) >= int(max_local_vis_areas):
            break
        if edit_cells:
            center_gx = int(round(sum(int(gx) for gx, _ in edit_cells) / float(len(edit_cells))))
            center_gy = int(round(sum(int(gy) for _, gy in edit_cells) / float(len(edit_cells))))
        else:
            center_gx = int(window.gx0) + 2
            center_gy = int(window.gy0) + 2
        area_gx0, area_gy0 = clamp_area_origin_cell(
            center_gx=center_gx,
            center_gy=center_gy,
            area_cells=int(area_cells),
            grid_w=int(grid_w),
            grid_h=int(grid_h),
        )
        area_key = (int(area_gx0), int(area_gy0))
        if area_key in seen:
            continue
        seen.add(area_key)
        level0_x = int(area_gx0) * int(tile_size_level0)
        level0_y = int(area_gy0) * int(tile_size_level0)
        before_img, crop_w0, crop_h0 = read_region_rgb_at_magnification(
            slide,
            x0=int(level0_x),
            y0=int(level0_y),
            out_w=int(local_vis_size),
            out_h=int(local_vis_size),
            target_magnification=float(target_magnification),
        )
        after_img = before_img.convert("RGB").copy()
        local_cells: list[tuple[int, int]] = []
        global_cells: list[tuple[int, int]] = []
        for global_cell, patch in sorted(edited_patch_by_cell.items(), key=lambda item: (item[0][1], item[0][0])):
            gx, gy = int(global_cell[0]), int(global_cell[1])
            if not (area_gx0 <= gx < area_gx0 + area_cells and area_gy0 <= gy < area_gy0 + area_cells):
                continue
            lx = int(gx) - int(area_gx0)
            ly = int(gy) - int(area_gy0)
            after_img.paste(patch.convert("RGB"), (lx * int(grid_step_px), ly * int(grid_step_px)))
            local_cells.append((lx, ly))
            global_cells.append((gx, gy))
        if not global_cells:
            continue
        area_dir = local_dir / f"area_{len(rows) + 1:03d}_gx{area_gx0}_gy{area_gy0}"
        area_dir.mkdir(parents=True, exist_ok=True)
        before_path = area_dir / "before_2048.png"
        after_path = area_dir / "after_selected_tiles_only.png"
        before_overlay_path = area_dir / "before_edit_overlay.png"
        after_overlay_path = area_dir / "after_edit_overlay.png"
        contact_path = area_dir / "before_after_contact_sheet.png"
        save_png(before_img, before_path)
        save_png(after_img, after_path)
        before_overlay = draw_cells_overlay(before_img, local_cells, grid_step_px=int(grid_step_px))
        after_overlay = draw_cells_overlay(after_img, local_cells, grid_step_px=int(grid_step_px))
        save_png(before_overlay, before_overlay_path)
        save_png(after_overlay, after_overlay_path)
        save_png(
            make_contact_sheet(
                [
                    ("before actual 2048", before_img),
                    ("after edited tiles only", after_img),
                    ("after highlighted", after_overlay),
                ]
            ),
            contact_path,
        )
        rows.append(
            {
                "area_index": int(len(rows) + 1),
                "area_gx0": int(area_gx0),
                "area_gy0": int(area_gy0),
                "level0_x": int(level0_x),
                "level0_y": int(level0_y),
                "crop_w_level0": int(crop_w0),
                "crop_h_level0": int(crop_h0),
                "target_magnification": float(target_magnification),
                "area_size_px": int(local_vis_size),
                "area_grid_cells": int(area_cells),
                "edited_cell_count": int(len(global_cells)),
                "edited_cells": ";".join(f"{gx},{gy}" for gx, gy in global_cells),
                "max_attention": float(max(attention_by_cell.get(cell, 0.0) for cell in global_cells)),
                "before_path": str(before_path),
                "after_path": str(after_path),
                "before_overlay_path": str(before_overlay_path),
                "after_overlay_path": str(after_overlay_path),
                "contact_sheet_path": str(contact_path),
            }
        )
    if rows:
        save_png(
            make_contact_sheet(
                [
                    (f"area {row['area_index']}", Image.open(row["contact_sheet_path"]).convert("RGB"))
                    for row in rows[: min(len(rows), 12)]
                ],
                thumb_w=768,
            ),
            local_dir / "local_area_contact_sheet.png",
        )
    return rows


def build_cell_maps(coords: np.ndarray, *, tile_size_level0: int) -> tuple[dict[tuple[int, int], int], dict[int, tuple[int, int]], int, int]:
    cell_to_index: dict[tuple[int, int], int] = {}
    index_to_cell: dict[int, tuple[int, int]] = {}
    for idx, (x, y) in enumerate(np.asarray(coords, dtype=np.int64).tolist()):
        gx = int(round(int(x) / float(tile_size_level0)))
        gy = int(round(int(y) / float(tile_size_level0)))
        cell = (gx, gy)
        cell_to_index[cell] = int(idx)
        index_to_cell[int(idx)] = cell
    grid_w = max(gx for gx, _ in cell_to_index) + 1
    grid_h = max(gy for _, gy in cell_to_index) + 1
    return cell_to_index, index_to_cell, int(grid_w), int(grid_h)


def all_window_cells(window: ProgressiveWindow) -> list[tuple[int, int]]:
    return [(int(window.gx0) + lx, int(window.gy0) + ly) for ly in range(4) for lx in range(4)]


def enumerate_candidate_windows_for_targets(
    *,
    target_cells: set[tuple[int, int]],
    grid_w: int,
    grid_h: int,
    cell_to_index: dict[tuple[int, int], int],
    require_full_window_features: bool,
    grid_step_px: int,
) -> tuple[list[ProgressiveWindow], dict[str, set[tuple[int, int]]], set[tuple[int, int]]]:
    windows: list[ProgressiveWindow] = []
    window_targets: dict[str, set[tuple[int, int]]] = {}
    coverable: set[tuple[int, int]] = set()
    for gy0 in range(0, max(1, int(grid_h) - 3)):
        if gy0 + 4 > int(grid_h):
            continue
        for gx0 in range(0, max(1, int(grid_w) - 3)):
            if gx0 + 4 > int(grid_w):
                continue
            window = ProgressiveWindow(
                window_id=f"r{gy0}_c{gx0}",
                row_index=int(gy0),
                col_index=int(gx0),
                gx0=int(gx0),
                gy0=int(gy0),
                grid_w=4,
                grid_h=4,
                left=int(gx0 * int(grid_step_px)),
                top=int(gy0 * int(grid_step_px)),
            )
            cells = all_window_cells(window)
            if bool(require_full_window_features) and any(cell not in cell_to_index for cell in cells):
                continue
            targets = center_support_global_cells(window).intersection(target_cells)
            if not targets:
                continue
            windows.append(window)
            window_targets[window.window_id] = set(targets)
            coverable.update(targets)
    return windows, window_targets, coverable


def plan_windows(
    *,
    target_cells_ordered: list[tuple[int, int]],
    grid_w: int,
    grid_h: int,
    cell_to_index: dict[tuple[int, int], int],
    require_full_window_features: bool,
    grid_step_px: int,
    max_windows: int,
) -> tuple[list[tuple[ProgressiveWindow, list[tuple[int, int]]]], list[tuple[int, int]]]:
    target_set = set(target_cells_ordered)
    windows, window_targets, coverable = enumerate_candidate_windows_for_targets(
        target_cells=target_set,
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        cell_to_index=cell_to_index,
        require_full_window_features=bool(require_full_window_features),
        grid_step_px=int(grid_step_px),
    )
    remaining = [cell for cell in target_cells_ordered if cell in coverable]
    skipped = [cell for cell in target_cells_ordered if cell not in coverable]
    steps: list[tuple[ProgressiveWindow, list[tuple[int, int]]]] = []
    used: set[str] = set()
    current: ProgressiveWindow | None = None
    while remaining:
        remaining_set = set(remaining)
        viable = [w for w in windows if w.window_id not in used and window_targets[w.window_id].intersection(remaining_set)]
        if not viable:
            skipped.extend(remaining)
            break
        if current is None:
            chosen = max(
                viable,
                key=lambda w: (
                    len(window_targets[w.window_id].intersection(remaining_set)),
                    -int(w.gy0),
                    -int(w.gx0),
                ),
            )
        else:
            chosen = max(
                viable,
                key=lambda w: (
                    len(window_targets[w.window_id].intersection(remaining_set)),
                    -(abs(int(w.gx0) - int(current.gx0)) + abs(int(w.gy0) - int(current.gy0))),
                    -int(w.gy0),
                    -int(w.gx0),
                ),
            )
        edit = [cell for cell in remaining if cell in window_targets[chosen.window_id]]
        steps.append((chosen, edit))
        used.add(chosen.window_id)
        current = chosen
        edit_set = set(edit)
        remaining = [cell for cell in remaining if cell not in edit_set]
        if int(max_windows) > 0 and len(steps) >= int(max_windows):
            skipped.extend(remaining)
            break
    return steps, skipped


@torch.no_grad()
def encode_patches_uni2(
    patches: list[Image.Image],
    *,
    uni_model,
    uni_transform,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    rows: list[torch.Tensor] = []
    for start in range(0, len(patches), int(batch_size)):
        batch = torch.stack([uni_transform(img.convert("RGB")) for img in patches[start : start + int(batch_size)]], dim=0).to(device=device)
        emb = uni_model(batch).detach().float().cpu()
        rows.append(emb)
    if not rows:
        return np.zeros((0, 1536), dtype=np.float32)
    return torch.cat(rows, dim=0).numpy().astype(np.float32, copy=False)


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dtype = torch.float16 if str(args.dtype) == "fp16" else torch.float32
    out_dir = args.out_dir / str(args.slide_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "experiment_args.json",
        {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "command": " ".join(shlex.quote(item) for item in sys.argv),
        },
    )

    if str(args.model_backend) == "mil":
        split_row = read_split_row(args.split_tsv, args.slide_key)
        source_label = int(split_row["label"])
        feature_path = args.features_root / f"{args.slide_key}.h5"
        coords_path = feature_path
        slide_path = find_slide_path(args.slides_dir, args.slide_key)
        if slide_path is None:
            raise FileNotFoundError(f"No matching SVS found in {args.slides_dir} for {args.slide_key}")
        features, coords = read_h5_features_coords(feature_path)
        classifier_model = build_mil_from_checkpoint(args.mil_ckpt, device=device)
        attention, pred_before, prob_pos_before = run_mil_attention(classifier_model, features, device=device)
        split_name = "split_tsv"
    else:
        labels_by_slide = load_clam_dataset_labels(args.clam_dataset_csv)
        split_members = read_clam_split_members(args.clam_splits_csv, str(args.clam_split))
        split_row = labels_by_slide.get(str(args.slide_key))
        if split_row is None:
            raise ValueError(f"CLAM slide id not found in dataset CSV: {args.slide_key}")
        if str(args.slide_key) not in split_members:
            raise ValueError(f"CLAM slide id {args.slide_key} is not in split {args.clam_split} ({args.clam_splits_csv})")
        source_label = int(split_row["label"])
        feature_path = args.clam_features_pt_dir / f"{args.slide_key}.pt"
        coords_path = args.clam_coords_h5_dir / f"{args.slide_key}.h5"
        slide_path = find_slide_path(args.slides_dir, args.slide_key)
        if slide_path is None:
            raise FileNotFoundError(f"No matching SVS found in {args.slides_dir} for {args.slide_key}")
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing CLAM pt features: {feature_path}")
        if not coords_path.exists():
            raise FileNotFoundError(f"Missing CLAM coord h5: {coords_path}")
        features = read_pt_features(feature_path)
        coords, _ = read_coord_h5(coords_path)
        if int(features.shape[0]) != int(coords.shape[0]):
            raise RuntimeError(f"{args.slide_key}: CLAM features n={features.shape[0]} but coords n={coords.shape[0]}")
        classifier_model = load_clam_model(args.clam_ckpt, device=device)
        attention, pred_before, prob_pos_before = run_clam_attention(
            classifier_model,
            features,
            device=device,
            attn_class=str(args.clam_attn_class),
        )
        split_name = str(args.clam_split)
    target_direction = target_direction_for_label(source_label, str(args.direction_mode))
    target_label = 1 if target_direction == "hpv_pos" else 0

    tile_size_level0 = infer_coord_tile_size(coords)
    cell_to_index, index_to_cell, grid_w, grid_h = build_cell_maps(coords, tile_size_level0=int(tile_size_level0))
    threshold = float(np.percentile(attention, float(args.attention_percentile)))
    selected_indices = np.where(np.asarray(attention) >= threshold)[0].astype(np.int64).tolist()
    selected_indices.sort(key=lambda idx: (-float(attention[int(idx)]), int(coords[int(idx), 1]), int(coords[int(idx), 0])))
    if int(args.max_edit_tiles) > 0:
        selected_indices = selected_indices[: int(args.max_edit_tiles)]
    target_cells_ordered = [index_to_cell[int(idx)] for idx in selected_indices]
    planned_steps, skipped_cells = plan_windows(
        target_cells_ordered=target_cells_ordered,
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        cell_to_index=cell_to_index,
        require_full_window_features=bool(args.require_full_window_features),
        grid_step_px=int(args.grid_step_px),
        max_windows=int(args.max_windows),
    )

    selected_rows = []
    for rank, idx in enumerate(selected_indices, start=1):
        gx, gy = index_to_cell[int(idx)]
        selected_rows.append(
            {
                "rank": int(rank),
                "tile_index": int(idx),
                "attention": float(attention[int(idx)]),
                "coord_x": int(coords[int(idx), 0]),
                "coord_y": int(coords[int(idx), 1]),
                "cell_gx": int(gx),
                "cell_gy": int(gy),
                "planned": bool((gx, gy) not in set(skipped_cells)),
            }
        )
    write_csv(out_dir / "selected_high_attention_tiles.csv", selected_rows)

    plan_rows = []
    for step_idx, (window, edit_cells) in enumerate(planned_steps, start=1):
        missing_context_cells = [cell for cell in all_window_cells(window) if cell not in cell_to_index]
        plan_rows.append(
            {
                "step": int(step_idx),
                "window_id": str(window.window_id),
                "gx0": int(window.gx0),
                "gy0": int(window.gy0),
                "level0_x": int(window.gx0) * int(tile_size_level0),
                "level0_y": int(window.gy0) * int(tile_size_level0),
                "edit_cells": encode_cells(edit_cells),
                "edit_tile_indices": ";".join(str(cell_to_index[cell]) for cell in edit_cells),
                "missing_context_cell_count": int(len(missing_context_cells)),
                "missing_context_cells": encode_cells(missing_context_cells),
            }
        )
    write_csv(out_dir / "window_plan.csv", plan_rows)

    if bool(args.dry_run):
        write_json(
            out_dir / "summary.json",
            {
                "slide_key": str(args.slide_key),
                "dry_run": True,
                "source_label": int(source_label),
                "target_direction": str(target_direction),
                "pred_before": int(pred_before),
                "prob_pos_before": float(prob_pos_before),
                "attention_percentile": float(args.attention_percentile),
                "attention_threshold": float(threshold),
                "selected_tile_count": int(len(selected_indices)),
                "planned_window_count": int(len(planned_steps)),
                "planned_tile_count": int(sum(len(edit_cells) for _, edit_cells in planned_steps)),
                "skipped_tile_count": int(len(skipped_cells)),
                "require_full_window_features": bool(args.require_full_window_features),
                "missing_context_cell_count": int(
                    sum(len([cell for cell in all_window_cells(window) if cell not in cell_to_index]) for window, _ in planned_steps)
                ),
                "tile_size_level0": int(tile_size_level0),
                "grid_shape": [int(grid_h), int(grid_w)],
                "model_backend": str(args.model_backend),
                "feature_path": str(feature_path),
                "coords_path": str(coords_path),
                "slide_path": str(slide_path),
                "split_name": str(split_name),
            },
        )
        print(f"[ok] dry run wrote plan to {out_dir}")
        return

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    chosen_latent = int(pos_latent if target_direction == "hpv_pos" else neg_latent)
    proto_vec = torch.from_numpy(np.asarray(proto_by_latent[chosen_latent], dtype=np.float32)).to(device=device)
    sae_strength = sae_strength_from_power(float(args.prototype_strength))
    condition_alpha_end = condition_alpha_end_from_power(base_alpha_end=float(args.mid_steer_alpha_end), value=float(args.prototype_strength))

    custom_pipeline_ref = resolve_custom_pipeline_ref(str(args.pix_pipeline_id))
    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=AutoencoderKL.from_pretrained(args.vae_model_id, subfolder=args.vae_subfolder, torch_dtype=dtype),
        custom_pipeline=custom_pipeline_ref,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_px, stride_px, cond_grid_side = resolve_pixcell_window_config(
        pix_model_id=str(args.pix_model_id),
        patch_px=0,
        stride_px=0,
    )
    uni_model, uni_transform = load_uni2(device=device)
    slide = open_slide(slide_path)

    edited_features = np.asarray(features, dtype=np.float32).copy()
    replacement_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    edited_patch_by_cell: dict[tuple[int, int], Image.Image] = {}
    attention_by_cell = {
        index_to_cell[int(idx)]: float(attention[int(idx)])
        for idx in range(int(len(attention)))
        if int(idx) in index_to_cell
    }
    visited_cells: set[tuple[int, int]] = set()
    for step_idx, (window, edit_cells_global) in enumerate(planned_steps, start=1):
        window_dir = out_dir / "windows" / f"step_{step_idx:03d}_{window.window_id}"
        window_dir.mkdir(parents=True, exist_ok=True)
        level0_x = int(window.gx0) * int(tile_size_level0)
        level0_y = int(window.gy0) * int(tile_size_level0)
        source_img, crop_w0, crop_h0 = read_region_rgb_at_magnification(
            slide,
            x0=int(level0_x),
            y0=int(level0_y),
            out_w=int(args.window_size),
            out_h=int(args.window_size),
            target_magnification=float(args.target_magnification),
        )
        local_z = np.zeros((4, 4, int(features.shape[1])), dtype=np.float32)
        missing_context_cells_global: list[tuple[int, int]] = []
        missing_context_cells_local: list[tuple[int, int]] = []
        for ly in range(4):
            for lx in range(4):
                cell = (int(window.gx0) + lx, int(window.gy0) + ly)
                if cell in cell_to_index:
                    local_z[ly, lx] = features[cell_to_index[cell]]
                else:
                    missing_context_cells_global.append(cell)
                    missing_context_cells_local.append((int(lx), int(ly)))
        local_base_t = torch.from_numpy(local_z).to(device=device, dtype=torch.float32)
        local_edit_t = local_base_t.clone()
        local_edit_cells = window_local_cells(window=window, global_cells=edit_cells_global)
        bad_local = [cell for cell in local_edit_cells if cell not in CENTER_2X2_LOCAL_CELLS]
        if bad_local:
            raise RuntimeError(f"Internal planning error: local edit cells outside center 2x2: {bad_local}")
        tile_mask = np.zeros((4, 4), dtype=np.float32)
        for lx, ly in local_edit_cells:
            tile_mask[int(ly), int(lx)] = 1.0
        local_edit_t, _ = edit_uni_z_grid_with_sae(
            sae_model=sae_model,
            z_grid=local_edit_t,
            target_latent_vector=proto_vec,
            target_latent_vector_strength=float(sae_strength),
            tile_mask=tile_mask,
            blend=float(args.steer_blend),
            keep_non_selected=True,
            return_debug=False,
        )
        source_np = np.asarray(source_img, dtype=np.float32) / 255.0
        source_t = torch.from_numpy(source_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
        preserve_source_latents = vae_encode_auto(
            pipeline.vae,
            source_t,
            use_tiled=False,
            tile_img=0,
            overlap_img=0,
        )
        preserve_map = build_history_aware_preserve_map(
            width=int(args.window_size),
            height=int(args.window_size),
            grid_step_px=int(args.grid_step_px),
            window=window,
            edit_cells_global=list(edit_cells_global),
            visited_cells_global=list(visited_cells),
            preserve_edit_strength=float(args.preserve_edit_strength),
            preserve_visited_strength=float(args.preserve_visited_strength),
            preserve_fresh_context_strength=float(args.preserve_fresh_context_strength),
        ).to(device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(int(args.seed) + int(step_idx), f"{args.slide_key}:{window.window_id}"))
        use_autocast = device.type == "cuda" and dtype == torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
        with torch.inference_mode(), ctx:
            img_t = sample_large_pixcell_multidiffusion(
                pipeline=pipeline,
                z_grid=local_base_t.to(device=device, dtype=dtype),
                scheduled_z_grid=local_edit_t.to(device=device, dtype=dtype),
                condition_start_ratio=float(args.mid_steer_start_ratio),
                condition_end_ratio=float(args.mid_steer_end_ratio),
                condition_alpha_start=float(args.mid_steer_alpha_start),
                condition_alpha_end=float(condition_alpha_end),
                condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
                out_h=int(args.window_size),
                out_w=int(args.window_size),
                patch_px=patch_px,
                stride_px=stride_px,
                cond_grid_side=cond_grid_side,
                guidance_scale=float(args.guidance),
                num_steps=int(args.steps),
                patch_batch=int(args.patch_batch),
                strength=0.0,
                init_latents=None,
                preserve_source_latents=preserve_source_latents,
                preserve_strength_map=preserve_map,
                preserve_outside_strength=float(args.preserve_fresh_context_strength),
                preserve_edit_strength=float(args.preserve_edit_strength),
                use_tiled_vae_decode=False,
                decode_tile_lat=128,
                decode_overlap_lat=16,
                generator=generator,
            )
        generated_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        generated_img = Image.fromarray(generated_np).convert("RGB")
        patches: list[Image.Image] = []
        patch_meta: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for local_cell in local_edit_cells:
            lx, ly = local_cell
            gx = int(window.gx0) + int(lx)
            gy = int(window.gy0) + int(ly)
            tile_idx = int(cell_to_index[(gx, gy)])
            x0 = int(lx) * int(args.grid_step_px)
            y0 = int(ly) * int(args.grid_step_px)
            patches.append(generated_img.crop((x0, y0, x0 + int(args.grid_step_px), y0 + int(args.grid_step_px))).convert("RGB"))
            patch_meta.append((tile_idx, (gx, gy), (lx, ly)))
        reencoded = encode_patches_uni2(
            patches,
            uni_model=uni_model,
            uni_transform=uni_transform,
            device=device,
        )
        for patch_img, emb, (tile_idx, global_cell, local_cell) in zip(patches, reencoded, patch_meta):
            edited_features[int(tile_idx)] = emb
            edited_patch_by_cell[(int(global_cell[0]), int(global_cell[1]))] = patch_img.convert("RGB")
            replacement_rows.append(
                {
                    "step": int(step_idx),
                    "window_id": str(window.window_id),
                    "tile_index": int(tile_idx),
                    "cell_gx": int(global_cell[0]),
                    "cell_gy": int(global_cell[1]),
                    "local_gx": int(local_cell[0]),
                    "local_gy": int(local_cell[1]),
                    "coord_x": int(coords[int(tile_idx), 0]),
                    "coord_y": int(coords[int(tile_idx), 1]),
                    "attention": float(attention[int(tile_idx)]),
                }
            )
        if bool(args.save_debug_artifacts):
            save_png(source_img, window_dir / "source_window.png")
            save_png(draw_cells_overlay(source_img, local_edit_cells, grid_step_px=int(args.grid_step_px)), window_dir / "source_selected_overlay.png")
            save_png(
                draw_window_context_overlay(
                    source_img,
                    edit_cells=local_edit_cells,
                    empty_cells=missing_context_cells_local,
                    grid_step_px=int(args.grid_step_px),
                ),
                window_dir / "source_context_overlay.png",
            )
            save_png(generated_img, window_dir / "steered_window.png")
            save_png(draw_cells_overlay(generated_img, local_edit_cells, grid_step_px=int(args.grid_step_px)), window_dir / "steered_selected_overlay.png")
            save_png(
                draw_window_context_overlay(
                    generated_img,
                    edit_cells=local_edit_cells,
                    empty_cells=missing_context_cells_local,
                    grid_step_px=int(args.grid_step_px),
                ),
                window_dir / "steered_context_overlay.png",
            )
            save_png(preserve_map_preview(preserve_map), window_dir / "preserve_map.png")
        window_rows.append(
            {
                "step": int(step_idx),
                "window_id": str(window.window_id),
                "level0_x": int(level0_x),
                "level0_y": int(level0_y),
                "crop_w_level0": int(crop_w0),
                "crop_h_level0": int(crop_h0),
                "edited_tile_count": int(len(local_edit_cells)),
                "missing_context_cell_count": int(len(missing_context_cells_global)),
                "missing_context_cells": encode_cells(missing_context_cells_global),
                "source_window": str(window_dir / "source_window.png") if bool(args.save_debug_artifacts) else "",
                "steered_window": str(window_dir / "steered_window.png") if bool(args.save_debug_artifacts) else "",
                "source_context_overlay": str(window_dir / "source_context_overlay.png") if bool(args.save_debug_artifacts) else "",
                "steered_context_overlay": str(window_dir / "steered_context_overlay.png") if bool(args.save_debug_artifacts) else "",
            }
        )
        visited_cells.update(all_window_cells(window))

    local_area_rows = save_local_edit_area_visuals(
        out_dir=out_dir,
        slide=slide,
        planned_steps=planned_steps,
        edited_patch_by_cell=edited_patch_by_cell,
        attention_by_cell=attention_by_cell,
        tile_size_level0=int(tile_size_level0),
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        target_magnification=float(args.target_magnification),
        local_vis_size=int(args.local_vis_size),
        grid_step_px=int(args.grid_step_px),
        max_local_vis_areas=int(args.max_local_vis_areas),
    )
    slide.close()
    if str(args.model_backend) == "mil":
        _, pred_after, prob_pos_after = run_mil_attention(classifier_model, edited_features, device=device)
    else:
        _, pred_after, prob_pos_after = run_clam_attention(
            classifier_model,
            edited_features,
            device=device,
            attn_class=str(args.clam_attn_class),
        )
    np.save(out_dir / "edited_features_selected_only.npy", edited_features)
    write_csv(out_dir / "replacement_manifest.csv", replacement_rows)
    write_csv(out_dir / "executed_windows.csv", window_rows)
    write_csv(out_dir / "local_edit_areas.csv", local_area_rows)
    summary = {
        "slide_key": str(args.slide_key),
        "source_label": int(source_label),
        "target_label": int(target_label),
        "target_direction": str(target_direction),
        "pred_before": int(pred_before),
        "prob_pos_before": float(prob_pos_before),
        "pred_after_selected_tile_reencode": int(pred_after),
        "prob_pos_after_selected_tile_reencode": float(prob_pos_after),
        "target_prob_before": float(target_prob(float(prob_pos_before), int(target_label))),
        "target_prob_after_selected_tile_reencode": float(target_prob(float(prob_pos_after), int(target_label))),
        "delta_target_prob": float(target_prob(float(prob_pos_after), int(target_label)) - target_prob(float(prob_pos_before), int(target_label))),
        "attention_percentile": float(args.attention_percentile),
        "attention_threshold": float(threshold),
        "selected_tile_count": int(len(selected_indices)),
        "planned_window_count": int(len(planned_steps)),
        "executed_window_count": int(len(window_rows)),
        "reencoded_tile_count": int(len(replacement_rows)),
        "skipped_tile_count": int(len(skipped_cells)),
        "require_full_window_features": bool(args.require_full_window_features),
        "missing_context_cell_count": int(sum(int(row["missing_context_cell_count"]) for row in window_rows)),
        "local_edit_area_count": int(len(local_area_rows)),
        "local_edit_areas_csv": str(out_dir / "local_edit_areas.csv"),
        "tile_size_level0": int(tile_size_level0),
        "feature_dim": int(features.shape[1]),
        "grid_shape": [int(grid_h), int(grid_w)],
        "slide_path": str(slide_path),
        "feature_path": str(feature_path),
        "coords_path": str(coords_path),
        "model_backend": str(args.model_backend),
        "split_name": str(split_name),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
