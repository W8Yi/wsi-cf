#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
import shlex
import sys

import numpy as np
import torch
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
    DEFAULT_SAE_CFG,
    DEFAULT_SAE_CKPT,
    ensure_legacy_repo_root_on_path,
)
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import parse_region_bank_csv, write_region_bank_csv
from wsi_cf.eval.hnsc_hpv import load_prototypes, pick_prototype_latent
from wsi_cf.generation.pixcell import resolve_pixcell_window_config, sample_large_pixcell_multidiffusion, vae_encode_auto
from wsi_cf.steering.cell_selection import (
    block_cells,
    decode_cells,
    encode_cells,
    parse_cell_specs,
    random_cells,
    random_connected_cells,
    validate_cells,
)

ensure_legacy_repo_root_on_path()

from utils.sae import load_sae_from_config  # type: ignore
from utils.sae_edit import edit_uni_z_grid_with_sae  # type: ignore


DEFAULT_CASES = [
    "baseline",
    "random_two",
    "neighbor_two",
    "neighbor_three",
    "block_2x2",
    "block_2x3",
    "manual",
]

DEFAULT_PROGRESSIVE_MODE = "none"
DEFAULT_PROGRESSIVE_COMMIT_MODE = "full_window_latest_wins"
DEFAULT_PROGRESSIVE_EDIT_SHAPES = [
    "center_one",
    "center_two_h",
    "center_2x2",
]


def _jsonify(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _serialize_args(args: argparse.Namespace) -> dict[str, object]:
    return {str(k): _jsonify(v) for k, v in vars(args).items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAE selected-cells steering cases on a prepared 10x region bank. "
            "This consumes saved region images and aligned UNI feature grids, then applies "
            "selected-cell SAE prototype steering for named test cases. It also supports a "
            "progressive 2048x2048 mode that steers overlapping 1024 windows across a larger region."
        )
    )
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cases", type=str, default="all", help="Comma-separated list or 'all'")
    parser.add_argument("--direction", type=str, default="hpv_pos", choices=["hpv_pos", "hpv_neg"])
    parser.add_argument("--source-label", type=int, default=None, choices=[0, 1], help="Optional label filter")
    parser.add_argument("--max-sources", type=int, default=0, help="Optional cap on number of source regions")
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--manual-cell", action="append", default=[], help="Repeatable gx,gy spec for the manual case")
    parser.add_argument("--neighbor-anchor-gx", type=int, default=1)
    parser.add_argument("--neighbor-anchor-gy", type=int, default=1)
    parser.add_argument("--block-2x2-origin-gx", type=int, default=1)
    parser.add_argument("--block-2x2-origin-gy", type=int, default=1)
    parser.add_argument("--block-2x3-origin-gx", type=int, default=1)
    parser.add_argument("--block-2x3-origin-gy", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--pix_model_id", type=str, default="StonyBrook-CVLab/PixCell-1024")
    parser.add_argument("--pix_pipeline_id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae_model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--preserve-outside-latents", action="store_true", help="Keep non-edited regions anchored to the original source image latents during denoising.")
    parser.add_argument("--preserve-outside-strength", type=float, default=1.0, help="How strongly to preserve non-edited regions in latent space.")
    parser.add_argument("--preserve-edit-strength", type=float, default=0.0, help="How strongly to preserve the selected edited region in latent space.")
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.0)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=1.0)
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
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progressive-mode", type=str, default=DEFAULT_PROGRESSIVE_MODE, choices=["none", "overlap_truth_right_strip"])
    parser.add_argument("--progressive-commit-mode", type=str, default=DEFAULT_PROGRESSIVE_COMMIT_MODE, choices=["full_window_latest_wins", "new_right_band_only"])
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--max-span", type=int, default=4)
    parser.add_argument("--edit-shapes", type=str, default="center_one,center_two_h,center_2x2")
    parser.add_argument("--target-row-index", type=int, default=1)
    parser.add_argument("--start-col-index", type=int, default=0)
    parser.add_argument("--save-progress-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-progressive", action=argparse.BooleanOptionalAction, default=False)
    return parser


def parse_cases(arg: str) -> list[str]:
    if str(arg).strip().lower() == "all":
        return list(DEFAULT_CASES)
    cases = [item.strip() for item in str(arg).split(",") if item.strip()]
    bad = [item for item in cases if item not in DEFAULT_CASES]
    if bad:
        raise ValueError(f"Unsupported cases: {bad}")
    return cases


def parse_edit_shapes(arg: str) -> list[str]:
    allowed = set(DEFAULT_PROGRESSIVE_EDIT_SHAPES)
    shapes = [item.strip() for item in str(arg).split(",") if item.strip()]
    if not shapes:
        raise ValueError("At least one progressive edit shape is required")
    bad = [item for item in shapes if item not in allowed]
    if bad:
        raise ValueError(f"Unsupported progressive edit shapes: {bad}")
    out: list[str] = []
    seen: set[str] = set()
    for shape in shapes:
        if shape not in seen:
            out.append(shape)
            seen.add(shape)
    return out


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def draw_selected_cells_overlay(img: Image.Image, *, cells: list[tuple[int, int]], grid_step_px: int) -> Image.Image:
    canvas = img.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for idx, (gx, gy) in enumerate(cells, start=1):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(canvas.size[0] - 1, x0 + int(grid_step_px) - 1)
        y1 = min(canvas.size[1] - 1, y0 + int(grid_step_px) - 1)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=6)
        draw.text((x0 + 8, y0 + 8), str(idx), fill=(255, 255, 0))
    return canvas


def make_edit_region_mask(
    *,
    width: int,
    height: int,
    cells: list[tuple[int, int]],
    grid_step_px: int,
) -> torch.Tensor:
    mask = torch.zeros((1, 1, int(height), int(width)), dtype=torch.float32)
    for gx, gy in cells:
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(int(width), x0 + int(grid_step_px))
        y1 = min(int(height), y0 + int(grid_step_px))
        mask[:, :, y0:y1, x0:x1] = 1.0
    return mask


def infer_grid_shape(z_grid: np.ndarray) -> tuple[int, int]:
    if z_grid.ndim != 3:
        raise ValueError(f"Expected z_grid to have shape [H,W,D], got {z_grid.shape}")
    return int(z_grid.shape[0]), int(z_grid.shape[1])


def make_window_starts(total: int, window: int, stride: int) -> list[int]:
    if int(window) <= 0 or int(stride) <= 0:
        raise ValueError("window and stride must be > 0")
    if int(total) < int(window):
        raise ValueError("window cannot be larger than total")
    if int(total) == int(window):
        return [0]
    starts = list(range(0, int(total) - int(window) + 1, int(stride)))
    if starts[-1] != int(total) - int(window):
        starts.append(int(total) - int(window))
    return starts


def enumerate_progressive_windows(
    *,
    region_w_px: int,
    region_h_px: int,
    window_size: int,
    window_stride: int,
    grid_step_px: int,
) -> list[dict[str, int]]:
    xs = make_window_starts(int(region_w_px), int(window_size), int(window_stride))
    ys = make_window_starts(int(region_h_px), int(window_size), int(window_stride))
    windows: list[dict[str, int]] = []
    idx = 0
    window_cells = int(window_size) // int(grid_step_px)
    for row_idx, top in enumerate(ys):
        for col_idx, left in enumerate(xs):
            windows.append(
                {
                    "window_index": idx,
                    "row_index": int(row_idx),
                    "col_index": int(col_idx),
                    "left": int(left),
                    "top": int(top),
                    "gx0": int(left) // int(grid_step_px),
                    "gy0": int(top) // int(grid_step_px),
                    "grid_w": int(window_cells),
                    "grid_h": int(window_cells),
                }
            )
            idx += 1
    return windows


def progressive_shape_cells(edit_shape: str) -> list[tuple[int, int]]:
    mapping = {
        "center_one": [(1, 1)],
        "center_two_h": [(1, 1), (2, 1)],
        "center_2x2": [(1, 1), (2, 1), (1, 2), (2, 2)],
    }
    if str(edit_shape) not in mapping:
        raise ValueError(f"Unsupported progressive edit shape: {edit_shape}")
    return list(mapping[str(edit_shape)])


def update_full_zgrid_selected_cells(
    *,
    full_zgrid: np.ndarray,
    edited_local_zgrid: np.ndarray,
    gx0: int,
    gy0: int,
    selected_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(full_zgrid, dtype=np.float32).copy()
    for lx, ly in selected_cells:
        out[int(gy0) + int(ly), int(gx0) + int(lx), :] = edited_local_zgrid[int(ly), int(lx), :]
    return out


def commit_progressive_window_update(
    *,
    current_canvas: np.ndarray,
    steered_img: Image.Image,
    left: int,
    top: int,
    window_size: int,
    window_stride: int,
    step_index: int,
    commit_mode: str,
) -> tuple[np.ndarray, tuple[int, int, int, int], str]:
    out = np.asarray(current_canvas, dtype=np.float32).copy()
    img = np.asarray(steered_img, dtype=np.float32) / 255.0
    chosen_mode = str(commit_mode)
    if int(step_index) == 0:
        x0_local, x1_local = 0, int(window_size)
        chosen_mode = "full_window"
    elif chosen_mode == "full_window_latest_wins":
        x0_local, x1_local = 0, int(window_size)
    elif chosen_mode == "new_right_band_only":
        x0_local = int(window_size) - int(window_stride)
        x1_local = int(window_size)
    else:
        raise ValueError(f"Unsupported progressive commit mode: {commit_mode}")
    y0_local, y1_local = 0, int(window_size)
    dst_x0 = int(left) + int(x0_local)
    dst_x1 = int(left) + int(x1_local)
    dst_y0 = int(top) + int(y0_local)
    dst_y1 = int(top) + int(y1_local)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[y0_local:y1_local, x0_local:x1_local]
    return out, (dst_x0, dst_y0, dst_x1, dst_y1), chosen_mode


def draw_progressive_overlay(
    img: Image.Image,
    *,
    windows: list[dict[str, int]],
    selected_cells_by_window: list[list[tuple[int, int]]],
    committed_boxes: list[tuple[int, int, int, int]],
    grid_step_px: int,
) -> Image.Image:
    canvas = img.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for window, cells in zip(windows, selected_cells_by_window):
        left = int(window["left"])
        top = int(window["top"])
        draw.rectangle([left, top, left + int(window["grid_w"]) * int(grid_step_px) - 1, top + int(window["grid_h"]) * int(grid_step_px) - 1], outline=(0, 255, 255), width=5)
        for idx, (lx, ly) in enumerate(cells, start=1):
            x0 = left + int(lx) * int(grid_step_px)
            y0 = top + int(ly) * int(grid_step_px)
            x1 = x0 + int(grid_step_px) - 1
            y1 = y0 + int(grid_step_px) - 1
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
            draw.text((x0 + 8, y0 + 8), str(idx), fill=(255, 255, 0))
    for x0, y0, x1, y1 in committed_boxes:
        draw.rectangle([int(x0), int(y0), int(x1) - 1, int(y1) - 1], outline=(0, 255, 0), width=4)
    return canvas


def build_contact_sheet(items: list[tuple[str, Image.Image]], *, thumb_size: int = 256, ncols: int = 3, pad: int = 12) -> Image.Image:
    if not items:
        return Image.new("RGB", (thumb_size, thumb_size), (245, 245, 245))
    label_h = 26
    ncols = max(1, int(ncols))
    nrows = (len(items) + ncols - 1) // ncols
    width = pad + ncols * (thumb_size + pad)
    height = pad + nrows * (thumb_size + label_h + pad)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(items):
        row = idx // ncols
        col = idx % ncols
        x0 = pad + col * (thumb_size + pad)
        y0 = pad + row * (thumb_size + label_h + pad)
        thumb = img.convert("RGB").resize((thumb_size, thumb_size), resample=Image.BILINEAR)
        canvas.paste(thumb, (x0, y0))
        draw.rectangle([x0, y0, x0 + thumb_size - 1, y0 + thumb_size - 1], outline=(180, 180, 180), width=1)
        draw.text((x0, y0 + thumb_size + 4), label, fill=(20, 20, 20))
    return canvas


def resolve_case_cells(
    *,
    case_name: str,
    grid_w: int,
    grid_h: int,
    manual_cells: list[tuple[int, int]],
    rng: random.Random,
    neighbor_anchor: tuple[int, int],
    block_2x2_origin: tuple[int, int],
    block_2x3_origin: tuple[int, int],
) -> list[tuple[int, int]]:
    if case_name == "baseline":
        return []
    if case_name == "random_two":
        return random_cells(grid_w=int(grid_w), grid_h=int(grid_h), count=2, rng=rng)
    if case_name == "neighbor_two":
        return random_connected_cells(
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            count=2,
            rng=rng,
            start=(int(neighbor_anchor[0]), int(neighbor_anchor[1])),
        )
    if case_name == "neighbor_three":
        return random_connected_cells(
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            count=3,
            rng=rng,
            start=(int(neighbor_anchor[0]), int(neighbor_anchor[1])),
        )
    if case_name == "block_2x2":
        return block_cells(
            origin_gx=int(block_2x2_origin[0]),
            origin_gy=int(block_2x2_origin[1]),
            width=2,
            height=2,
            grid_w=int(grid_w),
            grid_h=int(grid_h),
        )
    if case_name == "block_2x3":
        return block_cells(
            origin_gx=int(block_2x3_origin[0]),
            origin_gy=int(block_2x3_origin[1]),
            width=2,
            height=3,
            grid_w=int(grid_w),
            grid_h=int(grid_h),
        )
    if case_name == "manual":
        if not manual_cells:
            raise ValueError("manual case requested but no --manual-cell values were provided")
        return validate_cells(manual_cells, grid_w=int(grid_w), grid_h=int(grid_h))
    raise ValueError(f"Unsupported case_name: {case_name}")


def build_manifest(
    source_rows: list[dict[str, object]],
    *,
    cases: list[str],
    grid_w: int,
    grid_h: int,
    manual_cells: list[tuple[int, int]],
    seed: int,
    direction: str,
    prototype_latent: int,
    neighbor_anchor: tuple[int, int],
    block_2x2_origin: tuple[int, int],
    block_2x3_origin: tuple[int, int],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for source in source_rows:
        region_id = str(source["region_id"])
        for case_name in cases:
            row = {
                "source_region_id": region_id,
                "source_label": int(source["label"]),
                "source_slide_key": str(source["slide_key"]),
                "source_image_path": str(source["image_path"]),
                "source_feature_grid_path": str(source["feature_grid_path"]),
                "case_name": case_name,
                "condition": "baseline" if case_name == "baseline" else f"to_{direction}_selected_cells",
                "steer_mode": "none" if case_name == "baseline" else "selected_cells",
                "prototype_direction": "" if case_name == "baseline" else str(direction),
                "prototype_latent": "" if case_name == "baseline" else int(prototype_latent),
                "steer_cells": "",
                "steer_cell_count": 0,
            }
            if case_name != "baseline":
                rng = random.Random(f"{int(seed)}::{region_id}::{case_name}")
                cells = resolve_case_cells(
                    case_name=str(case_name),
                    grid_w=int(grid_w),
                    grid_h=int(grid_h),
                    manual_cells=manual_cells,
                    rng=rng,
                    neighbor_anchor=neighbor_anchor,
                    block_2x2_origin=block_2x2_origin,
                    block_2x3_origin=block_2x3_origin,
                )
                row["steer_cells"] = encode_cells(cells)
                row["steer_cell_count"] = len(cells)
            out.append(row)
    return out


def make_summary_row(row: dict[str, object], *, out_path: Path) -> dict[str, object]:
    return {
        "source_region_id": str(row["source_region_id"]),
        "source_label": int(row["source_label"]),
        "source_slide_key": str(row["source_slide_key"]),
        "case_name": str(row["case_name"]),
        "condition": str(row["condition"]),
        "steer_mode": str(row["steer_mode"]),
        "prototype_direction": str(row["prototype_direction"]),
        "prototype_latent": row["prototype_latent"],
        "steer_cells": str(row["steer_cells"]),
        "steer_cell_count": int(row["steer_cell_count"]),
        "output_path": str(out_path),
    }


def write_source_comparison(
    *,
    source_row: dict[str, object],
    source_runs: list[dict[str, object]],
    out_dir: Path,
    grid_step_px: int,
) -> None:
    source_region_id = str(source_row["region_id"])
    compare_dir = out_dir / "by_source" / source_region_id
    compare_dir.mkdir(parents=True, exist_ok=True)

    source_img = load_image(str(source_row["image_path"]))
    save_png(source_img, compare_dir / "source_region_actual.png")

    order = {name: idx for idx, name in enumerate(DEFAULT_CASES)}
    contact_items: list[tuple[str, Image.Image]] = [("source_actual", source_img)]
    for run in sorted(source_runs, key=lambda r: order.get(str(r["case_name"]), 10**6)):
        out_path = Path(str(run["output_path"]))
        if not out_path.exists():
            continue
        gen_img = load_image(str(out_path))
        case_name = str(run["case_name"])
        save_png(gen_img, compare_dir / f"{case_name}.png")
        if case_name == "baseline":
            save_png(gen_img, compare_dir / "source_region.png")
            save_png(gen_img, compare_dir / "source_region_generated.png")
            save_png(gen_img, compare_dir / "baseline_generated.png")
            contact_items.append(("source_generated", gen_img))
        cells = decode_cells(str(run.get("steer_cells", "")))
        if cells:
            overlay = draw_selected_cells_overlay(source_img, cells=cells, grid_step_px=int(grid_step_px))
            save_png(overlay, compare_dir / f"{case_name}__selected_overlay.png")
        if case_name != "baseline":
            contact_items.append((case_name, gen_img))

    sheet = build_contact_sheet(contact_items, thumb_size=256, ncols=3, pad=12)
    save_png(sheet, compare_dir / "comparison_contact_sheet.png")


def write_progressive_source_comparison(
    *,
    source_row: dict[str, object],
    source_runs: list[dict[str, object]],
    out_dir: Path,
) -> None:
    source_region_id = str(source_row["region_id"])
    compare_dir = out_dir / "by_source" / source_region_id
    compare_dir.mkdir(parents=True, exist_ok=True)
    source_img = load_image(str(source_row["image_path"]))
    save_png(source_img, compare_dir / "source_region_actual.png")

    def sort_key(row: dict[str, object]) -> tuple[str, int]:
        return (str(row.get("edit_shape", "")), int(row.get("span", 0)))

    contact_items: list[tuple[str, Image.Image]] = [("source_actual", source_img)]
    for run in sorted(source_runs, key=sort_key):
        out_path = Path(str(run["output_path"]))
        if not out_path.exists():
            continue
        label = f"{run['edit_shape']}_span{run['span']}"
        img = load_image(str(out_path))
        save_png(img, compare_dir / f"{label}.png")
        contact_items.append((label, img))
    sheet = build_contact_sheet(contact_items, thumb_size=256, ncols=3, pad=12)
    save_png(sheet, compare_dir / "comparison_contact_sheet.png")


def run_progressive_region(
    *,
    args: argparse.Namespace,
    args_payload: dict[str, object],
    row: dict[str, object],
    source_img: Image.Image,
    source_zgrid: np.ndarray,
    source_region_id: str,
    edit_shape: str,
    span: int,
    windows_by_rc: dict[tuple[int, int], dict[str, int]],
    pipeline,
    patch_px: int,
    stride_px: int,
    cond_grid_side: int,
    proto_by_latent: dict[int, np.ndarray | torch.Tensor],
    chosen_latent: int,
    sae_model,
    device,
    dtype,
) -> tuple[dict[str, object], Path]:
    selected_cells = progressive_shape_cells(edit_shape)
    active_windows: list[dict[str, int]] = []
    current_canvas = np.asarray(source_img, dtype=np.float32) / 255.0
    current_zgrid = np.asarray(source_zgrid, dtype=np.float32).copy()
    committed_boxes: list[tuple[int, int, int, int]] = []
    selected_cells_by_window: list[list[tuple[int, int]]] = []

    run_dir = args.out_dir / args.progressive_mode / edit_shape / f"span_{int(span)}" / source_region_id
    final_out_path = run_dir / "generated.png"
    if bool(args.skip_existing) and final_out_path.exists():
        summary_row = {
            "source_region_id": source_region_id,
            "source_label": int(row["source_label"]),
            "source_slide_key": str(row["source_slide_key"]),
            "progressive_mode": str(args.progressive_mode),
            "edit_shape": str(edit_shape),
            "span": int(span),
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "target_row_index": int(args.target_row_index),
            "start_col_index": int(args.start_col_index),
            "output_path": str(final_out_path),
        }
        return summary_row, final_out_path

    run_dir.mkdir(parents=True, exist_ok=True)
    save_png(source_img, run_dir / "source_region_actual.png")
    step_rows: list[dict[str, object]] = []

    for step_idx in range(int(span)):
        rc = (int(args.target_row_index), int(args.start_col_index) + int(step_idx))
        window = windows_by_rc[rc]
        active_windows.append(window)
        selected_cells_by_window.append(list(selected_cells))

        current_canvas_img = Image.fromarray((np.clip(current_canvas, 0.0, 1.0) * 255.0).astype(np.uint8))
        left = int(window["left"])
        top = int(window["top"])
        local_source_img = current_canvas_img.crop((left, top, left + int(args.window_size), top + int(args.window_size))).convert("RGB")
        gx0 = int(window["gx0"])
        gy0 = int(window["gy0"])
        local_base_zgrid = np.asarray(current_zgrid[gy0 : gy0 + 4, gx0 : gx0 + 4, :], dtype=np.float32)
        local_base_t = torch.from_numpy(local_base_zgrid).to(device=device, dtype=torch.float32)
        local_edit_t = local_base_t.clone()

        tile_mask = np.zeros(local_base_zgrid.shape[:2], dtype=np.float32)
        for lx, ly in selected_cells:
            tile_mask[int(ly), int(lx)] = 1.0
        local_edit_t, _ = edit_uni_z_grid_with_sae(
            sae_model=sae_model,
            z_grid=local_edit_t,
            target_latent_vector=proto_by_latent[int(chosen_latent)],
            target_latent_vector_strength=float(args.prototype_strength),
            tile_mask=tile_mask,
            blend=float(args.steer_blend),
            keep_non_selected=True,
            return_debug=False,
        )

        z_grid_base_pix = local_base_t.to(device=device, dtype=dtype)
        scheduled_z_grid = local_edit_t.to(device=device, dtype=dtype)
        preserve_source_latents = None
        edit_region_mask = None
        if bool(args.preserve_outside_latents):
            source_np = np.asarray(local_source_img, dtype=np.float32) / 255.0
            source_img_t = torch.from_numpy(source_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
            preserve_source_latents = vae_encode_auto(
                pipeline.vae,
                source_img_t,
                use_tiled=False,
                tile_img=0,
                overlap_img=0,
            )
            edit_region_mask = make_edit_region_mask(
                width=int(local_source_img.size[0]),
                height=int(local_source_img.size[1]),
                cells=selected_cells,
                grid_step_px=int(args.grid_step_px),
            ).to(device=device)

        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed) + int(span) * 100 + int(step_idx))
        use_autocast = device.type == "cuda" and dtype == torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
        with torch.inference_mode(), ctx:
            img_t = sample_large_pixcell_multidiffusion(
                pipeline=pipeline,
                z_grid=z_grid_base_pix,
                scheduled_z_grid=scheduled_z_grid,
                condition_start_ratio=float(args.mid_steer_start_ratio),
                condition_end_ratio=float(args.mid_steer_end_ratio),
                condition_alpha_start=float(args.mid_steer_alpha_start),
                condition_alpha_end=float(args.mid_steer_alpha_end),
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
                edit_region_mask=edit_region_mask,
                preserve_outside_strength=float(args.preserve_outside_strength),
                preserve_edit_strength=float(args.preserve_edit_strength),
                use_tiled_vae_decode=False,
                decode_tile_lat=128,
                decode_overlap_lat=16,
                generator=generator,
            )

        img_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        steered_img = Image.fromarray(img_np)
        current_canvas, commit_box, commit_mode = commit_progressive_window_update(
            current_canvas=current_canvas,
            steered_img=steered_img,
            left=left,
            top=top,
            window_size=int(args.window_size),
            window_stride=int(args.window_stride),
            step_index=int(step_idx),
            commit_mode=str(args.progressive_commit_mode),
        )
        committed_boxes.append(commit_box)
        current_zgrid = update_full_zgrid_selected_cells(
            full_zgrid=current_zgrid,
            edited_local_zgrid=local_edit_t.detach().cpu().numpy().astype(np.float32),
            gx0=gx0,
            gy0=gy0,
            selected_cells=selected_cells,
        )

        step_dir = run_dir / "steps" / f"step_{int(step_idx) + 1:02d}"
        if bool(args.save_progress_steps):
            step_dir.mkdir(parents=True, exist_ok=True)
            save_png(local_source_img, step_dir / "source_window_from_current_canvas.png")
            save_png(draw_selected_cells_overlay(local_source_img, cells=selected_cells, grid_step_px=int(args.grid_step_px)), step_dir / "selected_cells_overlay.png")
            save_png(steered_img, step_dir / "steered_window.png")
        step_rows.append(
            {
                "source_region_id": source_region_id,
                "edit_shape": str(edit_shape),
                "span": int(span),
                "step_index": int(step_idx),
                "row_index": int(window["row_index"]),
                "col_index": int(window["col_index"]),
                "window_index": int(window["window_index"]),
                "left": int(left),
                "top": int(top),
                "gx0": int(gx0),
                "gy0": int(gy0),
                "selected_cells_local": encode_cells(selected_cells),
                "selected_cells_global": encode_cells([(gx0 + int(lx), gy0 + int(ly)) for lx, ly in selected_cells]),
                "commit_mode": str(commit_mode),
                "commit_bounds_global": f"{commit_box[0]},{commit_box[1]},{commit_box[2]},{commit_box[3]}",
                "source_window_path": str(step_dir / "source_window_from_current_canvas.png") if bool(args.save_progress_steps) else "",
                "overlay_path": str(step_dir / "selected_cells_overlay.png") if bool(args.save_progress_steps) else "",
                "steered_window_path": str(step_dir / "steered_window.png") if bool(args.save_progress_steps) else "",
            }
        )

    final_img = Image.fromarray((np.clip(current_canvas, 0.0, 1.0) * 255.0).astype(np.uint8))
    save_png(final_img, final_out_path)
    save_png(
        draw_progressive_overlay(
            source_img,
            windows=active_windows,
            selected_cells_by_window=selected_cells_by_window,
            committed_boxes=committed_boxes,
            grid_step_px=int(args.grid_step_px),
        ),
        run_dir / "source_region_edit_overlay.png",
    )
    save_png(
        draw_progressive_overlay(
            final_img,
            windows=active_windows,
            selected_cells_by_window=selected_cells_by_window,
            committed_boxes=committed_boxes,
            grid_step_px=int(args.grid_step_px),
        ),
        run_dir / "generated_edit_overlay.png",
    )
    write_region_bank_csv(run_dir / "step_manifest.csv", step_rows)
    write_json(
        run_dir / "run_meta.json",
        {
            "source_region_id": source_region_id,
            "source_label": int(row["source_label"]),
            "source_slide_key": str(row["source_slide_key"]),
            "source_image_path": str(row["source_image_path"]),
            "source_feature_grid_path": str(row["source_feature_grid_path"]),
            "progressive_mode": str(args.progressive_mode),
            "edit_shape": str(edit_shape),
            "span": int(span),
            "grid_shape": [int(source_zgrid.shape[0]), int(source_zgrid.shape[1])],
            "region_size": [int(source_img.size[0]), int(source_img.size[1])],
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "progressive_commit_mode": str(args.progressive_commit_mode),
            "target_row_index": int(args.target_row_index),
            "start_col_index": int(args.start_col_index),
            "committed_boxes": [
                {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}
                for x0, y0, x1, y1 in committed_boxes
            ],
            "prototype_direction": str(args.direction),
            "prototype_latent": int(chosen_latent),
            "prototype_key": str(args.prototype_key),
            "prototype_strength": float(args.prototype_strength),
            "steer_blend": float(args.steer_blend),
            "preserve_outside_latents": bool(args.preserve_outside_latents),
            "preserve_outside_strength": float(args.preserve_outside_strength),
            "preserve_edit_strength": float(args.preserve_edit_strength),
            "mid_steer_start_ratio": float(args.mid_steer_start_ratio),
            "mid_steer_end_ratio": float(args.mid_steer_end_ratio),
            "mid_steer_alpha_start": float(args.mid_steer_alpha_start),
            "mid_steer_alpha_end": float(args.mid_steer_alpha_end),
            "mid_steer_alpha_schedule": str(args.mid_steer_alpha_schedule),
            "pix_model_id": str(args.pix_model_id),
            "grid_step_px": int(args.grid_step_px),
            "seed": int(args.seed),
            "steps": int(args.steps),
            "guidance": float(args.guidance),
            "baseline_progressive": bool(args.baseline_progressive),
            "output_path": str(final_out_path),
            "step_manifest_path": str(run_dir / "step_manifest.csv"),
            "experiment_args_path": str(args.out_dir / "experiment_args.json"),
            "cli_args": args_payload["cli_args"],
            "command": args_payload["command"],
        },
    )
    summary_row = {
        "source_region_id": source_region_id,
        "source_label": int(row["source_label"]),
        "source_slide_key": str(row["source_slide_key"]),
        "progressive_mode": str(args.progressive_mode),
        "edit_shape": str(edit_shape),
        "span": int(span),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "progressive_commit_mode": str(args.progressive_commit_mode),
        "target_row_index": int(args.target_row_index),
        "start_col_index": int(args.start_col_index),
        "output_path": str(final_out_path),
    }
    return summary_row, final_out_path


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    set_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "cli_args": _serialize_args(args),
        "argv": list(argv) if argv is not None else list(sys.argv[1:]),
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else list(sys.argv[1:])))),
        "cwd": str(Path.cwd()),
    }
    write_json(args.out_dir / "experiment_args.json", args_payload)

    region_rows = parse_region_bank_csv(args.region_bank_csv)
    source_rows = [
        {
            "region_id": row.region_id,
            "label": int(row.label),
            "slide_key": row.slide_key,
            "image_path": row.image_path,
            "feature_grid_path": row.feature_grid_path,
            "region_w": int(row.region_w),
            "region_h": int(row.region_h),
        }
        for row in region_rows
    ]
    source_rows = sorted(source_rows, key=lambda row: (int(row["label"]), str(row["slide_key"]), str(row["region_id"])))
    if args.source_label is not None:
        source_rows = [row for row in source_rows if int(row["label"]) == int(args.source_label)]
    if int(args.max_sources) > 0:
        source_rows = source_rows[: int(args.max_sources)]
    if not source_rows:
        raise ValueError("No source rows selected from region_bank.csv")

    manual_cells = parse_cell_specs(list(args.manual_cell))
    first_zgrid = np.asarray(np.load(str(source_rows[0]["feature_grid_path"])), dtype=np.float32)
    grid_h, grid_w = infer_grid_shape(first_zgrid)

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    chosen_latent = int(pos_latent if str(args.direction) == "hpv_pos" else neg_latent)

    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=AutoencoderKL.from_pretrained(args.vae_model_id, subfolder=args.vae_subfolder, torch_dtype=dtype),
        custom_pipeline=args.pix_pipeline_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_px, stride_px, cond_grid_side = resolve_pixcell_window_config(
        pix_model_id=args.pix_model_id,
        patch_px=0,
        stride_px=0,
    )

    if str(args.progressive_mode) == "none":
        manifest_rows = build_manifest(
            source_rows,
            cases=parse_cases(args.cases),
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            manual_cells=manual_cells,
            seed=int(args.seed),
            direction=str(args.direction),
            prototype_latent=int(chosen_latent),
            neighbor_anchor=(int(args.neighbor_anchor_gx), int(args.neighbor_anchor_gy)),
            block_2x2_origin=(int(args.block_2x2_origin_gx), int(args.block_2x2_origin_gy)),
            block_2x3_origin=(int(args.block_2x3_origin_gx), int(args.block_2x3_origin_gy)),
        )
        write_region_bank_csv(args.out_dir / "experiment_manifest.csv", manifest_rows)

        summary_rows: list[dict[str, object]] = []
        for row in manifest_rows:
            source_region_id = str(row["source_region_id"])
            case_name = str(row["case_name"])
            run_dir = args.out_dir / case_name / source_region_id
            out_path = run_dir / "generated.png"
            if bool(args.skip_existing) and out_path.exists():
                summary_rows.append(make_summary_row(row, out_path=out_path))
                continue

            source_img = load_image(str(row["source_image_path"]))
            source_zgrid = np.asarray(np.load(str(row["source_feature_grid_path"])), dtype=np.float32)
            z_grid_base_t = torch.from_numpy(source_zgrid).to(device=device, dtype=torch.float32)
            z_grid_edit_t = z_grid_base_t.clone()

            cells = decode_cells(str(row.get("steer_cells", "")))
            steer_mode = str(row["steer_mode"])
            if steer_mode == "selected_cells":
                tile_mask = np.zeros(source_zgrid.shape[:2], dtype=np.float32)
                cells = validate_cells(cells, grid_w=source_zgrid.shape[1], grid_h=source_zgrid.shape[0])
                for gx, gy in cells:
                    tile_mask[gy, gx] = 1.0
                z_grid_edit_t, _ = edit_uni_z_grid_with_sae(
                    sae_model=sae_model,
                    z_grid=z_grid_edit_t,
                    target_latent_vector=proto_by_latent[int(row["prototype_latent"])],
                    target_latent_vector_strength=float(args.prototype_strength),
                    tile_mask=tile_mask,
                    blend=float(args.steer_blend),
                    keep_non_selected=True,
                    return_debug=False,
                )

            z_grid_base_pix = z_grid_base_t.to(device=device, dtype=dtype)
            scheduled_z_grid = None if steer_mode == "none" else z_grid_edit_t.to(device=device, dtype=dtype)
            preserve_source_latents = None
            edit_region_mask = None
            if bool(args.preserve_outside_latents) and cells:
                source_np = np.asarray(source_img.convert("RGB"), dtype=np.float32) / 255.0
                source_img_t = torch.from_numpy(source_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
                preserve_source_latents = vae_encode_auto(
                    pipeline.vae,
                    source_img_t,
                    use_tiled=False,
                    tile_img=0,
                    overlap_img=0,
                )
                edit_region_mask = make_edit_region_mask(
                    width=int(source_img.size[0]),
                    height=int(source_img.size[1]),
                    cells=cells,
                    grid_step_px=int(args.grid_step_px),
                ).to(device=device)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.seed))
            h, w = source_img.size[1], source_img.size[0]
            use_autocast = device.type == "cuda" and dtype == torch.float16
            ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
            with torch.inference_mode(), ctx:
                img_t = sample_large_pixcell_multidiffusion(
                    pipeline=pipeline,
                    z_grid=z_grid_base_pix,
                    scheduled_z_grid=scheduled_z_grid,
                    condition_start_ratio=float(args.mid_steer_start_ratio),
                    condition_end_ratio=float(args.mid_steer_end_ratio),
                    condition_alpha_start=float(args.mid_steer_alpha_start),
                    condition_alpha_end=float(args.mid_steer_alpha_end),
                    condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
                    out_h=h,
                    out_w=w,
                    patch_px=patch_px,
                    stride_px=stride_px,
                    cond_grid_side=cond_grid_side,
                    guidance_scale=float(args.guidance),
                    num_steps=int(args.steps),
                    patch_batch=int(args.patch_batch),
                    strength=0.0,
                    init_latents=None,
                    preserve_source_latents=preserve_source_latents,
                    edit_region_mask=edit_region_mask,
                    preserve_outside_strength=float(args.preserve_outside_strength),
                    preserve_edit_strength=float(args.preserve_edit_strength),
                    use_tiled_vae_decode=False,
                    decode_tile_lat=128,
                    decode_overlap_lat=16,
                    generator=generator,
                )

            run_dir.mkdir(parents=True, exist_ok=True)
            save_png(source_img, run_dir / "source.png")
            if cells:
                save_png(draw_selected_cells_overlay(source_img, cells=cells, grid_step_px=int(args.grid_step_px)), run_dir / "selected_overlay.png")
            np.save(run_dir / "edited_zgrid.npy", z_grid_edit_t.detach().cpu().numpy().astype(np.float32))
            img_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            save_png(Image.fromarray(img_np), out_path)
            write_json(
                run_dir / "run_meta.json",
                {
                    "source_region_id": source_region_id,
                    "source_label": int(row["source_label"]),
                    "source_slide_key": str(row["source_slide_key"]),
                    "source_image_path": str(row["source_image_path"]),
                    "source_feature_grid_path": str(row["source_feature_grid_path"]),
                    "case_name": case_name,
                    "condition": str(row["condition"]),
                    "steer_mode": steer_mode,
                    "prototype_direction": str(row["prototype_direction"]),
                    "prototype_latent": row["prototype_latent"],
                    "prototype_key": str(args.prototype_key),
                    "prototype_strength": float(args.prototype_strength),
                    "steer_blend": float(args.steer_blend),
                    "preserve_outside_latents": bool(args.preserve_outside_latents),
                    "preserve_outside_strength": float(args.preserve_outside_strength),
                    "preserve_edit_strength": float(args.preserve_edit_strength),
                    "mid_steer_start_ratio": float(args.mid_steer_start_ratio),
                    "mid_steer_end_ratio": float(args.mid_steer_end_ratio),
                    "mid_steer_alpha_start": float(args.mid_steer_alpha_start),
                    "mid_steer_alpha_end": float(args.mid_steer_alpha_end),
                    "mid_steer_alpha_schedule": str(args.mid_steer_alpha_schedule),
                    "steer_cells": str(row.get("steer_cells", "")),
                    "steer_cell_count": int(row.get("steer_cell_count", 0)),
                    "pix_model_id": str(args.pix_model_id),
                    "grid_step_px": int(args.grid_step_px),
                    "seed": int(args.seed),
                    "steps": int(args.steps),
                    "guidance": float(args.guidance),
                    "output_path": str(out_path),
                    "experiment_args_path": str(args.out_dir / "experiment_args.json"),
                    "cli_args": args_payload["cli_args"],
                    "command": args_payload["command"],
                },
            )
            summary_rows.append(make_summary_row(row, out_path=out_path))
            print(f"[ok] wrote {out_path}")

        write_region_bank_csv(args.out_dir / "run_summary.csv", summary_rows)
        write_json(
            args.out_dir / "run_summary.json",
            {
                "n_sources": len(source_rows),
                "n_runs_planned": len(manifest_rows),
                "n_runs_completed": len(summary_rows),
                "direction": str(args.direction),
                "experiment_manifest_csv": str(args.out_dir / "experiment_manifest.csv"),
                "prototype_key": str(args.prototype_key),
                "prototype_strength": float(args.prototype_strength),
                "prototype_latent": int(chosen_latent),
                "experiment_args_path": str(args.out_dir / "experiment_args.json"),
                "cli_args": args_payload["cli_args"],
                "command": args_payload["command"],
            },
        )

        source_map = {str(row["region_id"]): row for row in source_rows}
        runs_by_source: dict[str, list[dict[str, object]]] = {}
        for row in summary_rows:
            runs_by_source.setdefault(str(row["source_region_id"]), []).append(row)
        for source_region_id, source_runs in runs_by_source.items():
            source_row = source_map.get(source_region_id)
            if source_row is None:
                continue
            write_source_comparison(
                source_row=source_row,
                source_runs=source_runs,
                out_dir=args.out_dir,
                grid_step_px=int(args.grid_step_px),
            )
        return

    if int(args.window_size) != 1024:
        raise ValueError("progressive mode currently supports only window-size=1024")
    if int(args.window_stride) <= 0 or int(args.window_stride) > int(args.window_size):
        raise ValueError("window-stride must be in [1, window-size]")
    if int(args.window_size) % int(args.grid_step_px) != 0 or int(args.window_stride) % int(args.grid_step_px) != 0:
        raise ValueError("window-size and window-stride must be divisible by grid-step-px")
    progressive_shapes = parse_edit_shapes(args.edit_shapes)
    manifest_rows: list[dict[str, object]] = []
    summary_rows = []
    source_map = {str(row["region_id"]): row for row in source_rows}

    for source in source_rows:
        source_img = load_image(str(source["image_path"]))
        source_zgrid = np.asarray(np.load(str(source["feature_grid_path"])), dtype=np.float32)
        source_grid_h, source_grid_w = infer_grid_shape(source_zgrid)
        if source_grid_h < 4 or source_grid_w < 4:
            raise ValueError("progressive mode requires at least a 4x4 source grid")
        windows = enumerate_progressive_windows(
            region_w_px=int(source_img.size[0]),
            region_h_px=int(source_img.size[1]),
            window_size=int(args.window_size),
            window_stride=int(args.window_stride),
            grid_step_px=int(args.grid_step_px),
        )
        windows_by_rc = {(int(item["row_index"]), int(item["col_index"])): item for item in windows}
        row_windows = [item for item in windows if int(item["row_index"]) == int(args.target_row_index)]
        if not row_windows:
            raise ValueError(f"target-row-index={args.target_row_index} is invalid for source {source['region_id']}")
        max_col_index = max(int(item["col_index"]) for item in row_windows)
        if not (0 <= int(args.start_col_index) <= max_col_index):
            raise ValueError(f"start-col-index must be in [0,{max_col_index}] for source {source['region_id']}")
        span_limit = min(int(args.max_span), max_col_index - int(args.start_col_index) + 1)
        if span_limit <= 0:
            raise ValueError(f"No valid progressive span for source {source['region_id']}")

        for edit_shape in progressive_shapes:
            for span in range(1, int(span_limit) + 1):
                manifest_rows.append(
                    {
                        "source_region_id": str(source["region_id"]),
                        "source_label": int(source["label"]),
                        "source_slide_key": str(source["slide_key"]),
                        "source_image_path": str(source["image_path"]),
                        "source_feature_grid_path": str(source["feature_grid_path"]),
                        "progressive_mode": str(args.progressive_mode),
                        "edit_shape": str(edit_shape),
                        "span": int(span),
                        "target_row_index": int(args.target_row_index),
                        "start_col_index": int(args.start_col_index),
                        "window_size": int(args.window_size),
                        "window_stride": int(args.window_stride),
                    }
                )
                summary_row, out_path = run_progressive_region(
                    args=args,
                    args_payload=args_payload,
                    row={
                        "source_region_id": str(source["region_id"]),
                        "source_label": int(source["label"]),
                        "source_slide_key": str(source["slide_key"]),
                        "source_image_path": str(source["image_path"]),
                        "source_feature_grid_path": str(source["feature_grid_path"]),
                    },
                    source_img=source_img,
                    source_zgrid=source_zgrid,
                    source_region_id=str(source["region_id"]),
                    edit_shape=str(edit_shape),
                    span=int(span),
                    windows_by_rc=windows_by_rc,
                    pipeline=pipeline,
                    patch_px=patch_px,
                    stride_px=stride_px,
                    cond_grid_side=cond_grid_side,
                    proto_by_latent=proto_by_latent,
                    chosen_latent=int(chosen_latent),
                    sae_model=sae_model,
                    device=device,
                    dtype=dtype,
                )
                summary_rows.append(summary_row)
                print(f"[ok] wrote {out_path}")

    write_region_bank_csv(args.out_dir / "experiment_manifest.csv", manifest_rows)
    write_region_bank_csv(args.out_dir / "run_summary.csv", summary_rows)
    write_json(
        args.out_dir / "run_summary.json",
        {
            "n_sources": len(source_rows),
            "n_runs_planned": len(manifest_rows),
            "n_runs_completed": len(summary_rows),
            "direction": str(args.direction),
            "progressive_mode": str(args.progressive_mode),
            "edit_shapes": progressive_shapes,
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "progressive_commit_mode": str(args.progressive_commit_mode),
            "max_span_requested": int(args.max_span),
            "experiment_manifest_csv": str(args.out_dir / "experiment_manifest.csv"),
            "prototype_key": str(args.prototype_key),
            "prototype_strength": float(args.prototype_strength),
            "prototype_latent": int(chosen_latent),
            "experiment_args_path": str(args.out_dir / "experiment_args.json"),
            "cli_args": args_payload["cli_args"],
            "command": args_payload["command"],
        },
    )

    runs_by_source: dict[str, list[dict[str, object]]] = {}
    for summary_row in summary_rows:
        runs_by_source.setdefault(str(summary_row["source_region_id"]), []).append(summary_row)
    for source_region_id, source_runs in runs_by_source.items():
        source_row = source_map.get(source_region_id)
        if source_row is None:
            continue
        write_progressive_source_comparison(
            source_row=source_row,
            source_runs=source_runs,
            out_dir=args.out_dir / str(args.progressive_mode),
        )


if __name__ == "__main__":
    main()
