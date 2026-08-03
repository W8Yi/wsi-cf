#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from PIL import Image, ImageFilter

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
    DEFAULT_SAE_VARIANT,
    DEFAULT_SHOWCASE_EDIT_MANIFEST,
    DEFAULT_SHOWCASE_OUT_DIR,
    DEFAULT_SHOWCASE_REGION_IMAGE,
    DEFAULT_TASK,
    SAE_VARIANTS,
    resolve_sae_paths,
)
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import RegionBankRow, parse_region_bank_csv
from wsi_cf.data.slides import open_slide, read_region_rgb_at_magnification
from wsi_cf.eval.hnsc_hpv import load_prototypes, pick_prototype_latent
from wsi_cf.generation.pixcell import (
    build_uni_grid_from_image,
    load_uni2,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
    vae_encode_auto,
)
from wsi_cf.steering.sae_edit import edit_uni_z_grid_with_sae
from wsi_cf.steering.edit_policy import add_edit_policy_args, apply_edit_policy
from wsi_cf.steering.progressive import (
    CENTER_2X2_LOCAL_CELLS,
    EDIT_SUPPORT_CHOICES,
    EDIT_SUPPORT_CENTER_2X2,
    EDIT_SUPPORT_PADDED_CENTER_2X2,
    advance_progressive_state,
    build_history_aware_preserve_map,
    draw_cells_overlay,
    draw_step_edit_area_zoom_4x4,
    draw_step_region_overlay,
    load_progressive_edit_manifest,
    local_edit_support_global_cells,
    make_initial_progressive_state,
    plan_progressive_steps,
    preserve_map_preview,
    split_cells_by_edit_support,
    window_local_cells,
)
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_decode_latents, sae_encode_features


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
            "Run the canonical manifest-driven progressive region editor. "
            "The editor automatically plans overlapping PixCell-1024 windows from target grid cells, "
            "tracks edit/visit history, and uses history-aware preservation during diffusion."
        )
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--region-image", type=Path, default=DEFAULT_SHOWCASE_REGION_IMAGE)
    parser.add_argument("--region-bank-csv", type=Path, default=None)
    parser.add_argument("--edit-manifest", type=Path, default=DEFAULT_SHOWCASE_EDIT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SHOWCASE_OUT_DIR)
    add_edit_policy_args(parser)
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument("--direction", type=str, default="hpv_pos", choices=["hpv_pos", "hpv_neg"])
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap on number of manifest runs to execute")
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
    parser.add_argument("--prototype-strength", type=float, default=0.9)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--preserve-edit-strength", type=float, default=0.0)
    parser.add_argument("--preserve-visited-strength", type=float, default=0.84)
    parser.add_argument("--preserve-fresh-context-strength", type=float, default=0.22)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.55)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=0.4)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument(
        "--steer-context-halo-weight",
        type=float,
        default=0.0,
        help=(
            "Weak SAE steering weight for non-committed context cells adjacent to the steered support. "
            "Use values like 0.15-0.35 to soften the center-support conditioning boundary."
        ),
    )
    parser.add_argument(
        "--steer-context-halo-radius-cells",
        type=int,
        default=1,
        help="Chebyshev radius, in local UNI cells, for --steer-context-halo-weight.",
    )
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--sae-ckpt", type=Path, default=None, help=f"Explicit SAE checkpoint override. Defaults to --sae-variant ({DEFAULT_SAE_CKPT}).")
    parser.add_argument("--sae-cfg", type=Path, default=None, help=f"Explicit SAE config override. Defaults to --sae-variant ({DEFAULT_SAE_CFG}).")
    parser.add_argument(
        "--prototype-npz",
        type=Path,
        default=DEFAULT_HNSCC_PROTOTYPE_NPZ,
    )
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--concepts-json", type=Path, default=None, help="Optional selected_concepts.json for generic task concept steering.")
    parser.add_argument("--representative-tiles-csv", type=Path, default=None, help="Representative tile CSV used to infer target activation values for --concepts-json.")
    parser.add_argument("--concept-class-label", type=str, default="", help="Optional concept class label to keep from selected_concepts.json.")
    parser.add_argument("--concept-ranking-method", type=str, default="attention_weighted", choices=["attention_weighted", "activation"])
    parser.add_argument("--concept-target-stat", type=str, default="median", choices=["median", "mean", "q75", "max"])
    parser.add_argument("--concept-target-top-k", type=int, default=5, help="Use only the top K representative tiles per concept to estimate steering target activation; 0 uses all rows.")
    parser.add_argument(
        "--concept-steering-mode",
        type=str,
        default="prototype_vector",
        choices=["prototype_vector", "latent_target"],
        help=(
            "prototype_vector builds a full SAE-code prototype from representative tiles and steers selected cells toward it. "
            "latent_target only clamps the listed concept latent activation and is kept for ablations."
        ),
    )
    parser.add_argument("--max-concepts", type=int, default=0, help="0 means use all concepts in --concepts-json.")
    parser.add_argument(
        "--edit-support",
        type=str,
        default=EDIT_SUPPORT_PADDED_CENTER_2X2,
        choices=list(EDIT_SUPPORT_CHOICES),
        help=(
            "padded_center_2x2 expands slide-backed regions by a context halo, shifts targets inward, "
            "and edits only center cells; full_window is intended for deliberately naive baselines."
        ),
    )
    parser.add_argument(
        "--context-halo-cells",
        type=int,
        default=1,
        help="Context halo, in UNI grid cells, used by edit_support=padded_center_2x2.",
    )
    parser.add_argument(
        "--window-stride-cells",
        type=int,
        default=2,
        help=(
            "Progressive window stride in UNI grid cells. Use 1 for overlapping center-2x2 "
            "edits where each new window can reuse one already-edited row/column as context."
        ),
    )
    parser.add_argument(
        "--window-selection-mode",
        type=str,
        default="coverage",
        choices=["coverage", "overlap"],
        help=(
            "coverage maximizes newly covered target cells first. overlap prefers nearby windows "
            "that reuse already-edited support cells, reducing internal center-support boundaries."
        ),
    )
    parser.add_argument(
        "--steer-full-support",
        action="store_true",
        help=(
            "Steer the full allowed edit support for each window, not only newly covered target cells. "
            "Useful with --window-stride-cells 1 so overlap cells are regenerated together."
        ),
    )
    parser.add_argument(
        "--preserve-full-support",
        action="store_true",
        help=(
            "Build the PixCell preservation map with the full allowed edit support as editable, "
            "not only newly covered target cells. This lets stride-1 overlap regenerate the whole "
            "center 2x2 even when some cells were already steered."
        ),
    )
    parser.add_argument(
        "--commit-mode",
        type=str,
        default="support_cells",
        choices=["full_window", "support_cells", "edit_cells"],
        help=(
            "Which pixels from each generated PixCell window are committed back to the full region. "
            "full_window preserves legacy behavior; support_cells avoids visible 1024-window seams."
        ),
    )
    parser.add_argument(
        "--commit-feather-px",
        type=int,
        default=64,
        help="Optional Gaussian feather radius for support/edit-cell commit masks.",
    )
    parser.add_argument(
        "--commit-halo-cells",
        type=int,
        default=0,
        help=(
            "Optional Chebyshev-radius cell halo around commit cells to blend into the canvas. "
            "Support/edit cells keep alpha 1.0; halo cells use --commit-halo-alpha."
        ),
    )
    parser.add_argument(
        "--commit-halo-alpha",
        type=float,
        default=0.0,
        help="Pixel blend alpha for --commit-halo-cells. Use 0 to disable soft halo commit.",
    )
    parser.add_argument(
        "--preserve-invalid-feature-cells",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After generation, restore cells excluded by valid_feature_mask.npy from the source image. "
            "This prevents full-window context regeneration from tinting blank/background tiles."
        ),
    )
    parser.add_argument(
        "--invalid-feature-feather-px",
        type=int,
        default=64,
        help="Gaussian feather radius for the final valid-feature/source composite.",
    )
    parser.add_argument(
        "--preserve-outside-target-cells",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After generation, restore pixels outside the requested target cells from the source image. "
            "This keeps a localized intervention from changing full-window context pixels."
        ),
    )
    parser.add_argument(
        "--target-cell-feather-px",
        type=int,
        default=64,
        help="Gaussian feather radius for the final target-cell/source composite.",
    )
    parser.add_argument("--output-mode", type=str, default="debug", choices=["minimal", "debug"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def infer_grid_shape(z_grid: np.ndarray) -> tuple[int, int]:
    if z_grid.ndim != 3:
        raise ValueError(f"Expected z_grid to have shape [H,W,D], got {z_grid.shape}")
    return int(z_grid.shape[0]), int(z_grid.shape[1])


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else WSI_CF_ROOT / path


def resolve_existing_repo_file(path_value: str | Path) -> Path | None:
    text = str(path_value).strip()
    if not text:
        return None
    path = resolve_repo_path(text)
    if not path.exists() or not path.is_file():
        return None
    return path


def pad_image_edge(img: Image.Image, *, pad_px: int) -> Image.Image:
    if int(pad_px) <= 0:
        return img.convert("RGB")
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    padded = np.pad(
        arr,
        ((int(pad_px), int(pad_px)), (int(pad_px), int(pad_px)), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded)


def infer_coord_step(coords: np.ndarray, *, fallback: int) -> int:
    candidates: list[int] = []
    arr = np.asarray(coords, dtype=np.int64)
    for axis in (0, 1):
        values = np.unique(arr[:, axis])
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    return int(min(candidates)) if candidates else int(fallback)


def aggregate_pt_features_to_supercells(
    features: np.ndarray,
    coords: np.ndarray,
    *,
    tile_size_level0: int,
    block_ratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = int(block_ratio)
    if ratio <= 1:
        return np.asarray(features, dtype=np.float32), np.asarray(coords, dtype=np.int64)
    super_members: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(np.asarray(coords, dtype=np.int64).tolist()):
        raw_gx = int(round(int(x) / float(tile_size_level0)))
        raw_gy = int(round(int(y) / float(tile_size_level0)))
        cell = (raw_gx // ratio, raw_gy // ratio)
        super_members.setdefault(cell, []).append(int(idx))
    ordered_cells = sorted(super_members.keys(), key=lambda cell: (int(cell[1]), int(cell[0])))
    agg_features = np.zeros((len(ordered_cells), int(features.shape[1])), dtype=np.float32)
    agg_coords = np.zeros((len(ordered_cells), 2), dtype=np.int64)
    agg_tile = int(tile_size_level0) * ratio
    for out_idx, cell in enumerate(ordered_cells):
        member_idx = super_members[cell]
        agg_features[out_idx] = np.asarray(features[member_idx], dtype=np.float32).mean(axis=0)
        agg_coords[out_idx] = np.asarray([int(cell[0]) * agg_tile, int(cell[1]) * agg_tile], dtype=np.int64)
    return agg_features, agg_coords


def build_padded_feature_grid(
    *,
    source_zgrid: np.ndarray,
    row: RegionBankRow,
    halo_cells: int,
    level0_step_px: int,
) -> tuple[np.ndarray, dict[str, object]]:
    halo = int(halo_cells)
    if halo <= 0:
        return np.asarray(source_zgrid, dtype=np.float32), {"feature_context": "none"}
    fallback = np.pad(
        np.asarray(source_zgrid, dtype=np.float32),
        ((halo, halo), (halo, halo), (0, 0)),
        mode="edge",
    )
    feature_path = resolve_existing_repo_file(str(row.canonical_h5_path))
    if feature_path is None:
        return fallback, {"feature_context": "edge_pad", "reason": "missing_canonical_h5_path"}

    if feature_path.suffix.lower() in {".pt", ".pth"}:
        coords_path = resolve_existing_repo_file(str(getattr(row, "coords_path", "")))
        if coords_path is None:
            return fallback, {
                "feature_context": "edge_pad",
                "reason": "missing_coords_path_for_pt_features",
                "canonical_h5_path": str(feature_path),
            }
        loaded = torch.load(feature_path, map_location="cpu")
        if not isinstance(loaded, torch.Tensor):
            raise TypeError(f"Unexpected PT feature payload in {feature_path}: {type(loaded)}")
        features = loaded.detach().cpu().float().numpy().astype(np.float32, copy=False)
        with h5py.File(coords_path, "r") as handle:
            coords = np.asarray(handle["coords"], dtype=np.int64)
        raw_step = int(getattr(row, "raw_coord_tile_size_level0", 0)) or infer_coord_step(coords, fallback=int(level0_step_px))
        ratio = int(getattr(row, "clam_aggregate_ratio", 1) or 1)
        if ratio > 1:
            features, coords = aggregate_pt_features_to_supercells(
                features,
                coords,
                tile_size_level0=int(raw_step),
                block_ratio=int(ratio),
            )
        feature_context = "whole_slide_pt"
    else:
        with h5py.File(feature_path, "r") as handle:
            features = np.asarray(handle["features"], dtype=np.float32)
            coords = np.asarray(handle["coords"], dtype=np.int64)
        feature_context = "whole_slide_h5"
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if int(features.shape[0]) != int(coords.shape[0]):
        raise ValueError(f"{feature_path}: feature/coord count mismatch {features.shape[0]} != {coords.shape[0]}")
    step = infer_coord_step(coords, fallback=int(level0_step_px))
    coord_to_index = {(int(x), int(y)): int(idx) for idx, (x, y) in enumerate(coords)}

    padded = fallback.copy()
    missing = 0
    grid_h, grid_w = source_zgrid.shape[:2]
    for py in range(int(grid_h) + 2 * halo):
        for px in range(int(grid_w) + 2 * halo):
            local_x = int(px) - halo
            local_y = int(py) - halo
            coord = (
                int(row.region_x) + int(local_x) * int(step),
                int(row.region_y) + int(local_y) * int(step),
            )
            idx = coord_to_index.get(coord)
            if idx is None:
                missing += 1
                continue
            padded[py, px, :] = features[idx]
    return padded, {
        "feature_context": feature_context,
        "canonical_h5_path": str(feature_path),
        "coords_path": str(getattr(row, "coords_path", "")),
        "level0_step_px": int(step),
        "missing_context_cells": int(missing),
    }


def build_padded_runtime_region(
    *,
    source_img: Image.Image,
    source_zgrid: np.ndarray,
    row: RegionBankRow,
    halo_cells: int,
    target_magnification: float,
) -> tuple[Image.Image, np.ndarray, dict[str, object]]:
    halo = max(0, int(halo_cells))
    grid_step = int(row.grid_step_px)
    pad_px = int(halo * grid_step)
    if halo <= 0:
        return source_img, np.asarray(source_zgrid, dtype=np.float32), {
            "enabled": False,
            "halo_cells": 0,
            "pad_px": 0,
            "target_cell_shift": [0, 0],
            "final_crop_box": [0, 0, int(source_img.size[0]), int(source_img.size[1])],
        }

    padded_img = pad_image_edge(source_img, pad_px=pad_px)
    image_context = "edge_pad"
    level0_step_px = int(grid_step)
    slide_path = resolve_existing_repo_file(str(row.slide_path))
    if slide_path is not None:
        slide = None
        try:
            slide = open_slide(slide_path)
            _, crop_w0, _ = read_region_rgb_at_magnification(
                slide,
                x0=int(row.region_x),
                y0=int(row.region_y),
                out_w=int(grid_step),
                out_h=int(grid_step),
                target_magnification=float(target_magnification),
            )
            level0_step_px = int(crop_w0)
            padded_img, _, _ = read_region_rgb_at_magnification(
                slide,
                x0=int(row.region_x) - halo * int(level0_step_px),
                y0=int(row.region_y) - halo * int(level0_step_px),
                out_w=int(source_img.size[0]) + 2 * pad_px,
                out_h=int(source_img.size[1]) + 2 * pad_px,
                target_magnification=float(target_magnification),
            )
            image_context = "whole_slide_svs"
        except Exception as exc:
            warnings.warn(
                f"Could not open slide context {slide_path}; using edge-padded image context instead: {exc}",
                RuntimeWarning,
            )
        finally:
            if slide is not None:
                slide.close()

    padded_zgrid, feature_meta = build_padded_feature_grid(
        source_zgrid=np.asarray(source_zgrid, dtype=np.float32),
        row=row,
        halo_cells=halo,
        level0_step_px=level0_step_px,
    )
    meta = {
        "enabled": True,
        "mode": EDIT_SUPPORT_PADDED_CENTER_2X2,
        "halo_cells": int(halo),
        "pad_px": int(pad_px),
        "target_cell_shift": [int(halo), int(halo)],
        "final_crop_box": [
            int(pad_px),
            int(pad_px),
            int(pad_px) + int(source_img.size[0]),
            int(pad_px) + int(source_img.size[1]),
        ],
        "image_context": image_context,
        "slide_path": str(slide_path) if slide_path is not None else "",
        **feature_meta,
    }
    return padded_img, padded_zgrid, meta


def shift_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], *, dx: int, dy: int) -> list[tuple[int, int]]:
    return [(int(gx) + int(dx), int(gy) + int(dy)) for gx, gy in cells]


def build_image_first_region_row(
    *,
    region_image: Path,
    request_region_id: str,
    out_dir: Path,
    grid_step_px: int,
    device: torch.device,
) -> RegionBankRow:
    """Encode a standalone region image and expose it through the RegionBankRow interface."""
    source_img = load_image(str(region_image))
    width, height = source_img.size
    if width % int(grid_step_px) != 0 or height % int(grid_step_px) != 0:
        raise ValueError(
            f"Image-first progressive editing requires image dimensions divisible by grid_step_px={grid_step_px}; "
            f"got {width}x{height} for {region_image}"
        )
    source_dir = out_dir / "_image_first_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    image_path = source_dir / "region.png"
    zgrid_path = source_dir / "region_zgrid.npy"
    save_png(source_img, image_path)
    if not zgrid_path.exists():
        uni_model, uni_transform = load_uni2(device)
        z_grid = build_uni_grid_from_image(
            source_img,
            uni_model=uni_model,
            uni_transform=uni_transform,
            grid_step_px=int(grid_step_px),
            device=device,
            out_dtype=torch.float32,
        )
        np.save(zgrid_path, z_grid.detach().cpu().numpy().astype(np.float32))
        del uni_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    z_grid_np = np.asarray(np.load(zgrid_path), dtype=np.float32)
    return RegionBankRow(
        region_id=str(request_region_id),
        split="showcase",
        label=1,
        hpv_status="HPV+",
        case_id=Path(region_image).stem,
        slide_key=Path(region_image).stem,
        slide_path=str(region_image),
        canonical_h5_path="",
        region_x=0,
        region_y=0,
        region_w=int(width),
        region_h=int(height),
        grid_step_px=int(grid_step_px),
        feature_dim=int(z_grid_np.shape[-1]),
        tissue_score=1.0,
        seed=0,
        image_path=str(image_path),
        feature_grid_path=str(zgrid_path),
        cell_preview_path="",
        region_dir=str(source_dir),
    )


def build_commit_alpha_mask(
    *,
    width: int,
    height: int,
    grid_step_px: int,
    cells_local: list[tuple[int, int]] | None,
    feather_px: int,
    halo_cells: int = 0,
    halo_alpha: float = 0.0,
) -> np.ndarray:
    if cells_local is None:
        return np.ones((int(height), int(width), 1), dtype=np.float32)
    mask = np.zeros((int(height), int(width)), dtype=np.float32)
    cell_set = {(int(lx), int(ly)) for lx, ly in cells_local}
    halo_radius = int(halo_cells)
    halo_value = float(halo_alpha)
    if halo_radius > 0 and halo_value > 0.0:
        grid_w = int(np.ceil(float(width) / float(grid_step_px)))
        grid_h = int(np.ceil(float(height) / float(grid_step_px)))
        for lx, ly in cell_set:
            for hy in range(max(0, int(ly) - halo_radius), min(grid_h, int(ly) + halo_radius + 1)):
                for hx in range(max(0, int(lx) - halo_radius), min(grid_w, int(lx) + halo_radius + 1)):
                    if (int(hx), int(hy)) not in cell_set:
                        x0 = int(hx) * int(grid_step_px)
                        y0 = int(hy) * int(grid_step_px)
                        x1 = min(int(width), x0 + int(grid_step_px))
                        y1 = min(int(height), y0 + int(grid_step_px))
                        if x1 > x0 and y1 > y0:
                            mask[y0:y1, x0:x1] = max(float(mask[y0:y1, x0:x1].max()), halo_value)
    for lx, ly in cell_set:
        x0 = int(lx) * int(grid_step_px)
        y0 = int(ly) * int(grid_step_px)
        x1 = min(int(width), x0 + int(grid_step_px))
        y1 = min(int(height), y0 + int(grid_step_px))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1.0
    if int(feather_px) > 0 and np.any(mask > 0):
        pil = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=int(feather_px)))
        mask = np.asarray(blurred, dtype=np.float32) / 255.0
        max_value = float(mask.max())
        if max_value > 0.0:
            mask = np.clip(mask / max_value, 0.0, 1.0)
    return mask[:, :, None].astype(np.float32, copy=False)


def commit_window(
    *,
    current_canvas: np.ndarray,
    steered_img: Image.Image,
    left: int,
    top: int,
    alpha_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = np.asarray(current_canvas, dtype=np.float32).copy()
    img = np.asarray(steered_img, dtype=np.float32) / 255.0
    height, width = img.shape[:2]
    alpha = np.asarray(alpha_mask, dtype=np.float32)
    if alpha.shape[:2] != img.shape[:2]:
        raise ValueError(f"Commit alpha mask shape {alpha.shape} does not match steered image shape {img.shape}")
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    alpha = np.clip(alpha, 0.0, 1.0)
    dst_x0 = int(left)
    dst_y0 = int(top)
    dst_x1 = int(left) + int(width)
    dst_y1 = int(top) + int(height)
    current_crop = out[dst_y0:dst_y1, dst_x0:dst_x1]
    out[dst_y0:dst_y1, dst_x0:dst_x1] = current_crop * (1.0 - alpha) + img * alpha
    return out, (dst_x0, dst_y0, dst_x1, dst_y1)


def build_valid_feature_alpha_mask(
    *,
    valid_feature_mask: np.ndarray,
    width: int,
    height: int,
    grid_step_px: int,
    feather_px: int,
) -> np.ndarray:
    mask = np.asarray(valid_feature_mask)
    expected_shape = (
        int(np.ceil(float(height) / float(grid_step_px))),
        int(np.ceil(float(width) / float(grid_step_px))),
    )
    if mask.shape != expected_shape:
        raise ValueError(f"valid_feature_mask shape {mask.shape} does not match expected grid {expected_shape}")
    if int(feather_px) < 0:
        raise ValueError("--invalid-feature-feather-px must be >= 0")

    alpha = np.zeros((int(height), int(width)), dtype=np.uint8)
    for gy, gx in np.argwhere(mask > 0):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(int(width), x0 + int(grid_step_px))
        y1 = min(int(height), y0 + int(grid_step_px))
        alpha[y0:y1, x0:x1] = 255
    alpha_img = Image.fromarray(alpha)
    if int(feather_px) > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=int(feather_px)))
    return (np.asarray(alpha_img, dtype=np.float32) / 255.0)[:, :, None]


def composite_invalid_feature_cells(
    *,
    generated_img: Image.Image,
    source_img: Image.Image,
    valid_feature_mask: np.ndarray,
    grid_step_px: int,
    feather_px: int,
) -> tuple[Image.Image, np.ndarray]:
    generated = np.asarray(generated_img.convert("RGB"), dtype=np.float32)
    source = np.asarray(source_img.convert("RGB"), dtype=np.float32)
    if generated.shape != source.shape:
        raise ValueError(f"Generated image shape {generated.shape} does not match source shape {source.shape}")
    alpha = build_valid_feature_alpha_mask(
        valid_feature_mask=valid_feature_mask,
        width=int(generated_img.size[0]),
        height=int(generated_img.size[1]),
        grid_step_px=int(grid_step_px),
        feather_px=int(feather_px),
    )
    composited = generated * alpha + source * (1.0 - alpha)
    return Image.fromarray(np.clip(np.rint(composited), 0, 255).astype(np.uint8)), alpha


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def load_concept_targets(
    *,
    concepts_json: Path,
    representative_tiles_csv: Path | None,
    class_label: str,
    ranking_method: str,
    target_stat: str,
    target_top_k: int,
    max_concepts: int,
) -> tuple[list[int], dict[int, float], dict[str, object]]:
    payload = json.loads(concepts_json.read_text())
    concepts = list(payload.get("concepts", []))
    if class_label:
        concepts = [row for row in concepts if str(row.get("class_label", "")) == str(class_label)]
    concepts.sort(key=lambda row: (int(row.get("concept_rank", 10**9)), -float(row.get("final_score", 0.0)), int(row["latent_idx"])))
    if int(max_concepts) > 0:
        concepts = concepts[: int(max_concepts)]
    if not concepts:
        raise ValueError(f"No concepts found in {concepts_json} for class_label={class_label!r}")
    latent_ids = [int(row["latent_idx"]) for row in concepts]
    values_by_latent: dict[int, list[float]] = {latent: [] for latent in latent_ids}
    if representative_tiles_csv is not None and representative_tiles_csv.exists():
        rows_by_latent: dict[int, list[dict[str, str]]] = {latent: [] for latent in latent_ids}
        for row in read_csv_rows(representative_tiles_csv):
            latent = int(row.get("latent_idx", -1))
            if latent not in rows_by_latent:
                continue
            if str(row.get("ranking_method", "")) != str(ranking_method):
                continue
            rows_by_latent[latent].append(row)
        for latent, rows in rows_by_latent.items():
            rows.sort(key=lambda row: int(row.get("tile_rank", 10**9)))
            if int(target_top_k) > 0:
                rows = rows[: int(target_top_k)]
            for row in rows:
                activation = row.get("activation", "")
                if str(activation).strip():
                    values_by_latent[latent].append(float(activation))
    target_values: dict[int, float] = {}
    for concept in concepts:
        latent = int(concept["latent_idx"])
        vals = np.asarray(values_by_latent.get(latent, []), dtype=np.float32)
        if vals.size:
            if target_stat == "median":
                target = float(np.median(vals))
            elif target_stat == "mean":
                target = float(np.mean(vals))
            elif target_stat == "q75":
                target = float(np.percentile(vals, 75.0))
            else:
                target = float(np.max(vals))
        else:
            # Fallback: association summaries from fraction metrics can be tiny,
            # but this keeps the run defined if representative rows are absent.
            target = float(concept.get("mean_class", concept.get("top_activation", 1.0)))
        target_values[latent] = target
    meta = {
        "concepts_json": str(concepts_json),
        "representative_tiles_csv": "" if representative_tiles_csv is None else str(representative_tiles_csv),
        "class_label": class_label or str(payload.get("class_label", "")),
        "ranking_method": str(ranking_method),
        "target_stat": str(target_stat),
        "target_top_k": int(target_top_k),
        "latent_ids": latent_ids,
        "target_tile_counts": {str(k): int(len(v)) for k, v in values_by_latent.items()},
        "target_values": {str(k): float(v) for k, v in target_values.items()},
    }
    return latent_ids, target_values, meta


def _read_h5_feature(path: Path, tile_index: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{path}: missing dataset 'features'")
        feats = handle["features"]
        if feats.ndim == 3 and feats.shape[0] == 1:
            arr = np.asarray(feats[0, int(tile_index)], dtype=np.float32)
        elif feats.ndim == 2:
            arr = np.asarray(feats[int(tile_index)], dtype=np.float32)
        else:
            raise ValueError(f"{path}: unsupported features shape {tuple(feats.shape)}")
    return arr.astype(np.float32, copy=False)


def representative_rows_for_latent(
    rows_by_latent_method: dict[tuple[int, str], list[dict[str, str]]],
    *,
    latent: int,
    ranking_method: str,
) -> tuple[list[dict[str, str]], str]:
    preferred = list(rows_by_latent_method.get((int(latent), str(ranking_method)), []))
    if preferred:
        return preferred, str(ranking_method)
    if str(ranking_method) != "activation":
        fallback = list(rows_by_latent_method.get((int(latent), "activation"), []))
        if fallback:
            return fallback, "activation"
    available_methods = sorted(
        method
        for candidate_latent, method in rows_by_latent_method
        if int(candidate_latent) == int(latent)
    )
    for method in available_methods:
        rows = list(rows_by_latent_method.get((int(latent), str(method)), []))
        if rows:
            return rows, str(method)
    return [], str(ranking_method)


@torch.no_grad()
def load_concept_prototype_vector(
    *,
    sae_model: torch.nn.Module,
    concepts_json: Path,
    representative_tiles_csv: Path | None,
    class_label: str,
    ranking_method: str,
    target_stat: str,
    target_top_k: int,
    max_concepts: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    if representative_tiles_csv is None or not representative_tiles_csv.exists():
        raise FileNotFoundError(
            "Full concept-prototype steering requires --representative-tiles-csv with source H5 tile references."
        )
    payload = json.loads(concepts_json.read_text())
    concepts = list(payload.get("concepts", []))
    if class_label:
        concepts = [row for row in concepts if str(row.get("class_label", "")) == str(class_label)]
    concepts.sort(key=lambda row: (int(row.get("concept_rank", 10**9)), -float(row.get("final_score", 0.0)), int(row["latent_idx"])))
    if int(max_concepts) > 0:
        concepts = concepts[: int(max_concepts)]
    if not concepts:
        raise ValueError(f"No concepts found in {concepts_json} for class_label={class_label!r}")

    latent_ids = [int(row["latent_idx"]) for row in concepts]
    rows_by_latent_method: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in read_csv_rows(representative_tiles_csv):
        latent = int(row.get("latent_idx", -1))
        if latent not in latent_ids:
            continue
        method = str(row.get("ranking_method", ""))
        rows_by_latent_method.setdefault((latent, method), []).append(row)

    device = next(sae_model.parameters()).device
    concept_vectors: list[torch.Tensor] = []
    per_concept_meta: list[dict[str, object]] = []
    for concept in concepts:
        latent = int(concept["latent_idx"])
        rows, resolved_ranking_method = representative_rows_for_latent(
            rows_by_latent_method,
            latent=latent,
            ranking_method=str(ranking_method),
        )
        rows.sort(key=lambda row: int(row.get("tile_rank", 10**9)))
        if int(target_top_k) > 0:
            rows = rows[: int(target_top_k)]
        if not rows:
            raise ValueError(
                f"No representative rows for concept latent_idx={latent}, ranking_method={ranking_method!r}; "
                "cannot build full-code concept prototype."
            )

        features = np.stack([_read_h5_feature(Path(str(row["h5_path"])), int(row["tile_index"])) for row in rows], axis=0)
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        z = sae_encode_features(sae_model, x).float()
        if target_stat == "median":
            proto = torch.median(z, dim=0).values
        elif target_stat == "mean":
            proto = torch.mean(z, dim=0)
        elif target_stat == "q75":
            proto = torch.quantile(z, q=0.75, dim=0)
        else:
            proto = torch.max(z, dim=0).values
        concept_vectors.append(proto)
        per_concept_meta.append(
            {
                "concept_rank": int(concept.get("concept_rank", len(per_concept_meta) + 1)),
                "latent_idx": int(latent),
                "source_latent_idx": int(concept.get("source_latent_idx", latent)),
                "prototype_basis": "representative_tiles_encoded_with_runtime_sae",
                "requested_ranking_method": str(ranking_method),
                "resolved_ranking_method": str(resolved_ranking_method),
                "representative_tile_count": int(len(rows)),
                "prototype_norm": float(proto.norm().detach().cpu()),
                "prototype_target_activation_at_latent": float(proto[int(latent)].detach().cpu()),
            }
        )

    stacked = torch.stack(concept_vectors, dim=0)
    # Multiple concept cards in one run become a single target SAE-code
    # prototype. Individual-concept scripts pass one concept at a time.
    prototype = torch.mean(stacked, dim=0)
    meta = {
        "concepts_json": str(concepts_json),
        "representative_tiles_csv": str(representative_tiles_csv),
        "class_label": class_label or str(payload.get("class_label", "")),
        "ranking_method": str(ranking_method),
        "ranking_method_fallback_order": ["requested", "activation", "first_available"],
        "target_stat": str(target_stat),
        "target_top_k": int(target_top_k),
        "steering_mode": "prototype_vector",
        "prototype_basis": "representative_tiles_encoded_with_runtime_sae",
        "latent_ids": latent_ids,
        "n_concepts": int(len(concepts)),
        "prototype_aggregation": "mean_across_concepts",
        "prototype_norm": float(prototype.norm().detach().cpu()),
        "concept_prototypes": per_concept_meta,
    }
    return prototype.detach(), meta


@torch.no_grad()
def edit_uni_z_grid_with_concept_targets(
    *,
    sae_model: torch.nn.Module,
    z_grid: torch.Tensor,
    latent_target_values: dict[int, float],
    target_strength: float,
    tile_mask: np.ndarray,
    blend: float,
) -> torch.Tensor:
    if z_grid.dim() != 3:
        raise ValueError(f"Expected local z_grid [Gh,Gw,D], got {tuple(z_grid.shape)}")
    gh, gw, d = z_grid.shape
    device = next(sae_model.parameters()).device
    x = z_grid.reshape(gh * gw, d).to(device=device, dtype=torch.float32)
    z_lat = sae_encode_features(sae_model, x)
    mask = np.asarray(tile_mask, dtype=np.float32)
    if mask.shape != (gh, gw):
        raise ValueError(f"tile_mask must be shape {(gh, gw)}, got {mask.shape}")
    sel = np.flatnonzero(mask.reshape(-1) > 0.0)
    if sel.size == 0:
        return z_grid
    sel_t = torch.as_tensor(sel, device=z_lat.device, dtype=torch.long)
    z_edit = z_lat.clone()
    for latent_idx, target_value in latent_target_values.items():
        latent = int(latent_idx)
        if latent < 0 or latent >= z_edit.shape[1]:
            raise ValueError(f"Concept latent_idx={latent} outside SAE latent dim={z_edit.shape[1]}")
        cur = z_edit[sel_t, latent]
        tgt = torch.full_like(cur, float(target_value))
        z_edit[sel_t, latent] = (1.0 - float(target_strength)) * cur + float(target_strength) * tgt
    x_rec = sae_decode_latents(sae_model, z_edit)
    w = torch.from_numpy(mask.reshape(gh * gw, 1)).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    x_new = x * (1.0 - float(blend) * w) + x_rec * (float(blend) * w)
    return x_new.reshape(gh, gw, d).to(device=z_grid.device, dtype=z_grid.dtype)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)
    args = apply_edit_policy(args, parser=parser, argv=raw_argv, root=WSI_CF_ROOT)
    if int(args.window_stride_cells) <= 0:
        raise ValueError("--window-stride-cells must be > 0")
    if not (0.0 <= float(args.steer_context_halo_weight) <= 1.0):
        raise ValueError("--steer-context-halo-weight must be in [0, 1]")
    if int(args.steer_context_halo_radius_cells) < 0:
        raise ValueError("--steer-context-halo-radius-cells must be >= 0")
    if int(args.commit_halo_cells) < 0:
        raise ValueError("--commit-halo-cells must be >= 0")
    if not (0.0 <= float(args.commit_halo_alpha) <= 1.0):
        raise ValueError("--commit-halo-alpha must be in [0, 1]")
    if int(args.invalid_feature_feather_px) < 0:
        raise ValueError("--invalid-feature-feather-px must be >= 0")
    if int(args.target_cell_feather_px) < 0:
        raise ValueError("--target-cell-feather-px must be >= 0")
    args.sae_ckpt, args.sae_cfg = resolve_sae_paths(args.sae_variant, args.sae_ckpt, args.sae_cfg)
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    set_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "cli_args": _serialize_args(args),
        "argv": raw_argv,
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + raw_argv)),
        "cwd": str(Path.cwd()),
    }
    write_json(args.out_dir / "experiment_args.json", args_payload)

    edit_requests = load_progressive_edit_manifest(args.edit_manifest)
    if int(args.max_runs) > 0:
        edit_requests = edit_requests[: int(args.max_runs)]
    if not edit_requests:
        raise ValueError("No edit requests found in edit manifest")
    run_id_counts: dict[str, int] = {}
    for request in edit_requests:
        run_id_counts[str(request.run_id)] = run_id_counts.get(str(request.run_id), 0) + 1
    duplicate_run_ids = {run_id for run_id, count in run_id_counts.items() if count > 1}
    if duplicate_run_ids:
        raise ValueError(f"Duplicate run_id values in edit manifest: {sorted(duplicate_run_ids)}")

    if args.region_bank_csv is not None:
        region_rows = parse_region_bank_csv(args.region_bank_csv)
    else:
        first_region_id = str(edit_requests[0].region_id)
        region_rows = [
            build_image_first_region_row(
                region_image=args.region_image,
                request_region_id=first_region_id,
                out_dir=args.out_dir,
                grid_step_px=256,
                device=device,
            )
        ]
    region_by_id = {str(row.region_id): row for row in region_rows}

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    concept_latent_targets: dict[int, float] | None = None
    concept_prototype_vector: torch.Tensor | None = None
    concept_meta: dict[str, object] | None = None
    chosen_latent = -1
    if args.concepts_json is not None:
        if str(args.concept_steering_mode) == "prototype_vector":
            concept_prototype_vector, concept_meta = load_concept_prototype_vector(
                sae_model=sae_model,
                concepts_json=args.concepts_json,
                representative_tiles_csv=args.representative_tiles_csv,
                class_label=str(args.concept_class_label),
                ranking_method=str(args.concept_ranking_method),
                target_stat=str(args.concept_target_stat),
                target_top_k=int(args.concept_target_top_k),
                max_concepts=int(args.max_concepts),
            )
        else:
            _, concept_latent_targets, concept_meta = load_concept_targets(
                concepts_json=args.concepts_json,
                representative_tiles_csv=args.representative_tiles_csv,
                class_label=str(args.concept_class_label),
                ranking_method=str(args.concept_ranking_method),
                target_stat=str(args.concept_target_stat),
                target_top_k=int(args.concept_target_top_k),
                max_concepts=int(args.max_concepts),
            )
            if concept_meta is not None:
                concept_meta["steering_mode"] = "latent_target"
    else:
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

    summary_rows: list[dict[str, object]] = []
    for request in edit_requests:
        row = region_by_id.get(str(request.region_id))
        if row is None:
            raise ValueError(f"Manifest references unknown region_id '{request.region_id}'")

        original_source_img = load_image(str(row.image_path))
        original_source_zgrid = np.asarray(np.load(str(row.feature_grid_path)), dtype=np.float32)
        original_grid_h, original_grid_w = infer_grid_shape(original_source_zgrid)
        padding_meta: dict[str, object] = {
            "enabled": False,
            "halo_cells": 0,
            "pad_px": 0,
            "target_cell_shift": [0, 0],
            "final_crop_box": [0, 0, int(original_source_img.size[0]), int(original_source_img.size[1])],
        }
        planning_edit_support = str(args.edit_support)
        target_shift_x = 0
        target_shift_y = 0
        source_img = original_source_img
        source_zgrid = original_source_zgrid
        if str(args.edit_support) == EDIT_SUPPORT_PADDED_CENTER_2X2:
            source_img, source_zgrid, padding_meta = build_padded_runtime_region(
                source_img=original_source_img,
                source_zgrid=original_source_zgrid,
                row=row,
                halo_cells=max(1, int(args.context_halo_cells)),
                target_magnification=float(args.target_magnification),
            )
            target_shift_x = int(padding_meta["target_cell_shift"][0])  # type: ignore[index]
            target_shift_y = int(padding_meta["target_cell_shift"][1])  # type: ignore[index]
            planning_edit_support = EDIT_SUPPORT_CENTER_2X2
        grid_h, grid_w = infer_grid_shape(source_zgrid)
        run_dir = args.out_dir / str(request.run_id)
        final_out_path = run_dir / "generated.png"
        if bool(args.skip_existing) and final_out_path.exists():
            summary_rows.append(
                {
                    "run_id": str(request.run_id),
                    "region_id": str(request.region_id),
                    "output_path": str(final_out_path),
                    "num_targets": int(len(request.target_cells)),
                }
            )
            continue

        shifted_request_cells = shift_cells(
            list(request.target_cells),
            dx=int(target_shift_x),
            dy=int(target_shift_y),
        )
        supported_targets, unsupported_targets = split_cells_by_edit_support(
            target_cells=shifted_request_cells,
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            window_grid_side=4,
            stride_cells=int(args.window_stride_cells),
            grid_step_px=int(row.grid_step_px),
            edit_support=str(planning_edit_support),
        )
        if unsupported_targets and not supported_targets:
            raise RuntimeError(
                f"All requested target cells are outside edit_support={args.edit_support}: {list(unsupported_targets)}"
            )
        active_target_cells = tuple(supported_targets)
        active_target_cells_original = tuple(
            (int(gx) - int(target_shift_x), int(gy) - int(target_shift_y))
            for gx, gy in active_target_cells
        )
        unsupported_targets_original = tuple(
            (int(gx) - int(target_shift_x), int(gy) - int(target_shift_y))
            for gx, gy in unsupported_targets
        )
        planned_steps = plan_progressive_steps(
            target_cells=list(active_target_cells),
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            window_grid_side=4,
            stride_cells=int(args.window_stride_cells),
            grid_step_px=int(row.grid_step_px),
            edit_support=str(planning_edit_support),
            selection_mode=str(args.window_selection_mode),
        )
        current_canvas = np.asarray(source_img, dtype=np.float32) / 255.0
        current_zgrid = np.asarray(source_zgrid, dtype=np.float32).copy()
        target_metadata = dict(request.metadata)
        target_metadata["padding_context"] = padding_meta
        if unsupported_targets_original:
            target_metadata["dropped_unsupported_target_cells"] = [
                {"gx": int(gx), "gy": int(gy)} for gx, gy in unsupported_targets_original
            ]
            target_metadata["original_target_cell_count"] = int(len(request.target_cells))
        state = make_initial_progressive_state(target_cells=list(active_target_cells))
        run_dir.mkdir(parents=True, exist_ok=True)
        save_png(original_source_img, run_dir / "source_region_actual.png")
        if bool(padding_meta.get("enabled")):
            save_png(source_img, run_dir / "padded_source_region_actual.png")

        if str(args.output_mode) == "debug":
            save_png(
                draw_cells_overlay(original_source_img, cells=list(active_target_cells_original), grid_step_px=int(row.grid_step_px)),
                run_dir / "source_targets_overlay.png",
            )
            if bool(padding_meta.get("enabled")):
                save_png(
                    draw_cells_overlay(source_img, cells=list(active_target_cells), grid_step_px=int(row.grid_step_px)),
                    run_dir / "padded_source_targets_overlay.png",
                )

        step_records: list[dict[str, object]] = []
        for step in planned_steps:
            window = step.window
            window_px_w = int(window.grid_w) * int(row.grid_step_px)
            window_px_h = int(window.grid_h) * int(row.grid_step_px)
            local_source_img = Image.fromarray((np.clip(current_canvas, 0.0, 1.0) * 255.0).astype(np.uint8)).crop(
                (
                    int(window.left),
                    int(window.top),
                    int(window.left) + int(window_px_w),
                    int(window.top) + int(window_px_h),
                )
            ).convert("RGB")
            gx0 = int(window.gx0)
            gy0 = int(window.gy0)
            local_base_zgrid = np.asarray(current_zgrid[gy0 : gy0 + 4, gx0 : gx0 + 4, :], dtype=np.float32)
            local_base_t = torch.from_numpy(local_base_zgrid).to(device=device, dtype=torch.float32)
            local_edit_t = local_base_t.clone()

            local_edit_cells = window_local_cells(window=window, global_cells=step.edit_cells_global)
            allowed_global_cells = local_edit_support_global_cells(
                window,
                grid_w=int(grid_w),
                grid_h=int(grid_h),
                edit_support=str(planning_edit_support),
            )
            allowed_local_cells = window_local_cells(
                window=window,
                global_cells=allowed_global_cells,
            )
            bad_local_cells = [cell for cell in local_edit_cells if cell not in set(allowed_local_cells)]
            if bad_local_cells:
                raise RuntimeError(
                    f"Planned local edit cells are outside edit_support={args.edit_support}, got {bad_local_cells} "
                    f"for window {window.window_id}. Center support is {CENTER_2X2_LOCAL_CELLS}."
                )
            local_steer_cells = list(allowed_local_cells) if bool(args.steer_full_support) else list(local_edit_cells)
            tile_mask = np.zeros(local_base_zgrid.shape[:2], dtype=np.float32)
            halo_weight = float(args.steer_context_halo_weight)
            halo_radius = int(args.steer_context_halo_radius_cells)
            if halo_weight > 0.0 and halo_radius > 0:
                for lx, ly in local_steer_cells:
                    for hy in range(max(0, int(ly) - halo_radius), min(tile_mask.shape[0], int(ly) + halo_radius + 1)):
                        for hx in range(max(0, int(lx) - halo_radius), min(tile_mask.shape[1], int(lx) + halo_radius + 1)):
                            tile_mask[int(hy), int(hx)] = max(float(tile_mask[int(hy), int(hx)]), halo_weight)
            for lx, ly in local_steer_cells:
                tile_mask[int(ly), int(lx)] = 1.0
            nonzero_tile_weights = [
                {"gx": int(lx), "gy": int(ly), "weight": float(tile_mask[int(ly), int(lx)])}
                for ly, lx in np.argwhere(tile_mask > 0.0)
            ]
            if concept_prototype_vector is not None:
                local_edit_t, _ = edit_uni_z_grid_with_sae(
                    sae_model=sae_model,
                    z_grid=local_edit_t,
                    target_latent_vector=concept_prototype_vector,
                    target_latent_vector_strength=float(args.prototype_strength),
                    tile_mask=tile_mask,
                    blend=float(args.steer_blend),
                    keep_non_selected=True,
                    return_debug=False,
                )
            elif concept_latent_targets is not None:
                local_edit_t = edit_uni_z_grid_with_concept_targets(
                    sae_model=sae_model,
                    z_grid=local_edit_t,
                    latent_target_values=concept_latent_targets,
                    target_strength=float(args.prototype_strength),
                    tile_mask=tile_mask,
                    blend=float(args.steer_blend),
                )
            else:
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

            source_np = np.asarray(local_source_img, dtype=np.float32) / 255.0
            source_img_t = torch.from_numpy(source_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
            preserve_source_latents = vae_encode_auto(
                pipeline.vae,
                source_img_t,
                use_tiled=False,
                tile_img=0,
                overlap_img=0,
            )
            preserve_edit_cells_global = list(allowed_global_cells) if bool(args.preserve_full_support) else list(step.edit_cells_global)
            preserve_map = build_history_aware_preserve_map(
                width=int(local_source_img.size[0]),
                height=int(local_source_img.size[1]),
                grid_step_px=int(row.grid_step_px),
                window=window,
                edit_cells_global=preserve_edit_cells_global,
                visited_cells_global=list(state.visited_cells),
                preserve_edit_strength=float(args.preserve_edit_strength),
                preserve_visited_strength=float(args.preserve_visited_strength),
                preserve_fresh_context_strength=float(args.preserve_fresh_context_strength),
            ).to(device=device)

            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.seed) + int(step.step_index))
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
                    condition_alpha_end=float(args.mid_steer_alpha_end),
                    condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
                    out_h=int(local_source_img.size[1]),
                    out_w=int(local_source_img.size[0]),
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

            img_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            steered_img = Image.fromarray(img_np)
            if str(args.commit_mode) == "full_window":
                commit_cells_local = None
            elif str(args.commit_mode) == "support_cells":
                commit_cells_local = list(allowed_local_cells)
            else:
                commit_cells_local = list(local_edit_cells)
            commit_alpha = build_commit_alpha_mask(
                width=int(steered_img.size[0]),
                height=int(steered_img.size[1]),
                grid_step_px=int(row.grid_step_px),
                cells_local=commit_cells_local,
                feather_px=int(args.commit_feather_px),
                halo_cells=int(args.commit_halo_cells),
                halo_alpha=float(args.commit_halo_alpha),
            )
            current_canvas, commit_box = commit_window(
                current_canvas=current_canvas,
                steered_img=steered_img,
                left=int(window.left),
                top=int(window.top),
                alpha_mask=commit_alpha,
            )
            current_zgrid = update_full_zgrid_selected_cells(
                full_zgrid=current_zgrid,
                edited_local_zgrid=local_edit_t.detach().cpu().numpy().astype(np.float32),
                gx0=int(window.gx0),
                gy0=int(window.gy0),
                selected_cells=local_steer_cells,
            )
            step_record = {
                "step_index": int(step.step_index),
                "window_id": str(window.window_id),
                "row_index": int(window.row_index),
                "col_index": int(window.col_index),
                "gx0": int(window.gx0),
                "gy0": int(window.gy0),
                "left": int(window.left),
                "top": int(window.top),
                "edit_cells_global": [{"gx": int(gx), "gy": int(gy)} for gx, gy in step.edit_cells_global],
                "edit_cells_local": [{"gx": int(gx), "gy": int(gy)} for gx, gy in local_edit_cells],
                "steer_cells_global": [
                    {"gx": int(window.gx0) + int(lx), "gy": int(window.gy0) + int(ly)}
                    for lx, ly in local_steer_cells
                ],
                "steer_cells_local": [{"gx": int(gx), "gy": int(gy)} for gx, gy in local_steer_cells],
                "steer_tile_weights_local": nonzero_tile_weights,
                "visited_cells_local_before_step": [
                    {"gx": int(gx), "gy": int(gy)}
                    for gx, gy in window_local_cells(window=window, global_cells=state.visited_cells)
                ],
                "commit_bounds_global": {
                    "x0": int(commit_box[0]),
                    "y0": int(commit_box[1]),
                    "x1": int(commit_box[2]),
                    "y1": int(commit_box[3]),
                },
                "commit_mode": str(args.commit_mode),
                "commit_feather_px": int(args.commit_feather_px),
                "commit_halo_cells": int(args.commit_halo_cells),
                "commit_halo_alpha": float(args.commit_halo_alpha),
                "preserve_full_support": bool(args.preserve_full_support),
                "preserve_edit_cells_global": [
                    {"gx": int(gx), "gy": int(gy)} for gx, gy in sorted(preserve_edit_cells_global, key=lambda item: (int(item[1]), int(item[0])))
                ],
                "preserve_edit_cells_local": [
                    {"gx": int(gx), "gy": int(gy)}
                    for gx, gy in window_local_cells(window=window, global_cells=preserve_edit_cells_global)
                ],
                "commit_cells_local": (
                    []
                    if commit_cells_local is None
                    else [{"gx": int(gx), "gy": int(gy)} for gx, gy in commit_cells_local]
                ),
            }

            if str(args.output_mode) == "debug":
                step_dir = run_dir / "steps" / f"step_{int(step.step_index) + 1:02d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                save_png(local_source_img, step_dir / "source_window.png")
                save_png(draw_cells_overlay(local_source_img, cells=local_edit_cells, grid_step_px=int(row.grid_step_px)), step_dir / "selected_cells_overlay.png")
                save_png(
                    draw_step_region_overlay(
                        source_img,
                        window=window,
                        edit_cells_global=step.edit_cells_global,
                        support_cells_global=allowed_global_cells,
                        grid_step_px=int(row.grid_step_px),
                    ),
                    step_dir / "edit_area_on_region.png",
                )
                save_png(
                    draw_step_edit_area_zoom_4x4(
                        source_img,
                        window=window,
                        edit_cells_global=step.edit_cells_global,
                        support_cells_global=allowed_global_cells,
                        grid_step_px=int(row.grid_step_px),
                    ),
                    step_dir / "edit_area_zoom_4x4.png",
                )
                save_png(preserve_map_preview(preserve_map), step_dir / "preserve_map.png")
                save_png(steered_img, step_dir / "steered_window.png")
                save_png(Image.fromarray((np.clip(commit_alpha[:, :, 0], 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"), step_dir / "commit_mask.png")
                step_record["source_window_path"] = str(step_dir / "source_window.png")
                step_record["selected_overlay_path"] = str(step_dir / "selected_cells_overlay.png")
                step_record["region_edit_area_overlay_path"] = str(step_dir / "edit_area_on_region.png")
                step_record["edit_area_zoom_4x4_path"] = str(step_dir / "edit_area_zoom_4x4.png")
                step_record["preserve_map_path"] = str(step_dir / "preserve_map.png")
                step_record["steered_window_path"] = str(step_dir / "steered_window.png")
                step_record["commit_mask_path"] = str(step_dir / "commit_mask.png")

            step_records.append(step_record)
            state = advance_progressive_state(
                state,
                window=window,
                edit_cells_global=list(step.edit_cells_global),
            )

        runtime_final_img = Image.fromarray((np.clip(current_canvas, 0.0, 1.0) * 255.0).astype(np.uint8))
        crop_box = tuple(int(v) for v in padding_meta.get("final_crop_box", [0, 0, runtime_final_img.size[0], runtime_final_img.size[1]]))
        final_img = runtime_final_img.crop(crop_box) if bool(padding_meta.get("enabled")) else runtime_final_img
        invalid_feature_composite: dict[str, object] = {
            "enabled": bool(args.preserve_invalid_feature_cells),
            "feather_px": int(args.invalid_feature_feather_px),
        }
        if bool(args.preserve_invalid_feature_cells):
            raw_mask_path = target_metadata.get("valid_feature_mask_path")
            mask_path = Path(str(raw_mask_path)) if raw_mask_path else Path(row.feature_grid_path).with_name("valid_feature_mask.npy")
            if not mask_path.is_absolute():
                mask_path = WSI_CF_ROOT / mask_path
            if not mask_path.exists():
                raise FileNotFoundError(
                    "--preserve-invalid-feature-cells requires valid_feature_mask.npy; "
                    f"looked for {mask_path}"
                )
            valid_feature_mask = np.load(mask_path)
            final_img, final_valid_alpha = composite_invalid_feature_cells(
                generated_img=final_img,
                source_img=original_source_img,
                valid_feature_mask=valid_feature_mask,
                grid_step_px=int(row.grid_step_px),
                feather_px=int(args.invalid_feature_feather_px),
            )
            invalid_feature_composite.update(
                {
                    "valid_feature_mask_path": str(mask_path),
                    "valid_cell_count": int(np.count_nonzero(valid_feature_mask)),
                    "invalid_cell_count": int(valid_feature_mask.size - np.count_nonzero(valid_feature_mask)),
                }
            )
            if str(args.output_mode) == "debug":
                save_png(
                    Image.fromarray((np.clip(final_valid_alpha[:, :, 0], 0.0, 1.0) * 255.0).astype(np.uint8)),
                    run_dir / "final_valid_feature_alpha.png",
                )
        target_cell_composite: dict[str, object] = {
            "enabled": bool(args.preserve_outside_target_cells),
            "feather_px": int(args.target_cell_feather_px),
        }
        if bool(args.preserve_outside_target_cells):
            target_mask = np.zeros((int(original_grid_h), int(original_grid_w)), dtype=np.uint8)
            for gx, gy in active_target_cells_original:
                target_mask[int(gy), int(gx)] = 1
            final_img, final_target_alpha = composite_invalid_feature_cells(
                generated_img=final_img,
                source_img=original_source_img,
                valid_feature_mask=target_mask,
                grid_step_px=int(row.grid_step_px),
                feather_px=int(args.target_cell_feather_px),
            )
            target_cell_composite["target_cell_count"] = int(np.count_nonzero(target_mask))
            if str(args.output_mode) == "debug":
                save_png(
                    Image.fromarray((np.clip(final_target_alpha[:, :, 0], 0.0, 1.0) * 255.0).astype(np.uint8)),
                    run_dir / "final_target_cell_alpha.png",
                )
        if bool(padding_meta.get("enabled")) and str(args.output_mode) == "debug":
            save_png(runtime_final_img, run_dir / "padded_generated_full_context.png")
        save_png(final_img, final_out_path)
        if str(args.output_mode) == "debug":
            save_png(
                draw_cells_overlay(final_img, cells=list(active_target_cells_original), grid_step_px=int(row.grid_step_px)),
                run_dir / "generated_targets_overlay.png",
            )

        run_manifest = {
            "run_id": str(request.run_id),
            "region_id": str(request.region_id),
            "source_image_path": str(row.image_path),
            "source_feature_grid_path": str(row.feature_grid_path),
            "region_size": [int(original_source_img.size[0]), int(original_source_img.size[1])],
            "grid_shape": [int(original_grid_h), int(original_grid_w)],
            "runtime_region_size": [int(source_img.size[0]), int(source_img.size[1])],
            "runtime_grid_shape": [int(grid_h), int(grid_w)],
            "grid_step_px": int(row.grid_step_px),
            "target_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in active_target_cells_original],
            "runtime_target_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in active_target_cells],
            "requested_target_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in request.target_cells],
            "dropped_unsupported_target_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in unsupported_targets_original],
            "edited_cells": [
                {"gx": int(gx) - int(target_shift_x), "gy": int(gy) - int(target_shift_y)}
                for gx, gy in state.edited_cells
            ],
            "runtime_edited_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in state.edited_cells],
            "visited_cells": [
                {"gx": int(gx) - int(target_shift_x), "gy": int(gy) - int(target_shift_y)}
                for gx, gy in state.visited_cells
                if 0 <= int(gx) - int(target_shift_x) < int(original_grid_w)
                and 0 <= int(gy) - int(target_shift_y) < int(original_grid_h)
            ],
            "runtime_visited_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in state.visited_cells],
            "window_history": list(step_records),
            "target_metadata": target_metadata,
            "prototype_direction": str(args.direction),
            "prototype_latent": int(chosen_latent),
            "prototype_key": str(args.prototype_key),
            "concept_steering": concept_meta,
            "prototype_strength": float(args.prototype_strength),
            "steer_blend": float(args.steer_blend),
            "preserve_edit_strength": float(args.preserve_edit_strength),
            "preserve_visited_strength": float(args.preserve_visited_strength),
            "preserve_fresh_context_strength": float(args.preserve_fresh_context_strength),
            "mid_steer_start_ratio": float(args.mid_steer_start_ratio),
            "mid_steer_end_ratio": float(args.mid_steer_end_ratio),
            "mid_steer_alpha_start": float(args.mid_steer_alpha_start),
            "mid_steer_alpha_end": float(args.mid_steer_alpha_end),
            "mid_steer_alpha_schedule": str(args.mid_steer_alpha_schedule),
            "steer_context_halo_weight": float(args.steer_context_halo_weight),
            "steer_context_halo_radius_cells": int(args.steer_context_halo_radius_cells),
            "pix_model_id": str(args.pix_model_id),
            "pix_pipeline_id": str(args.pix_pipeline_id),
            "steps": int(args.steps),
            "guidance": float(args.guidance),
            "seed": int(args.seed),
            "edit_support": str(args.edit_support),
            "planning_edit_support": str(planning_edit_support),
            "context_halo_cells": int(args.context_halo_cells),
            "window_stride_cells": int(args.window_stride_cells),
            "window_selection_mode": str(args.window_selection_mode),
            "steer_full_support": bool(args.steer_full_support),
            "preserve_full_support": bool(args.preserve_full_support),
            "padding_context": padding_meta,
            "commit_mode": str(args.commit_mode),
            "commit_feather_px": int(args.commit_feather_px),
            "commit_halo_cells": int(args.commit_halo_cells),
            "commit_halo_alpha": float(args.commit_halo_alpha),
            "invalid_feature_composite": invalid_feature_composite,
            "target_cell_composite": target_cell_composite,
            "output_mode": str(args.output_mode),
            "output_path": str(final_out_path),
            "experiment_args_path": str(args.out_dir / "experiment_args.json"),
            "cli_args": args_payload["cli_args"],
            "command": args_payload["command"],
        }
        write_json(run_dir / "run_manifest.json", run_manifest)
        summary_rows.append(
            {
                "run_id": str(request.run_id),
                "region_id": str(request.region_id),
                "output_path": str(final_out_path),
                "num_targets": int(len(active_target_cells)),
                "num_requested_targets": int(len(request.target_cells)),
                "num_dropped_unsupported_targets": int(len(unsupported_targets)),
                "num_windows": int(len(step_records)),
            }
        )

    write_json(args.out_dir / "run_summary.json", {"runs": summary_rows})


if __name__ == "__main__":
    main()
