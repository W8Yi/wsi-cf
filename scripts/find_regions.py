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
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png, write_json
from wsi_cf.common.paths import (
    DEFAULT_HNSCC_CLAM_CKPT,
    DEFAULT_HNSCC_CLAM_COORDS_H5_DIR,
    DEFAULT_HNSCC_CLAM_DATASET_CSV,
    DEFAULT_HNSCC_CLAM_FEATURES_PT_DIR,
    DEFAULT_HNSCC_CLAM_SPLITS_CSV,
    DEFAULT_HNSCC_MIL_CKPT,
    DEFAULT_HNSCC_PROTOTYPE_NPZ,
    DEFAULT_HNSCC_SPLIT_TSV,
    DEFAULT_SAE_CFG,
    DEFAULT_SAE_CKPT,
    DEFAULT_SAE_VARIANT,
    DEFAULT_SHOWCASE_REGION_IMAGE,
    SAE_VARIANTS,
    resolve_sae_paths,
)
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.borderline_selection import (
    borderline_region_score,
    select_nonoverlapping_regions,
    source_target_probabilities,
)
from wsi_cf.data.donor_pool import load_split_rows
from wsi_cf.data.region_bank import make_region_cells_preview
from wsi_cf.data.region_selection import select_final_region_candidates
from wsi_cf.data.slides import find_slide_path, infer_objective_power, open_slide, quick_region_quality_metrics, read_region_rgb_at_magnification
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, load_prototypes, pick_prototype_latent, run_mil_attention
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2
from wsi_cf.steering.progressive import split_cells_by_edit_support
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find regions for counterfactual steering. "
            "Candidates combine MIL attention, SAE current-label prototype match, tissue quality, "
            "valid feature density, and controlled high-importance tile counts."
        )
    )
    parser.add_argument("--mode", type=str, default="attention", choices=["attention", "manual", "random"])
    parser.add_argument("--backend", dest="model_backend", type=str, default="clam", choices=["mil", "clam"])
    parser.add_argument("--region-image", type=Path, default=DEFAULT_SHOWCASE_REGION_IMAGE)
    parser.add_argument("--region-id", type=str, default="manual_region")
    parser.add_argument("--split-tsv", type=Path, default=DEFAULT_HNSCC_SPLIT_TSV)
    parser.add_argument("--split", type=str, default="test", help="Use 'all' to scan every split.")
    parser.add_argument("--model-backend", type=str, default="clam", choices=["mil", "clam"])
    parser.add_argument("--features-root", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"))
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/test"))
    parser.add_argument("--classifier-run-dir", type=Path, default=None, help="Generic classifier bundle; when set, use its task_manifest.csv and best_model.pt.")
    parser.add_argument("--task-manifest-csv", type=Path, default=None)
    parser.add_argument("--slides-root", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features"))
    parser.add_argument("--source-label", type=str, default="")
    parser.add_argument("--target-label", type=str, default="")
    parser.add_argument("--max-regions", type=int, default=0, help="Generic classifier mode cap. 0 uses final-regions-per-label.")
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/pathology_aware_2048_regions")
    parser.add_argument("--mil-ckpt", type=Path, default=DEFAULT_HNSCC_MIL_CKPT)
    parser.add_argument("--clam-ckpt", type=Path, default=DEFAULT_HNSCC_CLAM_CKPT)
    parser.add_argument("--clam-dataset-csv", type=Path, default=DEFAULT_HNSCC_CLAM_DATASET_CSV)
    parser.add_argument("--clam-splits-csv", type=Path, default=DEFAULT_HNSCC_CLAM_SPLITS_CSV)
    parser.add_argument("--clam-features-pt-dir", type=Path, default=DEFAULT_HNSCC_CLAM_FEATURES_PT_DIR)
    parser.add_argument("--clam-coords-h5-dir", type=Path, default=DEFAULT_HNSCC_CLAM_COORDS_H5_DIR)
    parser.add_argument("--clam-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--clam-source-from-task-split",
        action="store_true",
        help="For CLAM attention, use --split-tsv/--split source slides instead of the CLAM split CSV.",
    )
    parser.add_argument("--clam-attn-class", type=str, default="pred", choices=["pred", "pos", "neg"])
    parser.add_argument(
        "--clam-use-target-mag-equivalent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For curated CLAM 40x tiles, aggregate raw 256px cells into target-magnification-equivalent supercells before region mining/export.",
    )
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--sae-ckpt", type=Path, default=None, help=f"Explicit SAE checkpoint override. Defaults to --sae-variant ({DEFAULT_SAE_CKPT}).")
    parser.add_argument("--sae-cfg", type=Path, default=None, help=f"Explicit SAE config override. Defaults to --sae-variant ({DEFAULT_SAE_CFG}).")
    parser.add_argument("--prototype-npz", type=Path, default=DEFAULT_HNSCC_PROTOTYPE_NPZ)
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument("--region-size", type=int, default=2048)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--grid-step-tolerance-px", type=float, default=16.0, help="Reject slides whose effective feature spacing at target magnification differs too much from grid-step-px.")
    parser.add_argument("--window-stride-cells", type=int, default=4, help="Candidate stride in 256-cell units. 4 gives non-overlapping 2048 regions.")
    parser.add_argument("--max-slides-per-label", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--max-candidates-per-slide", type=int, default=24)
    parser.add_argument("--image-qc-topk-per-slide", type=int, default=32, help="Only crop/quality-check the top N cheap-scored regions per slide.")
    parser.add_argument("--final-regions-per-label", type=int, default=10)
    parser.add_argument("--final-slides-per-label", type=int, default=0, help="When set with --final-regions-per-slide, export this many slides per HPV label. 0 means all eligible slides.")
    parser.add_argument("--final-regions-per-slide", type=int, default=0, help="Export this many regions from each selected slide. With --final-slides-per-label 0, exports this many for every eligible slide.")
    parser.add_argument("--region-selection-mode", type=str, default="attention_sae", choices=["attention_sae", "borderline"])
    parser.add_argument("--borderline-attention-seed-fraction", type=float, default=0.359375)
    parser.add_argument("--borderline-attention-fraction-width", type=float, default=0.30)
    parser.add_argument("--borderline-target-prob-peak", type=float, default=0.55)
    parser.add_argument("--borderline-target-prob-width", type=float, default=0.45)
    parser.add_argument("--borderline-nms-overlap", type=float, default=0.50)
    parser.add_argument("--attention-percentile", type=float, default=90.0)
    parser.add_argument("--sae-percentile", type=float, default=90.0)
    parser.add_argument("--combined-percentile", type=float, default=90.0)
    parser.add_argument("--sae-attention-gate-percentile", type=float, default=75.0, help="Only compute SAE prototype scores on tiles above this attention percentile.")
    parser.add_argument("--attention-weight", type=float, default=0.6)
    parser.add_argument("--sae-weight", type=float, default=0.4)
    parser.add_argument("--min-tissue", type=float, default=0.50)
    parser.add_argument("--min-dark-fraction", type=float, default=0.08, help="Approximate cellularity/tumor-density proxy.")
    parser.add_argument("--min-saturation-fraction", type=float, default=0.08)
    parser.add_argument("--min-valid-feature-fraction", type=float, default=0.75)
    parser.add_argument("--min-high-importance-cells", type=int, default=2)
    parser.add_argument("--max-high-importance-cells", type=int, default=32)
    parser.add_argument("--max-high-importance-fraction", type=float, default=0.50)
    parser.add_argument("--min-largest-high-component", type=int, default=1, help="Require a contiguous high-importance component of at least this many cells.")
    parser.add_argument("--min-central-high-importance-fraction", type=float, default=0.0, help="Fraction of high-importance cells that must lie in the central area of the region.")
    parser.add_argument("--central-fraction", type=float, default=0.5, help="Fractional side-length of the central box used for central-high-importance checks.")
    parser.add_argument("--min-selected-cells", type=int, default=2)
    parser.add_argument("--max-selected-cells", type=int, default=12)
    parser.add_argument("--target-importance-mass", type=float, default=0.35)
    parser.add_argument(
        "--edit-cell-selection-mode",
        type=str,
        default="attention_percentile_smooth",
        choices=["attention_percentile_smooth", "showcase_smoothed28", "importance_mass_sae_neighbors"],
        help=(
            "How to choose target edit cells inside an eligible 2048 region. "
            "attention_percentile_smooth thresholds local attention scores and smooths the mask without a fixed tile count. "
            "showcase_smoothed28 is the historical fixed top-23/prune-28 selector."
        ),
    )
    parser.add_argument("--edit-cell-attention-percentile", type=float, default=64.0625)
    parser.add_argument("--edit-cell-smooth-min-neighbors", type=int, default=3)
    parser.add_argument("--edit-cell-smooth-border-relax", type=int, default=1)
    parser.add_argument("--edit-cell-smooth-iterations", type=int, default=1)
    parser.add_argument("--showcase-selection-top-n", type=int, default=23)
    parser.add_argument("--showcase-selection-target-count", type=int, default=28)
    parser.add_argument("--showcase-selection-smooth-min-neighbors", type=int, default=3)
    parser.add_argument("--showcase-selection-smooth-border-relax", type=int, default=1)
    parser.add_argument("--showcase-selection-smooth-iterations", type=int, default=1)
    parser.add_argument(
        "--expand-selected-by-sae-neighbors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand the seed edit set to 4-neighbor tiles whose SAE embeddings are similar enough to already-selected tiles.",
    )
    parser.add_argument("--neighbor-similarity-threshold", type=float, default=0.92)
    parser.add_argument("--neighbor-min-combined-importance", type=float, default=0.10)
    parser.add_argument(
        "--max-expanded-cells",
        type=int,
        default=0,
        help="0 means use 2x max-selected-cells as the expansion budget.",
    )
    parser.add_argument("--require-label-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-label-confidence", type=float, default=0.65)
    parser.add_argument("--save-candidate-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-slide-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def run_manual_image_mode(args: argparse.Namespace, *, device: torch.device) -> None:
    img = Image.open(args.region_image).convert("RGB")
    width, height = img.size
    if width % int(args.grid_step_px) != 0 or height % int(args.grid_step_px) != 0:
        raise ValueError(f"Manual region image must be divisible by grid_step_px={args.grid_step_px}: {args.region_image}")
    region_id = str(args.region_id)
    region_dir = args.out_dir / region_id
    region_dir.mkdir(parents=True, exist_ok=True)
    image_path = region_dir / "region.png"
    zgrid_path = region_dir / "region_zgrid.npy"
    mask_path = region_dir / "valid_feature_mask.npy"
    preview_path = region_dir / "region_cells.png"
    save_png(img, image_path)
    uni_model, uni_transform = load_uni2(device)
    z_grid = build_uni_grid_from_image(
        img,
        uni_model=uni_model,
        uni_transform=uni_transform,
        grid_step_px=int(args.grid_step_px),
        device=device,
        out_dtype=torch.float32,
    ).detach().cpu().numpy().astype(np.float32)
    np.save(zgrid_path, z_grid)
    np.save(mask_path, np.ones(z_grid.shape[:2], dtype=np.uint8))
    save_png(make_region_cells_preview(img, grid_step_px=int(args.grid_step_px)), preview_path)
    center_x0 = max(0, z_grid.shape[1] // 2 - 1)
    center_y0 = max(0, z_grid.shape[0] // 2 - 1)
    target_cells = [
        {"gx": int(center_x0), "gy": int(center_y0)},
        {"gx": int(center_x0 + 1), "gy": int(center_y0)},
        {"gx": int(center_x0), "gy": int(center_y0 + 1)},
        {"gx": int(center_x0 + 1), "gy": int(center_y0 + 1)},
    ]
    row = {
        "region_id": region_id,
        "split": "manual",
        "label": 1,
        "hpv_status": "HPV+",
        "case_id": region_id,
        "slide_key": region_id,
        "slide_path": str(args.region_image),
        "canonical_h5_path": "",
        "region_x": 0,
        "region_y": 0,
        "region_w": int(width),
        "region_h": int(height),
        "grid_step_px": int(args.grid_step_px),
        "feature_dim": int(z_grid.shape[-1]),
        "tissue_score": 1.0,
        "seed": int(args.seed),
        "image_path": str(image_path),
        "feature_grid_path": str(zgrid_path),
        "cell_preview_path": str(preview_path),
        "region_dir": str(region_dir),
        "selected_cells_local": ";".join(f"{c['gx']},{c['gy']}" for c in target_cells),
    }
    write_csv(args.out_dir / "region_bank.csv", [row])
    write_json(
        args.out_dir / "progressive_edit_manifest.json",
        [{"run_id": f"{region_id}__manual_center_2x2", "region_id": region_id, "target_cells": target_cells}],
    )
    write_json(
        region_dir / "region_meta.json",
        {
            "mode": "manual",
            "region_id": region_id,
            "source_image": str(args.region_image),
            "image_size": [int(width), int(height)],
            "grid_shape": [int(z_grid.shape[0]), int(z_grid.shape[1])],
            "grid_step_px": int(args.grid_step_px),
            "target_cells": target_cells,
        },
    )
    write_json(args.out_dir / "summary.json", {"mode": "manual", "region_count": 1, "region_bank_csv": str(args.out_dir / "region_bank.csv")})


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


def find_prefixed_file(root: Path, slide_key: str, suffix: str) -> Path | None:
    exact = root / f"{slide_key}{suffix}"
    if exact.exists():
        return exact
    matches = sorted(root.glob(f"{slide_key}*{suffix}"))
    return matches[0] if matches else None


def read_coord_h5(h5_path: Path) -> tuple[np.ndarray, int]:
    with h5py.File(h5_path, "r") as handle:
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
        patch_size = int(handle["coords"].attrs.get("patch_size", 256))
    return coords, patch_size


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
                "slide_key": slide_id,
                "slide_id": slide_id,
                "hpv_status": str(row.get("hpv_status", "")).strip(),
                "label": int(hpv_label_from_status(str(row.get("hpv_status", "")).strip())),
            }
    return out


def read_clam_split_slide_ids(path: Path, split_name: str) -> list[str]:
    out: list[str] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slide_id = str(row.get(split_name, "")).strip()
            if slide_id:
                out.append(slide_id)
    return out


def infer_coord_tile_size(coords: np.ndarray, fallback: int = 512) -> int:
    arr = np.asarray(coords, dtype=np.int64)
    candidates: list[int] = []
    for axis in (0, 1):
        vals = np.unique(arr[:, axis])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    return int(min(candidates)) if candidates else int(fallback)


def infer_integer_downsample_ratio(*, objective_power: float, target_magnification: float, tolerance: float = 0.2) -> int:
    obj = float(objective_power)
    target = float(target_magnification)
    if target <= 0.0 or obj <= 0.0 or target >= obj:
        return 1
    ratio = obj / target
    rounded = int(round(ratio))
    if rounded <= 1:
        return 1
    if abs(ratio - float(rounded)) > float(tolerance):
        raise ValueError(f"Cannot safely form target-magnification-equivalent CLAM cells: objective={obj}, target={target}, ratio={ratio:.4f}")
    return int(rounded)


def aggregate_cells_to_supercells(
    *,
    features: np.ndarray,
    coords: np.ndarray,
    tile_size_level0: int,
    block_ratio: int,
) -> tuple[np.ndarray, np.ndarray, int, dict[tuple[int, int], list[int]], list[tuple[int, int]]]:
    if int(block_ratio) <= 1:
        raw_cells = [
            (int(round(int(x) / float(tile_size_level0))), int(round(int(y) / float(tile_size_level0))))
            for x, y in np.asarray(coords, dtype=np.int64).tolist()
        ]
        members = {cell: [int(idx)] for idx, cell in enumerate(raw_cells)}
        return (
            np.asarray(features, dtype=np.float32),
            np.asarray(coords, dtype=np.int64),
            int(tile_size_level0),
            members,
            raw_cells,
        )

    super_members: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(np.asarray(coords, dtype=np.int64).tolist()):
        raw_gx = int(round(int(x) / float(tile_size_level0)))
        raw_gy = int(round(int(y) / float(tile_size_level0)))
        super_cell = (raw_gx // int(block_ratio), raw_gy // int(block_ratio))
        super_members.setdefault(super_cell, []).append(int(idx))

    ordered_cells = sorted(super_members.keys(), key=lambda cell: (int(cell[1]), int(cell[0])))
    agg_features = np.zeros((len(ordered_cells), int(features.shape[1])), dtype=np.float32)
    agg_coords = np.zeros((len(ordered_cells), 2), dtype=np.int64)
    agg_tile_size = int(tile_size_level0) * int(block_ratio)
    for out_idx, cell in enumerate(ordered_cells):
        member_idx = super_members[cell]
        agg_features[out_idx] = np.asarray(features[member_idx], dtype=np.float32).mean(axis=0)
        agg_coords[out_idx] = np.asarray([int(cell[0]) * agg_tile_size, int(cell[1]) * agg_tile_size], dtype=np.int64)
    return agg_features, agg_coords, int(agg_tile_size), super_members, ordered_cells


def aggregate_scores_by_supercell(
    *,
    values: np.ndarray,
    ordered_cells: Sequence[tuple[int, int]],
    members_by_cell: dict[tuple[int, int], list[int]],
    reducer: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.zeros((len(ordered_cells),), dtype=np.float32)
    for out_idx, cell in enumerate(ordered_cells):
        member_vals = arr[np.asarray(members_by_cell[cell], dtype=np.int64)]
        if member_vals.size == 0:
            continue
        if reducer == "max":
            out[out_idx] = float(np.max(member_vals))
        elif reducer == "sum":
            out[out_idx] = float(np.sum(member_vals))
        else:
            out[out_idx] = float(np.mean(member_vals))
    return out


def build_cell_maps(coords: np.ndarray, tile_size_level0: int) -> tuple[dict[tuple[int, int], int], dict[int, tuple[int, int]], int, int]:
    cell_to_index: dict[tuple[int, int], int] = {}
    index_to_cell: dict[int, tuple[int, int]] = {}
    for idx, (x, y) in enumerate(np.asarray(coords, dtype=np.int64).tolist()):
        cell = (int(round(int(x) / float(tile_size_level0))), int(round(int(y) / float(tile_size_level0))))
        cell_to_index[cell] = int(idx)
        index_to_cell[int(idx)] = cell
    grid_w = max(gx for gx, _ in cell_to_index) + 1
    grid_h = max(gy for _, gy in cell_to_index) + 1
    return cell_to_index, index_to_cell, int(grid_w), int(grid_h)


def effective_grid_step_at_target_magnification(*, tile_size_level0: int, objective_power: float, target_magnification: float) -> float:
    return float(tile_size_level0) * float(target_magnification) / max(float(objective_power), 1e-8)


def normalize_01(values: np.ndarray, *, lo_percentile: float = 5.0, hi_percentile: float = 95.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    lo = float(np.percentile(arr, lo_percentile))
    hi = float(np.percentile(arr, hi_percentile))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


@torch.no_grad()
def compute_sae_prototype_scores(
    *,
    sae_model: torch.nn.Module,
    features: np.ndarray,
    prototype_vector: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    proto = torch.from_numpy(np.asarray(prototype_vector, dtype=np.float32)).to(device=device)
    proto = proto / proto.norm().clamp(min=1e-8)
    scores: list[torch.Tensor] = []
    for start in range(0, int(features.shape[0]), int(batch_size)):
        x = torch.from_numpy(np.asarray(features[start : start + int(batch_size)], dtype=np.float32)).to(device=device)
        z = sae_encode_features(sae_model, x).float()
        z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scores.append((z @ proto).detach().cpu())
    return torch.cat(scores, dim=0).numpy().astype(np.float32, copy=False)


@torch.no_grad()
def compute_sae_prototype_scores_gated(
    *,
    sae_model: torch.nn.Module,
    features: np.ndarray,
    prototype_vector: np.ndarray,
    attention: np.ndarray,
    gate_percentile: float,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    attn = np.asarray(attention, dtype=np.float32).reshape(-1)
    gate_threshold = float(np.percentile(attn, float(gate_percentile)))
    gate_mask = attn >= gate_threshold
    gate_indices = np.flatnonzero(gate_mask).astype(np.int64)
    if gate_indices.size == 0:
        gate_indices = np.asarray([int(np.argmax(attn))], dtype=np.int64)
        gate_mask = np.zeros_like(attn, dtype=bool)
        gate_mask[gate_indices] = True
    scores = np.zeros((int(features.shape[0]),), dtype=np.float32)
    gated_scores = compute_sae_prototype_scores(
        sae_model=sae_model,
        features=np.asarray(features[gate_indices], dtype=np.float32),
        prototype_vector=prototype_vector,
        device=device,
        batch_size=int(batch_size),
    )
    scores[gate_indices] = gated_scores
    return scores, gate_mask


@torch.no_grad()
def compute_region_sae_embeddings(
    *,
    sae_model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device=device)
    z = sae_encode_features(sae_model, x).float()
    z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return z.detach().cpu().numpy().astype(np.float32, copy=False)


def load_or_compute_slide_scores(
    *,
    cache_dir: Path,
    slide_key: str,
    features: np.ndarray,
    attention: np.ndarray,
    pred: int,
    prob_pos: float,
    label: int,
    sae_model: torch.nn.Module,
    prototype_vector: np.ndarray,
    gate_percentile: float,
    device: torch.device,
    use_cache: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{slide_key}__scores.npz"
    if bool(use_cache) and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        meta_gate = float(data["sae_attention_gate_percentile"].reshape(-1)[0])
        meta_label = int(data["label"].reshape(-1)[0])
        if (
            int(data["n_tiles"].reshape(-1)[0]) == int(features.shape[0])
            and meta_label == int(label)
            and abs(meta_gate - float(gate_percentile)) < 1e-6
        ):
            return (
                np.asarray(data["sae_match"], dtype=np.float32),
                np.asarray(data["sae_gate_mask"], dtype=bool),
            )
    sae_match, gate_mask = compute_sae_prototype_scores_gated(
        sae_model=sae_model,
        features=features,
        prototype_vector=prototype_vector,
        attention=attention,
        gate_percentile=float(gate_percentile),
        device=device,
    )
    if bool(use_cache):
        np.savez_compressed(
            cache_path,
            slide_key=np.asarray([str(slide_key)]),
            label=np.asarray([int(label)], dtype=np.int64),
            pred=np.asarray([int(pred)], dtype=np.int64),
            prob_pos=np.asarray([float(prob_pos)], dtype=np.float32),
            n_tiles=np.asarray([int(features.shape[0])], dtype=np.int64),
            sae_attention_gate_percentile=np.asarray([float(gate_percentile)], dtype=np.float32),
            sae_match=sae_match.astype(np.float32, copy=False),
            sae_gate_mask=np.asarray(gate_mask, dtype=np.uint8),
        )
    return sae_match, gate_mask


def label_confidence_ok(label: int, pred: int, prob_pos: float, min_conf: float, require_match: bool) -> tuple[bool, float]:
    label = int(label)
    pred = int(pred)
    prob_pos = float(prob_pos)
    label_prob = prob_pos if label == 1 else 1.0 - prob_pos
    if bool(require_match) and pred != label:
        return False, float(label_prob)
    return bool(label_prob >= float(min_conf)), float(label_prob)


def load_clam_model(ckpt_path: Path, *, device: torch.device):
    from wsi_cf.models.clam import CLAM_MB

    model = CLAM_MB(gate=True, size_arg="small", n_classes=2, embed_dim=1536)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_clam_attention(model: Any, features: np.ndarray, *, device: torch.device, attn_class: str) -> tuple[np.ndarray, int, float]:
    feats_t = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device=device, dtype=torch.float32)
    logits, y_prob, y_hat, a_raw, _ = model(feats_t)
    attn = F.softmax(a_raw, dim=1)
    pred = int(y_hat.item())
    prob_pos = float(y_prob[0, 1].item())
    if str(attn_class) == "pred":
        row = pred
    elif str(attn_class) == "pos":
        row = 1
    else:
        row = 0
    return attn[row].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1), pred, prob_pos


def run_local_region_classifier(
    model: Any,
    features: np.ndarray,
    *,
    model_backend: str,
    device: torch.device,
    clam_attn_class: str,
) -> tuple[int, float]:
    if str(model_backend) == "mil":
        _, pred, prob_pos = run_mil_attention(model, np.asarray(features, dtype=np.float32), device=device)
    else:
        _, pred, prob_pos = run_clam_attention(
            model,
            np.asarray(features, dtype=np.float32),
            device=device,
            attn_class=str(clam_attn_class),
        )
    return int(pred), float(prob_pos)


def iter_region_starts(grid_w: int, grid_h: int, side: int, stride: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for gy0 in range(0, max(1, int(grid_h) - int(side) + 1), int(stride)):
        for gx0 in range(0, max(1, int(grid_w) - int(side) + 1), int(stride)):
            out.append((int(gx0), int(gy0)))
    return out


def aligned_start_candidates_for_cell(*, gx: int, gy: int, grid_w: int, grid_h: int, side: int, stride: int) -> list[tuple[int, int]]:
    max_gx0 = int(grid_w) - int(side)
    max_gy0 = int(grid_h) - int(side)
    if max_gx0 < 0 or max_gy0 < 0:
        return []
    min_gx0 = max(0, int(gx) - int(side) + 1)
    min_gy0 = max(0, int(gy) - int(side) + 1)
    max_gx0_for_cell = min(int(gx), max_gx0)
    max_gy0_for_cell = min(int(gy), max_gy0)
    if min_gx0 > max_gx0_for_cell or min_gy0 > max_gy0_for_cell:
        return []
    first_gx0 = int(math.ceil(float(min_gx0) / float(stride)) * int(stride))
    first_gy0 = int(math.ceil(float(min_gy0) / float(stride)) * int(stride))
    out: list[tuple[int, int]] = []
    for gy0 in range(first_gy0, int(max_gy0_for_cell) + 1, int(stride)):
        for gx0 in range(first_gx0, int(max_gx0_for_cell) + 1, int(stride)):
            out.append((int(gx0), int(gy0)))
    return out


def candidate_region_starts_from_seed_cells(
    *,
    seed_cells: Sequence[tuple[int, int]],
    grid_w: int,
    grid_h: int,
    side: int,
    stride: int,
) -> list[tuple[int, int]]:
    starts: set[tuple[int, int]] = set()
    for gx, gy in seed_cells:
        starts.update(
            aligned_start_candidates_for_cell(
                gx=int(gx),
                gy=int(gy),
                grid_w=int(grid_w),
                grid_h=int(grid_h),
                side=int(side),
                stride=int(stride),
            )
        )
    out = sorted(starts, key=lambda item: (int(item[1]), int(item[0])))
    if out:
        return out
    return iter_region_starts(int(grid_w), int(grid_h), int(side), int(stride))


def region_cells(gx0: int, gy0: int, side: int) -> list[tuple[int, int]]:
    return [(int(gx0) + lx, int(gy0) + ly) for ly in range(int(side)) for lx in range(int(side))]


def largest_4connected_component_size(cells: Sequence[tuple[int, int]]) -> int:
    remaining = { (int(x), int(y)) for x, y in cells }
    best = 0
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 1
        while stack:
            x, y = stack.pop()
            for nb in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(nb)
                    size += 1
        best = max(best, size)
    return int(best)


def central_local_box(side: int, central_fraction: float) -> tuple[int, int, int, int]:
    side = int(side)
    frac = float(max(0.0, min(1.0, central_fraction)))
    central_side = max(1, int(round(float(side) * frac)))
    start = max(0, (side - central_side) // 2)
    end = min(side, start + central_side)
    return int(start), int(start), int(end), int(end)


def fraction_in_central_box(
    cells: Sequence[tuple[int, int]],
    *,
    gx0: int,
    gy0: int,
    side: int,
    central_fraction: float,
) -> float:
    if not cells:
        return 0.0
    cx0, cy0, cx1, cy1 = central_local_box(int(side), float(central_fraction))
    count = 0
    for gx, gy in cells:
        lx = int(gx) - int(gx0)
        ly = int(gy) - int(gy0)
        if cx0 <= lx < cx1 and cy0 <= ly < cy1:
            count += 1
    return float(count) / float(len(cells))


def select_cells_by_mass(
    cells: Sequence[tuple[int, int]],
    importance_by_cell: dict[tuple[int, int], float],
    *,
    target_mass: float,
    min_cells: int,
    max_cells: int,
) -> list[tuple[int, int]]:
    ordered = sorted(cells, key=lambda cell: (-float(importance_by_cell.get(cell, 0.0)), int(cell[1]), int(cell[0])))
    total = float(sum(max(0.0, float(importance_by_cell.get(cell, 0.0))) for cell in ordered))
    selected: list[tuple[int, int]] = []
    mass = 0.0
    for cell in ordered:
        if len(selected) >= int(max_cells):
            break
        selected.append(cell)
        mass += max(0.0, float(importance_by_cell.get(cell, 0.0)))
        if len(selected) >= int(min_cells) and (total <= 0.0 or mass / total >= float(target_mass)):
            break
    return selected


def local_support_count(cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    gx, gy = int(cell[0]), int(cell[1])
    return sum(1 for nb in iter_neighbor_cells((gx, gy)) if nb in cells)


def smooth_fill_local_holes(
    cells: Sequence[tuple[int, int]],
    *,
    valid_cells: Sequence[tuple[int, int]],
    grid_w: int,
    grid_h: int,
    min_neighbors: int,
    border_relax: int,
    iterations: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    selected = {(int(gx), int(gy)) for gx, gy in cells}
    valid = {(int(gx), int(gy)) for gx, gy in valid_cells}
    added: list[tuple[int, int]] = []
    for _ in range(max(1, int(iterations))):
        to_add: list[tuple[int, int]] = []
        for gx, gy in sorted(valid, key=lambda item: (int(item[1]), int(item[0]))):
            cell = (int(gx), int(gy))
            if cell in selected:
                continue
            support = local_support_count(selected, cell)
            touches_border = int(gx) in {0, int(grid_w) - 1} or int(gy) in {0, int(grid_h) - 1}
            required = max(1, int(min_neighbors) - (int(border_relax) if touches_border else 0))
            if int(support) >= int(required):
                to_add.append(cell)
        if not to_add:
            break
        for cell in to_add:
            selected.add(cell)
            added.append(cell)
    return sorted(selected, key=lambda cell: (int(cell[1]), int(cell[0]))), added


def prune_selection_to_count(
    cells: Sequence[tuple[int, int]],
    score_by_cell: dict[tuple[int, int], float],
    *,
    target_count: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    selected = {(int(gx), int(gy)) for gx, gy in cells}
    pruned: list[tuple[int, int]] = []
    if int(target_count) <= 0 or len(selected) <= int(target_count):
        return sorted(selected, key=lambda cell: (int(cell[1]), int(cell[0]))), pruned
    while len(selected) > int(target_count):
        drop = min(
            selected,
            key=lambda cell: (
                local_support_count(selected, cell),
                float(score_by_cell.get(cell, 0.0)),
                int(cell[1]),
                int(cell[0]),
            ),
        )
        selected.remove(drop)
        pruned.append(drop)
    return sorted(selected, key=lambda cell: (int(cell[1]), int(cell[0]))), pruned


def select_showcase_smoothed28_cells(
    *,
    valid_cells: Sequence[tuple[int, int]],
    score_by_cell: dict[tuple[int, int], float],
    grid_w: int,
    grid_h: int,
    top_n: int,
    target_count: int,
    smooth_min_neighbors: int,
    smooth_border_relax: int,
    smooth_iterations: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    valid = sorted({(int(gx), int(gy)) for gx, gy in valid_cells}, key=lambda cell: (-float(score_by_cell.get(cell, 0.0)), int(cell[1]), int(cell[0])))
    seed = valid[: max(0, min(int(top_n), len(valid)))]
    smoothed, added = smooth_fill_local_holes(
        seed,
        valid_cells=valid_cells,
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        min_neighbors=int(smooth_min_neighbors),
        border_relax=int(smooth_border_relax),
        iterations=int(smooth_iterations),
    )
    selected, pruned = prune_selection_to_count(
        smoothed,
        score_by_cell,
        target_count=int(target_count),
    )
    return selected, sorted(seed, key=lambda cell: (int(cell[1]), int(cell[0]))), added, pruned


def select_attention_percentile_smooth_cells(
    *,
    valid_cells: Sequence[tuple[int, int]],
    score_by_cell: dict[tuple[int, int], float],
    grid_w: int,
    grid_h: int,
    percentile: float,
    smooth_min_neighbors: int,
    smooth_border_relax: int,
    smooth_iterations: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], float]:
    valid = sorted({(int(gx), int(gy)) for gx, gy in valid_cells}, key=lambda cell: (int(cell[1]), int(cell[0])))
    if not valid:
        return [], [], [], float("nan")
    scores = np.asarray([float(score_by_cell.get(cell, 0.0)) for cell in valid], dtype=np.float32)
    threshold = float(np.percentile(scores, float(percentile)))
    seed = [
        cell
        for cell in valid
        if float(score_by_cell.get(cell, 0.0)) >= float(threshold)
    ]
    if not seed:
        best = max(valid, key=lambda cell: (float(score_by_cell.get(cell, 0.0)), -int(cell[1]), -int(cell[0])))
        seed = [best]
    selected, added = smooth_fill_local_holes(
        seed,
        valid_cells=valid,
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        min_neighbors=int(smooth_min_neighbors),
        border_relax=int(smooth_border_relax),
        iterations=int(smooth_iterations),
    )
    return selected, sorted(seed, key=lambda cell: (int(cell[1]), int(cell[0]))), added, threshold


def select_attention_threshold_smooth_cells(
    *,
    valid_cells: Sequence[tuple[int, int]],
    score_by_cell: dict[tuple[int, int], float],
    grid_w: int,
    grid_h: int,
    threshold: float,
    smooth_min_neighbors: int,
    smooth_border_relax: int,
    smooth_iterations: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    valid = sorted({(int(gx), int(gy)) for gx, gy in valid_cells}, key=lambda cell: (int(cell[1]), int(cell[0])))
    seed = [
        cell
        for cell in valid
        if float(score_by_cell.get(cell, 0.0)) >= float(threshold)
    ]
    if not seed and valid:
        best = max(valid, key=lambda cell: (float(score_by_cell.get(cell, 0.0)), -int(cell[1]), -int(cell[0])))
        seed = [best]
    selected, added = smooth_fill_local_holes(
        seed,
        valid_cells=valid,
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        min_neighbors=int(smooth_min_neighbors),
        border_relax=int(smooth_border_relax),
        iterations=int(smooth_iterations),
    )
    return selected, sorted(seed, key=lambda cell: (int(cell[1]), int(cell[0]))), added


def iter_neighbor_cells(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, y = int(cell[0]), int(cell[1])
    return ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))


def expand_selected_cells_by_sae_neighbors(
    *,
    seed_cells: Sequence[tuple[int, int]],
    valid_cells: Sequence[tuple[int, int]],
    region_features: np.ndarray,
    sae_model: torch.nn.Module,
    device: torch.device,
    combined_by_cell: dict[tuple[int, int], float],
    similarity_threshold: float,
    min_combined_importance: float,
    max_cells: int,
) -> list[tuple[int, int]]:
    if not seed_cells:
        return []
    valid_order = [tuple((int(x), int(y))) for x, y in valid_cells]
    emb = compute_region_sae_embeddings(
        sae_model=sae_model,
        features=np.asarray(region_features, dtype=np.float32),
        device=device,
    )
    emb_by_cell = {cell: emb[idx] for idx, cell in enumerate(valid_order)}
    valid_set = set(valid_order)
    expanded: set[tuple[int, int]] = {tuple((int(x), int(y))) for x, y in seed_cells}

    while len(expanded) < int(max_cells):
        candidates: list[tuple[float, float, int, int]] = []
        for cell in sorted(expanded, key=lambda item: (int(item[1]), int(item[0]))):
            for nb in iter_neighbor_cells(cell):
                if nb not in valid_set or nb in expanded:
                    continue
                if float(combined_by_cell.get(nb, 0.0)) < float(min_combined_importance):
                    continue
                sim = max(
                    float(np.dot(emb_by_cell[nb], emb_by_cell[src]))
                    for src in expanded
                )
                if sim >= float(similarity_threshold):
                    candidates.append((sim, float(combined_by_cell.get(nb, 0.0)), int(nb[1]), int(nb[0])))
        if not candidates:
            break
        candidates.sort(key=lambda item: (-float(item[0]), -float(item[1]), int(item[2]), int(item[3])))
        added_any = False
        for _, _, gy, gx in candidates:
            nb = (int(gx), int(gy))
            if nb in expanded:
                continue
            expanded.add(nb)
            added_any = True
            if len(expanded) >= int(max_cells):
                break
        if not added_any:
            break
    return sorted(expanded, key=lambda cell: (int(cell[1]), int(cell[0])))


def draw_region_overlay(
    img: Image.Image,
    *,
    selected_cells: Sequence[tuple[int, int]],
    seed_cells: Sequence[tuple[int, int]],
    high_cells: Sequence[tuple[int, int]],
    grid_step_px: int,
) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    selected = set((int(gx), int(gy)) for gx, gy in selected_cells)
    seed = set((int(gx), int(gy)) for gx, gy in seed_cells)
    for gx, gy in high_cells:
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        draw.rectangle(
            [x0, y0, x0 + int(grid_step_px) - 1, y0 + int(grid_step_px) - 1],
            fill=(255, 165, 0, 70),
            outline=(255, 165, 0, 220),
            width=3,
        )
    for gx, gy in sorted(selected - seed, key=lambda cell: (int(cell[1]), int(cell[0]))):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        draw.rectangle(
            [x0, y0, x0 + int(grid_step_px) - 1, y0 + int(grid_step_px) - 1],
            fill=(0, 255, 255, 45),
            outline=(0, 255, 255, 235),
            width=5,
        )
    for rank, (gx, gy) in enumerate(seed_cells, start=1):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        draw.rectangle(
            [x0, y0, x0 + int(grid_step_px) - 1, y0 + int(grid_step_px) - 1],
            outline=(255, 255, 0, 255),
            width=6,
        )
        draw.text((x0 + 8, y0 + 8), f"edit {rank}", fill=(255, 0, 0, 255))
    return out


def encode_local_cells(cells: Sequence[tuple[int, int]]) -> str:
    return ";".join(f"{int(gx)},{int(gy)}" for gx, gy in cells)


@torch.no_grad()
def run_generic_mil_attention(model: torch.nn.Module, features: np.ndarray, *, device: torch.device) -> tuple[np.ndarray, int, float]:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    _, y_prob, y_hat, a_raw, _ = model(x)
    attn = F.softmax(a_raw, dim=1).detach().cpu().numpy().reshape(-1).astype(np.float32)
    pred = int(y_hat.detach().cpu().reshape(-1)[0].item())
    probs = y_prob.detach().cpu().numpy().reshape(-1)
    prob_pred = float(probs[pred]) if 0 <= pred < len(probs) else float(np.max(probs))
    return attn, pred, prob_pred


def load_classifier_manifest_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    manifest = args.task_manifest_csv or (args.classifier_run_dir / "task_manifest.csv" if args.classifier_run_dir is not None else None)
    if manifest is None or not manifest.exists():
        raise FileNotFoundError(f"Missing task manifest for generic classifier mode: {manifest}")
    rows: list[dict[str, str]] = []
    with manifest.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if args.source_label and str(row.get("label_name", "")) != str(args.source_label):
                continue
            h5_path = Path(str(row.get("h5_path", "")))
            if not h5_path.exists():
                continue
            rows.append(dict(row))
    rows.sort(key=lambda r: (str(r.get("split", "")) != "test", str(r.get("case_id", "")), str(r.get("slide_key", ""))))
    return rows


def find_generic_slide_path(slides_root: Path, project: str, slide_key: str) -> Path | None:
    ready_root = slides_root / ".tmp/ready_buffer/slides" / project
    if slides_root.name == "TCGA_features":
        ready_root = slides_root.parent / ".tmp/ready_buffer/slides" / project
    for base in (
        slides_root / project / "slides",
        ready_root,
    ):
        direct = find_slide_path(base, slide_key)
        if direct is not None:
            return direct
        matches = sorted(base.glob(f"{slide_key}*/*.svs"))
        if matches:
            return matches[0]
        matches = sorted(base.rglob(f"{slide_key}*.svs")) if base.exists() else []
        if matches:
            return matches[0]
    return None


def run_generic_classifier_mode(args: argparse.Namespace, *, device: torch.device) -> None:
    if args.classifier_run_dir is None:
        raise ValueError("--classifier-run-dir is required for generic classifier region finding")
    model = build_mil_from_checkpoint(args.classifier_run_dir / "best_model.pt", device=device)
    source_rows = load_classifier_manifest_rows(args)
    if not source_rows:
        raise RuntimeError(f"No source rows found for source_label={args.source_label!r}")
    region_side = int(round(float(args.region_size) / float(args.grid_step_px)))
    if region_side * int(args.grid_step_px) != int(args.region_size):
        raise ValueError("region_size must be divisible by grid_step_px")
    max_regions = int(args.max_regions) if int(args.max_regions) > 0 else int(args.final_regions_per_label)

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    edit_manifest: list[dict[str, Any]] = []
    seen_slides = 0
    for row in source_rows:
        if len(selected_rows) >= max_regions:
            break
        slide_key = str(row["slide_key"])
        project = str(row["project_dir"])
        slide_path = find_generic_slide_path(args.slides_root, project, slide_key)
        if slide_path is None:
            candidate_rows.append({**row, "eligible": False, "reason": "missing_slide"})
            continue
        h5_path = Path(str(row["h5_path"]))
        try:
            features, coords = read_h5_features_coords(h5_path)
        except Exception as exc:
            candidate_rows.append({**row, "eligible": False, "reason": f"h5_read_error:{exc}"})
            continue
        tile_size_level0 = infer_coord_tile_size(coords)
        cell_to_index, index_to_cell, grid_w, grid_h = build_cell_maps(coords, tile_size_level0)
        slide = open_slide(slide_path)
        try:
            objective_power = infer_objective_power(slide)
            effective_step = effective_grid_step_at_target_magnification(
                tile_size_level0=int(tile_size_level0),
                objective_power=float(objective_power),
                target_magnification=float(args.target_magnification),
            )
            if abs(float(effective_step) - float(args.grid_step_px)) > float(args.grid_step_tolerance_px):
                candidate_rows.append(
                    {
                        **row,
                        "eligible": False,
                        "reason": "feature_grid_spacing_mismatch",
                        "tile_size_level0": int(tile_size_level0),
                        "objective_power": float(objective_power),
                        "effective_grid_step": float(effective_step),
                    }
                )
                continue
            attention, pred, prob_pred = run_generic_mil_attention(model, features, device=device)
            label_id = int(row.get("label_id", -1))
            if bool(args.require_label_match) and pred != label_id:
                candidate_rows.append({**row, "eligible": False, "reason": "prediction_mismatch", "pred": int(pred), "prob_pred": float(prob_pred)})
                continue
            seen_slides += 1
            attn_norm = normalize_01(attention)
            order = np.argsort(-attention)
            tried_starts: set[tuple[int, int]] = set()
            selected_this_slide = 0
            for tile_idx in order[: max(int(args.max_candidates_per_slide) * 4, 16)].tolist():
                if len(selected_rows) >= max_regions:
                    break
                if selected_this_slide >= int(args.max_candidates_per_slide):
                    break
                gx, gy = index_to_cell[int(tile_idx)]
                gx0 = min(max(0, int(gx) - region_side // 2), max(0, int(grid_w) - region_side))
                gy0 = min(max(0, int(gy) - region_side // 2), max(0, int(grid_h) - region_side))
                if (gx0, gy0) in tried_starts:
                    continue
                tried_starts.add((gx0, gy0))
                cells = region_cells(gx0, gy0, region_side)
                valid_cells = [cell for cell in cells if cell in cell_to_index]
                valid_fraction = float(len(valid_cells) / max(len(cells), 1))
                if valid_fraction < float(args.min_valid_feature_fraction):
                    continue
                local_idxs = [cell_to_index[cell] for cell in valid_cells]
                local_attention = np.asarray([attention[idx] for idx in local_idxs], dtype=np.float32)
                importance_by_cell = {cell: float(attn_norm[cell_to_index[cell]]) for cell in valid_cells}
                threshold = float(np.percentile(local_attention, float(args.attention_percentile)))
                high_cells = [cell for cell in valid_cells if float(attention[cell_to_index[cell]]) >= threshold]
                if len(high_cells) < int(args.min_selected_cells):
                    continue
                selected_global = select_cells_by_mass(
                    high_cells,
                    importance_by_cell,
                    target_mass=float(args.target_importance_mass),
                    min_cells=int(args.min_selected_cells),
                    max_cells=int(args.max_selected_cells),
                )
                selected_local = [(int(x) - gx0, int(y) - gy0) for x, y in selected_global]
                region_img, crop_w0, crop_h0 = read_region_rgb_at_magnification(
                    slide,
                    x0=int(gx0) * int(tile_size_level0),
                    y0=int(gy0) * int(tile_size_level0),
                    out_w=int(args.region_size),
                    out_h=int(args.region_size),
                    target_magnification=float(args.target_magnification),
                )
                quality = quick_region_quality_metrics(region_img)
                if quality["tissue_score"] < float(args.min_tissue):
                    continue
                if quality["dark_fraction"] < float(args.min_dark_fraction):
                    continue
                if quality["saturation_fraction"] < float(args.min_saturation_fraction):
                    continue
                zgrid = np.zeros((region_side, region_side, int(features.shape[1])), dtype=np.float32)
                valid_mask = np.zeros((region_side, region_side), dtype=np.uint8)
                for ly in range(region_side):
                    for lx in range(region_side):
                        cell = (gx0 + lx, gy0 + ly)
                        if cell in cell_to_index:
                            zgrid[ly, lx] = features[cell_to_index[cell]]
                            valid_mask[ly, lx] = 1
                region_id = f"{slide_key}__{str(args.source_label).lower()}_to_{str(args.target_label).lower()}__mag_{str(args.target_magnification).replace('.', 'p')}__gx_{gx0}__gy_{gy0}"
                region_dir = args.out_dir / f"label_{str(args.source_label)}" / region_id
                region_dir.mkdir(parents=True, exist_ok=True)
                image_path = region_dir / "region.png"
                feature_grid_path = region_dir / "region_zgrid.npy"
                valid_mask_path = region_dir / "valid_feature_mask.npy"
                cells_path = region_dir / "region_cells.png"
                overlay_path = region_dir / "importance_overlay.png"
                save_png(region_img, image_path)
                np.save(feature_grid_path, zgrid)
                np.save(valid_mask_path, valid_mask)
                save_png(make_region_cells_preview(region_img, grid_step_px=int(args.grid_step_px)), cells_path)
                high_local = [(int(x) - gx0, int(y) - gy0) for x, y in high_cells]
                save_png(
                    draw_region_overlay(region_img, selected_cells=selected_local, seed_cells=selected_local, high_cells=high_local, grid_step_px=int(args.grid_step_px)),
                    overlay_path,
                )
                public = {
                    "region_id": region_id,
                    "split": str(row.get("split", "")),
                    "label": int(label_id),
                    "hpv_status": str(args.source_label),
                    "case_id": str(row.get("case_id", "")),
                    "slide_key": slide_key,
                    "slide_path": str(slide_path),
                    "canonical_h5_path": str(h5_path),
                    "region_x": int(gx0) * int(tile_size_level0),
                    "region_y": int(gy0) * int(tile_size_level0),
                    "region_w": int(args.region_size),
                    "region_h": int(args.region_size),
                    "grid_step_px": int(args.grid_step_px),
                    "feature_dim": int(features.shape[1]),
                    "tissue_score": float(quality["tissue_score"]),
                    "seed": int(args.seed),
                    "image_path": str(image_path),
                    "feature_grid_path": str(feature_grid_path),
                    "cell_preview_path": str(cells_path),
                    "region_dir": str(region_dir),
                    "project_dir": project,
                    "source_label": str(args.source_label),
                    "target_label": str(args.target_label),
                    "pred": int(pred),
                    "prob_pred": float(prob_pred),
                    "region_gx0": int(gx0),
                    "region_gy0": int(gy0),
                    "selected_cells_local": encode_local_cells(selected_local),
                    "high_importance_cells_local": encode_local_cells(high_local),
                    "valid_feature_fraction": float(valid_fraction),
                    "max_attention": float(local_attention.max()),
                    "mean_attention": float(local_attention.mean()),
                    "importance_overlay_path": str(overlay_path),
                }
                write_json(region_dir / "region_meta.json", public)
                candidate_rows.append({**public, "eligible": True, "reason": "selected"})
                selected_rows.append(public)
                selected_this_slide += 1
                edit_manifest.append(
                    {
                        "run_id": f"{region_id}__to_{str(args.target_label).lower()}_concepts",
                        "region_id": region_id,
                        "source_label": str(args.source_label),
                        "target_label": str(args.target_label),
                        "target_cells": [{"gx": int(x), "gy": int(y)} for x, y in selected_local],
                        "selector": "classifier_attention",
                    }
                )
        finally:
            slide.close()

    write_csv(args.out_dir / "candidate_scan.csv", candidate_rows)
    write_csv(args.out_dir / "selected_regions.csv", selected_rows)
    write_csv(args.out_dir / "region_bank.csv", selected_rows)
    write_json(args.out_dir / "progressive_edit_manifest.json", edit_manifest)
    summary = {
        "mode": "generic_classifier",
        "source_label": str(args.source_label),
        "target_label": str(args.target_label),
        "n_selected_rows": int(len(selected_rows)),
        "slides_scanned": int(seen_slides),
        "region_bank_csv": str(args.out_dir / "region_bank.csv"),
        "progressive_edit_manifest": str(args.out_dir / "progressive_edit_manifest.json"),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    args.sae_ckpt, args.sae_cfg = resolve_sae_paths(args.sae_variant, args.sae_ckpt, args.sae_cfg)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "experiment_args.json",
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "command": " ".join(shlex.quote(item) for item in sys.argv),
        },
    )
    if str(args.mode) == "manual":
        run_manual_image_mode(args, device=device)
        return
    if str(args.mode) == "random":
        raise NotImplementedError("Random mode is reserved for the next region finder pass; use --mode attention or --mode manual.")
    if args.classifier_run_dir is not None:
        run_generic_classifier_mode(args, device=device)
        return

    if str(args.model_backend) == "mil":
        source_rows = load_split_rows(args.split_tsv, split_filter=str(args.split))
    else:
        source_rows = []
        if bool(args.clam_source_from_task_split):
            for row in load_split_rows(args.split_tsv, split_filter=str(args.split)):
                slide_key = str(row["slide_key"])
                feature_path = find_prefixed_file(args.clam_features_pt_dir, slide_key, ".pt")
                coords_path = find_prefixed_file(args.clam_coords_h5_dir, slide_key, ".h5")
                source_rows.append(
                    {
                        "split": str(row["split"]),
                        "case_id": str(row["case_id"]),
                        "slide_key": slide_key,
                        "label": int(row["label"]),
                        "hpv_status": str(row["hpv_status"]),
                        "feature_path": "" if feature_path is None else str(feature_path),
                        "coords_path": "" if coords_path is None else str(coords_path),
                    }
                )
        else:
            labels_by_slide = load_clam_dataset_labels(args.clam_dataset_csv)
            for slide_id in read_clam_split_slide_ids(args.clam_splits_csv, str(args.clam_split)):
                meta = labels_by_slide.get(str(slide_id))
                if meta is None:
                    continue
                source_rows.append(
                    {
                        "split": str(args.clam_split),
                        "case_id": str(meta["case_id"]),
                        "slide_key": str(meta["slide_key"]),
                        "label": int(meta["label"]),
                        "hpv_status": str(meta["hpv_status"]),
                        "feature_path": str(args.clam_features_pt_dir / f"{slide_id}.pt"),
                        "coords_path": str(args.clam_coords_h5_dir / f"{slide_id}.h5"),
                    }
                )
    by_label_seen = {0: 0, 1: 0}
    scan_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_count_by_label = {0: 0, 1: 0}
    cache_dir = args.out_dir / "cache" / "slide_scores"

    if str(args.model_backend) == "mil":
        attention_model = build_mil_from_checkpoint(args.mil_ckpt, device=device)
    else:
        attention_model = load_clam_model(args.clam_ckpt, device=device)
    sae_model, d_in, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    label_proto = {0: proto_by_latent[int(neg_latent)], 1: proto_by_latent[int(pos_latent)]}

    region_side = int(round(float(args.region_size) / float(args.grid_step_px)))
    if region_side * int(args.grid_step_px) != int(args.region_size):
        raise ValueError("region_size must be divisible by grid_step_px")

    for split_row in source_rows:
        label = int(split_row["label"])
        if label not in (0, 1):
            continue
        if int(args.max_slides_per_label) > 0 and by_label_seen[label] >= int(args.max_slides_per_label):
            continue
        slide_key = str(split_row["slide_key"])
        slide_path = find_slide_path(args.slides_dir, slide_key)
        if str(args.model_backend) == "mil":
            feature_path = args.features_root / f"{slide_key}.h5"
            coords_path = feature_path
        else:
            feature_path = Path(str(split_row["feature_path"]))
            coords_path = Path(str(split_row["coords_path"]))
        if slide_path is None or not feature_path.exists() or not coords_path.exists():
            continue
        by_label_seen[label] += 1
        if str(args.model_backend) == "mil":
            raw_features, raw_coords = read_h5_features_coords(feature_path)
        else:
            raw_features = read_pt_features(feature_path)
            raw_coords, _ = read_coord_h5(coords_path)
            if int(raw_features.shape[0]) != int(raw_coords.shape[0]):
                scan_rows.append(
                    {
                        "slide_key": slide_key,
                        "label": int(label),
                        "eligible": False,
                        "reason": "feature_coord_count_mismatch",
                        "feature_path": str(feature_path),
                        "coords_path": str(coords_path),
                        "n_features": int(raw_features.shape[0]),
                        "n_coords": int(raw_coords.shape[0]),
                    }
                )
                continue
        if raw_features.shape[1] != int(d_in):
            continue
        raw_tile_size_level0 = infer_coord_tile_size(raw_coords)
        slide = open_slide(slide_path)
        try:
            objective_power = infer_objective_power(slide)
        finally:
            slide.close()
        agg_ratio = 1
        if str(args.model_backend) == "clam" and bool(args.clam_use_target_mag_equivalent):
            try:
                agg_ratio = infer_integer_downsample_ratio(
                    objective_power=float(objective_power),
                    target_magnification=float(args.target_magnification),
                )
            except ValueError as exc:
                scan_rows.append(
                    {
                        "slide_key": slide_key,
                        "label": int(label),
                        "eligible": False,
                        "reason": "unsupported_target_magnification_equivalent",
                        "feature_path": str(feature_path),
                        "coords_path": str(coords_path),
                        "objective_power": float(objective_power),
                        "target_magnification": float(args.target_magnification),
                        "error": str(exc),
                    }
                )
                continue
        tile_size_level0 = int(raw_tile_size_level0) * int(agg_ratio)
        effective_step = effective_grid_step_at_target_magnification(
            tile_size_level0=int(tile_size_level0),
            objective_power=float(objective_power),
            target_magnification=float(args.target_magnification),
        )
        if abs(float(effective_step) - float(args.grid_step_px)) > float(args.grid_step_tolerance_px):
            scan_rows.append(
                {
                    "slide_key": slide_key,
                    "label": int(label),
                    "eligible": False,
                    "reason": "feature_grid_spacing_mismatch",
                    "coord_tile_size_level0": int(tile_size_level0),
                    "raw_coord_tile_size_level0": int(raw_tile_size_level0),
                    "clam_aggregate_ratio": int(agg_ratio),
                    "objective_power": float(objective_power),
                    "effective_grid_step_at_target_mag": float(effective_step),
                    "expected_grid_step_px": int(args.grid_step_px),
                    "feature_path": str(feature_path),
                    "coords_path": str(coords_path),
                }
            )
            continue
        if str(args.model_backend) == "mil":
            raw_attention, pred, prob_pos = run_mil_attention(attention_model, raw_features, device=device)
        else:
            raw_attention, pred, prob_pos = run_clam_attention(
                attention_model,
                raw_features,
                device=device,
                attn_class=str(args.clam_attn_class),
            )
        label_ok, label_prob = label_confidence_ok(
            label=label,
            pred=pred,
            prob_pos=prob_pos,
            min_conf=float(args.min_label_confidence),
            require_match=bool(args.require_label_match),
        )
        if not label_ok:
            scan_rows.append(
                {
                    "slide_key": slide_key,
                    "label": int(label),
                    "eligible": False,
                    "reason": "slide_label_prediction_filter_failed",
                    "pred": int(pred),
                    "prob_pos": float(prob_pos),
                    "label_prob": float(label_prob),
                    "feature_path": str(feature_path),
                    "coords_path": str(coords_path),
                }
            )
            continue

        if str(args.model_backend) == "clam" and int(agg_ratio) > 1:
            features, coords, tile_size_level0, members_by_cell, ordered_cells = aggregate_cells_to_supercells(
                features=raw_features,
                coords=raw_coords,
                tile_size_level0=int(raw_tile_size_level0),
                block_ratio=int(agg_ratio),
            )
            attention = aggregate_scores_by_supercell(
                values=raw_attention,
                ordered_cells=ordered_cells,
                members_by_cell=members_by_cell,
                reducer="max",
            )
        else:
            features = np.asarray(raw_features, dtype=np.float32)
            coords = np.asarray(raw_coords, dtype=np.int64)
            attention = np.asarray(raw_attention, dtype=np.float32)
        cell_to_index, index_to_cell, grid_w, grid_h = build_cell_maps(coords, tile_size_level0)

        sae_match, sae_gate_mask = load_or_compute_slide_scores(
            cache_dir=cache_dir,
            slide_key=f"{slide_key}__backend_{args.model_backend}__mag_{str(args.target_magnification).replace('.', 'p')}__ratio_{int(agg_ratio)}",
            features=features,
            attention=attention,
            pred=int(pred),
            prob_pos=float(prob_pos),
            label=int(label),
            sae_model=sae_model,
            prototype_vector=label_proto[label],
            gate_percentile=float(args.sae_attention_gate_percentile),
            device=device,
            use_cache=bool(args.cache_slide_scores),
        )
        attn_norm = normalize_01(attention)
        sae_norm = normalize_01(sae_match)
        combined = float(args.attention_weight) * attn_norm + float(args.sae_weight) * sae_norm
        attn_threshold = float(np.percentile(attention, float(args.attention_percentile)))
        sae_threshold = float(np.percentile(sae_match, float(args.sae_percentile)))
        combined_threshold = float(np.percentile(combined, float(args.combined_percentile)))
        slide_edit_attention_threshold = float(np.percentile(attention, float(args.edit_cell_attention_percentile)))
        importance_by_cell = {index_to_cell[idx]: float(combined[idx]) for idx in range(len(combined))}
        seed_cells = [
            index_to_cell[idx]
            for idx in range(int(len(combined)))
            if float(attention[idx]) >= attn_threshold
            or bool(sae_gate_mask[idx] and float(sae_match[idx]) >= sae_threshold)
            or float(combined[idx]) >= combined_threshold
        ]

        if str(args.region_selection_mode) == "borderline":
            candidate_starts = iter_region_starts(
                grid_w=int(grid_w),
                grid_h=int(grid_h),
                side=int(region_side),
                stride=int(args.window_stride_cells),
            )
        else:
            candidate_starts = candidate_region_starts_from_seed_cells(
                seed_cells=seed_cells,
                grid_w=int(grid_w),
                grid_h=int(grid_h),
                side=int(region_side),
                stride=int(args.window_stride_cells),
            )

        cheap_candidates: list[dict[str, Any]] = []
        for gx0, gy0 in candidate_starts:
            cells = region_cells(gx0, gy0, region_side)
            valid_cells = [cell for cell in cells if cell in cell_to_index]
            valid_fraction = float(len(valid_cells) / float(len(cells)))
            if valid_fraction < float(args.min_valid_feature_fraction):
                continue
            local_idxs = [cell_to_index[cell] for cell in valid_cells]
            local_attention = np.asarray([attention[idx] for idx in local_idxs], dtype=np.float32)
            local_sae = np.asarray([sae_match[idx] for idx in local_idxs], dtype=np.float32)
            local_combined = np.asarray([combined[idx] for idx in local_idxs], dtype=np.float32)
            if str(args.region_selection_mode) == "borderline":
                high_cells = [
                    cell
                    for cell, idx in zip(valid_cells, local_idxs)
                    if float(attention[idx]) >= float(slide_edit_attention_threshold)
                ]
            else:
                high_cells = [
                    cell
                    for cell, idx in zip(valid_cells, local_idxs)
                    if float(attention[idx]) >= attn_threshold
                    or float(sae_match[idx]) >= sae_threshold
                    or float(combined[idx]) >= combined_threshold
                ]
            high_count = int(len(high_cells))
            high_fraction = float(high_count / float(len(cells)))
            largest_component = largest_4connected_component_size(high_cells)
            central_high_fraction = fraction_in_central_box(
                high_cells,
                gx0=int(gx0),
                gy0=int(gy0),
                side=int(region_side),
                central_fraction=float(args.central_fraction),
            )
            if str(args.region_selection_mode) != "borderline":
                if high_count < int(args.min_high_importance_cells):
                    continue
                if high_count > int(args.max_high_importance_cells):
                    continue
                if high_fraction > float(args.max_high_importance_fraction):
                    continue
                if int(largest_component) < int(args.min_largest_high_component):
                    continue
                if float(central_high_fraction) < float(args.min_central_high_importance_fraction):
                    continue

            smoothing_added_local: list[tuple[int, int]] = []
            pruned_local: list[tuple[int, int]] = []
            selection_threshold = float("nan")
            if str(args.region_selection_mode) == "borderline":
                valid_local = [(int(x) - int(gx0), int(y) - int(gy0)) for x, y in valid_cells]
                local_attention_by_cell = {
                    (int(x) - int(gx0), int(y) - int(gy0)): float(attention[cell_to_index[(int(x), int(y))]])
                    for x, y in valid_cells
                }
                selected_local, seed_selected_local, smoothing_added_local = select_attention_threshold_smooth_cells(
                    valid_cells=valid_local,
                    score_by_cell=local_attention_by_cell,
                    grid_w=int(region_side),
                    grid_h=int(region_side),
                    threshold=float(slide_edit_attention_threshold),
                    smooth_min_neighbors=int(args.edit_cell_smooth_min_neighbors),
                    smooth_border_relax=int(args.edit_cell_smooth_border_relax),
                    smooth_iterations=int(args.edit_cell_smooth_iterations),
                )
                selection_threshold = float(slide_edit_attention_threshold)
                selected_global = [(int(gx0) + int(x), int(gy0) + int(y)) for x, y in selected_local]
                seed_selected_global = [(int(gx0) + int(x), int(gy0) + int(y)) for x, y in seed_selected_local]
            elif str(args.edit_cell_selection_mode) == "attention_percentile_smooth":
                valid_local = [(int(x) - int(gx0), int(y) - int(gy0)) for x, y in valid_cells]
                local_attention_by_cell = {
                    (int(x) - int(gx0), int(y) - int(gy0)): float(attention[cell_to_index[(int(x), int(y))]])
                    for x, y in valid_cells
                }
                selected_local, seed_selected_local, smoothing_added_local, selection_threshold = select_attention_percentile_smooth_cells(
                    valid_cells=valid_local,
                    score_by_cell=local_attention_by_cell,
                    grid_w=int(region_side),
                    grid_h=int(region_side),
                    percentile=float(args.edit_cell_attention_percentile),
                    smooth_min_neighbors=int(args.edit_cell_smooth_min_neighbors),
                    smooth_border_relax=int(args.edit_cell_smooth_border_relax),
                    smooth_iterations=int(args.edit_cell_smooth_iterations),
                )
                selected_global = [(int(gx0) + int(x), int(gy0) + int(y)) for x, y in selected_local]
                seed_selected_global = [(int(gx0) + int(x), int(gy0) + int(y)) for x, y in seed_selected_local]
            elif str(args.edit_cell_selection_mode) == "showcase_smoothed28":
                valid_local = [(int(x) - int(gx0), int(y) - int(gy0)) for x, y in valid_cells]
                local_attention_by_cell = {
                    (int(x) - int(gx0), int(y) - int(gy0)): float(attention[cell_to_index[(int(x), int(y))]])
                    for x, y in valid_cells
                }
                selected_local, seed_selected_local, smoothing_added_local, pruned_local = select_showcase_smoothed28_cells(
                    valid_cells=valid_local,
                    score_by_cell=local_attention_by_cell,
                    grid_w=int(region_side),
                    grid_h=int(region_side),
                    top_n=int(args.showcase_selection_top_n),
                    target_count=int(args.showcase_selection_target_count),
                    smooth_min_neighbors=int(args.showcase_selection_smooth_min_neighbors),
                    smooth_border_relax=int(args.showcase_selection_smooth_border_relax),
                    smooth_iterations=int(args.showcase_selection_smooth_iterations),
                )
                selected_global = [(int(gx0) + int(x), int(gy0) + int(y)) for x, y in selected_local]
                seed_selected_global = [(int(gx0) + int(x), int(gy0) + int(y)) for x, y in seed_selected_local]
            else:
                selected_global = select_cells_by_mass(
                    high_cells,
                    importance_by_cell,
                    target_mass=float(args.target_importance_mass),
                    min_cells=int(args.min_selected_cells),
                    max_cells=int(args.max_selected_cells),
                )
                if not (int(args.min_selected_cells) <= len(selected_global) <= int(args.max_selected_cells)):
                    continue
                seed_selected_global = list(selected_global)
                expansion_budget = int(args.max_expanded_cells)
                if expansion_budget <= 0:
                    expansion_budget = max(int(args.max_selected_cells) * 2, int(args.min_selected_cells))
                if bool(args.expand_selected_by_sae_neighbors) and seed_selected_global:
                    region_features = np.asarray(features[local_idxs], dtype=np.float32)
                    selected_global = expand_selected_cells_by_sae_neighbors(
                        seed_cells=seed_selected_global,
                        valid_cells=valid_cells,
                        region_features=region_features,
                        sae_model=sae_model,
                        device=device,
                        combined_by_cell=importance_by_cell,
                        similarity_threshold=float(args.neighbor_similarity_threshold),
                        min_combined_importance=float(args.neighbor_min_combined_importance),
                        max_cells=int(expansion_budget),
                    )
            selected_local = [(int(gx) - int(gx0), int(gy) - int(gy0)) for gx, gy in selected_global]
            seed_selected_local = [(int(gx) - int(gx0), int(gy) - int(gy0)) for gx, gy in seed_selected_global]
            supported_selected_local, unsupported_selected_local = split_cells_by_edit_support(
                target_cells=selected_local,
                grid_w=int(region_side),
                grid_h=int(region_side),
                window_grid_side=4,
                stride_cells=2,
                grid_step_px=int(args.grid_step_px),
                edit_support="center_2x2",
            )
            if str(args.region_selection_mode) == "borderline":
                if not supported_selected_local:
                    continue
                selected_local = list(supported_selected_local)
                selected_global = [(int(gx0) + int(x), int(gy0) + int(y)) for x, y in selected_local]
                supported_seed_local = [cell for cell in seed_selected_local if cell in set(supported_selected_local)]
                seed_selected_local = supported_seed_local or list(selected_local)
                seed_selected_global = [(int(gx0) + int(x), int(gy0) + int(y)) for x, y in seed_selected_local]
            local_region_features = np.asarray(features[local_idxs], dtype=np.float32)
            if str(args.region_selection_mode) == "borderline":
                region_pred, region_prob_pos = run_local_region_classifier(
                    attention_model,
                    local_region_features,
                    model_backend=str(args.model_backend),
                    device=device,
                    clam_attn_class=str(args.clam_attn_class),
                )
            else:
                region_pred, region_prob_pos = int(pred), float(prob_pos)
            region_source_prob, region_target_prob = source_target_probabilities(
                label=int(label),
                prob_pos=float(region_prob_pos),
            )
            borderline_scores = borderline_region_score(
                attention_seed_fraction=float(high_fraction),
                region_target_prob=float(region_target_prob),
                tissue_score=0.0,
                valid_feature_fraction=float(valid_fraction),
                attention_center=float(args.borderline_attention_seed_fraction),
                attention_width=float(args.borderline_attention_fraction_width),
                target_prob_peak=float(args.borderline_target_prob_peak),
                target_prob_width=float(args.borderline_target_prob_width),
            )
            cheap_candidates.append(
                {
                    "slide_key": slide_key,
                    "case_id": str(split_row["case_id"]),
                    "split": str(split_row["split"]),
                    "label": int(label),
                    "hpv_status": "hpv_pos" if label == 1 else "hpv_neg",
                    "feature_path": str(feature_path),
                    "coords_path": str(coords_path),
                    "pred": int(pred),
                    "prob_pos": float(prob_pos),
                    "label_prob": float(label_prob),
                    "region_pred": int(region_pred),
                    "region_prob_pos": float(region_prob_pos),
                    "region_source_prob": float(region_source_prob),
                    "region_target_prob": float(region_target_prob),
                    "raw_coord_tile_size_level0": int(raw_tile_size_level0),
                    "selection_tile_size_level0": int(tile_size_level0),
                    "clam_aggregate_ratio": int(agg_ratio),
                    "region_gx0": int(gx0),
                    "region_gy0": int(gy0),
                    "region_x": int(gx0) * int(tile_size_level0),
                    "region_y": int(gy0) * int(tile_size_level0),
                    "valid_feature_fraction": float(valid_fraction),
                    "tissue_score": -1.0,
                    "dark_fraction": -1.0,
                    "saturation_fraction": -1.0,
                    "high_importance_cell_count": int(high_count),
                    "high_importance_fraction": float(high_fraction),
                    "attention_seed_fraction": float(high_fraction),
                    "slide_attention_seed_threshold": float(slide_edit_attention_threshold),
                    "largest_high_component": int(largest_component),
                    "central_high_importance_fraction": float(central_high_fraction),
                    "selected_cell_count": int(len(selected_global)),
                    "seed_selected_cell_count": int(len(seed_selected_global)),
                    "unsupported_center2x2_cell_count": int(len(unsupported_selected_local)),
                    "unsupported_center2x2_cells_local": encode_local_cells(unsupported_selected_local),
                    "mean_attention": float(local_attention.mean()),
                    "max_attention": float(local_attention.max()),
                    "mean_sae_match": float(local_sae.mean()),
                    "max_sae_match": float(local_sae.max()),
                    "mean_combined_importance": float(local_combined.mean()),
                    "max_combined_importance": float(local_combined.max()),
                    **borderline_scores,
                    "cheap_rank_score": float(
                        3.0 * float(local_combined.mean())
                        + 1.5 * float(local_combined.max())
                        + 0.5 * float(valid_fraction)
                        - 0.5 * float(high_fraction)
                        + 0.35 * float(largest_component)
                        + 0.5 * float(central_high_fraction)
                    ),
                    "high_importance_cells_global": encode_local_cells(high_cells),
                    "high_importance_cells_local": encode_local_cells([(int(gx) - int(gx0), int(gy) - int(gy0)) for gx, gy in high_cells]),
                    "selected_cells_global": encode_local_cells(selected_global),
                    "selected_cells_local": encode_local_cells(selected_local),
                    "seed_selected_cells_global": encode_local_cells(seed_selected_global),
                    "seed_selected_cells_local": encode_local_cells(seed_selected_local),
                    "smoothing_added_cells_local": encode_local_cells(smoothing_added_local),
                    "pruned_cells_local": encode_local_cells(pruned_local),
                    "edit_cell_selection_mode": str(args.edit_cell_selection_mode),
                    "edit_cell_attention_threshold": selection_threshold,
                    "region_selection_mode": str(args.region_selection_mode),
                }
            )

        if str(args.region_selection_mode) == "borderline":
            cheap_candidates.sort(
                key=lambda row: (
                    -float(row["borderline_region_score"]),
                    float(row["region_source_prob"]),
                    str(row["slide_key"]),
                    int(row["region_y"]),
                    int(row["region_x"]),
                )
            )
        else:
            cheap_candidates.sort(
                key=lambda row: (
                    -float(row["cheap_rank_score"]),
                    int(row["high_importance_cell_count"]),
                    str(row["slide_key"]),
                    int(row["region_y"]),
                    int(row["region_x"]),
                )
            )
        image_checked_candidates: list[dict[str, Any]] = []
        if cheap_candidates:
            slide = open_slide(slide_path)
            try:
                for cand in cheap_candidates[: int(args.image_qc_topk_per_slide)]:
                    region_img, crop_w0, crop_h0 = read_region_rgb_at_magnification(
                        slide,
                        x0=int(cand["region_x"]),
                        y0=int(cand["region_y"]),
                        out_w=int(args.region_size),
                        out_h=int(args.region_size),
                        target_magnification=float(args.target_magnification),
                    )
                    quality = quick_region_quality_metrics(region_img)
                    if quality["tissue_score"] < float(args.min_tissue):
                        continue
                    if quality["dark_fraction"] < float(args.min_dark_fraction):
                        continue
                    if quality["saturation_fraction"] < float(args.min_saturation_fraction):
                        continue
                    checked = dict(cand)
                    checked["crop_w_level0"] = int(crop_w0)
                    checked["crop_h_level0"] = int(crop_h0)
                    checked["tissue_score"] = float(quality["tissue_score"])
                    checked["dark_fraction"] = float(quality["dark_fraction"])
                    checked["saturation_fraction"] = float(quality["saturation_fraction"])
                    if str(args.region_selection_mode) == "borderline":
                        checked.update(
                            borderline_region_score(
                                attention_seed_fraction=float(checked["attention_seed_fraction"]),
                                region_target_prob=float(checked["region_target_prob"]),
                                tissue_score=float(checked["tissue_score"]),
                                valid_feature_fraction=float(checked["valid_feature_fraction"]),
                                attention_center=float(args.borderline_attention_seed_fraction),
                                attention_width=float(args.borderline_attention_fraction_width),
                                target_prob_peak=float(args.borderline_target_prob_peak),
                                target_prob_width=float(args.borderline_target_prob_width),
                            )
                        )
                    image_checked_candidates.append(checked)
            finally:
                slide.close()

        if str(args.region_selection_mode) == "borderline":
            image_checked_candidates.sort(
                key=lambda row: (
                    -float(row["borderline_region_score"]),
                    float(row["region_source_prob"]),
                    -float(row["tissue_score"]),
                    str(row["slide_key"]),
                    int(row["region_y"]),
                    int(row["region_x"]),
                )
            )
            selected_candidates = select_nonoverlapping_regions(
                image_checked_candidates,
                side=int(region_side),
                max_regions=int(args.max_candidates_per_slide),
                max_overlap_fraction=float(args.borderline_nms_overlap),
            )
            selected_keys = {
                (int(row["region_gx0"]), int(row["region_gy0"]))
                for row in selected_candidates
            }
            selected_rank = 0
            for cand_rank, cand in enumerate(image_checked_candidates, start=1):
                public = dict(cand)
                is_selected = (int(cand["region_gx0"]), int(cand["region_gy0"])) in selected_keys
                if is_selected:
                    selected_rank += 1
                public["candidate_rank_in_slide"] = int(selected_rank if is_selected else cand_rank)
                public["candidate_raw_rank_in_slide"] = int(cand_rank)
                public["eligible"] = bool(is_selected)
                public["reason"] = "eligible" if is_selected else "borderline_nms_or_rank_filtered"
                public["sae_gate_tile_count"] = int(np.count_nonzero(sae_gate_mask))
                scan_rows.append(public)
        else:
            image_checked_candidates.sort(
                key=lambda row: (
                    -float(row["mean_combined_importance"]),
                    -float(row["tissue_score"]),
                    int(row["high_importance_cell_count"]),
                    str(row["slide_key"]),
                    int(row["region_y"]),
                    int(row["region_x"]),
                )
            )
            for cand_rank, cand in enumerate(image_checked_candidates[: int(args.max_candidates_per_slide)], start=1):
                public = dict(cand)
                public["candidate_rank_in_slide"] = int(cand_rank)
                public["eligible"] = True
                public["reason"] = "eligible"
                public["sae_gate_tile_count"] = int(np.count_nonzero(sae_gate_mask))
                scan_rows.append(public)

    write_csv(args.out_dir / "candidate_scan.csv", scan_rows)

    eligible_by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    # Reconstruct only public candidates from scan rows, sorted globally.
    for row in scan_rows:
        if bool(row.get("eligible")):
            eligible_by_label[int(row["label"])].append(row)
    for label in (0, 1):
        if str(args.region_selection_mode) == "borderline":
            eligible_by_label[label].sort(
                key=lambda row: (
                    str(row["slide_key"]),
                    int(row.get("candidate_rank_in_slide", 0)),
                    -float(row["borderline_region_score"]),
                    int(row["region_y"]),
                    int(row["region_x"]),
                )
            )
        else:
            eligible_by_label[label].sort(
                key=lambda row: (
                    -float(row["mean_combined_importance"]),
                    -float(row["tissue_score"]),
                    str(row["slide_key"]),
                    int(row["region_y"]),
                    int(row["region_x"]),
                )
            )

    final_rows_to_export = select_final_region_candidates(
        eligible_by_label,
        final_regions_per_label=int(args.final_regions_per_label),
        final_slides_per_label=int(args.final_slides_per_label),
        final_regions_per_slide=int(args.final_regions_per_slide),
    )

    rows_by_slide: dict[str, list[dict[str, Any]]] = {}
    for row in final_rows_to_export:
        rows_by_slide.setdefault(str(row["slide_key"]), []).append(row)

    for slide_key, slide_rows in rows_by_slide.items():
        slide_path = find_slide_path(args.slides_dir, str(slide_key))
        if slide_path is None:
            continue
        first_row = slide_rows[0]
        feature_path = Path(str(first_row["feature_path"]))
        coords_path = Path(str(first_row["coords_path"]))
        if str(args.model_backend) == "mil":
            raw_features, raw_coords = read_h5_features_coords(feature_path)
        else:
            raw_features = read_pt_features(feature_path)
            raw_coords, _ = read_coord_h5(coords_path)
        raw_tile_size_level0 = infer_coord_tile_size(raw_coords)
        slide = open_slide(slide_path)
        try:
            objective_power = infer_objective_power(slide)
            agg_ratio = 1
            if str(args.model_backend) == "clam" and bool(args.clam_use_target_mag_equivalent):
                agg_ratio = infer_integer_downsample_ratio(
                    objective_power=float(objective_power),
                    target_magnification=float(args.target_magnification),
                )
            if str(args.model_backend) == "clam" and int(agg_ratio) > 1:
                features, coords, tile_size_level0, _, _ = aggregate_cells_to_supercells(
                    features=raw_features,
                    coords=raw_coords,
                    tile_size_level0=int(raw_tile_size_level0),
                    block_ratio=int(agg_ratio),
                )
            else:
                features = np.asarray(raw_features, dtype=np.float32)
                coords = np.asarray(raw_coords, dtype=np.int64)
                tile_size_level0 = int(raw_tile_size_level0)
            cell_to_index, _, _, _ = build_cell_maps(coords, tile_size_level0)
            for row in slide_rows:
                label = int(row["label"])
                region_img, crop_w0, crop_h0 = read_region_rgb_at_magnification(
                    slide,
                    x0=int(row["region_x"]),
                    y0=int(row["region_y"]),
                    out_w=int(args.region_size),
                    out_h=int(args.region_size),
                    target_magnification=float(args.target_magnification),
                )
                region_id = (
                    f"{row['slide_key']}__pathaware{int(args.region_size)}__mag_{str(args.target_magnification).replace('.', 'p')}"
                    f"__gx_{int(row['region_gx0'])}__gy_{int(row['region_gy0'])}"
                )
                region_dir = args.out_dir / f"label_{label}_{'hpv_pos' if label == 1 else 'hpv_neg'}" / region_id
                region_dir.mkdir(parents=True, exist_ok=True)

                zgrid = np.zeros((region_side, region_side, int(features.shape[1])), dtype=np.float32)
                valid_mask = np.zeros((region_side, region_side), dtype=np.uint8)
                for ly in range(region_side):
                    for lx in range(region_side):
                        global_cell = (int(row["region_gx0"]) + lx, int(row["region_gy0"]) + ly)
                        if global_cell in cell_to_index:
                            zgrid[ly, lx] = features[cell_to_index[global_cell]]
                            valid_mask[ly, lx] = 1

                selected_local = [
                    tuple(int(part) for part in token.split(","))
                    for token in str(row["selected_cells_local"]).split(";")
                    if token
                ]
                seed_selected_local = [
                    tuple(int(part) for part in token.split(","))
                    for token in str(row.get("seed_selected_cells_local", "")).split(";")
                    if token
                ]
                if not seed_selected_local:
                    seed_selected_local = list(selected_local)
                high_global = [
                    tuple(int(part) for part in token.split(","))
                    for token in str(row["high_importance_cells_global"]).split(";")
                    if token
                ]
                high_local = [(int(gx) - int(row["region_gx0"]), int(gy) - int(row["region_gy0"])) for gx, gy in high_global]
                image_path = region_dir / "region.png"
                feature_grid_path = region_dir / "region_zgrid.npy"
                valid_mask_path = region_dir / "valid_feature_mask.npy"
                cells_path = region_dir / "region_cells.png"
                overlay_path = region_dir / "importance_overlay.png"
                meta_path = region_dir / "region_meta.json"
                save_png(region_img, image_path)
                np.save(feature_grid_path, zgrid)
                np.save(valid_mask_path, valid_mask)
                save_png(make_region_cells_preview(region_img, grid_step_px=int(args.grid_step_px)), cells_path)
                save_png(
                    draw_region_overlay(
                        region_img,
                        selected_cells=selected_local,
                        seed_cells=seed_selected_local,
                        high_cells=high_local,
                        grid_step_px=int(args.grid_step_px),
                    ),
                    overlay_path,
                )
                public = dict(row)
                public.update(
                    {
                        "region_id": region_id,
                        "slide_path": str(slide_path),
                        "canonical_h5_path": str(feature_path),
                        "coords_path": str(coords_path),
                        "region_w": int(args.region_size),
                        "region_h": int(args.region_size),
                        "crop_w_level0": int(crop_w0),
                        "crop_h_level0": int(crop_h0),
                        "grid_step_px": int(args.grid_step_px),
                        "feature_dim": int(features.shape[1]),
                        "seed": int(args.seed),
                        "image_path": str(image_path),
                        "feature_grid_path": str(feature_grid_path),
                        "valid_mask_path": str(valid_mask_path),
                        "cell_preview_path": str(cells_path),
                        "importance_overlay_path": str(overlay_path),
                        "region_dir": str(region_dir),
                        "selection_mode": (
                            f"attention_percentile_smooth_p{str(float(args.edit_cell_attention_percentile)).replace('.', 'p')}"
                            if str(row.get("edit_cell_selection_mode", args.edit_cell_selection_mode)) == "attention_percentile_smooth"
                            else (
                                f"showcase_smoothed28_attention_top{int(args.showcase_selection_top_n)}"
                                f"_smooth4_prune{int(args.showcase_selection_target_count)}"
                                if str(row.get("edit_cell_selection_mode", args.edit_cell_selection_mode)) == "showcase_smoothed28"
                                else (
                                    "importance_mass_plus_sae_neighbor_expansion"
                                    if bool(args.expand_selected_by_sae_neighbors)
                                    else "importance_mass_only"
                                )
                            )
                        ),
                        "edit_cell_attention_percentile": float(args.edit_cell_attention_percentile),
                    }
                )
                write_json(meta_path, public)
                selected_rows.append(public)
                selected_count_by_label[label] += 1
        finally:
            slide.close()

    write_csv(args.out_dir / "selected_regions.csv", selected_rows)
    write_csv(args.out_dir / "region_bank.csv", selected_rows)
    # A manifest the canonical progressive editor can consume after adapting region_bank-style rows.
    edit_manifest = [
        {
            "run_id": f"{row['region_id']}__attention_sae_selected",
            "region_id": str(row["region_id"]),
            "selection_mode": str(row.get("selection_mode", "importance_mass_only")),
            "target_cells": [
                {"gx": int(cell.split(",")[0]), "gy": int(cell.split(",")[1])}
                for cell in str(row["selected_cells_local"]).split(";")
                if cell
            ],
            "seed_target_cells": [
                {"gx": int(cell.split(",")[0]), "gy": int(cell.split(",")[1])}
                for cell in str(row.get("seed_selected_cells_local", "")).split(";")
                if cell
            ],
            "selector": (
                "borderline_wsi_attention_plus_region_probability"
                if str(row.get("region_selection_mode", args.region_selection_mode)) == "borderline"
                else "mil_attention_plus_sae_current_label_prototype"
            ),
            "label": int(row["label"]),
            "direction_recommendation": "hpv_neg" if int(row["label"]) == 1 else "hpv_pos",
        }
        for row in selected_rows
    ]
    write_json(args.out_dir / "progressive_edit_manifest.json", edit_manifest)
    summary = {
        "n_candidate_rows": int(len(scan_rows)),
        "n_selected_rows": int(len(selected_rows)),
        "selected_count_by_label": {str(k): int(v) for k, v in selected_count_by_label.items()},
        "eligible_count_by_label": {str(k): int(len(v)) for k, v in eligible_by_label.items()},
        "slides_scanned_by_label": {str(k): int(v) for k, v in by_label_seen.items()},
        "region_size": int(args.region_size),
        "target_magnification": float(args.target_magnification),
        "grid_step_px": int(args.grid_step_px),
        "region_grid_side": int(region_side),
        "final_regions_per_label": int(args.final_regions_per_label),
        "final_slides_per_label": int(args.final_slides_per_label),
        "final_regions_per_slide": int(args.final_regions_per_slide),
        "seed": int(args.seed),
        "model_backend": str(args.model_backend),
        "region_selection_mode": str(args.region_selection_mode),
        "borderline_attention_seed_fraction": float(args.borderline_attention_seed_fraction),
        "borderline_attention_fraction_width": float(args.borderline_attention_fraction_width),
        "borderline_target_prob_peak": float(args.borderline_target_prob_peak),
        "borderline_target_prob_width": float(args.borderline_target_prob_width),
        "borderline_nms_overlap": float(args.borderline_nms_overlap),
        "attention_weight": float(args.attention_weight),
        "sae_weight": float(args.sae_weight),
        "edit_cell_selection_mode": str(args.edit_cell_selection_mode),
        "edit_cell_attention_percentile": float(args.edit_cell_attention_percentile),
        "edit_cell_smooth_min_neighbors": int(args.edit_cell_smooth_min_neighbors),
        "edit_cell_smooth_border_relax": int(args.edit_cell_smooth_border_relax),
        "edit_cell_smooth_iterations": int(args.edit_cell_smooth_iterations),
        "showcase_selection_top_n": int(args.showcase_selection_top_n),
        "showcase_selection_target_count": int(args.showcase_selection_target_count),
        "pos_latent": int(pos_latent),
        "neg_latent": int(neg_latent),
        "candidate_scan_csv": str(args.out_dir / "candidate_scan.csv"),
        "selected_regions_csv": str(args.out_dir / "selected_regions.csv"),
        "region_bank_csv": str(args.out_dir / "region_bank.csv"),
        "progressive_edit_manifest": str(args.out_dir / "progressive_edit_manifest.json"),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
