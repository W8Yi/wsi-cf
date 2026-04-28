#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from find_pathology_aware_2048_regions import (  # type: ignore
    compute_sae_prototype_scores_gated,
    draw_region_overlay,
    encode_local_cells,
    expand_selected_cells_by_sae_neighbors,
    hpv_label_from_status,
    load_clam_model,
    normalize_01,
    run_clam_attention,
    select_cells_by_mass,
)

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
from wsi_cf.data.region_bank import make_region_cells_preview
from wsi_cf.eval.hnsc_hpv import load_prototypes, pick_prototype_latent

ensure_legacy_repo_root_on_path()

from utils.sae import load_sae_from_config  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a manually chosen showcase region with local CLAM attention + SAE importance, then expand the edit mask."
    )
    parser.add_argument("--source-region-dir", type=Path, required=True)
    parser.add_argument("--crop-cell-x0", type=int, default=8)
    parser.add_argument("--crop-cell-y0", type=int, default=0)
    parser.add_argument("--crop-cell-side", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clam-ckpt", type=Path, default=Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/s_0_checkpoint.pt"))
    parser.add_argument("--clam-attn-class", type=str, default="pred", choices=["pred", "pos", "neg"])
    parser.add_argument("--sae-ckpt", type=Path, default=DEFAULT_SAE_CKPT)
    parser.add_argument("--sae-cfg", type=Path, default=DEFAULT_SAE_CFG)
    parser.add_argument("--prototype-npz", type=Path, default=DEFAULT_HNSCC_PROTOTYPE_NPZ)
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--attention-percentile", type=float, default=90.0)
    parser.add_argument("--sae-percentile", type=float, default=90.0)
    parser.add_argument("--combined-percentile", type=float, default=90.0)
    parser.add_argument("--sae-attention-gate-percentile", type=float, default=75.0)
    parser.add_argument("--attention-weight", type=float, default=0.6)
    parser.add_argument("--sae-weight", type=float, default=0.4)
    parser.add_argument("--min-selected-cells", type=int, default=6)
    parser.add_argument("--max-selected-cells", type=int, default=24)
    parser.add_argument("--target-importance-mass", type=float, default=0.45)
    parser.add_argument("--expand-selected-by-sae-neighbors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neighbor-similarity-threshold", type=float, default=0.90)
    parser.add_argument("--neighbor-min-combined-importance", type=float, default=0.08)
    parser.add_argument("--max-expanded-cells", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def crop_grid(arr: np.ndarray, *, x0: int, y0: int, side: int) -> np.ndarray:
    return np.asarray(arr[y0 : y0 + side, x0 : x0 + side]).copy()


def parse_hpv_label(value: str) -> int:
    s = str(value).strip().lower()
    if s in {"hpv_pos", "hpv+", "pos", "positive"}:
        return 1
    if s in {"hpv_neg", "hpv-", "neg", "negative"}:
        return 0
    return int(hpv_label_from_status(str(value)))


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_dir = args.source_region_dir
    region_img = Image.open(source_dir / "region.png").convert("RGB")
    full_zgrid = np.load(source_dir / "region_zgrid.npy")
    full_mask = np.load(source_dir / "valid_feature_mask.npy")
    source_meta = json.load(open(source_dir / "region_meta.json"))

    x0 = int(args.crop_cell_x0)
    y0 = int(args.crop_cell_y0)
    side = int(args.crop_cell_side)
    grid_step_px = int(source_meta.get("grid_step_px", 256))
    crop_px = side * grid_step_px
    crop_box = (x0 * grid_step_px, y0 * grid_step_px, (x0 + side) * grid_step_px, (y0 + side) * grid_step_px)

    local_img = region_img.crop(crop_box)
    local_zgrid = crop_grid(full_zgrid, x0=x0, y0=y0, side=side)
    local_mask = crop_grid(full_mask, x0=x0, y0=y0, side=side)

    valid_cells: list[tuple[int, int]] = []
    valid_features: list[np.ndarray] = []
    for gy in range(side):
        for gx in range(side):
            if int(local_mask[gy, gx]) > 0:
                valid_cells.append((gx, gy))
                valid_features.append(np.asarray(local_zgrid[gy, gx], dtype=np.float32))
    if not valid_cells:
        raise ValueError("Selected showcase crop has no valid feature cells.")
    valid_features_np = np.stack(valid_features, axis=0).astype(np.float32, copy=False)

    clam_model = load_clam_model(args.clam_ckpt, device=device)
    attention, pred, prob_pos = run_clam_attention(clam_model, valid_features_np, device=device, attn_class=str(args.clam_attn_class))

    label = int(source_meta.get("label", parse_hpv_label(source_meta["hpv_status"])))
    hpv_status = str(source_meta.get("hpv_status", "hpv_pos" if label == 1 else "hpv_neg"))
    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    prototype_vector = proto_by_latent[int(pos_latent if label == 1 else neg_latent)]
    sae_match, sae_gate_mask = compute_sae_prototype_scores_gated(
        sae_model=sae_model,
        features=valid_features_np,
        prototype_vector=prototype_vector,
        attention=np.asarray(attention, dtype=np.float32),
        gate_percentile=float(args.sae_attention_gate_percentile),
        device=device,
    )

    attn_norm = normalize_01(attention)
    sae_norm = normalize_01(sae_match)
    combined = float(args.attention_weight) * attn_norm + float(args.sae_weight) * sae_norm
    attn_threshold = float(np.percentile(attention, float(args.attention_percentile)))
    sae_threshold = float(np.percentile(sae_match, float(args.sae_percentile)))
    combined_threshold = float(np.percentile(combined, float(args.combined_percentile)))

    importance_by_cell = {cell: float(combined[idx]) for idx, cell in enumerate(valid_cells)}
    high_cells = [
        cell
        for idx, cell in enumerate(valid_cells)
        if float(attention[idx]) >= attn_threshold
        or bool(sae_gate_mask[idx] and float(sae_match[idx]) >= sae_threshold)
        or float(combined[idx]) >= combined_threshold
    ]
    seed_selected = select_cells_by_mass(
        high_cells,
        importance_by_cell,
        target_mass=float(args.target_importance_mass),
        min_cells=int(args.min_selected_cells),
        max_cells=int(args.max_selected_cells),
    )
    selected = list(seed_selected)
    if bool(args.expand_selected_by_sae_neighbors):
        selected = expand_selected_cells_by_sae_neighbors(
            seed_cells=seed_selected,
            valid_cells=valid_cells,
            region_features=valid_features_np,
            sae_model=sae_model,
            device=device,
            combined_by_cell=importance_by_cell,
            similarity_threshold=float(args.neighbor_similarity_threshold),
            min_combined_importance=float(args.neighbor_min_combined_importance),
            max_cells=int(args.max_expanded_cells),
        )

    attention_map = np.zeros((side, side), dtype=np.float32)
    sae_map = np.zeros((side, side), dtype=np.float32)
    combined_map = np.zeros((side, side), dtype=np.float32)
    for idx, (gx, gy) in enumerate(valid_cells):
        attention_map[gy, gx] = float(attention[idx])
        sae_map[gy, gx] = float(sae_match[idx])
        combined_map[gy, gx] = float(combined[idx])

    image_path = args.out_dir / "region.png"
    overlay_path = args.out_dir / "importance_overlay.png"
    zgrid_path = args.out_dir / "region_zgrid.npy"
    mask_path = args.out_dir / "valid_feature_mask.npy"
    cells_path = args.out_dir / "region_cells.png"
    npy_attention_path = args.out_dir / "attention_map.npy"
    npy_sae_path = args.out_dir / "sae_match_map.npy"
    npy_combined_path = args.out_dir / "combined_importance_map.npy"

    save_png(local_img, image_path)
    save_png(make_region_cells_preview(local_img, grid_step_px=grid_step_px), cells_path)
    save_png(
        draw_region_overlay(
            local_img,
            selected_cells=selected,
            seed_cells=seed_selected,
            high_cells=high_cells,
            grid_step_px=grid_step_px,
        ),
        overlay_path,
    )
    np.save(zgrid_path, local_zgrid)
    np.save(mask_path, local_mask)
    np.save(npy_attention_path, attention_map)
    np.save(npy_sae_path, sae_map)
    np.save(npy_combined_path, combined_map)

    meta = {
        "source_region_dir": str(source_dir),
        "source_region_id": str(source_meta.get("region_id", source_dir.name)),
        "crop_cell_x0": int(x0),
        "crop_cell_y0": int(y0),
        "crop_cell_side": int(side),
        "crop_box_px_xyxy": [int(v) for v in crop_box],
        "slide_key": str(source_meta.get("slide_key")),
        "label": int(label),
        "hpv_status": hpv_status,
        "pred": int(pred),
        "prob_pos": float(prob_pos),
        "attention_threshold": float(attn_threshold),
        "sae_threshold": float(sae_threshold),
        "combined_threshold": float(combined_threshold),
        "high_importance_cell_count": int(len(high_cells)),
        "seed_selected_cell_count": int(len(seed_selected)),
        "selected_cell_count": int(len(selected)),
        "high_importance_cells_local": encode_local_cells(high_cells),
        "seed_selected_cells_local": encode_local_cells(seed_selected),
        "selected_cells_local": encode_local_cells(selected),
        "selection_mode": "importance_mass_plus_sae_neighbor_expansion" if bool(args.expand_selected_by_sae_neighbors) else "importance_mass_only",
        "grid_step_px": int(grid_step_px),
        "feature_dim": int(local_zgrid.shape[-1]),
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "feature_grid_path": str(zgrid_path),
        "valid_mask_path": str(mask_path),
        "attention_map_path": str(npy_attention_path),
        "sae_match_map_path": str(npy_sae_path),
        "combined_importance_map_path": str(npy_combined_path),
        "command": " ".join(shlex.quote(item) for item in sys.argv),
    }
    write_json(args.out_dir / "region_meta.json", meta)
    write_json(
        args.out_dir / "progressive_edit_manifest.json",
        [
            {
                "run_id": f"{source_dir.name}__manual_showcase",
                "region_id": str(source_dir.name),
                "selection_mode": str(meta["selection_mode"]),
                "target_cells": [{"gx": int(tok.split(",")[0]), "gy": int(tok.split(",")[1])} for tok in str(meta["selected_cells_local"]).split(";") if tok],
                "seed_target_cells": [{"gx": int(tok.split(",")[0]), "gy": int(tok.split(",")[1])} for tok in str(meta["seed_selected_cells_local"]).split(";") if tok],
                "selector": "manual_showcase_clam_attention_plus_sae",
                "label": int(label),
                "direction_recommendation": "hpv_neg" if int(label) == 1 else "hpv_pos",
            }
        ],
    )


if __name__ == "__main__":
    main()
