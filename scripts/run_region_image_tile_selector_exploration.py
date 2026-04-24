#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_region_image_tile_selector import (  # type: ignore
    build_uni_grid_from_image,
    draw_selection_overlay,
    encode_cells,
    expand_selected_cells_by_feature_neighbors,
    expand_by_connected_support,
    fill_selection_gaps,
    infer_grid_shape_from_image_size,
    parse_label_inputs,
    patch_quality_mask,
)
from find_pathology_aware_2048_regions import (  # type: ignore
    compute_sae_prototype_scores_gated,
    expand_selected_cells_by_sae_neighbors,
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
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import make_region_cells_preview
from wsi_cf.eval.hnsc_hpv import load_prototypes, pick_prototype_latent
from wsi_cf.generation.pixcell import load_uni2

from utils.sae import load_sae_from_config  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explore multiple attention thresholds and expansion methods for a single region image. "
            "Encodes the image once, scores it once, then writes overlays and manifests for each combination."
        )
    )
    parser.add_argument("--region-image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--region-id", type=str, default="")
    parser.add_argument("--label", type=int, choices=[0, 1], required=True)
    parser.add_argument("--hpv-status", type=str, default="")
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--uni-dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("--clam-ckpt", type=Path, default=Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/s_0_checkpoint.pt"))
    parser.add_argument("--clam-attn-class", type=str, default="pred", choices=["pred", "pos", "neg"])
    parser.add_argument("--sae-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt"))
    parser.add_argument("--sae-cfg", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json"))
    parser.add_argument("--prototype-npz", type=Path, default=WSI_CF_ROOT / "artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz")
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--attention-percentiles", type=str, default="80,85,90,95")
    parser.add_argument("--sae-percentile", type=float, default=90.0)
    parser.add_argument("--combined-percentile", type=float, default=90.0)
    parser.add_argument("--sae-attention-gate-percentile", type=float, default=75.0)
    parser.add_argument("--attention-weight", type=float, default=0.6)
    parser.add_argument("--sae-weight", type=float, default=0.4)
    parser.add_argument("--target-importance-mass", type=float, default=0.45)
    parser.add_argument("--min-selected-cells", type=int, default=6)
    parser.add_argument("--max-selected-cells", type=int, default=24)
    parser.add_argument("--default-neighbor-similarity-space", type=str, default="feature", choices=["feature", "sae"])
    parser.add_argument("--neighbor-similarity-threshold", type=float, default=0.90)
    parser.add_argument("--neighbor-min-combined-importance", type=float, default=0.08)
    parser.add_argument("--max-expanded-cells", type=int, default=48)
    parser.add_argument("--connected-support-min-component", type=int, default=2)
    parser.add_argument("--connected-support-min-touching-neighbors", type=int, default=1)
    parser.add_argument("--gap-fill-min-neighbors", type=int, default=2)
    parser.add_argument("--gap-fill-max-iters", type=int, default=4)
    parser.add_argument(
        "--expansion-methods",
        type=str,
        default="seed_only,feature_neighbors,sae_neighbors,connected_support,feature_plus_connected,sae_plus_connected,full_stack_feature,full_stack_sae",
        help="Comma-separated subset of: seed_only, feature_neighbors, sae_neighbors, connected_support, feature_plus_connected, sae_plus_connected, full_stack_feature, full_stack_sae",
    )
    parser.add_argument("--min-tissue", type=float, default=0.35)
    parser.add_argument("--min-dark-fraction", type=float, default=0.02)
    parser.add_argument("--min-saturation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def parse_float_list(raw: str) -> list[float]:
    vals = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("Expected at least one numeric value")
    return vals


def parse_string_list(raw: str) -> list[str]:
    vals = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    if not vals:
        raise ValueError("Expected at least one method")
    return vals


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def apply_expansion_method(
    *,
    method: str,
    seed_selected: list[tuple[int, int]],
    high_cells: list[tuple[int, int]],
    valid_cells: list[tuple[int, int]],
    valid_features_np: np.ndarray,
    sae_model: torch.nn.Module,
    device: torch.device,
    importance_by_cell: dict[tuple[int, int], float],
    default_neighbor_similarity_space: str,
    neighbor_similarity_threshold: float,
    neighbor_min_combined_importance: float,
    max_expanded_cells: int,
    connected_support_min_component: int,
    connected_support_min_touching_neighbors: int,
    gap_fill_min_neighbors: int,
    gap_fill_max_iters: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], str]:
    selected = list(seed_selected)
    neighbor_added: list[tuple[int, int]] = []
    support_added: list[tuple[int, int]] = []
    gap_fill_added: list[tuple[int, int]] = []

    neighbor_space = str(default_neighbor_similarity_space)
    if method.startswith("feature_") or method.endswith("_feature"):
        neighbor_space = "feature"
    elif method.startswith("sae_") or method.endswith("_sae"):
        neighbor_space = "sae"

    if method in {"feature_neighbors", "sae_neighbors", "feature_plus_connected", "sae_plus_connected", "full_stack_feature", "full_stack_sae"}:
        if neighbor_space == "sae":
            neighbor_expanded = expand_selected_cells_by_sae_neighbors(
                seed_cells=selected,
                valid_cells=valid_cells,
                region_features=valid_features_np,
                sae_model=sae_model,
                device=device,
                combined_by_cell=importance_by_cell,
                similarity_threshold=float(neighbor_similarity_threshold),
                min_combined_importance=float(neighbor_min_combined_importance),
                max_cells=int(max_expanded_cells),
            )
        else:
            neighbor_expanded = expand_selected_cells_by_feature_neighbors(
                seed_cells=selected,
                valid_cells=valid_cells,
                region_features=valid_features_np,
                combined_by_cell=importance_by_cell,
                similarity_threshold=float(neighbor_similarity_threshold),
                min_combined_importance=float(neighbor_min_combined_importance),
                max_cells=int(max_expanded_cells),
            )
        neighbor_added = sorted(set(neighbor_expanded) - set(selected), key=lambda cell: (int(cell[1]), int(cell[0])))
        selected = sorted(set(neighbor_expanded), key=lambda cell: (int(cell[1]), int(cell[0])))

    if method in {"connected_support", "feature_plus_connected", "sae_plus_connected", "full_stack_feature", "full_stack_sae"}:
        support_added = expand_by_connected_support(
            selected_cells=selected,
            high_cells=high_cells,
            importance_by_cell=importance_by_cell,
            min_component_size=int(connected_support_min_component),
            min_touching_neighbors=int(connected_support_min_touching_neighbors),
        )
        selected = sorted(set(selected).union(support_added), key=lambda cell: (int(cell[1]), int(cell[0])))

    if method in {"full_stack_feature", "full_stack_sae"}:
        gap_fill_added = fill_selection_gaps(
            selected_cells=selected,
            valid_cells=set(valid_cells),
            min_neighbors=int(gap_fill_min_neighbors),
            max_iters=int(gap_fill_max_iters),
        )
        selected = sorted(set(selected).union(gap_fill_added), key=lambda cell: (int(cell[1]), int(cell[0])))

    return selected, neighbor_added, support_added, gap_fill_added, neighbor_space


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "experiment_args.json",
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "command": " ".join(shlex.quote(x) for x in sys.argv)},
    )

    label, hpv_status = parse_label_inputs(label=args.label, hpv_status=args.hpv_status)
    region_img = Image.open(args.region_image).convert("RGB")
    grid_h, grid_w = infer_grid_shape_from_image_size(width=region_img.size[0], height=region_img.size[1], grid_step_px=int(args.grid_step_px))
    region_id = str(args.region_id).strip() or args.region_image.stem

    valid_mask, quality_rows = patch_quality_mask(
        region_img,
        grid_step_px=int(args.grid_step_px),
        min_tissue=float(args.min_tissue),
        min_dark_fraction=float(args.min_dark_fraction),
        min_saturation_fraction=float(args.min_saturation_fraction),
    )
    valid_cells = [(gx, gy) for gy in range(grid_h) for gx in range(grid_w) if int(valid_mask[gy, gx]) > 0]
    if not valid_cells:
        raise ValueError("No valid cells were found under the current patch-quality thresholds.")

    uni_model, uni_transform = load_uni2(device=device)
    out_dtype = torch.float16 if str(args.uni_dtype) == "fp16" else torch.float32
    z_grid_t = build_uni_grid_from_image(
        region_img,
        uni_model=uni_model,
        uni_transform=uni_transform,
        grid_step_px=int(args.grid_step_px),
        device=device,
        out_dtype=out_dtype,
    )
    z_grid = z_grid_t.detach().float().cpu().numpy().astype(np.float32, copy=False)
    valid_features_np = np.stack([z_grid[gy, gx] for gx, gy in valid_cells], axis=0).astype(np.float32, copy=False)

    clam_model = load_clam_model(args.clam_ckpt, device=device)
    attention_score, pred, prob_pos = run_clam_attention(clam_model, valid_features_np, device=device, attn_class=str(args.clam_attn_class))
    attention_score = np.asarray(attention_score, dtype=np.float32)

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    prototype_vector = proto_by_latent[int(pos_latent if label == 1 else neg_latent)]
    sae_match, sae_gate_mask = compute_sae_prototype_scores_gated(
        sae_model=sae_model,
        features=valid_features_np,
        prototype_vector=prototype_vector,
        attention=attention_score,
        gate_percentile=float(args.sae_attention_gate_percentile),
        device=device,
    )
    attn_norm = normalize_01(attention_score)
    sae_norm = normalize_01(sae_match)
    combined = float(args.attention_weight) * attn_norm + float(args.sae_weight) * sae_norm
    importance_by_cell = {cell: float(combined[idx]) for idx, cell in enumerate(valid_cells)}

    # Save shared artifacts once at root.
    save_png(region_img, args.out_dir / "region.png")
    save_png(make_region_cells_preview(region_img, grid_step_px=int(args.grid_step_px)), args.out_dir / "region_cells.png")
    np.save(args.out_dir / "region_zgrid.npy", z_grid)
    np.save(args.out_dir / "valid_feature_mask.npy", valid_mask)
    attn_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    sae_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    combined_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    for idx, (gx, gy) in enumerate(valid_cells):
        attn_map[gy, gx] = float(attention_score[idx])
        sae_map[gy, gx] = float(sae_match[idx])
        combined_map[gy, gx] = float(combined[idx])
    np.save(args.out_dir / "attention_map.npy", attn_map)
    np.save(args.out_dir / "sae_match_map.npy", sae_map)
    np.save(args.out_dir / "combined_importance_map.npy", combined_map)

    attention_percentiles = parse_float_list(args.attention_percentiles)
    expansion_methods = parse_string_list(args.expansion_methods)
    allowed_methods = {"seed_only", "feature_neighbors", "sae_neighbors", "connected_support", "feature_plus_connected", "sae_plus_connected", "full_stack_feature", "full_stack_sae"}
    invalid = [m for m in expansion_methods if m not in allowed_methods]
    if invalid:
        raise ValueError(f"Unsupported expansion methods: {invalid}")

    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for attn_pct in attention_percentiles:
        attn_threshold = float(np.percentile(attention_score, float(attn_pct)))
        sae_threshold = float(np.percentile(sae_match, float(args.sae_percentile)))
        combined_threshold = float(np.percentile(combined, float(args.combined_percentile)))
        high_cells = [
            cell
            for idx, cell in enumerate(valid_cells)
            if float(attention_score[idx]) >= attn_threshold
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

        for method in expansion_methods:
            run_name = f"attn_p{str(attn_pct).replace('.', 'p')}__{method}"
            run_dir = args.out_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            selected, neighbor_added, support_added, gap_fill_added, neighbor_space = apply_expansion_method(
                method=method,
                seed_selected=seed_selected,
                high_cells=high_cells,
                valid_cells=valid_cells,
                valid_features_np=valid_features_np,
                sae_model=sae_model,
                device=device,
                importance_by_cell=importance_by_cell,
                default_neighbor_similarity_space=str(args.default_neighbor_similarity_space),
                neighbor_similarity_threshold=float(args.neighbor_similarity_threshold),
                neighbor_min_combined_importance=float(args.neighbor_min_combined_importance),
                max_expanded_cells=int(args.max_expanded_cells),
                connected_support_min_component=int(args.connected_support_min_component),
                connected_support_min_touching_neighbors=int(args.connected_support_min_touching_neighbors),
                gap_fill_min_neighbors=int(args.gap_fill_min_neighbors),
                gap_fill_max_iters=int(args.gap_fill_max_iters),
            )
            save_png(
                draw_selection_overlay(
                    region_img,
                    high_cells=high_cells,
                    seed_cells=seed_selected,
                    neighbor_expanded_cells=neighbor_added,
                    support_expanded_cells=support_added,
                    gap_fill_cells=gap_fill_added,
                    grid_step_px=int(args.grid_step_px),
                ),
                run_dir / "importance_overlay.png",
            )
            row = {
                "run_name": run_name,
                "attention_percentile": float(attn_pct),
                "expansion_method": method,
                "neighbor_similarity_space": neighbor_space,
                "pred": int(pred),
                "prob_pos": float(prob_pos),
                "high_importance_cell_count": int(len(high_cells)),
                "seed_selected_cell_count": int(len(seed_selected)),
                "neighbor_expanded_cell_count": int(len(neighbor_added)),
                "sae_expanded_cell_count": int(len(neighbor_added)) if neighbor_space == "sae" else 0,
                "connected_support_added_cell_count": int(len(support_added)),
                "gap_fill_added_cell_count": int(len(gap_fill_added)),
                "selected_cell_count": int(len(selected)),
                "high_importance_cells_local": encode_cells(high_cells),
                "seed_selected_cells_local": encode_cells(seed_selected),
                "selected_cells_local": encode_cells(selected),
                "overlay_path": str(run_dir / "importance_overlay.png"),
            }
            write_json(run_dir / "run_meta.json", row)
            rows.append(row)
            manifests.append(
                {
                    "run_id": f"{region_id}__{run_name}",
                    "region_id": str(region_id),
                    "selection_mode": f"exploration_{method}",
                    "target_cells": [{"gx": int(tok.split(",")[0]), "gy": int(tok.split(",")[1])} for tok in str(row["selected_cells_local"]).split(";") if tok],
                    "seed_target_cells": [{"gx": int(tok.split(",")[0]), "gy": int(tok.split(",")[1])} for tok in str(row["seed_selected_cells_local"]).split(";") if tok],
                    "selector": "region_image_exploration_clam_attention_plus_sae",
                    "label": int(label),
                    "direction_recommendation": "hpv_neg" if int(label) == 1 else "hpv_pos",
                    "attention_percentile": float(attn_pct),
                    "expansion_method": method,
                    "neighbor_similarity_space": neighbor_space,
                }
            )

    write_csv(args.out_dir / "exploration_summary.csv", rows)
    write_json(
        args.out_dir / "summary.json",
        {
            "region_id": region_id,
            "region_image": str(args.region_image),
            "label": int(label),
            "hpv_status": hpv_status,
            "pred": int(pred),
            "prob_pos": float(prob_pos),
            "grid_h": int(grid_h),
            "grid_w": int(grid_w),
            "attention_percentiles": attention_percentiles,
            "expansion_methods": expansion_methods,
            "n_runs": int(len(rows)),
            "summary_csv": str(args.out_dir / "exploration_summary.csv"),
            "manifest_json": str(args.out_dir / "progressive_edit_manifest.json"),
        },
    )
    write_json(args.out_dir / "progressive_edit_manifest.json", manifests)


if __name__ == "__main__":
    main()
