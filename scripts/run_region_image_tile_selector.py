#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from wsi_cf.data.region_bank import make_region_cells_preview
from wsi_cf.data.slides import quick_region_quality_metrics
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, load_prototypes, pick_prototype_latent, run_mil_attention
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from find_pathology_aware_2048_regions import (  # type: ignore
    compute_sae_prototype_scores_gated,
    expand_selected_cells_by_sae_neighbors,
    hpv_label_from_status,
    load_clam_model,
    normalize_01,
    run_clam_attention,
    select_cells_by_mass,
)

ensure_legacy_repo_root_on_path()

from utils.sae import load_sae_from_config  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start from a region image, encode a fresh UNI grid, score tiles with local attention "
            "and SAE importance, and output a broader tile-selection overlay for editing."
        )
    )
    parser.add_argument("--region-image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--region-id", type=str, default="")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--hpv-status", type=str, default="")
    parser.add_argument("--model-backend", type=str, default="clam", choices=["clam", "mil", "both"])
    parser.add_argument("--clam-ckpt", type=Path, default=Path("/common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/s_0_checkpoint.pt"))
    parser.add_argument("--clam-attn-class", type=str, default="pred", choices=["pred", "pos", "neg"])
    parser.add_argument("--mil-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt"))
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--uni-dtype", type=str, default="fp32", choices=["fp16", "fp32"])
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
    parser.add_argument("--neighbor-similarity-space", type=str, default="feature", choices=["feature", "sae"])
    parser.add_argument("--neighbor-similarity-threshold", type=float, default=0.90)
    parser.add_argument("--neighbor-min-combined-importance", type=float, default=0.08)
    parser.add_argument("--max-expanded-cells", type=int, default=48)
    parser.add_argument("--connected-support-min-component", type=int, default=2)
    parser.add_argument("--connected-support-min-touching-neighbors", type=int, default=1)
    parser.add_argument("--fill-gaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gap-fill-min-neighbors", type=int, default=3)
    parser.add_argument("--gap-fill-max-iters", type=int, default=4)
    parser.add_argument("--min-tissue", type=float, default=0.35)
    parser.add_argument("--min-dark-fraction", type=float, default=0.02)
    parser.add_argument("--min-saturation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def parse_label_inputs(*, label: int | None, hpv_status: str) -> tuple[int, str]:
    if label is not None:
        return int(label), "hpv_pos" if int(label) == 1 else "hpv_neg"
    if str(hpv_status).strip():
        parsed = int(hpv_label_from_status(str(hpv_status)))
        return parsed, "hpv_pos" if parsed == 1 else "hpv_neg"
    raise ValueError("Provide either --label or --hpv-status so SAE current-label importance can be computed.")


def infer_grid_shape_from_image_size(*, width: int, height: int, grid_step_px: int) -> tuple[int, int]:
    if int(grid_step_px) <= 0:
        raise ValueError("grid_step_px must be > 0")
    if int(width) % int(grid_step_px) != 0 or int(height) % int(grid_step_px) != 0:
        raise ValueError("Image width and height must be divisible by grid_step_px")
    return int(height) // int(grid_step_px), int(width) // int(grid_step_px)


def encode_cells(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> str:
    ordered = sorted({(int(gx), int(gy)) for gx, gy in cells}, key=lambda cell: (int(cell[1]), int(cell[0])))
    return ";".join(f"{int(gx)},{int(gy)}" for gx, gy in ordered)


def patch_quality_mask(
    img: Image.Image,
    *,
    grid_step_px: int,
    min_tissue: float,
    min_dark_fraction: float,
    min_saturation_fraction: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    grid_h, grid_w = infer_grid_shape_from_image_size(width=img.size[0], height=img.size[1], grid_step_px=int(grid_step_px))
    mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
    rows: list[dict[str, float]] = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            crop = img.crop(
                (
                    int(gx) * int(grid_step_px),
                    int(gy) * int(grid_step_px),
                    (int(gx) + 1) * int(grid_step_px),
                    (int(gy) + 1) * int(grid_step_px),
                )
            )
            quality = quick_region_quality_metrics(crop)
            valid = (
                float(quality["tissue_score"]) >= float(min_tissue)
                and float(quality["dark_fraction"]) >= float(min_dark_fraction)
                and float(quality["saturation_fraction"]) >= float(min_saturation_fraction)
            )
            if valid:
                mask[gy, gx] = 1
            rows.append(
                {
                    "cell_gx": int(gx),
                    "cell_gy": int(gy),
                    "tissue_score": float(quality["tissue_score"]),
                    "dark_fraction": float(quality["dark_fraction"]),
                    "saturation_fraction": float(quality["saturation_fraction"]),
                    "is_valid": bool(valid),
                }
            )
    return mask, rows


def connected_components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set((int(x), int(y)) for x, y in cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        queue: deque[tuple[int, int]] = deque([start])
        comp: list[tuple[int, int]] = [start]
        while queue:
            x, y = queue.popleft()
            for nb in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nb in remaining:
                    remaining.remove(nb)
                    queue.append(nb)
                    comp.append(nb)
        components.append(sorted(comp, key=lambda cell: (int(cell[1]), int(cell[0]))))
    components.sort(key=lambda comp: (-len(comp), comp[0][1], comp[0][0]))
    return components


def expand_by_connected_support(
    *,
    selected_cells: list[tuple[int, int]],
    high_cells: list[tuple[int, int]],
    importance_by_cell: dict[tuple[int, int], float],
    min_component_size: int,
    min_touching_neighbors: int,
) -> list[tuple[int, int]]:
    selected = {(int(gx), int(gy)) for gx, gy in selected_cells}
    high_set = {(int(gx), int(gy)) for gx, gy in high_cells}
    added: set[tuple[int, int]] = set()
    for comp in connected_components(high_set):
        comp_set = set(comp)
        if len(comp_set) < int(min_component_size):
            continue
        touch_count = 0
        for cell in comp_set:
            if any(nb in selected for nb in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1))):
                touch_count += 1
        if touch_count >= int(min_touching_neighbors):
            added.update(comp_set - selected)
    return sorted(added, key=lambda cell: (-float(importance_by_cell.get(cell, 0.0)), int(cell[1]), int(cell[0])))


def fill_selection_gaps(
    *,
    selected_cells: list[tuple[int, int]],
    valid_cells: set[tuple[int, int]],
    min_neighbors: int,
    max_iters: int,
) -> list[tuple[int, int]]:
    selected = {(int(gx), int(gy)) for gx, gy in selected_cells}
    added_total: set[tuple[int, int]] = set()
    for _ in range(max(0, int(max_iters))):
        added_this_round: set[tuple[int, int]] = set()
        for cell in sorted(valid_cells - selected, key=lambda item: (int(item[1]), int(item[0]))):
            neighbors = ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1))
            count = sum(1 for nb in neighbors if nb in selected)
            bridge = ((cell[0] - 1, cell[1]) in selected and (cell[0] + 1, cell[1]) in selected) or (
                (cell[0], cell[1] - 1) in selected and (cell[0], cell[1] + 1) in selected
            )
            if count >= int(min_neighbors) or bridge:
                added_this_round.add(cell)
        if not added_this_round:
            break
        selected.update(added_this_round)
        added_total.update(added_this_round)
    return sorted(added_total, key=lambda cell: (int(cell[1]), int(cell[0])))


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    rows = np.asarray(arr, dtype=np.float32)
    denom = np.linalg.norm(rows, axis=1, keepdims=True)
    denom = np.clip(denom, a_min=1e-8, a_max=None)
    return rows / denom


def expand_selected_cells_by_feature_neighbors(
    *,
    seed_cells: list[tuple[int, int]],
    valid_cells: list[tuple[int, int]],
    region_features: np.ndarray,
    combined_by_cell: dict[tuple[int, int], float],
    similarity_threshold: float,
    min_combined_importance: float,
    max_cells: int,
) -> list[tuple[int, int]]:
    if not seed_cells:
        return []
    valid_order = [tuple((int(x), int(y))) for x, y in valid_cells]
    emb = _normalize_rows(np.asarray(region_features, dtype=np.float32))
    emb_by_cell = {cell: emb[idx] for idx, cell in enumerate(valid_order)}
    valid_set = set(valid_order)
    expanded: set[tuple[int, int]] = {tuple((int(x), int(y))) for x, y in seed_cells}

    while len(expanded) < int(max_cells):
        candidates: list[tuple[float, float, int, int]] = []
        for cell in sorted(expanded, key=lambda item: (int(item[1]), int(item[0]))):
            for nb in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)):
                if nb not in valid_set or nb in expanded:
                    continue
                if float(combined_by_cell.get(nb, 0.0)) < float(min_combined_importance):
                    continue
                sim = max(float(np.dot(emb_by_cell[nb], emb_by_cell[src])) for src in expanded)
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


def draw_selection_overlay(
    img: Image.Image,
    *,
    high_cells: list[tuple[int, int]],
    seed_cells: list[tuple[int, int]],
    neighbor_expanded_cells: list[tuple[int, int]],
    support_expanded_cells: list[tuple[int, int]],
    gap_fill_cells: list[tuple[int, int]],
    grid_step_px: int,
) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    high_set = {(int(gx), int(gy)) for gx, gy in high_cells}
    seed_set = {(int(gx), int(gy)) for gx, gy in seed_cells}
    neighbor_set = {(int(gx), int(gy)) for gx, gy in neighbor_expanded_cells}
    support_set = {(int(gx), int(gy)) for gx, gy in support_expanded_cells}
    fill_set = {(int(gx), int(gy)) for gx, gy in gap_fill_cells}

    for gx, gy in sorted(high_set, key=lambda cell: (int(cell[1]), int(cell[0]))):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        draw.rectangle(
            [x0, y0, x0 + int(grid_step_px) - 1, y0 + int(grid_step_px) - 1],
            fill=(255, 165, 0, 60),
            outline=(255, 165, 0, 190),
            width=2,
        )
    for gx, gy in sorted(neighbor_set, key=lambda cell: (int(cell[1]), int(cell[0]))):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        draw.rectangle(
            [x0, y0, x0 + int(grid_step_px) - 1, y0 + int(grid_step_px) - 1],
            fill=(0, 255, 255, 55),
            outline=(0, 255, 255, 220),
            width=4,
        )
    for gx, gy in sorted(support_set, key=lambda cell: (int(cell[1]), int(cell[0]))):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        draw.rectangle(
            [x0, y0, x0 + int(grid_step_px) - 1, y0 + int(grid_step_px) - 1],
            fill=(50, 220, 80, 55),
            outline=(50, 220, 80, 220),
            width=4,
        )
    for gx, gy in sorted(fill_set, key=lambda cell: (int(cell[1]), int(cell[0]))):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        draw.rectangle(
            [x0, y0, x0 + int(grid_step_px) - 1, y0 + int(grid_step_px) - 1],
            fill=(0, 180, 0, 40),
            outline=(0, 180, 0, 210),
            width=4,
        )
    for rank, (gx, gy) in enumerate(sorted(seed_set, key=lambda cell: (int(cell[1]), int(cell[0]))), start=1):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        draw.rectangle(
            [x0, y0, x0 + int(grid_step_px) - 1, y0 + int(grid_step_px) - 1],
            outline=(255, 255, 0, 255),
            width=6,
        )
        draw.text((x0 + 8, y0 + 8), f"{rank}", fill=(255, 0, 0, 255))
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    label, hpv_status = parse_label_inputs(label=args.label, hpv_status=args.hpv_status)
    region_img = Image.open(args.region_image).convert("RGB")
    grid_h, grid_w = infer_grid_shape_from_image_size(width=region_img.size[0], height=region_img.size[1], grid_step_px=int(args.grid_step_px))

    valid_mask, quality_rows = patch_quality_mask(
        region_img,
        grid_step_px=int(args.grid_step_px),
        min_tissue=float(args.min_tissue),
        min_dark_fraction=float(args.min_dark_fraction),
        min_saturation_fraction=float(args.min_saturation_fraction),
    )
    if int(valid_mask.sum()) == 0:
        raise ValueError("No valid tissue-rich cells were found in the region image under the current patch-quality thresholds.")

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
    del z_grid_t

    valid_cells = [(gx, gy) for gy in range(grid_h) for gx in range(grid_w) if int(valid_mask[gy, gx]) > 0]
    valid_features_np = np.stack([z_grid[gy, gx] for gx, gy in valid_cells], axis=0).astype(np.float32, copy=False)

    attention_backend = str(args.model_backend)
    attention_maps: dict[str, np.ndarray] = {}
    pred_rows: dict[str, dict[str, float | int]] = {}
    if attention_backend in {"clam", "both"}:
        clam_model = load_clam_model(args.clam_ckpt, device=device)
        clam_attention, clam_pred, clam_prob_pos = run_clam_attention(clam_model, valid_features_np, device=device, attn_class=str(args.clam_attn_class))
        attention_maps["clam"] = np.asarray(clam_attention, dtype=np.float32)
        pred_rows["clam"] = {"pred": int(clam_pred), "prob_pos": float(clam_prob_pos)}
    if attention_backend in {"mil", "both"}:
        mil_model = build_mil_from_checkpoint(args.mil_ckpt, device=device)
        mil_attention, mil_pred, mil_prob_pos = run_mil_attention(mil_model, valid_features_np, device=device)
        attention_maps["mil"] = np.asarray(mil_attention, dtype=np.float32)
        pred_rows["mil"] = {"pred": int(mil_pred), "prob_pos": float(mil_prob_pos)}
    if not attention_maps:
        raise ValueError(f"Unsupported model_backend: {args.model_backend}")

    if attention_backend == "both":
        attention_score = np.mean(np.stack([normalize_01(values) for values in attention_maps.values()], axis=0), axis=0).astype(np.float32, copy=False)
        pred = int(pred_rows["clam"]["pred"])
        prob_pos = float(pred_rows["clam"]["prob_pos"])
    else:
        attention_score = next(iter(attention_maps.values()))
        only = next(iter(pred_rows.values()))
        pred = int(only["pred"])
        prob_pos = float(only["prob_pos"])

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    prototype_vector = proto_by_latent[int(pos_latent if label == 1 else neg_latent)]
    sae_match, sae_gate_mask = compute_sae_prototype_scores_gated(
        sae_model=sae_model,
        features=valid_features_np,
        prototype_vector=prototype_vector,
        attention=np.asarray(attention_score, dtype=np.float32),
        gate_percentile=float(args.sae_attention_gate_percentile),
        device=device,
    )

    attn_norm = normalize_01(attention_score)
    sae_norm = normalize_01(sae_match)
    combined = float(args.attention_weight) * attn_norm + float(args.sae_weight) * sae_norm
    attn_threshold = float(np.percentile(attention_score, float(args.attention_percentile)))
    sae_threshold = float(np.percentile(sae_match, float(args.sae_percentile)))
    combined_threshold = float(np.percentile(combined, float(args.combined_percentile)))

    importance_by_cell = {cell: float(combined[idx]) for idx, cell in enumerate(valid_cells)}
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
    selected = list(seed_selected)
    neighbor_added: list[tuple[int, int]] = []
    if bool(args.expand_selected_by_sae_neighbors):
        if str(args.neighbor_similarity_space) == "sae":
            neighbor_expanded = expand_selected_cells_by_sae_neighbors(
                seed_cells=selected,
                valid_cells=valid_cells,
                region_features=valid_features_np,
                sae_model=sae_model,
                device=device,
                combined_by_cell=importance_by_cell,
                similarity_threshold=float(args.neighbor_similarity_threshold),
                min_combined_importance=float(args.neighbor_min_combined_importance),
                max_cells=int(args.max_expanded_cells),
            )
        else:
            neighbor_expanded = expand_selected_cells_by_feature_neighbors(
                seed_cells=selected,
                valid_cells=valid_cells,
                region_features=valid_features_np,
                combined_by_cell=importance_by_cell,
                similarity_threshold=float(args.neighbor_similarity_threshold),
                min_combined_importance=float(args.neighbor_min_combined_importance),
                max_cells=int(args.max_expanded_cells),
            )
        neighbor_added = sorted(set(neighbor_expanded) - set(selected), key=lambda cell: (int(cell[1]), int(cell[0])))
        selected = sorted(set(neighbor_expanded), key=lambda cell: (int(cell[1]), int(cell[0])))

    support_added = expand_by_connected_support(
        selected_cells=selected,
        high_cells=high_cells,
        importance_by_cell=importance_by_cell,
        min_component_size=int(args.connected_support_min_component),
        min_touching_neighbors=int(args.connected_support_min_touching_neighbors),
    )
    selected = sorted(set(selected).union(support_added), key=lambda cell: (int(cell[1]), int(cell[0])))

    gap_fill_added: list[tuple[int, int]] = []
    if bool(args.fill_gaps):
        gap_fill_added = fill_selection_gaps(
            selected_cells=selected,
            valid_cells=set(valid_cells),
            min_neighbors=int(args.gap_fill_min_neighbors),
            max_iters=int(args.gap_fill_max_iters),
        )
        selected = sorted(set(selected).union(gap_fill_added), key=lambda cell: (int(cell[1]), int(cell[0])))

    attention_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    sae_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    combined_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    for idx, (gx, gy) in enumerate(valid_cells):
        attention_map[gy, gx] = float(attention_score[idx])
        sae_map[gy, gx] = float(sae_match[idx])
        combined_map[gy, gx] = float(combined[idx])

    region_id = str(args.region_id).strip() or args.region_image.stem
    save_png(region_img, args.out_dir / "region.png")
    save_png(make_region_cells_preview(region_img, grid_step_px=int(args.grid_step_px)), args.out_dir / "region_cells.png")
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
        args.out_dir / "importance_overlay.png",
    )
    np.save(args.out_dir / "region_zgrid.npy", z_grid)
    np.save(args.out_dir / "valid_feature_mask.npy", valid_mask)
    np.save(args.out_dir / "attention_map.npy", attention_map)
    np.save(args.out_dir / "sae_match_map.npy", sae_map)
    np.save(args.out_dir / "combined_importance_map.npy", combined_map)

    meta: dict[str, Any] = {
        "region_id": region_id,
        "region_image": str(args.region_image),
        "label": int(label),
        "hpv_status": str(hpv_status),
        "model_backend": str(args.model_backend),
        "pred": int(pred),
        "prob_pos": float(prob_pos),
        "grid_step_px": int(args.grid_step_px),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "feature_dim": int(z_grid.shape[-1]),
        "neighbor_similarity_space": str(args.neighbor_similarity_space),
        "attention_weight": float(args.attention_weight),
        "sae_weight": float(args.sae_weight),
        "attention_threshold": float(attn_threshold),
        "sae_threshold": float(sae_threshold),
        "combined_threshold": float(combined_threshold),
        "high_importance_cell_count": int(len(high_cells)),
        "seed_selected_cell_count": int(len(seed_selected)),
        "neighbor_expanded_cell_count": int(len(neighbor_added)),
        "sae_expanded_cell_count": int(len(neighbor_added)) if str(args.neighbor_similarity_space) == "sae" else 0,
        "connected_support_added_cell_count": int(len(support_added)),
        "gap_fill_added_cell_count": int(len(gap_fill_added)),
        "selected_cell_count": int(len(selected)),
        "high_importance_cells_local": encode_cells(high_cells),
        "seed_selected_cells_local": encode_cells(seed_selected),
        "neighbor_expanded_cells_local": encode_cells(neighbor_added),
        "sae_expanded_cells_local": encode_cells(neighbor_added) if str(args.neighbor_similarity_space) == "sae" else "",
        "connected_support_added_cells_local": encode_cells(support_added),
        "gap_fill_added_cells_local": encode_cells(gap_fill_added),
        "selected_cells_local": encode_cells(selected),
        "selection_mode": "importance_mass_plus_sae_neighbor_plus_connected_support_plus_gap_fill",
        "image_path": str(args.out_dir / "region.png"),
        "overlay_path": str(args.out_dir / "importance_overlay.png"),
        "feature_grid_path": str(args.out_dir / "region_zgrid.npy"),
        "valid_mask_path": str(args.out_dir / "valid_feature_mask.npy"),
        "attention_map_path": str(args.out_dir / "attention_map.npy"),
        "sae_match_map_path": str(args.out_dir / "sae_match_map.npy"),
        "combined_importance_map_path": str(args.out_dir / "combined_importance_map.npy"),
        "quality_rows": quality_rows,
        "backend_predictions": pred_rows,
        "command": " ".join(shlex.quote(item) for item in sys.argv),
    }
    write_json(args.out_dir / "region_meta.json", meta)
    write_json(
        args.out_dir / "progressive_edit_manifest.json",
        [
            {
                "run_id": f"{region_id}__image_first_tile_selection",
                "region_id": str(region_id),
                "selection_mode": str(meta["selection_mode"]),
                "target_cells": [{"gx": int(tok.split(",")[0]), "gy": int(tok.split(",")[1])} for tok in str(meta["selected_cells_local"]).split(";") if tok],
                "seed_target_cells": [{"gx": int(tok.split(",")[0]), "gy": int(tok.split(",")[1])} for tok in str(meta["seed_selected_cells_local"]).split(";") if tok],
                "selector": "region_image_clam_attention_plus_sae",
                "label": int(label),
                "direction_recommendation": "hpv_neg" if int(label) == 1 else "hpv_pos",
            }
        ],
    )


if __name__ == "__main__":
    main()
