#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shlex
import sys
from pathlib import Path

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
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import parse_region_bank_csv
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, load_prototypes, pick_prototype_latent, run_mil_attention
from wsi_cf.eval.local_region import (
    build_local_attention_rows,
    count_high_attention_cells,
    flatten_region_zgrid,
    passes_label_confidence,
    replace_selected_cells_in_local_bag,
    select_attention_mass_cells,
    select_balanced_region_rows,
    select_attention_cells,
    summarize_region_classifier_runs,
)
from wsi_cf.generation.pixcell import (
    build_uni_grid_from_image,
    load_uni2,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
    vae_encode_auto,
)
from wsi_cf.steering.cell_selection import encode_cells
from wsi_cf.steering.progressive import (
    advance_progressive_state,
    build_history_aware_preserve_map,
    center_support_global_cells,
    draw_cells_overlay,
    enumerate_progressive_windows,
    make_initial_progressive_state,
    plan_progressive_steps,
    preserve_map_preview,
    window_local_cells,
)

from wsi_cf.common.paths import ensure_legacy_repo_root_on_path

ensure_legacy_repo_root_on_path()

from utils.sae import load_sae_from_config  # type: ignore
from utils.sae_edit import edit_uni_z_grid_with_sae  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a region-level attention-guided counterfactual loop: classify a prepared region, "
            "select high-attention cells, steer them with SAE + PixCell using the canonical progressive "
            "editor by default, re-encode the generated region with UNI, and rerun the classifier on "
            "the edited local bag."
        )
    )
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--experiment-mode", type=str, default="comprehensive", choices=["comprehensive", "simple"])
    parser.add_argument("--source-label", type=int, default=None, choices=[0, 1], help="Optional label filter")
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--final-regions-total", type=int, default=20)
    parser.add_argument("--final-regions-per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--mil-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt"))
    parser.add_argument("--selection-mode", type=str, default="attention_mass", choices=["topk", "percentile", "attention_mass"])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--attention-percentile", type=float, default=90.0)
    parser.add_argument("--min-cells", type=int, default=2)
    parser.add_argument("--max-cells", type=int, default=6)
    parser.add_argument("--target-attention-mass", type=float, default=0.35)
    parser.add_argument("--min-high-attention-cells", type=int, default=2)
    parser.add_argument("--max-high-attention-cells", type=int, default=6)
    parser.add_argument("--require-label-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-label-confidence", type=float, default=0.75)
    parser.add_argument("--candidate-scan-only", action="store_true", help="Write candidate_scan.csv/selected_regions.csv and stop before loading PixCell.")
    parser.add_argument("--editor-mode", type=str, default="progressive", choices=["progressive", "single_pass"])
    parser.add_argument("--direction-mode", type=str, default="opposite_label", choices=["opposite_label", "hpv_pos", "hpv_neg"])
    parser.add_argument("--pix-model-id", type=str, default="StonyBrook-CVLab/PixCell-1024")
    parser.add_argument("--pix-pipeline-id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae-model-id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae-subfolder", type=str, default="vae")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--prototype-strengths", type=str, default="0.4,0.8,1.2")
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--preserve-outside-latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-outside-strength", type=float, default=0.2)
    parser.add_argument("--preserve-edit-strength", type=float, default=0.05)
    parser.add_argument("--preserve-visited-strength", type=float, default=0.95)
    parser.add_argument("--preserve-fresh-context-strength", type=float, default=0.35)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.5)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=0.5)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--sae-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt"))
    parser.add_argument("--sae-cfg", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json"))
    parser.add_argument(
        "--prototype-npz",
        type=Path,
        default=Path("/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"),
    )
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--output-mode", type=str, default="debug", choices=["minimal", "debug"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


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


def make_edit_region_mask(*, width: int, height: int, cells: list[tuple[int, int]], grid_step_px: int) -> torch.Tensor:
    mask = torch.zeros((1, 1, int(height), int(width)), dtype=torch.float32)
    for gx, gy in cells:
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        x1 = min(int(width), x0 + int(grid_step_px))
        y1 = min(int(height), y0 + int(grid_step_px))
        mask[:, :, y0:y1, x0:x1] = 1.0
    return mask


def target_direction_for_row(*, source_label: int, direction_mode: str) -> str:
    mode = str(direction_mode)
    if mode == "hpv_pos":
        return "hpv_pos"
    if mode == "hpv_neg":
        return "hpv_neg"
    if int(source_label) == 0:
        return "hpv_pos"
    return "hpv_neg"


def target_prob_from_prob_pos(prob_pos: float, *, target_label: int) -> float:
    return float(prob_pos) if int(target_label) == 1 else float(1.0 - prob_pos)


def stable_region_seed(base_seed: int, region_id: str) -> int:
    acc = int(base_seed)
    for byte in str(region_id).encode("utf-8"):
        acc = (acc * 131 + int(byte)) % (2**31 - 1)
    return int(acc)


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


def commit_full_window(*, current_canvas: np.ndarray, steered_img: Image.Image, left: int, top: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = np.asarray(current_canvas, dtype=np.float32).copy()
    img = np.asarray(steered_img, dtype=np.float32) / 255.0
    height, width = img.shape[:2]
    dst_x0 = int(left)
    dst_y0 = int(top)
    dst_x1 = int(left) + int(width)
    dst_y1 = int(top) + int(height)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img
    return out, (dst_x0, dst_y0, dst_x1, dst_y1)


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


def editable_cells_for_progressive_grid(*, grid_w: int, grid_h: int, grid_step_px: int) -> set[tuple[int, int]]:
    windows = enumerate_progressive_windows(
        grid_w=int(grid_w),
        grid_h=int(grid_h),
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=int(grid_step_px),
    )
    return set().union(*(center_support_global_cells(window) for window in windows))


def parse_float_list(arg: str) -> list[float]:
    out: list[float] = []
    for item in str(arg).split(","):
        text = item.strip()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError("At least one numeric value is required")
    return out


def strength_tag(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def sae_strength_from_power(value: float) -> float:
    return float(max(0.0, min(float(value), 1.0)))


def condition_alpha_end_from_power(*, base_alpha_end: float, value: float) -> float:
    if float(value) <= 1.0:
        return float(base_alpha_end)
    return float(base_alpha_end) * float(value)


def summarize_by_strength(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    strengths = sorted({float(row["prototype_strength"]) for row in rows})
    for strength in strengths:
        subset = [row for row in rows if float(row["prototype_strength"]) == float(strength)]
        summary = summarize_region_classifier_runs(
            [
                {
                    "pred_before": row["pred_before"],
                    "pred_after": row["pred_after"],
                    "source_label": row["source_label"],
                    "target_label": row["target_label"],
                    "target_prob_before": row["target_prob_before"],
                    "target_prob_after": row["target_prob_after"],
                }
                for row in subset
            ]
        )
        full_delta = np.asarray([float(row["delta_target_prob_full_reencode"]) for row in subset], dtype=np.float32)
        flip = np.asarray([float(int(row["pred_after_full_reencode"]) == int(row["target_label"])) for row in subset], dtype=np.float32)
        summary["mean_delta_target_prob_full_reencode"] = float(full_delta.mean()) if full_delta.size else 0.0
        summary["target_pred_rate_after_full_reencode"] = float(flip.mean()) if flip.size else 0.0
        out[str(strength)] = summary
    return out


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    rows = parse_region_bank_csv(args.region_bank_csv)
    if args.source_label is not None:
        rows = [row for row in rows if int(row.label) == int(args.source_label)]
    rows = sorted(rows, key=lambda row: (int(row.label), str(row.slide_key), str(row.region_id)))
    if int(args.max_sources) > 0:
        rows = rows[: int(args.max_sources)]
    if not rows:
        raise SystemExit("No region-bank rows matched the requested filters.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "experiment_args.json", args_payload)
    write_json(out_dir / "command.json", {"argv": sys.argv, "command": shlex.join(sys.argv)})

    mil_model = build_mil_from_checkpoint(args.mil_ckpt, device=device)
    candidate_records: list[dict[str, object]] = []
    candidate_by_region_id: dict[str, dict[str, object]] = {}
    label_matches = 0
    for row in rows:
        source_zgrid = np.load(row.feature_grid_path).astype(np.float32, copy=False)
        grid_h, grid_w, feature_dim = int(source_zgrid.shape[0]), int(source_zgrid.shape[1]), int(source_zgrid.shape[2])
        features_before, _ = flatten_region_zgrid(source_zgrid)
        attention_before, pred_before, prob_pos_before = run_mil_attention(mil_model, features_before, device=device)
        if int(pred_before) == int(row.label):
            label_matches += 1
        allowed_cells = None
        if str(args.editor_mode) == "progressive":
            allowed_cells = editable_cells_for_progressive_grid(
                grid_w=int(grid_w),
                grid_h=int(grid_h),
                grid_step_px=int(row.grid_step_px),
            )
        label_ok, label_reason, true_conf = passes_label_confidence(
            label=int(row.label),
            pred=int(pred_before),
            prob_pos=float(prob_pos_before),
            min_confidence=float(args.min_label_confidence),
            require_label_match=bool(args.require_label_match),
        )
        high_count, high_threshold = count_high_attention_cells(
            attention=attention_before,
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            percentile=float(args.attention_percentile),
            allowed_cells=allowed_cells,
        )
        high_ok = int(args.min_high_attention_cells) <= int(high_count) <= int(args.max_high_attention_cells)
        if str(args.selection_mode) == "attention_mass":
            selected_cells, selected_mass, selection_ok, selection_reason = select_attention_mass_cells(
                attention=attention_before,
                grid_w=int(grid_w),
                grid_h=int(grid_h),
                target_mass=float(args.target_attention_mass),
                min_cells=int(args.min_cells),
                max_cells=int(args.max_cells),
                allowed_cells=allowed_cells,
            )
        else:
            selected_cells = select_attention_cells(
                attention=attention_before,
                grid_w=int(grid_w),
                grid_h=int(grid_h),
                mode=str(args.selection_mode),
                top_k=int(args.top_k),
                percentile=float(args.attention_percentile),
                min_cells=int(args.min_cells),
                max_cells=int(args.max_cells),
                allowed_cells=allowed_cells,
            )
            selected_mass = float(
                sum(float(attention_before[int(gy) * int(grid_w) + int(gx)]) for gx, gy in selected_cells)
            )
            selection_ok = int(args.min_cells) <= len(selected_cells) <= int(args.max_cells)
            selection_reason = "eligible_selected_cell_count" if selection_ok else "selected_cell_count_out_of_range"
        eligible = bool(label_ok and high_ok and selection_ok)
        if not label_ok:
            eligibility_reason = str(label_reason)
        elif not high_ok:
            eligibility_reason = "high_attention_count_out_of_range"
        elif not selection_ok:
            eligibility_reason = str(selection_reason)
        else:
            eligibility_reason = "eligible"
        record = {
            "region_id": str(row.region_id),
            "slide_key": str(row.slide_key),
            "source_label": int(row.label),
            "label": int(row.label),
            "hpv_status": str(row.hpv_status),
            "split": str(row.split),
            "pred_before": int(pred_before),
            "prob_pos_before": float(prob_pos_before),
            "true_label_confidence": float(true_conf),
            "label_match": bool(int(pred_before) == int(row.label)),
            "grid_h": int(grid_h),
            "grid_w": int(grid_w),
            "feature_dim": int(feature_dim),
            "allowed_edit_cell_count": int(len(allowed_cells)) if allowed_cells is not None else int(grid_h * grid_w),
            "attention_threshold": float(high_threshold),
            "high_attention_editable_cell_count": int(high_count),
            "selected_cells": encode_cells(selected_cells),
            "selected_cell_count": int(len(selected_cells)),
            "selected_attention_mass": float(selected_mass),
            "eligible": bool(eligible),
            "eligibility_reason": str(eligibility_reason),
        }
        candidate_records.append(record)
        candidate_by_region_id[str(row.region_id)] = {
            **record,
            "_attention_before": attention_before,
            "_selected_cells": selected_cells,
            "_allowed_cells": allowed_cells,
        }
    write_csv(out_dir / "candidate_scan.csv", candidate_records)

    if str(args.experiment_mode) == "comprehensive":
        per_label = int(args.final_regions_per_label)
        if int(args.final_regions_total) > 0:
            per_label = int(args.final_regions_total) // 2
        selected_candidate_records, eligible_counts = select_balanced_region_rows(candidate_records, per_label=per_label)
        if any(eligible_counts[label] < per_label for label in (0, 1)) and not bool(args.candidate_scan_only):
            write_csv(out_dir / "selected_regions.csv", selected_candidate_records)
            raise SystemExit(
                "Not enough eligible regions after candidate scan: "
                f"label_0={eligible_counts[0]}/{per_label}, label_1={eligible_counts[1]}/{per_label}. "
                f"See {out_dir / 'candidate_scan.csv'}"
            )
        selected_region_ids = {str(record["region_id"]) for record in selected_candidate_records}
        rows = [row for row in rows if str(row.region_id) in selected_region_ids]
        rows.sort(key=lambda row: (int(row.label), str(row.slide_key), str(row.region_id)))
        write_csv(out_dir / "selected_regions.csv", selected_candidate_records)
    else:
        selected_candidate_records = [record for record in candidate_records if bool(record.get("eligible", False))]
        if int(args.max_sources) > 0:
            selected_candidate_records = selected_candidate_records[: int(args.max_sources)]
        selected_region_ids = {str(record["region_id"]) for record in selected_candidate_records}
        rows = [row for row in rows if str(row.region_id) in selected_region_ids]
        write_csv(out_dir / "selected_regions.csv", selected_candidate_records)

    scan_summary = {
        "experiment_mode": str(args.experiment_mode),
        "n_candidates": int(len(candidate_records)),
        "label_match_rate": float(label_matches / max(1, len(candidate_records))),
        "eligible_count_by_label": {
            str(label): int(sum(1 for record in candidate_records if int(record["label"]) == label and bool(record["eligible"])))
            for label in (0, 1)
        },
        "selected_count_by_label": {
            str(label): int(sum(1 for record in selected_candidate_records if int(record["label"]) == label))
            for label in (0, 1)
        },
        "candidate_scan_csv": str(out_dir / "candidate_scan.csv"),
        "selected_regions_csv": str(out_dir / "selected_regions.csv"),
    }
    write_json(out_dir / "candidate_scan_summary.json", scan_summary)
    if bool(args.candidate_scan_only):
        print(f"[ok] wrote {out_dir / 'candidate_scan.csv'}")
        print(f"[ok] wrote {out_dir / 'selected_regions.csv'}")
        print(f"[ok] wrote {out_dir / 'candidate_scan_summary.json'}")
        return
    if not rows:
        raise SystemExit("No selected eligible rows remained for generation.")

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")

    custom_pipeline_ref = resolve_custom_pipeline_ref(str(args.pix_pipeline_id))
    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=AutoencoderKL.from_pretrained(
            args.vae_model_id,
            subfolder=args.vae_subfolder,
            torch_dtype=dtype,
        ),
        custom_pipeline=custom_pipeline_ref,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    pipeline.set_progress_bar_config(disable=False)

    patch_px, stride_px, cond_grid_side = resolve_pixcell_window_config(
        pix_model_id=str(args.pix_model_id),
        patch_px=0,
        stride_px=0,
    )

    uni_model, uni_transform = load_uni2(device=device)

    prototype_strengths = parse_float_list(str(args.prototype_strengths))
    run_items = [(row, float(strength)) for row in rows for strength in prototype_strengths]
    summary_rows: list[dict[str, object]] = []
    for row, prototype_strength in run_items:
        target_direction = target_direction_for_row(source_label=int(row.label), direction_mode=str(args.direction_mode))
        run_dir = out_dir / str(row.region_id) / str(target_direction) / f"strength_{strength_tag(float(prototype_strength))}"
        run_dir.mkdir(parents=True, exist_ok=True)
        final_img_path = run_dir / "generated.png"
        if bool(args.skip_existing) and final_img_path.exists():
            continue
        sae_strength = sae_strength_from_power(float(prototype_strength))
        condition_alpha_end = condition_alpha_end_from_power(
            base_alpha_end=float(args.mid_steer_alpha_end),
            value=float(prototype_strength),
        )

        source_img = load_image(row.image_path)
        source_zgrid = np.load(row.feature_grid_path).astype(np.float32, copy=False)
        grid_h, grid_w, feature_dim = int(source_zgrid.shape[0]), int(source_zgrid.shape[1]), int(source_zgrid.shape[2])

        features_before, tile_rows = flatten_region_zgrid(source_zgrid)
        candidate_record = candidate_by_region_id[str(row.region_id)]
        attention_before = np.asarray(candidate_record["_attention_before"], dtype=np.float32)
        pred_before = int(candidate_record["pred_before"])
        prob_pos_before = float(candidate_record["prob_pos_before"])
        allowed_cells = candidate_record["_allowed_cells"]
        selected_cells = list(candidate_record["_selected_cells"])  # type: ignore[arg-type]
        selected_set = {(int(gx), int(gy)) for gx, gy in selected_cells}
        for local_idx, tile_row in enumerate(tile_rows):
            tile_row["attention_before"] = float(attention_before[local_idx])
            tile_row["is_high_attention"] = bool((int(tile_row["cell_gx"]), int(tile_row["cell_gy"])) in selected_set)

        target_label = 1 if target_direction == "hpv_pos" else 0
        chosen_latent = int(pos_latent if target_direction == "hpv_pos" else neg_latent)

        proto_vec_t = torch.from_numpy(np.asarray(proto_by_latent[int(chosen_latent)], dtype=np.float32)).to(device=device)
        step_records: list[dict[str, object]] = []
        if str(args.editor_mode) == "progressive":
            planned_steps = plan_progressive_steps(
                target_cells=selected_cells,
                grid_w=int(grid_w),
                grid_h=int(grid_h),
                window_grid_side=4,
                stride_cells=2,
                grid_step_px=int(row.grid_step_px),
            )
            current_canvas = np.asarray(source_img, dtype=np.float32) / 255.0
            current_zgrid = np.asarray(source_zgrid, dtype=np.float32).copy()
            edited_condition_zgrid = np.asarray(source_zgrid, dtype=np.float32).copy()
            state = make_initial_progressive_state(target_cells=selected_cells)
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
                tile_mask = np.zeros(local_base_zgrid.shape[:2], dtype=np.float32)
                for lx, ly in local_edit_cells:
                    tile_mask[int(ly), int(lx)] = 1.0
                local_edit_t, _ = edit_uni_z_grid_with_sae(
                    sae_model=sae_model,
                    z_grid=local_edit_t,
                    target_latent_vector=proto_vec_t,
                    target_latent_vector_strength=float(sae_strength),
                    tile_mask=tile_mask,
                    blend=float(args.steer_blend),
                    keep_non_selected=True,
                    return_debug=False,
                )
                source_np = np.asarray(local_source_img, dtype=np.float32) / 255.0
                source_t = torch.from_numpy(source_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
                preserve_source_latents = vae_encode_auto(
                    pipeline.vae,
                    source_t,
                    use_tiled=False,
                    tile_img=0,
                    overlap_img=0,
                )
                preserve_map = build_history_aware_preserve_map(
                    width=int(local_source_img.size[0]),
                    height=int(local_source_img.size[1]),
                    grid_step_px=int(row.grid_step_px),
                    window=window,
                    edit_cells_global=list(step.edit_cells_global),
                    visited_cells_global=list(state.visited_cells),
                    preserve_edit_strength=float(args.preserve_edit_strength),
                    preserve_visited_strength=float(args.preserve_visited_strength),
                    preserve_fresh_context_strength=float(args.preserve_fresh_context_strength),
                ).to(device=device)
                generator = torch.Generator(device=device)
                generator.manual_seed(stable_region_seed(int(args.seed) + int(step.step_index), str(row.region_id)))
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
                current_canvas, commit_box = commit_full_window(
                    current_canvas=current_canvas,
                    steered_img=steered_img,
                    left=int(window.left),
                    top=int(window.top),
                )
                current_zgrid = update_full_zgrid_selected_cells(
                    full_zgrid=current_zgrid,
                    edited_local_zgrid=local_edit_t.detach().cpu().numpy().astype(np.float32),
                    gx0=int(window.gx0),
                    gy0=int(window.gy0),
                    selected_cells=local_edit_cells,
                )
                edited_condition_zgrid = update_full_zgrid_selected_cells(
                    full_zgrid=edited_condition_zgrid,
                    edited_local_zgrid=local_edit_t.detach().cpu().numpy().astype(np.float32),
                    gx0=int(window.gx0),
                    gy0=int(window.gy0),
                    selected_cells=local_edit_cells,
                )
                step_record = {
                    "step_index": int(step.step_index),
                    "window_id": str(window.window_id),
                    "gx0": int(window.gx0),
                    "gy0": int(window.gy0),
                    "left": int(window.left),
                    "top": int(window.top),
                    "edit_cells_global": [{"gx": int(gx), "gy": int(gy)} for gx, gy in step.edit_cells_global],
                    "edit_cells_local": [{"gx": int(gx), "gy": int(gy)} for gx, gy in local_edit_cells],
                    "commit_bounds_global": {
                        "x0": int(commit_box[0]),
                        "y0": int(commit_box[1]),
                        "x1": int(commit_box[2]),
                        "y1": int(commit_box[3]),
                    },
                }
                if str(args.output_mode) == "debug":
                    step_dir = run_dir / "steps" / f"step_{int(step.step_index) + 1:02d}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    save_png(local_source_img, step_dir / "source_window.png")
                    save_png(draw_cells_overlay(local_source_img, cells=local_edit_cells, grid_step_px=int(row.grid_step_px)), step_dir / "selected_cells_overlay.png")
                    save_png(preserve_map_preview(preserve_map), step_dir / "preserve_map.png")
                    save_png(steered_img, step_dir / "steered_window.png")
                    step_record["source_window_path"] = str(step_dir / "source_window.png")
                    step_record["selected_overlay_path"] = str(step_dir / "selected_cells_overlay.png")
                    step_record["preserve_map_path"] = str(step_dir / "preserve_map.png")
                    step_record["steered_window_path"] = str(step_dir / "steered_window.png")
                step_records.append(step_record)
                state = advance_progressive_state(
                    state,
                    window=window,
                    edit_cells_global=list(step.edit_cells_global),
                )
            generated_img = Image.fromarray((np.clip(current_canvas, 0.0, 1.0) * 255.0).astype(np.uint8))
            edited_zgrid = edited_condition_zgrid
        else:
            source_zgrid_t = torch.from_numpy(source_zgrid).to(device=device, dtype=torch.float32)
            tile_mask = np.zeros(source_zgrid.shape[:2], dtype=np.float32)
            for gx, gy in selected_cells:
                tile_mask[int(gy), int(gx)] = 1.0
            edited_zgrid_t, _ = edit_uni_z_grid_with_sae(
                sae_model=sae_model,
                z_grid=source_zgrid_t.clone(),
                target_latent_vector=proto_vec_t,
                target_latent_vector_strength=float(sae_strength),
                tile_mask=tile_mask,
                blend=float(args.steer_blend),
                keep_non_selected=True,
                return_debug=False,
            )
            edited_zgrid = edited_zgrid_t.detach().cpu().numpy().astype(np.float32, copy=False)
            preserve_source_latents = None
            edit_region_mask = None
            if bool(args.preserve_outside_latents):
                source_np = np.asarray(source_img, dtype=np.float32) / 255.0
                source_t = torch.from_numpy(source_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
                preserve_source_latents = vae_encode_auto(
                    pipeline.vae,
                    source_t,
                    use_tiled=max(source_img.size) > 1024,
                    tile_img=min(1024, max(source_img.size)),
                    overlap_img=min(128, max(source_img.size) // 8),
                )
                edit_region_mask = make_edit_region_mask(
                    width=int(source_img.size[0]),
                    height=int(source_img.size[1]),
                    cells=selected_cells,
                    grid_step_px=int(row.grid_step_px),
                ).to(device=device)
            generator = torch.Generator(device=device)
            generator.manual_seed(stable_region_seed(int(args.seed), str(row.region_id)))
            z_grid_base_pix = source_zgrid_t.to(device=device, dtype=dtype)
            z_grid_sched_pix = edited_zgrid_t.to(device=device, dtype=dtype)
            use_autocast = device.type == "cuda" and dtype == torch.float16
            ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
            with torch.inference_mode(), ctx:
                img_t = sample_large_pixcell_multidiffusion(
                    pipeline=pipeline,
                    z_grid=z_grid_base_pix,
                    scheduled_z_grid=z_grid_sched_pix,
                    condition_start_ratio=float(args.mid_steer_start_ratio),
                    condition_end_ratio=float(args.mid_steer_end_ratio),
                    condition_alpha_start=float(args.mid_steer_alpha_start),
                    condition_alpha_end=float(condition_alpha_end),
                    condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
                    out_h=int(source_img.size[1]),
                    out_w=int(source_img.size[0]),
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
                    use_tiled_vae_decode=max(source_img.size) > 1024,
                    decode_tile_lat=128,
                    decode_overlap_lat=16,
                    generator=generator,
                )
            generated_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            generated_img = Image.fromarray(generated_np)
        save_png(source_img, run_dir / "source_region_actual.png")
        save_png(draw_selected_cells_overlay(source_img, cells=selected_cells, grid_step_px=int(row.grid_step_px)), run_dir / "source_attention_overlay.png")
        save_png(generated_img, final_img_path)
        save_png(draw_selected_cells_overlay(generated_img, cells=selected_cells, grid_step_px=int(row.grid_step_px)), run_dir / "generated_attention_overlay.png")
        np.save(run_dir / "steered_condition_zgrid.npy", edited_zgrid)
        if str(args.output_mode) == "debug" and str(args.editor_mode) == "progressive":
            save_png(
                draw_cells_overlay(source_img, cells=selected_cells, grid_step_px=int(row.grid_step_px)),
                run_dir / "source_targets_overlay.png",
            )
            save_png(
                draw_cells_overlay(generated_img, cells=selected_cells, grid_step_px=int(row.grid_step_px)),
                run_dir / "generated_targets_overlay.png",
            )

        generated_zgrid = build_uni_grid_from_image(
            generated_img,
            uni_model=uni_model,
            uni_transform=uni_transform,
            grid_step_px=int(row.grid_step_px),
            device=device,
            out_dtype=dtype,
        ).detach().float().cpu().numpy().astype("float32", copy=False)
        np.save(run_dir / "generated_reencoded_zgrid.npy", generated_zgrid)

        features_after_full, _ = flatten_region_zgrid(generated_zgrid)
        features_after_selected, replacement_manifest = replace_selected_cells_in_local_bag(
            features_local=features_before,
            tile_rows=tile_rows,
            replacement_grid=generated_zgrid,
            selected_cells=selected_cells,
        )

        attention_after_full, pred_after_full, prob_pos_after_full = run_mil_attention(mil_model, features_after_full, device=device)
        attention_after_sel, pred_after_sel, prob_pos_after_sel = run_mil_attention(mil_model, features_after_selected, device=device)

        attention_rows = build_local_attention_rows(
            tile_rows=tile_rows,
            attention_before=attention_before,
            attention_after=attention_after_sel,
            selected_cells=selected_cells,
        )
        write_csv(run_dir / "local_attention_rows.csv", attention_rows)
        write_csv(run_dir / "replacement_manifest.csv", replacement_manifest)

        target_prob_before = target_prob_from_prob_pos(prob_pos_before, target_label=target_label)
        target_prob_after_selected = target_prob_from_prob_pos(prob_pos_after_sel, target_label=target_label)
        target_prob_after_full = target_prob_from_prob_pos(prob_pos_after_full, target_label=target_label)

        run_manifest = {
            "region_id": str(row.region_id),
            "slide_key": str(row.slide_key),
            "source_label": int(row.label),
            "grid_shape": [int(grid_h), int(grid_w)],
            "feature_dim": int(feature_dim),
            "editor_mode": str(args.editor_mode),
            "selection_mode": str(args.selection_mode),
            "target_attention_mass": float(args.target_attention_mass),
            "selected_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in selected_cells],
            "selected_cells_encoded": encode_cells(selected_cells),
            "selected_cell_count": int(len(selected_cells)),
            "selected_attention_mass": float(candidate_record["selected_attention_mass"]),
            "high_attention_editable_cell_count": int(candidate_record["high_attention_editable_cell_count"]),
            "allowed_edit_cells_encoded": encode_cells(sorted(allowed_cells, key=lambda item: (item[1], item[0]))) if allowed_cells is not None else "",
            "target_direction": str(target_direction),
            "target_label": int(target_label),
            "prototype_latent": int(chosen_latent),
            "prototype_strength": float(prototype_strength),
            "sae_latent_strength": float(sae_strength),
            "condition_alpha_end_effective": float(condition_alpha_end),
            "pred_before": int(pred_before),
            "prob_pos_before": float(prob_pos_before),
            "pred_after_selected_replace": int(pred_after_sel),
            "prob_pos_after_selected_replace": float(prob_pos_after_sel),
            "pred_after_full_reencode": int(pred_after_full),
            "prob_pos_after_full_reencode": float(prob_pos_after_full),
            "target_prob_before": float(target_prob_before),
            "target_prob_after_selected_replace": float(target_prob_after_selected),
            "target_prob_after_full_reencode": float(target_prob_after_full),
            "paths": {
                "source_image": str(run_dir / "source_region_actual.png"),
                "source_overlay": str(run_dir / "source_attention_overlay.png"),
                "generated_image": str(final_img_path),
                "generated_overlay": str(run_dir / "generated_attention_overlay.png"),
                "steered_condition_zgrid": str(run_dir / "steered_condition_zgrid.npy"),
                "generated_reencoded_zgrid": str(run_dir / "generated_reencoded_zgrid.npy"),
                "local_attention_rows": str(run_dir / "local_attention_rows.csv"),
                "replacement_manifest": str(run_dir / "replacement_manifest.csv"),
            },
            "window_history": step_records,
        }
        write_json(run_dir / "run_manifest.json", run_manifest)

        summary_rows.append(
            {
                "region_id": str(row.region_id),
                "slide_key": str(row.slide_key),
                "source_label": int(row.label),
                "target_direction": str(target_direction),
                "target_label": int(target_label),
                "prototype_strength": float(prototype_strength),
                "sae_latent_strength": float(sae_strength),
                "condition_alpha_end_effective": float(condition_alpha_end),
                "selected_cells": encode_cells(selected_cells),
                "selected_cell_count": int(len(selected_cells)),
                "selected_attention_mass": float(candidate_record["selected_attention_mass"]),
                "high_attention_editable_cell_count": int(candidate_record["high_attention_editable_cell_count"]),
                "pred_before": int(pred_before),
                "prob_pos_before": float(prob_pos_before),
                "pred_after": int(pred_after_sel),
                "prob_pos_after": float(prob_pos_after_sel),
                "pred_after_full_reencode": int(pred_after_full),
                "prob_pos_after_full_reencode": float(prob_pos_after_full),
                "target_prob_before": float(target_prob_before),
                "target_prob_after": float(target_prob_after_selected),
                "target_prob_after_full_reencode": float(target_prob_after_full),
                "delta_target_prob": float(target_prob_after_selected - target_prob_before),
                "delta_target_prob_full_reencode": float(target_prob_after_full - target_prob_before),
                "run_dir": str(run_dir),
            }
        )

    write_csv(out_dir / "region_results.csv", summary_rows)
    summary = summarize_region_classifier_runs(
        [
            {
                "pred_before": row["pred_before"],
                "pred_after": row["pred_after"],
                "source_label": row["source_label"],
                "target_label": row["target_label"],
                "target_prob_before": row["target_prob_before"],
                "target_prob_after": row["target_prob_after"],
            }
            for row in summary_rows
        ]
    )
    summary["n_processed"] = int(len(summary_rows))
    summary["editor_mode"] = str(args.editor_mode)
    summary["selection_mode"] = str(args.selection_mode)
    summary["direction_mode"] = str(args.direction_mode)
    summary["pix_custom_pipeline_resolved"] = str(custom_pipeline_ref)
    summary["region_results_csv"] = str(out_dir / "region_results.csv")
    summary_by_strength = summarize_by_strength(summary_rows)
    summary["summary_by_strength_json"] = str(out_dir / "summary_by_strength.json")
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "summary_by_strength.json", summary_by_strength)
    print(f"[ok] wrote {out_dir / 'region_results.csv'}")
    print(f"[ok] wrote {out_dir / 'summary.json'}")
    print(f"[ok] wrote {out_dir / 'summary_by_strength.json'}")


if __name__ == "__main__":
    main()
