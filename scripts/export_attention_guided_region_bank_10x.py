#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shlex
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.attention_proposals import iter_attention_region_candidates, select_attention_region
from wsi_cf.data.region_bank import export_region_bundle, save_label_contact_sheets, write_region_bank_csv
from wsi_cf.data.slides import (
    find_slide_path,
    infer_objective_power,
    level0_size_for_target_magnification,
    level0_tile_size,
    open_slide,
    quick_region_quality_metrics,
    quick_tissue_score,
    read_region_rgb_at_magnification,
)
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, read_h5_features_coords, resolve_test_rows, run_mil_attention
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2
from wsi_cf.steering.cell_selection import encode_cells


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
            "Export attention-guided 10x 1024x1024 HNSCC regions that are suitable for local steering. "
            "A suitable region must contain at least one high-attention cell and at most 2/3 of the 4x4 grid "
            "marked high attention."
        )
    )
    parser.add_argument("--split-json", type=Path, default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.json"))
    parser.add_argument("--split-tsv", type=Path, default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv"))
    parser.add_argument("--features-root", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"))
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/HNSCC_slides"))
    parser.add_argument("--mil-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt"))
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/attention_guided_region_bank_10x")
    parser.add_argument("--target-magnification", type=float, default=10.0)
    parser.add_argument("--region-size", type=int, default=1024)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--tile-size-20x", type=int, default=256)
    parser.add_argument("--attention-percentile", type=float, default=90.0)
    parser.add_argument("--min-high-attention-cells", type=int, default=1)
    parser.add_argument("--max-high-attention-cells", type=int, default=10)
    parser.add_argument("--candidate-anchors-per-slide", type=int, default=32)
    parser.add_argument("--min-region-tissue-score", type=float, default=0.70)
    parser.add_argument("--min-region-dark-fraction", type=float, default=0.25)
    parser.add_argument("--min-region-saturation-fraction", type=float, default=0.60)
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--source-label", type=int, default=None, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    return parser


def save_rows_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def load_source_rows(args) -> list[dict[str, object]]:
    rows = resolve_test_rows(split_json=args.split_json, split_tsv=args.split_tsv, features_root=args.features_root)
    out: list[dict[str, object]] = []
    for row in rows:
        slide_path = find_slide_path(args.slides_dir, str(row["slide_key"]))
        if slide_path is None:
            continue
        out.append({**row, "slide_path": str(slide_path)})
    out.sort(key=lambda row: (int(row["label"]), str(row["slide_key"])))
    if args.source_label is not None:
        out = [row for row in out if int(row["label"]) == int(args.source_label)]
    if int(args.max_slides) > 0:
        out = out[: int(args.max_slides)]
    return out


def choose_quality_filtered_region(
    *,
    slide,
    slide_key: str,
    case_id: str,
    label: int,
    coords: np.ndarray,
    attention: np.ndarray,
    crop_w_level0: int,
    crop_h_level0: int,
    tile_size_level0: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], object, int, int, dict[str, float]]:
    candidates = iter_attention_region_candidates(
        slide_key=slide_key,
        case_id=case_id,
        label=label,
        coords=coords,
        attention=attention,
        slide_w=int(slide.dimensions[0]),
        slide_h=int(slide.dimensions[1]),
        crop_w_level0=int(crop_w_level0),
        crop_h_level0=int(crop_h_level0),
        tile_size_level0=int(tile_size_level0),
        grid_side=4,
        attention_percentile=float(args.attention_percentile),
        candidate_anchor_limit=int(args.candidate_anchors_per_slide),
    )
    fallback_choice = None
    for candidate in candidates:
        if not (int(args.min_high_attention_cells) <= int(candidate["selected_cell_count"]) <= int(args.max_high_attention_cells)):
            continue
        region_img, used_crop_w, used_crop_h = read_region_rgb_at_magnification(
            slide,
            x0=int(candidate["region_x"]),
            y0=int(candidate["region_y"]),
            out_w=int(args.region_size),
            out_h=int(args.region_size),
            target_magnification=float(args.target_magnification),
        )
        quality = quick_region_quality_metrics(region_img)
        candidate = {**candidate, "region_quality": quality}
        if fallback_choice is None:
            fallback_choice = (candidate, region_img, used_crop_w, used_crop_h, quality)
        if (
            float(quality["tissue_score"]) >= float(args.min_region_tissue_score)
            and float(quality["dark_fraction"]) >= float(args.min_region_dark_fraction)
            and float(quality["saturation_fraction"]) >= float(args.min_region_saturation_fraction)
        ):
            candidate["selection_fallback"] = False
            return candidate, region_img, used_crop_w, used_crop_h, quality
    if fallback_choice is not None:
        candidate, region_img, used_crop_w, used_crop_h, quality = fallback_choice
        candidate["selection_fallback"] = True
        candidate["selection_fallback_reason"] = "no_region_passed_quality_thresholds"
        return candidate, region_img, used_crop_w, used_crop_h, quality
    proposal = select_attention_region(
        slide_key=slide_key,
        case_id=case_id,
        label=label,
        coords=coords,
        attention=attention,
        slide_w=int(slide.dimensions[0]),
        slide_h=int(slide.dimensions[1]),
        crop_w_level0=int(crop_w_level0),
        crop_h_level0=int(crop_h_level0),
        tile_size_level0=int(tile_size_level0),
        grid_side=4,
        attention_percentile=float(args.attention_percentile),
        min_high_attention_cells=int(args.min_high_attention_cells),
        max_high_attention_cells=int(args.max_high_attention_cells),
        candidate_anchor_limit=int(args.candidate_anchors_per_slide),
    )
    region_img, used_crop_w, used_crop_h = read_region_rgb_at_magnification(
        slide,
        x0=int(proposal["region_x"]),
        y0=int(proposal["region_y"]),
        out_w=int(args.region_size),
        out_h=int(args.region_size),
        target_magnification=float(args.target_magnification),
    )
    quality = quick_region_quality_metrics(region_img)
    proposal = {**proposal, "region_quality": quality, "selection_fallback": True, "selection_fallback_reason": "no_attention_candidate"}
    return proposal, region_img, used_crop_w, used_crop_h, quality


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "cli_args": _serialize_args(args),
        "argv": list(argv) if argv is not None else list(sys.argv[1:]),
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else list(sys.argv[1:])))),
        "cwd": str(Path.cwd()),
    }
    write_json(args.out_dir / "export_args.json", args_payload)

    source_rows = load_source_rows(args)
    if not source_rows:
        raise ValueError("No source slides resolved for the requested split and slide directory.")

    mil_model = build_mil_from_checkpoint(args.mil_ckpt, device=device)
    uni_model, uni_transform = load_uni2(device=device)

    def build_feature_grid(region_img):
        z_grid = build_uni_grid_from_image(
            region_img,
            uni_model=uni_model,
            uni_transform=uni_transform,
            grid_step_px=int(args.grid_step_px),
            device=device,
            out_dtype=dtype,
        )
        return z_grid.detach().float().cpu().numpy().astype("float32", copy=False)

    exported_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for source_row in source_rows:
        slide_key = str(source_row["slide_key"])
        x, coords = read_h5_features_coords(str(source_row["h5_path"]))
        if coords is None:
            skipped_rows.append({"slide_key": slide_key, "reason": "missing_coords"})
            continue
        full_attention, full_pred, full_prob_pos = run_mil_attention(mil_model, x, device=device)

        slide = open_slide(Path(str(source_row["slide_path"])))
        try:
            objective = infer_objective_power(slide)
            crop_w_level0 = level0_size_for_target_magnification(
                int(args.region_size),
                float(args.target_magnification),
                objective,
            )
            crop_h_level0 = level0_size_for_target_magnification(
                int(args.region_size),
                float(args.target_magnification),
                objective,
            )
            tile_size_level0 = level0_tile_size(int(args.tile_size_20x), objective)
            proposal, region_img, crop_w_level0, crop_h_level0, region_quality = choose_quality_filtered_region(
                slide=slide,
                slide_key=slide_key,
                case_id=str(source_row["case_id"]),
                label=int(source_row["label"]),
                coords=coords,
                attention=full_attention,
                crop_w_level0=int(crop_w_level0),
                crop_h_level0=int(crop_h_level0),
                tile_size_level0=int(tile_size_level0),
                args=args,
            )
            if bool(proposal.get("selection_fallback", False)):
                skipped_rows.append({"slide_key": slide_key, "reason": "no_suitable_region"})
                continue
        finally:
            slide.close()

        row = {
            "split": str(source_row.get("split", "test")),
            "label": int(source_row["label"]),
            "hpv_status": "HPV+" if int(source_row["label"]) == 1 else "HPV-",
            "case_id": str(source_row["case_id"]),
            "slide_key": slide_key,
            "slide_path": str(source_row["slide_path"]),
            "canonical_h5_path": str(source_row["h5_path"]),
        }
        region_id = (
            f"{slide_key}__attn_local__mag_{str(args.target_magnification).replace('.', 'p')}"
            f"__x_{int(proposal['region_x'])}__y_{int(proposal['region_y'])}"
        )
        exported = export_region_bundle(
            region_img=region_img,
            out_dir=args.out_dir,
            region_id=region_id,
            row=row,
            region_x=int(proposal["region_x"]),
            region_y=int(proposal["region_y"]),
            region_size=int(args.region_size),
            grid_step_px=int(args.grid_step_px),
            tissue_score=float(quick_tissue_score(region_img)),
            seed=int(args.seed),
            build_feature_grid=build_feature_grid,
        )
        exported.update(
            {
                "target_magnification": float(args.target_magnification),
                "crop_w_level0": int(crop_w_level0),
                "crop_h_level0": int(crop_h_level0),
                "full_pred": int(full_pred),
                "full_prob_pos": float(full_prob_pos),
                "anchor_tile_index": int(proposal["anchor_tile_index"]),
                "anchor_attention": float(proposal["anchor_attention"]),
                "selected_cells": proposal["selected_cells"],
                "selected_cells_encoded": encode_cells([(int(gx), int(gy)) for gx, gy in proposal["selected_cells"]]),
                "selected_cell_count": int(proposal["selected_cell_count"]),
                "selected_cell_fraction": float(proposal["selected_cell_fraction"]),
                "high_attention_threshold": float(proposal["high_attention_threshold"]),
                "local_tile_count": int(proposal["local_tile_count"]),
                "selection_fallback": bool(proposal["selection_fallback"]),
                "region_quality": region_quality,
                "export_args_path": str(args.out_dir / "export_args.json"),
                "cli_args": args_payload["cli_args"],
                "command": args_payload["command"],
            }
        )
        write_json(Path(str(exported["region_dir"])) / "region_meta.json", exported)
        write_json(Path(str(exported["region_dir"])) / "region_proposal.json", proposal)
        full_rows = []
        tile_row_map = {int(tile["tile_index"]): tile for tile in proposal["tile_rows"]}
        full_ranks = np.argsort(-np.asarray(full_attention, dtype=np.float32))
        full_rank_map = {int(tile_idx): rank + 1 for rank, tile_idx in enumerate(full_ranks.tolist())}
        selected_set = {(int(gx), int(gy)) for gx, gy in proposal["selected_cells"]}
        for tile_idx, (coord_x, coord_y) in enumerate(np.asarray(coords, dtype=np.int64).tolist()):
            local = tile_row_map.get(int(tile_idx))
            cell_gx = local["cell_gx"] if local else ""
            cell_gy = local["cell_gy"] if local else ""
            full_rows.append(
                {
                    "tile_index": int(tile_idx),
                    "coord_x": int(coord_x),
                    "coord_y": int(coord_y),
                    "attention": float(full_attention[tile_idx]),
                    "attention_rank": int(full_rank_map[int(tile_idx)]),
                    "in_region": bool(local is not None),
                    "cell_gx": cell_gx,
                    "cell_gy": cell_gy,
                    "is_high_attention": bool(local["is_high_attention"]) if local else False,
                    "is_selected_cell": bool((int(cell_gx), int(cell_gy)) in selected_set) if local else False,
                }
            )
        save_rows_csv(Path(str(exported["region_dir"])) / "full_attention.csv", full_rows)
        exported_rows.append(exported)

    exported_rows.sort(key=lambda row: (int(row["label"]), str(row["slide_key"]), str(row["region_id"])))
    write_region_bank_csv(args.out_dir / "region_bank.csv", exported_rows)
    save_label_contact_sheets(rows=exported_rows, out_dir=args.out_dir)
    summary = {
        "n_regions": len(exported_rows),
        "n_skipped": len(skipped_rows),
        "target_magnification": float(args.target_magnification),
        "region_size": int(args.region_size),
        "grid_step_px": int(args.grid_step_px),
        "attention_percentile": float(args.attention_percentile),
        "min_high_attention_cells": int(args.min_high_attention_cells),
        "max_high_attention_cells": int(args.max_high_attention_cells),
        "region_bank_csv": str(args.out_dir / "region_bank.csv"),
        "export_args_path": str(args.out_dir / "export_args.json"),
        "cli_args": args_payload["cli_args"],
        "command": args_payload["command"],
        "skipped_rows": skipped_rows,
    }
    write_json(args.out_dir / "region_bank_summary.json", summary)
    print(f"[ok] wrote {args.out_dir / 'region_bank.csv'}")
    print(f"[ok] wrote {args.out_dir / 'region_bank_summary.json'}")


if __name__ == "__main__":
    main()
