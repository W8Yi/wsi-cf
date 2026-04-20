#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png, write_json
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.concept_bank import load_representative_tiles
from wsi_cf.data.region_bank import export_region_bundle, save_label_contact_sheets, write_region_bank_csv
from wsi_cf.data.slides import (
    crop_tile_rgb,
    infer_objective_power,
    level0_size_for_target_magnification,
    level0_tile_size,
    open_slide,
    quick_tissue_score,
    read_region_rgb_at_magnification,
)
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2


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
            "Export 10x concept-centered regions using current 20x representative tiles as anchors. "
            "Each output bundle contains the real 10x region, aligned 4x4 UNI feature grid, cell preview, "
            "and metadata linking back to the original SAE representative tile."
        )
    )
    parser.add_argument(
        "--prototype-json",
        type=Path,
        default=Path(
            "/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.json"
        ),
    )
    parser.add_argument(
        "--tiles-csv",
        type=Path,
        default=Path(
            "/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/sae_neuron_pipeline_batch_topk/split_0/top_neuron_tiles.csv"
        ),
    )
    parser.add_argument(
        "--split-tsv",
        type=Path,
        default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv"),
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"),
    )
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/HNSCC_slides"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WSI_CF_ROOT / "artifacts/hnscc_10x_concept_regions_from_20x_representatives",
    )
    parser.add_argument("--selected-direction", type=str, default="all", choices=["all", "hpv_pos", "hpv_neg"])
    parser.add_argument("--latent-id", action="append", type=int, default=[])
    parser.add_argument("--examples-per-latent", type=int, default=1)
    parser.add_argument("--max-latents", type=int, default=0)
    parser.add_argument("--target-magnification", type=float, default=10.0)
    parser.add_argument("--region-size", type=int, default=1024)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--tile-size-20x", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    return parser


def _clamp_region_origin(*, center_x: float, center_y: float, crop_w: int, crop_h: int, slide_w: int, slide_h: int) -> tuple[int, int]:
    x0 = int(round(float(center_x) - float(crop_w) / 2.0))
    y0 = int(round(float(center_y) - float(crop_h) / 2.0))
    x0 = max(0, min(x0, max(0, int(slide_w) - int(crop_w))))
    y0 = max(0, min(y0, max(0, int(slide_h) - int(crop_h))))
    return x0, y0


def _build_region_overlay(
    region_img: Image.Image,
    *,
    region_x: int,
    region_y: int,
    crop_w_level0: int,
    crop_h_level0: int,
    tile_x: int,
    tile_y: int,
    tile_w_level0: int,
    tile_h_level0: int,
) -> Image.Image:
    overlay = region_img.copy()
    draw = ImageDraw.Draw(overlay)
    sx = float(region_img.width) / float(max(1, crop_w_level0))
    sy = float(region_img.height) / float(max(1, crop_h_level0))
    left = (float(tile_x) - float(region_x)) * sx
    top = (float(tile_y) - float(region_y)) * sy
    right = (float(tile_x + tile_w_level0) - float(region_x)) * sx
    bottom = (float(tile_y + tile_h_level0) - float(region_y)) * sy
    draw.rectangle([left, top, right, bottom], outline=(255, 255, 0), width=4)
    return overlay


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    latent_ids = set(int(v) for v in args.latent_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "cli_args": _serialize_args(args),
        "argv": list(argv) if argv is not None else list(sys.argv[1:]),
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else list(sys.argv[1:])))),
        "cwd": str(Path.cwd()),
    }
    write_json(args.out_dir / "export_args.json", args_payload)

    representatives = load_representative_tiles(
        prototype_json=args.prototype_json,
        tiles_csv=args.tiles_csv,
        split_tsv=args.split_tsv,
        slides_dir=args.slides_dir,
        features_dir=args.features_dir,
        selected_direction=args.selected_direction,
        examples_per_latent=int(args.examples_per_latent),
        latent_ids=latent_ids or None,
    )
    if int(args.max_latents) > 0:
        kept = []
        seen: set[int] = set()
        for row in representatives:
            if row.latent_idx in seen or len(seen) < int(args.max_latents):
                kept.append(row)
                seen.add(row.latent_idx)
        representatives = kept
    if not representatives:
        raise SystemExit("No representative tiles matched the requested filters.")

    uni_model, uni_transform = load_uni2(device=device)

    def build_feature_grid(region_img: Image.Image) -> np.ndarray:
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
    for rep in representatives:
        slide = open_slide(Path(rep.slide_path))
        try:
            objective = infer_objective_power(slide)
            tile_w_level0 = level0_tile_size(int(args.tile_size_20x), objective)
            tile_h_level0 = tile_w_level0
            tile_center_x = float(rep.coord_x) + float(tile_w_level0) / 2.0
            tile_center_y = float(rep.coord_y) + float(tile_h_level0) / 2.0
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
            region_x, region_y = _clamp_region_origin(
                center_x=tile_center_x,
                center_y=tile_center_y,
                crop_w=int(crop_w_level0),
                crop_h=int(crop_h_level0),
                slide_w=int(slide.dimensions[0]),
                slide_h=int(slide.dimensions[1]),
            )
            region_img, crop_w_level0, crop_h_level0 = read_region_rgb_at_magnification(
                slide,
                x0=int(region_x),
                y0=int(region_y),
                out_w=int(args.region_size),
                out_h=int(args.region_size),
                target_magnification=float(args.target_magnification),
            )
            tissue_score = quick_tissue_score(region_img)
            center_tile_img, _ = crop_tile_rgb(
                slide,
                x=int(rep.coord_x),
                y=int(rep.coord_y),
                tile_size_20x=int(args.tile_size_20x),
                out_tile_size=int(args.tile_size_20x),
            )
        finally:
            slide.close()

        row = {
            "split": rep.split,
            "label": int(rep.label),
            "hpv_status": rep.hpv_status,
            "case_id": rep.case_id,
            "slide_key": rep.slide_key,
            "slide_path": rep.slide_path,
            "canonical_h5_path": rep.canonical_h5_path,
        }
        region_id = (
            f"latent_{rep.latent_idx}__{rep.selected_direction}__rank_{rep.prototype_rank:02d}"
            f"__{rep.slide_key}__mag_{str(args.target_magnification).replace('.', 'p')}"
            f"__x_{int(region_x)}__y_{int(region_y)}"
        )
        exported = export_region_bundle(
            region_img=region_img,
            out_dir=args.out_dir,
            region_id=region_id,
            row=row,
            region_x=int(region_x),
            region_y=int(region_y),
            region_size=int(args.region_size),
            grid_step_px=int(args.grid_step_px),
            tissue_score=float(tissue_score),
            seed=int(args.seed),
            build_feature_grid=build_feature_grid,
        )

        region_dir = Path(str(exported["region_dir"]))
        center_tile_path = region_dir / "center_tile_20x.png"
        region_overlay_path = region_dir / "region_with_center_tile.png"
        save_png(center_tile_img, center_tile_path)
        save_png(
            _build_region_overlay(
                region_img,
                region_x=int(region_x),
                region_y=int(region_y),
                crop_w_level0=int(crop_w_level0),
                crop_h_level0=int(crop_h_level0),
                tile_x=int(rep.coord_x),
                tile_y=int(rep.coord_y),
                tile_w_level0=int(tile_w_level0),
                tile_h_level0=int(tile_h_level0),
            ),
            region_overlay_path,
        )

        exported.update(
            {
                "target_magnification": float(args.target_magnification),
                "crop_w_level0": int(crop_w_level0),
                "crop_h_level0": int(crop_h_level0),
                "prototype_latent_idx": int(rep.latent_idx),
                "prototype_direction": str(rep.selected_direction),
                "prototype_rank": int(rep.prototype_rank),
                "representative_tile_index": int(rep.tile_index),
                "representative_coord_x": int(rep.coord_x),
                "representative_coord_y": int(rep.coord_y),
                "representative_tile_size_20x": int(args.tile_size_20x),
                "representative_tile_size_level0": int(tile_w_level0),
                "representative_tile_center_x": float(tile_center_x),
                "representative_tile_center_y": float(tile_center_y),
                "representative_old_h5_path": str(rep.old_h5_path),
                "representative_attention": float(rep.attention),
                "representative_sae_activation": float(rep.sae_activation),
                "representative_attention_weighted_activation": float(rep.attention_weighted_activation),
                "representative_pred": int(rep.pred),
                "representative_prob_pos": float(rep.prob_pos),
                "source_tiles_csv": str(args.tiles_csv),
                "prototype_json": str(args.prototype_json),
                "center_tile_image_path": str(center_tile_path),
                "region_overlay_path": str(region_overlay_path),
                "export_args_path": str(args.out_dir / "export_args.json"),
                "cli_args": args_payload["cli_args"],
                "command": args_payload["command"],
            }
        )
        write_json(region_dir / "region_meta.json", exported)
        exported_rows.append(exported)

    exported_rows.sort(
        key=lambda row: (
            str(row["prototype_direction"]),
            int(row["prototype_latent_idx"]),
            int(row["prototype_rank"]),
            str(row["slide_key"]),
        )
    )
    out_csv = args.out_dir / "region_bank.csv"
    write_region_bank_csv(out_csv, exported_rows)
    save_label_contact_sheets(rows=exported_rows, out_dir=args.out_dir)

    direction_counts: dict[str, int] = {}
    latent_counts: dict[str, int] = {}
    for row in exported_rows:
        direction = str(row["prototype_direction"])
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        latent_key = str(row["prototype_latent_idx"])
        latent_counts[latent_key] = latent_counts.get(latent_key, 0) + 1
    summary = {
        "n_regions": len(exported_rows),
        "target_magnification": float(args.target_magnification),
        "region_size": int(args.region_size),
        "grid_step_px": int(args.grid_step_px),
        "examples_per_latent": int(args.examples_per_latent),
        "selected_direction": str(args.selected_direction),
        "direction_counts": direction_counts,
        "latent_counts": latent_counts,
        "region_bank_csv": str(out_csv),
        "export_args_path": str(args.out_dir / "export_args.json"),
        "cli_args": args_payload["cli_args"],
        "command": args_payload["command"],
    }
    write_json(args.out_dir / "region_bank_summary.json", summary)
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {args.out_dir / 'region_bank_summary.json'}")


if __name__ == "__main__":
    main()
