#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from wsi_cf.common.paths import ensure_legacy_repo_root_on_path
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.attention_proposals import iter_attention_region_candidates, select_attention_region
from wsi_cf.data.slides import (
    find_slide_path,
    infer_objective_power,
    level0_size_for_target_magnification,
    level0_tile_size,
    open_slide,
    quick_region_quality_metrics,
    read_region_rgb_at_magnification,
)
from wsi_cf.eval.hnsc_hpv import (
    build_mil_from_checkpoint,
    load_prototypes,
    pick_prototype_latent,
    read_h5_features_coords,
    resolve_test_rows,
    run_mil_attention,
)
from wsi_cf.eval.local_region import build_local_attention_rows, replace_selected_cells_in_local_bag
from wsi_cf.generation.pixcell import (
    build_uni_grid_from_image,
    load_uni2,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
    vae_encode_auto,
)
from wsi_cf.steering.cell_selection import encode_cells

ensure_legacy_repo_root_on_path()

from utils.sae import load_sae_from_config  # type: ignore
from utils.sae_edit import edit_uni_z_grid_with_sae  # type: ignore


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
            "Attention-guided local 1024 counterfactual steering on HNSCC: "
            "use slide-level MIL attention to pick one 1024 region, steer only high-attention cells, "
            "re-encode the generated region with UNI, and score only the local region bag before and after."
        )
    )
    parser.add_argument("--split-json", type=Path, default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.json"))
    parser.add_argument("--split-tsv", type=Path, default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv"))
    parser.add_argument("--features-root", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"))
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/HNSCC_slides"))
    parser.add_argument("--mil-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt"))
    parser.add_argument("--sae-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt"))
    parser.add_argument("--sae-cfg", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json"))
    parser.add_argument("--prototype-npz", type=Path, default=Path("/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"))
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--direction", type=str, default="hpv_pos", choices=["hpv_pos", "hpv_neg"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
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
    parser.add_argument("--allow-fallback-regions", action="store_true")
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--source-label", type=int, default=None, choices=[0, 1])
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
    parser.add_argument("--preserve-outside-latents", dest="preserve_outside_latents", action="store_true", default=True)
    parser.add_argument("--no-preserve-outside-latents", dest="preserve_outside_latents", action="store_false")
    parser.add_argument("--preserve-outside-strength", type=float, default=1.0)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.0)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/attention_guided_local_1024")
    return parser


def load_image(path: Path) -> Image.Image:
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


def slide_summary_from_json(path: Path) -> dict[str, object]:
    import json
    return json.loads(path.read_text())


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
) -> tuple[dict[str, object], Image.Image, int, int, dict[str, float]]:
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
    fallback_choice: tuple[dict[str, object], Image.Image, int, int, dict[str, float]] | None = None
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

    source_rows = load_source_rows(args)
    if not source_rows:
        raise ValueError("No source slides resolved for the requested split and slide directory.")

    mil_model = build_mil_from_checkpoint(args.mil_ckpt, device=device)
    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    chosen_latent = int(pos_latent if str(args.direction) == "hpv_pos" else neg_latent)
    prototype_vec = proto_by_latent[chosen_latent]

    uni_model, uni_transform = load_uni2(device=device)
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

    cohort_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    contact_items: list[tuple[str, Image.Image]] = []
    for source_row in source_rows:
        slide_key = str(source_row["slide_key"])
        slide_dir = args.out_dir / "by_slide" / slide_key
        slide_dir.mkdir(parents=True, exist_ok=True)
        slide_summary_path = slide_dir / "slide_summary.json"
        if bool(args.skip_existing) and slide_summary_path.exists():
            cohort_rows.append(slide_summary_from_json(slide_summary_path))
            continue

        x, coords = read_h5_features_coords(str(source_row["h5_path"]))
        if coords is None:
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
            proposal, source_img, crop_w_level0, crop_h_level0, region_quality = choose_quality_filtered_region(
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
        finally:
            slide.close()
        if bool(proposal.get("selection_fallback", False)) and not bool(args.allow_fallback_regions):
            skipped_rows.append(
                {
                    "slide_key": slide_key,
                    "reason": str(proposal.get("selection_fallback_reason", "fallback_region")),
                    "anchor_tile_index": int(proposal["anchor_tile_index"]),
                    "selected_cell_count": int(proposal["selected_cell_count"]),
                    "region_quality": region_quality,
                }
            )
            print(f"[skip] {slide_key}: no suitable high-attention, cell-rich region found")
            continue

        source_zgrid = build_uni_grid_from_image(
            source_img,
            uni_model=uni_model,
            uni_transform=uni_transform,
            grid_step_px=int(args.grid_step_px),
            device=device,
            out_dtype=dtype,
        )
        selected_cells = [(int(gx), int(gy)) for gx, gy in proposal["selected_cells"]]
        tile_mask = np.zeros(tuple(source_zgrid.shape[:2]), dtype=np.float32)
        for gx, gy in selected_cells:
            tile_mask[int(gy), int(gx)] = 1.0
        steered_zgrid_t, _ = edit_uni_z_grid_with_sae(
            sae_model=sae_model,
            z_grid=source_zgrid.to(device=device, dtype=torch.float32),
            target_latent_vector=prototype_vec,
            target_latent_vector_strength=float(args.prototype_strength),
            tile_mask=tile_mask,
            blend=float(args.steer_blend),
            keep_non_selected=True,
            return_debug=False,
        )
        z_grid_base_pix = source_zgrid.to(device=device, dtype=dtype)
        z_grid_edit_pix = steered_zgrid_t.to(device=device, dtype=dtype)

        preserve_source_latents = None
        edit_region_mask = None
        if bool(args.preserve_outside_latents) and selected_cells:
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
                cells=selected_cells,
                grid_step_px=int(args.grid_step_px),
            ).to(device=device)

        generator_base = torch.Generator(device=device)
        generator_base.manual_seed(int(args.seed))
        use_autocast = device.type == "cuda" and dtype == torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
        with torch.inference_mode(), ctx:
            base_img_t = sample_large_pixcell_multidiffusion(
                pipeline=pipeline,
                z_grid=z_grid_base_pix,
                scheduled_z_grid=None,
                condition_start_ratio=float(args.mid_steer_start_ratio),
                condition_end_ratio=float(args.mid_steer_end_ratio),
                condition_alpha_start=float(args.mid_steer_alpha_start),
                condition_alpha_end=float(args.mid_steer_alpha_end),
                condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
                out_h=int(args.region_size),
                out_w=int(args.region_size),
                patch_px=patch_px,
                stride_px=stride_px,
                cond_grid_side=cond_grid_side,
                guidance_scale=float(args.guidance),
                num_steps=int(args.steps),
                patch_batch=int(args.patch_batch),
                strength=0.0,
                init_latents=None,
                preserve_source_latents=None,
                edit_region_mask=None,
                preserve_outside_strength=float(args.preserve_outside_strength),
                use_tiled_vae_decode=False,
                decode_tile_lat=128,
                decode_overlap_lat=16,
                generator=generator_base,
            )

        generator_edit = torch.Generator(device=device)
        generator_edit.manual_seed(int(args.seed))
        with torch.inference_mode(), ctx:
            steered_img_t = sample_large_pixcell_multidiffusion(
                pipeline=pipeline,
                z_grid=z_grid_base_pix,
                scheduled_z_grid=z_grid_edit_pix,
                condition_start_ratio=float(args.mid_steer_start_ratio),
                condition_end_ratio=float(args.mid_steer_end_ratio),
                condition_alpha_start=float(args.mid_steer_alpha_start),
                condition_alpha_end=float(args.mid_steer_alpha_end),
                condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
                out_h=int(args.region_size),
                out_w=int(args.region_size),
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
                use_tiled_vae_decode=False,
                decode_tile_lat=128,
                decode_overlap_lat=16,
                generator=generator_edit,
            )

        base_img = Image.fromarray((base_img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))
        steered_img = Image.fromarray((steered_img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))
        steered_reencoded = build_uni_grid_from_image(
            steered_img,
            uni_model=uni_model,
            uni_transform=uni_transform,
            grid_step_px=int(args.grid_step_px),
            device=device,
            out_dtype=torch.float32,
        ).detach().cpu().numpy().astype(np.float32)

        tile_rows = list(proposal["tile_rows"])
        local_tile_indices = [int(row["tile_index"]) for row in tile_rows]
        x_local = np.asarray(x[local_tile_indices], dtype=np.float32)
        local_attention_before, local_pred_before, local_prob_before = run_mil_attention(mil_model, x_local, device=device)
        x_local_cf, replacement_manifest = replace_selected_cells_in_local_bag(
            features_local=x_local,
            tile_rows=tile_rows,
            replacement_grid=steered_reencoded,
            selected_cells=selected_cells,
        )
        local_attention_after, local_pred_after, local_prob_after = run_mil_attention(mil_model, x_local_cf, device=device)
        local_attention_rows = build_local_attention_rows(
            tile_rows=tile_rows,
            attention_before=local_attention_before,
            attention_after=local_attention_after,
            selected_cells=selected_cells,
        )

        source_actual_path = slide_dir / "source_region_actual.png"
        source_generated_path = slide_dir / "source_region_generated.png"
        steered_path = slide_dir / "steered_region.png"
        selected_overlay_path = slide_dir / "selected_cells_overlay.png"
        full_attention_csv = slide_dir / "full_attention.csv"
        proposal_path = slide_dir / "region_proposal.json"
        local_replacement_path = slide_dir / "local_replacement_manifest.json"
        local_scores_path = slide_dir / "local_mil_before_after.json"
        source_zgrid_path = slide_dir / "source_region_zgrid.npy"
        steered_zgrid_path = slide_dir / "steered_region_zgrid.npy"

        save_png(source_img, source_actual_path)
        save_png(base_img, source_generated_path)
        save_png(steered_img, steered_path)
        save_png(draw_selected_cells_overlay(source_img, cells=selected_cells, grid_step_px=int(args.grid_step_px)), selected_overlay_path)
        np.save(source_zgrid_path, source_zgrid.detach().cpu().numpy().astype(np.float32))
        np.save(steered_zgrid_path, steered_reencoded)

        full_rows = []
        tile_row_map = {int(row["tile_index"]): row for row in tile_rows}
        full_ranks = np.argsort(-np.asarray(full_attention, dtype=np.float32))
        full_rank_map = {int(tile_idx): rank + 1 for rank, tile_idx in enumerate(full_ranks.tolist())}
        selected_set = {(int(gx), int(gy)) for gx, gy in selected_cells}
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
        save_rows_csv(full_attention_csv, full_rows)

        proposal_payload = {
            **proposal,
            "region_quality": region_quality,
            "crop_w_level0": int(crop_w_level0),
            "crop_h_level0": int(crop_h_level0),
            "target_magnification": float(args.target_magnification),
            "tile_size_level0": int(tile_size_level0),
            "selected_cells_encoded": encode_cells(selected_cells),
            "region_actual_path": str(source_actual_path),
            "region_generated_path": str(source_generated_path),
            "steered_region_path": str(steered_path),
        }
        write_json(proposal_path, proposal_payload)
        write_json(
            local_replacement_path,
            {
                "slide_key": slide_key,
                "selected_cells": selected_cells,
                "selected_cells_encoded": encode_cells(selected_cells),
                "local_tile_count": int(len(tile_rows)),
                "replacement_rows": replacement_manifest,
                "source_feature_grid_path": str(source_zgrid_path),
                "steered_feature_grid_path": str(steered_zgrid_path),
            },
        )
        write_json(
            local_scores_path,
            {
                "slide_key": slide_key,
                "local_pred_before": int(local_pred_before),
                "local_prob_pos_before": float(local_prob_before),
                "local_pred_after": int(local_pred_after),
                "local_prob_pos_after": float(local_prob_after),
                "delta_prob_pos": float(local_prob_after - local_prob_before),
                "attention_rows": local_attention_rows,
            },
        )

        slide_summary = {
            "slide_key": slide_key,
            "case_id": str(source_row["case_id"]),
            "label": int(source_row["label"]),
            "direction": str(args.direction),
            "full_pred": int(full_pred),
            "full_prob_pos": float(full_prob_pos),
            "local_pred_before": int(local_pred_before),
            "local_prob_pos_before": float(local_prob_before),
            "local_pred_after": int(local_pred_after),
            "local_prob_pos_after": float(local_prob_after),
            "delta_prob_pos": float(local_prob_after - local_prob_before),
            "anchor_tile_index": int(proposal["anchor_tile_index"]),
            "anchor_attention": float(proposal["anchor_attention"]),
            "selected_cell_count": int(proposal["selected_cell_count"]),
            "selected_cell_fraction": float(proposal["selected_cell_fraction"]),
            "high_attention_threshold": float(proposal["high_attention_threshold"]),
            "selection_fallback": bool(proposal["selection_fallback"]),
            "selected_cells": selected_cells,
            "region_quality": region_quality,
            "local_tile_count": int(proposal["local_tile_count"]),
            "source_region_actual_path": str(source_actual_path),
            "source_region_generated_path": str(source_generated_path),
            "steered_region_path": str(steered_path),
            "region_proposal_path": str(proposal_path),
            "local_replacement_manifest_path": str(local_replacement_path),
            "local_scores_path": str(local_scores_path),
            "full_attention_csv": str(full_attention_csv),
            "experiment_args_path": str(args.out_dir / "experiment_args.json"),
            "cli_args": args_payload["cli_args"],
            "command": args_payload["command"],
        }
        write_json(slide_summary_path, slide_summary)
        cohort_rows.append(slide_summary)
        contact_items.extend(
            [
                (f"{slide_key} actual", source_img),
                (f"{slide_key} baseline", base_img),
                (f"{slide_key} steered", steered_img),
            ]
        )
        print(f"[ok] wrote {slide_summary_path}")

    if cohort_rows:
        save_rows_csv(args.out_dir / "cohort_results.csv", cohort_rows)
        deltas = [float(row["delta_prob_pos"]) for row in cohort_rows]
        summary = {
            "n_slides": len(cohort_rows),
            "direction": str(args.direction),
            "prototype_key": str(args.prototype_key),
            "prototype_strength": float(args.prototype_strength),
            "steer_blend": float(args.steer_blend),
            "prototype_latent": int(chosen_latent),
            "mean_delta_prob_pos": float(np.mean(deltas)),
            "median_delta_prob_pos": float(np.median(deltas)),
            "n_positive_delta": int(sum(1 for delta in deltas if delta > 0)),
            "n_negative_delta": int(sum(1 for delta in deltas if delta < 0)),
            "n_fallback": int(sum(1 for row in cohort_rows if bool(row["selection_fallback"]))),
            "n_skipped": int(len(skipped_rows)),
            "skipped_rows": skipped_rows,
            "experiment_args_path": str(args.out_dir / "experiment_args.json"),
            "cli_args": args_payload["cli_args"],
            "command": args_payload["command"],
        }
        write_json(args.out_dir / "cohort_summary.json", summary)
        sheet = build_contact_sheet(contact_items, thumb_size=256, ncols=3, pad=12)
        save_png(sheet, args.out_dir / "source_baseline_steered_contact_sheet.png")
        print(f"[ok] wrote {args.out_dir / 'cohort_results.csv'}")
        print(f"[ok] wrote {args.out_dir / 'cohort_summary.json'}")


if __name__ == "__main__":
    main()
