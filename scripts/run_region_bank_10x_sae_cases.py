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
from wsi_cf.common.paths import ensure_legacy_repo_root_on_path
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
            "This consumes saved 1024x1024 region images and aligned 4x4 feature grids, "
            "then applies selected-cell SAE prototype steering for named test cases."
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
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.0)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=1.0)
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
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def parse_cases(arg: str) -> list[str]:
    if str(arg).strip().lower() == "all":
        return list(DEFAULT_CASES)
    cases = [item.strip() for item in str(arg).split(",") if item.strip()]
    bad = [item for item in cases if item not in DEFAULT_CASES]
    if bad:
        raise ValueError(f"Unsupported cases: {bad}")
    return cases


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
    grid_side = 4

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
    neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
    chosen_latent = int(pos_latent if str(args.direction) == "hpv_pos" else neg_latent)

    manifest_rows = build_manifest(
        source_rows,
        cases=parse_cases(args.cases),
        grid_w=int(grid_side),
        grid_h=int(grid_side),
        manual_cells=manual_cells,
        seed=int(args.seed),
        direction=str(args.direction),
        prototype_latent=int(chosen_latent),
        neighbor_anchor=(int(args.neighbor_anchor_gx), int(args.neighbor_anchor_gy)),
        block_2x2_origin=(int(args.block_2x2_origin_gx), int(args.block_2x2_origin_gy)),
        block_2x3_origin=(int(args.block_2x3_origin_gx), int(args.block_2x3_origin_gy)),
    )
    write_region_bank_csv(args.out_dir / "experiment_manifest.csv", manifest_rows)

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


if __name__ == "__main__":
    main()
