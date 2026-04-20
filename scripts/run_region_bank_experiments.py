#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
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
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import (
    build_experiment_manifest,
    build_region_pairs,
    parse_region_roles_csv,
    write_region_bank_csv,
)
from wsi_cf.generation.pixcell import resolve_pixcell_window_config, sample_large_pixcell_multidiffusion


DEFAULT_CONDITIONS = [
    "baseline",
    "same_label_one_cell",
    "cross_label_one_cell",
    "same_label_full_grid",
    "cross_label_full_grid",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first fixed PixCell-1024 experiment suite from a prepared region bank and region_roles.csv."
        )
    )
    parser.add_argument("--region-roles-csv", type=Path, default=WSI_CF_ROOT / "artifacts/hnscc_region_bank_1024/region_roles.csv")
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/hnscc_region_bank_1024_runs")
    parser.add_argument("--conditions", type=str, default="all", help="Comma-separated list or 'all'")
    parser.add_argument("--source-label", type=int, default=None, choices=[0, 1], help="Optional label filter for sources")
    parser.add_argument("--max-sources", type=int, default=0, help="Optional cap on number of source regions")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--pix_model_id", type=str, default="StonyBrook-CVLab/PixCell-1024")
    parser.add_argument("--pix_pipeline_id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae_model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.0)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--save-source-copy", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def parse_conditions(arg: str) -> list[str]:
    if str(arg).strip().lower() == "all":
        return list(DEFAULT_CONDITIONS)
    conditions = [item.strip() for item in str(arg).split(",") if item.strip()]
    bad = [item for item in conditions if item not in DEFAULT_CONDITIONS]
    if bad:
        raise ValueError(f"Unsupported conditions: {bad}")
    return conditions


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_run_meta(path: Path, payload: dict[str, object]) -> None:
    write_json(path, payload)


def crop_grid_cell(img: Image.Image, *, gx: int, gy: int, grid_step_px: int) -> Image.Image:
    x0 = int(gx) * int(grid_step_px)
    y0 = int(gy) * int(grid_step_px)
    patch = img.crop((x0, y0, min(img.size[0], x0 + int(grid_step_px)), min(img.size[1], y0 + int(grid_step_px))))
    if patch.size != (int(grid_step_px), int(grid_step_px)):
        bg = Image.new("RGB", (int(grid_step_px), int(grid_step_px)), (255, 255, 255))
        bg.paste(patch, (0, 0))
        patch = bg
    return patch


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


def write_source_comparison(
    *,
    source_row: dict[str, object],
    source_runs: list[dict[str, object]],
    out_dir: Path,
    grid_step_px: int,
) -> None:
    source_region_id = str(source_row["source_region_id"])
    compare_dir = out_dir / "by_source" / source_region_id
    compare_dir.mkdir(parents=True, exist_ok=True)

    source_img = load_image(str(source_row["source_image_path"]))
    save_png(source_img, compare_dir / "source_region_actual.png")

    source_center_tile = crop_grid_cell(source_img, gx=1, gy=1, grid_step_px=int(grid_step_px))
    save_png(source_center_tile, compare_dir / "source_tile_gx1_gy1.png")

    condition_to_img: list[tuple[str, Image.Image]] = [("source_actual", source_img)]

    for run in sorted(source_runs, key=lambda r: str(r["condition"])):
        condition = str(run["condition"])
        out_path = Path(str(run["output_path"]))
        if out_path.exists():
            gen_img = load_image(str(out_path))
            save_png(gen_img, compare_dir / f"{condition}.png")
            if condition == "baseline":
                save_png(gen_img, compare_dir / "source_region.png")
                save_png(gen_img, compare_dir / "source_region_generated.png")
                save_png(gen_img, compare_dir / "baseline_generated.png")
                condition_to_img.append(("source_generated", gen_img))
            else:
                condition_to_img.append((condition, gen_img))

        donor_region_id = str(run["donor_region_id"])
        if not donor_region_id:
            continue

        donor_region_img = load_image(str(run["donor_image_path"]))
        steer_mode = str(run["steer_mode"])
        if steer_mode == "one_cell":
            gx = int(run["donor_cell_gx"])
            gy = int(run["donor_cell_gy"])
            donor_tile = crop_grid_cell(donor_region_img, gx=gx, gy=gy, grid_step_px=int(grid_step_px))
            save_png(donor_tile, compare_dir / f"{condition}__donor_tile_gx{gx}_gy{gy}.png")
        elif steer_mode == "full_grid":
            save_png(donor_region_img, compare_dir / f"{condition}__donor_region.png")

    sheet = build_contact_sheet(condition_to_img, thumb_size=256, ncols=3, pad=12)
    save_png(sheet, compare_dir / "comparison_contact_sheet.png")


def make_summary_row(row: dict[str, object], *, out_path: Path, steer_mode: str, donor_region_id: str) -> dict[str, object]:
    donor_image_path = ""
    if donor_region_id:
        if donor_region_id == str(row["same_label_donor_region_id"]):
            donor_image_path = str(row["same_label_donor_image_path"])
        elif donor_region_id == str(row["cross_label_donor_region_id"]):
            donor_image_path = str(row["cross_label_donor_image_path"])
    return {
        "condition": str(row["condition"]),
        "source_region_id": str(row["source_region_id"]),
        "source_label": int(row["source_label"]),
        "source_image_path": str(row["source_image_path"]),
        "donor_region_id": donor_region_id,
        "donor_image_path": donor_image_path,
        "steer_mode": steer_mode,
        "donor_cell_gx": row["donor_cell_gx"],
        "donor_cell_gy": row["donor_cell_gy"],
        "output_path": str(out_path),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    set_seed(int(args.seed))

    roles = parse_region_roles_csv(args.region_roles_csv)
    pairs = build_region_pairs(roles)
    if args.source_label is not None:
        pairs = [row for row in pairs if int(row["source_label"]) == int(args.source_label)]
    if int(args.max_sources) > 0:
        pairs = pairs[: int(args.max_sources)]

    manifest_rows = build_experiment_manifest(pairs, conditions=parse_conditions(args.conditions), one_cell_gx=1, one_cell_gy=1)
    pair_map = {str(row["source_region_id"]): row for row in pairs}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairings_csv = args.out_dir / "pairings.csv"
    write_region_bank_csv(pairings_csv, pairs)
    manifest_csv = args.out_dir / "experiment_manifest.csv"
    write_region_bank_csv(manifest_csv, manifest_rows)

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
        condition = str(row["condition"])
        run_dir = args.out_dir / condition / source_region_id
        out_path = run_dir / "generated.png"
        steer_mode = str(row["steer_mode"])
        donor_region_id = str(row["donor_region_id"])
        if bool(args.skip_existing) and out_path.exists():
            summary_rows.append(make_summary_row(row, out_path=out_path, steer_mode=steer_mode, donor_region_id=donor_region_id))
            continue

        source_img = load_image(str(row["source_image_path"]))
        source_zgrid = np.load(str(row["source_feature_grid_path"]))
        z_grid_base = np.asarray(source_zgrid, dtype=np.float32).copy()
        z_grid_edit = np.asarray(source_zgrid, dtype=np.float32).copy()

        donor_feature_grid_path = str(row["donor_feature_grid_path"])
        alpha = float(args.steer_blend)
        if steer_mode == "one_cell":
            donor_zgrid = np.load(donor_feature_grid_path)
            gx = int(row["donor_cell_gx"])
            gy = int(row["donor_cell_gy"])
            donor_cell = np.asarray(donor_zgrid[gy, gx], dtype=np.float32)
            z_grid_edit[gy, gx] = (1.0 - alpha) * z_grid_edit[gy, gx] + alpha * donor_cell
        elif steer_mode == "full_grid":
            donor_zgrid = np.load(donor_feature_grid_path)
            donor_full = np.asarray(donor_zgrid, dtype=np.float32)
            z_grid_edit = (1.0 - alpha) * z_grid_edit + alpha * donor_full

        z_grid_base_t = torch.from_numpy(z_grid_base).to(device=device, dtype=dtype)
        scheduled_z_grid = None if steer_mode == "none" else torch.from_numpy(z_grid_edit).to(device=device, dtype=dtype)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed))
        h, w = source_img.size[1], source_img.size[0]
        use_autocast = device.type == "cuda" and dtype == torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
        with torch.inference_mode(), ctx:
            img_t = sample_large_pixcell_multidiffusion(
                pipeline=pipeline,
                z_grid=z_grid_base_t,
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
                use_tiled_vae_decode=False,
                decode_tile_lat=128,
                decode_overlap_lat=16,
                generator=generator,
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        if bool(args.save_source_copy):
            save_png(source_img, run_dir / "source.png")
        img_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        save_png(Image.fromarray(img_np), out_path)
        meta = {
            "condition": condition,
            "source_region_id": source_region_id,
            "source_label": int(row["source_label"]),
            "source_image_path": str(row["source_image_path"]),
            "source_feature_grid_path": str(row["source_feature_grid_path"]),
            "donor_region_id": donor_region_id,
            "donor_feature_grid_path": donor_feature_grid_path,
            "steer_mode": steer_mode,
            "donor_cell_gx": row["donor_cell_gx"],
            "donor_cell_gy": row["donor_cell_gy"],
            "pix_model_id": str(args.pix_model_id),
            "grid_step_px": int(args.grid_step_px),
            "seed": int(args.seed),
            "steps": int(args.steps),
            "guidance": float(args.guidance),
            "steer_blend": float(args.steer_blend),
            "mid_steer_start_ratio": float(args.mid_steer_start_ratio),
            "mid_steer_end_ratio": float(args.mid_steer_end_ratio),
            "mid_steer_alpha_start": float(args.mid_steer_alpha_start),
            "mid_steer_alpha_end": float(args.mid_steer_alpha_end),
            "mid_steer_alpha_schedule": str(args.mid_steer_alpha_schedule),
            "output_path": str(out_path),
        }
        save_run_meta(run_dir / "run_meta.json", meta)
        summary_rows.append(make_summary_row(row, out_path=out_path, steer_mode=steer_mode, donor_region_id=donor_region_id))
        print(f"[ok] wrote {out_path}")

    write_region_bank_csv(args.out_dir / "run_summary.csv", summary_rows)
    write_json(
        args.out_dir / "run_summary.json",
        {
            "n_pairs": len(pairs),
            "n_runs_planned": len(manifest_rows),
            "n_runs_completed": len(summary_rows),
            "pairings_csv": str(pairings_csv),
            "manifest_csv": str(manifest_csv),
        },
    )

    runs_by_source: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        runs_by_source.setdefault(str(row["source_region_id"]), []).append(row)
    for source_region_id, source_runs in runs_by_source.items():
        source_row = pair_map.get(source_region_id)
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
