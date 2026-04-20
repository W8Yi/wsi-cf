#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from wsi_cf.common.paths import ensure_legacy_repo_root_on_path
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import parse_region_roles_csv, write_region_bank_csv
from wsi_cf.eval.hnsc_hpv import load_prototypes, pick_prototype_latent
from wsi_cf.generation.pixcell import resolve_pixcell_window_config, sample_large_pixcell_multidiffusion
from wsi_cf.steering.cell_selection import decode_cells, encode_cells, parse_cell_specs, validate_cells

ensure_legacy_repo_root_on_path()

from utils.sae import load_sae_from_config  # type: ignore
from utils.sae_edit import edit_uni_z_grid_with_sae  # type: ignore


DEFAULT_CONDITIONS = [
    "baseline",
    "to_hpv_pos_one_cell",
    "to_hpv_neg_one_cell",
    "to_hpv_pos_selected_cells",
    "to_hpv_neg_selected_cells",
    "to_hpv_pos_full_grid",
    "to_hpv_neg_full_grid",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAE prototype steering on prepared 1024x1024 region-bank source regions. "
            "This edits the original source feature grid toward HPV+ / HPV- SAE prototypes "
            "instead of replacing cells with donor features."
        )
    )
    parser.add_argument("--region-roles-csv", type=Path, default=WSI_CF_ROOT / "artifacts/hnscc_region_bank_1024/region_roles.csv")
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/hnscc_region_bank_1024_sae_runs")
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
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--steer-gx", type=int, default=1)
    parser.add_argument("--steer-gy", type=int, default=1)
    parser.add_argument(
        "--steer-cell",
        action="append",
        default=[],
        help="Repeatable selected-cell spec as gx,gy. Used by *_selected_cells conditions.",
    )
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
        default=Path(
            "/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"
        ),
    )
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645, help="Preferred HPV+ prototype latent id")
    parser.add_argument("--neg-latent", type=int, default=7036, help="Preferred HPV- prototype latent id")
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


def build_sae_manifest(
    source_rows: list[dict[str, object]],
    *,
    conditions: list[str],
    steer_gx: int,
    steer_gy: int,
    selected_cells: list[tuple[int, int]],
    pos_latent: int,
    neg_latent: int,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for source in source_rows:
        base = {
            "source_region_id": str(source["region_id"]),
            "source_label": int(source["label"]),
            "source_slide_key": str(source["slide_key"]),
            "source_role_rank": int(source["role_rank"]),
            "source_image_path": str(source["image_path"]),
            "source_feature_grid_path": str(source["feature_grid_path"]),
        }
        for condition in conditions:
            row = dict(base)
            row["condition"] = condition
            row["steer_mode"] = "none"
            row["prototype_direction"] = ""
            row["prototype_latent"] = ""
            row["steer_cell_gx"] = ""
            row["steer_cell_gy"] = ""
            row["steer_cells"] = ""
            row["steer_cell_count"] = 0
            if condition == "baseline":
                pass
            elif condition == "to_hpv_pos_one_cell":
                row["steer_mode"] = "one_cell"
                row["prototype_direction"] = "hpv_pos"
                row["prototype_latent"] = int(pos_latent)
                row["steer_cell_gx"] = int(steer_gx)
                row["steer_cell_gy"] = int(steer_gy)
            elif condition == "to_hpv_neg_one_cell":
                row["steer_mode"] = "one_cell"
                row["prototype_direction"] = "hpv_neg"
                row["prototype_latent"] = int(neg_latent)
                row["steer_cell_gx"] = int(steer_gx)
                row["steer_cell_gy"] = int(steer_gy)
            elif condition == "to_hpv_pos_selected_cells":
                if not selected_cells:
                    raise ValueError("Selected-cells condition requested but no --steer-cell values were provided")
                row["steer_mode"] = "selected_cells"
                row["prototype_direction"] = "hpv_pos"
                row["prototype_latent"] = int(pos_latent)
                row["steer_cells"] = encode_cells(selected_cells)
                row["steer_cell_count"] = len(selected_cells)
            elif condition == "to_hpv_neg_selected_cells":
                if not selected_cells:
                    raise ValueError("Selected-cells condition requested but no --steer-cell values were provided")
                row["steer_mode"] = "selected_cells"
                row["prototype_direction"] = "hpv_neg"
                row["prototype_latent"] = int(neg_latent)
                row["steer_cells"] = encode_cells(selected_cells)
                row["steer_cell_count"] = len(selected_cells)
            elif condition == "to_hpv_pos_full_grid":
                row["steer_mode"] = "full_grid"
                row["prototype_direction"] = "hpv_pos"
                row["prototype_latent"] = int(pos_latent)
            elif condition == "to_hpv_neg_full_grid":
                row["steer_mode"] = "full_grid"
                row["prototype_direction"] = "hpv_neg"
                row["prototype_latent"] = int(neg_latent)
            else:
                raise ValueError(f"Unsupported condition: {condition}")
            out.append(row)
    return out


def make_summary_row(row: dict[str, object], *, out_path: Path) -> dict[str, object]:
    return {
        "condition": str(row["condition"]),
        "source_region_id": str(row["source_region_id"]),
        "source_label": int(row["source_label"]),
        "source_image_path": str(row["source_image_path"]),
        "steer_mode": str(row["steer_mode"]),
        "prototype_direction": str(row["prototype_direction"]),
        "prototype_latent": row["prototype_latent"],
        "steer_cell_gx": row["steer_cell_gx"],
        "steer_cell_gy": row["steer_cell_gy"],
        "steer_cells": str(row.get("steer_cells", "")),
        "steer_cell_count": int(row.get("steer_cell_count", 0)),
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
    source_center_tile = crop_grid_cell(source_img, gx=1, gy=1, grid_step_px=int(grid_step_px))
    save_png(source_center_tile, compare_dir / "source_tile_gx1_gy1.png")

    order = {name: idx for idx, name in enumerate(DEFAULT_CONDITIONS)}
    condition_to_img: list[tuple[str, Image.Image]] = [("source_actual", source_img)]
    for run in sorted(source_runs, key=lambda r: order.get(str(r["condition"]), 10**6)):
        out_path = Path(str(run["output_path"]))
        if not out_path.exists():
            continue
        gen_img = load_image(str(out_path))
        save_png(gen_img, compare_dir / f"{run['condition']}.png")
        if str(run["condition"]) == "baseline":
            save_png(gen_img, compare_dir / "source_region.png")
            save_png(gen_img, compare_dir / "source_region_generated.png")
            save_png(gen_img, compare_dir / "baseline_generated.png")
            condition_to_img.append(("source_generated", gen_img))
        else:
            condition_to_img.append((str(run["condition"]), gen_img))

    sheet = build_contact_sheet(condition_to_img, thumb_size=256, ncols=3, pad=12)
    save_png(sheet, compare_dir / "comparison_contact_sheet.png")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    set_seed(int(args.seed))

    roles = parse_region_roles_csv(args.region_roles_csv)
    source_rows = [
        {
            "region_id": row.region_id,
            "label": int(row.label),
            "slide_key": row.slide_key,
            "role_rank": int(row.role_rank),
            "image_path": row.image_path,
            "feature_grid_path": row.feature_grid_path,
        }
        for row in roles
        if row.role == "source"
    ]
    source_rows = sorted(source_rows, key=lambda row: (int(row["label"]), int(row["role_rank"]), str(row["slide_key"])))
    if args.source_label is not None:
        source_rows = [row for row in source_rows if int(row["label"]) == int(args.source_label)]
    if int(args.max_sources) > 0:
        source_rows = source_rows[: int(args.max_sources)]
    selected_cells = parse_cell_specs(list(args.steer_cell))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = pick_prototype_latent(
        proto_by_latent,
        direction_by_latent,
        preferred=int(args.pos_latent),
        direction="hpv_pos",
    )
    neg_latent = pick_prototype_latent(
        proto_by_latent,
        direction_by_latent,
        preferred=int(args.neg_latent),
        direction="hpv_neg",
    )

    manifest_rows = build_sae_manifest(
        source_rows,
        conditions=parse_conditions(args.conditions),
        steer_gx=int(args.steer_gx),
        steer_gy=int(args.steer_gy),
        selected_cells=selected_cells,
        pos_latent=int(pos_latent),
        neg_latent=int(neg_latent),
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
        condition = str(row["condition"])
        run_dir = args.out_dir / condition / source_region_id
        out_path = run_dir / "generated.png"
        if bool(args.skip_existing) and out_path.exists():
            summary_rows.append(make_summary_row(row, out_path=out_path))
            continue

        source_img = load_image(str(row["source_image_path"]))
        source_zgrid = np.asarray(np.load(str(row["source_feature_grid_path"])), dtype=np.float32)
        z_grid_base_t = torch.from_numpy(source_zgrid).to(device=device, dtype=torch.float32)
        z_grid_edit_t = z_grid_base_t.clone()

        steer_mode = str(row["steer_mode"])
        prototype_direction = str(row["prototype_direction"])
        prototype_latent = int(row["prototype_latent"]) if str(row["prototype_latent"]) != "" else None
        if steer_mode == "one_cell":
            tile_mask = np.zeros(source_zgrid.shape[:2], dtype=np.float32)
            gx = int(row["steer_cell_gx"])
            gy = int(row["steer_cell_gy"])
            if not (0 <= gx < source_zgrid.shape[1] and 0 <= gy < source_zgrid.shape[0]):
                raise ValueError(f"Selected one-cell target {(gx, gy)} out of bounds for grid {source_zgrid.shape[:2]}")
            tile_mask[gy, gx] = 1.0
            z_grid_edit_t, _ = edit_uni_z_grid_with_sae(
                sae_model=sae_model,
                z_grid=z_grid_edit_t,
                target_latent_vector=proto_by_latent[int(prototype_latent)],
                target_latent_vector_strength=float(args.prototype_strength),
                tile_mask=tile_mask,
                blend=float(args.steer_blend),
                keep_non_selected=True,
                return_debug=False,
            )
        elif steer_mode == "selected_cells":
            tile_mask = np.zeros(source_zgrid.shape[:2], dtype=np.float32)
            cells = decode_cells(str(row.get("steer_cells", "")))
            if not cells:
                raise ValueError("selected_cells steer_mode requires at least one cell")
            cells = validate_cells(cells, grid_w=source_zgrid.shape[1], grid_h=source_zgrid.shape[0])
            for gx, gy in cells:
                tile_mask[gy, gx] = 1.0
            z_grid_edit_t, _ = edit_uni_z_grid_with_sae(
                sae_model=sae_model,
                z_grid=z_grid_edit_t,
                target_latent_vector=proto_by_latent[int(prototype_latent)],
                target_latent_vector_strength=float(args.prototype_strength),
                tile_mask=tile_mask,
                blend=float(args.steer_blend),
                keep_non_selected=True,
                return_debug=False,
            )
        elif steer_mode == "full_grid":
            z_grid_edit_t, _ = edit_uni_z_grid_with_sae(
                sae_model=sae_model,
                z_grid=z_grid_edit_t,
                target_latent_vector=proto_by_latent[int(prototype_latent)],
                target_latent_vector_strength=float(args.prototype_strength),
                tile_mask=np.ones(source_zgrid.shape[:2], dtype=np.float32),
                blend=float(args.steer_blend),
                keep_non_selected=True,
                return_debug=False,
            )
        z_grid_base_pix = z_grid_base_t.to(device=device, dtype=dtype)
        scheduled_z_grid = None if steer_mode == "none" else z_grid_edit_t.to(device=device, dtype=dtype)
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
        write_json(
            run_dir / "run_meta.json",
            {
                "condition": condition,
                "source_region_id": source_region_id,
                "source_label": int(row["source_label"]),
                "source_image_path": str(row["source_image_path"]),
                "source_feature_grid_path": str(row["source_feature_grid_path"]),
                "steer_mode": steer_mode,
                "prototype_direction": prototype_direction,
                "prototype_latent": prototype_latent,
                "prototype_key": str(args.prototype_key),
                "prototype_strength": float(args.prototype_strength),
                "steer_blend": float(args.steer_blend),
                "mid_steer_start_ratio": float(args.mid_steer_start_ratio),
                "mid_steer_end_ratio": float(args.mid_steer_end_ratio),
                "mid_steer_alpha_start": float(args.mid_steer_alpha_start),
                "mid_steer_alpha_end": float(args.mid_steer_alpha_end),
                "mid_steer_alpha_schedule": str(args.mid_steer_alpha_schedule),
                "steer_cell_gx": row["steer_cell_gx"],
                "steer_cell_gy": row["steer_cell_gy"],
                "steer_cells": str(row.get("steer_cells", "")),
                "steer_cell_count": int(row.get("steer_cell_count", 0)),
                "pix_model_id": str(args.pix_model_id),
                "grid_step_px": int(args.grid_step_px),
                "seed": int(args.seed),
                "steps": int(args.steps),
                "guidance": float(args.guidance),
                "output_path": str(out_path),
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
            "experiment_manifest_csv": str(args.out_dir / "experiment_manifest.csv"),
            "prototype_key": str(args.prototype_key),
            "prototype_strength": float(args.prototype_strength),
            "pos_latent": int(pos_latent),
            "neg_latent": int(neg_latent),
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
