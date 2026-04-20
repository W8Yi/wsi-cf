#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.generation.pixcell import (
    build_uni_grid_from_image,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
    vae_encode_auto,
    load_uni2,
)


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
            "Run a seam inpainting pass on a stitched 4096x4096 output using PixCell-256 on seam-centered local patches. "
            "The stitched image is used as the init image, only the seam band is allowed to change, and patch features "
            "can come from the stitched patch, the original patch, or a blend of both."
        )
    )
    parser.add_argument("--stress-run-dir", type=Path, required=True)
    parser.add_argument("--stitch-mode", type=str, default="center_weighted_blend", choices=["hard_stitch", "overlap_average", "center_weighted_blend", "trusted_center_only"])
    parser.add_argument("--repair-method", type=str, default="seam_inpaint", choices=["seam_inpaint"])
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--seam-band-width", type=int, default=96)
    parser.add_argument("--patch-step", type=int, default=256)
    parser.add_argument(
        "--seam-grid-size",
        type=int,
        default=None,
        help="Coarse seam grid size in pixels. Defaults to the original 1024 block size from the stress run.",
    )
    parser.add_argument("--repair-scope", type=str, default="edited_boundaries", choices=["edited_boundaries", "edited_adjacent", "all_seams"])
    parser.add_argument("--feature-source", type=str, default="blend", choices=["stitched", "original", "blend"])
    parser.add_argument("--feature-blend-alpha", type=float, default=0.5, help="Used when feature-source=blend. 0 means stitched only, 1 means original only.")
    parser.add_argument("--pix_model_id", type=str, default="StonyBrook-CVLab/PixCell-256")
    parser.add_argument("--pix_pipeline_id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae_model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=64)
    parser.add_argument("--strength", type=float, default=0.35, help="Img2img strength for seam repair patches.")
    parser.add_argument("--preserve-outside-latents", action="store_true", default=True)
    parser.add_argument("--preserve-outside-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def save_rows_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def load_rows_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as handle:
        return list(csv.DictReader(handle))


def make_window_starts(total: int, window: int, stride: int) -> list[int]:
    if total <= window:
        return [0]
    starts = list(range(0, total - window + 1, stride))
    if starts[-1] != total - window:
        starts.append(total - window)
    return starts


def compute_coarse_seam_lines(*, canvas_size: int, seam_grid_size: int) -> list[int]:
    grid = max(1, int(seam_grid_size))
    return [line for line in range(grid, int(canvas_size), grid)]


def crop_patch(img: Image.Image, *, x0: int, y0: int, patch_size: int) -> Image.Image:
    return img.crop((int(x0), int(y0), int(x0) + int(patch_size), int(y0) + int(patch_size))).convert("RGB")


def seam_patch_origins(*, canvas_size: int, seam_lines: list[int], patch_size: int, patch_step: int, orientation: str) -> list[tuple[int, int]]:
    origins: list[tuple[int, int]] = []
    patch_half = int(patch_size) // 2
    if orientation == "vertical":
        for x in seam_lines:
            x0 = max(0, min(int(canvas_size) - int(patch_size), int(x) - patch_half))
            for y0 in range(0, max(1, int(canvas_size) - int(patch_size) + 1), int(patch_step)):
                origins.append((x0, int(y0)))
    elif orientation == "horizontal":
        for y in seam_lines:
            y0 = max(0, min(int(canvas_size) - int(patch_size), int(y) - patch_half))
            for x0 in range(0, max(1, int(canvas_size) - int(patch_size) + 1), int(patch_step)):
                origins.append((int(x0), y0))
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")
    return origins


def infer_target_seam_lines(
    *,
    stress_summary: dict[str, object],
    canvas_size: int,
    seam_grid_size: int,
    repair_scope: str,
) -> dict[str, list[int]]:
    default_lines = compute_coarse_seam_lines(canvas_size=int(canvas_size), seam_grid_size=int(seam_grid_size))
    if str(repair_scope) == "all_seams":
        return {"vertical": default_lines, "horizontal": default_lines}

    if str(stress_summary.get("edit_layout")) == "center_area":
        bounds = stress_summary.get("center_area_bounds")
        if isinstance(bounds, list) and len(bounds) == 4:
            x0, y0, x1, y1 = [int(v) for v in bounds]
            vertical = [line for line in (x0, x1) if 0 < line < int(canvas_size)]
            horizontal = [line for line in (y0, y1) if 0 < line < int(canvas_size)]
            return {"vertical": sorted(set(vertical)), "horizontal": sorted(set(horizontal))}

    return {"vertical": default_lines, "horizontal": default_lines}


def targeted_seam_patch_origins(
    *,
    window_rows: list[dict[str, str]],
    canvas_size: int,
    window_size: int,
    stride: int,
    seam_lines: list[int],
    patch_size: int,
    patch_step: int,
    repair_scope: str,
    orientation: str,
) -> list[tuple[int, int]]:
    if str(repair_scope) == "all_seams":
        return seam_patch_origins(
            canvas_size=int(canvas_size),
            seam_lines=list(seam_lines),
            patch_size=int(patch_size),
            patch_step=int(patch_step),
            orientation=orientation,
        )
    if seam_lines:
        return seam_patch_origins(
            canvas_size=int(canvas_size),
            seam_lines=list(seam_lines),
            patch_size=int(patch_size),
            patch_step=int(patch_step),
            orientation=orientation,
        )
    info = {}
    for row in window_rows:
        key = (int(row["row_index"]), int(row["col_index"]))
        info[key] = {
            "left": int(row["left"]),
            "top": int(row["top"]),
            "edited": int(row.get("selected_cell_count", "0") or 0) > 0,
        }
    row_starts = make_window_starts(int(canvas_size), int(window_size), int(stride))
    col_starts = make_window_starts(int(canvas_size), int(window_size), int(stride))
    origins: set[tuple[int, int]] = set()
    patch_half = int(patch_size) // 2
    if orientation == "vertical":
        for row_idx, top in enumerate(row_starts):
            y_positions = list(range(int(top), min(int(canvas_size) - int(patch_size), int(top) + int(window_size) - int(patch_size)) + 1, int(patch_step)))
            if not y_positions:
                y_positions = [max(0, min(int(canvas_size) - int(patch_size), int(top)))]
            for seam_col_idx in range(1, len(col_starts)):
                left_win = info.get((row_idx, seam_col_idx - 1))
                right_win = info.get((row_idx, seam_col_idx))
                if left_win is None or right_win is None:
                    continue
                left_edited = bool(left_win["edited"])
                right_edited = bool(right_win["edited"])
                if str(repair_scope) == "edited_boundaries":
                    keep = left_edited != right_edited
                else:
                    keep = left_edited or right_edited
                if not keep:
                    continue
                seam_x = int(col_starts[seam_col_idx])
                x0 = max(0, min(int(canvas_size) - int(patch_size), seam_x - patch_half))
                for y0 in y_positions:
                    origins.add((x0, int(y0)))
    elif orientation == "horizontal":
        for seam_row_idx in range(1, len(row_starts)):
            seam_y = int(row_starts[seam_row_idx])
            y0 = max(0, min(int(canvas_size) - int(patch_size), seam_y - patch_half))
            for col_idx, left in enumerate(col_starts):
                x_positions = list(range(int(left), min(int(canvas_size) - int(patch_size), int(left) + int(window_size) - int(patch_size)) + 1, int(patch_step)))
                if not x_positions:
                    x_positions = [max(0, min(int(canvas_size) - int(patch_size), int(left)))]
                top_win = info.get((seam_row_idx - 1, col_idx))
                bottom_win = info.get((seam_row_idx, col_idx))
                if top_win is None or bottom_win is None:
                    continue
                top_edited = bool(top_win["edited"])
                bottom_edited = bool(bottom_win["edited"])
                if str(repair_scope) == "edited_boundaries":
                    keep = top_edited != bottom_edited
                else:
                    keep = top_edited or bottom_edited
                if not keep:
                    continue
                for x0 in x_positions:
                    origins.add((int(x0), y0))
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")
    return sorted(origins)


def make_seam_band_mask(*, patch_size: int, seam_band_width: int, orientation: str) -> np.ndarray:
    size = int(patch_size)
    band = max(1, int(seam_band_width))
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    if orientation == "vertical":
        dist = np.abs(xx.astype(np.float32) - center)
    elif orientation == "horizontal":
        dist = np.abs(yy.astype(np.float32) - center)
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")
    half = band / 2.0
    feather = max(8.0, band / 4.0)
    mask = np.zeros((size, size), dtype=np.float32)
    inner = dist <= half
    outer = (dist > half) & (dist <= half + feather)
    mask[inner] = 1.0
    mask[outer] = 1.0 - ((dist[outer] - half) / feather)
    return mask


def compute_seam_metrics(
    *,
    stitched: np.ndarray,
    vertical_lines: list[int],
    horizontal_lines: list[int],
    seam_band_px: int = 16,
) -> dict[str, float]:
    arr = np.asarray(stitched, dtype=np.float32)
    vertical = [int(line) for line in vertical_lines]
    horizontal = [int(line) for line in horizontal_lines]
    vertical_jumps = []
    horizontal_jumps = []
    band = max(1, int(seam_band_px))
    for x in vertical:
        if 1 <= int(x) < arr.shape[1]:
            vertical_jumps.append(float(np.mean(np.abs(arr[:, int(x) - 1, :] - arr[:, int(x), :]))))
    for y in horizontal:
        if 1 <= int(y) < arr.shape[0]:
            horizontal_jumps.append(float(np.mean(np.abs(arr[int(y) - 1, :, :] - arr[int(y), :, :]))))
    seam_bands = []
    for x in vertical:
        x0 = max(0, int(x) - band)
        x1 = min(arr.shape[1], int(x) + band)
        seam_bands.append(float(np.mean(arr[:, x0:x1, :])))
    for y in horizontal:
        y0 = max(0, int(y) - band)
        y1 = min(arr.shape[0], int(y) + band)
        seam_bands.append(float(np.mean(arr[y0:y1, :, :])))
    return {
        "mean_vertical_jump_l1": float(np.mean(vertical_jumps)) if vertical_jumps else 0.0,
        "mean_horizontal_jump_l1": float(np.mean(horizontal_jumps)) if horizontal_jumps else 0.0,
        "mean_seam_jump_l1": float(np.mean(vertical_jumps + horizontal_jumps)) if (vertical_jumps or horizontal_jumps) else 0.0,
        "mean_seam_band_intensity": float(np.mean(seam_bands)) if seam_bands else 0.0,
    }


def draw_patch_overlay(base: Image.Image, patch_rows: list[dict[str, object]]) -> Image.Image:
    canvas = base.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for row in patch_rows:
        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x1 = x0 + int(row["patch_size"]) - 1
        y1 = y0 + int(row["patch_size"]) - 1
        color = (255, 0, 0) if row["orientation"] == "vertical" else (0, 255, 255)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    return canvas


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    stress_summary = load_json(args.stress_run_dir / "summary.json")
    stress_args = load_json(args.stress_run_dir / "experiment_args.json")
    cli_args = stress_args.get("cli_args", {})
    canvas_size = int(stress_summary["canvas_size"])
    window_size = int(stress_summary["window_size"])
    window_stride = int(stress_summary["window_stride"])
    target_magnification = float(cli_args.get("target_magnification", 10.0))
    seam_grid_size = int(args.seam_grid_size or window_size)
    target_lines = infer_target_seam_lines(
        stress_summary=stress_summary,
        canvas_size=int(canvas_size),
        seam_grid_size=int(seam_grid_size),
        repair_scope=str(args.repair_scope),
    )

    out_dir = args.out_dir or (args.stress_run_dir / f"{args.repair_method}_{args.stitch_mode}_pixcell256")
    out_dir.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "cli_args": _serialize_args(args),
        "argv": list(argv) if argv is not None else list(sys.argv[1:]),
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else list(sys.argv[1:])))),
        "cwd": str(Path.cwd()),
        "stress_run_dir": str(args.stress_run_dir),
    }
    write_json(out_dir / "experiment_args.json", args_payload)

    original_img = Image.open(args.stress_run_dir / "source_canvas_actual.png").convert("RGB")
    stitched_img = Image.open(args.stress_run_dir / f"stitched_{args.stitch_mode}.png").convert("RGB")
    window_rows_csv = load_rows_csv(args.stress_run_dir / "window_manifest.csv")

    uni_model, uni_transform = load_uni2(device=device)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=AutoencoderKL.from_pretrained(args.vae_model_id, subfolder=args.vae_subfolder, torch_dtype=dtype),
        custom_pipeline=args.pix_pipeline_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_px, stride_px, cond_grid_side = resolve_pixcell_window_config(pix_model_id=args.pix_model_id, patch_px=0, stride_px=0)

    patch_rows: list[dict[str, object]] = []
    repaired_accum = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    repaired_weight = np.zeros((canvas_size, canvas_size, 1), dtype=np.float32)

    for orientation in ("vertical", "horizontal"):
        seam_lines = target_lines[str(orientation)]
        origins = targeted_seam_patch_origins(
            window_rows=window_rows_csv,
            canvas_size=int(canvas_size),
            window_size=int(window_size),
            stride=int(window_stride),
            seam_lines=seam_lines,
            patch_size=int(args.patch_size),
            patch_step=int(args.patch_step),
            repair_scope=str(args.repair_scope),
            orientation=orientation,
        )
        seam_mask = make_seam_band_mask(
            patch_size=int(args.patch_size),
            seam_band_width=int(args.seam_band_width),
            orientation=orientation,
        )
        mask3d = seam_mask[..., None]
        for patch_idx, (x0, y0) in enumerate(origins):
            stitched_patch = crop_patch(stitched_img, x0=int(x0), y0=int(y0), patch_size=int(args.patch_size))
            original_patch = crop_patch(original_img, x0=int(x0), y0=int(y0), patch_size=int(args.patch_size))
            z_stitched = build_uni_grid_from_image(
                stitched_patch,
                uni_model=uni_model,
                uni_transform=uni_transform,
                grid_step_px=int(args.patch_size),
                device=device,
                out_dtype=dtype,
            )
            z_original = build_uni_grid_from_image(
                original_patch,
                uni_model=uni_model,
                uni_transform=uni_transform,
                grid_step_px=int(args.patch_size),
                device=device,
                out_dtype=dtype,
            )
            if str(args.feature_source) == "stitched":
                z_cond = z_stitched
            elif str(args.feature_source) == "original":
                z_cond = z_original
            else:
                alpha = float(max(0.0, min(1.0, args.feature_blend_alpha)))
                z_cond = z_stitched * (1.0 - alpha) + z_original * alpha

            init_np = np.asarray(stitched_patch, dtype=np.float32) / 255.0
            init_img_t = torch.from_numpy(init_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
            init_latents = vae_encode_auto(
                pipeline.vae,
                init_img_t,
                use_tiled=False,
                tile_img=0,
                overlap_img=0,
            )

            preserve_source_latents = None
            edit_region_mask = None
            if bool(args.preserve_outside_latents):
                preserve_source_latents = init_latents
                edit_region_mask = torch.from_numpy(seam_mask).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.seed) + (100000 if orientation == "horizontal" else 0) + int(patch_idx))
            use_autocast = device.type == "cuda" and dtype == torch.float16
            ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
            with torch.inference_mode(), ctx:
                repaired_t = sample_large_pixcell_multidiffusion(
                    pipeline=pipeline,
                    z_grid=z_cond.to(device=device, dtype=dtype),
                    scheduled_z_grid=None,
                    condition_start_ratio=0.0,
                    condition_end_ratio=1.0,
                    condition_alpha_start=1.0,
                    condition_alpha_end=1.0,
                    condition_alpha_schedule="linear",
                    out_h=int(args.patch_size),
                    out_w=int(args.patch_size),
                    patch_px=patch_px,
                    stride_px=stride_px,
                    cond_grid_side=cond_grid_side,
                    guidance_scale=float(args.guidance),
                    num_steps=int(args.steps),
                    patch_batch=int(args.patch_batch),
                    strength=float(args.strength),
                    init_latents=init_latents,
                    preserve_source_latents=preserve_source_latents,
                    edit_region_mask=edit_region_mask,
                    preserve_outside_strength=float(args.preserve_outside_strength),
                    use_tiled_vae_decode=False,
                    decode_tile_lat=64,
                    decode_overlap_lat=8,
                    generator=generator,
                )
            repaired_patch = Image.fromarray((repaired_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))

            patch_dir = out_dir / "patches" / f"{orientation}_{patch_idx:04d}"
            patch_dir.mkdir(parents=True, exist_ok=True)
            save_png(stitched_patch, patch_dir / "stitched_patch.png")
            save_png(original_patch, patch_dir / "original_patch.png")
            save_png(repaired_patch, patch_dir / "repaired_patch.png")
            seam_mask_img = Image.fromarray((seam_mask * 255.0).astype(np.uint8))
            save_png(seam_mask_img.convert("RGB"), patch_dir / "seam_mask.png")

            repaired_arr = np.asarray(repaired_patch, dtype=np.float32) / 255.0
            repaired_accum[int(y0) : int(y0) + int(args.patch_size), int(x0) : int(x0) + int(args.patch_size)] += repaired_arr * mask3d
            repaired_weight[int(y0) : int(y0) + int(args.patch_size), int(x0) : int(x0) + int(args.patch_size)] += mask3d

            patch_rows.append(
                {
                    "orientation": orientation,
                    "patch_index": int(patch_idx),
                    "seam_line": int(
                        min(
                            seam_lines,
                            key=lambda line: abs(
                                (int(x0) + int(args.patch_size) // 2 if orientation == "vertical" else int(y0) + int(args.patch_size) // 2)
                                - int(line)
                            ),
                        )
                    ) if seam_lines else None,
                    "x0": int(x0),
                    "y0": int(y0),
                    "patch_size": int(args.patch_size),
                    "feature_source": str(args.feature_source),
                    "repair_method": str(args.repair_method),
                    "stitched_patch_path": str(patch_dir / "stitched_patch.png"),
                    "original_patch_path": str(patch_dir / "original_patch.png"),
                    "repaired_patch_path": str(patch_dir / "repaired_patch.png"),
                    "seam_mask_path": str(patch_dir / "seam_mask.png"),
                }
            )
            print(f"[ok] repaired {orientation} patch {patch_idx:04d}")

    stitched_arr = np.asarray(stitched_img, dtype=np.float32) / 255.0
    write_weight = np.clip(repaired_weight, 0.0, 1.0)
    repaired_canvas = stitched_arr * (1.0 - write_weight) + np.divide(repaired_accum, np.clip(repaired_weight, 1e-8, None)) * write_weight
    repaired_canvas = np.clip(repaired_canvas, 0.0, 1.0)
    repaired_img = Image.fromarray((repaired_canvas * 255.0).astype(np.uint8))
    save_png(repaired_img, out_dir / f"repaired_{args.stitch_mode}.png")
    save_png(draw_patch_overlay(stitched_img, patch_rows), out_dir / f"stitched_{args.stitch_mode}_patch_overlay.png")
    save_png(draw_patch_overlay(repaired_img, patch_rows), out_dir / f"repaired_{args.stitch_mode}_patch_overlay.png")
    save_rows_csv(out_dir / "patch_manifest.csv", patch_rows)

    before_metrics = compute_seam_metrics(
        stitched=np.asarray(stitched_img, dtype=np.float32) / 255.0,
        vertical_lines=target_lines["vertical"],
        horizontal_lines=target_lines["horizontal"],
    )
    after_metrics = compute_seam_metrics(
        stitched=repaired_canvas,
        vertical_lines=target_lines["vertical"],
        horizontal_lines=target_lines["horizontal"],
    )
    summary = {
        "stress_run_dir": str(args.stress_run_dir),
        "stitch_mode": str(args.stitch_mode),
        "repair_method": str(args.repair_method),
        "target_magnification": float(target_magnification),
        "feature_source": str(args.feature_source),
        "feature_blend_alpha": float(args.feature_blend_alpha),
        "repair_scope": str(args.repair_scope),
        "seam_grid_size": int(seam_grid_size),
        "target_vertical_seam_lines": [int(v) for v in target_lines["vertical"]],
        "target_horizontal_seam_lines": [int(v) for v in target_lines["horizontal"]],
        "patch_count": int(len(patch_rows)),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "repaired_image_path": str(out_dir / f"repaired_{args.stitch_mode}.png"),
        "patch_manifest_csv": str(out_dir / "patch_manifest.csv"),
        "experiment_args_path": str(out_dir / "experiment_args.json"),
        "cli_args": args_payload["cli_args"],
        "command": args_payload["command"],
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[ok] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
