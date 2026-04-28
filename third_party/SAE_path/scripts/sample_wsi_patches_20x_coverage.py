#!/usr/bin/env python3
"""
Sample spatially covering tissue patches from WSI slides at 20x-equivalent scale.

For each WSI:
1) Build a coarse tissue mask on a low-resolution level.
2) Enumerate a 20x-equivalent grid at level-0 coordinates.
3) Keep candidates above a tissue-fraction threshold.
4) Select up to K patches by farthest-point sampling for spatial coverage.
5) Export sampled coordinate CSVs (patch locators) + optional overlay.

By default this script writes only patch locator metadata, not patch images.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import openslide

try:
    from cucim import CuImage
    HAS_CUCIM = True
except Exception:
    CuImage = None
    HAS_CUCIM = False


WSI_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}


@dataclass
class SlideSampleResult:
    slide: str
    status: str
    width: int
    height: int
    objective_power: Optional[float]
    level0_tile_size: int
    level0_step_size: int
    n_candidates: int
    n_selected: int
    locator_csv: str
    overlay_png: str
    read_backend: str = "openslide"
    sec_open: float = 0.0
    sec_mask: float = 0.0
    sec_filter: float = 0.0
    sec_sample: float = 0.0
    sec_write: float = 0.0
    sec_export: float = 0.0
    sec_overlay: float = 0.0
    sec_total: float = 0.0
    error: str = ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wsi_dir", type=Path, default="/common/users/wq50/SAE_path/wsi/tcga_balanced_core_subset10")
    ap.add_argument("--out_dir", type=Path, default="/common/users/wq50/SAE_path/outputs/tcga_locators")
    ap.add_argument("--tile_size_20x", type=int, default=256)
    ap.add_argument("--step_size_20x", type=int, default=400)
    ap.add_argument("--patches_per_slide", type=int, default=256)
    ap.add_argument("--mask_max_dim", type=int, default=2048)
    ap.add_argument("--sat_thresh", type=int, default=20)
    ap.add_argument("--value_max", type=int, default=245)
    ap.add_argument("--min_tissue_frac", type=float, default=0.3)
    ap.add_argument("--median_size", type=int, default=3, help="Odd median filter size before HSV thresholding.")
    ap.add_argument("--morph_size", type=int, default=5, help="Odd Max/Min filter size for mask cleanup.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_slides", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--reader", type=str, default="cucim", choices=["auto", "openslide", "cucim"])
    ap.add_argument("--overwrite", action="store_true", help="Recompute slides even if locator output already exists.")
    ap.add_argument("--no_overlay", action="store_true")
    ap.add_argument(
        "--export_patches",
        action="store_true",
        help="Also export patch JPGs under out_dir/patches (default: locator-only).",
    )
    ap.add_argument("--jpeg_quality", type=int, default=90, help="JPEG quality if --export_patches is enabled.")
    return ap.parse_args()


def infer_level0_scale_20x(slide: openslide.OpenSlide) -> float:
    obj_raw = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if obj_raw:
        try:
            obj = float(obj_raw)
            if obj > 0:
                return max(0.25, obj / 20.0)
        except ValueError:
            pass

    mpp_raw = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp_raw:
        try:
            mpp = float(mpp_raw)
            # Approximate 20x as 0.5 um/px
            if mpp > 0:
                return max(0.25, 0.5 / mpp)
        except ValueError:
            pass

    return 1.0


def infer_objective_power(slide: openslide.OpenSlide) -> Optional[float]:
    for key in (openslide.PROPERTY_NAME_OBJECTIVE_POWER, "aperio.AppMag"):
        raw = slide.properties.get(key)
        if not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if v > 0:
            return v
    return None


def level0_size_from_20x(tile_size_20x: int, objective_power: Optional[float]) -> int:
    if objective_power is not None and objective_power > 0:
        return max(1, int(round(float(tile_size_20x) * (float(objective_power) / 20.0))))
    return max(1, int(round(float(tile_size_20x) * 1.0)))


def choose_mask_level(slide: openslide.OpenSlide, target_max_dim: int) -> int:
    target = float(target_max_dim)
    best_level = 0
    best_error = float("inf")
    for level, dims in enumerate(slide.level_dimensions):
        current = float(max(dims))
        err = abs(current - target)
        if current <= target:
            return level
        if err < best_error:
            best_error = err
            best_level = level
    return best_level


def build_tissue_mask(
    slide: openslide.OpenSlide,
    cuimg,
    mask_level: int,
    sat_thresh: int,
    value_max: int,
    median_size: int,
    morph_size: int,
) -> np.ndarray:
    level_w, level_h = slide.level_dimensions[int(mask_level)]
    if cuimg is not None:
        arr = np.asarray(cuimg.read_region(location=(0, 0), size=(int(level_w), int(level_h)), level=int(mask_level)))
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        rgb = Image.fromarray(arr)
    else:
        rgb = slide.read_region((0, 0), int(mask_level), (int(level_w), int(level_h))).convert("RGB")
    if median_size > 1 and median_size % 2 == 1:
        rgb = rgb.filter(ImageFilter.MedianFilter(size=int(median_size)))

    hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    mask = (sat >= int(sat_thresh)) & (val <= int(value_max))

    if morph_size > 1 and morph_size % 2 == 1:
        mask_img = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=int(morph_size)))
        mask_img = mask_img.filter(ImageFilter.MinFilter(size=int(morph_size)))
        mask = np.asarray(mask_img, dtype=np.uint8) > 0

    return np.asarray(mask, dtype=np.uint8)


def filter_dense_grid_by_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    mask_u8: np.ndarray,
    *,
    level_downsample: float,
    tile_size_level0: int,
    min_tissue_frac: float,
) -> np.ndarray:
    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    mask_h, mask_w = mask_u8.shape
    scale = 1.0 / max(float(level_downsample), 1e-12)
    x1 = np.floor(xs.astype(np.float64) * scale).astype(np.int64)
    x2 = np.ceil((xs.astype(np.float64) + float(tile_size_level0)) * scale).astype(np.int64)
    x1 = np.clip(x1, 0, max(0, mask_w - 1))
    x2 = np.clip(x2, x1 + 1, mask_w)

    ii = np.pad(mask_u8.astype(np.uint32, copy=False), ((1, 0), (1, 0)), mode="constant")
    ii = ii.cumsum(axis=0).cumsum(axis=1)

    kept_rows: List[np.ndarray] = []
    ys_f = ys.astype(np.float64)
    y1_all = np.floor(ys_f * scale).astype(np.int64)
    y2_all = np.ceil((ys_f + float(tile_size_level0)) * scale).astype(np.int64)
    y1_all = np.clip(y1_all, 0, max(0, mask_h - 1))
    y2_all = np.clip(y2_all, y1_all + 1, mask_h)

    areas_x = (x2 - x1).astype(np.float64)
    for y0, y1, y2 in zip(ys, y1_all, y2_all):
        areas = areas_x * float(y2 - y1)
        sums = (
            ii[y2, x2]
            - ii[y1, x2]
            - ii[y2, x1]
            + ii[y1, x1]
        ).astype(np.float64, copy=False)
        keep = (sums / np.maximum(areas, 1.0)) >= float(min_tissue_frac)
        if np.any(keep):
            row = np.column_stack(
                [
                    xs[keep],
                    np.full(int(np.count_nonzero(keep)), int(y0), dtype=np.int64),
                ]
            )
            kept_rows.append(row)

    if not kept_rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(kept_rows, axis=0)


def farthest_point_sampling(coords: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = int(coords.shape[0])
    if k >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    pick0 = int(rng.integers(0, n))
    picked = np.empty((k,), dtype=np.int64)
    picked[0] = pick0

    pts = coords.astype(np.float64, copy=False)
    d2 = np.sum((pts - pts[pick0]) ** 2, axis=1)
    d2[pick0] = -1.0

    for i in range(1, k):
        nxt = int(np.argmax(d2))
        picked[i] = nxt
        dn = np.sum((pts - pts[nxt]) ** 2, axis=1)
        d2 = np.minimum(d2, dn)
        d2[nxt] = -1.0
    return picked


def render_overlay(
    slide: openslide.OpenSlide,
    coords: np.ndarray,
    level0_tile_size: int,
    out_png: Path,
    max_dim: int = 4096,
) -> None:
    w0, h0 = slide.dimensions
    scale = max(w0, h0) / float(max_dim) if max(w0, h0) > max_dim else 1.0
    level = int(slide.get_best_level_for_downsample(scale))
    lw, lh = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (lw, lh)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    sx = lw / float(w0)
    sy = lh / float(h0)
    tw = max(1, int(round(level0_tile_size * sx)))
    th = max(1, int(round(level0_tile_size * sy)))

    for x, y in coords:
        x0 = int(round(x * sx))
        y0 = int(round(y * sy))
        draw.rectangle([x0, y0, x0 + tw, y0 + th], outline=(0, 255, 0, 180), fill=(0, 255, 0, 45))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def read_level0_patch_rgb(slide: openslide.OpenSlide, cuimg, x: int, y: int, size: int) -> Image.Image:
    if cuimg is not None:
        arr = np.asarray(cuimg.read_region(location=(int(x), int(y)), size=(int(size), int(size)), level=0))
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return slide.read_region((int(x), int(y)), 0, (int(size), int(size))).convert("RGB")


def sample_one_slide(
    wsi_path: Path,
    out_dir: Path,
    tile_size_20x: int,
    step_size_20x: int,
    patches_per_slide: int,
    mask_max_dim: int,
    sat_thresh: int,
    value_max: int,
    min_tissue_frac: float,
    median_size: int,
    morph_size: int,
    seed: int,
    reader: str,
    overwrite: bool,
    jpeg_quality: int,
    save_overlay: bool,
    export_patches: bool,
) -> SlideSampleResult:
    slide_name = wsi_path.stem
    locator_csv = out_dir / "locators" / f"{slide_name}.csv"
    patch_dir = out_dir / "patches" / slide_name
    overlay_png = out_dir / "overlay" / f"{slide_name}__sampled_overlay.png"
    t_start = time.perf_counter()

    try:
        if not overwrite:
            ok_locator = locator_csv.exists()
            ok_overlay = (not save_overlay) or overlay_png.exists()
            ok_patches = (not export_patches) or patch_dir.exists()
            if ok_locator and ok_overlay and ok_patches:
                return SlideSampleResult(
                    slide=slide_name,
                    status="skipped",
                    width=0,
                    height=0,
                    objective_power=None,
                    level0_tile_size=0,
                    level0_step_size=0,
                    n_candidates=0,
                    n_selected=0,
                    locator_csv=str(locator_csv),
                    overlay_png=str(overlay_png if save_overlay else ""),
                    read_backend="skip",
                    sec_total=float(time.perf_counter() - t_start),
                )

        t0 = time.perf_counter()
        slide = openslide.OpenSlide(str(wsi_path))
        cuimg = None
        backend = "openslide"
        if reader == "cucim":
            if not HAS_CUCIM:
                raise RuntimeError("reader=cucim requested, but cucim is not installed in this environment.")
            cuimg = CuImage(str(wsi_path))
            backend = "cucim"
        elif reader == "auto":
            if HAS_CUCIM:
                try:
                    cuimg = CuImage(str(wsi_path))
                    backend = "cucim"
                except Exception:
                    cuimg = None
                    backend = "openslide"
        sec_open = float(time.perf_counter() - t0)

        w0, h0 = slide.dimensions
        objective = infer_objective_power(slide)
        if objective is None:
            scale20 = infer_level0_scale_20x(slide)
            tile0 = max(1, int(round(float(tile_size_20x) * float(scale20))))
            step0 = max(1, int(round(float(step_size_20x) * float(scale20))))
        else:
            tile0 = level0_size_from_20x(int(tile_size_20x), objective)
            step0 = level0_size_from_20x(int(step_size_20x), objective)

        t0 = time.perf_counter()
        mask_level = choose_mask_level(slide, int(mask_max_dim))
        level_downsample = float(slide.level_downsamples[int(mask_level)])
        mask_u8 = build_tissue_mask(
            slide=slide,
            cuimg=cuimg,
            mask_level=int(mask_level),
            sat_thresh=int(sat_thresh),
            value_max=int(value_max),
            median_size=int(median_size),
            morph_size=int(morph_size),
        )
        sec_mask = float(time.perf_counter() - t0)

        t0 = time.perf_counter()
        max_x = w0 - tile0
        max_y = h0 - tile0
        if max_x < 0 or max_y < 0:
            raise RuntimeError("Slide smaller than requested tile size at level 0.")

        xs = np.arange(0, max_x + 1, step0, dtype=np.int64)
        ys = np.arange(0, max_y + 1, step0, dtype=np.int64)
        cand_arr = filter_dense_grid_by_mask(
            xs=xs,
            ys=ys,
            mask_u8=mask_u8,
            level_downsample=float(level_downsample),
            tile_size_level0=int(tile0),
            min_tissue_frac=float(min_tissue_frac),
        )
        candidates: List[Tuple[int, int]] = [(int(x), int(y)) for x, y in cand_arr.tolist()]
        sec_filter = float(time.perf_counter() - t0)

        if not candidates:
            t0 = time.perf_counter()
            locator_csv.parent.mkdir(parents=True, exist_ok=True)
            with locator_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["slide", "wsi_path", "x", "y", "tile_size_20x", "level0_tile_size", "objective_power"])
            sec_write = float(time.perf_counter() - t0)
            return SlideSampleResult(
                slide=slide_name,
                status="ok",
                width=int(w0),
                height=int(h0),
                objective_power=objective,
                level0_tile_size=int(tile0),
                level0_step_size=int(step0),
                n_candidates=0,
                n_selected=0,
                locator_csv=str(locator_csv),
                overlay_png=str(overlay_png),
                read_backend=backend,
                sec_open=sec_open,
                sec_mask=sec_mask,
                sec_filter=sec_filter,
                sec_write=sec_write,
                sec_total=float(time.perf_counter() - t_start),
            )

        t0 = time.perf_counter()
        k = min(int(patches_per_slide), int(cand_arr.shape[0]))
        pick_idx = farthest_point_sampling(cand_arr.astype(np.float64), k=k, seed=seed)
        sel = cand_arr[pick_idx]
        sec_sample = float(time.perf_counter() - t0)

        t0 = time.perf_counter()
        locator_csv.parent.mkdir(parents=True, exist_ok=True)
        with locator_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slide", "wsi_path", "x", "y", "tile_size_20x", "level0_tile_size", "objective_power"])
            for x, y in sel.tolist():
                w.writerow(
                    [
                        slide_name,
                        str(wsi_path),
                        int(x),
                        int(y),
                        int(tile_size_20x),
                        int(tile0),
                        "" if objective is None else float(objective),
                    ]
                )
        sec_write = float(time.perf_counter() - t0)

        sec_export = 0.0
        if export_patches:
            t0 = time.perf_counter()
            patch_dir.mkdir(parents=True, exist_ok=True)
            for x, y in sel.tolist():
                img = read_level0_patch_rgb(slide=slide, cuimg=cuimg, x=int(x), y=int(y), size=int(tile0))
                if int(tile0) != int(tile_size_20x):
                    img = img.resize((int(tile_size_20x), int(tile_size_20x)), Image.BICUBIC)
                out_name = f"{slide_name}_x{int(x)}_y{int(y)}_20x_{int(tile_size_20x)}.jpg"
                img.save(patch_dir / out_name, quality=int(jpeg_quality))
            sec_export = float(time.perf_counter() - t0)

        sec_overlay = 0.0
        if save_overlay:
            t0 = time.perf_counter()
            render_overlay(slide, sel, level0_tile_size=int(tile0), out_png=overlay_png, max_dim=int(mask_max_dim))
            sec_overlay = float(time.perf_counter() - t0)

        return SlideSampleResult(
            slide=slide_name,
            status="ok",
            width=int(w0),
            height=int(h0),
            objective_power=objective,
            level0_tile_size=int(tile0),
            level0_step_size=int(step0),
            n_candidates=int(cand_arr.shape[0]),
            n_selected=int(sel.shape[0]),
            locator_csv=str(locator_csv),
            overlay_png=str(overlay_png if save_overlay else ""),
            read_backend=backend,
            sec_open=sec_open,
            sec_mask=sec_mask,
            sec_filter=sec_filter,
            sec_sample=sec_sample,
            sec_write=sec_write,
            sec_export=sec_export,
            sec_overlay=sec_overlay,
            sec_total=float(time.perf_counter() - t_start),
        )
    except Exception as exc:  # noqa: BLE001
        return SlideSampleResult(
            slide=slide_name,
            status="error",
            width=0,
            height=0,
            objective_power=None,
            level0_tile_size=0,
            level0_step_size=0,
            n_candidates=0,
            n_selected=0,
            locator_csv=str(locator_csv),
            overlay_png=str(overlay_png),
            sec_total=float(time.perf_counter() - t_start),
            error=str(exc),
        )


def main() -> None:
    args = parse_args()
    if not args.wsi_dir.exists():
        raise SystemExit(f"Missing wsi_dir: {args.wsi_dir}")
    if args.patches_per_slide <= 0:
        raise SystemExit("--patches_per_slide must be > 0")

    slides = sorted(
        p for p in args.wsi_dir.iterdir() if p.is_file() and p.suffix.lower() in WSI_SUFFIXES
    )
    if args.max_slides > 0:
        slides = slides[: int(args.max_slides)]
    if not slides:
        raise SystemExit(f"No slides found in {args.wsi_dir}")

    out_dir = args.out_dir.resolve()
    (out_dir / "locators").mkdir(parents=True, exist_ok=True)
    if args.export_patches:
        (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    if not args.no_overlay:
        (out_dir / "overlay").mkdir(parents=True, exist_ok=True)

    print(f"[start] slides={len(slides)} workers={args.workers} out_dir={out_dir}", flush=True)
    print(
        f"[cfg] tile_size_20x={args.tile_size_20x} step_size_20x={args.step_size_20x} "
        f"patches_per_slide={args.patches_per_slide} min_tissue_frac={args.min_tissue_frac} "
        f"reader={args.reader} overwrite={args.overwrite}",
        flush=True,
    )

    results: List[SlideSampleResult] = []
    done = 0
    ok = 0
    skipped = 0
    err = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {
            ex.submit(
                sample_one_slide,
                wsi_path=slide_path,
                out_dir=out_dir,
                tile_size_20x=int(args.tile_size_20x),
                step_size_20x=int(args.step_size_20x),
                patches_per_slide=int(args.patches_per_slide),
                mask_max_dim=int(args.mask_max_dim),
                sat_thresh=int(args.sat_thresh),
                value_max=int(args.value_max),
                min_tissue_frac=float(args.min_tissue_frac),
                median_size=int(args.median_size),
                morph_size=int(args.morph_size),
                seed=int(args.seed),
                reader=str(args.reader),
                overwrite=bool(args.overwrite),
                jpeg_quality=int(args.jpeg_quality),
                save_overlay=not bool(args.no_overlay),
                export_patches=bool(args.export_patches),
            ): slide_path
            for slide_path in slides
        }

        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            if res.status == "ok":
                ok += 1
                print(
                    f"[slide] {res.slide} ok backend={res.read_backend} cand={res.n_candidates} sel={res.n_selected} "
                    f"sec(total={res.sec_total:.2f}, open={res.sec_open:.2f}, mask={res.sec_mask:.2f}, "
                    f"filter={res.sec_filter:.2f}, sample={res.sec_sample:.2f}, write={res.sec_write:.2f}, "
                    f"export={res.sec_export:.2f}, overlay={res.sec_overlay:.2f})",
                    flush=True,
                )
            elif res.status == "skipped":
                skipped += 1
            else:
                err += 1
            if done % 10 == 0 or done == len(slides):
                print(f"[progress] {done}/{len(slides)} ok={ok} skipped={skipped} err={err}", flush=True)
                if res.status == "error":
                    print(f"[error] {res.slide}: {res.error}", flush=True)

    rows = [r.__dict__ for r in sorted(results, key=lambda x: x.slide)]
    backend_counts = {}
    for r in results:
        backend_counts[r.read_backend] = int(backend_counts.get(r.read_backend, 0) + 1)
    ok_rows = [r for r in results if r.status == "ok"]
    summary = {
        "wsi_dir": str(args.wsi_dir.resolve()),
        "out_dir": str(out_dir),
        "slides_total": int(len(slides)),
        "slides_ok": int(ok),
        "slides_skipped": int(skipped),
        "slides_error": int(err),
        "tile_size_20x": int(args.tile_size_20x),
        "step_size_20x": int(args.step_size_20x),
        "patches_per_slide": int(args.patches_per_slide),
        "min_tissue_frac": float(args.min_tissue_frac),
        "workers": int(args.workers),
        "reader": str(args.reader),
        "overwrite": bool(args.overwrite),
        "n_candidates_total": int(sum(r.n_candidates for r in results)),
        "n_selected_total": int(sum(r.n_selected for r in results)),
        "backend_counts": backend_counts,
        "timing_sec": {
            "total_ok_sum": float(sum(r.sec_total for r in ok_rows)),
            "open_sum": float(sum(r.sec_open for r in ok_rows)),
            "mask_sum": float(sum(r.sec_mask for r in ok_rows)),
            "filter_sum": float(sum(r.sec_filter for r in ok_rows)),
            "sample_sum": float(sum(r.sec_sample for r in ok_rows)),
            "write_sum": float(sum(r.sec_write for r in ok_rows)),
            "export_sum": float(sum(r.sec_export for r in ok_rows)),
            "overlay_sum": float(sum(r.sec_overlay for r in ok_rows)),
            "total_ok_mean": float(sum(r.sec_total for r in ok_rows) / max(1, len(ok_rows))),
        },
        "locator_csv_dir": str(out_dir / "locators"),
        "export_patches": bool(args.export_patches),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_slide_summary.json").write_text(json.dumps(rows, indent=2))

    print(f"[done] wrote {out_dir / 'summary.json'}", flush=True)
    print(f"[done] wrote {out_dir / 'per_slide_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
