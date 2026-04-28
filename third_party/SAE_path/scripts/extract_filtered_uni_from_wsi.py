#!/usr/bin/env python3
"""
Build a denser, user-controlled UNI2-h feature set from local WSIs.

For each .svs in a directory, this script:
1. Builds a dense tile grid in level-0 coordinates.
2. Applies a simple tissue mask at a lower-resolution slide level.
3. Keeps tiles whose mask overlap exceeds a threshold.
4. Crops the kept tiles from the WSI, resizes them to the requested 20x tile size,
   and runs UNI2-h to obtain 1536-d embeddings.
5. Saves a CLAM-style H5 with features + coords.

The filtering is intentionally simple and auditable:
- HSV saturation threshold
- brightness ceiling to reject white background
- optional small morphological cleanup using PIL filters
- overlap-based acceptance (not center-point only)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from e

try:
    from cucim import CuImage
    HAS_CUCIM = True
except Exception:
    CuImage = None
    HAS_CUCIM = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.uni import get_uni

UNI_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def safe_float(x: object, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def infer_objective_power(slide: "openslide.OpenSlide") -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            val = safe_float(props.get(key), -1.0)
            if val > 0:
                return val
    mpp_x = safe_float(props.get("openslide.mpp-x"), -1.0)
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def level0_size_from_20x(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(tile_size_20x * (objective_power / 20.0))))


def choose_mask_level(slide: "openslide.OpenSlide", target_max_dim: int) -> int:
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


def read_region_array(
    slide: "openslide.OpenSlide",
    cuimg,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    level: int,
) -> np.ndarray:
    if cuimg is not None:
        arr = np.asarray(
            cuimg.read_region(
                location=(int(x), int(y)),
                size=(int(w), int(h)),
                level=int(level),
            )
        )
        if arr.ndim != 3:
            raise ValueError(f"Unexpected CuImage read shape: {tuple(arr.shape)}")
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        return np.ascontiguousarray(arr)
    pil_img = slide.read_region((int(x), int(y)), int(level), (int(w), int(h))).convert("RGB")
    return np.ascontiguousarray(np.asarray(pil_img, dtype=np.uint8))


def read_region_rgb(
    slide: "openslide.OpenSlide",
    cuimg,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    level: int,
) -> Image.Image:
    return Image.fromarray(read_region_array(slide, cuimg, x=x, y=y, w=w, h=h, level=level))


def preprocess_batch_for_uni(
    batch_np: np.ndarray,
    *,
    tile_size_20x: int,
    device: torch.device,
) -> torch.Tensor:
    x = torch.from_numpy(batch_np).to(device=device, dtype=torch.uint8, non_blocking=True)
    x = x.permute(0, 3, 1, 2).contiguous()

    if x.shape[-2] != tile_size_20x or x.shape[-1] != tile_size_20x:
        x = F.interpolate(
            x.to(dtype=torch.float32),
            size=(tile_size_20x, tile_size_20x),
            mode="bilinear",
            align_corners=False,
        )
    else:
        x = x.to(dtype=torch.float32)

    if x.shape[-2] >= UNI_INPUT_SIZE and x.shape[-1] >= UNI_INPUT_SIZE:
        top = max(0, (x.shape[-2] - UNI_INPUT_SIZE) // 2)
        left = max(0, (x.shape[-1] - UNI_INPUT_SIZE) // 2)
        x = x[:, :, top:top + UNI_INPUT_SIZE, left:left + UNI_INPUT_SIZE]
    else:
        x = F.interpolate(
            x,
            size=(UNI_INPUT_SIZE, UNI_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

    x = x / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def preprocess_batch_tensor_for_uni(
    batch: torch.Tensor,
    *,
    tile_size_20x: int,
    device: torch.device,
) -> torch.Tensor:
    x = batch
    if x.ndim != 4:
        raise ValueError(f"Expected 4D batch tensor, got shape={tuple(x.shape)}")
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2).contiguous()
    elif x.shape[1] != 3:
        raise ValueError(f"Expected channels-last or channels-first RGB batch, got shape={tuple(x.shape)}")

    x = x.to(device=device, dtype=torch.uint8, non_blocking=True)
    if x.shape[-2] != tile_size_20x or x.shape[-1] != tile_size_20x:
        x = F.interpolate(
            x.to(dtype=torch.float32),
            size=(tile_size_20x, tile_size_20x),
            mode="bilinear",
            align_corners=False,
        )
    else:
        x = x.to(dtype=torch.float32)

    if x.shape[-2] >= UNI_INPUT_SIZE and x.shape[-1] >= UNI_INPUT_SIZE:
        top = max(0, (x.shape[-2] - UNI_INPUT_SIZE) // 2)
        left = max(0, (x.shape[-1] - UNI_INPUT_SIZE) // 2)
        x = x[:, :, top:top + UNI_INPUT_SIZE, left:left + UNI_INPUT_SIZE]
    else:
        x = F.interpolate(
            x,
            size=(UNI_INPUT_SIZE, UNI_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

    x = x / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def build_tissue_mask(
    slide: "openslide.OpenSlide",
    cuimg,
    mask_level: int,
    *,
    sat_thresh: int,
    value_max: int,
    median_size: int,
    morph_size: int,
) -> np.ndarray:
    level_w, level_h = slide.level_dimensions[mask_level]
    img = read_region_rgb(slide, cuimg, x=0, y=0, w=level_w, h=level_h, level=mask_level)
    if median_size > 1 and median_size % 2 == 1:
        img = img.filter(ImageFilter.MedianFilter(size=median_size))
    hsv = img.convert("HSV")
    hsv_np = np.asarray(hsv, dtype=np.uint8)
    sat = hsv_np[..., 1]
    val = hsv_np[..., 2]
    mask = (sat >= sat_thresh) & (val <= value_max)

    if morph_size > 1 and morph_size % 2 == 1:
        mask_img = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=morph_size))
        mask_img = mask_img.filter(ImageFilter.MinFilter(size=morph_size))
        mask = np.asarray(mask_img, dtype=np.uint8) > 0

    return np.asarray(mask, dtype=bool)


def build_dense_grid_axes(
    width: int,
    height: int,
    tile_size: int,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_x = max(0, width - tile_size)
    max_y = max(0, height - tile_size)
    xs = np.arange(0, max_x + 1, step_size, dtype=np.int64)
    ys = np.arange(0, max_y + 1, step_size, dtype=np.int64)
    return xs, ys


def filter_dense_grid_by_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    mask: np.ndarray,
    *,
    level_downsample: float,
    tile_size_level0: int,
    min_tissue_frac: float,
) -> np.ndarray:
    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    mask_h, mask_w = mask.shape
    scale = 1.0 / max(level_downsample, 1e-12)
    x1 = np.floor(xs.astype(np.float64) * scale).astype(np.int64)
    x2 = np.ceil((xs.astype(np.float64) + float(tile_size_level0)) * scale).astype(np.int64)
    x1 = np.clip(x1, 0, max(0, mask_w - 1))
    x2 = np.clip(x2, x1 + 1, mask_w)

    mask_i = mask.astype(np.uint32, copy=False)
    ii = np.pad(mask_i, ((1, 0), (1, 0)), mode="constant")
    ii = ii.cumsum(axis=0).cumsum(axis=1)

    kept_rows: list[np.ndarray] = []
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


def read_coords_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["coord_x"]), int(row["coord_y"])))
    return np.asarray(rows, dtype=np.int64) if rows else np.empty((0, 2), dtype=np.int64)


def write_coords_csv(path: Path, coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["coord_x", "coord_y"])
        for x, y in coords:
            writer.writerow([int(x), int(y)])


def write_h5(path: Path, features: np.ndarray, coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    annots = np.zeros((1, coords.shape[0], 1), dtype=np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features[None, :, :])
        f.create_dataset("coords", data=coords[None, :, :])
        f.create_dataset("coords_patching", data=coords)
        f.create_dataset("annots", data=annots)


def slide_summary_path(summary_dir: Path, slide_key: str) -> Path:
    return summary_dir / f"{slide_key}.summary.json"


def write_json_text(path: Path, payload: object) -> None:
    # Recreate parent in case temp directories were cleaned during long runs.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def save_filter_visualization(
    slide: "openslide.OpenSlide",
    cuimg,
    coords: np.ndarray,
    *,
    tile_size_level0: int,
    out_prefix: Path,
    thumb_max_dim: int,
    alpha: int,
    save_mode: str,
) -> None:
    render_level = choose_mask_level(slide, thumb_max_dim)
    render_w, render_h = slide.level_dimensions[render_level]
    render_downsample = float(slide.level_downsamples[render_level])
    base = read_region_rgb(slide, cuimg, x=0, y=0, w=render_w, h=render_h, level=render_level)
    base_np = np.asarray(base)

    mask = np.zeros((render_h, render_w), dtype=np.uint8)
    scale = 1.0 / max(render_downsample, 1e-12)
    for x0, y0 in coords:
        x1 = int(math.floor(float(x0) * scale))
        y1 = int(math.floor(float(y0) * scale))
        x2 = int(math.ceil((float(x0) + tile_size_level0) * scale))
        y2 = int(math.ceil((float(y0) + tile_size_level0) * scale))
        x1 = max(0, min(x1, render_w - 1))
        y1 = max(0, min(y1, render_h - 1))
        x2 = max(x1 + 1, min(x2, render_w))
        y2 = max(y1 + 1, min(y2, render_h))
        mask[y1:y2, x1:x2] = 255

    mask_rgb = np.zeros((render_h, render_w, 3), dtype=np.uint8)
    mask_rgb[..., 1] = mask
    overlay_rgba = np.zeros((render_h, render_w, 4), dtype=np.uint8)
    overlay_rgba[..., 1] = mask
    overlay_rgba[..., 3] = np.where(mask > 0, np.uint8(max(0, min(alpha, 255))), 0)
    overlay = Image.alpha_composite(
        base.convert("RGBA"),
        Image.fromarray(overlay_rgba),
    ).convert("RGB")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_prefix.with_name(out_prefix.name + "__overlay.png"))
    if save_mode == "all":
        base.save(out_prefix.with_name(out_prefix.name + "__thumb.png"))
        Image.fromarray(mask_rgb).save(out_prefix.with_name(out_prefix.name + "__mask.png"))


class SlidePatchDataset(Dataset):
    def __init__(
        self,
        wsi_path: Path,
        coords: np.ndarray,
        *,
        tile_size_level0: int,
        reader: str,
    ) -> None:
        self.wsi_path = str(wsi_path)
        self.coords = np.asarray(coords, dtype=np.int64)
        self.tile_size_level0 = int(tile_size_level0)
        self.reader = str(reader)
        self._slide = None
        self._cuimg = None
        self._reader_warned = False

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def _ensure_open(self) -> None:
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.wsi_path)
        if self._cuimg is None and self.reader in {"auto", "cucim"} and HAS_CUCIM:
            try:
                self._cuimg = CuImage(self.wsi_path)
            except Exception as e:
                if self.reader == "cucim":
                    raise
                if not self._reader_warned:
                    print(f"[warn] {Path(self.wsi_path).stem}: cucim open failed in loader, fallback to openslide: {e}", flush=True)
                    self._reader_warned = True

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._ensure_open()
        x0, y0 = self.coords[int(idx)]
        patch_np = read_region_array(
            self._slide,
            self._cuimg,
            x=int(x0),
            y=int(y0),
            w=self.tile_size_level0,
            h=self.tile_size_level0,
            level=0,
        )
        return torch.from_numpy(patch_np)

    def __del__(self) -> None:
        try:
            if self._slide is not None:
                self._slide.close()
        except Exception:
            pass


def embed_coords(
    wsi_path: Path,
    coords: np.ndarray,
    *,
    tile_size_level0: int,
    tile_size_20x: int,
    batch_size: int,
    model: torch.nn.Module,
    device: torch.device,
    reader: str,
    loader_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> np.ndarray:
    feats: list[np.ndarray] = []
    n = int(coords.shape[0])
    if n == 0:
        return np.empty((0, 1536), dtype=np.float32)

    dataset = SlidePatchDataset(
        wsi_path=wsi_path,
        coords=coords,
        tile_size_level0=tile_size_level0,
        reader=reader,
    )

    num_workers = max(0, int(loader_workers))
    loader_kwargs = dict(
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(bool(pin_memory) and device.type == "cuda"),
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    t0 = time.time()
    num_batches = max(1, math.ceil(n / max(1, batch_size)))
    for batch_idx, batch in enumerate(loader, start=1):
        x = preprocess_batch_tensor_for_uni(
            batch,
            tile_size_20x=tile_size_20x,
            device=device,
        )
        with torch.inference_mode():
            z = model(x).detach().cpu().numpy().astype(np.float32, copy=False)
        feats.append(z)
        done = min(batch_idx * batch_size, n)
        elapsed = time.time() - t0
        pct = 100.0 * done / max(1, n)
        print(
            f"    [embed] batch {batch_idx}/{num_batches} "
            f"tiles {done}/{n} ({pct:.1f}%) elapsed={elapsed:.1f}s",
            flush=True,
        )
    return np.concatenate(feats, axis=0)


def filter_one_slide(
    wsi_path: Path,
    *,
    coords_dir: Path | None,
    summary_dir: Path,
    out_dir: Path,
    viz_dir: Path | None,
    tile_size_20x: int,
    step_size_20x: int,
    mask_max_dim: int,
    sat_thresh: int,
    value_max: int,
    min_tissue_frac: float,
    median_size: int,
    morph_size: int,
    thumb_max_dim: int,
    overlay_alpha: int,
    viz_mode: str,
    reader: str,
    overwrite: bool,
) -> dict:
    t0 = time.time()
    slide_key = wsi_path.stem
    out_h5 = out_dir / f"{slide_key}.h5"
    out_json = slide_summary_path(summary_dir, slide_key)
    out_coords = (coords_dir / f"{slide_key}.coords.csv") if coords_dir is not None else None
    viz_prefix = (viz_dir / slide_key) if viz_dir is not None else None

    if out_h5.exists() and out_json.exists() and (not overwrite):
        filter_summary = None
        if out_json.exists():
            try:
                payload = json.loads(out_json.read_text())
                filter_summary = payload.get("filter", payload)
            except Exception:
                filter_summary = None
        if filter_summary is None:
            filter_summary = {
                "slide_key": slide_key,
                "wsi_path": str(wsi_path),
                "coords_csv": (str(out_coords) if out_coords is not None else None),
            }
        filter_summary["status"] = "skipped_existing_all_outputs"
        filter_summary["h5_path"] = str(out_h5)
        filter_summary["elapsed_sec"] = 0.0
        return filter_summary

    if out_coords is not None and out_coords.exists() and (not overwrite):
        if out_json.exists():
            payload = json.loads(out_json.read_text())
            summary = payload.get("filter", payload)
        else:
            coords_np = read_coords_csv(out_coords)
            summary = {
                "slide_key": slide_key,
                "wsi_path": str(wsi_path),
                "coords_csv": str(out_coords),
                "candidate_tiles": 0,
                "kept_tiles": int(coords_np.shape[0]),
                "keep_fraction": 0.0,
            }
        summary["status"] = "reused_filter"
        summary["elapsed_sec"] = 0.0
        return summary

    slide = openslide.OpenSlide(str(wsi_path))
    cuimg = None
    try:
        if reader in {"auto", "cucim"} and HAS_CUCIM:
            try:
                cuimg = CuImage(str(wsi_path))
            except Exception as e:
                if reader == "cucim":
                    raise
                print(f"[warn] {slide_key}: cucim open failed, falling back to openslide: {e}", flush=True)
        width0, height0 = slide.dimensions
        objective_power = infer_objective_power(slide)
        tile_size_level0 = level0_size_from_20x(tile_size_20x, objective_power)
        step_size_level0 = level0_size_from_20x(step_size_20x, objective_power)
        mask_level = choose_mask_level(slide, mask_max_dim)
        level_downsample = float(slide.level_downsamples[mask_level])
        reader_used = ("cucim" if cuimg is not None else "openslide")

        xs, ys = build_dense_grid_axes(width0, height0, tile_size_level0, step_size_level0)
        candidate_tiles = int(xs.size * ys.size)
        print(
            f"[filter:start] {slide_key}: dims={width0}x{height0} "
            f"obj={objective_power:.1f}x tile0={tile_size_level0} step0={step_size_level0} "
            f"grid={xs.size}x{ys.size} candidates={candidate_tiles} mask_level={mask_level} "
            f"reader={reader_used}",
            flush=True,
        )

        mask = build_tissue_mask(
            slide,
            cuimg,
            mask_level,
            sat_thresh=sat_thresh,
            value_max=value_max,
            median_size=median_size,
            morph_size=morph_size,
        )

        coords_np = filter_dense_grid_by_mask(
            xs,
            ys,
            mask,
            level_downsample=level_downsample,
            tile_size_level0=tile_size_level0,
            min_tissue_frac=min_tissue_frac,
        )

        if out_coords is not None:
            write_coords_csv(out_coords, coords_np)
        if viz_prefix is not None:
            save_filter_visualization(
                slide,
                cuimg,
                coords_np,
                tile_size_level0=tile_size_level0,
                out_prefix=viz_prefix,
                thumb_max_dim=thumb_max_dim,
                alpha=overlay_alpha,
                save_mode=viz_mode,
            )

        summary = {
            "slide_key": slide_key,
            "status": ("ok" if coords_np.size > 0 else "no_tiles_kept"),
            "wsi_path": str(wsi_path),
            "coords_csv": (str(out_coords) if out_coords is not None else None),
            "viz_prefix": (str(viz_prefix) if viz_prefix is not None else None),
            "objective_power": float(objective_power),
            "tile_size_20x": int(tile_size_20x),
            "step_size_20x": int(step_size_20x),
            "tile_size_level0": int(tile_size_level0),
            "step_size_level0": int(step_size_level0),
            "mask_level": int(mask_level),
            "mask_level_downsample": float(level_downsample),
            "reader_used": reader_used,
            "candidate_tiles": candidate_tiles,
            "kept_tiles": int(coords_np.shape[0]),
            "keep_fraction": float(coords_np.shape[0] / max(1, candidate_tiles)),
            "elapsed_sec": float(time.time() - t0),
        }
    finally:
        slide.close()

    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {"slide_key": slide_key, "filter": summary}
    write_json_text(out_json, payload)
    return summary


def encode_one_slide_from_saved_coords(
    filter_summary: dict,
    *,
    out_dir: Path,
    summary_dir: Path,
    tile_size_20x: int,
    batch_size: int,
    reader: str,
    overwrite: bool,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    loader_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> dict:
    t0 = time.time()
    slide_key = str(filter_summary["slide_key"])
    wsi_path = Path(str(filter_summary["wsi_path"]))
    out_h5 = out_dir / f"{slide_key}.h5"
    out_json = slide_summary_path(summary_dir, slide_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    coords_csv = filter_summary.get("coords_csv")
    if not coords_csv:
        summary = {
            "slide_key": slide_key,
            "status": "missing_coords_csv",
            "h5_path": str(out_h5),
            "elapsed_sec": 0.0,
        }
        write_json_text(out_json, {"slide_key": slide_key, "filter": filter_summary, "encode": summary})
        return summary
    coords_np = read_coords_csv(Path(str(coords_csv)))
    if coords_np.size == 0:
        summary = {
            "slide_key": slide_key,
            "status": "no_tiles_kept",
            "h5_path": str(out_h5),
            "kept_tiles": 0,
            "elapsed_sec": 0.0,
        }
        write_json_text(out_json, {"slide_key": slide_key, "filter": filter_summary, "encode": summary})
        return summary
    if out_h5.exists() and (not overwrite):
        if out_json.exists():
            payload = json.loads(out_json.read_text())
            summary = payload.get("encode", payload)
        else:
            feature_dim = 0
            try:
                with h5py.File(out_h5, "r") as f:
                    shape = tuple(f["features"].shape)
                if len(shape) >= 3:
                    feature_dim = int(shape[-1])
            except Exception:
                feature_dim = 0
            summary = {
                "slide_key": slide_key,
                "h5_path": str(out_h5),
                "kept_tiles": int(coords_np.shape[0]),
                "feature_dim": int(feature_dim),
            }
        summary["status"] = "reused_encode"
        summary["elapsed_sec"] = 0.0
        return summary

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        objective_power = infer_objective_power(slide)
        tile_size_level0 = level0_size_from_20x(tile_size_20x, objective_power)
        reader_used = reader
        print(
            f"[encode:start] {slide_key}: tiles={coords_np.shape[0]} "
            f"tile0={tile_size_level0} batch_size={batch_size} reader={reader_used}",
            flush=True,
        )
        feats_np = embed_coords(
            wsi_path,
            coords_np,
            tile_size_level0=tile_size_level0,
            tile_size_20x=tile_size_20x,
            batch_size=batch_size,
            model=model,
            device=device,
            reader=reader,
            loader_workers=loader_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
    finally:
        slide.close()

    write_h5(out_h5, feats_np, coords_np)
    summary = {
        "slide_key": slide_key,
        "status": "ok",
        "wsi_path": str(wsi_path),
        "h5_path": str(out_h5),
        "coords_csv": str(coords_csv),
        "kept_tiles": int(coords_np.shape[0]),
        "feature_dim": int(feats_np.shape[1]),
        "reader_used": reader_used,
        "elapsed_sec": float(time.time() - t0),
    }
    write_json_text(out_json, {"slide_key": slide_key, "filter": filter_summary, "encode": summary})
    return summary


def parse_gpu_list(gpu_list: str) -> list[str]:
    vals = []
    for x in str(gpu_list).split(","):
        t = x.strip()
        if t != "":
            vals.append(t)
    return vals


def _dynamic_encode_worker(
    gpu_id: str,
    task_queue: "mp.Queue",
    result_queue: "mp.Queue",
    *,
    out_dir: str,
    summary_dir: str,
    tile_size_20x: int,
    batch_size: int,
    reader: str,
    overwrite: bool,
    loader_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, transform = get_uni(str(device))
    while True:
        task = task_queue.get()
        if task is None:
            break
        slide_key = str(task.get("slide_key", ""))
        try:
            encode_summary = encode_one_slide_from_saved_coords(
                task,
                out_dir=Path(out_dir),
                summary_dir=Path(summary_dir),
                tile_size_20x=tile_size_20x,
                batch_size=batch_size,
                reader=reader,
                overwrite=overwrite,
                model=model,
                transform=transform,
                device=device,
                loader_workers=loader_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
            )
            result_queue.put({"ok": True, "slide_key": slide_key, "encode": encode_summary})
        except Exception as e:
            result_queue.put({"ok": False, "slide_key": slide_key, "error": repr(e)})


def encode_slides_dynamic_multi_gpu(
    filter_summaries: list[dict],
    *,
    gpu_ids: list[str],
    out_dir: Path,
    summary_dir: Path,
    tile_size_20x: int,
    batch_size: int,
    reader: str,
    overwrite: bool,
    loader_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> list[dict]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for fs in filter_summaries:
        task_queue.put(fs)
    for _ in gpu_ids:
        task_queue.put(None)

    workers = []
    for gpu_id in gpu_ids:
        p = ctx.Process(
            target=_dynamic_encode_worker,
            args=(gpu_id, task_queue, result_queue),
            kwargs=dict(
                out_dir=str(out_dir),
                summary_dir=str(summary_dir),
                tile_size_20x=int(tile_size_20x),
                batch_size=int(batch_size),
                reader=reader,
                overwrite=bool(overwrite),
                loader_workers=int(loader_workers),
                prefetch_factor=int(prefetch_factor),
                pin_memory=bool(pin_memory),
            ),
        )
        p.start()
        workers.append(p)

    by_slide = {str(s["slide_key"]): s for s in filter_summaries}
    total = len(filter_summaries)
    done = 0
    run_summary = []
    failures = []
    while done < total:
        msg = result_queue.get()
        done += 1
        slide_key = str(msg.get("slide_key", ""))
        if not bool(msg.get("ok", False)):
            failures.append(msg)
            print(f"[encode {done}/{total}] {slide_key}: FAILED {msg.get('error')}", flush=True)
            continue
        encode_summary = msg["encode"]
        print(
            f"[encode {done}/{total}] {slide_key}: status={encode_summary.get('status')} "
            f"tiles={encode_summary.get('kept_tiles', 0)} "
            f"elapsed={encode_summary.get('elapsed_sec', 0.0):.1f}s",
            flush=True,
        )
        run_summary.append(
            {
                "slide_key": slide_key,
                "filter": by_slide[slide_key],
                "encode": encode_summary,
            }
        )

    for p in workers:
        p.join()

    if failures:
        first = failures[0]
        raise RuntimeError(f"Dynamic multi-GPU encode failed for {first.get('slide_key')}: {first.get('error')}")

    run_summary.sort(key=lambda x: str(x["slide_key"]))
    return run_summary


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_optional_dir_arg(raw: object) -> Path | None:
    txt = str(raw).strip()
    if txt.lower() in {"", ".", "none", "null"}:
        return None
    return Path(txt)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wsi_dir", type=Path, default=Path("wsi/hnsc_hpv"), help="Directory of .svs slides.")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("extracted_features/hnsc_hpv_filtered_uni2h"),
        help="Directory for output H5 embedding files only.",
    )
    ap.add_argument(
        "--summary_dir",
        type=Path,
        default=Path("extracted_features/hnsc_hpv_filtered_uni2h_summary"),
        help="Directory for per-slide combined summaries and run summaries.",
    )
    ap.add_argument(
        "--coords_dir",
        type=str,
        default="extracted_features/hnsc_hpv_filtered_uni2h_coords",
        help="Directory for per-slide coordinate CSV files. Use '', '.', or 'none' to disable.",
    )
    ap.add_argument(
        "--viz_dir",
        type=str,
        default="extracted_features/hnsc_hpv_filtered_uni2h_viz",
        help="Directory for per-slide filtering thumbnails/overlays. Use '', '.', or 'none' to disable.",
    )
    ap.add_argument(
        "--viz_mode",
        type=str,
        default="overlay",
        choices=["overlay", "all"],
        help="Visualization output mode. Default writes only __overlay.png.",
    )
    ap.add_argument("--tile_size_20x", type=int, default=256, help="Tile size at 20x magnification.")
    ap.add_argument("--step_size_20x", type=int, default=256, help="Grid step at 20x magnification.")
    ap.add_argument("--mask_max_dim", type=int, default=2048, help="Target max size for tissue-mask level.")
    ap.add_argument("--thumb_max_dim", type=int, default=2048, help="Max dimension for filter thumbnails.")
    ap.add_argument("--overlay_alpha", type=int, default=120, help="Alpha for the filter overlay (0-255).")
    ap.add_argument("--sat_thresh", type=int, default=20, help="HSV saturation threshold (0-255).")
    ap.add_argument("--value_max", type=int, default=245, help="HSV value ceiling to reject bright background.")
    ap.add_argument(
        "--min_tissue_frac",
        type=float,
        default=0.15,
        help="Minimum fraction of mask-positive pixels within a tile to keep it.",
    )
    ap.add_argument(
        "--median_size",
        type=int,
        default=3,
        help="Odd median filter size for the low-res RGB image before thresholding.",
    )
    ap.add_argument(
        "--morph_size",
        type=int,
        default=5,
        help="Odd max/min filter size for simple morphological cleanup. Set 1 to disable.",
    )
    ap.add_argument("--batch_size", type=int, default=2048, help="UNI embedding batch size.")
    ap.add_argument(
        "--loader_workers",
        type=int,
        default=0,
        help="Patch loader worker processes for encode stage (0 disables DataLoader workers).",
    )
    ap.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor when loader_workers > 0.")
    ap.add_argument("--pin_memory", action="store_true", help="Use pinned host memory for faster H2D transfer.")
    ap.add_argument("--filter_workers", type=int, default=4, help="Thread count for filter+overlay stage.")
    ap.add_argument(
        "--reader",
        type=str,
        default="auto",
        choices=["auto", "openslide", "cucim"],
        help="Tile reader backend. 'auto' prefers cucim when available.",
    )
    ap.add_argument("--device", type=str, default="auto", help="cuda:0, cpu, or auto.")
    ap.add_argument("--gpu_list", type=str, default="", help="Comma-separated GPU IDs for dynamic multi-GPU encode.")
    ap.add_argument("--dynamic_gpu", action="store_true", help="Enable dynamic multi-GPU slide scheduling for encode stage.")
    ap.add_argument("--max_slides", type=int, default=0, help="If >0, only process the first N slides.")
    ap.add_argument("--slide_key", type=str, default="", help="If set, process only this slide stem.")
    ap.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the sorted slide list into this many shards for multi-process / multi-GPU runs.",
    )
    ap.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Which shard to process (0-based). Must be in [0, num_shards).",
    )
    ap.add_argument("--filter_only", action="store_true", help="Only compute coords + overlays; skip UNI encoding.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing H5 outputs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.wsi_dir.exists():
        raise SystemExit(f"Missing WSI directory: {args.wsi_dir}")

    coords_dir = parse_optional_dir_arg(args.coords_dir)
    summary_dir = args.summary_dir
    viz_dir = parse_optional_dir_arg(args.viz_dir)
    if (not args.filter_only) and coords_dir is None:
        raise SystemExit("coords_dir cannot be disabled unless --filter_only is set.")
    device = resolve_device(args.device)

    slides = sorted(p for p in args.wsi_dir.iterdir() if p.is_file() and p.suffix.lower() == ".svs")
    if args.slide_key:
        slides = [p for p in slides if p.stem == args.slide_key]
    if args.max_slides > 0:
        slides = slides[:args.max_slides]
    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.num_shards > 1:
        slides = [p for i, p in enumerate(slides) if (i % args.num_shards) == args.shard_index]
    if not slides:
        raise SystemExit("No matching .svs slides found.")

    if args.reader == "cucim" and (not HAS_CUCIM):
        raise SystemExit("Requested --reader cucim but cucim is not installed.")
    if args.reader == "auto":
        reader_msg = "cucim" if HAS_CUCIM else "openslide"
    else:
        reader_msg = args.reader
    print(f"[setup] reader preference: {reader_msg}", flush=True)
    if args.num_shards > 1:
        print(
            f"[setup] shard {args.shard_index + 1}/{args.num_shards}: {len(slides)} assigned slide(s)",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    if coords_dir is not None:
        coords_dir.mkdir(parents=True, exist_ok=True)
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage 1/2] filtering {len(slides)} slide(s)", flush=True)
    filter_kwargs = dict(
        coords_dir=coords_dir,
        summary_dir=summary_dir,
        out_dir=args.out_dir,
        viz_dir=viz_dir,
        tile_size_20x=args.tile_size_20x,
        step_size_20x=args.step_size_20x,
        mask_max_dim=args.mask_max_dim,
        sat_thresh=args.sat_thresh,
        value_max=args.value_max,
        min_tissue_frac=args.min_tissue_frac,
        median_size=args.median_size,
        morph_size=args.morph_size,
        thumb_max_dim=args.thumb_max_dim,
        overlay_alpha=args.overlay_alpha,
        viz_mode=args.viz_mode,
        reader=args.reader,
        overwrite=args.overwrite,
    )
    filter_summaries: list[dict] = []
    max_workers = max(1, int(args.filter_workers))
    total_filter = len(slides)
    if max_workers > 1 and len(slides) > 1:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(filter_one_slide, wsi_path, **filter_kwargs): wsi_path for wsi_path in slides}
            done_count = 0
            for fut in cf.as_completed(futures):
                summary = fut.result()
                done_count += 1
                print(
                    f"[filter {done_count}/{total_filter}] {summary['slide_key']}: status={summary.get('status')} "
                    f"kept={summary.get('kept_tiles', 0)} / {summary.get('candidate_tiles', 0)} "
                    f"({100.0 * summary.get('keep_fraction', 0.0):.1f}%) "
                    f"elapsed={summary.get('elapsed_sec', 0.0):.1f}s",
                    flush=True,
                )
                filter_summaries.append(summary)
    else:
        for idx, wsi_path in enumerate(slides, start=1):
            summary = filter_one_slide(wsi_path, **filter_kwargs)
            print(
                f"[filter {idx}/{total_filter}] {summary['slide_key']}: status={summary.get('status')} "
                f"kept={summary.get('kept_tiles', 0)} / {summary.get('candidate_tiles', 0)} "
                f"({100.0 * summary.get('keep_fraction', 0.0):.1f}%) "
                f"elapsed={summary.get('elapsed_sec', 0.0):.1f}s",
                flush=True,
            )
            filter_summaries.append(summary)
    filter_summaries.sort(key=lambda x: str(x["slide_key"]))
    total_candidates = sum(int(s.get("candidate_tiles", 0)) for s in filter_summaries)
    total_kept = sum(int(s.get("kept_tiles", 0)) for s in filter_summaries)
    print(
        f"[stage 1/2] done: kept {total_kept} / {total_candidates} "
        f"({100.0 * total_kept / max(1, total_candidates):.1f}%) across {len(filter_summaries)} slide(s)",
        flush=True,
    )

    if args.filter_only:
        summary_path = summary_dir / "run_summary.json"
        write_json_text(summary_path, [{"slide_key": s["slide_key"], "filter": s} for s in filter_summaries])
        print(f"[ok] wrote {args.out_dir} and {summary_dir}", flush=True)
        return

    run_summary = []
    gpu_ids = parse_gpu_list(args.gpu_list)
    use_dynamic_gpu = bool(args.dynamic_gpu and device.type == "cuda" and len(gpu_ids) >= 2)
    if use_dynamic_gpu:
        print(f"[stage 2/2] dynamic multi-GPU encode on GPUs: {gpu_ids}", flush=True)
        run_summary = encode_slides_dynamic_multi_gpu(
            filter_summaries,
            gpu_ids=gpu_ids,
            out_dir=args.out_dir,
            summary_dir=summary_dir,
            tile_size_20x=args.tile_size_20x,
            batch_size=args.batch_size,
            reader=args.reader,
            overwrite=args.overwrite,
            loader_workers=args.loader_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
        )
    else:
        print(f"[stage 2/2] loading UNI2-h on {device}", flush=True)
        model, transform = get_uni(str(device))

        total_encode = len(filter_summaries)
        for idx, filter_summary in enumerate(filter_summaries, start=1):
            encode_summary = encode_one_slide_from_saved_coords(
                filter_summary,
                out_dir=args.out_dir,
                summary_dir=summary_dir,
                tile_size_20x=args.tile_size_20x,
                batch_size=args.batch_size,
                reader=args.reader,
                overwrite=args.overwrite,
                model=model,
                transform=transform,
                device=device,
                loader_workers=args.loader_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
            )
            print(
                f"[encode {idx}/{total_encode}] {filter_summary['slide_key']}: status={encode_summary.get('status')} "
                f"tiles={encode_summary.get('kept_tiles', 0)} "
                f"elapsed={encode_summary.get('elapsed_sec', 0.0):.1f}s",
                flush=True,
            )
            run_summary.append(
                {
                    "slide_key": filter_summary["slide_key"],
                    "filter": filter_summary,
                    "encode": encode_summary,
                }
            )
    total_encoded_tiles = sum(int(x["encode"].get("kept_tiles", 0)) for x in run_summary)
    total_encode_time = sum(float(x["encode"].get("elapsed_sec", 0.0)) for x in run_summary)
    print(
        f"[stage 2/2] done: encoded {total_encoded_tiles} tiles across {len(run_summary)} slide(s) "
        f"total_encode_time={total_encode_time:.1f}s",
        flush=True,
    )

    summary_path = summary_dir / "run_summary.json"
    write_json_text(summary_path, run_summary)
    print(f"[ok] wrote {args.out_dir} and {summary_dir}", flush=True)


if __name__ == "__main__":
    main()
