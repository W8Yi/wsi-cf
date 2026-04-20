from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _require_openslide():
    try:
        import openslide  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only in real runtime
        raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from exc
    return openslide


def open_slide(path: Path):
    openslide = _require_openslide()
    return openslide.OpenSlide(str(path))


def read_region_rgb(slide: Any, x0: int, y0: int, w: int, h: int) -> Image.Image:
    rgba = slide.read_region((int(x0), int(y0)), 0, (int(w), int(h))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def quick_tissue_score(img: Image.Image) -> float:
    arr = np.asarray(img, dtype=np.uint8)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
    return float(np.mean(gray < 230.0))


def quick_region_quality_metrics(img: Image.Image) -> dict[str, float]:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
    channel_max = arr.max(axis=-1).astype(np.float32)
    channel_min = arr.min(axis=-1).astype(np.float32)
    tissue_score = float(np.mean(gray < 230.0))
    dark_fraction = float(np.mean(gray < 160.0))
    saturation_fraction = float(np.mean((channel_max - channel_min) > 25.0))
    return {
        "tissue_score": tissue_score,
        "dark_fraction": dark_fraction,
        "saturation_fraction": saturation_fraction,
    }


def find_slide_path(slides_dir: Path, slide_key: str) -> Path | None:
    matches = sorted(slides_dir.glob(f"{slide_key}*.svs"))
    if matches:
        return matches[0]
    return None


def infer_objective_power(slide: Any) -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            try:
                value = float(props.get(key))
            except Exception:
                value = -1.0
            if value > 0:
                return value
    try:
        mpp_x = float(props.get("openslide.mpp-x", -1.0))
    except Exception:
        mpp_x = -1.0
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(float(tile_size_20x) * (float(objective_power) / 20.0))))


def level0_size_for_target_magnification(output_size_px: int, target_magnification: float, objective_power: float) -> int:
    target = float(target_magnification)
    if target <= 0:
        raise ValueError("target_magnification must be > 0")
    return max(1, int(round(float(output_size_px) * (float(objective_power) / target))))


def read_region_rgb_at_magnification(
    slide: Any,
    *,
    x0: int,
    y0: int,
    out_w: int,
    out_h: int,
    target_magnification: float,
) -> tuple[Image.Image, int, int]:
    objective = infer_objective_power(slide)
    crop_w = level0_size_for_target_magnification(int(out_w), float(target_magnification), objective)
    crop_h = level0_size_for_target_magnification(int(out_h), float(target_magnification), objective)
    rgb = read_region_rgb(slide, int(x0), int(y0), int(crop_w), int(crop_h))
    if rgb.size != (int(out_w), int(out_h)):
        rgb = rgb.resize((int(out_w), int(out_h)), resample=Image.BILINEAR)
    return rgb, int(crop_w), int(crop_h)


def crop_tile_rgb(
    slide: Any,
    *,
    x: int,
    y: int,
    tile_size_20x: int,
    out_tile_size: int,
) -> tuple[Image.Image, int]:
    crop_px = level0_tile_size(tile_size_20x, infer_objective_power(slide))
    rgb = read_region_rgb(slide, int(x), int(y), int(crop_px), int(crop_px))
    if crop_px != int(out_tile_size):
        rgb = rgb.resize((int(out_tile_size), int(out_tile_size)), resample=Image.BILINEAR)
    return rgb, int(crop_px)
