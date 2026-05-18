#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json


DEFAULT_SHOWCASE_ROOT = WSI_CF_ROOT / "artifacts/showcase_regions"
DEFAULT_SELECTOR_DIR = (
    DEFAULT_SHOWCASE_ROOT / "TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_clean_sae_expand_attn_seed_p90"
)
DEFAULT_PROGRESSIVE_RUN_DIR = (
    DEFAULT_SHOWCASE_ROOT
    / "TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_progressive_clean_sae_p90_to_hpvneg_border_relaxed"
    / "case_clean_sae_expand_attn_seed_p90__image_first_tile_selection"
)
DEFAULT_BASE_DIR = DEFAULT_SHOWCASE_ROOT / "TCGA-P3-A5QE-01Z-00-DX1_top_right_2048"
DEFAULT_OUT_DIR = DEFAULT_SHOWCASE_ROOT / "figure4_element_pack"
DEFAULT_PYTHON = "/common/users/wq50/envs/pace/bin/python" if Path("/common/users/wq50/envs/pace/bin/python").exists() else sys.executable


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export atomic figure elements for the showcase progressive edit.")
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR_DIR)
    parser.add_argument("--progressive-run-dir", type=Path, default=DEFAULT_PROGRESSIVE_RUN_DIR)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--naive-run-dir", type=Path, default=DEFAULT_OUT_DIR / "naive_baseline_run")
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-naive-baseline", action="store_true")
    parser.add_argument("--require-naive", action="store_true")
    parser.add_argument("--boundary-crop-size", type=int, default=512)
    parser.add_argument("--boundary-context-crop-size", type=int, default=1024)
    parser.add_argument("--heatmap-scale", type=int, default=96)
    parser.add_argument("--compute-hpv-predictions", action="store_true")
    parser.add_argument("--mil-ckpt", type=Path, default=WSI_CF_ROOT / "resources/models/classifiers/hnscc_hpv/mil_split0.pt")
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(str(key))
                    seen.add(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst)


def image_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def add_manifest(
    elements: list[dict[str, Any]],
    *,
    name: str,
    path: Path,
    source: Path | str,
    kind: str,
    intended_use: str,
    crop_box: tuple[int, int, int, int] | None = None,
) -> None:
    item: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "source": str(source),
        "kind": kind,
        "intended_use": intended_use,
    }
    if crop_box is not None:
        item["crop_box"] = [int(v) for v in crop_box]
    elements.append(item)


def draw_grid(
    img: Image.Image,
    *,
    grid_w: int,
    grid_h: int,
    line_color: tuple[int, int, int, int] = (0, 0, 0, 150),
    line_width: int = 2,
) -> Image.Image:
    out = img.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    for gx in range(1, int(grid_w)):
        x = round(gx * w / int(grid_w))
        draw.line([(x, 0), (x, h)], fill=line_color, width=line_width)
    for gy in range(1, int(grid_h)):
        y = round(gy * h / int(grid_h))
        draw.line([(0, y), (w, y)], fill=line_color, width=line_width)
    return out.convert("RGB")


def draw_cell_mask(
    *,
    size: tuple[int, int],
    grid_w: int,
    grid_h: int,
    cells: list[dict[str, Any]],
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> Image.Image:
    out = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = size
    cw = w / int(grid_w)
    ch = h / int(grid_h)
    for cell in cells:
        gx = int(cell["gx"])
        gy = int(cell["gy"])
        x0 = int(round(gx * cw))
        y0 = int(round(gy * ch))
        x1 = int(round((gx + 1) * cw))
        y1 = int(round((gy + 1) * ch))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill, outline=outline, width=4)
    return out


def overlay_cells(
    img: Image.Image,
    *,
    grid_w: int,
    grid_h: int,
    cells: list[dict[str, Any]],
    fill: tuple[int, int, int, int] = (255, 0, 0, 70),
    outline: tuple[int, int, int, int] = (255, 0, 0, 230),
) -> Image.Image:
    out = img.convert("RGBA")
    mask = draw_cell_mask(size=out.size, grid_w=grid_w, grid_h=grid_h, cells=cells, fill=fill, outline=outline)
    out.alpha_composite(mask)
    return out.convert("RGB")


def colorize_array(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D heatmap array, got {arr.shape}")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        norm = np.zeros(arr.shape, dtype=np.float32)
    else:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        norm = (arr - lo) / max(hi - lo, 1e-8)
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    stops = np.asarray(
        [
            [13, 8, 135],
            [84, 3, 160],
            [139, 10, 165],
            [185, 50, 137],
            [219, 92, 104],
            [244, 136, 73],
            [253, 185, 43],
            [240, 249, 33],
        ],
        dtype=np.float32,
    )
    x = np.clip(norm, 0.0, 1.0) * (len(stops) - 1)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, len(stops) - 1)
    t = (x - i0)[..., None]
    rgb = stops[i0] * (1.0 - t) + stops[i1] * t
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def add_colorbar(img: Image.Image) -> Image.Image:
    h = img.size[1]
    grad = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    bar = colorize_array(grad).resize((28, h), resample=Image.BILINEAR)
    out = Image.new("RGB", (img.size[0] + 36, h), (255, 255, 255))
    out.paste(img, (0, 0))
    out.paste(bar, (img.size[0] + 8, 0))
    return out


def chart_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), int(size))
    return ImageFont.load_default()


def draw_text_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (int(xy[0] - (box[2] - box[0]) / 2), int(xy[1] - (box[3] - box[1]) / 2)),
        text,
        fill=fill,
        font=font,
    )


def draw_grouped_bar_chart(
    *,
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_label: str,
    out_path: Path,
    y_max: float | None = None,
    value_format: str = "{:.2f}",
    size: tuple[int, int] = (1200, 760),
) -> None:
    width, height = size
    margin_left = 120
    margin_right = 60
    margin_top = 90
    margin_bottom = 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_val = max([0.0] + [float(v) for _, vals, _ in series for v in vals])
    ymax = float(y_max) if y_max is not None else max(1e-6, max_val * 1.18)
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = chart_font(28, bold=True)
    axis_font = chart_font(18)
    tick_font = chart_font(16)
    value_font = chart_font(15)
    legend_font = chart_font(17)

    axis_color = (40, 40, 40)
    grid_color = (220, 220, 220)
    text_color = (25, 25, 25)
    draw.text((margin_left, 24), title, fill=text_color, font=title_font)
    draw.text((18, margin_top + plot_h // 2 - 8), y_label, fill=text_color, font=axis_font)
    draw.line([(margin_left, margin_top), (margin_left, margin_top + plot_h)], fill=axis_color, width=2)
    draw.line([(margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h)], fill=axis_color, width=2)

    for tick in range(0, 6):
        value = ymax * tick / 5.0
        y = int(round(margin_top + plot_h - (value / ymax) * plot_h))
        draw.line([(margin_left, y), (margin_left + plot_w, y)], fill=grid_color, width=1)
        label = value_format.format(value)
        draw.text((margin_left - 72, y - 8), label, fill=text_color, font=tick_font)

    n_cat = max(1, len(categories))
    n_series = max(1, len(series))
    group_w = plot_w / n_cat
    bar_w = min(70.0, group_w * 0.72 / n_series)
    for ci, category in enumerate(categories):
        center_x = margin_left + group_w * (ci + 0.5)
        start_x = center_x - (bar_w * n_series) / 2.0
        for si, (_, vals, color) in enumerate(series):
            value = float(vals[ci])
            x0 = int(round(start_x + si * bar_w))
            x1 = int(round(x0 + bar_w * 0.82))
            y1 = margin_top + plot_h
            y0 = int(round(y1 - (value / ymax) * plot_h))
            draw.rectangle([x0, y0, x1, y1], fill=color)
            draw_text_centered(draw, (int((x0 + x1) / 2), y0 - 16), value_format.format(value), fill=text_color, font=value_font)
        draw_text_centered(draw, (int(center_x), margin_top + plot_h + 34), category, fill=text_color, font=axis_font)

    legend_x = margin_left
    legend_y = height - 72
    for name, _, color in series:
        draw.rectangle([legend_x, legend_y, legend_x + 24, legend_y + 16], fill=color)
        draw.text((legend_x + 32, legend_y - 1), name, fill=text_color, font=legend_font)
        legend_x += 230

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def export_chart_elements(
    *,
    charts_dir: Path,
    hpv_csv: Path,
    diff_summary_csv: Path,
    path_plan_summary: Path,
) -> list[dict[str, Any]]:
    chart_items: list[dict[str, Any]] = []
    if hpv_csv.exists():
        rows = read_csv(hpv_csv)
        main_rows = [row for row in rows if str(row["sample"]) in {"source", "progressive"}]
        categories = ["original" if str(row["sample"]) == "source" else "progressive" for row in main_rows]
        hpv_pos = [float(row["prob_hpv_pos"]) for row in main_rows]
        hpv_neg = [float(row["prob_hpv_neg"]) for row in main_rows]
        out_path = charts_dir / "hpv_probability_shift_bar.png"
        draw_grouped_bar_chart(
            title="HPV+ Probability Shift",
            categories=categories,
            series=[("HPV+ probability", hpv_pos, (107, 75, 154))],
            y_label="Probability",
            out_path=out_path,
            y_max=1.0,
        )
        chart_items.append(
            {
                "name": "hpv_probability_shift_bar",
                "path": str(out_path),
                "source": str(hpv_csv),
                "kind": "chart",
                "intended_use": "Primary prediction-shift bar chart: original/source versus our progressive edited image only.",
            }
        )
        out_path = charts_dir / "hpv_class_probability_grouped_bar.png"
        draw_grouped_bar_chart(
            title="HPV Class Probabilities",
            categories=categories,
            series=[("HPV-", hpv_neg, (170, 170, 170)), ("HPV+", hpv_pos, (107, 75, 154))],
            y_label="Probability",
            out_path=out_path,
            y_max=1.0,
        )
        chart_items.append(
            {
                "name": "hpv_class_probability_grouped_bar",
                "path": str(out_path),
                "source": str(hpv_csv),
                "kind": "chart",
                "intended_use": "Grouped bar chart of HPV- and HPV+ classifier probabilities.",
            }
        )

        all_categories = ["original" if str(row["sample"]) == "source" else str(row["sample"]) for row in rows]
        all_hpv_pos = [float(row["prob_hpv_pos"]) for row in rows]
        all_hpv_neg = [float(row["prob_hpv_neg"]) for row in rows]
        out_path = charts_dir / "hpv_class_probability_with_naive_grouped_bar.png"
        draw_grouped_bar_chart(
            title="HPV Class Probabilities With Naive Baseline",
            categories=all_categories,
            series=[("HPV-", all_hpv_neg, (170, 170, 170)), ("HPV+", all_hpv_pos, (107, 75, 154))],
            y_label="Probability",
            out_path=out_path,
            y_max=1.0,
        )
        chart_items.append(
            {
                "name": "hpv_class_probability_with_naive_grouped_bar",
                "path": str(out_path),
                "source": str(hpv_csv),
                "kind": "chart",
                "intended_use": "Supporting classifier-probability chart including the naive baseline.",
            }
        )

    if diff_summary_csv.exists():
        rows = read_csv(diff_summary_csv)
        wanted = [row for row in rows if row["area"] in {"target_cells", "non_target_cells"}]
        comparisons = ["source_vs_progressive", "source_vs_naive", "progressive_vs_naive"]
        categories = ["src-prog", "src-naive", "prog-naive"]
        by_key = {(row["comparison"], row["area"]): float(row["mean_abs_rgb"]) for row in wanted}
        target_vals = [by_key.get((comparison, "target_cells"), 0.0) for comparison in comparisons]
        nontarget_vals = [by_key.get((comparison, "non_target_cells"), 0.0) for comparison in comparisons]
        out_path = charts_dir / "target_vs_nontarget_difference_bar.png"
        draw_grouped_bar_chart(
            title="Mean Absolute RGB Difference",
            categories=categories,
            series=[("target cells", target_vals, (224, 75, 75)), ("non-target cells", nontarget_vals, (80, 130, 190))],
            y_label="Mean abs RGB",
            out_path=out_path,
            value_format="{:.1f}",
        )
        chart_items.append(
            {
                "name": "target_vs_nontarget_difference_bar",
                "path": str(out_path),
                "source": str(diff_summary_csv),
                "kind": "chart",
                "intended_use": "Difference comparison chart for edited target cells versus non-target cells.",
            }
        )

        full_rows = [row for row in rows if row["area"] == "full_region" and row["comparison"] in comparisons]
        full_by_comp = {row["comparison"]: float(row["mean_abs_rgb"]) for row in full_rows}
        out_path = charts_dir / "full_region_difference_bar.png"
        draw_grouped_bar_chart(
            title="Full Region Difference",
            categories=categories,
            series=[("full region", [full_by_comp.get(comparison, 0.0) for comparison in comparisons], (88, 145, 105))],
            y_label="Mean abs RGB",
            out_path=out_path,
            value_format="{:.1f}",
        )
        chart_items.append(
            {
                "name": "full_region_difference_bar",
                "path": str(out_path),
                "source": str(diff_summary_csv),
                "kind": "chart",
                "intended_use": "Mean absolute RGB difference across the full region.",
            }
        )

    if path_plan_summary.exists():
        payload = read_json(path_plan_summary)
        categories = ["minimal", "scanline"]
        windows = [
            float(payload["minimal_greedy"]["window_count"]),
            float(payload["scanline_connected"]["window_count"]),
        ]
        out_path = charts_dir / "path_window_count_bar.png"
        draw_grouped_bar_chart(
            title="Path Window Count",
            categories=categories,
            series=[("windows", windows, (120, 120, 120))],
            y_label="Window count",
            out_path=out_path,
            y_max=max(windows) + 1.0,
            value_format="{:.0f}",
        )
        chart_items.append(
            {
                "name": "path_window_count_bar",
                "path": str(out_path),
                "source": str(path_plan_summary),
                "kind": "chart",
                "intended_use": "Bar chart comparing greedy and connected scanline path lengths.",
            }
        )
    return chart_items


def save_array_heatmap(arr: np.ndarray, out_path: Path, *, scale: int) -> None:
    heat = colorize_array(arr)
    heat = heat.resize((int(heat.size[0] * scale), int(heat.size[1] * scale)), resample=Image.NEAREST)
    heat = add_colorbar(heat)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heat.save(out_path)


def save_diff(source_path: Path, generated_path: Path, out_path: Path) -> None:
    src = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.int16)
    gen = np.asarray(Image.open(generated_path).convert("RGB"), dtype=np.int16)
    diff = np.abs(gen - src).astype(np.float32).mean(axis=2)
    if diff.max() > 0:
        diff = diff / diff.max()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    add_colorbar(colorize_array(diff)).save(out_path)


def draw_global_window_element(
    *,
    source_img: Image.Image,
    grid_w: int,
    grid_h: int,
    step: dict[str, Any],
    visited_cells: set[tuple[int, int]],
    out_path: Path,
) -> set[tuple[int, int]]:
    out = draw_grid(source_img, grid_w=grid_w, grid_h=grid_h).convert("RGBA")
    overlay = Image.new("RGBA", out.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    w, h = out.size
    cw = w / int(grid_w)
    ch = h / int(grid_h)

    for gx, gy in visited_cells:
        x0 = int(round(gx * cw))
        y0 = int(round(gy * ch))
        draw.rectangle([x0, y0, int(round((gx + 1) * cw)) - 1, int(round((gy + 1) * ch)) - 1], fill=(80, 120, 255, 55))

    for cell in step.get("edit_cells_global", []):
        gx = int(cell["gx"])
        gy = int(cell["gy"])
        x0 = int(round(gx * cw))
        y0 = int(round(gy * ch))
        draw.rectangle(
            [x0, y0, int(round((gx + 1) * cw)) - 1, int(round((gy + 1) * ch)) - 1],
            fill=(255, 0, 0, 90),
            outline=(255, 0, 0, 230),
            width=4,
        )

    x0 = int(step["left"])
    y0 = int(step["top"])
    x1 = x0 + int(4 * cw)
    y1 = y0 + int(4 * ch)
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0, 255), width=6)

    out.alpha_composite(overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(out_path)
    new_visited = set(visited_cells)
    for dy in range(4):
        for dx in range(4):
            new_visited.add((int(step["gx0"]) + dx, int(step["gy0"]) + dy))
    return new_visited


def crop_box(center_x: int, center_y: int, size: int, width: int, height: int) -> tuple[int, int, int, int]:
    half = int(size) // 2
    x0 = max(0, min(int(center_x) - half, int(width) - int(size)))
    y0 = max(0, min(int(center_y) - half, int(height) - int(size)))
    return int(x0), int(y0), int(x0 + int(size)), int(y0 + int(size))


def choose_boundary_boxes(steps: list[dict[str, Any]], *, image_size: tuple[int, int], crop_size: int) -> list[dict[str, Any]]:
    width, height = image_size
    boxes: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for step in steps:
        bounds = step.get("commit_bounds_global", {})
        candidates = []
        for x in (bounds.get("x0"), bounds.get("x1")):
            if x not in (None, 0, width):
                candidates.append((int(x), int((bounds.get("y0", 0) + bounds.get("y1", height)) / 2), "vertical"))
        for y in (bounds.get("y0"), bounds.get("y1")):
            if y not in (None, 0, height):
                candidates.append((int((bounds.get("x0", 0) + bounds.get("x1", width)) / 2), int(y), "horizontal"))
        for cx, cy, orientation in candidates:
            key = (round(cx / 128), round(cy / 128))
            if key in seen:
                continue
            seen.add(key)
            boxes.append(
                {
                    "boundary_index": len(boxes) + 1,
                    "source_step_index": int(step.get("step_index", 0)),
                    "orientation": orientation,
                    "center_x": int(cx),
                    "center_y": int(cy),
                    "crop_box": crop_box(cx, cy, int(crop_size), width, height),
                }
            )
            if len(boxes) >= 5:
                return boxes
    return boxes


def save_crop(src_path: Path, dst_path: Path, box: tuple[int, int, int, int]) -> None:
    img = Image.open(src_path).convert("RGB")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.crop(box).save(dst_path)


def paste_step_commit(canvas: Image.Image, step: dict[str, Any]) -> Image.Image:
    steered_path = Path(str(step.get("steered_window_path", "")))
    if not steered_path.exists():
        return canvas
    bounds = step.get("commit_bounds_global", {})
    x0 = int(bounds.get("x0", step.get("left", 0)))
    y0 = int(bounds.get("y0", step.get("top", 0)))
    x1 = int(bounds.get("x1", x0))
    y1 = int(bounds.get("y1", y0))
    left = int(step.get("left", x0))
    top = int(step.get("top", y0))
    if x1 <= x0 or y1 <= y0:
        return canvas
    steered = Image.open(steered_path).convert("RGB")
    local_box = (x0 - left, y0 - top, x1 - left, y1 - top)
    out = canvas.copy()
    out.paste(steered.crop(local_box), (x0, y0))
    return out


def cell_mask_array(
    *,
    size: tuple[int, int],
    grid_w: int,
    grid_h: int,
    cells: list[dict[str, Any]],
) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    cw = width / int(grid_w)
    ch = height / int(grid_h)
    for cell in cells:
        gx = int(cell["gx"])
        gy = int(cell["gy"])
        x0 = int(round(gx * cw))
        y0 = int(round(gy * ch))
        x1 = int(round((gx + 1) * cw))
        y1 = int(round((gy + 1) * ch))
        mask[y0:y1, x0:x1] = True
    return mask


def commit_union_mask(
    *,
    size: tuple[int, int],
    steps: list[dict[str, Any]],
) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    for step in steps:
        bounds = step.get("commit_bounds_global", {})
        x0 = max(0, min(width, int(bounds.get("x0", step.get("left", 0)))))
        y0 = max(0, min(height, int(bounds.get("y0", step.get("top", 0)))))
        x1 = max(0, min(width, int(bounds.get("x1", x0))))
        y1 = max(0, min(height, int(bounds.get("y1", y0))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def summarize_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    if not bool(mask.any()):
        return {
            "pixel_count": 0,
            "mean_abs_rgb": 0.0,
            "median_abs_rgb": 0.0,
            "rms_rgb": 0.0,
            "changed_fraction_gt10": 0.0,
            "changed_fraction_gt25": 0.0,
        }
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    pix = diff[mask]
    mean_per_pixel = pix.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(pix))))
    return {
        "pixel_count": int(pix.shape[0]),
        "mean_abs_rgb": float(mean_per_pixel.mean()),
        "median_abs_rgb": float(np.median(mean_per_pixel)),
        "rms_rgb": rms,
        "changed_fraction_gt10": float(np.mean(mean_per_pixel > 10.0)),
        "changed_fraction_gt25": float(np.mean(mean_per_pixel > 25.0)),
    }


def build_image_difference_tables(
    *,
    source_path: Path,
    progressive_path: Path,
    naive_path: Path | None,
    target_cells: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    grid_w: int,
    grid_h: int,
    out_dir: Path,
) -> tuple[Path, Path]:
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
    progressive = np.asarray(Image.open(progressive_path).convert("RGB"), dtype=np.uint8)
    naive = np.asarray(Image.open(naive_path).convert("RGB"), dtype=np.uint8) if naive_path is not None else None
    height, width = source.shape[:2]
    target_mask = cell_mask_array(size=(width, height), grid_w=grid_w, grid_h=grid_h, cells=target_cells)
    commit_mask = commit_union_mask(size=(width, height), steps=steps)
    full_mask = np.ones((height, width), dtype=bool)
    area_masks = {
        "full_region": full_mask,
        "target_cells": target_mask,
        "non_target_cells": ~target_mask,
        "committed_windows": commit_mask,
        "outside_committed_windows": ~commit_mask,
    }
    comparisons: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("source_vs_progressive", source, progressive),
    ]
    if naive is not None:
        comparisons.extend(
            [
                ("source_vs_naive", source, naive),
                ("progressive_vs_naive", progressive, naive),
            ]
        )
    rows: list[dict[str, Any]] = []
    for comparison, left, right in comparisons:
        for area_name, mask in area_masks.items():
            rows.append({"comparison": comparison, "area": area_name, **summarize_diff(left, right, mask)})

    per_cell_rows: list[dict[str, Any]] = []
    target_set = {(int(cell["gx"]), int(cell["gy"])) for cell in target_cells}
    commit_by_cell: set[tuple[int, int]] = set()
    for step in steps:
        gx0 = int(step.get("gx0", 0))
        gy0 = int(step.get("gy0", 0))
        for dy in range(4):
            for dx in range(4):
                commit_by_cell.add((gx0 + dx, gy0 + dy))
    for gy in range(int(grid_h)):
        for gx in range(int(grid_w)):
            mask = cell_mask_array(size=(width, height), grid_w=grid_w, grid_h=grid_h, cells=[{"gx": gx, "gy": gy}])
            row: dict[str, Any] = {
                "gx": int(gx),
                "gy": int(gy),
                "is_target_cell": bool((gx, gy) in target_set),
                "is_committed_window_cell": bool((gx, gy) in commit_by_cell),
                "source_vs_progressive_mean_abs_rgb": summarize_diff(source, progressive, mask)["mean_abs_rgb"],
            }
            if naive is not None:
                row["source_vs_naive_mean_abs_rgb"] = summarize_diff(source, naive, mask)["mean_abs_rgb"]
                row["progressive_vs_naive_mean_abs_rgb"] = summarize_diff(progressive, naive, mask)["mean_abs_rgb"]
            per_cell_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "image_difference_summary.csv"
    per_cell_csv = out_dir / "per_cell_image_difference.csv"
    write_csv(summary_csv, rows)
    write_csv(per_cell_csv, per_cell_rows)
    return summary_csv, per_cell_csv


def target_cells_from_manifest(run_manifest: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(cell["gx"]), int(cell["gy"])) for cell in run_manifest.get("target_cells", [])]


def supported_targets_for_window(
    *,
    window: dict[str, int | str],
    targets: set[tuple[int, int]],
    grid_w: int,
    grid_h: int,
    edit_support: str,
) -> set[tuple[int, int]]:
    gx0 = int(window["gx0"])
    gy0 = int(window["gy0"])
    allowed = {
        (gx0 + 1, gy0 + 1),
        (gx0 + 2, gy0 + 1),
        (gx0 + 1, gy0 + 2),
        (gx0 + 2, gy0 + 2),
    }
    if str(edit_support) == "border_relaxed":
        for dy in range(4):
            for dx in range(4):
                gx = gx0 + dx
                gy = gy0 + dy
                if gx in {0, int(grid_w) - 1} or gy in {0, int(grid_h) - 1}:
                    allowed.add((gx, gy))
    return {cell for cell in allowed if cell in targets}


def build_scanline_window_plan(run_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    grid_h, grid_w = [int(v) for v in run_manifest.get("grid_shape", [8, 8])]
    grid_step_px = int(run_manifest.get("grid_step_px", 256))
    targets = set(target_cells_from_manifest(run_manifest))
    remaining = set(targets)
    windows: list[dict[str, Any]] = []
    starts_x = list(range(0, grid_w - 4 + 1, 2))
    starts_y = list(range(0, grid_h - 4 + 1, 2))
    if starts_x[-1] != grid_w - 4:
        starts_x.append(grid_w - 4)
    if starts_y[-1] != grid_h - 4:
        starts_y.append(grid_h - 4)
    for row_index, gy0 in enumerate(starts_y):
        row_xs = starts_x if row_index % 2 == 0 else list(reversed(starts_x))
        for gx0 in row_xs:
            col_index = starts_x.index(gx0)
            windows.append(
                {
                    "window_id": f"r{row_index}_c{col_index}",
                    "row_index": int(row_index),
                    "col_index": int(col_index),
                    "gx0": int(gx0),
                    "gy0": int(gy0),
                    "left": int(gx0 * grid_step_px),
                    "top": int(gy0 * grid_step_px),
                }
            )

    steps: list[dict[str, Any]] = []
    edit_support = str(run_manifest.get("edit_support", "border_relaxed"))
    for step_index, window in enumerate(windows):
        covered = supported_targets_for_window(
            window=window,
            targets=targets,
            grid_w=grid_w,
            grid_h=grid_h,
            edit_support=edit_support,
        )
        edit_cells = sorted(covered.intersection(remaining), key=lambda item: (item[1], item[0]))
        remaining.difference_update(edit_cells)
        steps.append(
            {
                **window,
                "step_index": int(step_index),
                "edit_cells_global": [{"gx": int(gx), "gy": int(gy)} for gx, gy in edit_cells],
                "n_edit_cells": int(len(edit_cells)),
                "is_bridge_window": bool(len(edit_cells) == 0),
            }
        )
    if remaining:
        raise RuntimeError(f"Scanline plan failed to cover targets: {sorted(remaining, key=lambda item: (item[1], item[0]))}")
    return steps


def draw_path_plan_overview(
    *,
    source_img: Image.Image,
    grid_w: int,
    grid_h: int,
    steps: list[dict[str, Any]],
    out_path: Path,
) -> None:
    out = draw_grid(source_img, grid_w=grid_w, grid_h=grid_h).convert("RGBA")
    overlay = Image.new("RGBA", out.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = out.size
    cw = width / int(grid_w)
    ch = height / int(grid_h)
    centers: list[tuple[int, int]] = []
    n = max(1, len(steps) - 1)
    for idx, step in enumerate(steps):
        x0 = int(step["left"])
        y0 = int(step["top"])
        x1 = x0 + int(round(4 * cw))
        y1 = y0 + int(round(4 * ch))
        alpha = int(45 + 70 * idx / n)
        color = (255, 0, 0, alpha) if int(step.get("n_edit_cells", 1)) > 0 else (80, 120, 255, 45)
        outline = (255, 0, 0, 230) if int(step.get("n_edit_cells", 1)) > 0 else (80, 120, 255, 210)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color, outline=outline, width=5)
        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)
        centers.append((cx, cy))
        label_box = [cx - 24, cy - 24, cx + 24, cy + 24]
        draw.ellipse(label_box, fill=(255, 255, 255, 220), outline=outline, width=3)
        draw.text((cx - 10, cy - 9), str(idx + 1), fill=(0, 0, 0, 255))
    for p0, p1 in zip(centers[:-1], centers[1:]):
        draw.line([p0, p1], fill=(0, 0, 0, 210), width=5)
    out.alpha_composite(overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(out_path)


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: int = 18,
    gap: int = 10,
) -> None:
    x0, y0, x1, y1 = [int(v) for v in box]
    for x in range(x0, x1, dash + gap):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=fill, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=fill, width=width)
    for y in range(y0, y1, dash + gap):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=fill, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=fill, width=width)


def draw_demo_path_step(
    *,
    source_img: Image.Image,
    grid_w: int,
    grid_h: int,
    window: dict[str, int],
    visited_windows: list[dict[str, int]],
    out_size: int = 420,
) -> Image.Image:
    base = source_img.resize((out_size, out_size), resample=Image.BILINEAR).convert("RGBA")
    draw_grid_img = draw_grid(base.convert("RGB"), grid_w=grid_w, grid_h=grid_h, line_color=(60, 60, 60, 130), line_width=1).convert("RGBA")
    overlay = Image.new("RGBA", draw_grid_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    cw = out_size / int(grid_w)
    ch = out_size / int(grid_h)

    for visited in visited_windows:
        vx0 = int(round(int(visited["gx0"]) * cw))
        vy0 = int(round(int(visited["gy0"]) * ch))
        vx1 = int(round((int(visited["gx0"]) + 4) * cw))
        vy1 = int(round((int(visited["gy0"]) + 4) * ch))
        draw.rectangle([vx0, vy0, vx1 - 1, vy1 - 1], fill=(97, 135, 240, 75))

    gx0 = int(window["gx0"])
    gy0 = int(window["gy0"])
    x0 = int(round(gx0 * cw))
    y0 = int(round(gy0 * ch))
    x1 = int(round((gx0 + 4) * cw))
    y1 = int(round((gy0 + 4) * ch))
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(235, 235, 235, 85))

    for ly in (1, 2):
        for lx in (1, 2):
            cx0 = int(round((gx0 + lx) * cw))
            cy0 = int(round((gy0 + ly) * ch))
            cx1 = int(round((gx0 + lx + 1) * cw))
            cy1 = int(round((gy0 + ly + 1) * ch))
            draw.rectangle([cx0, cy0, cx1 - 1, cy1 - 1], fill=(255, 42, 42, 150), outline=(120, 0, 0, 240), width=3)

    draw_dashed_rectangle(draw, (x0, y0, x1 - 1, y1 - 1), fill=(255, 0, 0, 240), width=3)
    draw_grid_img.alpha_composite(overlay)
    return draw_grid_img.convert("RGB")


def export_demo_path_diagrams(
    *,
    source_img: Image.Image,
    grid_w: int,
    grid_h: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    demo_specs: list[tuple[str, str, list[dict[str, int]]]] = [
        (
            "demo_column_top_to_bottom",
            "Showcase-only column-major path: left column top-to-bottom, then the next column from the top.",
            [
                {"gx0": 0, "gy0": 0},
                {"gx0": 0, "gy0": 2},
                {"gx0": 0, "gy0": 4},
                {"gx0": 2, "gy0": 0},
            ],
        ),
        (
            "demo_diagonal",
            "Showcase-only four-step diagonal path with illustrative one-cell diagonal motion.",
            [
                {"gx0": 0, "gy0": 0},
                {"gx0": 1, "gy0": 1},
                {"gx0": 2, "gy0": 2},
                {"gx0": 3, "gy0": 3},
            ],
        ),
    ]
    elements: list[dict[str, Any]] = []
    data_files: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_w = 420
    panel_h = 480
    arrow_w = 70
    legend_h = 92
    title_font = chart_font(24, bold=True)
    label_font = chart_font(19, bold=True)
    legend_font = chart_font(17)
    for name, description, windows in demo_specs:
        rows: list[dict[str, Any]] = []
        step_images: list[Image.Image] = []
        visited: list[dict[str, int]] = []
        for idx, window in enumerate(windows, start=1):
            step = draw_demo_path_step(
                source_img=source_img,
                grid_w=grid_w,
                grid_h=grid_h,
                window=window,
                visited_windows=visited,
            )
            step_path = out_dir / f"{name}_step_{idx:02d}.png"
            step.save(step_path)
            elements.append(
                {
                    "name": f"{name}_step_{idx:02d}",
                    "path": str(step_path),
                    "source": "showcase_demo_path",
                    "kind": "planning_demo",
                    "intended_use": description,
                }
            )
            step_images.append(step)
            rows.append(
                {
                    "demo_path": name,
                    "step_index": idx - 1,
                    "display_step": idx,
                    "gx0": int(window["gx0"]),
                    "gy0": int(window["gy0"]),
                    "editable_center_cells": ";".join(
                        f"{int(window['gx0']) + lx},{int(window['gy0']) + ly}"
                        for ly in (1, 2)
                        for lx in (1, 2)
                    ),
                    "note": description,
                }
            )
            visited.append(dict(window))

        canvas_w = len(step_images) * panel_w + (len(step_images) - 1) * arrow_w
        canvas_h = panel_h + legend_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for idx, step_img in enumerate(step_images, start=1):
            draw.text((x + 20, 18), f"Step {idx}", fill=(20, 20, 20), font=label_font)
            canvas.paste(step_img, (x, 56))
            x += panel_w
            if idx < len(step_images):
                y = 56 + panel_w // 2
                draw.line([(x + 12, y), (x + arrow_w - 20, y)], fill=(0, 0, 0), width=4)
                draw.polygon([(x + arrow_w - 20, y - 10), (x + arrow_w - 20, y + 10), (x + arrow_w - 5, y)], fill=(0, 0, 0))
                x += arrow_w

        legend_y = panel_h + 22
        legend_items = [
            ("Editable center (2 x 2)", (255, 42, 42), "solid"),
            ("Visited / preserved", (97, 135, 240), "solid"),
            ("Fresh context", (235, 235, 235), "solid"),
            ("Current window (4 x 4)", (255, 0, 0), "dashed"),
        ]
        lx = 22
        for text, color, style in legend_items:
            if style == "dashed":
                draw_dashed_rectangle(draw, (lx, legend_y, lx + 36, legend_y + 24), fill=(*color, 255), width=3, dash=8, gap=5)
            else:
                draw.rectangle([lx, legend_y, lx + 36, legend_y + 24], fill=color, outline=(90, 90, 90))
            draw.text((lx + 48, legend_y + 1), text, fill=(20, 20, 20), font=legend_font)
            lx += 360

        combined_path = out_dir / f"{name}_4step_diagram.png"
        canvas.save(combined_path)
        elements.append(
            {
                "name": f"{name}_4step_diagram",
                "path": str(combined_path),
                "source": "showcase_demo_path",
                "kind": "planning_demo",
                "intended_use": description,
            }
        )
        csv_path = out_dir / f"{name}_4step_plan.csv"
        write_csv(csv_path, rows)
        data_files.append({"name": f"{name}_4step_plan", "path": str(csv_path), "source": "showcase_demo_path"})
    return elements, data_files


def write_path_plan_csv(path: Path, steps: list[dict[str, Any]], *, plan_name: str) -> None:
    rows: list[dict[str, Any]] = []
    for step in steps:
        rows.append(
            {
                "plan_name": plan_name,
                "step_index": int(step.get("step_index", 0)),
                "display_step": int(step.get("step_index", 0)) + 1,
                "window_id": str(step.get("window_id", "")),
                "gx0": int(step.get("gx0", 0)),
                "gy0": int(step.get("gy0", 0)),
                "left": int(step.get("left", 0)),
                "top": int(step.get("top", 0)),
                "n_edit_cells": int(len(step.get("edit_cells_global", []))),
                "is_bridge_window": bool(step.get("is_bridge_window", False)),
                "edit_cells_global": ";".join(f"{c['gx']},{c['gy']}" for c in step.get("edit_cells_global", [])),
            }
        )
    write_csv(path, rows)


def compute_hpv_predictions(
    *,
    images: list[tuple[str, Path]],
    grid_step_px: int,
    mil_ckpt: Path,
    device_name: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    import torch

    from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, run_mil_attention
    from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2

    device = torch.device("cuda" if str(device_name) == "auto" and torch.cuda.is_available() else str(device_name))
    uni_model, uni_transform = load_uni2(device)
    mil_model = build_mil_from_checkpoint(mil_ckpt, device=device)
    grid_dir = out_dir / "encoded_uni2_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, image_path in images:
        grid_path = grid_dir / f"{name}_uni2_grid.npy"
        if grid_path.exists():
            z_grid = np.asarray(np.load(grid_path), dtype=np.float32)
        else:
            z_t = build_uni_grid_from_image(
                Image.open(image_path).convert("RGB"),
                uni_model=uni_model,
                uni_transform=uni_transform,
                grid_step_px=int(grid_step_px),
                device=device,
                out_dtype=torch.float32,
            )
            z_grid = z_t.detach().cpu().numpy().astype(np.float32)
            np.save(grid_path, z_grid)
        bag = z_grid.reshape(-1, z_grid.shape[-1]).astype(np.float32, copy=False)
        _, pred, prob_pos = run_mil_attention(mil_model, bag, device=device)
        rows.append(
            {
                "sample": name,
                "image_path": str(image_path),
                "uni2_grid_path": str(grid_path),
                "classifier": str(mil_ckpt),
                "prediction_method": "image_reencoded_uni2_region_mil",
                "pred_label_id": int(pred),
                "pred_label": "HPV+" if int(pred) == 1 else "HPV-",
                "prob_hpv_pos": float(prob_pos),
                "prob_hpv_neg": float(1.0 - prob_pos),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hpv_predictions.csv"
    json_path = out_dir / "hpv_prediction_summary.json"
    source_row = next((row for row in rows if row["sample"] == "source"), None)
    progressive_row = next((row for row in rows if row["sample"] == "progressive"), None)
    if source_row is not None and progressive_row is not None:
        shift_csv = out_dir / "hpv_prediction_shift_source_progressive.csv"
        write_csv(
            shift_csv,
            [
                {
                    "comparison": "source_to_progressive",
                    "source_prob_hpv_pos": float(source_row["prob_hpv_pos"]),
                    "progressive_prob_hpv_pos": float(progressive_row["prob_hpv_pos"]),
                    "delta_prob_hpv_pos": float(progressive_row["prob_hpv_pos"]) - float(source_row["prob_hpv_pos"]),
                    "source_prob_hpv_neg": float(source_row["prob_hpv_neg"]),
                    "progressive_prob_hpv_neg": float(progressive_row["prob_hpv_neg"]),
                    "delta_prob_hpv_neg": float(progressive_row["prob_hpv_neg"]) - float(source_row["prob_hpv_neg"]),
                    "source_pred_label": source_row["pred_label"],
                    "progressive_pred_label": progressive_row["pred_label"],
                    "source_uni2_grid_path": source_row["uni2_grid_path"],
                    "progressive_uni2_grid_path": progressive_row["uni2_grid_path"],
                    "prediction_method": "image_reencoded_uni2_region_mil",
                }
            ],
        )
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "prediction_method": "image_reencoded_uni2_region_mil",
            "classifier": str(mil_ckpt),
            "grid_step_px": int(grid_step_px),
            "primary_shift_comparison": "source_to_progressive",
            "rows": rows,
        },
    )
    return csv_path, json_path


def build_naive_command(args: argparse.Namespace, run_manifest: dict[str, Any]) -> list[str]:
    cli = dict(run_manifest.get("cli_args", {}))
    region_bank = cli.get("region_bank_csv")
    edit_manifest = cli.get("edit_manifest")
    if not region_bank or not edit_manifest:
        raise ValueError("Cannot build naive command because run_manifest cli_args lacks region_bank_csv/edit_manifest")
    return [
        str(args.python),
        str(WSI_CF_ROOT / "scripts/run_progressive_region_edit.py"),
        "--region-bank-csv",
        str(region_bank),
        "--edit-manifest",
        str(edit_manifest),
        "--out-dir",
        str(args.naive_run_dir),
        "--direction",
        str(run_manifest.get("prototype_direction", "hpv_neg")),
        "--edit-support",
        str(run_manifest.get("edit_support", "border_relaxed")),
        "--seed",
        str(run_manifest.get("seed", 7)),
        "--device",
        str(args.device),
        "--dtype",
        str(cli.get("dtype", "fp16")),
        "--steps",
        str(run_manifest.get("steps", 30)),
        "--guidance",
        str(run_manifest.get("guidance", 2.0)),
        "--patch-batch",
        str(cli.get("patch_batch", 128)),
        "--prototype-strength",
        str(run_manifest.get("prototype_strength", 0.8)),
        "--steer-blend",
        str(run_manifest.get("steer_blend", 1.0)),
        "--preserve-edit-strength",
        "0.0",
        "--preserve-visited-strength",
        "0.0",
        "--preserve-fresh-context-strength",
        "0.0",
        "--mid-steer-start-ratio",
        str(run_manifest.get("mid_steer_start_ratio", 0.5)),
        "--mid-steer-end-ratio",
        str(run_manifest.get("mid_steer_end_ratio", 1.0)),
        "--mid-steer-alpha-start",
        str(run_manifest.get("mid_steer_alpha_start", 0.5)),
        "--mid-steer-alpha-end",
        str(run_manifest.get("mid_steer_alpha_end", 1.0)),
        "--mid-steer-alpha-schedule",
        str(run_manifest.get("mid_steer_alpha_schedule", "linear")),
        "--output-mode",
        "debug",
    ]


def find_naive_generated(naive_run_dir: Path, run_id: str) -> Path | None:
    candidates = [
        naive_run_dir / run_id / "generated.png",
        naive_run_dir / "case_clean_sae_expand_attn_seed_p90__image_first_tile_selection" / "generated.png",
    ]
    for path in candidates:
        if image_exists(path):
            return path
    hits = sorted(naive_run_dir.glob("*/generated.png"))
    return hits[0] if hits else None


def verify_required_inputs(paths: list[Path]) -> list[str]:
    missing = [str(path) for path in paths if not path.exists()]
    return missing


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_manifest_path = args.progressive_run_dir / "run_manifest.json"
    source_path = args.progressive_run_dir / "source_region_actual.png"
    generated_path = args.progressive_run_dir / "generated.png"
    generated_overlay_path = args.progressive_run_dir / "generated_targets_overlay.png"
    source_overlay_path = args.progressive_run_dir / "source_targets_overlay.png"
    whole_slide_path = args.base_dir / "whole_slide_boxed_showcase_region.png"
    selector_manifest_path = args.selector_dir / "progressive_edit_manifest.json"

    required = [
        run_manifest_path,
        source_path,
        generated_path,
        generated_overlay_path,
        source_overlay_path,
        args.selector_dir / "attention_map.npy",
        args.selector_dir / "combined_importance_map.npy",
        args.selector_dir / "region.png",
        selector_manifest_path,
    ]
    if args.require_naive:
        required.append(args.naive_run_dir)
    missing = verify_required_inputs(required)
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    run_manifest = read_json(run_manifest_path)
    naive_cmd = build_naive_command(args, run_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "source": args.out_dir / "source",
        "final": args.out_dir / "final",
        "maps": args.out_dir / "maps",
        "masks": args.out_dir / "masks",
        "steps_global_source": args.out_dir / "steps" / "global_source",
        "steps_global_progressive": args.out_dir / "steps" / "global_progressive_state",
        "steps_source_windows": args.out_dir / "steps" / "source_windows",
        "steps_selected": args.out_dir / "steps" / "selected_cells_overlay",
        "steps_preserve": args.out_dir / "steps" / "preserve_maps",
        "steps_steered": args.out_dir / "steps" / "steered_windows",
        "boundaries_tight": args.out_dir / "boundaries" / "tight",
        "boundaries_context": args.out_dir / "boundaries" / "context",
        "planning": args.out_dir / "planning",
        "planning_demo": args.out_dir / "planning" / "demo_4step_paths",
        "charts": args.out_dir / "charts",
        "comparison": args.out_dir / "comparison",
        "data": args.out_dir / "data",
    }
    for folder in out.values():
        folder.mkdir(parents=True, exist_ok=True)
    naive_script = args.out_dir / "run_naive_baseline.sh"
    naive_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(WSI_CF_ROOT)) + "\n" + " ".join(shlex.quote(part) for part in naive_cmd) + "\n")
    naive_script.chmod(0o755)

    if args.run_naive_baseline:
        subprocess.run(naive_cmd, check=True, cwd=str(WSI_CF_ROOT))

    if args.dry_run:
        print(json.dumps({"missing": missing, "naive_command": " ".join(shlex.quote(part) for part in naive_cmd)}, indent=2))
        return

    elements: list[dict[str, Any]] = []
    data_files: list[dict[str, Any]] = []
    source_img = Image.open(source_path).convert("RGB")
    width, height = source_img.size
    grid_h, grid_w = [int(v) for v in run_manifest.get("grid_shape", [8, 8])]

    source_raw = out["source"] / "source_raw.png"
    copy_image(source_path, source_raw)
    add_manifest(elements, name="source_raw", path=source_raw, source=source_path, kind="image", intended_use="Raw source region.")

    source_grid = out["source"] / "source_grid_8x8.png"
    draw_grid(source_img, grid_w=grid_w, grid_h=grid_h).save(source_grid)
    add_manifest(elements, name="source_grid_8x8", path=source_grid, source=source_path, kind="overlay", intended_use="Source region with 8x8 UNI grid.")

    source_targets = out["source"] / "source_targets_overlay.png"
    copy_image(source_overlay_path, source_targets)
    add_manifest(elements, name="source_targets_overlay", path=source_targets, source=source_overlay_path, kind="overlay", intended_use="Source region with selected target cells.")

    counterfactual_raw = out["final"] / "counterfactual_raw.png"
    copy_image(generated_path, counterfactual_raw)
    add_manifest(elements, name="counterfactual_raw", path=counterfactual_raw, source=generated_path, kind="image", intended_use="Final progressive counterfactual.")

    counterfactual_targets = out["final"] / "counterfactual_targets_overlay.png"
    copy_image(generated_overlay_path, counterfactual_targets)
    add_manifest(elements, name="counterfactual_targets_overlay", path=counterfactual_targets, source=generated_overlay_path, kind="overlay", intended_use="Final counterfactual with target-cell overlay.")

    diff_path = out["final"] / "diff_rgb_abs.png"
    save_diff(source_path, generated_path, diff_path)
    add_manifest(elements, name="diff_rgb_abs", path=diff_path, source=f"{source_path};{generated_path}", kind="heatmap", intended_use="Absolute RGB difference between source and progressive counterfactual.")

    map_specs = [
        ("map_attention", args.selector_dir / "attention_map.npy", "Attention score heatmap."),
        ("map_sae_similarity", args.selector_dir / "sae_match_map_all_tiles.npy", "SAE similarity score heatmap."),
        ("map_combined_importance", args.selector_dir / "combined_importance_map.npy", "Combined attention/SAE importance heatmap."),
    ]
    if not map_specs[1][1].exists():
        map_specs[1] = ("map_sae_similarity", args.selector_dir / "sae_match_map.npy", "SAE similarity score heatmap.")
    for name, npy_path, use in map_specs:
        out_path = out["maps"] / f"{name}.png"
        save_array_heatmap(np.load(npy_path), out_path, scale=int(args.heatmap_scale))
        add_manifest(elements, name=name, path=out_path, source=npy_path, kind="heatmap", intended_use=use)

    selector_items = read_json(selector_manifest_path)
    selector_item = selector_items[0] if selector_items else {}
    seed_cells = list(selector_item.get("seed_target_cells", []))
    final_cells = list(run_manifest.get("target_cells", selector_item.get("target_cells", [])))
    seed_mask = out["masks"] / "mask_seed_cells.png"
    draw_cell_mask(size=(width, height), grid_w=grid_w, grid_h=grid_h, cells=seed_cells, fill=(255, 200, 0, 120), outline=(255, 150, 0, 255)).save(seed_mask)
    add_manifest(elements, name="mask_seed_cells", path=seed_mask, source=selector_manifest_path, kind="mask", intended_use="Transparent mask of seed-selected cells.")
    final_mask = out["masks"] / "mask_final_targets.png"
    draw_cell_mask(size=(width, height), grid_w=grid_w, grid_h=grid_h, cells=final_cells, fill=(255, 0, 0, 100), outline=(255, 0, 0, 255)).save(final_mask)
    add_manifest(elements, name="mask_final_targets", path=final_mask, source=run_manifest_path, kind="mask", intended_use="Transparent mask of final edited target cells.")

    whole_slide_out = out["source"] / "whole_slide_showcase_box.png"
    if whole_slide_path.exists():
        copy_image(whole_slide_path, whole_slide_out)
        add_manifest(elements, name="whole_slide_showcase_box", path=whole_slide_out, source=whole_slide_path, kind="overview", intended_use="Whole-slide overview with showcase region box.")

    steps = list(run_manifest.get("window_history", []))
    visited: set[tuple[int, int]] = set()
    progressive_canvas = source_img.copy()
    step_rows: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        prefix = f"step_{idx:02d}"
        visited_before = set(visited)
        global_path = out["steps_global_source"] / f"{prefix}_global_window.png"
        draw_global_window_element(source_img=source_img, grid_w=grid_w, grid_h=grid_h, step=step, visited_cells=visited_before, out_path=global_path)
        add_manifest(elements, name=f"{prefix}_global_window", path=global_path, source=run_manifest_path, kind="overlay", intended_use="Source-based global grid with current window, edited cells, and visited context.")

        progressive_canvas = paste_step_commit(progressive_canvas, step)
        progressive_global_path = out["steps_global_progressive"] / f"{prefix}_global_window_progressive_state.png"
        visited = draw_global_window_element(
            source_img=progressive_canvas,
            grid_w=grid_w,
            grid_h=grid_h,
            step=step,
            visited_cells=visited_before,
            out_path=progressive_global_path,
        )
        add_manifest(
            elements,
            name=f"{prefix}_global_window_progressive_state",
            path=progressive_global_path,
            source=step.get("steered_window_path", run_manifest_path),
            kind="overlay",
            intended_use="Global grid drawn on the canvas after this progressive step has been committed.",
        )
        for key, suffix, use in [
            ("source_window_path", "source_window", "Source crop for this progressive step."),
            ("selected_overlay_path", "selected_cells_overlay", "Selected cells overlay for this progressive step."),
            ("preserve_map_path", "preserve_map", "History-aware preservation strength map."),
            ("steered_window_path", "steered_window", "Generated local window for this step."),
        ]:
            src = Path(str(step.get(key, "")))
            if src.exists():
                step_out_dir = {
                    "source_window": out["steps_source_windows"],
                    "selected_cells_overlay": out["steps_selected"],
                    "preserve_map": out["steps_preserve"],
                    "steered_window": out["steps_steered"],
                }[suffix]
                dst = step_out_dir / f"{prefix}_{suffix}.png"
                copy_image(src, dst)
                add_manifest(elements, name=f"{prefix}_{suffix}", path=dst, source=src, kind="step", intended_use=use)
        step_rows.append(
            {
                "step_index": int(step.get("step_index", idx - 1)),
                "window_id": str(step.get("window_id", "")),
                "gx0": int(step.get("gx0", 0)),
                "gy0": int(step.get("gy0", 0)),
                "left": int(step.get("left", 0)),
                "top": int(step.get("top", 0)),
                "edit_cells_global": ";".join(f"{c['gx']},{c['gy']}" for c in step.get("edit_cells_global", [])),
                "edit_cells_local": ";".join(f"{c['gx']},{c['gy']}" for c in step.get("edit_cells_local", [])),
                "commit_bounds_global": json.dumps(step.get("commit_bounds_global", {}), sort_keys=True),
            }
        )

    minimal_plan_steps = [
        {
            **step,
            "n_edit_cells": int(len(step.get("edit_cells_global", []))),
            "is_bridge_window": False,
        }
        for step in steps
    ]
    scanline_plan_steps = build_scanline_window_plan(run_manifest)
    minimal_path_png = out["planning"] / "minimal_greedy_path_overview.png"
    draw_path_plan_overview(source_img=source_img, grid_w=grid_w, grid_h=grid_h, steps=minimal_plan_steps, out_path=minimal_path_png)
    add_manifest(
        elements,
        name="minimal_greedy_path_overview",
        path=minimal_path_png,
        source=run_manifest_path,
        kind="planning",
        intended_use="Current greedy/fewest-window progressive edit order overview.",
    )
    scanline_path_png = out["planning"] / "scanline_connected_path_overview.png"
    draw_path_plan_overview(source_img=source_img, grid_w=grid_w, grid_h=grid_h, steps=scanline_plan_steps, out_path=scanline_path_png)
    add_manifest(
        elements,
        name="scanline_connected_path_overview",
        path=scanline_path_png,
        source=run_manifest_path,
        kind="planning",
        intended_use="Alternative connected row-major path: left to right, then top to bottom.",
    )
    scanline_visited: set[tuple[int, int]] = set()
    for idx, step in enumerate(scanline_plan_steps, start=1):
        scanline_step_path = out["planning"] / f"scanline_step_{idx:02d}_global_window.png"
        scanline_visited = draw_global_window_element(
            source_img=source_img,
            grid_w=grid_w,
            grid_h=grid_h,
            step=step,
            visited_cells=scanline_visited,
            out_path=scanline_step_path,
        )
        add_manifest(
            elements,
            name=f"scanline_step_{idx:02d}_global_window",
            path=scanline_step_path,
            source=run_manifest_path,
            kind="planning",
            intended_use="Alternative row-major connected path step on global grid.",
        )

    demo_elements, demo_data_files = export_demo_path_diagrams(
        source_img=source_img,
        grid_w=grid_w,
        grid_h=grid_h,
        out_dir=out["planning_demo"],
    )
    elements.extend(demo_elements)
    data_files.extend(demo_data_files)

    minimal_plan_csv = out["data"] / "minimal_greedy_path_plan.csv"
    scanline_plan_csv = out["data"] / "scanline_connected_path_plan.csv"
    write_path_plan_csv(minimal_plan_csv, minimal_plan_steps, plan_name="minimal_greedy")
    write_path_plan_csv(scanline_plan_csv, scanline_plan_steps, plan_name="scanline_connected")
    data_files.append({"name": "minimal_greedy_path_plan", "path": str(minimal_plan_csv), "source": str(run_manifest_path)})
    data_files.append({"name": "scanline_connected_path_plan", "path": str(scanline_plan_csv), "source": str(run_manifest_path)})
    path_plan_summary = out["data"] / "path_plan_summary.json"
    write_json(
        path_plan_summary,
        {
            "minimal_greedy": {
                "window_count": int(len(minimal_plan_steps)),
                "bridge_window_count": 0,
                "edited_cell_count": int(sum(len(step.get("edit_cells_global", [])) for step in minimal_plan_steps)),
                "window_ids": [str(step.get("window_id", "")) for step in minimal_plan_steps],
            },
            "scanline_connected": {
                "window_count": int(len(scanline_plan_steps)),
                "bridge_window_count": int(sum(1 for step in scanline_plan_steps if bool(step.get("is_bridge_window", False)))),
                "edited_cell_count": int(sum(len(step.get("edit_cells_global", [])) for step in scanline_plan_steps)),
                "window_ids": [str(step.get("window_id", "")) for step in scanline_plan_steps],
            },
            "note": "Scanline connected path visits every 4x4 window in a snake order: left-to-right on the first row, right-to-left on the next, then left-to-right again. This keeps neighboring windows spatially connected while moving top-to-bottom; bridge windows may have zero newly edited target cells.",
        },
    )
    data_files.append({"name": "path_plan_summary", "path": str(path_plan_summary), "source": str(run_manifest_path)})

    boundary_rows = choose_boundary_boxes(steps, image_size=(width, height), crop_size=int(args.boundary_crop_size))
    naive_generated = find_naive_generated(args.naive_run_dir, str(run_manifest.get("run_id", "")))
    for row in boundary_rows:
        idx = int(row["boundary_index"])
        box = tuple(int(v) for v in row["crop_box"])
        context_box = crop_box(int(row["center_x"]), int(row["center_y"]), int(args.boundary_context_crop_size), width, height)
        row["context_crop_box"] = context_box
        prog_path = out["boundaries_tight"] / f"boundary_{idx:02d}_progressive.png"
        save_crop(generated_path, prog_path, box)
        add_manifest(elements, name=f"boundary_{idx:02d}_progressive", path=prog_path, source=generated_path, kind="crop", intended_use="Progressive boundary zoom crop.", crop_box=box)
        src_crop = out["boundaries_tight"] / f"boundary_{idx:02d}_source.png"
        save_crop(source_path, src_crop, box)
        add_manifest(elements, name=f"boundary_{idx:02d}_source", path=src_crop, source=source_path, kind="crop", intended_use="Source boundary zoom crop.", crop_box=box)
        prog_context_path = out["boundaries_context"] / f"boundary_{idx:02d}_context_progressive.png"
        save_crop(generated_path, prog_context_path, context_box)
        add_manifest(elements, name=f"boundary_{idx:02d}_context_progressive", path=prog_context_path, source=generated_path, kind="crop", intended_use="Wider context crop around progressive boundary zoom location.", crop_box=context_box)
        src_context_path = out["boundaries_context"] / f"boundary_{idx:02d}_context_source.png"
        save_crop(source_path, src_context_path, context_box)
        add_manifest(elements, name=f"boundary_{idx:02d}_context_source", path=src_context_path, source=source_path, kind="crop", intended_use="Wider context crop around source boundary zoom location.", crop_box=context_box)
        if naive_generated is not None:
            naive_path = out["boundaries_tight"] / f"boundary_{idx:02d}_naive.png"
            save_crop(naive_generated, naive_path, box)
            add_manifest(elements, name=f"boundary_{idx:02d}_naive", path=naive_path, source=naive_generated, kind="crop", intended_use="Naive baseline boundary zoom crop.", crop_box=box)
            naive_context_path = out["boundaries_context"] / f"boundary_{idx:02d}_context_naive.png"
            save_crop(naive_generated, naive_context_path, context_box)
            add_manifest(elements, name=f"boundary_{idx:02d}_context_naive", path=naive_context_path, source=naive_generated, kind="crop", intended_use="Wider context crop around naive boundary zoom location.", crop_box=context_box)

    progressive_csv = out["data"] / "progressive_steps.csv"
    write_csv(progressive_csv, step_rows)
    data_files.append({"name": "progressive_steps", "path": str(progressive_csv), "source": str(run_manifest_path)})
    boundary_csv = out["data"] / "boundary_crops.csv"
    write_csv(
        boundary_csv,
        [
            {
                **{k: v for k, v in row.items() if k not in {"crop_box", "context_crop_box"}},
                "crop_box": json.dumps(row["crop_box"]),
                "context_crop_box": json.dumps(row["context_crop_box"]),
            }
            for row in boundary_rows
        ],
    )
    data_files.append({"name": "boundary_crops", "path": str(boundary_csv), "source": str(run_manifest_path)})

    diff_summary_csv, per_cell_diff_csv = build_image_difference_tables(
        source_path=source_path,
        progressive_path=generated_path,
        naive_path=naive_generated,
        target_cells=list(run_manifest.get("target_cells", [])),
        steps=steps,
        grid_w=grid_w,
        grid_h=grid_h,
        out_dir=out["comparison"],
    )
    data_files.append({"name": "image_difference_summary", "path": str(diff_summary_csv), "source": f"{source_path};{generated_path};{naive_generated or ''}"})
    data_files.append({"name": "per_cell_image_difference", "path": str(per_cell_diff_csv), "source": f"{source_path};{generated_path};{naive_generated or ''}"})

    hpv_csv = out["data"] / "hpv_predictions.csv"
    hpv_json = out["data"] / "hpv_prediction_summary.json"
    hpv_prediction_available = False
    if args.compute_hpv_predictions:
        prediction_images = [("source", source_path), ("progressive", generated_path)]
        if naive_generated is not None:
            prediction_images.append(("naive", naive_generated))
        hpv_csv, hpv_json = compute_hpv_predictions(
            images=prediction_images,
            grid_step_px=int(run_manifest.get("grid_step_px", 256)),
            mil_ckpt=args.mil_ckpt,
            device_name=str(args.device),
            out_dir=out["data"],
        )
    if hpv_csv.exists() and hpv_json.exists():
        hpv_prediction_available = True
        data_files.append({"name": "hpv_predictions", "path": str(hpv_csv), "source": str(args.mil_ckpt)})
        data_files.append({"name": "hpv_prediction_summary", "path": str(hpv_json), "source": str(args.mil_ckpt)})
        hpv_shift_csv = out["data"] / "hpv_prediction_shift_source_progressive.csv"
        if hpv_shift_csv.exists():
            data_files.append({"name": "hpv_prediction_shift_source_progressive", "path": str(hpv_shift_csv), "source": str(args.mil_ckpt)})

    elements.extend(
        export_chart_elements(
            charts_dir=out["charts"],
            hpv_csv=hpv_csv,
            diff_summary_csv=diff_summary_csv,
            path_plan_summary=path_plan_summary,
        )
    )

    edit_summary = {
        "region_id": run_manifest.get("region_id"),
        "run_id": run_manifest.get("run_id"),
        "region_size": run_manifest.get("region_size"),
        "grid_shape": run_manifest.get("grid_shape"),
        "grid_step_px": run_manifest.get("grid_step_px"),
        "target_cell_count": len(run_manifest.get("target_cells", [])),
        "window_count": len(steps),
        "prototype_direction": run_manifest.get("prototype_direction"),
        "prototype_latent": run_manifest.get("prototype_latent"),
        "prototype_strength": run_manifest.get("prototype_strength"),
        "preserve_edit_strength": run_manifest.get("preserve_edit_strength"),
        "preserve_visited_strength": run_manifest.get("preserve_visited_strength"),
        "preserve_fresh_context_strength": run_manifest.get("preserve_fresh_context_strength"),
        "progressive_command": run_manifest.get("command"),
        "naive_baseline_command": " ".join(shlex.quote(part) for part in naive_cmd),
        "naive_baseline_script": str(naive_script),
        "naive_baseline_available": naive_generated is not None,
        "naive_baseline_generated": "" if naive_generated is None else str(naive_generated),
        "organized_output": True,
        "hpv_prediction_available": hpv_prediction_available,
        "hpv_prediction_method": "image_reencoded_uni2_region_mil" if hpv_prediction_available else "",
    }
    edit_summary_path = out["data"] / "edit_summary.json"
    write_json(edit_summary_path, edit_summary)
    data_files.append({"name": "edit_summary", "path": str(edit_summary_path), "source": str(run_manifest_path)})

    element_manifest = {
        "out_dir": str(args.out_dir),
        "selector_dir": str(args.selector_dir),
        "progressive_run_dir": str(args.progressive_run_dir),
        "base_dir": str(args.base_dir),
        "naive_run_dir": str(args.naive_run_dir),
        "naive_baseline_available": naive_generated is not None,
        "elements": elements,
        "data_files": data_files,
    }
    manifest_path = args.out_dir / "element_manifest.json"
    write_json(manifest_path, element_manifest)

    missing_outputs = [item["path"] for item in elements if not image_exists(Path(item["path"]))]
    if missing_outputs:
        raise RuntimeError("Some listed elements are missing or empty:\n" + "\n".join(missing_outputs))
    print(json.dumps({"element_count": len(elements), "data_file_count": len(data_files), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
