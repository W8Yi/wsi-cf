#!/usr/bin/env python3
"""
Render high-attention tile overlays on whole-slide images.

Designed for local-PC use:
- reads the exported MIL attention CSV
- resolves WSI paths from tcga_wsi_index.json
- draws high-attention tile boxes on a slide thumbnail

Important scale rule:
- tile coords come from a 20x extraction grid with tile size 256
- if the source WSI is 40x, each 20x tile spans 512x512 pixels at level 0
- this script expands the tile box size automatically using the slide magnification
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageDraw

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from e


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def load_wsi_index(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


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

    # Fallback from MPP when objective power is missing.
    mpp_x = safe_float(props.get("openslide.mpp-x"), -1.0)
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(tile_size_20x * (objective_power / 20.0))))


def derive_svs_name_from_h5(h5_path: str) -> str:
    return f"{Path(h5_path).stem}.svs"


def resolve_wsi_path(
    row: dict,
    wsi_index: dict,
    search_root: Optional[Path],
    search_cache: Dict[str, Path],
) -> Path:
    svs_name = derive_svs_name_from_h5(row["h5_path"])

    if svs_name in wsi_index:
        for loc in wsi_index[svs_name].get("locations", []):
            p = Path(loc["path"])
            if p.exists():
                return p

    if search_root is not None:
        cached = search_cache.get(svs_name)
        if cached is not None and cached.exists():
            return cached
        matches = list(search_root.rglob(svs_name))
        if matches:
            search_cache[svs_name] = matches[0]
            return matches[0]

    raise FileNotFoundError(
        f"Could not resolve WSI for {row['slide_key']} ({svs_name}). "
        "Either fix tcga_wsi_index.json paths or pass --search_root."
    )


def group_rows(rows: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["slide_key"]].append(row)
    for slide_key in grouped:
        grouped[slide_key].sort(key=lambda r: int(r["tile_rank"]))
    return grouped


def pick_color(weight_01: float) -> tuple[int, int, int, int]:
    # Yellow -> orange -> red
    weight_01 = min(1.0, max(0.0, weight_01))
    r = 255
    g = int(round(255 * (1.0 - 0.75 * weight_01)))
    b = int(round(64 * (1.0 - weight_01)))
    a = int(round(70 + 110 * weight_01))
    return (r, g, b, a)


def render_overlay(
    *,
    slide_path: Path,
    rows: List[dict],
    out_path: Path,
    thumb_max_dim: int,
    tile_size_20x: int,
    line_width: int,
    annotate_top_n: int,
) -> None:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        w0, h0 = slide.dimensions
        objective_power = infer_objective_power(slide)
        tile_px = level0_tile_size(tile_size_20x=tile_size_20x, objective_power=objective_power)

        scale = min(thumb_max_dim / max(1, w0), thumb_max_dim / max(1, h0))
        thumb_w = max(1, int(round(w0 * scale)))
        thumb_h = max(1, int(round(h0 * scale)))

        thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGBA")
        overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        attn_vals = [safe_float(r["attention"], 0.0) for r in rows]
        attn_min = min(attn_vals) if attn_vals else 0.0
        attn_max = max(attn_vals) if attn_vals else 1.0
        denom = max(1e-12, attn_max - attn_min)

        for idx, row in enumerate(rows):
            x = int(float(row["coord_x"]))
            y = int(float(row["coord_y"]))
            attn = safe_float(row["attention"], 0.0)
            weight = (attn - attn_min) / denom if attn_max > attn_min else 1.0
            color = pick_color(weight)

            x1 = int(round(x * scale))
            y1 = int(round(y * scale))
            x2 = int(round((x + tile_px) * scale))
            y2 = int(round((y + tile_px) * scale))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(1, line_width))

            if idx < annotate_top_n:
                label = str(idx + 1)
                tx = min(max(0, x1 + 2), max(0, thumb.width - 20))
                ty = min(max(0, y1 + 2), max(0, thumb.height - 12))
                draw.rectangle([tx, ty, tx + 14, ty + 10], fill=(255, 255, 255, 180))
                draw.text((tx + 2, ty - 1), label, fill=(0, 0, 0, 255))

        merged = Image.alpha_composite(thumb, overlay).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.save(out_path)
    finally:
        slide.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles_csv",
        type=Path,
        required=True,
        help="Path to top_attention_tiles.csv from export_hnsc_hpv_mil_attention.py",
    )
    parser.add_argument(
        "--wsi_index",
        type=Path,
        default=Path("metadata/indexes/tcga_wsi_index.json"),
        help="JSON mapping slide filenames to local WSI locations.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("attention_overlay_pngs"),
        help="Directory to save overlay PNGs.",
    )
    parser.add_argument(
        "--slide_key",
        type=str,
        default="",
        help="Optional single slide_key to render. Empty means all slides in the CSV.",
    )
    parser.add_argument(
        "--max_slides",
        type=int,
        default=0,
        help="Optional limit on number of slides rendered.",
    )
    parser.add_argument(
        "--search_root",
        type=Path,
        default=None,
        help="Optional fallback root to search for .svs files by basename if index paths do not exist.",
    )
    parser.add_argument(
        "--thumb_max_dim",
        type=int,
        default=2048,
        help="Max width/height of the output thumbnail PNG.",
    )
    parser.add_argument(
        "--tile_size_20x",
        type=int,
        default=256,
        help="Tile size used at 20x extraction.",
    )
    parser.add_argument(
        "--line_width",
        type=int,
        default=2,
        help="Overlay rectangle width in thumbnail pixels.",
    )
    parser.add_argument(
        "--annotate_top_n",
        type=int,
        default=10,
        help="Write rank labels for the top N tiles per slide.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.tiles_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.tiles_csv}")

    grouped = group_rows(rows)
    wsi_index = load_wsi_index(args.wsi_index)
    search_cache: Dict[str, Path] = {}

    slide_items = list(grouped.items())
    if args.slide_key:
        slide_items = [item for item in slide_items if item[0] == args.slide_key]
    if args.max_slides > 0:
        slide_items = slide_items[: args.max_slides]
    if not slide_items:
        raise SystemExit("No matching slides to render.")

    for slide_key, slide_rows in slide_items:
        slide_path = resolve_wsi_path(
            row=slide_rows[0],
            wsi_index=wsi_index,
            search_root=args.search_root,
            search_cache=search_cache,
        )
        out_path = args.out_dir / f"{slide_key}__attn_overlay.png"
        print(f"[render] {slide_key} <- {slide_path}", flush=True)
        render_overlay(
            slide_path=slide_path,
            rows=slide_rows,
            out_path=out_path,
            thumb_max_dim=args.thumb_max_dim,
            tile_size_20x=args.tile_size_20x,
            line_width=args.line_width,
            annotate_top_n=args.annotate_top_n,
        )
        print(f"[ok] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
