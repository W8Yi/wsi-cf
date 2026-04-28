#!/usr/bin/env python3
"""
Visualize top SAE neuron tiles on real WSIs.

Input:
- top_neuron_tiles.csv from run_hnsc_hpv_sae_neuron_pipeline.py

Outputs:
- per-neuron individual tile PNGs
- per-neuron contact sheet PNGs
- global paged contact sheets of all exported tiles
- top1-per-neuron sheet
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from e


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(x: object, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: object, default: int) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def infer_objective_power(slide: "openslide.OpenSlide") -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            v = safe_float(props.get(key), -1.0)
            if v > 0:
                return v
    mpp_x = safe_float(props.get("openslide.mpp-x"), -1.0)
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(tile_size_20x * (objective_power / 20.0))))


class SlideCache:
    def __init__(self, max_open: int) -> None:
        self.max_open = max(1, int(max_open))
        self._cache: "OrderedDict[Path, openslide.OpenSlide]" = OrderedDict()

    def get(self, path: Path) -> "openslide.OpenSlide":
        if path in self._cache:
            s = self._cache.pop(path)
            self._cache[path] = s
            return s
        s = openslide.OpenSlide(str(path))
        self._cache[path] = s
        while len(self._cache) > self.max_open:
            _, old = self._cache.popitem(last=False)
            old.close()
        return s

    def close(self) -> None:
        for s in self._cache.values():
            s.close()
        self._cache.clear()


def resolve_wsi_path(slide_key: str, wsi_dir: Path, cache: Dict[str, Optional[Path]]) -> Optional[Path]:
    cached = cache.get(slide_key)
    if cached is not None:
        return cached

    direct = wsi_dir / f"{slide_key}.svs"
    if direct.exists():
        cache[slide_key] = direct
        return direct

    # Fallback in case filenames include suffixes.
    matches = sorted(wsi_dir.glob(f"{slide_key}*.svs"))
    if matches:
        cache[slide_key] = matches[0]
        return matches[0]

    cache[slide_key] = None
    return None


def crop_tile(
    slide: "openslide.OpenSlide",
    *,
    x: int,
    y: int,
    tile_size_20x: int,
    out_tile_size: int,
) -> Image.Image:
    objective = infer_objective_power(slide)
    crop_px = level0_tile_size(tile_size_20x=tile_size_20x, objective_power=objective)

    rgba = slide.read_region((x, y), 0, (crop_px, crop_px)).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    if crop_px != out_tile_size:
        rgb = rgb.resize((out_tile_size, out_tile_size), resample=Image.BILINEAR)
    return rgb


def add_label_overlay(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width, 16], fill=(255, 255, 255))
    draw.text((3, 2), text, fill=(0, 0, 0))
    return out


def make_contact_sheet(
    image_paths: List[Path],
    out_path: Path,
    *,
    ncols: int,
    tile_size: int,
    pad: int = 4,
    with_header: str = "",
) -> None:
    n = len(image_paths)
    if n == 0:
        return
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols
    header_h = 26 if with_header else 0
    w = ncols * tile_size + (ncols + 1) * pad
    h = nrows * tile_size + (nrows + 1) * pad + header_h

    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    if with_header:
        draw.rectangle([0, 0, w, header_h], fill=(230, 230, 230))
        draw.text((8, 6), with_header, fill=(0, 0, 0))

    y0_base = header_h
    for i, p in enumerate(image_paths):
        r = i // ncols
        c = i % ncols
        x0 = pad + c * (tile_size + pad)
        y0 = y0_base + pad + r * (tile_size + pad)
        im = Image.open(p).convert("RGB")
        if im.size != (tile_size, tile_size):
            im = im.resize((tile_size, tile_size), resample=Image.BILINEAR)
        canvas.paste(im, (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles_csv",
        type=Path,
        required=True,
        help="top_neuron_tiles.csv from SAE neuron pipeline",
    )
    parser.add_argument(
        "--wsi_dir",
        type=Path,
        default=Path("wsi/hnsc_hpv"),
        help="Directory containing downloaded HNSC HPV .svs files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: <tiles_csv_dir>/tile_visualizations)",
    )
    parser.add_argument("--tile_size_20x", type=int, default=256, help="Original extraction tile size at 20x")
    parser.add_argument("--out_tile_size", type=int, default=256, help="Output PNG tile size")
    parser.add_argument("--sheet_ncols", type=int, default=10, help="Contact-sheet columns")
    parser.add_argument("--sheet_nrows", type=int, default=8, help="Rows per global all-tiles page")
    parser.add_argument("--max_neurons", type=int, default=0, help="Optional cap on number of neurons")
    parser.add_argument("--max_tiles_per_neuron", type=int, default=0, help="Optional cap per neuron")
    parser.add_argument("--max_open_slides", type=int, default=8, help="Max open WSI handles")
    parser.add_argument("--skip_existing", action="store_true", help="Skip tile PNGs that already exist")
    parser.add_argument("--no_label_overlay", action="store_true", help="Do not render text strip on tile images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.tiles_csv)
    if not rows:
        raise SystemExit(f"No rows in {args.tiles_csv}")

    out_dir = args.out_dir or (args.tiles_csv.parent / "tile_visualizations")
    by_neuron_dir = out_dir / "by_neuron"
    sheets_dir = out_dir / "sheets"
    by_neuron_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    # Normalize and group.
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        r["latent_idx"] = safe_int(r.get("latent_idx"), -1)
        r["prototype_rank"] = safe_int(r.get("prototype_rank"), 10**9)
        r["coord_x"] = safe_int(r.get("coord_x"), -1)
        r["coord_y"] = safe_int(r.get("coord_y"), -1)
        grouped[r["latent_idx"]].append(r)

    latent_ids = sorted([x for x in grouped.keys() if x >= 0])
    if args.max_neurons > 0:
        latent_ids = latent_ids[: args.max_neurons]

    slide_path_cache: Dict[str, Optional[Path]] = {}
    slide_cache = SlideCache(max_open=args.max_open_slides)

    exported_rows = []
    missing_wsi = []
    missing_coords = 0
    total_saved = 0

    try:
        for li, latent_idx in enumerate(latent_ids, start=1):
            entries = sorted(grouped[latent_idx], key=lambda x: x["prototype_rank"])
            if args.max_tiles_per_neuron > 0:
                entries = entries[: args.max_tiles_per_neuron]

            latent_dir = by_neuron_dir / f"latent_{latent_idx}"
            tiles_dir = latent_dir / "tiles"
            tiles_dir.mkdir(parents=True, exist_ok=True)

            tile_paths: List[Path] = []
            for r in entries:
                if r["coord_x"] < 0 or r["coord_y"] < 0:
                    missing_coords += 1
                    continue

                slide_key = r["slide_key"]
                wsi_path = resolve_wsi_path(slide_key=slide_key, wsi_dir=args.wsi_dir, cache=slide_path_cache)
                if wsi_path is None:
                    missing_wsi.append(slide_key)
                    continue

                out_name = (
                    f"rank_{int(r['prototype_rank']):03d}"
                    f"__{slide_key}"
                    f"__tile_{safe_int(r.get('tile_index'), -1):06d}.png"
                )
                out_path = tiles_dir / out_name
                if args.skip_existing and out_path.exists():
                    tile_paths.append(out_path)
                else:
                    slide = slide_cache.get(wsi_path)
                    tile = crop_tile(
                        slide,
                        x=r["coord_x"],
                        y=r["coord_y"],
                        tile_size_20x=args.tile_size_20x,
                        out_tile_size=args.out_tile_size,
                    )
                    if not args.no_label_overlay:
                        label = (
                            f"L{latent_idx} R{int(r['prototype_rank'])} "
                            f"{slide_key} a={safe_float(r.get('attention'), 0.0):.3f}"
                        )
                        tile = add_label_overlay(tile, label)
                    tile.save(out_path)
                    tile_paths.append(out_path)
                    total_saved += 1

                exported_rows.append(
                    {
                        "latent_idx": latent_idx,
                        "prototype_rank": int(r["prototype_rank"]),
                        "selected_direction": r.get("selected_direction", ""),
                        "slide_key": slide_key,
                        "tile_index": safe_int(r.get("tile_index"), -1),
                        "coord_x": int(r["coord_x"]),
                        "coord_y": int(r["coord_y"]),
                        "attention": safe_float(r.get("attention"), 0.0),
                        "sae_activation": safe_float(r.get("sae_activation"), 0.0),
                        "attention_weighted_activation": safe_float(r.get("attention_weighted_activation"), 0.0),
                        "wsi_path": str(wsi_path),
                        "tile_png_path": str(out_path),
                    }
                )

            if tile_paths:
                make_contact_sheet(
                    tile_paths,
                    latent_dir / "sheet.png",
                    ncols=args.sheet_ncols,
                    tile_size=args.out_tile_size,
                    with_header=f"latent={latent_idx}  n_tiles={len(tile_paths)}",
                )
            if li % 5 == 0 or li == len(latent_ids):
                print(f"[progress] neurons {li}/{len(latent_ids)}", flush=True)
    finally:
        slide_cache.close()

    # Export manifest of produced tile pngs.
    exported_rows.sort(key=lambda r: (r["latent_idx"], r["prototype_rank"], r["slide_key"]))
    write_csv(
        out_dir / "exported_tiles_manifest.csv",
        [
            "latent_idx",
            "prototype_rank",
            "selected_direction",
            "slide_key",
            "tile_index",
            "coord_x",
            "coord_y",
            "attention",
            "sae_activation",
            "attention_weighted_activation",
            "wsi_path",
            "tile_png_path",
        ],
        exported_rows,
    )

    # Global sheets.
    all_tile_paths = [Path(r["tile_png_path"]) for r in exported_rows]
    page_size = max(1, int(args.sheet_ncols) * int(args.sheet_nrows))
    page_count = 0
    for start in range(0, len(all_tile_paths), page_size):
        page_paths = all_tile_paths[start:start + page_size]
        page_count += 1
        make_contact_sheet(
            page_paths,
            sheets_dir / f"all_tiles_sheet_page_{page_count:03d}.png",
            ncols=args.sheet_ncols,
            tile_size=args.out_tile_size,
            with_header=f"all tiles page {page_count}  n={len(page_paths)}",
        )

    # Top-1 per neuron sheet.
    top1_paths = []
    seen = set()
    for r in exported_rows:
        lid = int(r["latent_idx"])
        if lid in seen:
            continue
        seen.add(lid)
        top1_paths.append(Path(r["tile_png_path"]))
    if top1_paths:
        make_contact_sheet(
            top1_paths,
            sheets_dir / "top1_per_neuron_sheet.png",
            ncols=args.sheet_ncols,
            tile_size=args.out_tile_size,
            with_header=f"top1 per neuron  n_neurons={len(top1_paths)}",
        )

    summary = {
        "tiles_csv": str(args.tiles_csv),
        "wsi_dir": str(args.wsi_dir),
        "out_dir": str(out_dir),
        "neurons_requested": len(latent_ids),
        "tiles_rendered": int(total_saved),
        "rows_exported": int(len(exported_rows)),
        "missing_coord_rows": int(missing_coords),
        "missing_wsi_slide_keys": sorted(set(missing_wsi)),
        "n_missing_wsi_slide_keys": int(len(set(missing_wsi))),
        "global_sheet_pages": int(page_count),
        "files": {
            "exported_tiles_manifest_csv": str(out_dir / "exported_tiles_manifest.csv"),
            "by_neuron_dir": str(by_neuron_dir),
            "sheets_dir": str(sheets_dir),
        },
    }
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[ok] wrote {out_dir / 'summary.json'}", flush=True)
    print(f"[ok] wrote {out_dir / 'exported_tiles_manifest.csv'}", flush=True)
    print(f"[ok] sheets_dir={sheets_dir}", flush=True)


if __name__ == "__main__":
    main()
