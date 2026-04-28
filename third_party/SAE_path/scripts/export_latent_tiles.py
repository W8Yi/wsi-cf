#!/usr/bin/env python3
"""
export_latent_tiles.py

Given a latent-mining run folder (from latent_mining_simple.py), this script:
1) Loads a pass2_top_tiles_*.json (latest by default)
2) For each latent:
   - downloads the required WSI(s) via wsi_downloader.py (cached)
   - crops representative tile images using (x,y) coords
   - saves tiles into: <out_root>/<run_name>/latent_tiles/latent_<id>/tiles/
   - writes a contact-sheet PNG per latent (all tiles in a grid)
   - writes a JSONL metadata file per latent

Efficiency improvements in this version:
- Download-once: precompute required slides and download serially before cropping
- LRU OpenSlide cache: keep at most MAX_OPEN_SLIDES WSIs open (default 4)

Notes:
- UNI/CLAM coords are assumed to be level-0 top-left pixel coordinates.
- To visualize with a consistent 20x-equivalent FOV:
    * if level-0 is ~40x (mpp<=0.26), crop 512x512 at level 0 then resize to 256
    * if level-0 is ~20x (mpp<=0.55), crop 256x256 at level 0
  If mpp is missing, default to 40x behavior (crop 512->256), which is TCGA-safe.

Example:
  python export_latent_tiles.py \
    --out_root /common/users/wq50/SAE_path/runs \
    --run_name myrun \
    --gdc_client /common/users/wq50/SAE_path/gdc-client \
    --tile_size 256 --vis_size 256 --ncols 10 \
    --wsi_cache /common/users/wq50/wsi_cache --keep_wsi_cache
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python + system OpenSlide libs are required.") from e

from utils.wsi_downloader import download_one


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def find_latest_pass2(run_dir: Path) -> Path:
    files = sorted(
        run_dir.glob("pass2_top_tiles_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No pass2_top_tiles_*.json found in {run_dir}")
    return files[0]


def safe_rgb_tile_20x_equiv(
    slide: "openslide.OpenSlide",
    x: int,
    y: int,
    out_size: int = 256,
    fov_scale_20x_equiv: float = 1.0,
    context_grid: int = 1,
    draw_center_box: bool = False,
) -> Tuple[Image.Image, int, Optional[float]]:
    """
    Returns (rgb_img, crop_px_level0, mpp_x).

    Normalizes to a 20x-equivalent field of view:
      - if level-0 is ~20x (mpp ~0.5): read 256x256 at level 0
      - if level-0 is ~40x (mpp ~0.25): read 512x512 at level 0 then resize to 256x256

    fov_scale_20x_equiv enlarges the crop to reflect features derived from a wider field
    (e.g., 10x_pool2x2 SAE features built from 2x2 neighboring 20x tiles -> scale=2.0).
    context_grid (odd integer >=1) expands the crop to a kxk neighborhood with the queried tile
    at the center (useful for 20x context visualization, e.g., 3x3).

    Coordinates (x,y) are assumed to be level-0 top-left.
    If mpp is missing/invalid, defaults to 40x behavior (crop 512->256).
    """
    mpp_x_raw = slide.properties.get("openslide.mpp-x", None)
    mpp_x: Optional[float] = None
    if mpp_x_raw is not None:
        try:
            mpp_x = float(mpp_x_raw)
        except Exception:
            mpp_x = None

    # Decide crop size in level-0 pixels
    # 40x: ~0.25 um/px; 20x: ~0.50 um/px
    scale = max(1.0, float(fov_scale_20x_equiv))
    context_grid = int(context_grid)
    if context_grid < 1 or (context_grid % 2) != 1:
        raise ValueError(f"context_grid must be an odd integer >= 1, got {context_grid}")
    if mpp_x is None or not math.isfinite(mpp_x):
        base_crop = int(round(out_size * 2 * scale))  # TCGA-safe fallback: assume 40x
    elif mpp_x <= 0.26:
        base_crop = int(round(out_size * 2 * scale))
    else:
        base_crop = int(round(out_size * scale))

    base_crop = max(1, int(base_crop))
    crop = int(base_crop * context_grid)
    # coords are top-left of the "actual" tile; shift so this tile is centered in a context_grid x context_grid view.
    offset = (context_grid // 2) * base_crop
    x0 = int(x) - offset
    y0 = int(y) - offset

    rgba = slide.read_region((x0, y0), 0, (int(crop), int(crop))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")

    if crop != out_size:
        rgb = rgb.resize((out_size, out_size), resample=Image.BILINEAR)

    if draw_center_box and context_grid > 1:
        draw = ImageDraw.Draw(rgb)
        cell = float(out_size) / float(context_grid)
        r = context_grid // 2
        # Rectangle for the center tile in resized coordinates.
        x1 = r * cell
        y1 = r * cell
        x2 = (r + 1) * cell
        y2 = (r + 1) * cell
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=max(1, out_size // 128))

    return rgb, int(crop), mpp_x


def make_contact_sheet(images: List[Image.Image], ncols: int, vis_size: int, pad: int = 2) -> Image.Image:
    def _placeholder_tile(sz: int) -> Image.Image:
        tile = Image.new("RGB", (sz, sz), (232, 232, 232))
        d = ImageDraw.Draw(tile)
        w = max(2, sz // 64)
        # Border
        d.rectangle((1, 1, sz - 2, sz - 2), outline=(120, 120, 120), width=w)
        # Diagonal X to indicate intentionally empty slot
        d.line((2, 2, sz - 3, sz - 3), fill=(220, 40, 40), width=w)
        d.line((sz - 3, 2, 2, sz - 3), fill=(220, 40, 40), width=w)
        # Light center mark
        c = sz // 2
        r = max(2, sz // 20)
        d.ellipse((c - r, c - r, c + r, c + r), fill=(250, 250, 250), outline=(150, 150, 150))
        return tile

    if not images:
        return _placeholder_tile(vis_size)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(images) / ncols))
    w = ncols * vis_size + (ncols + 1) * pad
    h = nrows * vis_size + (nrows + 1) * pad
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    total_slots = nrows * ncols
    placeholder = _placeholder_tile(vis_size)
    for i in range(total_slots):
        r, c = divmod(i, ncols)
        x0 = pad + c * (vis_size + pad)
        y0 = pad + r * (vis_size + pad)
        if i < len(images):
            tile = images[i].resize((vis_size, vis_size), resample=Image.BILINEAR)
        else:
            tile = placeholder
        canvas.paste(tile, (x0, y0))
    return canvas


def build_h5_to_slide_map(manifest: dict) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for slide_key, e in manifest.items():
        h5 = e.get("h5_path")
        if h5:
            m[str(h5)] = str(slide_key)
    return m


def get_local_slide_path(slide_key: str, manifest: dict) -> Optional[Path]:
    """
    If manifest provides a local SVS path and it exists, return it.
    """
    entry = manifest.get(slide_key, {})
    p = entry.get("slide_path")
    if not p:
        return None
    p = Path(p)
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--pass2_json", type=Path, default=None)

    ap.add_argument("--gdc_client", type=Path, required=True, help="Path to gdc-client executable")
    ap.add_argument("--token", type=Path, default=None)

    # Output tile size (final image). For UNI-style visualization, keep 256.
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--vis_size", type=int, default=256)
    ap.add_argument("--ncols", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_latents", type=int, default=-1)
    ap.add_argument(
        "--max_tiles_per_slide_per_latent",
        type=int,
        default=-1,
        help="Display/export cap per latent per slide (post-pass2). Helps diversify contact sheets. -1 disables.",
    )
    ap.add_argument(
        "--context_grid",
        type=int,
        default=1,
        help="Odd context window grid for exported crops (e.g., 3 => 3x3 neighborhood with actual tile centered).",
    )
    ap.add_argument(
        "--draw_center_box",
        action="store_true",
        help="When --context_grid > 1, draw a red box marking the actual mined tile at the center.",
    )
    ap.add_argument(
        "--feature_magnification_override",
        type=str,
        default="",
        choices=["", "20x", "10x_pool2x2"],
        help="Override feature magnification used in pass2 (auto-detected from pass2 config by default).",
    )
    ap.add_argument(
        "--download_workers",
        type=int,
        default=8,
        help="Deprecated in slide-centric mode (kept for CLI compatibility).",
    )

    ap.add_argument(
        "--wsi_cache",
        type=Path,
        default=Path("./wsi_cache"),
        help="Directory to store downloaded WSIs (default: ./wsi_cache)",
    )
    ap.add_argument(
        "--keep_wsi_cache",
        action="store_true",
        help="Keep cached WSIs after finishing (default: delete cache contents).",
    )

    # LRU: keep at most this many WSIs open at once
    ap.add_argument(
        "--max_open_slides",
        type=int,
        default=4,
        help="Deprecated in slide-centric mode (kept for CLI compatibility).",
    )

    args = ap.parse_args()

    run_dir = (args.out_root / args.run_name).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    pass2_path = args.pass2_json or find_latest_pass2(run_dir)
    pass2 = load_json(pass2_path)
    top_tiles = pass2.get("top_tiles", {})
    if not top_tiles:
        raise ValueError(f"No 'top_tiles' found in {pass2_path}")
    sdf_hierarchy = pass2.get("sdf_hierarchy", {}) or {}
    level1_to_level2_raw = sdf_hierarchy.get("level1_to_level2_parent_selected", {}) or {}
    level1_to_level2 = {int(k): int(v) for k, v in level1_to_level2_raw.items()}

    manifest_path = Path(pass2.get("config_pass1", {}).get("index_json"))
    pass1_cfg = pass2.get("config_pass1", {}) or {}
    feature_mag = args.feature_magnification_override or str(pass1_cfg.get("magnification", "20x"))
    feature_fov_scale = 2.0 if feature_mag == "10x_pool2x2" else 1.0
    if args.context_grid < 1 or (args.context_grid % 2) != 1:
        raise ValueError(f"--context_grid must be an odd integer >=1, got {args.context_grid}")
    print(f"Using manifest: {manifest_path}")
    print(f"Feature magnification: {feature_mag} (export FOV scale={feature_fov_scale:.1f}x 20x-equivalent)")
    manifest = load_json(manifest_path)
    h5_to_slide = build_h5_to_slide_map(manifest)

    out_base = run_dir / f"latent_tiles_{pass2_path.stem}"
    out_base.mkdir(parents=True, exist_ok=True)
    
    sheets_dir = out_base / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)


    wsi_cache = args.wsi_cache.expanduser().resolve()
    wsi_cache.mkdir(parents=True, exist_ok=True)

    # Cache for resolved slide paths
    slide_path_cache: Dict[str, Path] = {}

    def slide_key_from_h5(h5_path: str) -> str:
        sk = h5_to_slide.get(h5_path)
        if not sk:
            raise KeyError(f"h5_path not found in manifest: {h5_path}")
        return sk

    def get_wsi_path(slide_key: str) -> Tuple[Path, bool]:
        """
        Returns (path, is_downloaded).
        is_downloaded=False means the path is from manifest local storage.
        """
        p = slide_path_cache.get(slide_key)
        if p is not None:
            # Cached paths are either local manifest paths or downloaded files.
            is_downloaded = str(p).startswith(str(wsi_cache))
            return p, is_downloaded

        local = get_local_slide_path(slide_key, manifest)
        if local is not None:
            slide_path_cache[slide_key] = local
            return local, False

        # download (outside lock)
        p = download_one(
            slide_key,
            index_json=manifest_path,
            out_dir=wsi_cache,
            gdc_client=args.gdc_client,
            token_path=args.token,
            latest=True,
            overwrite=False,
            skip_if_exists=True,
            verbose=True,
        )

        slide_path_cache[slide_key] = p
        return p, True

    latent_ids = sorted(top_tiles.keys(), key=lambda x: int(x))
    if args.max_latents > 0:
        latent_ids = latent_ids[: args.max_latents]

    print(f"Using pass2: {pass2_path}")
    print(f"WSI cache: {wsi_cache} (keep={args.keep_wsi_cache})")
    print("Slide-centric mode: each slide is processed once then removed if it was downloaded.")
    if args.download_workers != 8:
        print("[note] --download_workers is deprecated and ignored in slide-centric mode.")
    if args.max_open_slides != 4:
        print("[note] --max_open_slides is deprecated and ignored in slide-centric mode.")
    print(f"Exporting latents: {len(latent_ids)} -> {out_base}")
    if level1_to_level2:
        used_l2 = sorted({level1_to_level2.get(int(lid)) for lid in latent_ids if int(lid) in level1_to_level2})
        print(f"Hierarchy mode: enabled (level2 groups found for {len(used_l2)} level2 parents)")
    else:
        print("Hierarchy mode: disabled (no sdf_hierarchy in pass2)")

    # ----------------------------
    # Build slide-centric job list
    # ----------------------------
    latent_state: Dict[int, dict] = {}
    slide_jobs: Dict[str, List[dict]] = {}
    for lid_str in latent_ids:
        lid = int(lid_str)
        l2_parent = level1_to_level2.get(lid, None)
        if l2_parent is None:
            latent_dir = out_base / f"latent_{lid:05d}"
            sheet_path = sheets_dir / f"latent_{lid:05d}.png"
        else:
            latent_dir = out_base / f"level2_{l2_parent:05d}" / f"latent_{lid:05d}"
            sheet_path = sheets_dir / f"level2_{l2_parent:05d}_latent_{lid:05d}.png"
        tiles_dir = latent_dir / "tiles"
        latent_dir.mkdir(parents=True, exist_ok=True)
        tiles_dir.mkdir(parents=True, exist_ok=True)

        latent_state[lid] = {
            "latent_dir": latent_dir,
            "tiles_dir": tiles_dir,
            "meta_path": latent_dir / "tiles_meta.jsonl",
            "sheet_path": sheet_path,
            "meta_lines": [],
            "sheet_items": [],  # list of (rank, png_path)
            "l2_parent": l2_parent,
        }

        per_slide_count: Dict[str, int] = {}
        cap_per_slide = int(args.max_tiles_per_slide_per_latent)
        kept_rank = 0
        for rank, t in enumerate(top_tiles[lid_str]):
            h5_path = str(t["h5_path"])
            slide_key = h5_to_slide.get(h5_path)
            if not slide_key:
                print(f"[skip] latent {lid} rank {rank}: h5_path not in manifest: {h5_path}")
                continue
            if cap_per_slide > 0:
                n = per_slide_count.get(slide_key, 0)
                if n >= cap_per_slide:
                    continue
                per_slide_count[slide_key] = n + 1
            slide_jobs.setdefault(slide_key, []).append(
                {
                    "latent": lid,
                    "rank": kept_rank,
                    "score": float(t["score"]),
                    "x": int(t["x"]),
                    "y": int(t["y"]),
                    "h5_path": h5_path,
                    "tile_idx": int(t.get("tile_idx", -1)),
                }
            )
            kept_rank += 1

    print(f"Unique slides needed: {len(slide_jobs)}")


    try:
        # Track exported hierarchy
        exported_hierarchy: Dict[int, List[int]] = {}
        for lid in sorted(latent_state.keys()):
            l2_parent = latent_state[lid]["l2_parent"]
            if l2_parent is not None:
                exported_hierarchy.setdefault(int(l2_parent), []).append(int(lid))

        # Process each slide once, then delete it if downloaded by this script.
        for slide_idx, (slide_key, jobs) in enumerate(sorted(slide_jobs.items()), start=1):
            try:
                wsi_path, is_downloaded = get_wsi_path(slide_key)
            except Exception as e:
                print(f"[warn] slide resolve failed ({slide_key}): {e}")
                continue

            try:
                osr = openslide.OpenSlide(str(wsi_path))
            except Exception as e:
                print(f"[warn] openslide open failed ({slide_key}): {e}")
                continue

            print(f"[slide {slide_idx}/{len(slide_jobs)}] {slide_key} jobs={len(jobs)}")
            try:
                for j in jobs:
                    lid = int(j["latent"])
                    rank = int(j["rank"])
                    score = float(j["score"])
                    x = int(j["x"])
                    y = int(j["y"])
                    h5_path = str(j["h5_path"])
                    tile_idx = int(j["tile_idx"])
                    lat = latent_state[lid]
                    l2_parent = lat["l2_parent"]
                    tiles_dir = lat["tiles_dir"]

                    fn = f"r{rank:03d}_score{score:.4f}_{slide_key}_x{x}_y{y}.png"
                    out_png = tiles_dir / fn

                    mpp_x: Optional[float] = None
                    crop_px: Optional[int] = None

                    if out_png.exists() and not args.overwrite:
                        try:
                            Image.open(out_png).convert("RGB")
                        except Exception:
                            try:
                                im, crop_px, mpp_x = safe_rgb_tile_20x_equiv(
                                    osr,
                                    x=x,
                                    y=y,
                                    out_size=args.tile_size,
                                    fov_scale_20x_equiv=feature_fov_scale,
                                    context_grid=args.context_grid,
                                    draw_center_box=bool(args.draw_center_box),
                                )
                                im.save(out_png)
                            except Exception as e:
                                print(f"[skip] latent {lid} rank {rank}: read_region failed: {e}")
                                continue
                    else:
                        try:
                            im, crop_px, mpp_x = safe_rgb_tile_20x_equiv(
                                osr,
                                x=x,
                                y=y,
                                out_size=args.tile_size,
                                fov_scale_20x_equiv=feature_fov_scale,
                                context_grid=args.context_grid,
                                draw_center_box=bool(args.draw_center_box),
                            )
                            im.save(out_png)
                        except Exception as e:
                            print(f"[skip] latent {lid} rank {rank}: read_region failed: {e}")
                            continue

                    lat["sheet_items"].append((rank, out_png))
                    lat["meta_lines"].append(
                        json.dumps(
                            {
                                "latent": lid,
                                "level2_parent": int(l2_parent) if l2_parent is not None else None,
                                "rank": rank,
                                "score": score,
                                "slide": slide_key,
                                "x": x,
                                "y": y,
                                "h5_path": h5_path,
                                "tile_idx": tile_idx,
                                "png": str(out_png),
                                "mpp_x": float(mpp_x) if mpp_x is not None else None,
                                "crop_px_level0": int(crop_px) if crop_px is not None else None,
                                "out_size": int(args.tile_size),
                                "feature_magnification": feature_mag,
                                "feature_fov_scale_20x_equiv": float(feature_fov_scale),
                                "context_grid": int(args.context_grid),
                                "draw_center_box": bool(args.draw_center_box),
                            }
                        )
                    )
            finally:
                try:
                    osr.close()
                except Exception:
                    pass

                if is_downloaded and (not args.keep_wsi_cache):
                    try:
                        if wsi_path.is_dir():
                            shutil.rmtree(wsi_path, ignore_errors=True)
                        elif wsi_path.exists():
                            wsi_path.unlink()
                        slide_path_cache.pop(slide_key, None)
                    except Exception as e:
                        print(f"[warn] could not delete slide after use ({wsi_path}): {e}")

        # Finalize outputs per latent
        for lid in sorted(latent_state.keys()):
            lat = latent_state[lid]
            meta_path = lat["meta_path"]
            sheet_path = lat["sheet_path"]
            latent_dir = lat["latent_dir"]

            meta_lines = lat["meta_lines"]
            meta_path.write_text("\n".join(meta_lines) + ("\n" if meta_lines else ""))

            sheet_items = sorted(lat["sheet_items"], key=lambda t: t[0])
            sheet_imgs: List[Image.Image] = []
            for _, png_path in sheet_items:
                try:
                    sheet_imgs.append(Image.open(png_path).convert("RGB"))
                except Exception:
                    pass
            sheet = make_contact_sheet(sheet_imgs, ncols=args.ncols, vis_size=args.vis_size, pad=2)
            sheet.save(sheet_path)
            print(f"[ok] latent {lid:05d}: tiles={len(sheet_imgs)} -> {latent_dir}")

        if level1_to_level2:
            for k in list(exported_hierarchy.keys()):
                exported_hierarchy[k] = sorted(exported_hierarchy[k])
            hierarchy_summary = {
                "source_pass2": str(pass2_path),
                "level1_dim": sdf_hierarchy.get("level1_dim"),
                "level2_dim": sdf_hierarchy.get("level2_dim"),
                "exported_level2_count": len(exported_hierarchy),
                "exported_latent_count": sum(len(v) for v in exported_hierarchy.values()),
                "exported_level2_to_level1_latents": {int(k): v for k, v in sorted(exported_hierarchy.items())},
                "level1_to_level2_parent_selected": {int(k): int(v) for k, v in sorted(level1_to_level2.items())},
            }
            (out_base / "hierarchy_summary.json").write_text(json.dumps(hierarchy_summary, indent=2))
            print(f"[ok] wrote hierarchy summary: {out_base / 'hierarchy_summary.json'}")

    finally:
        # Default: delete remaining cache contents
        if not args.keep_wsi_cache:
            try:
                for p in wsi_cache.iterdir():
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        try:
                            p.unlink()
                        except Exception:
                            pass
                print(f"[cleanup] removed cached WSIs under: {wsi_cache}")
            except Exception as e:
                print(f"[warn] cache cleanup failed: {e}")


if __name__ == "__main__":
    print("Starting export latent tiles...")
    main()
