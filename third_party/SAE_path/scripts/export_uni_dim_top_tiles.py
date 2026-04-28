#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

try:
    import openslide
except Exception as e:  # pragma: no cover - runtime dependency
    raise RuntimeError("openslide-python + system OpenSlide libs are required.") from e

from utils.wsi_downloader import download_one, parse_slide_key


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Export image tiles for raw UNI-dimension top-tile results. "
            "Reads JSON from scripts.uni_dim_top_tiles or scripts.aggregate_uni_dim_top_tiles, "
            "downloads WSIs as needed, and crops 20x-equivalent 256x256 tiles."
        )
    )
    ap.add_argument("--input-json", type=Path, required=True, help="uni_dim_top_tiles JSON or merged aggregate JSON.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output root for exported tiles/contact sheets.")
    ap.add_argument(
        "--index-json",
        type=Path,
        default=Path("metadata/indexes/manifest_index.json"),
        help="Enriched manifest index used by utils.wsi_downloader (must contain gdc refs).",
    )
    ap.add_argument(
        "--gdc-client",
        type=Path,
        default=Path("/common/users/wq50/SAE_path/gdc/gdc-client"),
        help="Path to gdc-client executable.",
    )
    ap.add_argument("--token", type=Path, default=None, help="Optional GDC token file.")
    ap.add_argument(
        "--wsi-cache",
        type=Path,
        default=Path("/common/users/wq50/wsi_cache"),
        help="Directory to cache downloaded WSIs.",
    )
    ap.add_argument("--keep-wsi-cache", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing tile images/contact sheets.")
    ap.add_argument("--skip-if-exists", action="store_true", help="Skip tiles that already exist.")
    ap.add_argument("--max-axes", type=int, default=0, help="Optional cap on number of axes to export (0=all in input order).")
    ap.add_argument(
        "--axes",
        type=str,
        default="",
        help="Optional comma-separated axis indices to export. If empty, export all.",
    )
    ap.add_argument(
        "--signs",
        type=str,
        default="pos,neg",
        help="Comma-separated signs to export (pos,neg).",
    )
    ap.add_argument("--top-n", type=int, default=0, help="Optional cap on number of top tiles per axis/sign (0=use all in input JSON).")
    ap.add_argument("--tile-size", type=int, default=256, help="Final output tile size (default 256).")
    ap.add_argument("--vis-size", type=int, default=256, help="Contact sheet thumbnail size.")
    ap.add_argument("--ncols", type=int, default=10, help="Contact sheet columns.")
    ap.add_argument("--label-tiles", action="store_true", help="Overlay rank/score labels on saved tile images.")
    ap.add_argument(
        "--download-retries",
        type=int,
        default=2,
        help="Retries for transient gdc-client download failures per slide (default: 2).",
    )
    ap.add_argument(
        "--retry-sleep-sec",
        type=float,
        default=2.0,
        help="Sleep between download retries (default: 2.0s).",
    )
    ap.add_argument(
        "--sort-by",
        type=str,
        default="axis_index",
        choices=["axis_index", "diversity", "similarity", "stability", "contrast"],
        help="Ordering for axes if input is aggregate JSON.",
    )
    return ap


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _parse_axes_list(text: str) -> set[int]:
    if not text.strip():
        return set()
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def _parse_signs(text: str) -> list[str]:
    vals = [x.strip().lower() for x in text.split(",") if x.strip()]
    for v in vals:
        if v not in {"pos", "neg"}:
            raise SystemExit(f"Unsupported sign '{v}'")
    return vals


def _safe_rgb_tile_20x_equiv(
    slide: "openslide.OpenSlide",
    x: int,
    y: int,
    out_size: int = 256,
) -> tuple[Image.Image, int, float | None]:
    """
    Return (rgb_img, crop_px_level0, mpp_x) with 20x-equivalent FOV normalization.

    - level0 ~40x (mpp<=0.26): crop 512x512 then resize to 256x256
    - level0 ~20x (mpp>0.26): crop 256x256 at level 0
    - missing mpp: default to 40x behavior (512->256), TCGA-safe
    """
    mpp_x_raw = slide.properties.get("openslide.mpp-x", None)
    mpp_x: float | None = None
    if mpp_x_raw is not None:
        try:
            mpp_x = float(mpp_x_raw)
        except Exception:
            mpp_x = None

    if mpp_x is None or not math.isfinite(mpp_x):
        crop = int(out_size) * 2
    elif mpp_x <= 0.26:
        crop = int(out_size) * 2
    else:
        crop = int(out_size)

    rgba = slide.read_region((int(x), int(y)), 0, (int(crop), int(crop))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    if crop != int(out_size):
        rgb = rgb.resize((int(out_size), int(out_size)), resample=Image.BILINEAR)
    return rgb, int(crop), mpp_x


def _make_contact_sheet(images: list[Image.Image], labels: list[str], *, ncols: int, vis_size: int, pad: int = 4) -> Image.Image:
    if not images:
        return Image.new("RGB", (vis_size, vis_size), (255, 255, 255))
    ncols = max(1, int(ncols))
    label_h = 14
    cell_h = int(vis_size) + label_h
    nrows = int(math.ceil(len(images) / ncols))
    W = ncols * int(vis_size) + (ncols + 1) * pad
    H = nrows * cell_h + (nrows + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, im in enumerate(images):
        r, c = divmod(i, ncols)
        x0 = pad + c * (int(vis_size) + pad)
        y0 = pad + r * (cell_h + pad)
        thumb = im.resize((int(vis_size), int(vis_size)), resample=Image.BILINEAR)
        canvas.paste(thumb, (x0, y0))
        if i < len(labels):
            draw.text((x0 + 2, y0 + int(vis_size) + 1), labels[i], fill=(0, 0, 0), font=font)
    return canvas


def _tile_filename(rank: int, rec: dict[str, Any]) -> str:
    slide_key = parse_slide_key(str(rec["h5_path"]))
    return f"rank_{rank:03d}__{slide_key}__tile{int(rec['tile_idx'])}__x{int(rec['x'])}_y{int(rec['y'])}.png"


def _axis_sort_score(axis_entry: dict[str, Any], mode: str) -> float:
    if mode == "axis_index":
        return float(axis_entry.get("axis_index", 0))
    # Prefer pos sign if present, else any.
    signs = axis_entry.get("signs", {})
    m = signs.get("pos", signs.get(next(iter(signs.keys()), ""), {})).get("metrics", {}) if isinstance(signs, dict) and signs else {}
    if mode == "diversity":
        v = m.get("diversity_1_minus_mean_cosine")
        return float("inf") if v is None else float(v)
    if mode == "similarity":
        v = m.get("intra_set_mean_pairwise_cosine_uni")
        return float("-inf") if v is None else -float(v)
    if mode == "stability":
        v = m.get("bootstrap_tile_jaccard_at_n_estimate")
        return float("-inf") if v is None else -float(v)
    if mode == "contrast":
        c = m.get("activation_contrast_reservoir_approx", {})
        v = c.get("contrast_top1pct_over_median_eps") if isinstance(c, dict) else None
        return float("-inf") if v is None else -float(v)
    return float(axis_entry.get("axis_index", 0))


def _iter_axis_entries(payload: dict[str, Any], *, sort_by: str) -> list[tuple[int, dict[str, Any]]]:
    axis_results = payload.get("axis_results", {})
    if not isinstance(axis_results, dict):
        raise SystemExit("Input JSON missing axis_results")
    items: list[tuple[int, dict[str, Any]]] = []
    for ax_str, entry in axis_results.items():
        try:
            ax = int(ax_str)
        except Exception:
            continue
        if isinstance(entry, dict):
            if "axis_index" not in entry:
                entry["axis_index"] = ax
            items.append((ax, entry))
    if sort_by == "axis_index":
        return sorted(items, key=lambda t: t[0])
    return sorted(items, key=lambda t: _axis_sort_score(t[1], sort_by))


def main() -> None:
    args = _build_argparser().parse_args()
    signs = _parse_signs(args.signs)
    selected_axes = _parse_axes_list(args.axes)

    payload = _load_json(args.input_json)
    if not isinstance(payload, dict):
        raise SystemExit("Input JSON root must be dict")

    axis_items = _iter_axis_entries(payload, sort_by=args.sort_by)
    if selected_axes:
        axis_items = [(ax, e) for ax, e in axis_items if ax in selected_axes]
    if args.max_axes and int(args.max_axes) > 0:
        axis_items = axis_items[: int(args.max_axes)]
    if not axis_items:
        raise SystemExit("No axis entries selected.")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wsi_cache = args.wsi_cache.expanduser().resolve()
    wsi_cache.mkdir(parents=True, exist_ok=True)

    open_slides: dict[str, openslide.OpenSlide] = {}
    wsi_path_cache: dict[str, Path] = {}
    failed_slides: dict[str, str] = {}

    export_meta: dict[str, Any] = {
        "input_json": str(args.input_json),
        "index_json": str(args.index_json),
        "gdc_client": str(args.gdc_client),
        "tile_size": int(args.tile_size),
        "vis_size": int(args.vis_size),
        "ncols": int(args.ncols),
        "notes": [
            "Tiles are cropped as 20x-equivalent FOV and saved at 256x256 by default.",
            "If level0 is ~40x (mpp<=0.26), crop 512x512 then resize to 256x256.",
            "If level0 is ~20x, crop 256x256 at level0.",
        ],
        "axes_exported": [],
        "failures": [],
    }

    try:
        for ax, axis_entry in axis_items:
            axis_out = out_dir / f"axis_{ax:06d}"
            axis_out.mkdir(parents=True, exist_ok=True)
            signs_obj = axis_entry.get("signs", {})
            axis_meta = {"axis_index": int(ax), "signs": {}}

            for sign in signs:
                sign_entry = signs_obj.get(sign)
                if not isinstance(sign_entry, dict):
                    continue
                records = sign_entry.get("top_tiles", [])
                if not isinstance(records, list):
                    continue
                if args.top_n and int(args.top_n) > 0:
                    records = records[: int(args.top_n)]

                sign_out = axis_out / sign
                tiles_out = sign_out / "tiles"
                tiles_out.mkdir(parents=True, exist_ok=True)

                saved_imgs: list[Image.Image] = []
                labels: list[str] = []
                exported: list[dict[str, Any]] = []

                for rank, rec in enumerate(records, start=1):
                    try:
                        slide_key = parse_slide_key(str(rec["h5_path"]))
                    except Exception:
                        # Fall back to provided slide_id if parse fails
                        slide_key = str(rec.get("slide_id", "unknown"))

                    if slide_key in failed_slides:
                        sign_meta_err = {
                            "rank": int(rank),
                            "slide_key": slide_key,
                            "h5_path": str(rec.get("h5_path")),
                            "tile_idx": int(rec.get("tile_idx", -1)),
                            "x": int(rec.get("x", 0)),
                            "y": int(rec.get("y", 0)),
                            "error": f"previous slide failure: {failed_slides[slide_key]}",
                        }
                        export_meta["failures"].append(sign_meta_err)
                        continue

                    try:
                        if slide_key not in wsi_path_cache:
                            last_err: Exception | None = None
                            for attempt in range(int(args.download_retries) + 1):
                                try:
                                    wsi_path_cache[slide_key] = download_one(
                                        slide_key,
                                        index_json=args.index_json,
                                        out_dir=wsi_cache,
                                        gdc_client=args.gdc_client,
                                        token_path=args.token,
                                        latest=True,
                                        overwrite=False,
                                        skip_if_exists=True,
                                        verbose=True,
                                    )
                                    last_err = None
                                    break
                                except Exception as exc:
                                    last_err = exc
                                    if attempt >= int(args.download_retries):
                                        break
                                    print(
                                        f"[warn] download failed for {slide_key} (attempt {attempt+1}/{int(args.download_retries)+1}): {exc}"
                                    )
                                    if float(args.retry_sleep_sec) > 0:
                                        time.sleep(float(args.retry_sleep_sec))
                            if last_err is not None:
                                raise last_err

                        if slide_key not in open_slides:
                            open_slides[slide_key] = openslide.OpenSlide(str(wsi_path_cache[slide_key]))

                        slide = open_slides[slide_key]
                        x, y = int(rec["x"]), int(rec["y"])
                        im, crop_px, mpp_x = _safe_rgb_tile_20x_equiv(slide, x=x, y=y, out_size=int(args.tile_size))
                    except Exception as exc:
                        failed_slides[slide_key] = str(exc)
                        failure = {
                            "rank": int(rank),
                            "slide_key": slide_key,
                            "h5_path": str(rec.get("h5_path")),
                            "tile_idx": int(rec.get("tile_idx", -1)),
                            "x": int(rec.get("x", 0)),
                            "y": int(rec.get("y", 0)),
                            "error": str(exc),
                        }
                        export_meta["failures"].append(failure)
                        print(f"[fail] axis={ax} sign={sign} slide={slide_key} rank={rank}: {exc}")
                        continue

                    if args.label_tiles:
                        draw = ImageDraw.Draw(im)
                        font = ImageFont.load_default()
                        txt = f"#{rank} {float(rec.get('score', 0.0)):.3f}"
                        draw.rectangle([0, 0, min(im.width - 1, 130), 14], fill=(255, 255, 255))
                        draw.text((2, 2), txt, fill=(0, 0, 0), font=font)

                    tile_name = _tile_filename(rank, rec)
                    tile_path = tiles_out / tile_name
                    if not (args.skip_if_exists and tile_path.exists() and not args.overwrite):
                        if tile_path.exists() and args.overwrite:
                            tile_path.unlink()
                        im.save(tile_path)

                    saved_imgs.append(im)
                    labels.append(f"#{rank} {float(rec.get('score', 0.0)):.2f}")
                    exported.append(
                        {
                            **rec,
                            "rank": int(rank),
                            "slide_key": slide_key,
                            "saved_tile": str(tile_path),
                            "crop_px_level0": int(crop_px),
                            "mpp_x": None if mpp_x is None else float(mpp_x),
                        }
                    )

                if saved_imgs:
                    sheet = _make_contact_sheet(saved_imgs, labels, ncols=int(args.ncols), vis_size=int(args.vis_size))
                    sheet_path = sign_out / "contact_sheet.png"
                    if sheet_path.exists() and args.overwrite:
                        sheet_path.unlink()
                    if not (args.skip_if_exists and sheet_path.exists() and not args.overwrite):
                        sheet.save(sheet_path)
                else:
                    sheet_path = sign_out / "contact_sheet.png"

                sign_meta = {
                    "num_tiles_exported": len(exported),
                    "num_tiles_requested": len(records),
                    "contact_sheet": str(sheet_path),
                    "metrics": sign_entry.get("metrics", {}),
                    "tiles": exported,
                }
                (sign_out / "top_tiles.json").write_text(json.dumps(sign_meta, indent=2))
                axis_meta["signs"][sign] = sign_meta

            (axis_out / "axis_meta.json").write_text(json.dumps(axis_meta, indent=2))
            export_meta["axes_exported"].append({"axis_index": ax, "path": str(axis_out)})
    finally:
        for s in open_slides.values():
            try:
                s.close()
            except Exception:
                pass

    (out_dir / "export_meta.json").write_text(json.dumps(export_meta, indent=2))
    print("Saved export root:", out_dir)
    print("Saved export meta:", out_dir / "export_meta.json")

    if not args.keep_wsi_cache:
        print(f"WSIs kept in cache: {wsi_cache} (set --keep-wsi-cache to silence this message).")


if __name__ == "__main__":
    main()
