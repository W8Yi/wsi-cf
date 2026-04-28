#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from e


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export a balanced HNSCC donor pool of tile images + feature vectors. "
            "By default, picks 50 slides total, balanced by label, with one tile per slide."
        )
    )
    ap.add_argument(
        "--split-tsv",
        type=Path,
        default=REPO_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.tsv",
        help="TSV containing slide_key and label columns.",
    )
    ap.add_argument(
        "--features-dir",
        type=Path,
        default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"),
        help="Directory containing <slide_key>.h5 UNI2 feature files.",
    )
    ap.add_argument(
        "--slides-dir",
        type=Path,
        default=Path("/common/users/wq50/HNSCC/HNSCC_slides"),
        help="Directory containing local HNSCC slide .svs files.",
    )
    ap.add_argument(
        "--split-filter",
        type=str,
        default="all",
        choices=["all", "train", "test"],
        help="Which rows from split-tsv are eligible.",
    )
    ap.add_argument("--total-tiles", type=int, default=50, help="Total exported tiles across both labels.")
    ap.add_argument(
        "--tiles-per-slide",
        type=int,
        default=1,
        help="Tiles exported per selected slide. Keep 1 if you want every tile from a different slide.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tile-size-20x", type=int, default=256, help="Tile crop size in 20x-equivalent pixels.")
    ap.add_argument("--out-tile-size", type=int, default=256, help="Saved PNG size.")
    ap.add_argument(
        "--tile-select",
        type=str,
        default="random",
        choices=["random", "first", "center"],
        help="How to pick the tile index within each selected slide.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/hnscc_tile_feature_pool",
    )
    return ap.parse_args()


def _load_rows(split_tsv: Path, split_filter: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with split_tsv.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if split_filter != "all" and str(row.get("split", "")) != split_filter:
                continue
            try:
                label = int(row.get("label", -1))
            except Exception:
                continue
            if label not in (0, 1):
                continue
            slide_key = str(row.get("slide_key", "")).strip()
            if not slide_key:
                continue
            rows.append(
                {
                    "split": str(row.get("split", "")),
                    "label": int(label),
                    "hpv_status": str(row.get("hpv_status", "")),
                    "case_id": str(row.get("case_id", "")),
                    "slide_key": slide_key,
                }
            )
    rows.sort(key=lambda r: (int(r["label"]), str(r["slide_key"])))
    return rows


def _canonical_h5_path(features_dir: Path, slide_key: str) -> Path:
    return features_dir / f"{slide_key}.h5"


def _find_slide_path(slides_dir: Path, slide_key: str) -> Path | None:
    matches = sorted(slides_dir.glob(f"{slide_key}*.svs"))
    if matches:
        return matches[0]
    return None


def _read_h5_features_coords(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]
        coords = f["coords"]
        if feats.ndim == 2:
            x = feats[:]
        elif feats.ndim == 3 and int(feats.shape[0]) == 1:
            x = feats[0]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")
        if coords.ndim == 2 and int(coords.shape[1]) == 2:
            c = coords[:]
        elif coords.ndim == 3 and int(coords.shape[0]) == 1 and int(coords.shape[2]) == 2:
            c = coords[0]
        else:
            raise RuntimeError(f"{h5_path}: unsupported coords shape {tuple(coords.shape)}")
    x = np.asarray(x, dtype=np.float32)
    c = np.asarray(c, dtype=np.int64)
    if x.shape[0] != c.shape[0]:
        raise RuntimeError(f"{h5_path}: features/coords row mismatch {x.shape} vs {c.shape}")
    return x, c


def _infer_objective_power(slide: "openslide.OpenSlide") -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            try:
                v = float(props.get(key))
            except Exception:
                v = -1.0
            if v > 0:
                return v
    try:
        mpp_x = float(props.get("openslide.mpp-x", -1.0))
    except Exception:
        mpp_x = -1.0
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def _level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(float(tile_size_20x) * (float(objective_power) / 20.0))))


def _crop_tile_rgb(
    slide: "openslide.OpenSlide",
    *,
    x: int,
    y: int,
    tile_size_20x: int,
    out_tile_size: int,
) -> tuple[Image.Image, int]:
    crop_px = _level0_tile_size(tile_size_20x, _infer_objective_power(slide))
    rgba = slide.read_region((int(x), int(y)), 0, (int(crop_px), int(crop_px))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    if crop_px != int(out_tile_size):
        rgb = rgb.resize((int(out_tile_size), int(out_tile_size)), resample=Image.BILINEAR)
    return rgb, int(crop_px)


def _choose_tile_indices(coords: np.ndarray, *, mode: str, k: int, rng: random.Random) -> list[int]:
    n = int(coords.shape[0])
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if mode == "first":
        return list(range(k))
    if mode == "random":
        return rng.sample(range(n), k=k)
    if mode == "center":
        center = coords.mean(axis=0, keepdims=True)
        d2 = ((coords.astype(np.float64) - center) ** 2).sum(axis=1)
        order = np.argsort(d2)
        return [int(i) for i in order[:k].tolist()]
    raise ValueError(f"Unsupported tile select mode: {mode}")


def _make_contact_sheet(paths: list[Path], thumb_size: int = 128, ncols: int = 5, pad: int = 4) -> Image.Image:
    if not paths:
        return Image.new("RGB", (thumb_size, thumb_size), (245, 245, 245))
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(paths) / float(ncols)))
    w = pad + ncols * (thumb_size + pad)
    h = pad + nrows * (thumb_size + pad)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(paths):
        r = i // ncols
        c = i % ncols
        x0 = pad + c * (thumb_size + pad)
        y0 = pad + r * (thumb_size + pad)
        img = Image.open(p).convert("RGB").resize((thumb_size, thumb_size), resample=Image.BILINEAR)
        canvas.paste(img, (x0, y0))
        draw.rectangle([x0, y0, x0 + thumb_size - 1, y0 + thumb_size - 1], outline=(180, 180, 180), width=1)
    return canvas


def _label_slug(label: int) -> str:
    return "hpv_pos" if int(label) == 1 else "hpv_neg"


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))

    rows = _load_rows(args.split_tsv, args.split_filter)
    if not rows:
        raise SystemExit(f"No eligible rows found in {args.split_tsv} with split_filter={args.split_filter}")

    usable: list[dict[str, Any]] = []
    for row in rows:
        slide_key = str(row["slide_key"])
        h5_path = _canonical_h5_path(args.features_dir, slide_key)
        slide_path = _find_slide_path(args.slides_dir, slide_key)
        if not h5_path.exists() or slide_path is None or not slide_path.exists():
            continue
        usable.append(dict(row, h5_path=str(h5_path), slide_path=str(slide_path)))

    by_label = {
        0: [r for r in usable if int(r["label"]) == 0],
        1: [r for r in usable if int(r["label"]) == 1],
    }
    if not by_label[0] or not by_label[1]:
        raise SystemExit(
            f"Need both labels after local path filtering. Found counts: neg={len(by_label[0])}, pos={len(by_label[1])}"
        )

    n_total = int(args.total_tiles)
    if n_total <= 0:
        raise SystemExit("--total-tiles must be > 0")
    tiles_per_slide = int(args.tiles_per_slide)
    if tiles_per_slide <= 0:
        raise SystemExit("--tiles-per-slide must be > 0")

    target_pos = int(math.ceil(n_total / 2.0))
    target_neg = int(n_total - target_pos)
    need_slides_pos = int(math.ceil(target_pos / float(tiles_per_slide)))
    need_slides_neg = int(math.ceil(target_neg / float(tiles_per_slide)))
    if len(by_label[1]) < need_slides_pos or len(by_label[0]) < need_slides_neg:
        raise SystemExit(
            "Not enough different slides for requested pool. "
            f"Need neg={need_slides_neg}, pos={need_slides_pos}; "
            f"have neg={len(by_label[0])}, pos={len(by_label[1])}."
        )

    rng.shuffle(by_label[0])
    rng.shuffle(by_label[1])
    selected_rows = by_label[0][:need_slides_neg] + by_label[1][:need_slides_pos]
    selected_rows.sort(key=lambda r: (int(r["label"]), str(r["slide_key"])))

    export_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        slide_key = str(row["slide_key"])
        label = int(row["label"])
        h5_path = Path(str(row["h5_path"]))
        slide_path = Path(str(row["slide_path"]))
        x, coords = _read_h5_features_coords(h5_path)
        tile_indices = _choose_tile_indices(coords, mode=str(args.tile_select), k=tiles_per_slide, rng=rng)
        if not tile_indices:
            continue
        slide = openslide.OpenSlide(str(slide_path))
        try:
            for local_rank, tile_idx in enumerate(tile_indices, start=1):
                feat = np.asarray(x[int(tile_idx)], dtype=np.float32)
                coord_x = int(coords[int(tile_idx), 0])
                coord_y = int(coords[int(tile_idx), 1])
                tile_img, crop_px_level0 = _crop_tile_rgb(
                    slide,
                    x=coord_x,
                    y=coord_y,
                    tile_size_20x=int(args.tile_size_20x),
                    out_tile_size=int(args.out_tile_size),
                )
                label_dir = args.out_dir / f"label_{label}_{_label_slug(label)}"
                label_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{slide_key}__tile_{int(tile_idx):05d}__rank_{local_rank:02d}__x_{coord_x}__y_{coord_y}"
                img_path = label_dir / f"{stem}.png"
                feat_path = label_dir / f"{stem}.npy"
                tile_img.save(img_path)
                np.save(feat_path, feat)
                export_rows.append(
                    {
                        "label": int(label),
                        "label_name": _label_slug(label),
                        "split": str(row["split"]),
                        "hpv_status": str(row["hpv_status"]),
                        "case_id": str(row["case_id"]),
                        "slide_key": slide_key,
                        "slide_path": str(slide_path),
                        "h5_path": str(h5_path),
                        "tile_index": int(tile_idx),
                        "coord_x": int(coord_x),
                        "coord_y": int(coord_y),
                        "crop_px_level0": int(crop_px_level0),
                        "feature_dim": int(feat.shape[0]),
                        "image_path": str(img_path),
                        "feature_path": str(feat_path),
                    }
                )
        finally:
            slide.close()

    export_rows.sort(key=lambda r: (int(r["label"]), str(r["slide_key"]), int(r["tile_index"])))

    if len(export_rows) > n_total:
        # Trim deterministically in case ceil-based per-label allocation overran by 1.
        neg_rows = [r for r in export_rows if int(r["label"]) == 0][:target_neg]
        pos_rows = [r for r in export_rows if int(r["label"]) == 1][:target_pos]
        export_rows = neg_rows + pos_rows
        export_rows.sort(key=lambda r: (int(r["label"]), str(r["slide_key"]), int(r["tile_index"])))

    csv_path = args.out_dir / "tile_pool.csv"
    json_path = args.out_dir / "tile_pool_summary.json"
    if export_rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(export_rows[0].keys()))
            writer.writeheader()
            writer.writerows(export_rows)
    else:
        csv_path.write_text("")

    for label in (0, 1):
        label_rows = [r for r in export_rows if int(r["label"]) == label]
        img_paths = [Path(str(r["image_path"])) for r in label_rows]
        sheet = _make_contact_sheet(img_paths, thumb_size=128, ncols=5, pad=4)
        sheet.save(args.out_dir / f"contact_sheet__label_{label}_{_label_slug(label)}.png")

    summary = {
        "split_tsv": str(args.split_tsv),
        "features_dir": str(args.features_dir),
        "slides_dir": str(args.slides_dir),
        "split_filter": str(args.split_filter),
        "seed": int(args.seed),
        "total_tiles_requested": int(args.total_tiles),
        "tiles_per_slide": int(args.tiles_per_slide),
        "tile_select": str(args.tile_select),
        "tile_size_20x": int(args.tile_size_20x),
        "out_tile_size": int(args.out_tile_size),
        "usable_slide_counts": {
            "hpv_neg": int(len(by_label[0])),
            "hpv_pos": int(len(by_label[1])),
        },
        "exported_tile_counts": {
            "total": int(len(export_rows)),
            "hpv_neg": int(sum(1 for r in export_rows if int(r["label"]) == 0)),
            "hpv_pos": int(sum(1 for r in export_rows if int(r["label"]) == 1)),
        },
        "exported_slide_counts": {
            "total": int(len(set(str(r["slide_key"]) for r in export_rows))),
            "hpv_neg": int(len(set(str(r["slide_key"]) for r in export_rows if int(r["label"]) == 0))),
            "hpv_pos": int(len(set(str(r["slide_key"]) for r in export_rows if int(r["label"]) == 1))),
        },
        "tile_pool_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {json_path}")
    for label in (0, 1):
        print(f"[ok] wrote {args.out_dir / f'contact_sheet__label_{label}_{_label_slug(label)}.png'}")


if __name__ == "__main__":
    main()
