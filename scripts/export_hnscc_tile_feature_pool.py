#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.data.donor_pool import canonical_h5_path, choose_tile_indices, load_split_rows, make_contact_sheet
from wsi_cf.data.h5 import load_h5_features_coords
from wsi_cf.data.slides import crop_tile_rgb, find_slide_path, open_slide


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a balanced HNSCC donor pool of tile images + feature vectors. "
            "By default, picks 50 slides total, balanced by label, with one tile per slide."
        )
    )
    parser.add_argument("--split-tsv", type=Path, default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv"))
    parser.add_argument("--features-dir", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"))
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/HNSCC_slides"))
    parser.add_argument("--split-filter", type=str, default="all", choices=["all", "train", "test"])
    parser.add_argument("--total-tiles", type=int, default=50)
    parser.add_argument("--tiles-per-slide", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tile-size-20x", type=int, default=256)
    parser.add_argument("--out-tile-size", type=int, default=256)
    parser.add_argument("--tile-select", type=str, default="random", choices=["random", "first", "center"])
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/hnscc_tile_feature_pool")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rng = random.Random(int(args.seed))
    rows = load_split_rows(args.split_tsv, args.split_filter)
    by_label = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    per_label = int(args.total_tiles) // 2
    selected_rows: list[dict[str, object]] = []
    for label in (0, 1):
        label_rows = list(by_label[label])
        if len(label_rows) == 0:
            continue
        rng.shuffle(label_rows)
        exported = 0
        for row in label_rows:
            if exported >= per_label:
                break
            slide_key = str(row["slide_key"])
            h5_path = canonical_h5_path(args.features_dir, slide_key)
            slide_path = find_slide_path(args.slides_dir, slide_key)
            if not h5_path.exists() or slide_path is None:
                continue
            feats, coords = load_h5_features_coords(h5_path)
            if coords is None or feats.shape[0] == 0:
                continue
            tile_indices = choose_tile_indices(coords, mode=args.tile_select, k=int(args.tiles_per_slide), rng=rng)
            slide = open_slide(slide_path)
            try:
                for tile_index in tile_indices:
                    if exported >= per_label:
                        break
                    coord_x = int(coords[tile_index, 0])
                    coord_y = int(coords[tile_index, 1])
                    label_dir = args.out_dir / f"label_{label}_{'hpv_pos' if label == 1 else 'hpv_neg'}"
                    label_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"{slide_key}__tile_{int(tile_index):05d}__rank_01__x_{coord_x}__y_{coord_y}"
                    feat_path = label_dir / f"{stem}.npy"
                    img_path = label_dir / f"{stem}.png"
                    np.save(feat_path, np.asarray(feats[tile_index], dtype=np.float32))
                    tile_img, _ = crop_tile_rgb(
                        slide,
                        x=coord_x,
                        y=coord_y,
                        tile_size_20x=int(args.tile_size_20x),
                        out_tile_size=int(args.out_tile_size),
                    )
                    tile_img.save(img_path)
                    selected_rows.append(
                        {
                            "split": str(row["split"]),
                            "label": int(label),
                            "hpv_status": str(row["hpv_status"]),
                            "case_id": str(row["case_id"]),
                            "slide_key": slide_key,
                            "tile_index": int(tile_index),
                            "coord_x": coord_x,
                            "coord_y": coord_y,
                            "feature_path": str(feat_path),
                            "image_path": str(img_path),
                        }
                    )
                    exported += 1
            finally:
                slide.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "tile_pool.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = list(selected_rows[0].keys()) if selected_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(selected_rows)

    image_paths_by_label: dict[int, list[Path]] = defaultdict(list)
    for row in selected_rows:
        image_paths_by_label[int(row["label"])].append(Path(str(row["image_path"])))
    for label, paths in image_paths_by_label.items():
        sheet = make_contact_sheet(paths)
        sheet.save(args.out_dir / f"label_{label}_contact_sheet.png")

    write_json(
        args.out_dir / "tile_pool_summary.json",
        {
            "n_rows": len(selected_rows),
            "labels": {str(label): sum(1 for row in selected_rows if int(row["label"]) == label) for label in (0, 1)},
            "csv_path": str(csv_path),
        },
    )
    print(f"[ok] wrote {csv_path}")


if __name__ == "__main__":
    main()
