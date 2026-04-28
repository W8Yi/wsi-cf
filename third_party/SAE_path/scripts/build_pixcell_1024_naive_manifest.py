#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a naive PixCell-1024 4x4 steering manifest from 16 donor feature vectors. "
            "The output manifest can be passed to multidiff_img2img_wsi.py via --steer_manifest."
        )
    )
    ap.add_argument(
        "--tile-pool-csv",
        type=Path,
        default=REPO_ROOT / "outputs/hnscc_tile_feature_pool/tile_pool.csv",
        help="CSV created by export_hnscc_tile_feature_pool.py",
    )
    ap.add_argument(
        "--label",
        type=int,
        default=1,
        choices=[0, 1],
        help="Which label to draw donor features from.",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=16,
        help="Number of donor features to place into the manifest. For PixCell-1024, use 16.",
    )
    ap.add_argument(
        "--select",
        type=str,
        default="first",
        choices=["first", "random"],
        help="How to choose donor rows from the pool after filtering by label.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "outputs/pixcell_one_feature/naive_pixcell1024_manifest.json",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional CSV preview path. Defaults to <out-json>.csv",
    )
    return ap.parse_args()


def _load_rows(csv_path: Path, label: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_label = int(row.get("label", -1))
            except Exception:
                continue
            if row_label != int(label):
                continue
            feature_path = str(row.get("feature_path", "")).strip()
            if not feature_path:
                continue
            if not Path(feature_path).exists():
                continue
            rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.tile_pool_csv, args.label)
    if len(rows) < int(args.count):
        raise SystemExit(
            f"Not enough donor rows for label={args.label}. Need {args.count}, found {len(rows)} in {args.tile_pool_csv}"
        )

    if args.select == "random":
        rng = random.Random(int(args.seed))
        rows = rows.copy()
        rng.shuffle(rows)
    else:
        rows = list(rows)

    chosen = rows[: int(args.count)]
    manifest: list[dict[str, object]] = []
    preview: list[dict[str, object]] = []
    for idx, row in enumerate(chosen):
        gy = idx // 4
        gx = idx % 4
        feature_path = str(row["feature_path"])
        item = {
            "gx": int(gx),
            "gy": int(gy),
            "path": feature_path,
        }
        manifest.append(item)
        preview.append(
            {
                "gx": int(gx),
                "gy": int(gy),
                "label": int(row["label"]),
                "slide_key": str(row["slide_key"]),
                "tile_index": int(row["tile_index"]),
                "coord_x": int(row["coord_x"]),
                "coord_y": int(row["coord_y"]),
                "feature_path": feature_path,
                "image_path": str(row["image_path"]),
            }
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2))

    out_csv = args.out_csv if args.out_csv is not None else args.out_json.with_suffix(".csv")
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(preview[0].keys()))
        writer.writeheader()
        writer.writerows(preview)

    print(f"[ok] wrote {args.out_json}")
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
