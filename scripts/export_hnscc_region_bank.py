#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import (
    assign_region_roles,
    export_region_bundle,
    load_local_labeled_rows,
    make_region_bank_summary,
    sample_random_tissue_region,
    save_label_contact_sheets,
    validate_balanced_request,
    write_region_bank_csv,
)
from wsi_cf.data.slides import open_slide, read_region_rgb
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a balanced bank of 1024x1024 HNSCC regions with aligned real images and "
            "PixCell-ready 4x4 UNI2 feature grids."
        )
    )
    parser.add_argument("--split-tsv", type=Path, default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv"))
    parser.add_argument("--features-dir", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"))
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/HNSCC_slides"))
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/hnscc_region_bank_1024")
    parser.add_argument("--region-size", type=int, default=1024)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--selection-mode", type=str, default="random_tissue", choices=["random_tissue"])
    parser.add_argument("--min-tissue", type=float, default=0.35)
    parser.add_argument("--max-region-tries", type=int, default=64)
    parser.add_argument("--regions-total", type=int, default=40)
    parser.add_argument("--regions-per-slide", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(int(args.seed))
    rng = random.Random(int(args.seed))
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    eligible_rows = load_local_labeled_rows(
        split_tsv=args.split_tsv,
        slides_dir=args.slides_dir,
        features_dir=args.features_dir,
    )
    validate_balanced_request(
        eligible_rows=eligible_rows,
        total_regions=int(args.regions_total),
        regions_per_slide=int(args.regions_per_slide),
    )

    by_label: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for row in eligible_rows:
        by_label[int(row["label"])].append(dict(row))
    for label in (0, 1):
        rng.shuffle(by_label[label])

    need_per_label = int(args.regions_total) // 2
    selected_rows = {0: by_label[0][:need_per_label], 1: by_label[1][:need_per_label]}

    uni_model, uni_transform = load_uni2(device=device)

    def build_feature_grid(region_img):
        z_grid = build_uni_grid_from_image(
            region_img,
            uni_model=uni_model,
            uni_transform=uni_transform,
            grid_step_px=int(args.grid_step_px),
            device=device,
            out_dtype=dtype,
        )
        return z_grid.detach().float().cpu().numpy().astype(np.float32, copy=False)

    exported_rows: list[dict[str, object]] = []
    for label in (0, 1):
        for row in selected_rows[label]:
            slide = open_slide(Path(str(row["slide_path"])))
            try:
                region_x, region_y, tissue_score = sample_random_tissue_region(
                    slide,
                    region_size=int(args.region_size),
                    n_tries=int(args.max_region_tries),
                    min_tissue=float(args.min_tissue),
                    rng=rng,
                )
                region_img = read_region_rgb(slide, int(region_x), int(region_y), int(args.region_size), int(args.region_size))
            finally:
                slide.close()

            region_id = f"{row['slide_key']}__x_{int(region_x)}__y_{int(region_y)}"
            exported = export_region_bundle(
                region_img=region_img,
                out_dir=args.out_dir,
                region_id=region_id,
                row=row,
                region_x=int(region_x),
                region_y=int(region_y),
                region_size=int(args.region_size),
                grid_step_px=int(args.grid_step_px),
                tissue_score=float(tissue_score),
                seed=int(args.seed),
                build_feature_grid=build_feature_grid,
            )
            exported_rows.append(exported)

    exported_rows.sort(key=lambda r: (int(r["label"]), str(r["slide_key"]), str(r["region_id"])))
    out_csv = args.out_dir / "region_bank.csv"
    write_region_bank_csv(out_csv, exported_rows)

    roles = assign_region_roles(exported_rows, sources_per_label=10, donors_per_label=10)
    out_roles_csv = args.out_dir / "region_roles.csv"
    write_region_bank_csv(out_roles_csv, roles)

    save_label_contact_sheets(rows=exported_rows, out_dir=args.out_dir)
    summary = make_region_bank_summary(rows=exported_rows, roles=roles, out_csv=out_csv, out_roles_csv=out_roles_csv)
    write_json(args.out_dir / "region_bank_summary.json", summary)
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_roles_csv}")
    print(f"[ok] wrote {args.out_dir / 'region_bank_summary.json'}")


if __name__ == "__main__":
    main()
