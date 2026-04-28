#!/usr/bin/env python3
"""
Build the slide list for all labeled TCGA-HNSC HPV WSIs.

Outputs:
- metadata/manifests/hnsc_hpv_wsi/slide_keys.txt
- metadata/manifests/hnsc_hpv_wsi/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labels_tsv",
        type=Path,
        default=Path("metadata/labels/master/slide_labels_master.tsv"),
        help="Master slide labels table.",
    )
    ap.add_argument(
        "--manifest_index",
        type=Path,
        default=Path("metadata/indexes/manifest_index.json"),
        help="Manifest index with gdc refs and sizes.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("metadata/manifests/hnsc_hpv_wsi"),
        help="Output directory.",
    )
    ap.add_argument(
        "--include_conflicts",
        action="store_true",
        help="Keep hpv_conflict=1 slides too.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    manifest_index = json.loads(args.manifest_index.read_text())

    slide_rows = []
    total_bytes = 0

    with args.labels_tsv.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["project_dir"] != "TCGA-HNSC":
                continue
            if row["hpv_status"] not in {"HPV+", "HPV-"}:
                continue
            if (not args.include_conflicts) and row["hpv_conflict"] == "1":
                continue

            slide_key = row["slide_key"]
            entry = manifest_index.get(slide_key)
            if entry is None:
                raise SystemExit(f"{slide_key} missing from {args.manifest_index}")

            gdc = entry.get("gdc")
            if isinstance(gdc, list):
                if not gdc:
                    raise SystemExit(f"{slide_key} has empty gdc list in {args.manifest_index}")
                gdc_ref = gdc[0]
            elif isinstance(gdc, dict):
                gdc_ref = gdc
            else:
                raise SystemExit(f"{slide_key} has invalid gdc entry in {args.manifest_index}")

            total_bytes += int(gdc_ref.get("size", 0))
            slide_rows.append(
                {
                    "slide_key": slide_key,
                    "case_id": row["case_id"],
                    "hpv_status": row["hpv_status"],
                    "gdc_uuid": str(gdc_ref.get("id", "")),
                    "gdc_filename": str(gdc_ref.get("filename", "")),
                    "gdc_size_bytes": int(gdc_ref.get("size", 0)),
                }
            )

    slide_rows.sort(key=lambda r: (r["hpv_status"], r["slide_key"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slide_keys_path = args.out_dir / "slide_keys.txt"
    slide_keys_path.write_text("".join(f"{row['slide_key']}\n" for row in slide_rows))

    summary = {
        "labels_tsv": str(args.labels_tsv),
        "manifest_index": str(args.manifest_index),
        "include_conflicts": bool(args.include_conflicts),
        "slides_total": len(slide_rows),
        "cases_total": len({row["case_id"] for row in slide_rows}),
        "slides_hpv_pos": sum(1 for row in slide_rows if row["hpv_status"] == "HPV+"),
        "slides_hpv_neg": sum(1 for row in slide_rows if row["hpv_status"] == "HPV-"),
        "total_size_bytes": total_bytes,
        "total_size_gb_decimal": total_bytes / 1e9,
        "total_size_gib_binary": total_bytes / (1024 ** 3),
        "slide_keys_txt": str(slide_keys_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[ok] wrote {slide_keys_path}")
    print(f"[ok] wrote {args.out_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
