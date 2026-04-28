#!/usr/bin/env python3
"""
Build a slide-level label catalog for TCGA WSI feature files.

Data sources:
1) manifest_index.json (local slide inventory)
2) cBioPortal Pan-Cancer clinical table (downloaded from GitHub)

Outputs:
- <raw_dir>/cBioportal_data.tsv
- <out_dir>/tcga_slide_label_catalog.tsv
- <out_dir>/tcga_slide_label_catalog_summary.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen


DEFAULT_CBIO_URL = (
    "https://raw.githubusercontent.com/GerkeLab/TCGAclinical/master/data/cBioportal_data.tsv"
)


def parse_slide_key_from_h5_path(h5_path: str) -> Optional[str]:
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]-[0-9]{2}-DX[0-9]+)", str(h5_path).upper())
    return m.group(1) if m else None


def sample_prefix_from_slide_key(slide_key: str) -> Optional[str]:
    m = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})[A-Z]-", str(slide_key).upper())
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def sample_type_from_slide_key(slide_key: str) -> Optional[str]:
    m = re.match(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-([0-9]{2})[A-Z]-", str(slide_key).upper())
    return m.group(1) if m else None


def project_from_h5_path(h5_path: str) -> Optional[str]:
    m = re.search(r"/extracted_features/([^/]+)/", str(h5_path))
    return m.group(1) if m else None


def read_manifest_index(path: Path) -> Dict[str, dict]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def download_if_needed(url: str, out_path: Path, refresh: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not refresh:
        return
    with urlopen(url) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def load_cbio_rows(tsv_path: Path) -> List[dict]:
    text = tsv_path.read_text(errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    return list(reader)


def study_priority(study_id: str) -> int:
    sid = (study_id or "").lower()
    if "pan_can_atlas" in sid:
        return 0
    if "tcga" in sid:
        return 1
    return 2


def choose_best_cbio_row(rows: List[dict]) -> dict:
    if len(rows) == 1:
        return rows[0]
    return sorted(
        rows,
        key=lambda r: (
            study_priority(r.get("Study ID", "")),
            len(r.get("Study ID", "")),
            r.get("Study ID", ""),
        ),
    )[0]


def build_cbio_index(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        sample_id = (r.get("Sample ID") or "").upper()
        m = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2})", sample_id)
        if not m:
            continue
        out[m.group(1)].append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest_index",
        type=Path,
        default=Path("metadata/indexes/manifest_index.json"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("metadata/labels/sources/intermediate"),
        help="Directory for intermediate label tables",
    )
    ap.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("metadata/labels/sources/raw"),
        help="Directory for raw source snapshots",
    )
    ap.add_argument(
        "--cbio_url",
        type=str,
        default=DEFAULT_CBIO_URL,
    )
    ap.add_argument(
        "--refresh_cbio_download",
        action="store_true",
    )
    args = ap.parse_args()

    manifest = read_manifest_index(args.manifest_index)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    cbio_tsv = args.raw_dir / "cBioportal_data.tsv"
    print(f"[1/3] Downloading cBioPortal table -> {cbio_tsv}")
    download_if_needed(args.cbio_url, cbio_tsv, refresh=args.refresh_cbio_download)

    print("[2/3] Parsing cBioPortal rows")
    cbio_rows = load_cbio_rows(cbio_tsv)
    cbio_by_prefix = build_cbio_index(cbio_rows)

    print("[3/3] Building slide label catalog")
    out_tsv = args.out_dir / "tcga_slide_label_catalog.tsv"
    fieldnames = [
        "slide_key",
        "patient_id",
        "h5_path",
        "project_dir",
        "sample_type_code",
        "sample_prefix",
        # cBioPortal selected fields
        "cbio_study_id",
        "cbio_patient_id",
        "cbio_sample_id",
        "cbio_tcga_pancan_acronym",
        "cbio_cancer_type",
        "cbio_cancer_type_detailed",
        "cbio_subtype",
        "cbio_primary_diagnosis",
        "cbio_histologic_grade",
        "cbio_ajcc_stage",
        "cbio_os_months",
        "cbio_os_status",
        "cbio_pfs_months",
        "cbio_pfs_status",
    ]

    n_manifest = 0
    n_cbio = 0
    n_subtype = 0
    c_project = Counter()
    c_sample_type = Counter()
    c_cbio_subtype = Counter()

    with out_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for slide_key, meta in manifest.items():
            n_manifest += 1
            h5_path = str(meta.get("h5_path", ""))
            patient_id = str(meta.get("patient_id", "") or "")
            project_dir = project_from_h5_path(h5_path) or ""
            sample_code = sample_type_from_slide_key(slide_key) or ""
            sample_prefix = sample_prefix_from_slide_key(slide_key) or ""

            c_project[project_dir] += 1
            c_sample_type[sample_code] += 1

            best_cbio = None
            if sample_prefix and sample_prefix in cbio_by_prefix:
                best_cbio = choose_best_cbio_row(cbio_by_prefix[sample_prefix])
                n_cbio += 1
                subtype = (best_cbio.get("Subtype") or "").strip()
                if subtype and subtype != "NA":
                    n_subtype += 1
                    c_cbio_subtype[subtype] += 1

            row = {
                "slide_key": slide_key,
                "patient_id": patient_id,
                "h5_path": h5_path,
                "project_dir": project_dir,
                "sample_type_code": sample_code,
                "sample_prefix": sample_prefix,
                "cbio_study_id": (best_cbio or {}).get("Study ID", ""),
                "cbio_patient_id": (best_cbio or {}).get("Patient ID", ""),
                "cbio_sample_id": (best_cbio or {}).get("Sample ID", ""),
                "cbio_tcga_pancan_acronym": (best_cbio or {}).get("TCGA PanCanAtlas Cancer Type Acronym", ""),
                "cbio_cancer_type": (best_cbio or {}).get("Cancer Type", ""),
                "cbio_cancer_type_detailed": (best_cbio or {}).get("Cancer Type Detailed", ""),
                "cbio_subtype": (best_cbio or {}).get("Subtype", ""),
                "cbio_primary_diagnosis": (best_cbio or {}).get("Diagnosis", "") or (best_cbio or {}).get("Primary Diagnosis", "") or "",
                "cbio_histologic_grade": (best_cbio or {}).get("Neoplasm Histologic Grade", ""),
                "cbio_ajcc_stage": (best_cbio or {}).get("Neoplasm Disease Stage American Joint Committee on Cancer Code", ""),
                "cbio_os_months": (best_cbio or {}).get("Overall Survival (Months)", ""),
                "cbio_os_status": (best_cbio or {}).get("Overall Survival Status", ""),
                "cbio_pfs_months": (best_cbio or {}).get("Progression Free Survival (Months)", ""),
                "cbio_pfs_status": (best_cbio or {}).get("Progression Free Status", ""),
            }
            writer.writerow(row)

    summary = {
        "manifest_slides": n_manifest,
        "cbio_match_count": n_cbio,
        "cbio_match_rate": (float(n_cbio) / float(n_manifest)) if n_manifest else 0.0,
        "cbio_subtype_nonempty_count": n_subtype,
        "n_projects": len(c_project),
        "project_counts_top20": c_project.most_common(20),
        "sample_type_code_counts": dict(c_sample_type),
        "cbio_subtype_top30": c_cbio_subtype.most_common(30),
        "sources": {
            "manifest_index": str(args.manifest_index),
            "cbio_url": args.cbio_url,
            "cbio_tsv": str(cbio_tsv),
        },
    }

    out_summary = args.out_dir / "tcga_slide_label_catalog_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))

    print(f"[ok] wrote {out_tsv}")
    print(f"[ok] wrote {out_summary}")
    print(
        f"[summary] slides={n_manifest} cbio_matched={n_cbio} "
        f"({summary['cbio_match_rate']*100:.1f}%) subtype_nonempty={n_subtype}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
