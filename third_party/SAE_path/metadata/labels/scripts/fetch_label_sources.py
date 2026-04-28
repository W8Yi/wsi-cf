#!/usr/bin/env python3
"""
Download external raw label sources used by the label build pipeline.

Outputs are written under metadata/labels/sources/raw by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import requests


CBIO_URL = "https://raw.githubusercontent.com/GerkeLab/TCGAclinical/master/data/cBioportal_data.tsv"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data/{uuid}"


@dataclass(frozen=True)
class SourceSpec:
    filename: str
    url: str
    required: bool = True


SOURCES: Dict[str, SourceSpec] = {
    "cbioportal_data": SourceSpec(
        filename="cBioportal_data.tsv",
        url=CBIO_URL,
    ),
    "tcga_subtype_20170308": SourceSpec(
        filename="TCGASubtype.20170308.tsv",
        url=GDC_DATA_URL.format(uuid="0f31b768-7f67-4fc4-abc3-06ac5bd90bf0"),
    ),
    "viral_scores": SourceSpec(
        filename="viral.tsv",
        url=GDC_DATA_URL.format(uuid="a55229b3-da03-49fc-a310-9b1bf16b8512"),
    ),
    "absolute_mastercalls": SourceSpec(
        filename="TCGA_mastercalls.abs_tables_JSedit.fixed.txt",
        url=GDC_DATA_URL.format(uuid="4f277128-f793-4354-a13d-30cc7fe9f6b5"),
    ),
    "absolute_scores": SourceSpec(
        filename="ABSOLUTE_scores.tsv",
        url=GDC_DATA_URL.format(uuid="0e8831f4-dd7e-4673-8624-b4519c2e0d65"),
    ),
    "immune_subtype_mclust": SourceSpec(
        filename="immune_subtype_mclust.tsv.gz",
        url="https://raw.githubusercontent.com/CRI-iAtlas/ImmuneSubtypeClassifier/master/inst/extdata/five_signature_mclust_ensemble_results.tsv.gz",
    ),
}


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def download(url: str, timeout_sec: float) -> bytes:
    r = requests.get(url, timeout=timeout_sec)
    r.raise_for_status()
    return r.content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("metadata/labels/sources/raw"),
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download even if file exists",
    )
    ap.add_argument(
        "--timeout_sec",
        type=float,
        default=120.0,
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(args.out_dir),
        "sources": {},
    }

    for key, spec in SOURCES.items():
        out_path = args.out_dir / spec.filename
        if out_path.exists() and not args.refresh:
            data = out_path.read_bytes()
            status = "cached"
        else:
            print(f"[download] {key} -> {out_path}")
            data = download(spec.url, timeout_sec=args.timeout_sec)
            out_path.write_bytes(data)
            status = "downloaded"

        info = {
            "filename": spec.filename,
            "url": spec.url,
            "status": status,
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
        }
        manifest["sources"][key] = info

    out_manifest = args.out_dir / "source_download_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[ok] wrote {out_manifest}")


if __name__ == "__main__":
    main()
