#!/usr/bin/env python3
"""
Fetch TCGA slide-level labels from GDC API using file UUIDs from manifest_index.json.

Outputs:
- <out_dir>/tcga_gdc_slide_labels.tsv
- <out_dir>/tcga_gdc_slide_labels_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import requests


GDC_FILES_URL = "https://api.gdc.cancer.gov/files"


def chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


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
    )
    ap.add_argument("--batch_size", type=int, default=300)
    ap.add_argument("--sleep_sec", type=float, default=0.1)
    args = ap.parse_args()

    manifest = json.loads(args.manifest_index.read_text())
    file_to_slide: Dict[str, dict] = {}
    for slide_key, meta in manifest.items():
        gdc = meta.get("gdc") or []
        if not gdc:
            continue
        fid = (gdc[0] or {}).get("id")
        if not fid:
            continue
        file_to_slide[str(fid)] = {
            "slide_key": slide_key,
            "patient_id": str(meta.get("patient_id", "") or ""),
            "h5_path": str(meta.get("h5_path", "") or ""),
        }

    file_ids = sorted(file_to_slide.keys())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = args.out_dir / "tcga_gdc_slide_labels.tsv"
    out_summary = args.out_dir / "tcga_gdc_slide_labels_summary.json"

    fields = (
        "file_id,"
        "file_name,"
        "cases.case_id,"
        "cases.submitter_id,"
        "cases.project.project_id,"
        "cases.primary_site,"
        "cases.disease_type,"
        "cases.samples.sample_type,"
        "cases.samples.tumor_descriptor,"
        "cases.diagnoses.primary_diagnosis,"
        "cases.diagnoses.tumor_grade,"
        "cases.diagnoses.ajcc_pathologic_stage,"
        "cases.diagnoses.ajcc_pathologic_t,"
        "cases.diagnoses.ajcc_pathologic_n,"
        "cases.diagnoses.ajcc_pathologic_m,"
        "cases.diagnoses.days_to_death,"
        "cases.diagnoses.days_to_last_follow_up,"
        "cases.demographic.vital_status"
    )

    hits_by_fid: Dict[str, dict] = {}
    batches = chunked(file_ids, args.batch_size)
    for i, batch in enumerate(batches, start=1):
        payload = {
            "filters": {"op": "in", "content": {"field": "files.file_id", "value": batch}},
            "fields": fields,
            "format": "JSON",
            "size": len(batch),
        }
        r = requests.post(GDC_FILES_URL, json=payload, timeout=90)
        r.raise_for_status()
        obj = r.json()
        hits = obj.get("data", {}).get("hits", [])
        for h in hits:
            fid = str(h.get("file_id", ""))
            if fid:
                hits_by_fid[fid] = h
        print(f"[batch {i}/{len(batches)}] requested={len(batch)} returned={len(hits)}")
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    rows = []
    for fid in file_ids:
        base = file_to_slide[fid]
        h = hits_by_fid.get(fid, {})
        case = (h.get("cases") or [{}])[0]
        sample = (case.get("samples") or [{}])[0]
        diag = (case.get("diagnoses") or [{}])[0]
        demo = (case.get("demographic") or {})

        row = {
            "slide_key": base["slide_key"],
            "patient_id": base["patient_id"],
            "h5_path": base["h5_path"],
            "file_id": fid,
            "gdc_file_name": h.get("file_name", ""),
            "gdc_case_id": case.get("case_id", ""),
            "gdc_submitter_id": case.get("submitter_id", ""),
            "gdc_project_id": (case.get("project") or {}).get("project_id", ""),
            "gdc_primary_site": case.get("primary_site", ""),
            "gdc_disease_type": case.get("disease_type", ""),
            "gdc_sample_type": sample.get("sample_type", ""),
            "gdc_tumor_descriptor": sample.get("tumor_descriptor", ""),
            "gdc_primary_diagnosis": diag.get("primary_diagnosis", ""),
            "gdc_tumor_grade": diag.get("tumor_grade", ""),
            "gdc_ajcc_pathologic_stage": diag.get("ajcc_pathologic_stage", ""),
            "gdc_ajcc_pathologic_t": diag.get("ajcc_pathologic_t", ""),
            "gdc_ajcc_pathologic_n": diag.get("ajcc_pathologic_n", ""),
            "gdc_ajcc_pathologic_m": diag.get("ajcc_pathologic_m", ""),
            "gdc_days_to_death": diag.get("days_to_death", ""),
            "gdc_days_to_last_follow_up": diag.get("days_to_last_follow_up", ""),
            "gdc_vital_status": demo.get("vital_status", ""),
        }
        rows.append(row)

    header = list(rows[0].keys()) if rows else []
    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def non_empty_count(col: str) -> int:
        return sum(1 for r in rows if str(r.get(col, "") or "").strip() not in ("", "NA", "not reported", "Not Reported"))

    summary = {
        "slides_in_manifest": len(file_ids),
        "gdc_hit_count": len(hits_by_fid),
        "coverage": {
            "gdc_project_id": non_empty_count("gdc_project_id"),
            "gdc_primary_site": non_empty_count("gdc_primary_site"),
            "gdc_disease_type": non_empty_count("gdc_disease_type"),
            "gdc_sample_type": non_empty_count("gdc_sample_type"),
            "gdc_primary_diagnosis": non_empty_count("gdc_primary_diagnosis"),
            "gdc_tumor_grade": non_empty_count("gdc_tumor_grade"),
            "gdc_ajcc_pathologic_stage": non_empty_count("gdc_ajcc_pathologic_stage"),
            "gdc_vital_status": non_empty_count("gdc_vital_status"),
        },
        "top_counts": {
            "gdc_project_id": Counter(r["gdc_project_id"] for r in rows if r["gdc_project_id"]).most_common(20),
            "gdc_sample_type": Counter(r["gdc_sample_type"] for r in rows if r["gdc_sample_type"]).most_common(20),
            "gdc_primary_diagnosis": Counter(r["gdc_primary_diagnosis"] for r in rows if r["gdc_primary_diagnosis"]).most_common(20),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2))

    print(f"[ok] wrote {out_tsv}")
    print(f"[ok] wrote {out_summary}")
    print(f"[summary] manifest={len(file_ids)} gdc_hits={len(hits_by_fid)}")


if __name__ == "__main__":
    main()
