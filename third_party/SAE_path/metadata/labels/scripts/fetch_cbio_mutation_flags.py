#!/usr/bin/env python3
"""
Fetch TP53/KRAS mutation flags for cohort samples from cBioPortal TCGA PanCancer Atlas studies.

Outputs:
- metadata/labels/sources/raw/cbio_mutation_tp53_kras.tsv
- metadata/labels/sources/raw/cbio_mutation_tp53_kras_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

import requests


CBIO_BASE = "https://www.cbioportal.org/api"
TP53_ENTREZ = 7157
KRAS_ENTREZ = 3845


def cohort_samples_from_manifest(manifest_index: Path) -> Set[str]:
    obj = json.loads(manifest_index.read_text())
    samples: Set[str] = set()
    for slide_key in obj.keys():
        m = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2})", str(slide_key).upper())
        if m:
            samples.add(m.group(1))
    return samples


def get_json(url: str, timeout: float = 120.0) -> object:
    r = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_json(url: str, payload: dict, timeout: float = 120.0) -> object:
    r = requests.post(
        url,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest_index",
        type=Path,
        default=Path("metadata/indexes/manifest_index.json"),
    )
    ap.add_argument(
        "--out_tsv",
        type=Path,
        default=Path("metadata/labels/sources/raw/cbio_mutation_tp53_kras.tsv"),
    )
    ap.add_argument("--sleep_sec", type=float, default=0.05)
    args = ap.parse_args()

    cohort_samples = cohort_samples_from_manifest(args.manifest_index)
    if not cohort_samples:
        raise RuntimeError(f"No cohort samples parsed from {args.manifest_index}")

    studies = get_json(f"{CBIO_BASE}/studies?projection=SUMMARY&pageSize=5000&pageNumber=0")
    study_ids = sorted(
        s["studyId"]
        for s in studies
        if "tcga_pan_can_atlas_2018" in str(s.get("studyId", ""))
    )

    sequenced_samples: Set[str] = set()
    tp53_mut: Set[str] = set()
    kras_mut: Set[str] = set()
    sample_studies: Dict[str, Set[str]] = defaultdict(set)
    study_hits: Dict[str, int] = {}

    for sid in study_ids:
        sample_list_id = f"{sid}_sequenced"
        mut_profile_id = f"{sid}_mutations"

        sample_ids_url = f"{CBIO_BASE}/sample-lists/{sample_list_id}/sample-ids"
        try:
            sample_ids = get_json(sample_ids_url)
        except requests.HTTPError:
            continue

        sample_ids_in_cohort = [s for s in sample_ids if s in cohort_samples]
        if not sample_ids_in_cohort:
            continue

        for s in sample_ids_in_cohort:
            sequenced_samples.add(s)
            sample_studies[s].add(sid)

        payload = {"sampleListId": sample_list_id, "entrezGeneIds": [TP53_ENTREZ, KRAS_ENTREZ]}
        mutations_url = f"{CBIO_BASE}/molecular-profiles/{mut_profile_id}/mutations/fetch?projection=SUMMARY"
        try:
            mutations = post_json(mutations_url, payload)
        except requests.HTTPError:
            continue

        study_hits[sid] = len(mutations)
        for m in mutations:
            sample_id = str(m.get("sampleId", ""))
            if sample_id not in cohort_samples:
                continue
            gene_id = m.get("entrezGeneId")
            if gene_id == TP53_ENTREZ:
                tp53_mut.add(sample_id)
            elif gene_id == KRAS_ENTREZ:
                kras_mut.add(sample_id)

        print(
            f"[study] {sid}: cohort_sequenced={len(sample_ids_in_cohort)} mutation_records={len(mutations)}"
        )
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for sample_id in sorted(cohort_samples):
        has_data = 1 if sample_id in sequenced_samples else 0
        rows.append(
            {
                "sample_id": sample_id,
                "has_mutation_data": has_data,
                "tp53_mutated": 1 if sample_id in tp53_mut else 0 if has_data else "",
                "kras_mutated": 1 if sample_id in kras_mut else 0 if has_data else "",
                "source_studies": ",".join(sorted(sample_studies.get(sample_id, set()))),
            }
        )

    with args.out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_index": str(args.manifest_index),
        "cohort_samples_total": len(cohort_samples),
        "cohort_samples_with_mutation_data": len(sequenced_samples),
        "tp53_mutated_samples": len(tp53_mut),
        "kras_mutated_samples": len(kras_mut),
        "studies_queried": len(study_ids),
        "studies_with_hits": len(study_hits),
        "top_study_mutation_record_counts": Counter(study_hits).most_common(20),
    }
    out_summary = args.out_tsv.with_name(args.out_tsv.stem + "_summary.json")
    out_summary.write_text(json.dumps(summary, indent=2))

    print(f"[ok] wrote {args.out_tsv}")
    print(f"[ok] wrote {out_summary}")
    print(
        f"[summary] samples={len(cohort_samples)} with_mut_data={len(sequenced_samples)} "
        f"tp53_mut={len(tp53_mut)} kras_mut={len(kras_mut)}"
    )


if __name__ == "__main__":
    main()

