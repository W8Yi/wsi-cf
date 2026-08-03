#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import requests


WSI_CF_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = WSI_CF_ROOT / "artifacts/prad_gleason_inputs/slide_labels.csv"
DEFAULT_OUT_DIR = WSI_CF_ROOT / "artifacts/prad_gleason_inputs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a GDC SVS download manifest for labeled TCGA-PRAD Gleason slides.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--project", default="TCGA-PRAD")
    parser.add_argument("--sample-type", default="Primary Tumor")
    parser.add_argument("--refresh-gdc", action="store_true")
    return parser


def read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], *, delimiter: str = ",") -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_gdc_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "filename", "md5", "size", "state"])
        for row in rows:
            writer.writerow([row["id"], row["filename"], row["md5"], row["size"], row["state"]])


def query_gdc_prad_slides(project: str, sample_type: str) -> list[dict[str, Any]]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": [sample_type]}},
        ],
    }
    response = requests.get(
        "https://api.gdc.cancer.gov/files",
        params={
            "filters": json.dumps(filters),
            "fields": (
                "file_id,file_name,file_size,md5sum,state,"
                "cases.submitter_id,cases.samples.sample_type,cases.project.project_id"
            ),
            "format": "JSON",
            "size": "5000",
        },
        timeout=120,
    )
    response.raise_for_status()
    out: list[dict[str, Any]] = []
    for hit in response.json()["data"]["hits"]:
        file_name = str(hit["file_name"])
        slide_key = file_name.split(".")[0]
        cases = hit.get("cases") or [{}]
        out.append(
            {
                "slide_key": slide_key,
                "case_id": str(cases[0].get("submitter_id", "")),
                "id": str(hit["file_id"]),
                "filename": file_name,
                "md5": str(hit.get("md5sum", "")),
                "size": int(hit.get("file_size", 0)),
                "state": str(hit.get("state", "")),
            }
        )
    return sorted(out, key=lambda row: (str(row["case_id"]), str(row["slide_key"])))


def load_or_query_inventory(cache_path: Path, *, project: str, sample_type: str, refresh: bool) -> list[dict[str, Any]]:
    if cache_path.exists() and not bool(refresh):
        return list(json.loads(cache_path.read_text())["files"])
    rows = query_gdc_prad_slides(project, sample_type)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"project": project, "sample_type": sample_type, "files": rows}, indent=2) + "\n")
    return rows


def main() -> None:
    args = build_arg_parser().parse_args()
    label_rows = read_label_rows(args.labels)
    needed_by_slide = {str(row["slide_key"]): dict(row) for row in label_rows}
    inventory = load_or_query_inventory(
        args.out_dir / "gdc_prad_slide_files.json",
        project=str(args.project),
        sample_type=str(args.sample_type),
        refresh=bool(args.refresh_gdc),
    )
    by_slide = {str(row["slide_key"]): row for row in inventory}
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for slide_key, label in sorted(needed_by_slide.items()):
        hit = by_slide.get(slide_key)
        if hit is None:
            missing.append(label)
            continue
        matched.append(
            {
                **hit,
                "gleason_score_label": label.get("gleason_score_label", ""),
                "grade_group": label.get("grade_group", ""),
                "low_high_grade": label.get("low_high_grade", ""),
                "split": label.get("split", ""),
            }
        )
    write_gdc_manifest(args.out_dir / "prad_all.gdc_manifest.tsv", matched)
    write_csv(args.out_dir / "prad_gdc_slide_files_matched.csv", matched)
    write_csv(args.out_dir / "prad_gdc_slide_files_missing.csv", missing)
    summary = {
        "label_slide_count": int(len(label_rows)),
        "gdc_slide_inventory_count": int(len(inventory)),
        "matched_manifest_count": int(len(matched)),
        "missing_manifest_count": int(len(missing)),
        "total_manifest_size_gb": float(sum(int(row.get("size") or 0) for row in matched) / 1e9),
        "manifest": str(args.out_dir / "prad_all.gdc_manifest.tsv"),
        "missing_csv": str(args.out_dir / "prad_gdc_slide_files_missing.csv"),
    }
    (args.out_dir / "prad_slide_download_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
