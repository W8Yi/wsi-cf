#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import requests


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a GDC SVS download manifest for slide keys listed in a classifier task_manifest.csv. "
            "Useful when feature bags exist but local source SVS files are missing."
        )
    )
    parser.add_argument("--task-manifest-csv", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--gdc-project", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--label", default="")
    parser.add_argument("--sample-type", default="", help="Optional GDC sample type filter, e.g. 'Primary Tumor'.")
    parser.add_argument("--label-column", default="label_name")
    parser.add_argument("--slide-key-column", default="slide_key")
    parser.add_argument("--size", type=int, default=5000)
    return parser


def read_task_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def query_gdc_slides(*, gdc_project: str, sample_type: str, size: int) -> list[dict[str, Any]]:
    filters: dict[str, Any] = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [gdc_project]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
        ],
    }
    if sample_type:
        filters["content"].append(
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": [sample_type]}}
        )
    response = requests.get(
        "https://api.gdc.cancer.gov/files",
        params={
            "filters": json.dumps(filters),
            "fields": (
                "file_id,file_name,file_size,md5sum,state,"
                "cases.submitter_id,cases.samples.sample_type,cases.project.project_id"
            ),
            "format": "JSON",
            "size": str(int(size)),
        },
        timeout=120,
    )
    response.raise_for_status()
    return list(response.json()["data"]["hits"])


def slide_key_from_filename(filename: str) -> str:
    return str(filename).split(".")[0]


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "filename", "md5", "size", "state"])
        for row in rows:
            writer.writerow([row["file_id"], row["file_name"], row.get("md5sum", ""), row.get("file_size", 0), row.get("state", "")])


def main() -> None:
    args = build_arg_parser().parse_args()
    task_rows = read_task_rows(args.task_manifest_csv)
    selected = [
        row
        for row in task_rows
        if (not args.split or str(row.get("split", "")) == str(args.split))
        and (not args.label or str(row.get(args.label_column, "")) == str(args.label))
    ]
    wanted = {str(row.get(args.slide_key_column, "")).strip() for row in selected if str(row.get(args.slide_key_column, "")).strip()}
    if not wanted:
        raise ValueError(
            f"No slide keys selected from {args.task_manifest_csv} with split={args.split!r}, "
            f"label={args.label!r}, label_column={args.label_column!r}"
        )
    gdc_rows = query_gdc_slides(gdc_project=str(args.gdc_project), sample_type=str(args.sample_type), size=int(args.size))
    by_key = {slide_key_from_filename(str(row["file_name"])): row for row in gdc_rows}
    matched = [by_key[key] for key in sorted(wanted) if key in by_key]
    missing = sorted(wanted - set(by_key))
    write_manifest(args.out_manifest, matched)
    payload = {
        "task_manifest_csv": str(args.task_manifest_csv),
        "gdc_project": str(args.gdc_project),
        "split": str(args.split),
        "label": str(args.label),
        "sample_type": str(args.sample_type),
        "n_wanted": len(wanted),
        "n_matched": len(matched),
        "n_missing": len(missing),
        "missing_slide_keys": missing,
        "out_manifest": str(args.out_manifest),
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

