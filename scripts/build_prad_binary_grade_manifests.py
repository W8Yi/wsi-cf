#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


WSI_CF_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSIFIER_MANIFEST = WSI_CF_ROOT / "artifacts/classifier_training/prad_low_vs_high_grade/task_manifest.csv"
DEFAULT_LABELS = WSI_CF_ROOT / "artifacts/prad_gleason_inputs/slide_labels.csv"
DEFAULT_OUT_DIR = WSI_CF_ROOT / "artifacts/prad_gleason_inputs/binary_grade_manifests"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build grade-stratified task manifests for binary PRAD steering. "
            "The binary low/high classifier labels are preserved while source slides are filtered by grade group."
        )
    )
    parser.add_argument("--classifier-task-manifest", type=Path, default=DEFAULT_CLASSIFIER_MANIFEST)
    parser.add_argument("--slide-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--grade-column", default="grade_group")
    parser.add_argument("--grades", default="GG1,GG2,GG3,GG4,GG5")
    parser.add_argument("--require-existing-h5", action=argparse.BooleanOptionalAction, default=True)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_csv_tokens(text: str) -> list[str]:
    return [token.strip() for token in str(text).split(",") if token.strip()]


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_rows = read_csv(args.classifier_task_manifest)
    label_rows = read_csv(args.slide_labels)
    if not manifest_rows:
        raise ValueError(f"No rows in classifier task manifest: {args.classifier_task_manifest}")
    if not label_rows:
        raise ValueError(f"No rows in PRAD slide labels: {args.slide_labels}")
    if args.grade_column not in label_rows[0]:
        raise ValueError(f"{args.slide_labels}: missing grade column {args.grade_column!r}")

    labels_by_slide = {str(row["slide_key"]): row for row in label_rows}
    manifest_fields = list(manifest_rows[0].keys())
    extra_fields = [
        "gleason_primary",
        "gleason_secondary",
        "gleason_pattern",
        "gleason_score",
        "gleason_score_label",
        "grade_group",
        "low_high_grade",
    ]
    out_fields = manifest_fields + [field for field in extra_fields if field not in manifest_fields]
    grades = parse_csv_tokens(args.grades)
    if not grades:
        raise ValueError("--grades did not contain any grade values")

    summary_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "classifier_task_manifest": str(args.classifier_task_manifest),
        "slide_labels": str(args.slide_labels),
        "grade_column": str(args.grade_column),
        "grades": grades,
        "manifests": {},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for grade in grades:
        rows: list[dict[str, Any]] = []
        skipped = Counter()
        for row in manifest_rows:
            label = labels_by_slide.get(str(row.get("slide_key", "")))
            if label is None:
                skipped["missing_slide_label"] += 1
                continue
            if str(label.get(args.grade_column, "")) != grade:
                continue
            h5_path = Path(str(row.get("h5_path", "")))
            if bool(args.require_existing_h5) and not h5_path.exists():
                skipped["missing_h5"] += 1
                continue
            rows.append({**row, **{field: label.get(field, "") for field in extra_fields}})

        out_path = args.out_dir / f"prad_binary_{args.grade_column}_{grade}.csv"
        write_csv(out_path, rows, out_fields)
        label_counts = Counter(str(row.get("label_name", "")) for row in rows)
        split_counts = Counter(str(row.get("split", "")) for row in rows)
        row_summary = {
            "grade": grade,
            "manifest": str(out_path),
            "n_slides": int(len(rows)),
            "label_counts": dict(sorted(label_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "skipped": dict(sorted(skipped.items())),
        }
        summary["manifests"][grade] = row_summary
        summary_rows.append(row_summary)

    write_json(args.out_dir / "summary.json", summary)
    write_csv(
        args.out_dir / "summary.csv",
        summary_rows,
        ["grade", "manifest", "n_slides", "label_counts", "split_counts", "skipped"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
