#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WSI_CF_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = WSI_CF_ROOT / "artifacts/prad_gleason_inputs/slide_labels.csv"
DEFAULT_OUT = WSI_CF_ROOT / "artifacts/prad_gleason_inputs/splits/prad_grade_group_patient_stratified_80_20.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a PRAD-specific patient-level stratified split for Gleason grade-group classification."
    )
    parser.add_argument("--slide-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grade-column", default="grade_group")
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--min-test-per-grade", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-existing-h5", action=argparse.BooleanOptionalAction, default=True)
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def case_sort_key(case_id: str, *, seed: int) -> tuple[str, str]:
    digest = hashlib.md5(f"{int(seed)}::{case_id}".encode("utf-8")).hexdigest()
    return digest, case_id


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def summarize(rows: list[dict[str, str]], cases: set[str], grade_column: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["case_id"]) in cases]
    return {
        "slides": int(len(selected)),
        "patients": int(len(cases)),
        "grade_counts_slides": dict(sorted(Counter(str(row[grade_column]) for row in selected).items())),
        "grade_counts_patients": dict(
            sorted(
                Counter(
                    str(next(row[grade_column] for row in selected if str(row["case_id"]) == case_id))
                    for case_id in cases
                ).items()
            )
        ),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = read_rows(args.slide_labels)
    if not rows:
        raise ValueError(f"No rows found in {args.slide_labels}")
    required = {"case_id", "slide_key", "h5_path", args.grade_column}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{args.slide_labels}: missing columns {sorted(missing)}")

    filtered: list[dict[str, str]] = []
    for row in rows:
        h5_path = Path(str(row.get("h5_path", "")))
        if bool(args.require_existing_h5) and not h5_path.exists():
            continue
        if not str(row.get(args.grade_column, "")).strip():
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError("No labeled PRAD rows remained after filtering")

    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in filtered:
        by_case[str(row["case_id"])].append(row)

    grade_by_case: dict[str, str] = {}
    for case_id, case_rows in by_case.items():
        grades = {str(row[args.grade_column]) for row in case_rows}
        if len(grades) != 1:
            raise ValueError(f"Case {case_id} has multiple grade labels: {sorted(grades)}")
        grade_by_case[case_id] = next(iter(grades))

    cases_by_grade: dict[str, list[str]] = defaultdict(list)
    for case_id, grade in grade_by_case.items():
        cases_by_grade[str(grade)].append(case_id)

    train_cases: set[str] = set()
    test_cases: set[str] = set()
    split_plan: dict[str, Any] = {}
    for grade, cases in sorted(cases_by_grade.items()):
        ordered = sorted(cases, key=lambda case_id: case_sort_key(case_id, seed=int(args.seed)))
        n_test = int(math.ceil(float(args.test_frac) * len(ordered)))
        n_test = max(int(args.min_test_per_grade), n_test)
        if len(ordered) <= 1:
            n_test = 1
        else:
            n_test = min(n_test, len(ordered) - 1)
        grade_test = set(ordered[:n_test])
        grade_train = set(ordered[n_test:])
        test_cases.update(grade_test)
        train_cases.update(grade_train)
        split_plan[grade] = {
            "patients": int(len(ordered)),
            "test_patients": int(len(grade_test)),
            "train_patients": int(len(grade_train)),
        }

    overlap = train_cases & test_cases
    if overlap:
        raise RuntimeError(f"Internal split error: train/test overlap {sorted(overlap)[:5]}")

    train_paths = [
        str(row["h5_path"])
        for row in sorted(filtered, key=lambda row: (str(row["case_id"]), str(row["slide_key"])))
        if str(row["case_id"]) in train_cases
    ]
    test_paths = [
        str(row["h5_path"])
        for row in sorted(filtered, key=lambda row: (str(row["case_id"]), str(row["slide_key"])))
        if str(row["case_id"]) in test_cases
    ]
    payload = {
        "train": train_paths,
        "test": test_paths,
        "meta": {
            "project": "TCGA-PRAD",
            "split": "patient_level_stratified_by_grade_group",
            "grade_column": str(args.grade_column),
            "test_frac": float(args.test_frac),
            "min_test_per_grade": int(args.min_test_per_grade),
            "seed": int(args.seed),
            "source_slide_labels": str(args.slide_labels),
            "split_plan": split_plan,
            "train": summarize(filtered, train_cases, str(args.grade_column)),
            "test": summarize(filtered, test_cases, str(args.grade_column)),
            "total": summarize(filtered, set(by_case), str(args.grade_column)),
        },
    }
    write_json(args.out, payload)
    print(json.dumps(payload["meta"], indent=2))
    print(f"[ok] wrote split manifest: {args.out}")


if __name__ == "__main__":
    main()
