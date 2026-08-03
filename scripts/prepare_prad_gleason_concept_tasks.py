#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


WSI_CF_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURE_DIR = Path("/research/projects/mllab/WSI/TCGA_features/TCGA-PRAD/features_uni2")
DEFAULT_SPLIT_MANIFEST = WSI_CF_ROOT / "resources/manifests/sae_manifests_tcga_patient_train_test_90_10.json"
DEFAULT_OUT_DIR = WSI_CF_ROOT / "artifacts/prad_gleason_inputs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build local TCGA-PRAD Gleason score and grade-group concept labels from GDC clinical metadata."
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--refresh-gdc", action="store_true", help="Refresh cached GDC clinical metadata.")
    return parser


def case_id_from_slide_key(slide_key: str) -> str:
    return "-".join(str(slide_key).split("-")[:3])


def slide_key_from_path(path_value: str) -> str:
    return Path(str(path_value)).name.split(".")[0]


def pattern_number(value: Any) -> int | None:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else None


def isup_grade_group(primary: int, secondary: int) -> int:
    score = int(primary) + int(secondary)
    if score <= 6:
        return 1
    if (int(primary), int(secondary)) == (3, 4):
        return 2
    if (int(primary), int(secondary)) == (4, 3):
        return 3
    if score == 8:
        return 4
    return 5


def morphology_group_from_patterns(primary: int, secondary: int) -> tuple[str, str]:
    """Map slide-level Gleason patterns to a three-class morphology label.

    The class is assigned by the highest pattern present, so a 4+5 or 5+4
    diagnosis is treated as pattern 5 morphology.
    """
    highest = max(int(primary), int(secondary))
    if highest <= 3:
        return (
            "pattern_1_3_well_formed",
            "DISCRETE WELL-FORMED GLANDS (GLEASON PATTERNS 1-3)",
        )
    if highest == 4:
        return (
            "pattern_4_cribriform_poorly_formed_fused",
            "CRIBRIFORM/POORLY-FORMED/FUSED GLANDS (GLEASON PATTERN 4)",
        )
    return (
        "pattern_5_solid_single_necrosis",
        "SHEETS/CORDS/SINGLE CELLS/SOLID NESTS/NECROSIS (GLEASON PATTERN 5)",
    )


def select_prostate_diagnosis(case: dict[str, Any]) -> dict[str, Any] | None:
    diagnoses = list(case.get("diagnoses") or [])
    with_patterns = [
        diagnosis
        for diagnosis in diagnoses
        if pattern_number(diagnosis.get("primary_gleason_grade")) is not None
        and pattern_number(diagnosis.get("secondary_gleason_grade")) is not None
    ]
    for diagnosis in with_patterns:
        if str(diagnosis.get("tissue_or_organ_of_origin", "")).lower() == "prostate gland":
            return diagnosis
    return with_patterns[0] if with_patterns else None


def query_gdc_prad_cases() -> list[dict[str, Any]]:
    filters = {"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-PRAD"]}}
    fields = (
        "submitter_id,"
        "diagnoses.submitter_id,"
        "diagnoses.tissue_or_organ_of_origin,"
        "diagnoses.primary_gleason_grade,"
        "diagnoses.secondary_gleason_grade"
    )
    response = requests.get(
        "https://api.gdc.cancer.gov/cases",
        params={"filters": json.dumps(filters), "fields": fields, "format": "JSON", "size": "1000"},
        timeout=120,
    )
    response.raise_for_status()
    return list(response.json()["data"]["hits"])


def load_or_query_gdc_cases(cache_path: Path, *, refresh: bool) -> list[dict[str, Any]]:
    if cache_path.exists() and not bool(refresh):
        with cache_path.open("r") as handle:
            payload = json.load(handle)
        return list(payload["cases"])
    cases = query_gdc_prad_cases()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as handle:
        json.dump({"project": "TCGA-PRAD", "source": "GDC cases API", "cases": cases}, handle, indent=2)
        handle.write("\n")
    return cases


def load_split_cases(path: Path) -> dict[str, str]:
    with path.open("r") as handle:
        payload = json.load(handle)
    splits: dict[str, str] = {}
    for split in ("train", "test"):
        for item in payload.get(split, []):
            case_id = case_id_from_slide_key(slide_key_from_path(str(item)))
            prior = splits.get(case_id)
            if prior and prior != split:
                raise ValueError(f"{path}: patient {case_id} appears in both train and test")
            splits[case_id] = split
    return splits


def build_case_labels(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("submitter_id", ""))
        diagnosis = select_prostate_diagnosis(case)
        if not case_id or diagnosis is None:
            continue
        primary = pattern_number(diagnosis.get("primary_gleason_grade"))
        secondary = pattern_number(diagnosis.get("secondary_gleason_grade"))
        if primary is None or secondary is None:
            continue
        score = int(primary) + int(secondary)
        grade_group = isup_grade_group(primary, secondary)
        morphology_group, morphology_group_description = morphology_group_from_patterns(primary, secondary)
        labels[case_id] = {
            "case_id": case_id,
            "gleason_primary": int(primary),
            "gleason_secondary": int(secondary),
            "gleason_pattern": f"{primary}+{secondary}",
            "gleason_score": int(score),
            "gleason_score_label": f"GS{score}",
            "grade_group": f"GG{grade_group}",
            "low_high_grade": "low" if grade_group <= 2 else "high",
            "morphology_group": morphology_group,
            "morphology_group_description": morphology_group_description,
        }
    return labels


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "slide_key",
        "sample_id",
        "project_dir",
        "h5_path",
        "split",
        "gleason_primary",
        "gleason_secondary",
        "gleason_pattern",
        "gleason_score",
        "gleason_score_label",
        "grade_group",
        "low_high_grade",
        "morphology_group",
        "morphology_group_description",
        "label_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def counter_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.feature_dir.exists():
        raise FileNotFoundError(f"Missing PRAD UNI2 feature directory: {args.feature_dir}")
    cases = load_or_query_gdc_cases(args.out_dir / "gdc_prad_cases.json", refresh=bool(args.refresh_gdc))
    labels = build_case_labels(cases)
    splits = load_split_cases(args.split_manifest)

    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for h5_path in sorted(args.feature_dir.glob("*.h5")):
        slide_key = h5_path.stem
        case_id = case_id_from_slide_key(slide_key)
        if case_id not in labels:
            skipped["missing_gleason_label"] += 1
            continue
        if case_id not in splits:
            skipped["missing_train_test_split"] += 1
            continue
        rows.append(
            {
                **labels[case_id],
                "slide_key": slide_key,
                "sample_id": f"{case_id}-{slide_key.split('-')[3][:2]}",
                "project_dir": "TCGA-PRAD",
                "h5_path": str(h5_path),
                "split": splits[case_id],
                "label_source": "GDC diagnoses.primary_gleason_grade+secondary_gleason_grade",
            }
        )
    if not rows:
        raise RuntimeError("No local TCGA-PRAD UNI2 bags matched GDC Gleason labels and split metadata")

    label_path = args.out_dir / "slide_labels.csv"
    write_csv(label_path, rows)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
    summary = {
        "project": "TCGA-PRAD",
        "gdc_case_count": int(len(cases)),
        "gdc_cases_with_gleason_patterns": int(len(labels)),
        "local_feature_dir": str(args.feature_dir),
        "local_feature_file_count": int(len(list(args.feature_dir.glob("*.h5")))),
        "matched_slide_count": int(len(rows)),
        "matched_patient_count": int(len({row["case_id"] for row in rows})),
        "gleason_score_slide_counts": counter_by(rows, "gleason_score_label"),
        "grade_group_slide_counts": counter_by(rows, "grade_group"),
        "low_high_slide_counts": counter_by(rows, "low_high_grade"),
        "morphology_group_slide_counts": counter_by(rows, "morphology_group"),
        "split_counts": {
            split: {
                "slides": int(len(split_rows)),
                "patients": int(len({row["case_id"] for row in split_rows})),
                "gleason_score": counter_by(split_rows, "gleason_score_label"),
                "low_high_grade": counter_by(split_rows, "low_high_grade"),
                "morphology_group": counter_by(split_rows, "morphology_group"),
            }
            for split, split_rows in sorted(by_split.items())
        },
        "skipped_feature_files": dict(skipped),
        "label_file": str(label_path),
    }
    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"[ok] wrote {label_path}")
    print(
        f"[cohort] slides={summary['matched_slide_count']} patients={summary['matched_patient_count']} "
        f"scores={summary['gleason_score_slide_counts']} binary={summary['low_high_slide_counts']}"
    )
    print(f"[split] {summary['split_counts']}")


if __name__ == "__main__":
    main()
