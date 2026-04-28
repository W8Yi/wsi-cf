#!/usr/bin/env python3
"""
Build a clean TCGA-HNSC HPV dataset split from the master slide labels table.

Outputs:
- JSON manifest with train/val/test lists of H5 feature paths
- TSV with one row per slide and explicit split + label columns

Default behavior:
- Uses metadata/labels/master/slide_labels_master.tsv
- Keeps only TCGA-HNSC rows with hpv_status in {HPV+, HPV-}
- Excludes rows flagged with hpv_conflict=1
- Splits at the patient (case_id) level to avoid slide leakage
- Stratifies deterministically within each class using an MD5 hash of case_id
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


LABEL_TO_INT = {"HPV-": 0, "HPV+": 1}


def stable_hash_int(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bounded_split_count(total: int, frac: float, remaining_after: int) -> int:
    count = int(round(total * frac))
    max_allowed = max(0, total - remaining_after)
    return min(count, max_allowed)


def assign_case_splits(
    label_to_cases: Dict[str, List[str]],
    val_frac: float,
    test_frac: float,
) -> Dict[str, str]:
    case_to_split: Dict[str, str] = {}

    for label, case_ids in sorted(label_to_cases.items()):
        ordered = sorted(case_ids, key=lambda cid: (stable_hash_int(cid), cid))
        n_total = len(ordered)

        # Keep at least one case in train when possible.
        n_test = bounded_split_count(n_total, test_frac, remaining_after=1)
        n_val = bounded_split_count(n_total - n_test, val_frac / max(1e-12, 1.0 - test_frac), remaining_after=1)

        test_cases = ordered[:n_test]
        val_cases = ordered[n_test:n_test + n_val]
        train_cases = ordered[n_test + n_val:]

        for case_id in train_cases:
            case_to_split[case_id] = "train"
        for case_id in val_cases:
            case_to_split[case_id] = "val"
        for case_id in test_cases:
            case_to_split[case_id] = "test"

    return case_to_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels_tsv",
        type=Path,
        default=Path("metadata/labels/master/slide_labels_master.tsv"),
        help="Master slide labels table.",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        default=Path("metadata/manifests/hnsc_hpv_patient_split_80_10_10.json"),
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--out_tsv",
        type=Path,
        default=Path("metadata/manifests/hnsc_hpv_patient_split_80_10_10.tsv"),
        help="Output TSV with split assignments.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.10,
        help="Validation fraction at the case level.",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.10,
        help="Test fraction at the case level.",
    )
    parser.add_argument(
        "--include_conflicts",
        action="store_true",
        help="Keep rows with hpv_conflict=1 instead of excluding them.",
    )
    parser.add_argument(
        "--allow_missing_h5",
        action="store_true",
        help="Keep rows even if the H5 path does not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_frac < 0 or args.test_frac < 0 or args.val_frac + args.test_frac >= 1:
        raise SystemExit("val_frac and test_frac must be >= 0 and sum to < 1.")

    rows = read_tsv(args.labels_tsv)

    filtered: List[dict] = []
    dropped_missing_h5 = 0
    dropped_conflicts = 0

    for row in rows:
        if row["project_dir"] != "TCGA-HNSC":
            continue
        if row["hpv_status"] not in LABEL_TO_INT:
            continue
        if not args.include_conflicts and row["hpv_conflict"] == "1":
            dropped_conflicts += 1
            continue
        if not args.allow_missing_h5 and not Path(row["h5_path"]).exists():
            dropped_missing_h5 += 1
            continue
        filtered.append(row)

    if not filtered:
        raise SystemExit("No HNSC HPV rows matched the requested filters.")

    case_to_rows: Dict[str, List[dict]] = defaultdict(list)
    case_to_label: Dict[str, str] = {}

    for row in filtered:
        case_id = row["case_id"]
        case_to_rows[case_id].append(row)
        label = row["hpv_status"]
        prior = case_to_label.get(case_id)
        if prior is not None and prior != label:
            raise SystemExit(f"Inconsistent hpv_status within case {case_id}: {prior} vs {label}")
        case_to_label[case_id] = label

    label_to_cases: Dict[str, List[str]] = defaultdict(list)
    for case_id, label in sorted(case_to_label.items()):
        label_to_cases[label].append(case_id)

    case_to_split = assign_case_splits(label_to_cases, val_frac=args.val_frac, test_frac=args.test_frac)

    manifest = {
        "meta": {
            "source_labels_tsv": str(args.labels_tsv.resolve()),
            "project_dir": "TCGA-HNSC",
            "label_field": "hpv_status",
            "label_map": LABEL_TO_INT,
            "split_level": "case_id",
            "stratified_by": "hpv_status",
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "exclude_conflicts": not args.include_conflicts,
            "require_h5_exists": not args.allow_missing_h5,
            "dropped_conflict_rows": dropped_conflicts,
            "dropped_missing_h5_rows": dropped_missing_h5,
        },
        "train": [],
        "val": [],
        "test": [],
    }

    dataset_rows: List[dict] = []
    split_case_counts = Counter()
    split_slide_counts = Counter()
    split_case_label_counts: Dict[str, Counter] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    for case_id in sorted(case_to_rows):
        split = case_to_split[case_id]
        label = case_to_label[case_id]
        split_case_counts[split] += 1
        split_case_label_counts[split][label] += 1

        case_rows = sorted(case_to_rows[case_id], key=lambda row: (row["slide_key"], row["h5_path"]))
        for row in case_rows:
            h5_path = row["h5_path"]
            manifest[split].append(h5_path)
            split_slide_counts[split] += 1
            dataset_rows.append(
                {
                    "split": split,
                    "label": LABEL_TO_INT[label],
                    "hpv_status": label,
                    "case_id": row["case_id"],
                    "slide_key": row["slide_key"],
                    "sample_id": row["sample_id"],
                    "project_dir": row["project_dir"],
                    "h5_path": h5_path,
                    "hpv_source": row["hpv_source"],
                    "hpv_conflict": row["hpv_conflict"],
                    "hpv_score_viral": row["hpv_score_viral"],
                    "hpv_status_from_viral_threshold6": row["hpv_status_from_viral_threshold6"],
                }
            )

    manifest["meta"]["counts"] = {
        "cases_total": len(case_to_rows),
        "slides_total": len(dataset_rows),
        "cases_by_label": dict(Counter(case_to_label.values())),
        "slides_by_label": dict(Counter(row["hpv_status"] for row in filtered)),
        "cases_by_split": dict(split_case_counts),
        "slides_by_split": dict(split_slide_counts),
        "cases_by_split_and_label": {
            split: dict(counter) for split, counter in split_case_label_counts.items()
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    fieldnames = [
        "split",
        "label",
        "hpv_status",
        "case_id",
        "slide_key",
        "sample_id",
        "project_dir",
        "h5_path",
        "hpv_source",
        "hpv_conflict",
        "hpv_score_viral",
        "hpv_status_from_viral_threshold6",
    ]
    dataset_rows.sort(key=lambda row: (row["split"], row["label"], row["case_id"], row["slide_key"]))
    write_tsv(args.out_tsv, fieldnames, dataset_rows)

    print(f"[ok] wrote {args.out_json}")
    print(f"[ok] wrote {args.out_tsv}")
    print(json.dumps(manifest["meta"], indent=2))


if __name__ == "__main__":
    main()
