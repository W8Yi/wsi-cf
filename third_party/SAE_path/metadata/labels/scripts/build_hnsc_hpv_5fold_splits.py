#!/usr/bin/env python3
"""
Build deterministic stratified case-level 5-fold HNSC HPV splits.

By default this creates 5 folds, with each split using:
- ~80% train
- ~20% test

Outputs go into one folder, by default:
  metadata/manifests/hnsc_hpv_5fold/

Per split:
- split_{i}.json
- split_{i}.tsv

Also writes:
- index.json
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


def assign_cases_to_folds(
    label_to_cases: Dict[str, List[str]],
    num_splits: int,
) -> Dict[str, int]:
    case_to_fold: Dict[str, int] = {}
    for label, case_ids in sorted(label_to_cases.items()):
        ordered = sorted(case_ids, key=lambda cid: (stable_hash_int(cid), cid))
        for idx, case_id in enumerate(ordered):
            case_to_fold[case_id] = idx % num_splits
    return case_to_fold


def build_case_assignments(
    case_to_fold: Dict[str, int],
    split_idx: int,
) -> Dict[str, str]:
    return {
        case_id: ("test" if fold_idx == split_idx else "train")
        for case_id, fold_idx in case_to_fold.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels_tsv",
        type=Path,
        default=Path("metadata/labels/master/slide_labels_master.tsv"),
        help="Master slide labels table.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("metadata/manifests/hnsc_hpv_5fold"),
        help="Directory where split files will be written.",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=5,
        help="Number of stratified folds to generate.",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.20,
        help="Target test fraction (informational); for K folds this should be close to 1/K.",
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
    if args.num_splits <= 1:
        raise SystemExit("num_splits must be > 1.")
    if args.test_frac <= 0 or args.test_frac >= 1:
        raise SystemExit("test_frac must be > 0 and < 1.")
    expected_test_frac = 1.0 / float(args.num_splits)
    if abs(args.test_frac - expected_test_frac) > 1e-6:
        print(
            f"[warn] --test_frac={args.test_frac:.4f} does not match 1/num_splits={expected_test_frac:.4f}; "
            "using 1/num_splits for metadata."
        )
    effective_test_frac = expected_test_frac

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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "meta": {
            "source_labels_tsv": str(args.labels_tsv.resolve()),
            "out_dir": str(args.out_dir.resolve()),
            "project_dir": "TCGA-HNSC",
            "label_field": "hpv_status",
            "label_map": LABEL_TO_INT,
            "split_level": "case_id",
            "split_style": "stratified_kfold",
            "num_splits": args.num_splits,
            "test_frac": effective_test_frac,
            "train_frac": 1.0 - effective_test_frac,
            "exclude_conflicts": not args.include_conflicts,
            "require_h5_exists": not args.allow_missing_h5,
            "dropped_conflict_rows": dropped_conflicts,
            "dropped_missing_h5_rows": dropped_missing_h5,
            "cases_total": len(case_to_rows),
            "slides_total": len(filtered),
            "cases_by_label": dict(Counter(case_to_label.values())),
            "slides_by_label": dict(Counter(row["hpv_status"] for row in filtered)),
        },
        "splits": [],
    }

    fieldnames = [
        "split_name",
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
    case_to_fold = assign_cases_to_folds(label_to_cases=label_to_cases, num_splits=args.num_splits)

    for split_idx in range(args.num_splits):
        case_to_split = build_case_assignments(
            case_to_fold=case_to_fold,
            split_idx=split_idx,
        )
        split_name = f"split_{split_idx}"

        manifest = {
            "meta": {
                "split_name": split_name,
                "split_index": split_idx,
                "source_labels_tsv": str(args.labels_tsv.resolve()),
                "project_dir": "TCGA-HNSC",
                "label_field": "hpv_status",
                "label_map": LABEL_TO_INT,
                "split_level": "case_id",
                "split_style": "stratified_kfold",
                "test_frac": effective_test_frac,
                "train_frac": 1.0 - effective_test_frac,
                "exclude_conflicts": not args.include_conflicts,
                "require_h5_exists": not args.allow_missing_h5,
            },
            "train": [],
            "test": [],
        }

        dataset_rows: List[dict] = []
        split_case_counts = Counter()
        split_slide_counts = Counter()
        split_case_label_counts: Dict[str, Counter] = {"train": Counter(), "test": Counter()}

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
                        "split_name": split_name,
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
            "cases_by_split": dict(split_case_counts),
            "slides_by_split": dict(split_slide_counts),
            "cases_by_split_and_label": {
                split: dict(counter) for split, counter in split_case_label_counts.items()
            },
        }

        json_path = args.out_dir / f"{split_name}.json"
        tsv_path = args.out_dir / f"{split_name}.tsv"

        with json_path.open("w") as handle:
            json.dump(manifest, handle, indent=2)

        dataset_rows.sort(key=lambda row: (row["split"], row["label"], row["case_id"], row["slide_key"]))
        write_tsv(tsv_path, fieldnames, dataset_rows)

        index["splits"].append(
            {
                "split_name": split_name,
                "json_path": str(json_path.resolve()),
                "tsv_path": str(tsv_path.resolve()),
                "counts": manifest["meta"]["counts"],
            }
        )

        print(f"[ok] wrote {json_path}")
        print(f"[ok] wrote {tsv_path}")

    index_path = args.out_dir / "index.json"
    with index_path.open("w") as handle:
        json.dump(index, handle, indent=2)
    print(f"[ok] wrote {index_path}")
    print(json.dumps(index["meta"], indent=2))


if __name__ == "__main__":
    main()
