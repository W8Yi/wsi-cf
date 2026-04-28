#!/usr/bin/env python3
"""
Build a pan-cancer "Balanced Core" TCGA subset with deterministic 90/10 split.

Design goals:
- Cover every TCGA project present in case_labels_master.tsv.
- Keep project contribution balanced (fixed max cases per project).
- Prefer label-diverse cases using a greedy coverage objective.
- Emit train/test manifests compatible with existing H5-based pipelines.

Outputs (under --out_dir):
- manifest.json
- summary.json
- balanced_core_cases.tsv
- balanced_core_slides.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


UNKNOWN_VALUES = {"", "Unknown", "NA", "NaN", "None", "Null"}


def is_informative(value: str) -> bool:
    return str(value or "").strip() not in UNKNOWN_VALUES


def stable_hash_int(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--case_labels_tsv",
        type=Path,
        default=Path("metadata/labels/master/case_labels_master.tsv"),
        help="Case-level master label table.",
    )
    ap.add_argument(
        "--slide_labels_tsv",
        type=Path,
        default=Path("metadata/labels/master/slide_labels_master.tsv"),
        help="Slide-level master label table with H5 paths.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("metadata/manifests/tcga_balanced_core_90_10"),
        help="Output directory for manifest and reports.",
    )
    ap.add_argument(
        "--cases_per_project",
        type=int,
        default=40,
        help="Maximum selected cases per project.",
    )
    ap.add_argument(
        "--test_frac",
        type=float,
        default=0.10,
        help="Target test fraction, applied within each project.",
    )
    ap.add_argument(
        "--slides_per_case",
        type=int,
        default=1,
        help="Representative slides per case (<=0 means keep all case slides).",
    )
    ap.add_argument(
        "--coverage_weight",
        type=float,
        default=5.0,
        help="Weight for adding unseen label tokens in greedy selection.",
    )
    ap.add_argument(
        "--require_h5_exists",
        action="store_true",
        help="Require selected slide H5 paths to exist on disk.",
    )
    return ap.parse_args()


def build_case_tokens(case_row: dict) -> Set[str]:
    tokens: Set[str] = set()

    categorical_fields = [
        "hpv_status",
        "immune_subtype",
        "msi_status",
        "pam50_subtype",
        "stage",
        "tumor_grade",
        "os_status",
        "pfs_status",
    ]
    for field in categorical_fields:
        value = str(case_row.get(field, "")).strip()
        if is_informative(value):
            tokens.add(f"{field}={value}")

    mutation_has_data = str(case_row.get("mutation_has_data", "")).strip() == "1"
    if mutation_has_data:
        for field in ["tp53_mutated", "kras_mutated"]:
            value = str(case_row.get(field, "")).strip()
            if is_informative(value):
                tokens.add(f"{field}={value}")

    purity_raw = str(case_row.get("tumor_purity", "")).strip()
    if is_informative(purity_raw):
        try:
            purity = float(purity_raw)
            purity_bin = min(9, max(0, int(math.floor(purity * 10.0))))
            tokens.add(f"tumor_purity_bin={purity_bin}")
        except ValueError:
            pass

    return tokens


def choose_case_subset_for_project(
    case_ids: List[str],
    case_tokens: Dict[str, Set[str]],
    token_freq: Dict[str, int],
    coverage_weight: float,
) -> List[str]:
    selected: List[str] = []
    covered: Set[str] = set()
    remaining = set(case_ids)

    # Deterministic tie-break.
    ordered_ids = sorted(case_ids, key=lambda cid: (stable_hash_int(cid), cid))
    index = {cid: idx for idx, cid in enumerate(ordered_ids)}

    while remaining:
        best_case = None
        best_gain = None
        for cid in remaining:
            toks = case_tokens.get(cid, set())
            new_tokens = toks - covered
            rarity_gain = sum(1.0 / max(1, token_freq.get(t, 1)) for t in new_tokens)
            base_gain = sum(1.0 / max(1, token_freq.get(t, 1)) for t in toks)
            gain = coverage_weight * float(len(new_tokens)) + rarity_gain + 0.1 * base_gain
            rank = (gain, -index[cid], cid)
            if best_gain is None or rank > best_gain:
                best_gain = rank
                best_case = cid
        assert best_case is not None
        selected.append(best_case)
        covered.update(case_tokens.get(best_case, set()))
        remaining.remove(best_case)
    return selected


def pick_project_train_test(case_ids: List[str], test_frac: float) -> Tuple[Set[str], Set[str]]:
    ordered = sorted(case_ids, key=lambda cid: (stable_hash_int(cid), cid))
    n = len(ordered)
    if n <= 1:
        return set(ordered), set()

    n_test = int(round(float(n) * float(test_frac)))
    n_test = max(1, min(n_test, n - 1))
    test_ids = set(ordered[:n_test])
    train_ids = set(ordered[n_test:])
    return train_ids, test_ids


def slide_priority(row: dict) -> Tuple[int, int, str]:
    sample_type = str(row.get("sample_type_code", "")).strip()
    slide_key = str(row.get("slide_key", "")).strip()
    # Primary tumor first, then DX1, then lexical.
    sample_rank = 0 if sample_type == "01" else 1
    dx1_rank = 0 if "-DX1" in slide_key else 1
    return sample_rank, dx1_rank, slide_key


def select_slides_for_case(
    slide_rows: List[dict],
    slides_per_case: int,
    require_h5_exists: bool,
) -> List[dict]:
    rows = sorted(slide_rows, key=slide_priority)
    out: List[dict] = []
    for row in rows:
        h5_path = str(row.get("h5_path", "")).strip()
        if not h5_path:
            continue
        if require_h5_exists and not Path(h5_path).exists():
            continue
        out.append(row)
    if slides_per_case > 0:
        return out[:slides_per_case]
    return out


def main() -> None:
    args = parse_args()
    if args.cases_per_project <= 0:
        raise SystemExit("--cases_per_project must be > 0")
    if not (0.0 < args.test_frac < 1.0):
        raise SystemExit("--test_frac must be in (0, 1)")

    case_rows = read_tsv(args.case_labels_tsv)
    slide_rows = read_tsv(args.slide_labels_tsv)

    case_by_id: Dict[str, dict] = {}
    project_to_cases: Dict[str, List[str]] = defaultdict(list)
    for row in case_rows:
        case_id = str(row.get("case_id", "")).strip()
        project = str(row.get("project_dir", "")).strip()
        if not case_id or not project:
            continue
        case_by_id[case_id] = row
        project_to_cases[project].append(case_id)

    slides_by_case: Dict[str, List[dict]] = defaultdict(list)
    for row in slide_rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            slides_by_case[case_id].append(row)

    # Keep only cases that have at least one slide row.
    for project, cids in list(project_to_cases.items()):
        keep = [cid for cid in cids if cid in slides_by_case]
        if keep:
            project_to_cases[project] = sorted(set(keep))
        else:
            del project_to_cases[project]

    case_tokens: Dict[str, Set[str]] = {cid: build_case_tokens(case_by_id[cid]) for cid in case_by_id}
    token_freq: Counter[str] = Counter()
    for cid in case_by_id:
        token_freq.update(case_tokens[cid])

    selected_cases_by_project: Dict[str, List[str]] = {}
    for project in sorted(project_to_cases):
        case_ids = project_to_cases[project]
        ordered = choose_case_subset_for_project(
            case_ids=case_ids,
            case_tokens=case_tokens,
            token_freq=dict(token_freq),
            coverage_weight=float(args.coverage_weight),
        )
        take_n = min(len(ordered), int(args.cases_per_project))
        selected_cases_by_project[project] = ordered[:take_n]

    train_case_ids: Set[str] = set()
    test_case_ids: Set[str] = set()
    for project, case_ids in selected_cases_by_project.items():
        project_train, project_test = pick_project_train_test(case_ids, test_frac=float(args.test_frac))
        train_case_ids.update(project_train)
        test_case_ids.update(project_test)

    selected_case_ids = train_case_ids | test_case_ids

    case_table_rows: List[dict] = []
    slide_table_rows: List[dict] = []
    train_h5: List[str] = []
    test_h5: List[str] = []

    selected_tokens_all: Set[str] = set()

    for case_id in sorted(selected_case_ids):
        row = case_by_id[case_id]
        project = str(row.get("project_dir", "")).strip()
        split = "test" if case_id in test_case_ids else "train"
        tokens = sorted(case_tokens.get(case_id, set()))
        selected_tokens_all.update(tokens)
        rarity_score = sum(1.0 / max(1, token_freq.get(t, 1)) for t in tokens)

        case_table_rows.append(
            {
                "split": split,
                "project_dir": project,
                "case_id": case_id,
                "num_tokens": len(tokens),
                "rarity_score": f"{rarity_score:.6f}",
                "tokens": "|".join(tokens),
            }
        )

        chosen_slides = select_slides_for_case(
            slide_rows=slides_by_case.get(case_id, []),
            slides_per_case=int(args.slides_per_case),
            require_h5_exists=bool(args.require_h5_exists),
        )
        for srow in chosen_slides:
            out_row = {
                "split": split,
                "project_dir": project,
                "case_id": case_id,
                "sample_id": srow.get("sample_id", ""),
                "slide_key": srow.get("slide_key", ""),
                "sample_type_code": srow.get("sample_type_code", ""),
                "h5_path": srow.get("h5_path", ""),
            }
            slide_table_rows.append(out_row)
            if split == "train":
                train_h5.append(str(srow.get("h5_path", "")))
            else:
                test_h5.append(str(srow.get("h5_path", "")))

    # Deduplicate while preserving deterministic order.
    train_h5 = sorted(set([p for p in train_h5 if p]))
    test_h5 = sorted(set([p for p in test_h5 if p]))

    # Coverage report.
    all_tokens = set(token_freq.keys())
    coverage_ratio = (len(selected_tokens_all) / len(all_tokens)) if all_tokens else 0.0

    case_counts_by_project: Dict[str, Dict[str, int]] = {}
    for project in sorted(selected_cases_by_project):
        cids = selected_cases_by_project[project]
        case_counts_by_project[project] = {
            "selected_cases": len(cids),
            "train_cases": sum(1 for cid in cids if cid in train_case_ids),
            "test_cases": sum(1 for cid in cids if cid in test_case_ids),
            "available_cases": len(project_to_cases[project]),
        }

    slide_counts_by_project: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train_slides": 0, "test_slides": 0})
    for r in slide_table_rows:
        project = r["project_dir"]
        if r["split"] == "train":
            slide_counts_by_project[project]["train_slides"] += 1
        else:
            slide_counts_by_project[project]["test_slides"] += 1

    summary = {
        "design": {
            "name": "Balanced Core",
            "description": "Project-balanced, diversity-aware case subset with deterministic split.",
            "cases_per_project": int(args.cases_per_project),
            "test_frac": float(args.test_frac),
            "slides_per_case": int(args.slides_per_case),
            "coverage_weight": float(args.coverage_weight),
            "require_h5_exists": bool(args.require_h5_exists),
        },
        "inputs": {
            "case_labels_tsv": str(args.case_labels_tsv.resolve()),
            "slide_labels_tsv": str(args.slide_labels_tsv.resolve()),
        },
        "counts": {
            "projects_selected": len(selected_cases_by_project),
            "cases_total": len(selected_case_ids),
            "cases_train": len(train_case_ids),
            "cases_test": len(test_case_ids),
            "slides_total": len(slide_table_rows),
            "slides_train": sum(1 for r in slide_table_rows if r["split"] == "train"),
            "slides_test": sum(1 for r in slide_table_rows if r["split"] == "test"),
            "train_h5": len(train_h5),
            "test_h5": len(test_h5),
        },
        "token_coverage": {
            "selected_unique_tokens": len(selected_tokens_all),
            "all_unique_tokens": len(all_tokens),
            "coverage_ratio": coverage_ratio,
        },
        "case_counts_by_project": case_counts_by_project,
        "slide_counts_by_project": dict(sorted(slide_counts_by_project.items())),
    }

    manifest = {
        "train": train_h5,
        "test": test_h5,
        "meta": {
            "split_level": "case_id",
            "subset": "tcga_balanced_core",
            "summary_path": "summary.json",
            "slide_table": "balanced_core_slides.tsv",
            "case_table": "balanced_core_cases.tsv",
            **summary["design"],
            "counts": summary["counts"],
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        args.out_dir / "balanced_core_cases.tsv",
        fieldnames=["split", "project_dir", "case_id", "num_tokens", "rarity_score", "tokens"],
        rows=sorted(case_table_rows, key=lambda r: (r["split"], r["project_dir"], r["case_id"])),
    )
    write_tsv(
        args.out_dir / "balanced_core_slides.tsv",
        fieldnames=["split", "project_dir", "case_id", "sample_id", "slide_key", "sample_type_code", "h5_path"],
        rows=sorted(slide_table_rows, key=lambda r: (r["split"], r["project_dir"], r["case_id"], r["slide_key"])),
    )
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[ok] wrote {(args.out_dir / 'manifest.json')}")
    print(f"[ok] wrote {(args.out_dir / 'summary.json')}")
    print(f"[ok] wrote {(args.out_dir / 'balanced_core_cases.tsv')}")
    print(f"[ok] wrote {(args.out_dir / 'balanced_core_slides.tsv')}")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
