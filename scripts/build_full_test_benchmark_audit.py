#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("paper_outputs/full_test_streaming_benchmark_v1/audit")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write a denominator audit for the full-test paper benchmark. "
            "The audit starts from held-out source slides and checks which of "
            "them are represented in the current region banks."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--task-direction",
        action="append",
        default=None,
        help="Optional task/direction filter such as hnscc_hpv/hpv_pos_to_hpv_neg.",
    )
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def case_from_slide(slide_key: str) -> str:
    text = str(slide_key)
    return text[:12] if text.startswith("TCGA-") and len(text) >= 12 else text.split("__", 1)[0].split(".", 1)[0]


def normalize_hpv(value: str) -> str:
    text = str(value).strip()
    aliases = {"HPV+": "hpv_pos", "HPV-": "hpv_neg", "hpv+": "hpv_pos", "hpv-": "hpv_neg"}
    return aliases.get(text, text)


def hnscc_checkpoint_slides(split: str, source_label: str) -> list[dict[str, str]]:
    labels = {row["slide_id"]: row for row in read_csv(Path("resources/models/classifiers/hnscc_hpv/HNSCC.csv"))}
    rows: list[dict[str, str]] = []
    for row in read_csv(Path("resources/models/classifiers/hnscc_hpv/splits_0.csv")):
        slide = str(row.get(split, "")).strip()
        if not slide:
            continue
        label_row = labels.get(slide, {})
        label = normalize_hpv(label_row.get("hpv_status", "unknown"))
        if label != source_label:
            continue
        rows.append(
            {
                "task_name": "hnscc_hpv",
                "slide_key": slide,
                "case_id": label_row.get("case_id") or case_from_slide(slide),
                "source_label": source_label,
                "split": split,
                "h5_path": "",
            }
        )
    return rows


def manifest_slides(path: Path, *, task_name: str, source_label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(path):
        if str(row.get("split", "")) != "test":
            continue
        if str(row.get("label_name", row.get("raw_label", ""))) != source_label:
            continue
        rows.append(
            {
                "task_name": task_name,
                "slide_key": str(row.get("slide_key", "")).strip(),
                "case_id": str(row.get("case_id", "")).strip() or case_from_slide(str(row.get("slide_key", ""))),
                "source_label": source_label,
                "split": "test",
                "h5_path": str(row.get("h5_path", "")).strip(),
            }
        )
    return rows


def direction_specs() -> list[dict[str, str]]:
    normal_root = Path("artifacts/classifier_training_normal_tumor")
    prad_manifest = Path("artifacts/classifier_training/prad_morphology_group/task_manifest.csv")
    specs = [
        {
            "task_name": "hnscc_hpv",
            "direction": "hpv_pos_to_hpv_neg",
            "source_label": "hpv_pos",
            "target_label": "hpv_neg",
            "region_bank_csv": "artifacts/prediction_transition_region_banks_test_only_unbalanced/hnscc_hpv/region_bank.csv",
        },
        {
            "task_name": "hnscc_hpv",
            "direction": "hpv_neg_to_hpv_pos",
            "source_label": "hpv_neg",
            "target_label": "hpv_pos",
            "region_bank_csv": "artifacts/prediction_transition_region_banks_test_only_unbalanced/hnscc_hpv/region_bank.csv",
        },
    ]
    for task_name in ["luad_normal_tumor", "coad_normal_tumor", "kirc_normal_tumor", "brca_normal_tumor"]:
        specs.append(
            {
                "task_name": task_name,
                "direction": "normal_to_tumor",
                "source_label": "normal",
                "target_label": "tumor",
                "manifest_csv": str(normal_root / task_name / "task_manifest.csv"),
                "region_bank_csv": f"artifacts/prediction_transition_region_banks_test_only_unbalanced/normal_tumor/{task_name}/region_bank.csv",
            }
        )
    specs.extend(
        [
            {
                "task_name": "prad_morphology_group",
                "direction": "well_to_p4",
                "source_label": "pattern_1_3_well_formed",
                "target_label": "pattern_4_cribriform_poorly_formed_fused",
                "manifest_csv": str(prad_manifest),
                "region_bank_csv": "artifacts/prediction_transition_region_banks_test_only_unbalanced/prad_morphology_group/well_to_p4/region_bank.csv",
            },
            {
                "task_name": "prad_morphology_group",
                "direction": "p4_to_p5",
                "source_label": "pattern_4_cribriform_poorly_formed_fused",
                "target_label": "pattern_5_solid_single_necrosis",
                "manifest_csv": str(prad_manifest),
                "region_bank_csv": "artifacts/prediction_transition_region_banks_test_only_unbalanced/prad_morphology_group/p4_to_p5/region_bank.csv",
            },
        ]
    )
    return specs


def eligible_for_spec(spec: dict[str, str]) -> list[dict[str, str]]:
    if spec["task_name"] == "hnscc_hpv":
        return hnscc_checkpoint_slides("test", spec["source_label"])
    return manifest_slides(Path(spec["manifest_csv"]), task_name=spec["task_name"], source_label=spec["source_label"])


def region_rows_by_slide(region_bank_csv: str) -> dict[str, list[dict[str, str]]]:
    path = Path(region_bank_csv)
    if not path.exists():
        return {}
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        slide = str(row.get("slide_key") or row.get("slide_id") or "").strip()
        if not slide:
            slide = str(row.get("region_id", "")).split("__", 1)[0]
        out[slide].append(row)
    return out


def build_audit(task_direction_filters: set[str] | None = None) -> dict[str, Any]:
    eligible_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in direction_specs():
        key = f"{spec['task_name']}/{spec['direction']}"
        if task_direction_filters and key not in task_direction_filters:
            continue
        eligible = eligible_for_spec(spec)
        regions_by_slide = region_rows_by_slide(spec["region_bank_csv"])
        included_slides = 0
        excluded = Counter()
        region_count = 0
        for slide in eligible:
            base = {**spec, **slide, "task_direction": key}
            eligible_rows.append(base)
            regions = regions_by_slide.get(slide["slide_key"], [])
            if regions:
                included_slides += 1
                region_count += len(regions)
                status = "included"
                reason = ""
            else:
                status = "excluded"
                reason = "no_region_bank_rows_for_eligible_test_slide"
                excluded[reason] += 1
            audit_rows.append(
                {
                    **base,
                    "audit_status": status,
                    "exclusion_reason": reason,
                    "n_region_bank_rows": len(regions),
                }
            )
        summary_rows.append(
            {
                "task_name": spec["task_name"],
                "direction": spec["direction"],
                "source_label": spec["source_label"],
                "target_label": spec["target_label"],
                "eligible_test_slides": len(eligible),
                "eligible_test_subjects": len({row["case_id"] for row in eligible}),
                "included_slides_with_regions": included_slides,
                "excluded_slides_without_regions": len(eligible) - included_slides,
                "region_bank_rows": region_count,
                "region_bank_csv": spec["region_bank_csv"],
                "exclusion_reasons": "; ".join(f"{k}:{v}" for k, v in sorted(excluded.items())),
            }
        )
    return {"eligible": eligible_rows, "audit": audit_rows, "summary": summary_rows}


def main() -> None:
    args = build_arg_parser().parse_args()
    filters = set(args.task_direction or []) or None
    payload = build_audit(filters)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "eligible_test_slides.csv", payload["eligible"])
    write_csv(args.out_dir / "region_discovery_audit.csv", payload["audit"])
    write_csv(args.out_dir / "benchmark_denominator_summary.csv", payload["summary"])
    write_json(args.out_dir / "benchmark_denominator_summary.json", payload["summary"])
    print(json.dumps({"out_dir": str(args.out_dir), "n_directions": len(payload["summary"])}, indent=2))


if __name__ == "__main__":
    main()
