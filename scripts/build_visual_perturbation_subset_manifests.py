#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE_ROOT = Path("paper_outputs/prediction_transition_benchmark_test_only_unbalanced")
DEFAULT_OUT_ROOT = Path("paper_outputs/visual_perturbation_paper_subset")

DEFAULT_TASK_DIRECTIONS = [
    "hnscc_hpv/hpv_pos_to_hpv_neg",
    "luad_normal_tumor/normal_to_tumor",
    "prad_morphology_group/p4_to_p5",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small deterministic visual-perturbation subset from an existing "
            "prediction-transition benchmark. The subset keeps attention-ranked runs "
            "at a few budgets so our edit policy and the bad naive policy can be "
            "regenerated and compared without storing the full benchmark images."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--task-direction",
        action="append",
        default=None,
        help=(
            "Task/direction key, e.g. hnscc_hpv/hpv_pos_to_hpv_neg. "
            "Can be repeated. Defaults to the recommended paper subset."
        ),
    )
    parser.add_argument("--budgets", type=str, default="1,8,16,32")
    parser.add_argument("--regions-per-direction", type=int, default=4)
    parser.add_argument(
        "--slides-per-direction",
        type=int,
        default=0,
        help=(
            "If >0, select regions in a slide-balanced way: up to this many "
            "slides per task/direction. Takes precedence over "
            "--regions-per-direction."
        ),
    )
    parser.add_argument(
        "--regions-per-slide",
        type=int,
        default=0,
        help=(
            "When --slides-per-direction is used, select up to this many complete "
            "regions from each selected slide. Defaults to --regions-per-direction."
        ),
    )
    parser.add_argument("--selector", type=str, default="attention")
    parser.add_argument(
        "--region-sort",
        type=str,
        choices=["first", "best_delta"],
        default="first",
        help="How to choose regions with complete requested budgets.",
    )
    return parser


def load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def load_manifest_requests(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("edit_requests") or payload.get("requests") or []
    else:
        raise ValueError(f"Unsupported manifest payload in {path}: {type(payload)}")
    return [dict(row) for row in rows if isinstance(row, dict) and str(row.get("run_id", "")).strip()]


def parse_budgets(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Budgets must be positive integers, got {value}")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("No budgets requested")
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_int(row: dict[str, Any], key: str) -> int:
    return int(float(str(row.get(key, "")).strip()))


def as_float(row: dict[str, Any], key: str) -> float:
    text = str(row.get(key, "")).strip()
    return float(text) if text else float("nan")


def select_region_ids(
    rows: list[dict[str, str]],
    *,
    budgets: list[int],
    selector: str,
    regions_per_direction: int,
    slides_per_direction: int = 0,
    regions_per_slide: int = 0,
    region_sort: str,
) -> tuple[list[str], dict[str, Any]]:
    wanted_budgets = set(int(v) for v in budgets)
    by_region: dict[str, dict[int, dict[str, str]]] = {}
    region_order: list[str] = []
    for row in rows:
        if str(row.get("selector", "")) != selector:
            continue
        try:
            budget = as_int(row, "budget")
        except ValueError:
            continue
        if budget not in wanted_budgets:
            continue
        region_id = str(row.get("region_id", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if not region_id or not run_id:
            continue
        if region_id not in by_region:
            by_region[region_id] = {}
            region_order.append(region_id)
        by_region[region_id].setdefault(budget, row)

    complete = [region_id for region_id in region_order if wanted_budgets.issubset(set(by_region[region_id]))]
    if region_sort == "best_delta":
        report_budget = max(wanted_budgets)

        def sort_key(region_id: str) -> tuple[float, str]:
            row = by_region[region_id][report_budget]
            return (-as_float(row, "target_probability_delta"), region_id)

        complete = sorted(complete, key=sort_key)

    if int(slides_per_direction) <= 0:
        selected = complete[: max(0, int(regions_per_direction))]
        slide_keys = [str(by_region[region_id][min(wanted_budgets)].get("slide_key", "")).strip() for region_id in selected]
        return selected, {
            "selection_mode": "regions_per_direction",
            "requested_regions_per_direction": int(regions_per_direction),
            "available_complete_regions": len(complete),
            "n_selected_slides": len(set(slide_keys)),
            "selected_slide_keys": sorted(set(slide_keys)),
        }

    per_slide = int(regions_per_slide) if int(regions_per_slide) > 0 else int(regions_per_direction)
    if per_slide <= 0:
        raise ValueError("--regions-per-slide must be positive when --slides-per-direction is used")

    by_slide: dict[str, list[str]] = {}
    slide_order: list[str] = []
    for region_id in complete:
        row = by_region[region_id][min(wanted_budgets)]
        slide_key = str(row.get("slide_key", "")).strip()
        if not slide_key:
            slide_key = str(row.get("slide_id", "")).strip()
        if not slide_key:
            slide_key = region_id.split("__", 1)[0]
        if slide_key not in by_slide:
            by_slide[slide_key] = []
            slide_order.append(slide_key)
        by_slide[slide_key].append(region_id)

    eligible_slides = [slide_key for slide_key in slide_order if len(by_slide[slide_key]) >= per_slide]
    selected_slides = eligible_slides[: max(0, int(slides_per_direction))]
    selected: list[str] = []
    for slide_key in selected_slides:
        selected.extend(by_slide[slide_key][:per_slide])

    return selected, {
        "selection_mode": "slides_per_direction",
        "requested_slides_per_direction": int(slides_per_direction),
        "requested_regions_per_slide": per_slide,
        "available_complete_regions": len(complete),
        "available_eligible_slides": len(eligible_slides),
        "n_selected_slides": len(selected_slides),
        "selected_slide_keys": selected_slides,
        "selected_regions_per_slide": {slide_key: len(by_slide[slide_key][:per_slide]) for slide_key in selected_slides},
    }


def build_subset_for_task(
    *,
    reference_root: Path,
    out_root: Path,
    task_name: str,
    direction: str,
    budgets: list[int],
    selector: str,
    regions_per_direction: int,
    slides_per_direction: int,
    regions_per_slide: int,
    region_sort: str,
) -> dict[str, Any]:
    metrics_csv = reference_root / "metrics" / task_name / direction / "prediction_transition_by_run.csv"
    manifest_path = reference_root / "manifests" / task_name / direction / "combined_manifest.json"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing prediction metrics: {metrics_csv}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

    metric_rows = read_csv_rows(metrics_csv)
    requests = load_manifest_requests(manifest_path)
    requests_by_run_id = {str(row["run_id"]): row for row in requests}
    selected_region_ids, selection_summary = select_region_ids(
        metric_rows,
        budgets=budgets,
        selector=selector,
        regions_per_direction=regions_per_direction,
        slides_per_direction=slides_per_direction,
        regions_per_slide=regions_per_slide,
        region_sort=region_sort,
    )
    selected_region_set = set(selected_region_ids)
    selected_metric_rows = [
        row
        for row in metric_rows
        if str(row.get("selector", "")) == selector
        and str(row.get("region_id", "")).strip() in selected_region_set
        and as_int(row, "budget") in set(budgets)
    ]
    selected_metric_rows.sort(
        key=lambda row: (
            selected_region_ids.index(str(row["region_id"])),
            budgets.index(as_int(row, "budget")),
        )
    )

    subset_requests: list[dict[str, Any]] = []
    subset_index_rows: list[dict[str, Any]] = []
    missing_requests: list[str] = []
    for row in selected_metric_rows:
        run_id = str(row["run_id"])
        request = requests_by_run_id.get(run_id)
        if request is None:
            missing_requests.append(run_id)
            continue
        subset_requests.append(dict(request))
        subset_index_rows.append(
            {
                "task_name": task_name,
                "direction": direction,
                "run_id": run_id,
                "region_id": row.get("region_id", ""),
                "slide_key": row.get("slide_key", ""),
                "selector": row.get("selector", ""),
                "budget": as_int(row, "budget"),
                "source_label": row.get("source_label", ""),
                "target_label": row.get("target_label", ""),
                "source_target_probability": row.get("source_target_probability", ""),
                "edited_target_probability": row.get("edited_target_probability", ""),
                "target_probability_delta": row.get("target_probability_delta", ""),
            }
        )

    task_out = out_root / "manifests" / task_name / direction
    task_out.mkdir(parents=True, exist_ok=True)
    subset_manifest = task_out / "visual_subset_manifest.json"
    subset_index = task_out / "visual_subset_index.csv"
    subset_manifest.write_text(json.dumps(subset_requests, indent=2) + "\n")
    write_csv(subset_index, subset_index_rows)
    return {
        "task_name": task_name,
        "direction": direction,
        "source_metrics": str(metrics_csv),
        "source_manifest": str(manifest_path),
        "subset_manifest": str(subset_manifest),
        "subset_index": str(subset_index),
        "selector": selector,
        "budgets": budgets,
        "region_sort": region_sort,
        "n_regions": len(selected_region_ids),
        "region_ids": selected_region_ids,
        "n_requests": len(subset_requests),
        "n_missing_requests": len(missing_requests),
        "missing_request_preview": missing_requests[:20],
        **selection_summary,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    budgets = parse_budgets(args.budgets)
    task_directions = args.task_direction or DEFAULT_TASK_DIRECTIONS
    summaries: list[dict[str, Any]] = []
    for item in task_directions:
        if "/" not in item:
            raise ValueError(f"--task-direction must look like task/direction, got {item!r}")
        task_name, direction = item.split("/", 1)
        summaries.append(
            build_subset_for_task(
                reference_root=args.reference_root,
                out_root=args.out_root,
                task_name=task_name,
                direction=direction,
                budgets=budgets,
                selector=args.selector,
                regions_per_direction=int(args.regions_per_direction),
                slides_per_direction=int(args.slides_per_direction),
                regions_per_slide=int(args.regions_per_slide),
                region_sort=str(args.region_sort),
            )
        )
    payload = {
        "reference_root": str(args.reference_root),
        "out_root": str(args.out_root),
        "task_directions": task_directions,
        "budgets": budgets,
        "selector": str(args.selector),
        "regions_per_direction": int(args.regions_per_direction),
        "slides_per_direction": int(args.slides_per_direction),
        "regions_per_slide": int(args.regions_per_slide),
        "region_sort": str(args.region_sort),
        "summaries": summaries,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "visual_subset_manifest_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
