#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_prediction_transition_edits as evaluator  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge prediction-transition evaluator shard outputs into the canonical metric files."
    )
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--edit-manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--classifier-run-dir", type=Path, default=None)
    parser.add_argument("--classifier-ckpt", type=Path, default=None)
    parser.add_argument("--score-scope", default="local_region")
    parser.add_argument("--local-source", default="source_image")
    parser.add_argument("--strict-count", action="store_true", help="Fail if merged row count differs from manifest length.")
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def merge(args: argparse.Namespace) -> dict[str, Any]:
    shard_files = sorted(args.shard_root.glob("shard_*/prediction_transition_by_run.csv"))
    if not shard_files:
        raise FileNotFoundError(f"No shard prediction CSVs found under {args.shard_root}")

    rows_by_run: dict[str, dict[str, Any]] = {}
    duplicate_run_ids: list[str] = []
    for path in shard_files:
        for row in read_csv_rows(path):
            run_id = str(row.get("run_id", ""))
            if not run_id:
                raise ValueError(f"Row without run_id in {path}")
            if run_id in rows_by_run:
                duplicate_run_ids.append(run_id)
            rows_by_run[run_id] = row

    rows = sorted(
        rows_by_run.values(),
        key=lambda row: (
            str(row.get("task_name", "")),
            str(row.get("direction", "")),
            str(row.get("region_id", "")),
            str(row.get("selector", "")),
            int(float(row.get("repeat_id") or -1)),
            int(float(row.get("budget") or 0)),
            str(row.get("run_id", "")),
        ),
    )
    expected_requests = read_json(args.edit_manifest)
    if args.strict_count and len(rows) != len(expected_requests):
        raise ValueError(
            f"Merged {len(rows)} scored rows, but manifest has {len(expected_requests)} requests. "
            f"Missing {len(expected_requests) - len(rows)}."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    evaluator.write_csv(args.out_dir / "prediction_transition_by_run.csv", rows)
    evaluator.write_csv(
        args.out_dir / "prediction_transition_summary_by_budget.csv",
        evaluator.summarize_group(rows, ["task_name", "direction", "selector", "budget"]),
    )
    evaluator.write_csv(
        args.out_dir / "prediction_transition_summary_by_direction.csv",
        evaluator.summarize_group(rows, ["task_name", "direction", "selector"]),
    )
    evaluator.write_csv(args.out_dir / "random_vs_attention_summary.csv", evaluator.random_vs_attention_summary(rows))
    evaluator.write_csv(args.out_dir / "prediction_transition_auc_by_region.csv", evaluator.transition_auc_rows(rows))

    summary = {
        "task_name": str(args.task_name),
        "direction": str(args.direction),
        "region_bank_csv": str(args.region_bank_csv),
        "edit_manifest": str(args.edit_manifest),
        "generated_root": str(args.generated_root),
        "score_scope": str(args.score_scope),
        "local_source": str(args.local_source) if str(args.score_scope) == "local_region" else "",
        "classifier_run_dir": "" if args.classifier_run_dir is None else str(args.classifier_run_dir),
        "classifier_ckpt": "" if args.classifier_ckpt is None else str(args.classifier_ckpt),
        "shard_root": str(args.shard_root),
        "shard_files": [str(path) for path in shard_files],
        "n_requests": int(len(expected_requests)),
        "n_scored": int(len(rows)),
        "n_duplicate_run_ids": int(len(duplicate_run_ids)),
        "duplicate_run_ids": duplicate_run_ids[:100],
        "metrics": {
            "by_run": str(args.out_dir / "prediction_transition_by_run.csv"),
            "summary_by_budget": str(args.out_dir / "prediction_transition_summary_by_budget.csv"),
            "summary_by_direction": str(args.out_dir / "prediction_transition_summary_by_direction.csv"),
            "random_vs_attention": str(args.out_dir / "random_vs_attention_summary.csv"),
            "auc_by_region": str(args.out_dir / "prediction_transition_auc_by_region.csv"),
        },
    }
    evaluator.write_json(args.out_dir / "benchmark_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(merge(build_arg_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
