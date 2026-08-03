#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_prediction_transition_edits as evaluator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run prediction-transition generation in small chunks, evaluate each chunk immediately, "
            "and optionally delete generated images after scoring to save disk."
        )
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--direction-name", required=True)
    parser.add_argument("--runner-direction", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--label-order", required=True)
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--classifier-run-dir", type=Path, default=None)
    parser.add_argument("--classifier-ckpt", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--edit-policy", type=Path, required=True)
    parser.add_argument("--sae-variant", type=str, default="relu_sae_base")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--edit-support", type=str, default="padded_center_2x2")
    parser.add_argument("--window-stride-cells", type=int, default=1)
    parser.add_argument("--window-selection-mode", type=str, default="overlap")
    parser.add_argument("--commit-mode", type=str, default="full_window")
    parser.add_argument("--output-mode", type=str, default="minimal")
    parser.add_argument("--score-scope", choices=["slide_bag", "local_region"], default="local_region")
    parser.add_argument("--local-source", choices=["source_image", "feature_grid"], default="source_image")
    parser.add_argument("--concepts-json", type=Path, default=None)
    parser.add_argument("--representative-tiles-csv", type=Path, default=None)
    parser.add_argument("--concept-class-label", type=str, default="")
    parser.add_argument("--concept-ranking-method", type=str, default="attention_weighted")
    parser.add_argument("--concept-target-stat", type=str, default="median")
    parser.add_argument("--concept-target-top-k", type=int, default=5)
    parser.add_argument("--concept-steering-mode", type=str, default="prototype_vector")
    parser.add_argument("--max-concepts", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--keep-temp-manifests", action="store_true")
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--no-score-existing-images",
        action="store_true",
        help="Do not first score generated.png files that already exist but are absent from metrics.",
    )
    parser.add_argument(
        "--no-skip-scored",
        action="store_true",
        help="Regenerate/evaluate run_ids already present in prediction_transition_by_run.csv.",
    )
    return parser


def load_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("edit_requests") or payload.get("requests") or []
    else:
        raise ValueError(f"Unsupported manifest payload in {path}: {type(payload)}")
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and str(row.get("run_id", "")).strip():
            out.append(dict(row))
    return out


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def chunks(rows: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(chunk_size))
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def remove_run_dirs(root: Path, requests: list[dict[str, Any]]) -> None:
    for request in requests:
        run_dir = root / str(request["run_id"])
        if run_dir.exists():
            shutil.rmtree(run_dir)


def progressive_command(args: argparse.Namespace, manifest_path: Path) -> list[str]:
    cmd = [
        str(args.python),
        "scripts/run_progressive_region_edit.py",
        "--task",
        str(args.task_name),
        "--region-bank-csv",
        str(args.region_bank_csv),
        "--edit-manifest",
        str(manifest_path),
        "--out-dir",
        str(args.generated_root),
        "--edit-policy",
        str(args.edit_policy),
        "--direction",
        str(args.runner_direction),
        "--target-magnification",
        "20",
        "--sae-variant",
        str(args.sae_variant),
        "--steps",
        str(args.steps),
        "--patch-batch",
        str(args.patch_batch),
        "--edit-support",
        str(args.edit_support),
        "--window-stride-cells",
        str(args.window_stride_cells),
        "--window-selection-mode",
        str(args.window_selection_mode),
        "--commit-mode",
        str(args.commit_mode),
        "--output-mode",
        str(args.output_mode),
        "--device",
        str(args.device),
    ]
    if args.concepts_json is not None:
        cmd.extend(
            [
                "--concepts-json",
                str(args.concepts_json),
                "--representative-tiles-csv",
                str(args.representative_tiles_csv),
                "--concept-class-label",
                str(args.concept_class_label),
                "--concept-ranking-method",
                str(args.concept_ranking_method),
                "--concept-target-stat",
                str(args.concept_target_stat),
                "--concept-target-top-k",
                str(args.concept_target_top_k),
                "--concept-steering-mode",
                str(args.concept_steering_mode),
                "--max-concepts",
                str(args.max_concepts),
            ]
        )
    return cmd


def evaluate_chunk_args(args: argparse.Namespace, manifest_path: Path, chunk_out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        region_bank_csv=args.region_bank_csv,
        edit_manifest=manifest_path,
        generated_root=args.generated_root,
        classifier_run_dir=args.classifier_run_dir,
        classifier_ckpt=args.classifier_ckpt,
        out_dir=chunk_out_dir,
        task_name=args.task_name,
        direction=args.direction_name,
        target_label=args.target_label,
        label_order=args.label_order,
        grade_label_order="GG1,GG2,GG3,GG4,GG5",
        score_scope=args.score_scope,
        local_source=args.local_source,
        device=args.device,
        force_reencode=args.force_reencode,
        allow_missing=args.allow_missing,
        skip_source_grid_check=False,
    )


def write_merged_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], *, n_requests: int, n_missing_generated: int = 0) -> None:
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
    evaluator.write_json(
        args.out_dir / "benchmark_summary.json",
        {
            "task_name": str(args.task_name),
            "direction": str(args.direction_name),
            "region_bank_csv": str(args.region_bank_csv),
            "edit_manifest": str(args.manifest),
            "generated_root": str(args.generated_root),
            "score_scope": str(args.score_scope),
            "local_source": str(args.local_source) if str(args.score_scope) == "local_region" else "",
            "classifier_run_dir": "" if args.classifier_run_dir is None else str(args.classifier_run_dir),
            "classifier_ckpt": "" if args.classifier_ckpt is None else str(args.classifier_ckpt),
            "n_requests": int(n_requests),
            "n_scored": int(len(rows)),
            "n_missing_generated": int(n_missing_generated),
            "streamed": True,
            "keep_images": bool(args.keep_images),
            "chunk_size": int(args.chunk_size),
            "metrics": {
                "by_run": str(args.out_dir / "prediction_transition_by_run.csv"),
                "summary_by_budget": str(args.out_dir / "prediction_transition_summary_by_budget.csv"),
                "summary_by_direction": str(args.out_dir / "prediction_transition_summary_by_direction.csv"),
                "random_vs_attention": str(args.out_dir / "random_vs_attention_summary.csv"),
                "auc_by_region": str(args.out_dir / "prediction_transition_auc_by_region.csv"),
            },
        },
    )


def score_chunk(
    args: argparse.Namespace,
    *,
    chunk: list[dict[str, Any]],
    chunk_manifest: Path,
    chunk_out_dir: Path,
    chunk_idx: int,
    total_chunks: int,
    merged_by_run_id: dict[str, dict[str, Any]],
    n_requests: int,
) -> None:
    if chunk_out_dir.exists():
        shutil.rmtree(chunk_out_dir)
    print(f"[chunk {chunk_idx}/{total_chunks}] evaluating {len(chunk)} run(s)", file=sys.stderr)
    summary = evaluator.evaluate(evaluate_chunk_args(args, chunk_manifest, chunk_out_dir))
    chunk_rows = read_csv_rows(chunk_out_dir / "prediction_transition_by_run.csv")
    for row in chunk_rows:
        merged_by_run_id[str(row["run_id"])] = dict(row)
    write_merged_outputs(
        args,
        list(merged_by_run_id.values()),
        n_requests=n_requests,
        n_missing_generated=int(summary["n_missing_generated"]),
    )
    if not bool(args.keep_images):
        remove_run_dirs(args.generated_root, chunk)
    if not bool(args.keep_temp_manifests):
        shutil.rmtree(chunk_out_dir, ignore_errors=True)
        chunk_manifest.unlink(missing_ok=True)


def run_stream(args: argparse.Namespace) -> dict[str, Any]:
    all_requests = load_requests(args.manifest)
    if int(args.max_runs) > 0:
        all_requests = all_requests[: int(args.max_runs)]
    existing_rows = read_csv_rows(args.out_dir / "prediction_transition_by_run.csv")
    scored_run_ids = {str(row.get("run_id", "")) for row in existing_rows if str(row.get("run_id", "")).strip()}
    if not bool(args.no_skip_scored):
        requests = [row for row in all_requests if str(row["run_id"]) not in scored_run_ids]
    else:
        requests = list(all_requests)
        existing_rows = []
        scored_run_ids = set()

    tmp_dir = args.out_dir / "_stream_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    args.generated_root.mkdir(parents=True, exist_ok=True)
    merged_by_run_id: dict[str, dict[str, Any]] = {str(row["run_id"]): dict(row) for row in existing_rows if row.get("run_id")}

    if not bool(args.no_skip_scored) and not bool(args.no_score_existing_images):
        existing_generated = [
            row
            for row in requests
            if (args.generated_root / str(row["run_id"]) / "generated.png").exists()
        ]
        existing_chunks = chunks(existing_generated, int(args.chunk_size))
        for chunk_idx, chunk in enumerate(existing_chunks, start=1):
            chunk_manifest = tmp_dir / f"{args.task_name}__{args.direction_name}__existing_{chunk_idx:05d}.json"
            chunk_out_dir = tmp_dir / f"{args.task_name}__{args.direction_name}__existing_{chunk_idx:05d}_metrics"
            chunk_manifest.write_text(json.dumps(chunk, indent=2) + "\n")
            score_chunk(
                args,
                chunk=chunk,
                chunk_manifest=chunk_manifest,
                chunk_out_dir=chunk_out_dir,
                chunk_idx=chunk_idx,
                total_chunks=len(existing_chunks),
                merged_by_run_id=merged_by_run_id,
                n_requests=len(all_requests),
            )
        if existing_generated:
            newly_scored = {str(row["run_id"]) for row in existing_generated}
            requests = [row for row in requests if str(row["run_id"]) not in newly_scored]

    if not requests:
        write_merged_outputs(args, list(merged_by_run_id.values()), n_requests=len(all_requests))
        return {
            "task_name": args.task_name,
            "direction": args.direction_name,
            "n_requests": len(all_requests),
            "n_scored_existing": len(merged_by_run_id),
            "n_to_process": 0,
            "out_dir": str(args.out_dir),
        }

    request_chunks = chunks(requests, int(args.chunk_size))

    for chunk_idx, chunk in enumerate(request_chunks, start=1):
        chunk_manifest = tmp_dir / f"{args.task_name}__{args.direction_name}__chunk_{chunk_idx:05d}.json"
        chunk_out_dir = tmp_dir / f"{args.task_name}__{args.direction_name}__chunk_{chunk_idx:05d}_metrics"
        chunk_manifest.write_text(json.dumps(chunk, indent=2) + "\n")

        remove_run_dirs(args.generated_root, chunk)
        print(f"[chunk {chunk_idx}/{len(request_chunks)}] generating {len(chunk)} run(s)", file=sys.stderr)
        subprocess.run(progressive_command(args, chunk_manifest), cwd=ROOT, check=True)

        score_chunk(
            args,
            chunk=chunk,
            chunk_manifest=chunk_manifest,
            chunk_out_dir=chunk_out_dir,
            chunk_idx=chunk_idx,
            total_chunks=len(request_chunks),
            merged_by_run_id=merged_by_run_id,
            n_requests=len(all_requests),
        )

    payload = {
        "task_name": args.task_name,
        "direction": args.direction_name,
        "n_requests": len(all_requests),
        "n_scored_existing": len(scored_run_ids),
        "n_processed": len(requests),
        "n_scored_total": len(merged_by_run_id),
        "keep_images": bool(args.keep_images),
        "out_dir": str(args.out_dir),
    }
    evaluator.write_json(args.out_dir / "streamed_prediction_transition_summary.json", payload)
    return payload


def main() -> None:
    print(json.dumps(run_stream(build_arg_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
