#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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

import compare_edit_visual_perturbation as visual
import evaluate_prediction_transition_edits as pred_eval


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified streaming runner for the full-test paper benchmark. It generates each "
            "checkpoint once, computes prediction-transition and visual/border metrics, "
            "then deletes images/grids by default."
        )
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--direction-name", required=True)
    parser.add_argument("--runner-direction", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--label-order", required=True)
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--legacy-ours-root", type=Path, default=None)
    parser.add_argument("--legacy-naive-root", type=Path, default=None)
    parser.add_argument("--classifier-run-dir", type=Path, default=None)
    parser.add_argument("--classifier-ckpt", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ours-policy", type=Path, required=True)
    parser.add_argument("--naive-policy", type=Path, required=True)
    parser.add_argument("--sae-variant", type=str, default="relu_sae_base")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--edit-support", type=str, default="padded_center_2x2")
    parser.add_argument("--window-stride-cells", type=int, default=1)
    parser.add_argument("--window-selection-mode", type=str, default="overlap")
    parser.add_argument("--commit-mode", type=str, default="full_window")
    parser.add_argument("--output-mode", type=str, default="minimal")
    parser.add_argument("--concepts-json", type=Path, default=None)
    parser.add_argument("--representative-tiles-csv", type=Path, default=None)
    parser.add_argument("--concept-class-label", type=str, default="")
    parser.add_argument("--concept-ranking-method", type=str, default="attention_weighted")
    parser.add_argument("--concept-target-stat", type=str, default="median")
    parser.add_argument("--concept-target-top-k", type=int, default=5)
    parser.add_argument("--concept-steering-mode", type=str, default="prototype_vector")
    parser.add_argument("--max-concepts", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--score-scope", choices=["slide_bag", "local_region"], default="local_region")
    parser.add_argument("--local-source", choices=["source_image", "feature_grid"], default="source_image")
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--no-skip-scored", action="store_true")
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--keep-encoded-grids", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--keep-gallery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gallery-regions-per-direction", type=int, default=2)
    parser.add_argument("--gallery-budgets", type=str, default="1,32,64")
    parser.add_argument("--write-per-cell", action="store_true")
    parser.add_argument("--formats", type=str, default="png,pdf,svg")
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_requests(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload if isinstance(payload, list) else payload.get("edit_requests") or payload.get("requests") or []
    return [dict(row) for row in rows if isinstance(row, dict) and str(row.get("run_id", "")).strip()]


def chunks(rows: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(chunk_size))
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def request_cache_key(args: argparse.Namespace, request: dict[str, Any], *, method: str) -> str:
    payload = {
        "method": method,
        "task_name": args.task_name,
        "direction_name": args.direction_name,
        "run_id": request.get("run_id"),
        "region_id": request.get("region_id"),
        "target_cells": request.get("target_cells", []),
        "selector": request.get("selector", ""),
        "repeat_id": request.get("repeat_id", -1),
        "budget": request.get("budget", ""),
        "policy_sha": file_sha(args.ours_policy if method == "ours" else args.naive_policy),
        "sae_variant": args.sae_variant,
        "steps": int(args.steps),
        "patch_batch": int(args.patch_batch),
        "edit_support": args.edit_support,
        "window_stride_cells": int(args.window_stride_cells),
        "window_selection_mode": args.window_selection_mode,
        "commit_mode": args.commit_mode,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def add_provenance(args: argparse.Namespace, rows: list[dict[str, Any]], requests_by_run: dict[str, dict[str, Any]], *, method: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row.get("run_id", ""))
        request = requests_by_run.get(run_id, {})
        row_method = str(method or row.get("method") or "ours")
        out.append(
            {
                **row,
                "method": row_method,
                "generation_mode": "cumulative_checkpoint",
                "trajectory_cache_key": request_cache_key(args, request or row, method=row_method),
            }
        )
    return out


def remove_run_dirs(root: Path, requests: list[dict[str, Any]]) -> None:
    for request in requests:
        run_dir = root / str(request["run_id"])
        if run_dir.exists():
            shutil.rmtree(run_dir)


def copy_legacy_run_dirs(legacy_root: Path | None, tmp_root: Path, requests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    if legacy_root is None:
        return copied, list(requests)
    for request in requests:
        run_id = str(request["run_id"])
        src = legacy_root / run_id
        dst = tmp_root / run_id
        if (src / "generated.png").exists() and (src / "run_manifest.json").exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied.append(request)
        else:
            missing.append(request)
    return copied, missing


def maybe_copy_gallery(args: argparse.Namespace, requests: list[dict[str, Any]], *, method: str, root: Path) -> None:
    if not bool(args.keep_gallery):
        return
    wanted_budgets = {int(token) for token in str(args.gallery_budgets).split(",") if token.strip()}
    allowed_regions: list[str] = []
    for request in requests:
        region_id = str(request.get("region_id", ""))
        if str(request.get("selector", "")) != "attention" or int(request.get("repeat_id", -1)) != -1:
            continue
        if region_id not in allowed_regions:
            allowed_regions.append(region_id)
        if len(allowed_regions) >= int(args.gallery_regions_per_direction):
            break
    allowed_region_set = set(allowed_regions)
    gallery_root = args.out_root / "gallery" / args.task_name / args.direction_name / method
    for request in requests:
        if str(request.get("region_id", "")) not in allowed_region_set:
            continue
        if str(request.get("selector", "")) != "attention" or int(request.get("repeat_id", -1)) != -1:
            continue
        if int(request.get("budget", -1)) not in wanted_budgets:
            continue
        run_id = str(request["run_id"])
        src_dir = root / run_id
        if not src_dir.exists():
            continue
        dst_dir = gallery_root / run_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ["source_region_actual.png", "generated.png", "run_manifest.json"]:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)


def progressive_command(args: argparse.Namespace, manifest_path: Path, *, out_dir: Path, policy: Path) -> list[str]:
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
        str(out_dir),
        "--edit-policy",
        str(policy),
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


def pred_args(args: argparse.Namespace, manifest_path: Path, generated_root: Path, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        region_bank_csv=args.region_bank_csv,
        edit_manifest=manifest_path,
        generated_root=generated_root,
        classifier_run_dir=args.classifier_run_dir,
        classifier_ckpt=args.classifier_ckpt,
        out_dir=out_dir,
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


def visual_args(args: argparse.Namespace, run_ids_file: Path, ours_root: Path, naive_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        ours_root=ours_root,
        naive_root=naive_root,
        out_dir=args.out_root / "metrics_visual" / args.task_name / args.direction_name,
        manifest=args.manifest,
        ours_name="ours",
        naive_name="bad_naive",
        source_image_name="source_region_actual.png",
        generated_image_name="generated.png",
        run_manifest_name="run_manifest.json",
        grid_step_px=256,
        max_runs=0,
        run_ids_file=run_ids_file,
        no_per_cell=not bool(args.write_per_cell),
        no_seam=False,
        allow_missing=args.allow_missing,
        title=f"{args.task_name} {args.direction_name}: full-test streaming visual perturbation",
        formats="",
    )


def write_prediction_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], n_requests: int) -> None:
    out_dir = args.out_root / "metrics" / args.task_name / args.direction_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "prediction_transition_by_checkpoint.csv", rows)
    write_csv(out_dir / "prediction_transition_summary_by_budget.csv", pred_eval.summarize_group(rows, ["task_name", "direction", "method", "selector", "budget"]))
    write_csv(out_dir / "prediction_transition_summary_by_direction.csv", pred_eval.summarize_group(rows, ["task_name", "direction", "method", "selector"]))
    write_csv(out_dir / "random_vs_attention_summary.csv", pred_eval.random_vs_attention_summary(rows))
    write_json(
        out_dir / "benchmark_summary.json",
        {
            "task_name": args.task_name,
            "direction": args.direction_name,
            "n_requests": int(n_requests),
            "n_scored": int(len(rows)),
            "keep_images": bool(args.keep_images),
            "keep_encoded_grids": bool(args.keep_encoded_grids),
            "generation_mode": "cumulative_checkpoint",
        },
    )


def write_visual_outputs(args: argparse.Namespace, summary_rows: list[dict[str, Any]], seam_rows: list[dict[str, Any]]) -> None:
    out_dir = args.out_root / "metrics_visual" / args.task_name / args.direction_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_rows = visual.pairwise_rows(summary_rows, "ours", "bad_naive")
    seam_pairwise = visual.pairwise_seam_rows(seam_rows, "ours", "bad_naive")
    write_csv(out_dir / "visual_perturbation_by_checkpoint.csv", summary_rows)
    write_csv(out_dir / "visual_perturbation_summary_by_budget.csv", visual.aggregate_rows(summary_rows))
    write_csv(out_dir / "ours_vs_naive_paired_by_checkpoint.csv", paired_rows)
    write_csv(out_dir / "ours_vs_naive_paired_summary.csv", visual.aggregate_pairwise(paired_rows))
    write_csv(out_dir / "border_discontinuity_by_checkpoint.csv", seam_rows)
    write_csv(out_dir / "border_discontinuity_summary.csv", visual.aggregate_seam_rows(seam_rows))
    write_csv(out_dir / "border_discontinuity_paired_by_checkpoint.csv", seam_pairwise)
    write_csv(out_dir / "border_discontinuity_paired_summary.csv", visual.aggregate_pairwise_seam(seam_pairwise))
    write_json(
        out_dir / "visual_perturbation_summary.json",
        {
            "task_name": args.task_name,
            "direction": args.direction_name,
            "n_visual_rows": int(len(summary_rows)),
            "n_border_rows": int(len(seam_rows)),
            "keep_images": bool(args.keep_images),
        },
    )


def completed_run_ids(args: argparse.Namespace) -> set[str]:
    if bool(args.no_skip_scored):
        return set()
    pred_rows = read_csv_rows(args.out_root / "metrics" / args.task_name / args.direction_name / "prediction_transition_by_checkpoint.csv")
    visual_rows = read_csv_rows(args.out_root / "metrics_visual" / args.task_name / args.direction_name / "visual_perturbation_by_checkpoint.csv")
    pred_ids = {str(row.get("run_id", "")) for row in pred_rows if str(row.get("run_id", "")).strip()}
    visual_ids = {str(row.get("run_id", "")) for row in visual_rows if str(row.get("run_id", "")).strip()}
    return pred_ids & visual_ids


def run_stream(args: argparse.Namespace) -> dict[str, Any]:
    all_requests = load_requests(args.manifest)
    if int(args.max_runs) > 0:
        all_requests = all_requests[: int(args.max_runs)]
    requests_by_run = {str(row["run_id"]): dict(row) for row in all_requests}
    done = completed_run_ids(args)
    pending = [row for row in all_requests if str(row["run_id"]) not in done]
    args.out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out_root / "_stream_tmp" / args.task_name / args.direction_name
    ours_root = args.out_root / "_generated_tmp" / "ours" / args.task_name / args.direction_name
    naive_root = args.out_root / "_generated_tmp" / "bad_naive" / args.task_name / args.direction_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ours_root.mkdir(parents=True, exist_ok=True)
    naive_root.mkdir(parents=True, exist_ok=True)

    pred_rows = read_csv_rows(args.out_root / "metrics" / args.task_name / args.direction_name / "prediction_transition_by_checkpoint.csv")
    visual_rows_acc = read_csv_rows(args.out_root / "metrics_visual" / args.task_name / args.direction_name / "visual_perturbation_by_checkpoint.csv")
    seam_rows_acc = read_csv_rows(args.out_root / "metrics_visual" / args.task_name / args.direction_name / "border_discontinuity_by_checkpoint.csv")
    progress_path = args.out_root / "streaming_progress.jsonl"
    failed_path = args.out_root / "failed_checkpoints.csv"
    failed_rows: list[dict[str, Any]] = read_csv_rows(failed_path)

    total_chunks = len(chunks(pending, int(args.chunk_size)))
    for chunk_idx, chunk in enumerate(chunks(pending, int(args.chunk_size)), start=1):
        chunk_manifest = tmp_dir / f"chunk_{chunk_idx:06d}.json"
        run_ids_file = tmp_dir / f"chunk_{chunk_idx:06d}.run_ids.txt"
        chunk_pred_dir = tmp_dir / f"chunk_{chunk_idx:06d}_prediction"
        chunk_manifest.write_text(json.dumps(chunk, indent=2) + "\n")
        run_ids_file.write_text("\n".join(str(row["run_id"]) for row in chunk) + "\n")
        try:
            remove_run_dirs(ours_root, chunk)
            remove_run_dirs(naive_root, chunk)
            copied_ours, missing_ours = copy_legacy_run_dirs(args.legacy_ours_root, ours_root, chunk)
            if missing_ours:
                missing_ours_manifest = tmp_dir / f"chunk_{chunk_idx:06d}_missing_ours.json"
                missing_ours_manifest.write_text(json.dumps(missing_ours, indent=2) + "\n")
                print(
                    f"[chunk {chunk_idx}/{total_chunks}] ours generation: {len(missing_ours)} run(s); reused {len(copied_ours)}",
                    file=sys.stderr,
                )
                subprocess.run(progressive_command(args, missing_ours_manifest, out_dir=ours_root, policy=args.ours_policy), cwd=ROOT, check=True)
                if not bool(args.keep_temp):
                    missing_ours_manifest.unlink(missing_ok=True)
            else:
                print(f"[chunk {chunk_idx}/{total_chunks}] ours generation skipped; reused {len(copied_ours)}", file=sys.stderr)
            maybe_copy_gallery(args, chunk, method="ours", root=ours_root)

            pred_summary = pred_eval.evaluate(pred_args(args, chunk_manifest, ours_root, chunk_pred_dir))
            chunk_pred_rows = add_provenance(
                args,
                read_csv_rows(chunk_pred_dir / "prediction_transition_by_run.csv"),
                requests_by_run,
                method="ours",
            )
            pred_rows_by_id = {str(row["run_id"]): row for row in pred_rows}
            for row in chunk_pred_rows:
                pred_rows_by_id[str(row["run_id"])] = row
            pred_rows = list(pred_rows_by_id.values())

            copied_naive, missing_naive = copy_legacy_run_dirs(args.legacy_naive_root, naive_root, chunk)
            if missing_naive:
                missing_naive_manifest = tmp_dir / f"chunk_{chunk_idx:06d}_missing_naive.json"
                missing_naive_manifest.write_text(json.dumps(missing_naive, indent=2) + "\n")
                print(
                    f"[chunk {chunk_idx}/{total_chunks}] bad-naive generation: {len(missing_naive)} run(s); reused {len(copied_naive)}",
                    file=sys.stderr,
                )
                subprocess.run(progressive_command(args, missing_naive_manifest, out_dir=naive_root, policy=args.naive_policy), cwd=ROOT, check=True)
                if not bool(args.keep_temp):
                    missing_naive_manifest.unlink(missing_ok=True)
            else:
                print(f"[chunk {chunk_idx}/{total_chunks}] bad-naive generation skipped; reused {len(copied_naive)}", file=sys.stderr)
            maybe_copy_gallery(args, chunk, method="bad_naive", root=naive_root)

            v_args = visual_args(args, run_ids_file, ours_root, naive_root)
            visual_result = visual.collect_visual_rows(v_args)
            new_visual_rows = add_provenance(args, list(visual_result["summary_rows"]), requests_by_run, method="")
            new_seam_rows = add_provenance(args, list(visual_result["seam_rows"]), requests_by_run, method="")
            visual_rows_acc = [row for row in visual_rows_acc if str(row.get("run_id", "")) not in {str(req["run_id"]) for req in chunk}]
            seam_rows_acc = [row for row in seam_rows_acc if str(row.get("run_id", "")) not in {str(req["run_id"]) for req in chunk}]
            visual_rows_acc.extend(new_visual_rows)
            seam_rows_acc.extend(new_seam_rows)

            write_prediction_outputs(args, pred_rows, n_requests=len(all_requests))
            write_visual_outputs(args, visual_rows_acc, seam_rows_acc)
            with progress_path.open("a") as handle:
                handle.write(json.dumps({"event": "chunk_complete", "task_name": args.task_name, "direction": args.direction_name, "chunk_idx": chunk_idx, "n_runs": len(chunk), "prediction": pred_summary}) + "\n")
        except Exception as exc:
            for request in chunk:
                failed_rows.append(
                    {
                        "task_name": args.task_name,
                        "direction": args.direction_name,
                        "run_id": str(request["run_id"]),
                        "region_id": str(request.get("region_id", "")),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            write_csv(failed_path, failed_rows)
            raise
        finally:
            if not bool(args.keep_images):
                remove_run_dirs(ours_root, chunk)
                remove_run_dirs(naive_root, chunk)
            if not bool(args.keep_encoded_grids):
                shutil.rmtree(chunk_pred_dir / "encoded_generated_grids", ignore_errors=True)
                shutil.rmtree(chunk_pred_dir / "encoded_source_grids", ignore_errors=True)
            if not bool(args.keep_temp):
                shutil.rmtree(chunk_pred_dir, ignore_errors=True)
                chunk_manifest.unlink(missing_ok=True)
                run_ids_file.unlink(missing_ok=True)

    if not bool(args.keep_images):
        shutil.rmtree(ours_root, ignore_errors=True)
        shutil.rmtree(naive_root, ignore_errors=True)
    payload = {
        "task_name": args.task_name,
        "direction": args.direction_name,
        "manifest": str(args.manifest),
        "n_requests": int(len(all_requests)),
        "n_previously_completed": int(len(done)),
        "n_processed": int(len(pending)),
        "out_root": str(args.out_root),
        "keep_images": bool(args.keep_images),
        "keep_encoded_grids": bool(args.keep_encoded_grids),
        "generation_mode": "cumulative_checkpoint",
    }
    write_json(args.out_root / "streamed_full_test_summary" / args.task_name / f"{args.direction_name}.json", payload)
    return payload


def main() -> None:
    print(json.dumps(run_stream(build_arg_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
