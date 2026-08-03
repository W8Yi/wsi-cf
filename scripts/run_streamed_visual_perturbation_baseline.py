#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a naive edit baseline in small chunks, score visual perturbation immediately, "
            "and delete the generated naive images after each chunk."
        )
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--direction-name", required=True)
    parser.add_argument("--runner-direction", required=True)
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ours-root", type=Path, required=True)
    parser.add_argument("--naive-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--edit-policy", type=Path, default=Path("configs/edit_policies/bad_naive_no_preserve_no_sliding.json"))
    parser.add_argument("--sae-variant", type=str, default="relu_sae_base")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--edit-support", type=str, default="padded_center_2x2")
    parser.add_argument("--window-stride-cells", type=int, default=4)
    parser.add_argument("--window-selection-mode", type=str, default="coverage")
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
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--allow-missing-ours", action="store_true")
    parser.add_argument("--keep-naive-images", action="store_true")
    parser.add_argument("--keep-temp-manifests", action="store_true")
    parser.add_argument("--write-per-cell", action="store_true")
    parser.add_argument("--formats", type=str, default="png,pdf,svg")
    return parser


def load_requests(path: Path) -> list[dict[str, Any]]:
    payload = visual.load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("edit_requests") or payload.get("requests") or []
    else:
        raise ValueError(f"Unsupported manifest payload in {path}: {type(payload)}")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        run_id = str(row.get("run_id", "")).strip()
        if run_id:
            out.append(dict(row))
    return out


def eligible_requests(
    requests: list[dict[str, Any]],
    *,
    ours_root: Path,
    generated_image_name: str,
    max_runs: int,
    allow_missing_ours: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    kept: list[dict[str, Any]] = []
    missing: list[str] = []
    for request in requests:
        run_id = str(request["run_id"])
        if not (ours_root / run_id / generated_image_name).exists():
            missing.append(run_id)
            if not allow_missing_ours:
                continue
        kept.append(request)
        if int(max_runs) > 0 and len(kept) >= int(max_runs):
            break
    return kept, missing


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
        str(args.naive_root),
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


def make_visual_args(args: argparse.Namespace, run_ids_file: Path, *, formats: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        ours_root=args.ours_root,
        naive_root=args.naive_root,
        out_dir=args.out_dir,
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
        allow_missing=False,
        title=f"{args.task_name} {args.direction_name}: visual perturbation",
        formats=args.formats if formats is None else formats,
    )


def run_stream(args: argparse.Namespace) -> dict[str, Any]:
    all_requests = load_requests(args.manifest)
    requests, missing_ours = eligible_requests(
        all_requests,
        ours_root=args.ours_root,
        generated_image_name="generated.png",
        max_runs=int(args.max_runs),
        allow_missing_ours=bool(args.allow_missing_ours),
    )
    if not requests:
        raise ValueError(f"No eligible requests found in {args.manifest}")

    tmp_dir = args.out_dir / "_stream_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    args.naive_root.mkdir(parents=True, exist_ok=True)
    accumulated_summary: list[dict[str, Any]] = []
    accumulated_cells: list[dict[str, Any]] = []
    accumulated_seams: list[dict[str, Any]] = []
    total_chunks = len(chunks(requests, int(args.chunk_size)))

    for chunk_idx, chunk in enumerate(chunks(requests, int(args.chunk_size)), start=1):
        chunk_manifest = tmp_dir / f"{args.task_name}__{args.direction_name}__chunk_{chunk_idx:05d}.json"
        run_ids_file = tmp_dir / f"{args.task_name}__{args.direction_name}__chunk_{chunk_idx:05d}.run_ids.txt"
        chunk_manifest.write_text(json.dumps(chunk, indent=2) + "\n")
        run_ids_file.write_text("\n".join(str(row["run_id"]) for row in chunk) + "\n")

        remove_run_dirs(args.naive_root, chunk)
        print(f"[chunk {chunk_idx}/{total_chunks}] generating {len(chunk)} naive run(s)", file=sys.stderr)
        subprocess.run(progressive_command(args, chunk_manifest), cwd=ROOT, check=True)

        visual_args = make_visual_args(args, run_ids_file)
        result = visual.collect_visual_rows(visual_args)
        accumulated_summary.extend(result["summary_rows"])
        accumulated_cells.extend(result["cell_rows"])
        accumulated_seams.extend(result["seam_rows"])

        merged = {
            "paired": sorted({str(row["run_id"]) for row in accumulated_summary}),
            "missing": [],
            "summary_rows": accumulated_summary,
            "cell_rows": accumulated_cells,
            "aggregate": visual.aggregate_rows(accumulated_summary),
            "paired_rows": visual.pairwise_rows(accumulated_summary, "ours", "bad_naive"),
            "paired_aggregate": visual.aggregate_pairwise(visual.pairwise_rows(accumulated_summary, "ours", "bad_naive")),
            "seam_rows": accumulated_seams,
            "seam_aggregate": visual.aggregate_seam_rows(accumulated_seams),
            "seam_pairwise": visual.pairwise_seam_rows(accumulated_seams, "ours", "bad_naive"),
            "seam_pairwise_aggregate": visual.aggregate_pairwise_seam(
                visual.pairwise_seam_rows(accumulated_seams, "ours", "bad_naive")
            ),
        }
        plot_formats = args.formats if chunk_idx == total_chunks else ""
        visual.write_visual_outputs(make_visual_args(args, run_ids_file, formats=plot_formats), merged)

        if not bool(args.keep_naive_images):
            remove_run_dirs(args.naive_root, chunk)
        if not bool(args.keep_temp_manifests):
            chunk_manifest.unlink(missing_ok=True)
            run_ids_file.unlink(missing_ok=True)

    payload = {
        "task_name": args.task_name,
        "direction_name": args.direction_name,
        "manifest": str(args.manifest),
        "ours_root": str(args.ours_root),
        "naive_root": str(args.naive_root),
        "out_dir": str(args.out_dir),
        "chunk_size": int(args.chunk_size),
        "keep_naive_images": bool(args.keep_naive_images),
        "n_requests_total": int(len(all_requests)),
        "n_requests_scored": int(len(requests)),
        "n_missing_ours": int(len(missing_ours)),
        "missing_ours_preview": missing_ours[:20],
    }
    (args.out_dir / "streamed_visual_perturbation_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(run_stream(args), indent=2))


if __name__ == "__main__":
    main()
