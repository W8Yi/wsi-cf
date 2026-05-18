#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.common.paths import DEFAULT_SAE_CFG, DEFAULT_SAE_CKPT, DEFAULT_SAE_VARIANT, SAE_VARIANTS, resolve_sae_paths
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features


DEFAULT_TASKS = "hnsc_hpv,cesc_hpv,kirc_grade,msi_coad_stad"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CSV-only representative tile rankings for task-associated SAE latents. "
            "Tiles are ranked by raw SAE latent activation."
        )
    )
    parser.add_argument("--association-root", type=Path, default=WSI_CF_ROOT / "artifacts/concept_label_associations_all")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/representative_tiles_all_tasks")
    parser.add_argument("--metric", type=str, default="fraction")
    parser.add_argument("--top-latents-per-class", type=int, default=20)
    parser.add_argument("--top-tiles-per-latent", type=int, default=25)
    parser.add_argument("--min-cohen-d", type=float, default=0.0)
    parser.add_argument("--max-slides-per-task", type=int, default=0)
    parser.add_argument("--max-slides-per-cohort", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--sae-ckpt", type=Path, default=None, help=f"Explicit SAE checkpoint override. Defaults to --sae-variant ({DEFAULT_SAE_CKPT}).")
    parser.add_argument("--sae-cfg", type=Path, default=None, help=f"Explicit SAE config override. Defaults to --sae-variant ({DEFAULT_SAE_CFG}).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-per-task-files", action="store_true")
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(str(key))
                    seen.add(str(key))
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_tasks(value: str) -> list[str]:
    tasks = [token.strip() for token in str(value).split(",") if token.strip()]
    if not tasks:
        raise ValueError("--tasks must contain at least one task id")
    return tasks


def read_h5_features_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        feats = np.asarray(handle["features"][:], dtype=np.float32)
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if feats.ndim != 2:
        raise ValueError(f"{path}: expected features [N,D] or [1,N,D], got {feats.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path}: expected coords [N,2] or [1,N,2], got {coords.shape}")
    if int(feats.shape[0]) != int(coords.shape[0]):
        raise ValueError(f"{path}: features N={feats.shape[0]} but coords N={coords.shape[0]}")
    return feats, coords


def select_latents_for_task(
    *,
    task: str,
    task_dir: Path,
    metric: str,
    top_latents_per_class: int,
    min_cohen_d: float,
) -> list[dict[str, Any]]:
    assoc_path = task_dir / "latent_label_associations.csv"
    if not assoc_path.exists():
        raise FileNotFoundError(f"Missing association file for task={task}: {assoc_path}")
    rows = read_csv_rows(assoc_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("metric", "")) != str(metric):
            continue
        diff = float(row.get("diff_class_minus_rest", 0.0))
        cohen_d = float(row.get("cohen_d", 0.0))
        if diff <= 0.0 or cohen_d < float(min_cohen_d):
            continue
        class_label = str(row["class_label"])
        item = {
            "task": task,
            "latent_idx": int(row["latent_idx"]),
            "metric": str(row["metric"]),
            "class_label": class_label,
            "n_class": int(row["n_class"]),
            "n_rest": int(row["n_rest"]),
            "mean_class": float(row["mean_class"]),
            "mean_rest": float(row["mean_rest"]),
            "diff_class_minus_rest": diff,
            "abs_diff": float(row["abs_diff"]),
            "cohen_d": cohen_d,
        }
        grouped[class_label].append(item)

    selected: list[dict[str, Any]] = []
    for class_label, class_rows in sorted(grouped.items()):
        class_rows.sort(key=lambda r: (-float(r["cohen_d"]), -float(r["abs_diff"]), int(r["latent_idx"])))
        for rank, row in enumerate(class_rows[: max(0, int(top_latents_per_class))], start=1):
            selected.append({**row, "latent_rank_in_class": int(rank)})
    return selected


def heap_push_top(heap: list[tuple[float, int, dict[str, Any]]], row: dict[str, Any], *, limit: int, counter: int) -> int:
    score = float(row["activation"])
    item = (score, int(counter), row)
    if len(heap) < int(limit):
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)
    return int(counter) + 1


def heap_to_ranked_rows(heap: list[tuple[float, int, dict[str, Any]]], *, rank_key: str) -> list[dict[str, Any]]:
    rows = [item[2] for item in heap]
    rows.sort(key=lambda r: (-float(r["activation"]), str(r["slide_key"]), int(r["tile_index"])))
    return [{**row, rank_key: int(rank)} for rank, row in enumerate(rows, start=1)]


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.sae_ckpt, args.sae_cfg = resolve_sae_paths(args.sae_variant, args.sae_ckpt, args.sae_cfg)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expected_outputs = [
        args.out_dir / "task_representative_tiles.csv",
        args.out_dir / "cohort_representative_tiles.csv",
        args.out_dir / "selected_latents.csv",
        args.out_dir / "summary.json",
    ]
    if bool(args.skip_existing) and all(path.exists() for path in expected_outputs):
        print(f"[skip] representative tile outputs already exist in {args.out_dir}")
        return

    sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    sae_model.eval()

    tasks = parse_tasks(args.tasks)
    all_selected_latents: list[dict[str, Any]] = []
    task_heaps: dict[tuple[str, str, int], list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    cohort_heaps: dict[tuple[str, str, str, int], list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    heap_counter = 0
    summary: dict[str, Any] = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:]))),
        "tasks": {},
        "skipped_slides": [],
        "sae_d_in": int(d_in),
        "sae_d_latent": int(d_latent),
    }

    for task in tasks:
        task_dir = args.association_root / task
        cohort_path = task_dir / "cohort_slides.csv"
        task_summary = {
            "task_dir": str(task_dir),
            "selected_latent_count": 0,
            "slides_seen": 0,
            "slides_processed": 0,
            "slides_skipped": 0,
            "tiles_processed": 0,
            "classes": {},
            "cohorts": {},
        }
        if not cohort_path.exists():
            summary["tasks"][task] = {**task_summary, "error": f"Missing {cohort_path}"}
            continue

        selected_latents = select_latents_for_task(
            task=task,
            task_dir=task_dir,
            metric=str(args.metric),
            top_latents_per_class=int(args.top_latents_per_class),
            min_cohen_d=float(args.min_cohen_d),
        )
        all_selected_latents.extend(selected_latents)
        task_summary["selected_latent_count"] = int(len(selected_latents))
        latents_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected_latents:
            latents_by_label[str(row["class_label"])].append(row)
        for label, rows in latents_by_label.items():
            task_summary["classes"][label] = int(len(rows))

        slides = read_csv_rows(cohort_path)
        processed_by_cohort: dict[str, int] = defaultdict(int)
        processed_for_task = 0
        for slide_row in slides:
            task_summary["slides_seen"] += 1
            if int(args.max_slides_per_task) > 0 and processed_for_task >= int(args.max_slides_per_task):
                break
            label = str(slide_row.get("label", ""))
            project_dir = str(slide_row.get("project_dir", ""))
            h5_path = Path(str(slide_row.get("h5_path", "")))
            relevant_latents = latents_by_label.get(label, [])
            if not relevant_latents:
                continue
            if int(args.max_slides_per_cohort) > 0 and processed_by_cohort[project_dir] >= int(args.max_slides_per_cohort):
                continue
            if not h5_path.exists():
                task_summary["slides_skipped"] += 1
                summary["skipped_slides"].append(
                    {"task": task, "slide_key": slide_row.get("slide_key", ""), "h5_path": str(h5_path), "reason": "missing_h5"}
                )
                continue

            latent_ids = [int(row["latent_idx"]) for row in relevant_latents]
            latent_positions = {latent: idx for idx, latent in enumerate(latent_ids)}
            try:
                features, coords = read_h5_features_coords(h5_path)
            except Exception as exc:
                task_summary["slides_skipped"] += 1
                summary["skipped_slides"].append(
                    {"task": task, "slide_key": slide_row.get("slide_key", ""), "h5_path": str(h5_path), "reason": str(exc)}
                )
                continue
            if int(features.shape[1]) != int(d_in):
                task_summary["slides_skipped"] += 1
                summary["skipped_slides"].append(
                    {
                        "task": task,
                        "slide_key": slide_row.get("slide_key", ""),
                        "h5_path": str(h5_path),
                        "reason": f"feature_dim_{features.shape[1]}_ne_sae_d_in_{d_in}",
                    }
                )
                continue

            processed_for_task += 1
            processed_by_cohort[project_dir] += 1
            task_summary["slides_processed"] += 1
            task_summary["tiles_processed"] += int(features.shape[0])
            task_summary["cohorts"][project_dir] = int(processed_by_cohort[project_dir])

            for start in range(0, int(features.shape[0]), int(args.batch_size)):
                end = min(int(features.shape[0]), start + int(args.batch_size))
                x = torch.from_numpy(features[start:end]).to(device=device, dtype=torch.float32)
                with torch.inference_mode():
                    z = sae_encode_features(sae_model, x)
                    z_sel = z[:, latent_ids].detach().cpu()
                for latent_idx in latent_ids:
                    pos = latent_positions[int(latent_idx)]
                    scores = z_sel[:, pos]
                    k = min(int(args.top_tiles_per_latent), int(scores.numel()))
                    if k <= 0:
                        continue
                    vals, inds = torch.topk(scores, k=k, largest=True)
                    for val, local_idx_t in zip(vals.tolist(), inds.tolist()):
                        tile_index = int(start) + int(local_idx_t)
                        base_row = {
                            "task": task,
                            "class_label": label,
                            "project_dir": project_dir,
                            "latent_idx": int(latent_idx),
                            "activation": float(val),
                            "case_id": str(slide_row.get("case_id", "")),
                            "slide_key": str(slide_row.get("slide_key", "")),
                            "h5_path": str(h5_path),
                            "tile_index": int(tile_index),
                            "coord_x": int(coords[tile_index, 0]),
                            "coord_y": int(coords[tile_index, 1]),
                            "feature_dim": int(features.shape[1]),
                            "n_tiles_slide": int(features.shape[0]),
                        }
                        task_key = (task, label, int(latent_idx))
                        cohort_key = (task, project_dir, label, int(latent_idx))
                        heap_counter = heap_push_top(
                            task_heaps[task_key],
                            base_row,
                            limit=int(args.top_tiles_per_latent),
                            counter=heap_counter,
                        )
                        heap_counter = heap_push_top(
                            cohort_heaps[cohort_key],
                            base_row,
                            limit=int(args.top_tiles_per_latent),
                            counter=heap_counter,
                        )

        summary["tasks"][task] = task_summary

    selected_latent_fields = [
        "task",
        "class_label",
        "latent_rank_in_class",
        "latent_idx",
        "metric",
        "n_class",
        "n_rest",
        "mean_class",
        "mean_rest",
        "diff_class_minus_rest",
        "abs_diff",
        "cohen_d",
    ]
    write_csv(args.out_dir / "selected_latents.csv", all_selected_latents, selected_latent_fields)

    task_rows: list[dict[str, Any]] = []
    for key in sorted(task_heaps, key=lambda item: (item[0], item[1], item[2])):
        task_rows.extend(heap_to_ranked_rows(task_heaps[key], rank_key="tile_rank_in_task_latent"))
    cohort_rows: list[dict[str, Any]] = []
    for key in sorted(cohort_heaps, key=lambda item: (item[0], item[1], item[2], item[3])):
        cohort_rows.extend(heap_to_ranked_rows(cohort_heaps[key], rank_key="tile_rank_in_cohort_latent"))

    rep_fields = [
        "task",
        "project_dir",
        "class_label",
        "latent_idx",
        "tile_rank_in_task_latent",
        "tile_rank_in_cohort_latent",
        "activation",
        "case_id",
        "slide_key",
        "h5_path",
        "tile_index",
        "coord_x",
        "coord_y",
        "feature_dim",
        "n_tiles_slide",
    ]
    write_csv(args.out_dir / "task_representative_tiles.csv", task_rows, rep_fields)
    write_csv(args.out_dir / "cohort_representative_tiles.csv", cohort_rows, rep_fields)

    if bool(args.write_per_task_files):
        for task in tasks:
            task_dir = args.out_dir / task
            write_csv(
                task_dir / "task_representative_tiles.csv",
                [row for row in task_rows if str(row["task"]) == task],
                rep_fields,
            )
            write_csv(
                task_dir / "cohort_representative_tiles.csv",
                [row for row in cohort_rows if str(row["task"]) == task],
                rep_fields,
            )
            write_csv(
                task_dir / "selected_latents.csv",
                [row for row in all_selected_latents if str(row["task"]) == task],
                selected_latent_fields,
            )

    summary["output_counts"] = {
        "selected_latents": int(len(all_selected_latents)),
        "task_representative_tiles": int(len(task_rows)),
        "cohort_representative_tiles": int(len(cohort_rows)),
        "skipped_slides": int(len(summary["skipped_slides"])),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary["output_counts"], indent=2))


if __name__ == "__main__":
    main()
