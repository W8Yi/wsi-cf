#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a progressive-edit manifest by selecting local region cells from attention scores only."
    )
    parser.add_argument("--attention-map", type=Path, required=True, help="2D .npy attention map in local region grid order, shape H x W.")
    parser.add_argument("--region-bank-csv", type=Path, required=True, help="Region bank containing the source region row.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--region-id", type=str, default="", help="Region id to select from the region bank. Defaults to the first row.")
    parser.add_argument("--selection", type=str, default="top_n", choices=["top_n", "percentile"])
    parser.add_argument("--top-n", type=int, default=28)
    parser.add_argument("--attention-percentile", type=float, default=56.25)
    parser.add_argument("--run-suffix", type=str, default="attention_only_top28")
    parser.add_argument("--compare-manifest", type=Path, default=None, help="Optional existing manifest to compare overlap against.")
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument(
        "--smooth-fill-holes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After the initial attention selection, add unselected cells with enough selected neighbors.",
    )
    parser.add_argument("--smooth-neighborhood", type=int, default=4, choices=[4, 8])
    parser.add_argument("--smooth-min-neighbors", type=int, default=3)
    parser.add_argument(
        "--smooth-border-relax",
        type=int,
        default=1,
        help="Reduce the neighbor requirement by this amount for candidate cells on the grid border.",
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=1,
        help="Number of smoothing passes. 1 avoids cascading fills; higher values can grow larger masks.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=0,
        help="Optional final target count. If smoothing selects too many cells, prune back deterministically.",
    )
    parser.add_argument(
        "--prune-strategy",
        type=str,
        default="low_support",
        choices=["low_support", "low_attention"],
        help="How to prune when --target-count is set below the selected cell count.",
    )
    return parser


def read_region_bank(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fields.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def draw_overlay(image_path: Path, cells: list[tuple[int, int]], *, grid_step_px: int, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    for rank, (gx, gy) in enumerate(cells, start=1):
        x0 = int(gx) * int(grid_step_px)
        y0 = int(gy) * int(grid_step_px)
        draw.rectangle(
            [x0, y0, x0 + int(grid_step_px) - 1, y0 + int(grid_step_px) - 1],
            fill=(255, 165, 0, 55),
            outline=(255, 210, 0, 245),
            width=5,
        )
        draw.text((x0 + 8, y0 + 8), str(rank), fill=(180, 0, 0, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def manifest_cells(path: Path) -> set[tuple[int, int]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        return set()
    return {(int(cell["gx"]), int(cell["gy"])) for cell in data[0].get("target_cells", [])}


def select_cells(attention: np.ndarray, *, selection: str, top_n: int, percentile: float) -> tuple[list[tuple[int, int]], float | None]:
    if attention.ndim != 2:
        raise ValueError(f"attention map must be 2D, got shape {attention.shape}")
    ranked = [
        ((int(gx), int(gy)), float(attention[int(gy), int(gx)]))
        for gy in range(int(attention.shape[0]))
        for gx in range(int(attention.shape[1]))
    ]
    ranked.sort(key=lambda item: (-float(item[1]), int(item[0][1]), int(item[0][0])))
    if selection == "top_n":
        if int(top_n) <= 0 or int(top_n) > len(ranked):
            raise ValueError(f"--top-n must be in [1, {len(ranked)}]")
        return [cell for cell, _ in ranked[: int(top_n)]], None
    threshold = float(np.percentile(attention.reshape(-1), float(percentile)))
    cells = [cell for cell, score in ranked if float(score) >= threshold]
    return cells, threshold


def neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if int(connectivity) == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if int(connectivity) == 8:
        return ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
    raise ValueError(f"Unsupported smoothing connectivity: {connectivity}")


def smooth_fill_holes(
    cells: list[tuple[int, int]],
    *,
    grid_w: int,
    grid_h: int,
    connectivity: int,
    min_neighbors: int,
    border_relax: int,
    iterations: int,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    selected = {(int(gx), int(gy)) for gx, gy in cells}
    offsets = neighbor_offsets(int(connectivity))
    added_rows: list[dict[str, Any]] = []
    for iteration in range(1, max(1, int(iterations)) + 1):
        to_add: list[tuple[int, int, int, int, bool]] = []
        for gy in range(int(grid_h)):
            for gx in range(int(grid_w)):
                cell = (int(gx), int(gy))
                if cell in selected:
                    continue
                neighbor_count = 0
                for dx, dy in offsets:
                    nx = int(gx) + int(dx)
                    ny = int(gy) + int(dy)
                    if (nx, ny) in selected:
                        neighbor_count += 1
                touches_border = int(gx) in {0, int(grid_w) - 1} or int(gy) in {0, int(grid_h) - 1}
                required = int(min_neighbors) - (int(border_relax) if touches_border else 0)
                required = max(1, int(required))
                if int(neighbor_count) >= int(required):
                    to_add.append((int(gx), int(gy), int(neighbor_count), int(required), bool(touches_border)))
        if not to_add:
            break
        for gx, gy, neighbor_count, required, touches_border in to_add:
            selected.add((int(gx), int(gy)))
            added_rows.append(
                {
                    "gx": int(gx),
                    "gy": int(gy),
                    "iteration": int(iteration),
                    "selected_neighbor_count": int(neighbor_count),
                    "required_neighbor_count": int(required),
                    "touches_grid_border": bool(touches_border),
                }
            )
    return sorted(selected, key=lambda cell: (int(cell[1]), int(cell[0]))), added_rows


def local_support_count(cells: set[tuple[int, int]], cell: tuple[int, int], *, connectivity: int) -> int:
    gx, gy = int(cell[0]), int(cell[1])
    count = 0
    for dx, dy in neighbor_offsets(int(connectivity)):
        if (gx + int(dx), gy + int(dy)) in cells:
            count += 1
    return int(count)


def prune_to_target_count(
    cells: list[tuple[int, int]],
    *,
    attention: np.ndarray,
    target_count: int,
    connectivity: int,
    strategy: str,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    selected = {(int(gx), int(gy)) for gx, gy in cells}
    pruned_rows: list[dict[str, Any]] = []
    if int(target_count) <= 0 or len(selected) <= int(target_count):
        return sorted(selected, key=lambda cell: (int(cell[1]), int(cell[0]))), pruned_rows
    while len(selected) > int(target_count):
        candidates = sorted(selected, key=lambda cell: (int(cell[1]), int(cell[0])))
        if str(strategy) == "low_attention":
            drop = min(
                candidates,
                key=lambda cell: (
                    float(attention[int(cell[1]), int(cell[0])]),
                    local_support_count(selected, cell, connectivity=int(connectivity)),
                    int(cell[1]),
                    int(cell[0]),
                ),
            )
        else:
            drop = min(
                candidates,
                key=lambda cell: (
                    local_support_count(selected, cell, connectivity=int(connectivity)),
                    float(attention[int(cell[1]), int(cell[0])]),
                    int(cell[1]),
                    int(cell[0]),
                ),
            )
        selected.remove(drop)
        pruned_rows.append(
            {
                "gx": int(drop[0]),
                "gy": int(drop[1]),
                "attention": float(attention[int(drop[1]), int(drop[0])]),
                "support_count_before_prune": int(local_support_count(selected | {drop}, drop, connectivity=int(connectivity))),
                "prune_step": int(len(pruned_rows) + 1),
            }
        )
    return sorted(selected, key=lambda cell: (int(cell[1]), int(cell[0]))), pruned_rows


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = read_region_bank(args.region_bank_csv)
    if not rows:
        raise ValueError(f"Empty region bank: {args.region_bank_csv}")
    if args.region_id:
        matches = [row for row in rows if str(row.get("region_id", "")) == str(args.region_id)]
        if not matches:
            raise ValueError(f"Region id {args.region_id!r} not found in {args.region_bank_csv}")
        region_row = matches[0]
    else:
        region_row = rows[0]
    region_id = str(region_row["region_id"])

    attention = np.asarray(np.load(args.attention_map), dtype=np.float32)
    initial_selected, threshold = select_cells(
        attention,
        selection=str(args.selection),
        top_n=int(args.top_n),
        percentile=float(args.attention_percentile),
    )
    selected = list(initial_selected)
    smooth_added_rows: list[dict[str, Any]] = []
    pruned_rows: list[dict[str, Any]] = []
    if bool(args.smooth_fill_holes):
        selected, smooth_added_rows = smooth_fill_holes(
            selected,
            grid_w=int(attention.shape[1]),
            grid_h=int(attention.shape[0]),
            connectivity=int(args.smooth_neighborhood),
            min_neighbors=int(args.smooth_min_neighbors),
            border_relax=int(args.smooth_border_relax),
            iterations=int(args.smooth_iterations),
        )
    selected, pruned_rows = prune_to_target_count(
        selected,
        attention=attention,
        target_count=int(args.target_count),
        connectivity=int(args.smooth_neighborhood),
        strategy=str(args.prune_strategy),
    )
    selected_row_major = sorted(selected, key=lambda cell: (int(cell[1]), int(cell[0])))
    selected_set = set(selected_row_major)
    initial_selected_set = {(int(gx), int(gy)) for gx, gy in initial_selected}
    smooth_added_set = {(int(row["gx"]), int(row["gy"])) for row in smooth_added_rows}

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.region_bank_csv, out_dir / "region_bank.csv")

    run_id = f"{region_id}__{args.run_suffix}"
    manifest = [
        {
            "run_id": run_id,
            "region_id": region_id,
            "selection_mode": f"attention_score_only_{args.selection}",
            "target_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in selected_row_major],
            "seed_target_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in sorted(initial_selected_set, key=lambda cell: (cell[1], cell[0]))],
            "selector": "attention_score_only",
            "attention_map": str(args.attention_map),
            "top_n": int(args.top_n) if str(args.selection) == "top_n" else None,
            "attention_percentile": float(args.attention_percentile) if str(args.selection) == "percentile" else None,
            "attention_threshold": threshold,
            "smoothing": {
                "enabled": bool(args.smooth_fill_holes),
                "neighborhood": int(args.smooth_neighborhood),
                "min_neighbors": int(args.smooth_min_neighbors),
                "border_relax": int(args.smooth_border_relax),
                "iterations": int(args.smooth_iterations),
                "added_cells": [{"gx": int(row["gx"]), "gy": int(row["gy"])} for row in smooth_added_rows],
            },
        }
    ]
    write_json(out_dir / "progressive_edit_manifest.json", manifest)
    if smooth_added_rows:
        write_csv(out_dir / "smoothing_added_cells.csv", smooth_added_rows)
    if pruned_rows:
        write_csv(out_dir / "pruned_cells.csv", pruned_rows)

    ranked_rows = []
    ranked = [
        ((int(gx), int(gy)), float(attention[int(gy), int(gx)]))
        for gy in range(int(attention.shape[0]))
        for gx in range(int(attention.shape[1]))
    ]
    ranked.sort(key=lambda item: (-float(item[1]), int(item[0][1]), int(item[0][0])))
    for rank, (cell, score) in enumerate(ranked, start=1):
        ranked_rows.append(
            {
                "attention_rank": int(rank),
                "gx": int(cell[0]),
                "gy": int(cell[1]),
                "attention": float(score),
                "selected_initial": bool(cell in initial_selected_set),
                "selected_by_smoothing": bool(cell in smooth_added_set),
                "selected": bool(cell in selected_set),
            }
        )
    write_csv(out_dir / "attention_cells.csv", ranked_rows)

    overlay_path = None
    image_path = Path(str(region_row.get("image_path", "")))
    if image_path.exists():
        overlay_path = out_dir / "attention_only_selected_overlay.png"
        draw_overlay(image_path, selected_row_major, grid_step_px=int(args.grid_step_px), out_path=overlay_path)

    summary: dict[str, Any] = {
        "region_id": region_id,
        "run_id": run_id,
        "selection": str(args.selection),
        "initial_selected_count": int(len(initial_selected_set)),
        "selected_count": int(len(selected_row_major)),
        "smooth_added_count": int(len(smooth_added_rows)),
        "target_count": int(args.target_count) if int(args.target_count) > 0 else None,
        "prune_strategy": str(args.prune_strategy) if int(args.target_count) > 0 else None,
        "pruned_count": int(len(pruned_rows)),
        "pruned_cells": [{"gx": int(row["gx"]), "gy": int(row["gy"])} for row in pruned_rows],
        "smoothing": {
            "enabled": bool(args.smooth_fill_holes),
            "neighborhood": int(args.smooth_neighborhood),
            "min_neighbors": int(args.smooth_min_neighbors),
            "border_relax": int(args.smooth_border_relax),
            "iterations": int(args.smooth_iterations),
            "added_cells": [{"gx": int(row["gx"]), "gy": int(row["gy"])} for row in smooth_added_rows],
        },
        "attention_map_shape": [int(attention.shape[0]), int(attention.shape[1])],
        "top_n": int(args.top_n) if str(args.selection) == "top_n" else None,
        "attention_percentile": float(args.attention_percentile) if str(args.selection) == "percentile" else None,
        "attention_threshold": threshold,
        "region_bank_csv": str(out_dir / "region_bank.csv"),
        "progressive_edit_manifest": str(out_dir / "progressive_edit_manifest.json"),
        "attention_cells_csv": str(out_dir / "attention_cells.csv"),
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
    }
    if args.compare_manifest is not None:
        reference = manifest_cells(args.compare_manifest)
        overlap = selected_set & reference
        union = selected_set | reference
        summary.update(
            {
                "compare_manifest": str(args.compare_manifest),
                "reference_count": int(len(reference)),
                "overlap_count": int(len(overlap)),
                "jaccard": float(len(overlap) / len(union)) if union else 0.0,
                "missing_reference_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in sorted(reference - selected_set, key=lambda c: (c[1], c[0]))],
                "extra_selected_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in sorted(selected_set - reference, key=lambda c: (c[1], c[0]))],
            }
        )
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
