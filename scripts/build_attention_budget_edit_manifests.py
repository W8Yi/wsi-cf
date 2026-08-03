#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


WSI_CF_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MANIFEST = (
    WSI_CF_ROOT
    / "artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/progressive_edit_manifest.json"
)
DEFAULT_ATTENTION_CSV = (
    WSI_CF_ROOT
    / "artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/attention_cells.csv"
)
DEFAULT_OUT_DIR = WSI_CF_ROOT / "artifacts/hnscc_hpv_showcase_attention_budget_sweep/manifests"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build cumulative progressive-edit manifests that start from the reviewed showcase "
            "target cells and add more cells by descending attention rank."
        )
    )
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--attention-cells-csv", type=Path, default=DEFAULT_ATTENTION_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--budgets",
        type=str,
        default="28,32,40,48,56,64",
        help="Comma-separated requested target cell counts. Default starts at the current reviewed 28-cell request.",
    )
    parser.add_argument(
        "--start-mode",
        type=str,
        choices=["base_then_attention", "attention_only"],
        default="base_then_attention",
        help="base_then_attention preserves current target cells first, then fills remaining cells by attention rank.",
    )
    parser.add_argument("--run-suffix", type=str, default="attn_budget")
    return parser


def read_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def parse_budgets(text: str) -> list[int]:
    budgets: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Budgets must be positive, got {value}")
        budgets.append(value)
    if not budgets:
        raise ValueError("--budgets did not contain any values")
    return sorted(set(budgets))


def normalize_cell(cell: Any) -> tuple[int, int]:
    if isinstance(cell, dict):
        return int(cell["gx"]), int(cell["gy"])
    if isinstance(cell, (list, tuple)) and len(cell) == 2:
        return int(cell[0]), int(cell[1])
    raise ValueError(f"Unsupported cell value: {cell!r}")


def cell_dict(cell: tuple[int, int]) -> dict[str, int]:
    return {"gx": int(cell[0]), "gy": int(cell[1])}


def read_attention_cells(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "attention_rank": int(row["attention_rank"]),
                "gx": int(row["gx"]),
                "gy": int(row["gy"]),
                "attention": float(row["attention"]),
                "selected_current": str(row.get("selected", "")).lower() == "true",
                "selected_initial": str(row.get("selected_initial", "")).lower() == "true",
                "selected_by_smoothing": str(row.get("selected_by_smoothing", "")).lower() == "true",
            }
        )
    out.sort(key=lambda row: (int(row["attention_rank"]), int(row["gy"]), int(row["gx"])))
    seen = {(int(row["gx"]), int(row["gy"])) for row in out}
    if len(seen) != len(out):
        raise ValueError(f"{path}: duplicate cells in attention CSV")
    return out


def ordered_budget_cells(
    *,
    base_cells: list[tuple[int, int]],
    attention_rows: list[dict[str, Any]],
    budget: int,
    start_mode: str,
) -> list[tuple[int, int]]:
    attention_cells = [(int(row["gx"]), int(row["gy"])) for row in attention_rows]
    if int(budget) > len(attention_cells):
        raise ValueError(f"Budget {budget} exceeds attention cell count {len(attention_cells)}")

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    if str(start_mode) == "base_then_attention":
        for cell in base_cells:
            if cell in seen:
                continue
            ordered.append(cell)
            seen.add(cell)
    for cell in attention_cells:
        if cell in seen:
            continue
        ordered.append(cell)
        seen.add(cell)
    return ordered[: int(budget)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()
    base_payload = read_json(args.base_manifest)
    if not isinstance(base_payload, list) or not base_payload:
        raise ValueError(f"{args.base_manifest}: expected non-empty list manifest")
    if len(base_payload) != 1:
        raise ValueError(f"{args.base_manifest}: expected exactly one showcase request")
    base = dict(base_payload[0])
    region_id = str(base["region_id"])
    base_run_id = str(base.get("run_id") or region_id)
    base_cells = [normalize_cell(cell) for cell in base["target_cells"]]
    attention_rows = read_attention_cells(args.attention_cells_csv)
    attention_by_cell = {(int(row["gx"]), int(row["gy"])): row for row in attention_rows}
    missing_base = [cell for cell in base_cells if cell not in attention_by_cell]
    if missing_base:
        raise ValueError(f"Base target cells missing from attention CSV: {missing_base}")

    budgets = parse_budgets(args.budgets)
    all_requests: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for budget in budgets:
        cells = ordered_budget_cells(
            base_cells=base_cells,
            attention_rows=attention_rows,
            budget=int(budget),
            start_mode=str(args.start_mode),
        )
        attention_ranks = [int(attention_by_cell[cell]["attention_rank"]) for cell in cells]
        run_id = f"{base_run_id}__{args.run_suffix}_{int(budget):03d}"
        request = {
            "run_id": run_id,
            "region_id": region_id,
            "selection_mode": "attention_budget_sweep",
            "target_budget": int(budget),
            "start_mode": str(args.start_mode),
            "source_base_manifest": str(args.base_manifest),
            "source_attention_cells_csv": str(args.attention_cells_csv),
            "target_cells": [cell_dict(cell) for cell in cells],
            "attention_rank_min": int(min(attention_ranks)),
            "attention_rank_max": int(max(attention_ranks)),
            "attention_rank_mean": float(sum(attention_ranks) / max(len(attention_ranks), 1)),
            "notes": (
                "Cumulative target set for pathologist review: current showcase cells are preserved first, "
                "then remaining cells are added by descending attention score."
            ),
        }
        all_requests.append(request)
        manifest_path = args.out_dir / f"cells_{int(budget):03d}.json"
        write_json(manifest_path, [request])
        summary_rows.append(
            {
                "budget": int(budget),
                "run_id": run_id,
                "manifest_path": str(manifest_path),
                "num_cells": int(len(cells)),
                "num_base_cells_retained": int(sum(1 for cell in base_cells if cell in set(cells))),
                "attention_rank_min": int(min(attention_ranks)),
                "attention_rank_max": int(max(attention_ranks)),
                "attention_rank_mean": float(sum(attention_ranks) / max(len(attention_ranks), 1)),
            }
        )

    write_json(args.out_dir / "all_budgets_manifest.json", all_requests)
    write_csv(args.out_dir / "budget_summary.csv", summary_rows)
    write_csv(args.out_dir / "attention_ranked_cells.csv", attention_rows)
    write_json(
        args.out_dir / "summary.json",
        {
            "base_manifest": str(args.base_manifest),
            "attention_cells_csv": str(args.attention_cells_csv),
            "region_id": region_id,
            "base_run_id": base_run_id,
            "base_target_cell_count": int(len(base_cells)),
            "attention_cell_count": int(len(attention_rows)),
            "budgets": budgets,
            "start_mode": str(args.start_mode),
            "combined_manifest": str(args.out_dir / "all_budgets_manifest.json"),
            "budget_summary_csv": str(args.out_dir / "budget_summary.csv"),
        },
    )
    print(f"[ok] wrote {args.out_dir / 'all_budgets_manifest.json'}")
    for row in summary_rows:
        print(
            f"[budget] cells={row['budget']} base_retained={row['num_base_cells_retained']} "
            f"rank_max={row['attention_rank_max']} manifest={row['manifest_path']}"
        )


if __name__ == "__main__":
    main()
