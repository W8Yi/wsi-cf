#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


WSI_CF_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(WSI_CF_ROOT) not in sys.path:
    sys.path.insert(0, str(WSI_CF_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build normal-to-tumor progressive edit manifests that steer increasing "
            "fractions of each selected region's valid cells, ordered by classifier attention."
        )
    )
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--classifier-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fractions", type=str, default="0.5,0.65,0.8,1.0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run-suffix", type=str, default="attn_frac")
    parser.add_argument(
        "--use-existing-valid-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use valid_feature_mask.npy when present; otherwise derive valid cells from the H5 coordinates.",
    )
    return parser


def read_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_fractions(text: str) -> list[float]:
    values: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"Fractions must be in (0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("--fractions did not contain any values")
    return sorted(set(values))


def fraction_slug(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def parse_cells(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in str(text or "").split(";"):
        token = token.strip()
        if not token:
            continue
        x, y = token.split(",", maxsplit=1)
        out.append((int(x), int(y)))
    return out


def cell_dict(cell: tuple[int, int]) -> dict[str, int]:
    return {"gx": int(cell[0]), "gy": int(cell[1])}


def infer_region_side(row: dict[str, str]) -> int:
    if row.get("region_w") and row.get("grid_step_px"):
        side = int(round(float(row["region_w"]) / float(row["grid_step_px"])))
        if side > 0:
            return side
    cells = parse_cells(str(row.get("selected_cells_local", "")))
    if cells:
        return max(max(x for x, _ in cells), max(y for _, y in cells)) + 1
    return 8


def valid_local_cells_from_mask(row: dict[str, str], *, region_side: int) -> list[tuple[int, int]] | None:
    raw_mask_path = str(row.get("valid_mask_path", "")).strip()
    mask_path = Path(raw_mask_path) if raw_mask_path else Path()
    if not raw_mask_path or not mask_path.is_file():
        region_dir = Path(str(row.get("region_dir", "")))
        candidate = region_dir / "valid_feature_mask.npy"
        mask_path = candidate if candidate.is_file() else mask_path
    if not mask_path.is_file():
        return None
    mask = np.load(mask_path)
    if mask.shape[0] < region_side or mask.shape[1] < region_side:
        raise ValueError(f"{mask_path}: mask shape {mask.shape} is smaller than region side {region_side}")
    return [
        (int(lx), int(ly))
        for ly in range(region_side)
        for lx in range(region_side)
        if int(mask[ly, lx]) != 0
    ]


def rank_local_cells_by_attention(
    *,
    row: dict[str, str],
    attention: np.ndarray,
    cell_to_index: dict[tuple[int, int], int],
    region_side: int,
    use_existing_valid_mask: bool,
) -> list[dict[str, Any]]:
    gx0 = int(float(row["region_gx0"]))
    gy0 = int(float(row["region_gy0"]))
    valid_local = None
    if use_existing_valid_mask:
        valid_local = valid_local_cells_from_mask(row, region_side=region_side)
    if valid_local is None:
        valid_local = []
        for ly in range(region_side):
            for lx in range(region_side):
                if (gx0 + lx, gy0 + ly) in cell_to_index:
                    valid_local.append((int(lx), int(ly)))

    ranked: list[dict[str, Any]] = []
    for lx, ly in valid_local:
        global_cell = (gx0 + int(lx), gy0 + int(ly))
        idx = cell_to_index.get(global_cell)
        if idx is None:
            continue
        ranked.append(
            {
                "gx": int(lx),
                "gy": int(ly),
                "global_gx": int(global_cell[0]),
                "global_gy": int(global_cell[1]),
                "attention": float(attention[int(idx)]),
            }
        )
    ranked.sort(key=lambda item: (-float(item["attention"]), int(item["gy"]), int(item["gx"])))
    for rank, item in enumerate(ranked, start=1):
        item["attention_rank"] = int(rank)
    return ranked


def choose_fraction_cells(ranked_cells: list[dict[str, Any]], fraction: float) -> list[tuple[int, int]]:
    if not ranked_cells:
        return []
    n_cells = len(ranked_cells) if float(fraction) >= 1.0 else int(math.ceil(float(fraction) * len(ranked_cells)))
    n_cells = max(1, min(int(n_cells), len(ranked_cells)))
    selected = ranked_cells[:n_cells]
    return [(int(row["gx"]), int(row["gy"])) for row in selected]


def load_attention_context(classifier_run_dir: Path, device_name: str):
    import torch

    from scripts.find_regions import build_cell_maps, infer_coord_tile_size, read_h5_features_coords, run_generic_mil_attention
    from wsi_cf.common.runtime import resolve_device
    from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint

    device = resolve_device(device_name)
    model = build_mil_from_checkpoint(classifier_run_dir / "best_model.pt", device=device)
    return torch, device, model, read_h5_features_coords, infer_coord_tile_size, build_cell_maps, run_generic_mil_attention


def build_manifests(args: argparse.Namespace) -> None:
    fractions = parse_fractions(args.fractions)
    rows = read_csv(args.region_bank_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.region_bank_csv}")

    (
        _torch,
        device,
        model,
        read_h5_features_coords,
        infer_coord_tile_size,
        build_cell_maps,
        run_generic_mil_attention,
    ) = load_attention_context(args.classifier_run_dir, args.device)

    attention_cache: dict[str, tuple[np.ndarray, dict[tuple[int, int], int], int]] = {}
    requests_by_fraction: dict[float, list[dict[str, Any]]] = {fraction: [] for fraction in fractions}
    ranked_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for row in rows:
        h5_path = Path(str(row["canonical_h5_path"]))
        cache_key = str(h5_path.resolve())
        if cache_key not in attention_cache:
            features, coords = read_h5_features_coords(h5_path)
            tile_size = infer_coord_tile_size(coords)
            cell_to_index, _, _, _ = build_cell_maps(coords, int(tile_size))
            attention, pred, prob_pred = run_generic_mil_attention(model, features, device=device)
            attention_cache[cache_key] = (attention, cell_to_index, int(tile_size))
        attention, cell_to_index, tile_size = attention_cache[cache_key]
        region_side = infer_region_side(row)
        ranked = rank_local_cells_by_attention(
            row=row,
            attention=attention,
            cell_to_index=cell_to_index,
            region_side=int(region_side),
            use_existing_valid_mask=bool(args.use_existing_valid_mask),
        )
        if not ranked:
            raise ValueError(f"{row.get('region_id', '<unknown>')}: no valid cells available for sweep")

        region_id = str(row["region_id"])
        base_run_id = f"{region_id}__to_{str(row.get('target_label', 'tumor')).lower()}_concepts"
        for ranked_cell in ranked:
            ranked_rows.append(
                {
                    "region_id": region_id,
                    "slide_key": row.get("slide_key", ""),
                    "attention_rank": int(ranked_cell["attention_rank"]),
                    "gx": int(ranked_cell["gx"]),
                    "gy": int(ranked_cell["gy"]),
                    "global_gx": int(ranked_cell["global_gx"]),
                    "global_gy": int(ranked_cell["global_gy"]),
                    "attention": float(ranked_cell["attention"]),
                }
            )

        for fraction in fractions:
            cells = choose_fraction_cells(ranked, fraction)
            slug = fraction_slug(fraction)
            request = {
                "run_id": f"{base_run_id}__{args.run_suffix}_{slug}",
                "region_id": region_id,
                "source_label": str(row.get("source_label", "normal")),
                "target_label": str(row.get("target_label", "tumor")),
                "selection_mode": "classifier_attention_cell_fraction_sweep",
                "target_fraction": float(fraction),
                "valid_cell_count": int(len(ranked)),
                "target_cell_count": int(len(cells)),
                "source_region_bank_csv": str(args.region_bank_csv),
                "target_cells": [cell_dict(cell) for cell in cells],
                "selector": "classifier_attention_valid_cell_fraction",
            }
            requests_by_fraction[fraction].append(request)
            selected_ranks = [int(ranked[i]["attention_rank"]) for i in range(len(cells))]
            summary_rows.append(
                {
                    "fraction": float(fraction),
                    "fraction_slug": slug,
                    "region_id": region_id,
                    "slide_key": row.get("slide_key", ""),
                    "valid_cell_count": int(len(ranked)),
                    "target_cell_count": int(len(cells)),
                    "attention_rank_min": int(min(selected_ranks)),
                    "attention_rank_max": int(max(selected_ranks)),
                    "tile_size_level0": int(tile_size),
                }
            )

    all_requests: list[dict[str, Any]] = []
    for fraction in fractions:
        slug = fraction_slug(fraction)
        manifest_path = args.out_dir / f"fraction_{slug}.json"
        requests = requests_by_fraction[fraction]
        write_json(manifest_path, requests)
        all_requests.extend(requests)
    write_json(args.out_dir / "all_fractions_manifest.json", all_requests)
    write_csv(args.out_dir / "fraction_summary.csv", summary_rows)
    write_csv(args.out_dir / "attention_ranked_cells.csv", ranked_rows)
    write_json(
        args.out_dir / "summary.json",
        {
            "region_bank_csv": str(args.region_bank_csv),
            "classifier_run_dir": str(args.classifier_run_dir),
            "fractions": fractions,
            "region_count": int(len(rows)),
            "combined_manifest": str(args.out_dir / "all_fractions_manifest.json"),
            "fraction_summary_csv": str(args.out_dir / "fraction_summary.csv"),
            "attention_ranked_cells_csv": str(args.out_dir / "attention_ranked_cells.csv"),
        },
    )
    print(f"[ok] wrote {args.out_dir / 'all_fractions_manifest.json'}")
    for fraction in fractions:
        slug = fraction_slug(fraction)
        print(f"[fraction] {slug}: {args.out_dir / f'fraction_{slug}.json'}")


def main() -> None:
    build_manifests(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
