#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import h5py


SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(WSI_CF_ROOT) not in sys.path:
    sys.path.insert(0, str(WSI_CF_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build cumulative attention-ranked and true-random progressive edit manifests "
            "for prediction-transition benchmarks."
        )
    )
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--classifier-run-dir", type=Path, default=None)
    parser.add_argument("--classifier-ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--source-label", default="")
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--budgets", default="1,2,4,8,16,32,48,64")
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--max-regions-per-slide", type=int, default=0)
    parser.add_argument("--max-regions", type=int, default=0)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--use-existing-valid-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-full-endpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Always add the valid full-cell budget after clamping requested budgets. "
            "Disable for quick probes where the full endpoint would be too expensive."
        ),
    )
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WSI_CF_ROOT / path


def infer_coord_step(coords: np.ndarray, fallback: int = 256) -> int:
    candidates: list[int] = []
    arr = np.asarray(coords, dtype=np.int64)
    for axis in (0, 1):
        values = np.unique(arr[:, axis])
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    return int(min(candidates)) if candidates else int(fallback)


def aggregate_pt_features_to_supercells(
    features: np.ndarray,
    coords: np.ndarray,
    *,
    tile_size_level0: int,
    block_ratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = int(block_ratio)
    if ratio <= 1:
        return np.asarray(features, dtype=np.float32), np.asarray(coords, dtype=np.int64)
    super_members: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(np.asarray(coords, dtype=np.int64).tolist()):
        raw_gx = int(round(int(x) / float(tile_size_level0)))
        raw_gy = int(round(int(y) / float(tile_size_level0)))
        cell = (raw_gx // ratio, raw_gy // ratio)
        super_members.setdefault(cell, []).append(int(idx))
    ordered_cells = sorted(super_members.keys(), key=lambda cell: (int(cell[1]), int(cell[0])))
    agg_features = np.zeros((len(ordered_cells), int(features.shape[1])), dtype=np.float32)
    agg_coords = np.zeros((len(ordered_cells), 2), dtype=np.int64)
    agg_tile = int(tile_size_level0) * ratio
    for out_idx, cell in enumerate(ordered_cells):
        member_idx = super_members[cell]
        agg_features[out_idx] = np.asarray(features[member_idx], dtype=np.float32).mean(axis=0)
        agg_coords[out_idx] = np.asarray([int(cell[0]) * agg_tile, int(cell[1]) * agg_tile], dtype=np.int64)
    return agg_features, agg_coords


def read_feature_bag_for_region(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    feature_path = resolve_repo_path(row.get("canonical_h5_path") or row.get("feature_path") or "")
    if not str(feature_path):
        raise ValueError(f"{row.get('region_id', '<unknown>')}: missing feature path")
    if feature_path.suffix.lower() in {".pt", ".pth"}:
        import torch

        coords_path = resolve_repo_path(row.get("coords_path", ""))
        if not coords_path.is_file():
            raise FileNotFoundError(f"{row.get('region_id', '<unknown>')}: missing coords_path for PT features: {coords_path}")
        feats = torch.load(feature_path, map_location="cpu")
        if not isinstance(feats, torch.Tensor):
            raise TypeError(f"Unexpected PT payload in {feature_path}: {type(feats)}")
        with h5py.File(coords_path, "r") as handle:
            coords = np.asarray(handle["coords"][:], dtype=np.int64)
        features = feats.detach().cpu().float().numpy().astype(np.float32, copy=False)
        raw_step = int(float(row.get("raw_coord_tile_size_level0") or 0)) or infer_coord_step(coords)
        ratio = int(float(row.get("clam_aggregate_ratio") or 1))
        if ratio > 1:
            features, coords = aggregate_pt_features_to_supercells(
                features,
                coords,
                tile_size_level0=int(raw_step),
                block_ratio=int(ratio),
            )
        if int(features.shape[0]) != int(coords.shape[0]):
            raise ValueError(f"{feature_path}: feature/coord count mismatch {features.shape[0]} != {coords.shape[0]}")
        return features, coords
    with h5py.File(feature_path, "r") as handle:
        features = np.asarray(handle["features"], dtype=np.float32)
        coords = np.asarray(handle["coords"], dtype=np.int64)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    return features, coords


def parse_budgets(value: str) -> list[int]:
    out: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        budget = int(token)
        if budget <= 0:
            raise ValueError(f"Budgets must be positive, got {budget}")
        out.append(budget)
    if not out:
        raise ValueError("--budgets did not contain any values")
    return sorted(set(out))


def effective_budgets(requested: list[int], valid_count: int, *, include_full_endpoint: bool = True) -> list[int]:
    if int(valid_count) <= 0:
        return []
    clamped = {max(1, min(int(budget), int(valid_count))) for budget in requested}
    if bool(include_full_endpoint):
        clamped.add(int(valid_count))
    return sorted(clamped)


def slug_token(value: str) -> str:
    out = []
    for ch in str(value).lower():
        out.append(ch if ch.isalnum() else "_")
    return "_".join("".join(out).split("_"))


def stable_seed(*parts: object, base_seed: int) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**32)


def normalize_label(value: str) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    aliases = {
        "hpv+": "hpv_pos",
        "hpv-positive": "hpv_pos",
        "positive": "hpv_pos",
        "hpv-": "hpv_neg",
        "hpv-negative": "hpv_neg",
        "negative": "hpv_neg",
    }
    return aliases.get(lower, text)


def row_matches_source(row: dict[str, str], source_label: str) -> bool:
    if not source_label:
        return True
    expected = normalize_label(source_label)
    candidates = [
        row.get("source_label", ""),
        row.get("hpv_status", ""),
        row.get("label_name", ""),
        row.get("grade_group", ""),
    ]
    return any(normalize_label(candidate) == expected for candidate in candidates if str(candidate).strip())


def parse_cells(text: str) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for token in str(text or "").split(";"):
        token = token.strip()
        if not token:
            continue
        gx, gy = token.split(",", maxsplit=1)
        cells.append((int(gx), int(gy)))
    return cells


def infer_region_side(row: dict[str, str]) -> int:
    if row.get("region_w") and row.get("grid_step_px"):
        side = int(round(float(row["region_w"]) / float(row["grid_step_px"])))
        if side > 0:
            return side
    cells = parse_cells(row.get("selected_cells_local", ""))
    if cells:
        return max(max(gx for gx, _ in cells), max(gy for _, gy in cells)) + 1
    return 8


def valid_local_cells_from_mask(row: dict[str, str], *, region_side: int) -> list[tuple[int, int]] | None:
    raw_path = str(row.get("valid_mask_path", "")).strip()
    candidates = []
    if raw_path:
        candidates.append(resolve_repo_path(raw_path))
    region_dir = str(row.get("region_dir", "")).strip()
    if region_dir:
        candidates.append(resolve_repo_path(region_dir) / "valid_feature_mask.npy")
    for path in candidates:
        if not path.is_file():
            continue
        mask = np.load(path)
        if mask.shape[0] < region_side or mask.shape[1] < region_side:
            raise ValueError(f"{path}: mask shape {mask.shape} is smaller than region side {region_side}")
        return [
            (int(gx), int(gy))
            for gy in range(region_side)
            for gx in range(region_side)
            if int(mask[gy, gx]) != 0
        ]
    return None


def rank_local_cells_by_attention(
    *,
    row: dict[str, str],
    attention: np.ndarray,
    cell_to_index: dict[tuple[int, int], int],
    region_side: int,
    use_existing_valid_mask: bool,
) -> list[dict[str, Any]]:
    gx0 = int(float(row.get("region_gx0", 0)))
    gy0 = int(float(row.get("region_gy0", 0)))
    valid_cells = valid_local_cells_from_mask(row, region_side=region_side) if use_existing_valid_mask else None
    if valid_cells is None:
        valid_cells = [
            (gx, gy)
            for gy in range(region_side)
            for gx in range(region_side)
            if (gx0 + gx, gy0 + gy) in cell_to_index
        ]
    ranked: list[dict[str, Any]] = []
    for gx, gy in valid_cells:
        global_cell = (gx0 + int(gx), gy0 + int(gy))
        bag_index = cell_to_index.get(global_cell)
        if bag_index is None:
            continue
        ranked.append(
            {
                "gx": int(gx),
                "gy": int(gy),
                "global_gx": int(global_cell[0]),
                "global_gy": int(global_cell[1]),
                "attention": float(attention[int(bag_index)]),
            }
        )
    ranked.sort(key=lambda item: (-float(item["attention"]), int(item["gy"]), int(item["gx"])))
    for rank, item in enumerate(ranked, start=1):
        item["attention_rank"] = int(rank)
    return ranked


def resolve_classifier_ckpt(classifier_run_dir: Path | None, classifier_ckpt: Path | None) -> Path:
    if classifier_ckpt is not None:
        return classifier_ckpt
    if classifier_run_dir is None:
        raise ValueError("Provide --classifier-run-dir or --classifier-ckpt")
    return classifier_run_dir / "best_model.pt"


def load_attention_context(classifier_run_dir: Path | None, classifier_ckpt: Path | None, device_name: str):
    from scripts.find_regions import build_cell_maps, infer_coord_tile_size, read_h5_features_coords, run_generic_mil_attention
    from wsi_cf.common.runtime import resolve_device
    from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint

    device = resolve_device(device_name)
    model = build_mil_from_checkpoint(resolve_classifier_ckpt(classifier_run_dir, classifier_ckpt), device=device)
    return device, model, read_h5_features_coords, infer_coord_tile_size, build_cell_maps, run_generic_mil_attention


def select_region_rows(
    rows: list[dict[str, str]],
    *,
    source_label: str,
    max_slides: int,
    max_regions_per_slide: int,
    max_regions: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    slide_counts: dict[str, int] = defaultdict(int)
    seen_slides: set[str] = set()
    for row in rows:
        if not row_matches_source(row, source_label):
            continue
        slide_key = str(row.get("slide_key") or row.get("case_id") or row.get("region_id", ""))
        if max_slides > 0 and slide_key not in seen_slides and len(seen_slides) >= max_slides:
            continue
        if max_regions_per_slide > 0 and slide_counts[slide_key] >= max_regions_per_slide:
            continue
        selected.append(row)
        slide_counts[slide_key] += 1
        seen_slides.add(slide_key)
        if max_regions > 0 and len(selected) >= max_regions:
            break
    return selected


def cell_payload(cells: list[tuple[int, int]]) -> list[dict[str, int]]:
    return [{"gx": int(gx), "gy": int(gy)} for gx, gy in cells]


def build_region_requests(
    *,
    row: dict[str, str],
    ranked_cells: list[dict[str, Any]],
    requested_budgets: list[int],
    task_name: str,
    direction: str,
    source_label: str,
    target_label: str,
    random_repeats: int,
    seed: int,
    run_prefix: str,
    include_full_endpoint: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_count = len(ranked_cells)
    budgets = effective_budgets(requested_budgets, valid_count, include_full_endpoint=bool(include_full_endpoint))
    region_id = str(row["region_id"])
    run_base = "__".join(token for token in [run_prefix, region_id, direction] if token)
    requests: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ranked_cell_tuples = [(int(item["gx"]), int(item["gy"])) for item in ranked_cells]
    attention_by_cell = {(int(item["gx"]), int(item["gy"])): float(item["attention"]) for item in ranked_cells}
    attention_rank_by_cell = {(int(item["gx"]), int(item["gy"])): int(item["attention_rank"]) for item in ranked_cells}

    def add_request(selector: str, repeat_id: int, budget: int, ordered_cells: list[tuple[int, int]]) -> None:
        chosen = ordered_cells[:budget]
        run_id = f"{run_base}__{slug_token(selector)}"
        if repeat_id >= 0:
            run_id += f"_r{repeat_id:02d}"
        run_id += f"__cells_{int(budget):03d}"
        request = {
            "run_id": run_id,
            "region_id": region_id,
            "task_name": task_name,
            "direction": direction,
            "source_label": source_label or row.get("source_label", row.get("hpv_status", "")),
            "target_label": target_label,
            "selector": selector,
            "repeat_id": int(repeat_id),
            "budget": int(budget),
            "budget_is_full": bool(int(budget) == int(valid_count)),
            "valid_cell_count": int(valid_count),
            "target_cells": cell_payload(chosen),
            "source_region_bank_csv": "",
        }
        requests.append(request)
        summary_rows.append(
            {
                "task_name": task_name,
                "direction": direction,
                "selector": selector,
                "repeat_id": int(repeat_id),
                "budget": int(budget),
                "budget_is_full": int(budget == valid_count),
                "region_id": region_id,
                "slide_key": row.get("slide_key", ""),
                "valid_cell_count": int(valid_count),
                "target_cell_count": int(len(chosen)),
            }
        )
        for selection_rank, cell in enumerate(chosen, start=1):
            selected_rows.append(
                {
                    "task_name": task_name,
                    "direction": direction,
                    "region_id": region_id,
                    "slide_key": row.get("slide_key", ""),
                    "selector": selector,
                    "repeat_id": int(repeat_id),
                    "budget": int(budget),
                    "rank": int(selection_rank),
                    "attention_rank": int(attention_rank_by_cell[cell]),
                    "gx": int(cell[0]),
                    "gy": int(cell[1]),
                    "attention": float(attention_by_cell[cell]),
                    "is_valid": 1,
                    "run_id": run_id,
                }
            )

    for budget in budgets:
        add_request("attention", -1, int(budget), ranked_cell_tuples)

    for repeat_id in range(int(random_repeats)):
        rng = np.random.default_rng(stable_seed(task_name, direction, region_id, repeat_id, base_seed=int(seed)))
        random_cells = list(ranked_cell_tuples)
        rng.shuffle(random_cells)
        for budget in budgets:
            if int(budget) == int(valid_count):
                continue
            add_request("random", repeat_id, int(budget), random_cells)

    return requests, selected_rows, summary_rows


def build_manifests(args: argparse.Namespace) -> dict[str, Any]:
    requested_budgets = parse_budgets(args.budgets)
    rows = select_region_rows(
        read_csv(args.region_bank_csv),
        source_label=str(args.source_label),
        max_slides=int(args.max_slides),
        max_regions_per_slide=int(args.max_regions_per_slide),
        max_regions=int(args.max_regions),
    )
    if not rows:
        raise ValueError(f"No region rows selected from {args.region_bank_csv}")

    (
        device,
        model,
        read_h5_features_coords,
        infer_coord_tile_size,
        build_cell_maps,
        run_generic_mil_attention,
    ) = load_attention_context(args.classifier_run_dir, args.classifier_ckpt, args.device)

    attention_cache: dict[str, tuple[np.ndarray, dict[tuple[int, int], int]]] = {}
    all_requests: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for row in rows:
        feature_path = resolve_repo_path(row.get("canonical_h5_path") or row.get("feature_path") or "")
        coords_path = resolve_repo_path(row.get("coords_path", ""))
        cache_key = f"{feature_path.resolve()}::{coords_path.resolve() if coords_path.exists() else ''}"
        if cache_key not in attention_cache:
            if feature_path.suffix.lower() in {".pt", ".pth"}:
                features, coords = read_feature_bag_for_region(row)
            else:
                features, coords = read_h5_features_coords(str(feature_path))
            tile_size = infer_coord_tile_size(coords)
            cell_to_index, _, _, _ = build_cell_maps(coords, int(tile_size))
            attention, _pred, _prob_pred = run_generic_mil_attention(model, features, device=device)
            attention_cache[cache_key] = (attention, cell_to_index)
        attention, cell_to_index = attention_cache[cache_key]
        ranked = rank_local_cells_by_attention(
            row=row,
            attention=attention,
            cell_to_index=cell_to_index,
            region_side=infer_region_side(row),
            use_existing_valid_mask=bool(args.use_existing_valid_mask),
        )
        if not ranked:
            raise ValueError(f"{row.get('region_id', '<unknown>')}: no valid feature-backed cells")
        requests, selected, summaries = build_region_requests(
            row=row,
            ranked_cells=ranked,
            requested_budgets=requested_budgets,
            task_name=str(args.task_name),
            direction=str(args.direction),
            source_label=str(args.source_label),
            target_label=str(args.target_label),
            random_repeats=int(args.random_repeats),
            seed=int(args.seed),
            run_prefix=str(args.run_prefix),
            include_full_endpoint=bool(args.include_full_endpoint),
        )
        for request in requests:
            request["source_region_bank_csv"] = str(args.region_bank_csv)
        all_requests.extend(requests)
        selection_rows.extend(selected)
        summary_rows.extend(summaries)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    attention_requests = [req for req in all_requests if req["selector"] == "attention"]
    random_requests = [req for req in all_requests if req["selector"] == "random"]
    write_json(args.out_dir / "combined_manifest.json", all_requests)
    write_json(args.out_dir / "attention_manifest.json", attention_requests)
    write_json(args.out_dir / "random_manifest.json", random_requests)
    write_csv(args.out_dir / "selection_cells.csv", selection_rows)
    write_csv(args.out_dir / "manifest_summary.csv", summary_rows)
    payload = {
        "task_name": str(args.task_name),
        "direction": str(args.direction),
        "source_label": str(args.source_label),
        "target_label": str(args.target_label),
        "region_bank_csv": str(args.region_bank_csv),
        "classifier_run_dir": "" if args.classifier_run_dir is None else str(args.classifier_run_dir),
        "classifier_ckpt": "" if args.classifier_ckpt is None else str(args.classifier_ckpt),
        "requested_budgets": requested_budgets,
        "include_full_endpoint": bool(args.include_full_endpoint),
        "random_repeats": int(args.random_repeats),
        "seed": int(args.seed),
        "n_regions": int(len(rows)),
        "n_requests": int(len(all_requests)),
        "combined_manifest": str(args.out_dir / "combined_manifest.json"),
        "selection_cells_csv": str(args.out_dir / "selection_cells.csv"),
    }
    write_json(args.out_dir / "summary.json", payload)
    return payload


def main() -> None:
    summary = build_manifests(build_arg_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
