#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-tumor_purity_low_high}"

N_PAIRS="${N_PAIRS:-10}"
SEED="${SEED:-7}"
STEPS="${STEPS:-30}"
OUTPUT_MODE="${OUTPUT_MODE:-minimal}"
TARGET_MAGNIFICATION="${TARGET_MAGNIFICATION:-20}"
REGION_SIZE="${REGION_SIZE:-2048}"
GRID_STEP_PX="${GRID_STEP_PX:-256}"

FEATURES_ROOT="${FEATURES_ROOT:-/research/projects/mllab/WSI/TCGA_features}"
SLIDES_ROOT="${SLIDES_ROOT:-/research/projects/mllab/WSI/TCGA/store,/research/projects/mllab/WSI/.tmp/ready_buffer/slides}"
DOWNLOAD_SLIDES="${DOWNLOAD_SLIDES:-1}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/research/projects/mllab/WSI/.tmp/ready_buffer/slides}"
MAX_DOWNLOADS="${MAX_DOWNLOADS:-40}"
MANIFEST_CSV="${MANIFEST_CSV:-artifacts/label_sources/tumor_purity_low_high_manifest.csv}"

CONCEPT_ROOT="${CONCEPT_ROOT:-artifacts/label_only_concepts_top10_top50/${TASK}}"
ASSOC_ROOT="${ASSOC_ROOT:-artifacts/concept_label_associations_labels_only}"
OUT_ROOT="${OUT_ROOT:-artifacts/tumor_purity_low_high_10_pairs_labels_only}"

CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
CONCEPT_TARGET_STAT="${CONCEPT_TARGET_STAT:-median}"
CONCEPT_STEERING_MODE="${CONCEPT_STEERING_MODE:-prototype_vector}"
MAX_CONCEPTS="${MAX_CONCEPTS:-0}"

MIN_TISSUE="${MIN_TISSUE:-0.45}"
MIN_DARK_FRACTION="${MIN_DARK_FRACTION:-0.04}"
MIN_SATURATION_FRACTION="${MIN_SATURATION_FRACTION:-0.04}"
MAX_REGION_TRIES_PER_SLIDE="${MAX_REGION_TRIES_PER_SLIDE:-128}"
MAX_SLIDE_SCAN="${MAX_SLIDE_SCAN:-500}"

ENSURE_CONCEPTS="${ENSURE_CONCEPTS:-1}"
SKIP_REGION_PREP="${SKIP_REGION_PREP:-0}"
SKIP_EDITS="${SKIP_EDITS:-0}"

if [[ "${ENSURE_CONCEPTS}" == "1" ]]; then
  if [[ ! -f "${CONCEPT_ROOT}/low/selected_concepts.json" || ! -f "${CONCEPT_ROOT}/high/selected_concepts.json" ]]; then
    echo "[concepts] missing labels-only tumor purity concepts; building them first" >&2
    ASSOC_ROOT="${ASSOC_ROOT}" \
    CONCEPT_OUT="$(dirname "${CONCEPT_ROOT}")" \
    MAX_SLIDES="${MAX_SLIDE_SCAN}" \
    DEVICE="${DEVICE}" \
    bash examples/tumor_purity/00_find_low_high_purity_concepts_labels_only.sh
  fi
fi

LOW_TO_HIGH_DIR="${OUT_ROOT}/low_to_high_regions"
HIGH_TO_LOW_DIR="${OUT_ROOT}/high_to_low_regions"
LOW_TO_HIGH_OUT="${OUT_ROOT}/low_to_high_edits"
HIGH_TO_LOW_OUT="${OUT_ROOT}/high_to_low_edits"

if [[ "${SKIP_REGION_PREP}" != "1" ]]; then
  echo "[regions] sampling ${N_PAIRS} low sources and ${N_PAIRS} high sources" >&2
  "$PY" - \
    --manifest-csv "${MANIFEST_CSV}" \
    --slides-root "${SLIDES_ROOT}" \
    --download-slides "${DOWNLOAD_SLIDES}" \
    --download-root "${DOWNLOAD_ROOT}" \
    --max-downloads "${MAX_DOWNLOADS}" \
    --out-root "${OUT_ROOT}" \
    --n-pairs "${N_PAIRS}" \
    --seed "${SEED}" \
    --region-size "${REGION_SIZE}" \
    --grid-step-px "${GRID_STEP_PX}" \
    --target-magnification "${TARGET_MAGNIFICATION}" \
    --min-tissue "${MIN_TISSUE}" \
    --min-dark-fraction "${MIN_DARK_FRACTION}" \
    --min-saturation-fraction "${MIN_SATURATION_FRACTION}" \
    --max-region-tries-per-slide "${MAX_REGION_TRIES_PER_SLIDE}" \
    --max-slide-scan "${MAX_SLIDE_SCAN}" <<'PY'
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import requests

sys.path.insert(0, str(Path.cwd() / "src"))

from wsi_cf.common.io import save_png, write_json
from wsi_cf.data.region_bank import make_region_cells_preview, write_region_bank_csv
from wsi_cf.data.slides import (
    find_slide_path,
    infer_objective_power,
    open_slide,
    quick_region_quality_metrics,
    read_region_rgb_at_magnification,
)
from wsi_cf.steering.progressive import draw_cells_overlay


def read_rows(path):
    with Path(path).open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_h5_features_coords(path):
    with h5py.File(path, "r") as handle:
        feats = np.asarray(handle["features"][:], dtype=np.float32)
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if feats.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"unsupported H5 shapes: features={feats.shape}, coords={coords.shape}")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(f"feature/coord count mismatch: {feats.shape[0]} vs {coords.shape[0]}")
    return feats, coords


def infer_coord_tile_size(coords, fallback=512):
    arr = np.asarray(coords, dtype=np.int64)
    candidates = []
    for axis in (0, 1):
        vals = np.unique(arr[:, axis])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    return int(min(candidates)) if candidates else int(fallback)


def build_cell_maps(coords, tile_size):
    cell_to_index = {}
    for idx, (x, y) in enumerate(np.asarray(coords, dtype=np.int64)):
        cell_to_index[(int(round(int(x) / int(tile_size))), int(round(int(y) / int(tile_size))))] = int(idx)
    return cell_to_index


def valid_starts(cell_to_index, side):
    cells = set(cell_to_index)
    starts = []
    for gx, gy in cells:
        ok = True
        for dy in range(side):
            for dx in range(side):
                if (gx + dx, gy + dy) not in cells:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            starts.append((int(gx), int(gy)))
    return starts


def effective_grid_step_at_target_mag(tile_size_level0, objective_power, target_magnification):
    return float(tile_size_level0) * float(target_magnification) / max(float(objective_power), 1e-8)


DOWNLOAD_COUNT = 0


def query_gdc_slide(case_id, slide_key):
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": [case_id]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size",
        "format": "JSON",
        "size": "100",
    }
    resp = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=60)
    resp.raise_for_status()
    hits = resp.json().get("data", {}).get("hits", [])
    for hit in hits:
        name = str(hit.get("file_name", ""))
        if name.startswith(str(slide_key)) and ".svs" in name.lower():
            return str(hit["file_id"]), name
    return None


def download_gdc_slide(download_root, project, case_id, slide_key):
    global DOWNLOAD_COUNT
    found = query_gdc_slide(case_id, slide_key)
    if found is None:
        return None
    file_id, file_name = found
    out_dir = Path(download_root) / project / slide_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_name
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    print(f"[download] {slide_key} -> {out_path}", flush=True)
    with requests.get(f"https://api.gdc.cancer.gov/data/{file_id}", stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.rename(out_path)
    DOWNLOAD_COUNT += 1
    return out_path


def slide_path_for(args, project, case_id, slide_key):
    roots = [Path(token.strip()) for token in str(args.slides_root).split(",") if token.strip()]
    for root in roots:
        for base in (root / project, root):
            found = find_slide_path(base, slide_key)
            if found is not None:
                return found
            if not base.exists():
                continue
            hits = sorted(base.glob(f"{slide_key}*/*.svs")) + sorted(base.glob(f"{slide_key}*.svs"))
            if hits:
                return hits[0]
    if str(args.download_slides) == "1" and DOWNLOAD_COUNT < int(args.max_downloads):
        try:
            return download_gdc_slide(args.download_root, project, case_id, slide_key)
        except Exception as exc:
            print(f"[warn] download failed for {slide_key}: {exc}", file=sys.stderr)
    return None


def diverse_order(rows, rng):
    by_project = defaultdict(list)
    for row in rows:
        by_project[row["project_dir"]].append(row)
    for project_rows in by_project.values():
        rng.shuffle(project_rows)
    projects = sorted(by_project, key=lambda p: (-len(by_project[p]), p))
    rng.shuffle(projects)
    ordered = []
    while projects:
        next_projects = []
        for project in projects:
            if by_project[project]:
                ordered.append(by_project[project].pop())
            if by_project[project]:
                next_projects.append(project)
        projects = next_projects
    return ordered


def sample_region(args, row, label, pair_index, direction, rng):
    h5_path = Path(row["h5_path"])
    if not h5_path.exists():
        return None
    slide_path = slide_path_for(args, row["project_dir"], row["case_id"], row["slide_key"])
    if slide_path is None:
        return None
    features, coords = read_h5_features_coords(h5_path)
    if int(features.shape[1]) != 1536:
        return None
    side = int(args.region_size) // int(args.grid_step_px)
    if side * int(args.grid_step_px) != int(args.region_size):
        raise ValueError("region-size must be divisible by grid-step-px")
    tile_size = infer_coord_tile_size(coords)
    cell_to_index = build_cell_maps(coords, tile_size)
    starts = valid_starts(cell_to_index, side)
    rng.shuffle(starts)
    slide = open_slide(slide_path)
    try:
        objective = infer_objective_power(slide)
        effective = effective_grid_step_at_target_mag(tile_size, objective, args.target_magnification)
        if abs(float(effective) - float(args.grid_step_px)) > 16.0:
            return None
        for gx0, gy0 in starts[: int(args.max_region_tries_per_slide)]:
            x0 = int(gx0) * int(tile_size)
            y0 = int(gy0) * int(tile_size)
            image, crop_w0, crop_h0 = read_region_rgb_at_magnification(
                slide,
                x0=x0,
                y0=y0,
                out_w=int(args.region_size),
                out_h=int(args.region_size),
                target_magnification=float(args.target_magnification),
            )
            quality = quick_region_quality_metrics(image)
            if quality["tissue_score"] < float(args.min_tissue):
                continue
            if quality["dark_fraction"] < float(args.min_dark_fraction):
                continue
            if quality["saturation_fraction"] < float(args.min_saturation_fraction):
                continue
            zgrid = np.zeros((side, side, int(features.shape[1])), dtype=np.float32)
            for ly in range(side):
                for lx in range(side):
                    zgrid[ly, lx] = features[cell_to_index[(gx0 + lx, gy0 + ly)]]

            center = side // 2
            target_cells = [(center - 1, center - 1), (center, center - 1), (center - 1, center), (center, center)]
            region_id = (
                f"pair_{int(pair_index):02d}__{direction}__{row['slide_key']}"
                f"__mag_{str(args.target_magnification).replace('.', 'p')}__gx_{gx0}__gy_{gy0}"
            )
            region_dir = Path(args.out_root) / f"{direction}_regions" / f"label_{label}" / region_id
            region_dir.mkdir(parents=True, exist_ok=True)
            image_path = region_dir / "region.png"
            zgrid_path = region_dir / "region_zgrid.npy"
            cells_path = region_dir / "region_cells.png"
            overlay_path = region_dir / "edit_overlay.png"
            save_png(image, image_path)
            np.save(zgrid_path, zgrid)
            save_png(make_region_cells_preview(image, grid_step_px=int(args.grid_step_px)), cells_path)
            save_png(draw_cells_overlay(image, cells=target_cells, grid_step_px=int(args.grid_step_px)), overlay_path)
            out = {
                "region_id": region_id,
                "split": "labels_only_diverse",
                "label": 0 if label == "low" else 1,
                "hpv_status": label,
                "case_id": row["case_id"],
                "slide_key": row["slide_key"],
                "slide_path": str(slide_path),
                "canonical_h5_path": str(h5_path),
                "region_x": int(x0),
                "region_y": int(y0),
                "region_w": int(args.region_size),
                "region_h": int(args.region_size),
                "grid_step_px": int(args.grid_step_px),
                "feature_dim": int(features.shape[1]),
                "tissue_score": float(quality["tissue_score"]),
                "seed": int(args.seed),
                "image_path": str(image_path),
                "feature_grid_path": str(zgrid_path),
                "cell_preview_path": str(cells_path),
                "region_dir": str(region_dir),
                "project_dir": row["project_dir"],
                "source_label": label,
                "tumor_purity": row.get("tumor_purity", ""),
                "pair_index": int(pair_index),
                "direction": direction,
                "region_gx0": int(gx0),
                "region_gy0": int(gy0),
                "crop_w_level0": int(crop_w0),
                "crop_h_level0": int(crop_h0),
                "objective_power": float(objective),
                "tile_size_level0": int(tile_size),
                "effective_grid_step_at_target_mag": float(effective),
                "edit_overlay_path": str(overlay_path),
                "target_cells": ";".join(f"{x},{y}" for x, y in target_cells),
            }
            write_json(region_dir / "region_meta.json", out)
            return out
    finally:
        slide.close()
    return None


def build_direction(args, source_label, target_label, direction):
    rng = random.Random(int(args.seed) + (11 if source_label == "low" else 29))
    rows = [r for r in read_rows(args.manifest_csv) if r.get("label") == source_label]
    rows = diverse_order(rows, rng)
    used_cases = set()
    selected = []
    skipped = []
    for row in rows[: int(args.max_slide_scan)]:
        if len(selected) >= int(args.n_pairs):
            break
        if row["case_id"] in used_cases:
            continue
        try:
            region = sample_region(args, row, source_label, len(selected) + 1, direction, rng)
        except Exception as exc:
            skipped.append({**row, "reason": str(exc)})
            continue
        if region is None:
            skipped.append({**row, "reason": "no_valid_region_or_slide"})
            continue
        selected.append(region)
        used_cases.add(row["case_id"])
        print(f"[region] {direction} {len(selected)}/{args.n_pairs}: {region['region_id']}", flush=True)
    if len(selected) < int(args.n_pairs):
        raise RuntimeError(f"{direction}: only found {len(selected)} valid regions out of {args.n_pairs}")

    out_dir = Path(args.out_root) / f"{direction}_regions"
    write_region_bank_csv(out_dir / "region_bank.csv", selected)
    write_csv(out_dir / "skipped_region_candidates.csv", skipped)
    edit_manifest = []
    for row in selected:
        cells = [
            {"gx": int(token.split(",")[0]), "gy": int(token.split(",")[1])}
            for token in str(row["target_cells"]).split(";")
            if token
        ]
        edit_manifest.append(
            {
                "run_id": f"{row['region_id']}__to_{target_label}_concepts",
                "region_id": row["region_id"],
                "source_label": source_label,
                "target_label": target_label,
                "target_cells": cells,
                "selector": "random_tissue_center_2x2_no_classifier",
            }
        )
    write_json(out_dir / "progressive_edit_manifest.json", edit_manifest)
    return selected


parser = argparse.ArgumentParser()
parser.add_argument("--manifest-csv", type=Path, required=True)
parser.add_argument("--slides-root", type=Path, required=True)
parser.add_argument("--download-slides", type=str, default="1")
parser.add_argument("--download-root", type=Path, required=True)
parser.add_argument("--max-downloads", type=int, default=40)
parser.add_argument("--out-root", type=Path, required=True)
parser.add_argument("--n-pairs", type=int, required=True)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--region-size", type=int, default=2048)
parser.add_argument("--grid-step-px", type=int, default=256)
parser.add_argument("--target-magnification", type=float, default=20.0)
parser.add_argument("--min-tissue", type=float, default=0.45)
parser.add_argument("--min-dark-fraction", type=float, default=0.04)
parser.add_argument("--min-saturation-fraction", type=float, default=0.04)
parser.add_argument("--max-region-tries-per-slide", type=int, default=128)
parser.add_argument("--max-slide-scan", type=int, default=500)
args = parser.parse_args()
args.out_root.mkdir(parents=True, exist_ok=True)

low_regions = build_direction(args, "low", "high", "low_to_high")
high_regions = build_direction(args, "high", "low", "high_to_low")
write_json(
    args.out_root / "region_pair_summary.json",
    {
        "n_pairs": int(args.n_pairs),
        "low_to_high_regions": len(low_regions),
        "high_to_low_regions": len(high_regions),
        "low_to_high_region_bank": str(args.out_root / "low_to_high_regions/region_bank.csv"),
        "high_to_low_region_bank": str(args.out_root / "high_to_low_regions/region_bank.csv"),
    },
)
PY
fi

if [[ "${SKIP_EDITS}" != "1" ]]; then
  echo "[edit] low -> high" >&2
  "$PY" scripts/run_progressive_region_edit.py \
    --task "${TASK}" \
    --region-bank-csv "${LOW_TO_HIGH_DIR}/region_bank.csv" \
    --edit-manifest "${LOW_TO_HIGH_DIR}/progressive_edit_manifest.json" \
    --out-dir "${LOW_TO_HIGH_OUT}" \
    --concepts-json "${CONCEPT_ROOT}/high/selected_concepts.json" \
    --representative-tiles-csv "${CONCEPT_ROOT}/high/representative_tiles.csv" \
    --concept-class-label high \
    --concept-ranking-method activation \
    --concept-target-stat "${CONCEPT_TARGET_STAT}" \
    --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
    --concept-steering-mode "${CONCEPT_STEERING_MODE}" \
    --max-concepts "${MAX_CONCEPTS}" \
    --max-runs "${N_PAIRS}" \
    --target-magnification "${TARGET_MAGNIFICATION}" \
    --edit-support center_2x2 \
    --prototype-strength 0.8 \
    --steer-blend 1.0 \
    --preserve-edit-strength 0.05 \
    --preserve-visited-strength 0.95 \
    --preserve-fresh-context-strength 0.35 \
    --mid-steer-start-ratio 0.5 \
    --mid-steer-end-ratio 1.0 \
    --mid-steer-alpha-start 0.5 \
    --mid-steer-alpha-end 1.0 \
    --mid-steer-alpha-schedule linear \
    --steps "${STEPS}" \
    --guidance 2.0 \
    --patch-batch 256 \
    --output-mode "${OUTPUT_MODE}" \
    --device "${DEVICE}"

  echo "[edit] high -> low" >&2
  "$PY" scripts/run_progressive_region_edit.py \
    --task "${TASK}" \
    --region-bank-csv "${HIGH_TO_LOW_DIR}/region_bank.csv" \
    --edit-manifest "${HIGH_TO_LOW_DIR}/progressive_edit_manifest.json" \
    --out-dir "${HIGH_TO_LOW_OUT}" \
    --concepts-json "${CONCEPT_ROOT}/low/selected_concepts.json" \
    --representative-tiles-csv "${CONCEPT_ROOT}/low/representative_tiles.csv" \
    --concept-class-label low \
    --concept-ranking-method activation \
    --concept-target-stat "${CONCEPT_TARGET_STAT}" \
    --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
    --concept-steering-mode "${CONCEPT_STEERING_MODE}" \
    --max-concepts "${MAX_CONCEPTS}" \
    --max-runs "${N_PAIRS}" \
    --target-magnification "${TARGET_MAGNIFICATION}" \
    --edit-support center_2x2 \
    --prototype-strength 0.8 \
    --steer-blend 1.0 \
    --preserve-edit-strength 0.05 \
    --preserve-visited-strength 0.95 \
    --preserve-fresh-context-strength 0.35 \
    --mid-steer-start-ratio 0.5 \
    --mid-steer-end-ratio 1.0 \
    --mid-steer-alpha-start 0.5 \
    --mid-steer-alpha-end 1.0 \
    --mid-steer-alpha-schedule linear \
    --steps "${STEPS}" \
    --guidance 2.0 \
    --patch-batch 256 \
    --output-mode "${OUTPUT_MODE}" \
    --device "${DEVICE}"
fi

echo "[ok] low->high regions: ${LOW_TO_HIGH_DIR}" >&2
echo "[ok] low->high edits:   ${LOW_TO_HIGH_OUT}" >&2
echo "[ok] high->low regions: ${HIGH_TO_LOW_DIR}" >&2
echo "[ok] high->low edits:   ${HIGH_TO_LOW_OUT}" >&2
