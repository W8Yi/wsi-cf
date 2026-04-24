#!/usr/bin/env bash
set -euo pipefail

PACE_PY=/common/users/wq50/envs/pace/bin/python
WSI_CF=/common/users/wq50/wsi_cf

BANK_DIR=${WSI_CF}/artifacts/hnscc_region_bank_20x_2048_sample20
BANK_CSV=${BANK_DIR}/region_bank.csv
MANIFEST=${WSI_CF}/artifacts/edit_manifests/all_regions_20x_random_targets.json

OUT_POS=${WSI_CF}/artifacts/progressive_edit_20x_all_regions_to_hpv_pos
OUT_NEG=${WSI_CF}/artifacts/progressive_edit_20x_all_regions_to_hpv_neg

mkdir -p "${WSI_CF}/artifacts/edit_manifests"

# 1. Sample 20 total 20x regions, balanced across labels.
#    The script name says "10x" historically, but target magnification is configurable.
${PACE_PY} ${WSI_CF}/scripts/export_hnscc_region_bank_10x.py \
  --out-dir "${BANK_DIR}" \
  --target-magnification 20 \
  --region-size 2048 \
  --grid-step-px 256 \
  --regions-total 20 \
  --regions-per-slide 1 \
  --min-tissue 0.35 \
  --max-region-tries 64 \
  --seed 7 \
  --device cuda:0 \
  --dtype fp16

# 2. Build one manifest item per region.
#    Each region gets a random valid target pattern.
#    Targets are restricted to cells that are editable under the enforced center-2x2 rule.
${PACE_PY} - <<'PY'
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

WSI_CF = Path("/common/users/wq50/wsi_cf")
sys.path.insert(0, str(WSI_CF / "src"))

from wsi_cf.steering.progressive import enumerate_progressive_windows, center_support_global_cells

bank_csv = Path("/common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_20x_2048_sample20/region_bank.csv")
manifest_path = Path("/common/users/wq50/wsi_cf/artifacts/edit_manifests/all_regions_20x_random_targets.json")

rng = random.Random(7)

def neighbors4(cell, valid_set):
    gx, gy = cell
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (gx + dx, gy + dy)
        if nxt in valid_set:
            out.append(nxt)
    return out

def sample_connected(valid_set, count, rng):
    start = rng.choice(sorted(valid_set))
    selected = [start]
    selected_set = {start}
    while len(selected) < count:
        frontier = []
        for cell in selected:
            for nxt in neighbors4(cell, valid_set):
                if nxt not in selected_set and nxt not in frontier:
                    frontier.append(nxt)
        if not frontier:
            break
        nxt = rng.choice(frontier)
        selected.append(nxt)
        selected_set.add(nxt)
    return sorted(selected, key=lambda x: (x[1], x[0]))

def sample_two_cluster(valid_set, rng):
    first_n = rng.randint(1, 3)
    second_n = rng.randint(1, 3)
    cluster1 = sample_connected(valid_set, first_n, rng)
    remaining = set(valid_set)
    for c in cluster1:
        remaining.discard(c)
        for n in neighbors4(c, valid_set):
            remaining.discard(n)
    if not remaining:
        return cluster1
    cluster2 = sample_connected(remaining, second_n, rng)
    merged = sorted(set(cluster1).union(cluster2), key=lambda x: (x[1], x[0]))
    return merged

def sample_line(valid_set, length, orientation, rng):
    candidates = []
    for gx, gy in sorted(valid_set):
        if orientation == "h":
            cells = [(gx + i, gy) for i in range(length)]
        else:
            cells = [(gx, gy + i) for i in range(length)]
        if all(cell in valid_set for cell in cells):
            candidates.append(cells)
    if not candidates:
        return None
    return sorted(rng.choice(candidates), key=lambda x: (x[1], x[0]))

def sample_block(valid_set, w, h, rng):
    candidates = []
    for gx, gy in sorted(valid_set):
        cells = [(gx + dx, gy + dy) for dy in range(h) for dx in range(w)]
        if all(cell in valid_set for cell in cells):
            candidates.append(cells)
    if not candidates:
        return None
    return sorted(rng.choice(candidates), key=lambda x: (x[1], x[0]))

def choose_random_targets(valid_cells, rng):
    valid_set = set(valid_cells)
    modes = [
        "single",
        "pair",
        "triple",
        "line_h_4",
        "line_v_4",
        "block_2x2",
        "connected_big",
        "two_cluster",
    ]
    mode = rng.choice(modes)

    if mode == "single":
        targets = [rng.choice(sorted(valid_set))]
    elif mode == "pair":
        targets = sample_connected(valid_set, 2, rng)
    elif mode == "triple":
        targets = sample_connected(valid_set, 3, rng)
    elif mode == "line_h_4":
        targets = sample_line(valid_set, 4, "h", rng) or sample_connected(valid_set, 4, rng)
    elif mode == "line_v_4":
        targets = sample_line(valid_set, 4, "v", rng) or sample_connected(valid_set, 4, rng)
    elif mode == "block_2x2":
        targets = sample_block(valid_set, 2, 2, rng) or sample_connected(valid_set, 4, rng)
    elif mode == "connected_big":
        targets = sample_connected(valid_set, rng.randint(4, 6), rng)
    elif mode == "two_cluster":
        targets = sample_two_cluster(valid_set, rng)
    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    targets = sorted(set(targets), key=lambda x: (x[1], x[0]))
    return mode, targets

rows = list(csv.DictReader(bank_csv.open()))
rows.sort(key=lambda r: (int(r["label"]), str(r["slide_key"]), str(r["region_id"])))

manifest = []
for idx, row in enumerate(rows, start=1):
    zgrid = np.load(row["feature_grid_path"])
    grid_h, grid_w = int(zgrid.shape[0]), int(zgrid.shape[1])

    windows = enumerate_progressive_windows(
        grid_w=grid_w,
        grid_h=grid_h,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=int(row["grid_step_px"]),
    )
    valid_cells = sorted(
        set().union(*[center_support_global_cells(w) for w in windows]),
        key=lambda x: (x[1], x[0]),
    )
    mode, targets = choose_random_targets(valid_cells, rng)

    manifest.append(
        {
            "run_id": f"test_{idx:02d}_{row['slide_key']}",
            "region_id": row["region_id"],
            "target_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in targets],
            "source_method": "random_shape_batch",
            "shape_kind": mode,
            "num_targets": len(targets),
            "source_label": int(row["label"]),
            "source_slide_key": row["slide_key"],
            "note": "20x random target-shape test over all sampled regions"
        }
    )

manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"[ok] wrote {manifest_path}")
print(f"[ok] total runs: {len(manifest)}")
label_counts = {}
for item in manifest:
    label = item["source_label"]
    label_counts[label] = label_counts.get(label, 0) + 1
print(f"[ok] source label counts: {label_counts}")
PY

# 3. Run all regions steering toward HPV+
${PACE_PY} ${WSI_CF}/scripts/run_progressive_region_edit.py \
  --region-bank-csv "${BANK_CSV}" \
  --edit-manifest "${MANIFEST}" \
  --out-dir "${OUT_POS}" \
  --direction hpv_pos \
  --device cuda:0

# 4. Run the same regions and same target patterns steering toward HPV-
${PACE_PY} ${WSI_CF}/scripts/run_progressive_region_edit.py \
  --region-bank-csv "${BANK_CSV}" \
  --edit-manifest "${MANIFEST}" \
  --out-dir "${OUT_NEG}" \
  --direction hpv_neg \
  --device cuda:0
