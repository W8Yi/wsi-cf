#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

# This test isolates why KIRC grade edits may look subtle. It reuses one fixed
# low->high region and one fixed high-grade concept, then varies strength,
# target sharpness, diffusion schedule, preservation, and edit area.
SOURCE_REGION_DIR="${SOURCE_REGION_DIR:-artifacts/kirc_low_high_20x_2048_each_concept_strength_sweep_prototype_retest/low_to_high_regions3}"
if [[ ! -f "${SOURCE_REGION_DIR}/region_bank.csv" ]]; then
  SOURCE_REGION_DIR="artifacts/kirc_low_high_20x_2048_each_concept_strength_sweep/low_to_high_regions3"
fi

CONCEPT_JSON="${CONCEPT_JSON:-artifacts/kirc_low_high_20x_2048_each_concept_strength_sweep_prototype_retest/_single_concept_jsons/high/high_rank_01_latent_4766.json}"
if [[ ! -f "${CONCEPT_JSON}" ]]; then
  CONCEPT_JSON="artifacts/kirc_low_high_20x_2048_each_concept_strength_sweep/_single_concept_jsons/high/high_rank_01_latent_4766.json"
fi
REPRESENTATIVE_TILES_CSV="${REPRESENTATIVE_TILES_CSV:-artifacts/classifier_label_concepts_all_top10_top50/kirc_low_vs_high_grade/high/representative_tiles.csv}"

OUT_ROOT="${OUT_ROOT:-artifacts/kirc_grade_visibility_ablation_prototype}"
PREP_DIR="${OUT_ROOT}/_prepared"
mkdir -p "${PREP_DIR}"

STEPS="${STEPS:-24}"
GUIDANCE="${GUIDANCE:-2.0}"
PATCH_BATCH="${PATCH_BATCH:-256}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
RUN_FULL_GRID="${RUN_FULL_GRID:-0}"

if [[ ! -f "${SOURCE_REGION_DIR}/region_bank.csv" || ! -f "${SOURCE_REGION_DIR}/progressive_edit_manifest.json" ]]; then
  echo "[error] missing source KIRC region bundle: ${SOURCE_REGION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CONCEPT_JSON}" || ! -f "${REPRESENTATIVE_TILES_CSV}" ]]; then
  echo "[error] missing concept inputs:" >&2
  echo "  CONCEPT_JSON=${CONCEPT_JSON}" >&2
  echo "  REPRESENTATIVE_TILES_CSV=${REPRESENTATIVE_TILES_CSV}" >&2
  exit 1
fi

"$PY" - <<PY
import csv
import json
import shutil
from pathlib import Path

source_dir = Path("${SOURCE_REGION_DIR}")
prep_dir = Path("${PREP_DIR}")
prep_dir.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader((source_dir / "region_bank.csv").open()))
if not rows:
    raise SystemExit("empty region_bank.csv")
row = rows[0]
region_id = row["region_id"]

with (prep_dir / "region_bank_one.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerow(row)

source_manifest = json.loads((source_dir / "progressive_edit_manifest.json").read_text())
request = next((item for item in source_manifest if item.get("region_id") == region_id), source_manifest[0])
original_cells = [(int(c["gx"]), int(c["gy"])) for c in request["target_cells"]]

def write_manifest(name, cells, note):
    seen = set()
    deduped = []
    for gx, gy in cells:
        if 0 <= gx < 8 and 0 <= gy < 8 and (gx, gy) not in seen:
            deduped.append({"gx": int(gx), "gy": int(gy)})
            seen.add((gx, gy))
    payload = [{
        "run_id": f"{region_id}__{name}",
        "region_id": region_id,
        "target_cells": deduped,
        "selector": "visibility_ablation",
        "ablation_area": name,
        "note": note,
    }]
    path = prep_dir / f"manifest_{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\\n")
    return path, len(deduped)

cx, cy = 3, 3
center_2x2 = [(3, 3), (4, 3), (3, 4), (4, 4)]
center_4x4 = [(gx, gy) for gy in range(2, 6) for gx in range(2, 6)]
expanded_original = sorted({
    (gx + dx, gy + dy)
    for gx, gy in original_cells
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if 0 <= gx + dx < 8 and 0 <= gy + dy < 8
}, key=lambda c: (c[1], c[0]))
full_8x8 = [(gx, gy) for gy in range(8) for gx in range(8)]

manifest_info = {}
for name, cells, note in [
    ("original", original_cells, "Original attention-selected cells from find_regions.py."),
    ("expanded_original", expanded_original, "Original cells plus 8-neighbor expansion."),
    ("center_2x2", center_2x2, "A contiguous central 2x2 block."),
    ("center_4x4", center_4x4, "A contiguous central 4x4 block."),
    ("full_8x8", full_8x8, "All cells in the 2048 region; optional heavy ablation."),
]:
    path, count = write_manifest(name, cells, note)
    manifest_info[name] = {"path": str(path), "cell_count": count, "note": note}

summary = {
    "source_region_dir": str(source_dir),
    "region_id": region_id,
    "region_bank_csv": str(prep_dir / "region_bank_one.csv"),
    "original_cells": original_cells,
    "manifests": manifest_info,
}
(prep_dir / "ablation_inputs.json").write_text(json.dumps(summary, indent=2) + "\\n")
print(json.dumps(summary, indent=2))
PY

run_case() {
  local case_name="$1"
  local manifest_name="$2"
  local strength="$3"
  local target_stat="$4"
  local target_top_k="$5"
  local preserve_edit="$6"
  local preserve_visited="$7"
  local preserve_fresh="$8"
  local mid_start="$9"
  local mid_alpha_start="${10}"
  local edit_support="${11}"

  local out_dir="${OUT_ROOT}/${case_name}"
  echo "[run] ${case_name}"
  "$PY" scripts/run_progressive_region_edit.py \
    --task kirc_low_vs_high_grade \
    --region-bank-csv "${PREP_DIR}/region_bank_one.csv" \
    --edit-manifest "${PREP_DIR}/manifest_${manifest_name}.json" \
    --out-dir "${out_dir}" \
    --concepts-json "${CONCEPT_JSON}" \
    --representative-tiles-csv "${REPRESENTATIVE_TILES_CSV}" \
    --concept-class-label high \
    --concept-ranking-method attention_weighted \
    --concept-target-stat "${target_stat}" \
    --concept-target-top-k "${target_top_k}" \
    --concept-steering-mode prototype_vector \
    --max-concepts 1 \
    --max-runs 1 \
    --target-magnification 20 \
    --edit-support "${edit_support}" \
    --prototype-strength "${strength}" \
    --steer-blend 1.0 \
    --preserve-edit-strength "${preserve_edit}" \
    --preserve-visited-strength "${preserve_visited}" \
    --preserve-fresh-context-strength "${preserve_fresh}" \
    --mid-steer-start-ratio "${mid_start}" \
    --mid-steer-end-ratio 1.0 \
    --mid-steer-alpha-start "${mid_alpha_start}" \
    --mid-steer-alpha-end 1.0 \
    --mid-steer-alpha-schedule linear \
    --steps "${STEPS}" \
    --guidance "${GUIDANCE}" \
    --patch-batch "${PATCH_BATCH}" \
    --output-mode "${OUTPUT_MODE}" \
    --device "${DEVICE}"
}

# Baseline: current corrected prototype-vector behavior.
run_case "01_baseline_original_s0p4_median_top5" "original" "0.4" "median" "5" "0.05" "0.95" "0.35" "0.5" "0.5" "border_relaxed"

# Strength: same cells, same prototype definition.
run_case "02_strength_original_s1p0_median_top5" "original" "1.0" "median" "5" "0.05" "0.95" "0.35" "0.5" "0.5" "border_relaxed"
run_case "03_strength_original_s1p5_median_top5" "original" "1.5" "median" "5" "0.05" "0.95" "0.35" "0.5" "0.5" "border_relaxed"

# Prototype sharpness: use only the strongest representative tile and max prototype.
run_case "04_sharp_proto_original_s1p0_max_top1" "original" "1.0" "max" "1" "0.05" "0.95" "0.35" "0.5" "0.5" "border_relaxed"

# Schedule: apply steered conditioning from the beginning of diffusion.
run_case "05_full_schedule_original_s1p0" "original" "1.0" "median" "5" "0.05" "0.95" "0.35" "0.0" "1.0" "border_relaxed"

# Preservation: let edited cells and fresh context move more freely.
run_case "06_low_preserve_original_s1p0" "original" "1.0" "median" "5" "0.0" "0.75" "0.15" "0.5" "0.5" "border_relaxed"
run_case "06b_low_preserve_original_s1p5" "original" "1.5" "median" "5" "0.0" "0.75" "0.15" "0.5" "0.5" "border_relaxed"

# Edit-area tests: same prototype, larger contiguous target footprints.
run_case "07_center2x2_s1p0" "center_2x2" "1.0" "median" "5" "0.05" "0.95" "0.35" "0.5" "0.5" "border_relaxed"
run_case "08_center4x4_s1p0" "center_4x4" "1.0" "median" "5" "0.05" "0.95" "0.35" "0.5" "0.5" "border_relaxed"
run_case "09_expanded_original_s1p0" "expanded_original" "1.0" "median" "5" "0.05" "0.95" "0.35" "0.5" "0.5" "border_relaxed"

# Combined strongest reasonable test: larger area, sharper prototype, lower preservation, full schedule.
run_case "10_center4x4_combined_strong" "center_4x4" "1.5" "max" "1" "0.0" "0.75" "0.15" "0.0" "1.0" "border_relaxed"

if [[ "${RUN_FULL_GRID}" == "1" ]]; then
  run_case "11_full8x8_optional_heavy" "full_8x8" "1.0" "median" "5" "0.0" "0.75" "0.15" "0.0" "1.0" "border_relaxed"
fi

"$PY" - <<PY
import csv
import json
from pathlib import Path

root = Path("${OUT_ROOT}")
rows = []
for manifest_path in sorted(root.glob("*/**/run_manifest.json")):
    data = json.loads(manifest_path.read_text())
    rows.append({
        "case": manifest_path.parts[len(root.parts)],
        "run_id": data.get("run_id", ""),
        "output_path": data.get("output_path", ""),
        "source_image_path": data.get("source_image_path", ""),
        "n_target_cells": len(data.get("target_cells", [])),
        "n_windows": len(data.get("window_history", [])),
        "prototype_strength": data.get("prototype_strength", ""),
        "concept_steering_mode": data.get("concept_steering", {}).get("steering_mode", ""),
        "prototype_norm": data.get("concept_steering", {}).get("prototype_norm", ""),
        "target_stat": data.get("concept_steering", {}).get("target_stat", ""),
        "target_top_k": data.get("concept_steering", {}).get("target_top_k", ""),
        "preserve_edit_strength": data.get("preserve_edit_strength", ""),
        "preserve_visited_strength": data.get("preserve_visited_strength", ""),
        "preserve_fresh_context_strength": data.get("preserve_fresh_context_strength", ""),
        "mid_steer_start_ratio": data.get("mid_steer_start_ratio", ""),
        "mid_steer_alpha_start": data.get("mid_steer_alpha_start", ""),
    })
out = root / "ablation_summary.csv"
if rows:
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"[ok] wrote {out} with {len(rows)} rows")
PY

echo "[ok] KIRC visibility ablation written to ${OUT_ROOT}"
echo "     summary: ${OUT_ROOT}/ablation_summary.csv"
echo "     inputs:  ${PREP_DIR}/ablation_inputs.json"
