#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
REGION_DIR="${REGION_DIR:-artifacts/luad_to_lusc_20x_2048_regions10}"
OUT_DIR="${OUT_DIR:-artifacts/luad_to_lusc_20x_2048_regions10_concept_edit}"
CONCEPT_DIR="${CONCEPT_DIR:-artifacts/classifier_label_concepts_all_top10_top50/luad_lusc/LUSC}"
STEPS="${STEPS:-30}"
MAX_RUNS="${MAX_RUNS:-10}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"

"$PY" scripts/find_regions.py \
  --mode attention \
  --classifier-run-dir artifacts/classifier_training/luad_lusc \
  --source-label LUAD \
  --target-label LUSC \
  --slides-root /research/projects/mllab/WSI/TCGA_features \
  --target-magnification 20 \
  --region-size 2048 \
  --grid-step-px 256 \
  --max-regions "${MAX_RUNS}" \
  --max-candidates-per-slide 4 \
  --attention-percentile 85 \
  --min-selected-cells 4 \
  --max-selected-cells 12 \
  --target-importance-mass 0.45 \
  --min-tissue 0.45 \
  --min-dark-fraction 0.04 \
  --min-saturation-fraction 0.04 \
  --out-dir "${REGION_DIR}" \
  --device "${DEVICE}"

"$PY" scripts/run_progressive_region_edit.py \
  --task luad_lusc \
  --region-bank-csv "${REGION_DIR}/region_bank.csv" \
  --edit-manifest "${REGION_DIR}/progressive_edit_manifest.json" \
  --out-dir "${OUT_DIR}" \
  --concepts-json "${CONCEPT_DIR}/selected_concepts.json" \
  --representative-tiles-csv "${CONCEPT_DIR}/representative_tiles.csv" \
  --concept-class-label LUSC \
  --concept-ranking-method attention_weighted \
  --concept-target-stat median \
  --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
  --max-concepts 0 \
  --max-runs "${MAX_RUNS}" \
  --target-magnification 20 \
  --edit-support border_relaxed \
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
  --output-mode minimal \
  --device "${DEVICE}"

echo "[ok] regions: ${REGION_DIR}"
echo "[ok] edits:   ${OUT_DIR}"
