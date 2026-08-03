#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
N_REGIONS="${N_REGIONS:-1}"
SEED="${SEED:-7}"
OUT_DIR="${OUT_DIR:-paper_outputs/steering_strength_sweeps/kirc_low_vs_high_grade/low_to_high}"
REGION_ROOT="${REGION_ROOT:-artifacts/morphology_label_concept_review_top1_showcase_best_10slides/_regions/03_kirc_low_to_high}"
CONCEPT_ROOT="${CONCEPT_ROOT:-artifacts/classifier_label_concepts_all_top10_top50/kirc_low_vs_high_grade/high}"

extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

"${PYTHON_BIN}" scripts/run_steering_strength_sweep.py \
  --task-name kirc_low_vs_high_grade \
  --direction low_to_high \
  --runner-direction hpv_pos \
  --source-label low \
  --target-label high \
  --label-order high,low \
  --region-bank-csv "${REGION_ROOT}/region_bank.csv" \
  --base-edit-manifest "${REGION_ROOT}/progressive_edit_manifest.json" \
  --classifier-run-dir artifacts/classifier_training/kirc_low_vs_high_grade \
  --concepts-json "${CONCEPT_ROOT}/selected_concepts.json" \
  --representative-tiles-csv "${CONCEPT_ROOT}/representative_tiles.csv" \
  --concept-class-label high \
  --max-concepts 1 \
  --sae-variant relu_sae_base \
  --n-regions "${N_REGIONS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --out-dir "${OUT_DIR}" \
  --title "KIRC low to high grade: controlled steering strength" \
  "${extra_args[@]}"

