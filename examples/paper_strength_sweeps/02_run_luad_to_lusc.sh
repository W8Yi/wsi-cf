#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
N_REGIONS="${N_REGIONS:-1}"
SEED="${SEED:-7}"
OUT_DIR="${OUT_DIR:-paper_outputs/steering_strength_sweeps/luad_lusc/luad_to_lusc}"
CONCEPT_ROOT="${CONCEPT_ROOT:-artifacts/classifier_label_concepts_all_top10_top50/luad_lusc/LUSC}"

extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

"${PYTHON_BIN}" scripts/run_steering_strength_sweep.py \
  --task-name luad_lusc \
  --direction luad_to_lusc \
  --runner-direction hpv_pos \
  --source-label LUAD \
  --target-label LUSC \
  --label-order LUAD,LUSC \
  --region-bank-csv artifacts/luad_to_lusc_20x_2048_regions10/region_bank.csv \
  --base-edit-manifest artifacts/luad_to_lusc_20x_2048_regions10/progressive_edit_manifest.json \
  --classifier-run-dir artifacts/classifier_training/luad_lusc \
  --concepts-json "${CONCEPT_ROOT}/selected_concepts.json" \
  --representative-tiles-csv "${CONCEPT_ROOT}/representative_tiles.csv" \
  --concept-class-label LUSC \
  --max-concepts 1 \
  --sae-variant relu_sae_base \
  --n-regions "${N_REGIONS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --out-dir "${OUT_DIR}" \
  --title "LUAD to LUSC: controlled steering strength" \
  "${extra_args[@]}"

