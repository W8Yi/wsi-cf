#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-kirc_normal_tumor}"
N_REGIONS="${N_REGIONS:-1}"
SEED="${SEED:-7}"
PATCH_BATCH="${PATCH_BATCH:-128}"
TITLE="${TITLE:-KIRC normal tissue to tumor: controlled steering-strength sweep}"
EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"
OUT_DIR="${OUT_DIR:-paper_outputs/steering_strength_sweeps/${TASK}/normal_to_tumor}"
REGION_ROOT="${REGION_ROOT:-artifacts/normal_to_tumor_regions_showcase_sae_cell_fraction_sweep_padded_center_commit_rerun_diverse/${TASK}}"
CLASSIFIER_ROOT="${CLASSIFIER_ROOT:-artifacts/classifier_training_normal_tumor/${TASK}}"
CONCEPT_ROOT="${CONCEPT_ROOT:-artifacts/concept_discovery_normal_tumor_relu_sae_base/${TASK}/labels/tumor}"

extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi
if [[ "${ALL_REGION_CELLS:-0}" == "1" ]]; then
  extra_args+=(--all-region-cells)
fi

"${PYTHON_BIN}" scripts/run_steering_strength_sweep.py \
  --task-name "${TASK}" \
  --direction normal_to_tumor \
  --runner-direction hpv_pos \
  --source-label normal \
  --target-label tumor \
  --label-order normal,tumor \
  --region-bank-csv "${REGION_ROOT}/region_bank.csv" \
  --base-edit-manifest "${REGION_ROOT}/progressive_edit_manifest.json" \
  --classifier-run-dir "${CLASSIFIER_ROOT}" \
  --concepts-json "${CONCEPT_ROOT}/selected_concepts.json" \
  --representative-tiles-csv "${CONCEPT_ROOT}/representative_tiles.csv" \
  --concept-class-label tumor \
  --max-concepts 1 \
  --sae-variant relu_sae_base \
  --n-regions "${N_REGIONS}" \
  --seed "${SEED}" \
  --patch-batch "${PATCH_BATCH}" \
  --edit-policy "${EDIT_POLICY}" \
  --device "${DEVICE}" \
  --out-dir "${OUT_DIR}" \
  --title "${TITLE}" \
  "${extra_args[@]}"
