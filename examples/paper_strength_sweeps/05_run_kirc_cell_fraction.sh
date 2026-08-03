#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:5}"
TASK="${TASK:-kirc_normal_tumor}"
REGION_ID="${REGION_ID:-TCGA-B2-3923-11A-01-TS1__normal_to_tumor__mag_20p0__gx_35__gy_21}"
SEED="${SEED:-7}"
FIXED_STRENGTH="${FIXED_STRENGTH:-0.9}"
FRACTIONS="${FRACTIONS:-0,0.2,0.4,0.6,0.8,1.0}"
PATCH_BATCH="${PATCH_BATCH:-128}"
EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"
REGION_ROOT="${REGION_ROOT:-artifacts/normal_to_tumor_regions_showcase_sae_cell_fraction_sweep_padded_center_commit_rerun_diverse/${TASK}}"
CLASSIFIER_ROOT="${CLASSIFIER_ROOT:-artifacts/classifier_training_normal_tumor/${TASK}}"
CONCEPT_ROOT="${CONCEPT_ROOT:-artifacts/concept_discovery_normal_tumor_relu_sae_base/${TASK}/labels/tumor}"
OUT_DIR="${OUT_DIR:-paper_outputs/cell_fraction_sweeps/${TASK}/normal_to_tumor_strength090_seed7_policy09}"

extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

"${PYTHON_BIN}" scripts/run_cell_fraction_sweep.py \
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
  --region-id "${REGION_ID}" \
  --fractions "${FRACTIONS}" \
  --fixed-strength "${FIXED_STRENGTH}" \
  --seed "${SEED}" \
  --patch-batch "${PATCH_BATCH}" \
  --edit-policy "${EDIT_POLICY}" \
  --runner-extra "--preserve-invalid-feature-cells --invalid-feature-feather-px 64 --preserve-outside-target-cells --target-cell-feather-px 64" \
  --device "${DEVICE}" \
  --out-dir "${OUT_DIR}" \
  "${extra_args[@]}"
