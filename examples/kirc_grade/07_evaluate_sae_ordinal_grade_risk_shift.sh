#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EDIT_ROOT="${EDIT_ROOT:-artifacts/morphology_label_concept_review_top1_showcase_best_10slides}"
MODEL_CKPT="${MODEL_CKPT:-artifacts/sae_grade_risk_training/kirc_batch_topk_ordinal/best_model.pt}"

"$PY" scripts/evaluate_kirc_sae_grade_risk_edits.py \
  --model-ckpt "${MODEL_CKPT}" \
  --edit-root "${EDIT_ROOT}" \
  --out-dir "${EDIT_ROOT}/sae_grade_risk_eval" \
  --max-runs-per-direction 10 \
  --edit-origin binary_classifier_selected_existing_edits \
  --device "${DEVICE}" \
  "$@"
