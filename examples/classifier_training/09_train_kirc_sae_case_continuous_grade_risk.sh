#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-artifacts/sae_grade_risk_training}"
TASK_NAME="${TASK_NAME:-kirc_batch_topk_case_ordinal_aux}"
FEATURE_CACHE="${FEATURE_CACHE:-artifacts/sae_grade_risk_training/kirc_batch_topk_ordinal/slide_sae_features.npz}"
TOP_CONCEPTS="${TOP_CONCEPTS:-30}"
SELECTION_FOLDS="${SELECTION_FOLDS:-5}"
EPOCHS="${EPOCHS:-200}"
LOW_HIGH_LOSS_WEIGHT="${LOW_HIGH_LOSS_WEIGHT:-0.25}"
MONOTONICITY_PENALTY="${MONOTONICITY_PENALTY:-0.10}"

"$PY" scripts/train_sae_concept_grade_risk.py \
  --task-name "${TASK_NAME}" \
  --projects TCGA-KIRC \
  --label-column tumor_grade \
  --target-map G1:0.0,G2:0.33,G3:0.66,G4:1.0 \
  --sae-variant tcga_sae_batch_topk_20x_interp \
  --sae-batch-size 4096 \
  --active-threshold 1e-6 \
  --top-fraction 0.05 \
  --feature-cache "${FEATURE_CACHE}" \
  --reuse-feature-cache \
  --case-level \
  --top-concepts "${TOP_CONCEPTS}" \
  --selection-folds "${SELECTION_FOLDS}" \
  --epochs "${EPOCHS}" \
  --early-stopping-patience 20 \
  --lr 1e-3 \
  --weight-decay 1e-3 \
  --ranking-loss-weight 0.25 \
  --ranking-margin 0.10 \
  --low-high-loss-weight "${LOW_HIGH_LOSS_WEIGHT}" \
  --val-monotonicity-penalty "${MONOTONICITY_PENALTY}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
