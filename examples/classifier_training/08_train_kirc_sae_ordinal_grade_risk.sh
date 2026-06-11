#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-artifacts/sae_grade_risk_training}"
REUSE_FEATURE_CACHE="${REUSE_FEATURE_CACHE:-0}"

REUSE_ARGS=()
if [[ "${REUSE_FEATURE_CACHE}" == "1" ]]; then
  REUSE_ARGS+=(--reuse-feature-cache)
fi

"$PY" scripts/train_sae_concept_grade_risk.py \
  --task-name kirc_batch_topk_ordinal \
  --projects TCGA-KIRC \
  --label-column tumor_grade \
  --target-map G1:0.0,G2:0.33,G3:0.66,G4:1.0 \
  --sae-variant tcga_sae_batch_topk_20x_interp \
  --sae-batch-size 4096 \
  --active-threshold 1e-6 \
  --top-fraction 0.05 \
  --top-concepts 100 \
  --epochs 200 \
  --early-stopping-patience 20 \
  --lr 1e-3 \
  --weight-decay 1e-3 \
  --ranking-loss-weight 0.25 \
  --ranking-margin 0.10 \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  "${REUSE_ARGS[@]}" \
  "$@"
