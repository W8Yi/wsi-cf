#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-20}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-4096}"

"$PY" scripts/train_grade_risk_regressor.py \
  --task-name kirc_continuous_grade_risk \
  --projects TCGA-KIRC \
  --label-column tumor_grade \
  --target-map G1:0.0,G2:0.33,G3:0.66,G4:1.0 \
  --out-dir artifacts/grade_risk_training \
  --epochs "${EPOCHS}" \
  --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
  --device "${DEVICE}" \
  "$@"
