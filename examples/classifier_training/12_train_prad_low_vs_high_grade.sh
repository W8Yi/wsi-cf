#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-20}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-4096}"
OUT_DIR="${OUT_DIR:-artifacts/classifier_training}"

echo "[labels] preparing TCGA-PRAD Gleason low/high labels" >&2
"${PY}" scripts/prepare_prad_gleason_concept_tasks.py \
  --out-dir artifacts/prad_gleason_inputs

echo "[train] TCGA-PRAD low(GG1-GG2) vs high(GG3-GG5) classifier" >&2
"${PY}" scripts/train_attention_classifier.py \
  --task-name prad_low_vs_high_grade \
  --label-source artifacts/prad_gleason_inputs/slide_labels.csv \
  --projects TCGA-PRAD \
  --label-column low_high_grade \
  --include-labels low,high \
  --label-order low,high \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
  --device "${DEVICE}" \
  "$@"
