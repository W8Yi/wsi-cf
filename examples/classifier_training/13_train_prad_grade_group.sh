#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-30}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-4096}"
OUT_DIR="${OUT_DIR:-artifacts/classifier_training}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-artifacts/prad_gleason_inputs/splits/prad_grade_group_patient_stratified_80_20.json}"

echo "[labels] preparing TCGA-PRAD Gleason grade-group labels" >&2
"${PY}" scripts/prepare_prad_gleason_concept_tasks.py \
  --out-dir artifacts/prad_gleason_inputs

echo "[split] building PRAD grade-stratified patient split: ${SPLIT_MANIFEST}" >&2
"${PY}" scripts/build_prad_grade_group_split.py \
  --slide-labels artifacts/prad_gleason_inputs/slide_labels.csv \
  --out "${SPLIT_MANIFEST}" \
  --grade-column grade_group

echo "[train] TCGA-PRAD 5-way grade-group classifier: GG1/GG2/GG3/GG4/GG5" >&2
"${PY}" scripts/train_attention_classifier.py \
  --task-name prad_grade_group \
  --label-source artifacts/prad_gleason_inputs/slide_labels.csv \
  --projects TCGA-PRAD \
  --label-column grade_group \
  --include-labels GG1,GG2,GG3,GG4,GG5 \
  --label-order GG1,GG2,GG3,GG4,GG5 \
  --split-manifest "${SPLIT_MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
  --device "${DEVICE}" \
  "$@"
