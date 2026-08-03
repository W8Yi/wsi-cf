#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-30}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-4096}"
OUT_DIR="${OUT_DIR:-artifacts/classifier_training}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-artifacts/prad_gleason_inputs/splits/prad_morphology_group_patient_stratified_80_20.json}"

LABEL_ORDER="pattern_1_3_well_formed,pattern_4_cribriform_poorly_formed_fused,pattern_5_solid_single_necrosis"

echo "[labels] preparing TCGA-PRAD Gleason morphology-group labels" >&2
"${PY}" scripts/prepare_prad_gleason_concept_tasks.py \
  --out-dir artifacts/prad_gleason_inputs

echo "[split] building PRAD morphology-group patient split: ${SPLIT_MANIFEST}" >&2
"${PY}" scripts/build_prad_grade_group_split.py \
  --slide-labels artifacts/prad_gleason_inputs/slide_labels.csv \
  --out "${SPLIT_MANIFEST}" \
  --grade-column morphology_group \
  --min-test-per-grade 4

echo "[train] TCGA-PRAD 3-way morphology classifier" >&2
"${PY}" scripts/train_attention_classifier.py \
  --task-name prad_morphology_group \
  --label-source artifacts/prad_gleason_inputs/slide_labels.csv \
  --projects TCGA-PRAD \
  --label-column morphology_group \
  --include-labels "${LABEL_ORDER}" \
  --label-order "${LABEL_ORDER}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --epochs "${EPOCHS}" \
  --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
  --device "${DEVICE}" \
  "$@"
