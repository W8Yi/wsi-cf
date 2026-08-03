#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
LABEL_SOURCE="${LABEL_SOURCE:-artifacts/prad_gleason_inputs/slide_labels.csv}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-artifacts/prad_gleason_inputs/splits/prad_grade_group_patient_stratified_80_20.json}"
TEST_FRAC="${TEST_FRAC:-0.20}"
MIN_TEST_PER_GRADE="${MIN_TEST_PER_GRADE:-4}"
SEED="${SEED:-7}"

"${PY}" scripts/build_prad_grade_group_split.py \
  --slide-labels "${LABEL_SOURCE}" \
  --out "${SPLIT_MANIFEST}" \
  --grade-column grade_group \
  --test-frac "${TEST_FRAC}" \
  --min-test-per-grade "${MIN_TEST_PER_GRADE}" \
  --seed "${SEED}" \
  "$@"
