#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EDIT_ROOT="${EDIT_ROOT:-paper_outputs/prad_gleason_grade_to_grade_steering}"
CLASSIFIER_RUN_DIR="${CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_grade_group}"
OUT_DIR="${OUT_DIR:-${EDIT_ROOT}/classifier_shift_eval}"
MAX_RUNS_PER_DIRECTION="${MAX_RUNS_PER_DIRECTION:-1000000}"

"${PY}" scripts/evaluate_prad_classifier_edits.py \
  --edit-root "${EDIT_ROOT}" \
  --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
  --out-dir "${OUT_DIR}" \
  --directions auto \
  --max-runs-per-direction "${MAX_RUNS_PER_DIRECTION}" \
  --device "${DEVICE}" \
  "$@"

echo "[ok] PRAD grade-to-grade classifier-shift eval: ${OUT_DIR}" >&2
