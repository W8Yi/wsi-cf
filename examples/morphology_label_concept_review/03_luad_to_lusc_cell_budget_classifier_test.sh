#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SOURCE_ROOT="${SOURCE_ROOT:-artifacts/morphology_label_concept_review_top1_showcase_best_10slides}"
OUT_DIR="${OUT_DIR:-artifacts/luad_to_lusc_cell_budget_showcase_best_10slides}"
CELL_COUNTS="${CELL_COUNTS:-1,4,16,32,64}"
MAX_RUNS="${MAX_RUNS:-10}"
RUN_EDITS="${RUN_EDITS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUTPUT_MODE="${OUTPUT_MODE:-minimal}"

ARGS=()
if [[ "${RUN_EDITS}" == "1" ]]; then
  ARGS+=(--run-edits)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

"${PY}" scripts/evaluate_luad_lusc_cell_budget_edits.py \
  --source-root "${SOURCE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --cell-counts "${CELL_COUNTS}" \
  --max-runs "${MAX_RUNS}" \
  --output-mode "${OUTPUT_MODE}" \
  --device "${DEVICE}" \
  "${ARGS[@]}" \
  "$@"

echo "[ok] LUAD-to-LUSC cell-budget classifier test outputs: ${OUT_DIR}" >&2
