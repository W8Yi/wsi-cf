#!/usr/bin/env bash
set -euo pipefail

# Rebuild paper-ready tables and plots from prediction-transition benchmark metrics.
#
# Default behavior intentionally uses only complete task/direction runs, so an
# active streamed direction such as PRAD p4->p5 does not silently enter figures
# before all requested edits are scored.

ROOT_DIR="${ROOT_DIR:-/common/users/wq50/wsi_cf}"
PYTHON_BIN="${PYTHON_BIN:-/common/users/wq50/envs/pace/bin/python}"

METRICS_ROOT="${METRICS_ROOT:-paper_outputs/prediction_transition_benchmark_test_only_unbalanced/metrics}"
OUT_DIR="${OUT_DIR:-paper_outputs/prediction_transition_benchmark_test_only_unbalanced/plots/paper_complete}"
PREFIX="${PREFIX:-paper_complete_transition}"
TITLE="${TITLE:-Prediction transition benchmark}"
REPORT_BUDGET="${REPORT_BUDGET:-32}"
MAX_SHARED_BUDGET="${MAX_SHARED_BUDGET:-48}"
FORMATS="${FORMATS:-png,pdf,svg}"
COMPLETE_ONLY="${COMPLETE_ONLY:-1}"
EXCLUDE="${EXCLUDE:-}"

cd "${ROOT_DIR}"

args=(
  scripts/plot_prediction_transition_metrics.py
  --metrics-root "${METRICS_ROOT}"
  --out-dir "${OUT_DIR}"
  --prefix "${PREFIX}"
  --title "${TITLE}"
  --report-budget "${REPORT_BUDGET}"
  --max-shared-budget "${MAX_SHARED_BUDGET}"
  --formats "${FORMATS}"
)

if [[ "${COMPLETE_ONLY}" != "0" ]]; then
  args+=(--complete-only)
fi

for pattern in ${EXCLUDE}; do
  args+=(--exclude "${pattern}")
done

"${PYTHON_BIN}" "${args[@]}"
