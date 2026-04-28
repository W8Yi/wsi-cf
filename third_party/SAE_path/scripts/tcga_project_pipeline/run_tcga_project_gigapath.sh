#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/common/users/wq50/envs/pace/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-8}"

exec "$PYTHON_BIN" \
  "$ROOT_DIR/scripts/tcga_project_pipeline/run_tcga_project_gigapath.py" \
  "$@"
