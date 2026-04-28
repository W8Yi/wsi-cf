#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/common/users/wq50/envs/pace/bin/python}"

exec "$PYTHON_BIN" \
  "$ROOT_DIR/scripts/tcga_project_pipeline/run_tcga_project_uni2h.py" \
  "$@"
