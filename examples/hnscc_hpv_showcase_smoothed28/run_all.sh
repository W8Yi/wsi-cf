#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/01_run_progressive_edit.sh"
"${SCRIPT_DIR}/02_run_naive_baseline.sh"
"${SCRIPT_DIR}/03_compute_metrics.sh"
