#!/usr/bin/env bash
set -euo pipefail

# Keep the core workflow entrypoint aligned with the paper showcase steering
# configuration and its prototype/SAE provenance.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../hnscc_hpv_showcase_smoothed28/01_run_progressive_edit.sh" "$@"
