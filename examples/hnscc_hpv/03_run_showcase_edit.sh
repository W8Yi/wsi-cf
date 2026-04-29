#!/usr/bin/env bash
set -euo pipefail

python scripts/run_progressive_region_edit.py \
  --task hnscc_hpv \
  --direction hpv_neg \
  --output-mode debug \
  --out-dir artifacts/hnscc_hpv_showcase_progressive_edit
