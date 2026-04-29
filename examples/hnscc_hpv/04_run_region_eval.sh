#!/usr/bin/env bash
set -euo pipefail

python scripts/run_progressive_region_edit.py \
  --task hnscc_hpv \
  --region-bank-csv artifacts/hnscc_hpv_found_regions/region_bank.csv \
  --edit-manifest artifacts/hnscc_hpv_found_regions/progressive_edit_manifest.json \
  --direction hpv_neg \
  --output-mode minimal \
  --out-dir artifacts/hnscc_hpv_region_eval_progressive_edit
