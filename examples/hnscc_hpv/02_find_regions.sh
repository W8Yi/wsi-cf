#!/usr/bin/env bash
set -euo pipefail

python scripts/find_regions.py \
  --mode attention \
  --backend clam \
  --target-magnification 20 \
  --region-size 2048 \
  --final-regions-per-label 2 \
  --out-dir artifacts/hnscc_hpv_found_regions
