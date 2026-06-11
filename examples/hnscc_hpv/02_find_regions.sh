#!/usr/bin/env bash
set -euo pipefail

python scripts/find_regions.py \
  --mode attention \
  --backend clam \
  --clam-source-from-task-split \
  --slides-dir /common/users/wq50/CLAM/HNSCC_slides \
  --target-magnification 20 \
  --region-size 2048 \
  --final-regions-per-label 2 \
  --out-dir paper_example/hnscc_hpv_found_regions
