#!/usr/bin/env bash
set -euo pipefail

python scripts/visualize_attention.py \
  --task hnscc_hpv \
  --backend clam \
  --max-slides 1 \
  --thumbnail-max-side 4096 \
  --out-dir artifacts/hnscc_hpv_attention_vis
