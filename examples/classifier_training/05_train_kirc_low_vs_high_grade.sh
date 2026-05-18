#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

/common/users/wq50/envs/pace/bin/python scripts/train_attention_classifier.py \
  --task-name kirc_low_vs_high_grade \
  --projects TCGA-KIRC \
  --label-column tumor_grade \
  --label-map G1:low,G2:low,G3:high,G4:high \
  --include-labels low,high \
  --out-dir artifacts/classifier_training \
  --epochs 8 \
  --max-tiles-per-slide 4096 \
  --device cuda:0 \
  "$@"
