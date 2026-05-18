#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

/common/users/wq50/envs/pace/bin/python scripts/train_attention_classifier.py \
  --task-name kirc_grade \
  --projects TCGA-KIRC \
  --label-column tumor_grade \
  --include-labels G1,G2,G3,G4 \
  --out-dir artifacts/classifier_training \
  --epochs 20 \
  --max-tiles-per-slide 2048 \
  --device cuda:0 \
  "$@"
