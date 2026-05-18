#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

/common/users/wq50/envs/pace/bin/python scripts/train_attention_classifier.py \
  --task-name cancer_type_all_tcga \
  --projects all \
  --label-column project_dir \
  --min-slides-per-class 30 \
  --out-dir artifacts/classifier_training \
  --epochs 20 \
  --max-tiles-per-slide 2048 \
  --device cuda:0 \
  "$@"
