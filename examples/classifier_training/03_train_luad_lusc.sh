#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

/common/users/wq50/envs/pace/bin/python scripts/train_attention_classifier.py \
  --task-name luad_lusc \
  --projects TCGA-LUAD,TCGA-LUSC \
  --label-column project_dir \
  --label-map TCGA-LUAD:LUAD,TCGA-LUSC:LUSC \
  --include-labels LUAD,LUSC \
  --out-dir artifacts/classifier_training \
  --epochs 20 \
  --max-tiles-per-slide 2048 \
  --device cuda:0 \
  "$@"
