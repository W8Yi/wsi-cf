#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

/common/users/wq50/envs/pace/bin/python scripts/train_attention_classifier.py \
  --task-name msi_coad_stad \
  --projects TCGA-COAD,TCGA-STAD \
  --label-column msi_status \
  --include-labels MSI,NonMSI \
  --out-dir artifacts/classifier_training \
  --epochs 20 \
  --max-tiles-per-slide 2048 \
  --device cuda:0 \
  "$@"
