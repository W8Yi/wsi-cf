#!/usr/bin/env bash
set -euo pipefail

PACE_PY=/common/users/wq50/envs/pace/bin/python
WSI_CF=/common/users/wq50/wsi_cf

${PACE_PY} ${WSI_CF}/scripts/visualize_mil_attention_whole_slide.py \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --features-root /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2 \
  --slides-dir /common/users/wq50/HNSCC/test \
  --mil-ckpt /common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt \
  --out-dir ${WSI_CF}/artifacts/whole_slide_attention_vis_gdc_test \
  --split test \
  --max-slides-per-label 2 \
  --max-slides 4 \
  --top-k 40 \
  --attention-percentile 90 \
  --thumbnail-max-side 2048 \
  --heatmap-blur-px 4 \
  --heatmap-alpha 255 \
  --device cpu
