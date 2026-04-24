#!/usr/bin/env bash
set -euo pipefail

PACE_PY=/common/users/wq50/envs/pace/bin/python
WSI_CF=/common/users/wq50/wsi_cf

${PACE_PY} ${WSI_CF}/scripts/visualize_clam_attention_whole_slide.py \
  --splits-csv /common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/splits_0.csv \
  --dataset-csv /common/users/wq50/CLAM/dataset_csv/HNSCC.csv \
  --features-pt-dir /common/users/wq50/CLAM/features/HPV_UNI2_features/pt_files \
  --coords-h5-dir /common/users/wq50/CLAM/HNSCC_cases/patches \
  --slides-dir /common/users/wq50/CLAM/HNSCC_slides \
  --ckpt /common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/s_0_checkpoint.pt \
  --out-dir ${WSI_CF}/artifacts/clam_attention_vis_split0_test \
  --split test \
  --top-k 0 \
  --attention-percentile 90 \
  --thumbnail-max-side 4000 \
  --heatmap-blur-px 3 \
  --heatmap-gamma 0.6 \
  --heatmap-alpha 200 \
  --attn-class pred \
  --max-slides 0 \
  --device cuda:0
