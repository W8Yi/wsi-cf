#!/usr/bin/env bash
set -euo pipefail

PACE_PY=/common/users/wq50/envs/pace/bin/python
WSI_CF=/common/users/wq50/wsi_cf

${PACE_PY} ${WSI_CF}/scripts/find_pathology_aware_2048_regions.py \
  --model-backend clam \
  --slides-dir /common/users/wq50/CLAM/HNSCC_slides \
  --out-dir ${WSI_CF}/artifacts/pathology_aware_2048_regions_clam_curated_20x \
  --clam-ckpt /common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/s_0_checkpoint.pt \
  --clam-dataset-csv /common/users/wq50/CLAM/dataset_csv/HNSCC.csv \
  --clam-splits-csv /common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/splits_0.csv \
  --clam-features-pt-dir /common/users/wq50/CLAM/features/HPV_UNI2_features/pt_files \
  --clam-coords-h5-dir /common/users/wq50/CLAM/HNSCC_cases/patches \
  --clam-split test \
  --clam-attn-class pred \
  --target-magnification 20 \
  --region-size 2048 \
  --grid-step-px 256 \
  --window-stride-cells 4 \
  --image-qc-topk-per-slide 32 \
  --max-candidates-per-slide 24 \
  --final-regions-per-label 10 \
  --attention-percentile 90 \
  --sae-percentile 90 \
  --combined-percentile 90 \
  --sae-attention-gate-percentile 75 \
  --attention-weight 0.6 \
  --sae-weight 0.4 \
  --min-tissue 0.50 \
  --min-dark-fraction 0.08 \
  --min-saturation-fraction 0.08 \
  --min-valid-feature-fraction 0.75 \
  --min-high-importance-cells 2 \
  --max-high-importance-cells 32 \
  --max-high-importance-fraction 0.50 \
  --min-selected-cells 2 \
  --max-selected-cells 12 \
  --target-importance-mass 0.35 \
  --clam-use-target-mag-equivalent \
  --expand-selected-by-sae-neighbors \
  --neighbor-similarity-threshold 0.92 \
  --neighbor-min-combined-importance 0.10 \
  --max-expanded-cells 24 \
  --require-label-match \
  --min-label-confidence 0.65 \
  --cache-slide-scores \
  --device cuda:0
