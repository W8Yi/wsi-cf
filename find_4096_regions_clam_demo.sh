#!/usr/bin/env bash
set -euo pipefail

PACE_PY=/common/users/wq50/envs/pace/bin/python
WSI_CF=/common/users/wq50/wsi_cf

${PACE_PY} ${WSI_CF}/scripts/find_pathology_aware_2048_regions.py \
  --model-backend clam \
  --slides-dir /common/users/wq50/CLAM/HNSCC_slides \
  --out-dir ${WSI_CF}/artifacts/pathology_aware_4096_regions_clam_demo_20x \
  --clam-ckpt /common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/s_0_checkpoint.pt \
  --clam-dataset-csv /common/users/wq50/CLAM/dataset_csv/HNSCC.csv \
  --clam-splits-csv /common/users/wq50/CLAM/results/HPV_CLAM_50_mb_s1/splits_0.csv \
  --clam-features-pt-dir /common/users/wq50/CLAM/features/HPV_UNI2_features/pt_files \
  --clam-coords-h5-dir /common/users/wq50/CLAM/HNSCC_cases/patches \
  --clam-split test \
  --clam-attn-class pred \
  --target-magnification 20 \
  --region-size 4096 \
  --grid-step-px 256 \
  --window-stride-cells 8 \
  --image-qc-topk-per-slide 24 \
  --max-candidates-per-slide 12 \
  --final-regions-per-label 6 \
  --attention-percentile 90 \
  --sae-percentile 90 \
  --combined-percentile 90 \
  --sae-attention-gate-percentile 75 \
  --attention-weight 0.65 \
  --sae-weight 0.35 \
  --min-tissue 0.60 \
  --min-dark-fraction 0.10 \
  --min-saturation-fraction 0.10 \
  --min-valid-feature-fraction 0.80 \
  --min-high-importance-cells 12 \
  --max-high-importance-cells 96 \
  --max-high-importance-fraction 0.45 \
  --min-largest-high-component 10 \
  --min-central-high-importance-fraction 0.20 \
  --central-fraction 0.60 \
  --min-selected-cells 6 \
  --max-selected-cells 24 \
  --target-importance-mass 0.45 \
  --clam-use-target-mag-equivalent \
  --expand-selected-by-sae-neighbors \
  --neighbor-similarity-threshold 0.90 \
  --neighbor-min-combined-importance 0.08 \
  --max-expanded-cells 48 \
  --require-label-match \
  --min-label-confidence 0.70 \
  --cache-slide-scores \
  --device cuda:0
