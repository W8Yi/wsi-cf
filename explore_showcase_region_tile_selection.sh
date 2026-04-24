#!/usr/bin/env bash
set -euo pipefail

PACE_PY=/common/users/wq50/envs/pace/bin/python
WSI_CF=/common/users/wq50/wsi_cf

${PACE_PY} ${WSI_CF}/scripts/run_region_image_tile_selector_exploration.py \
  --region-image ${WSI_CF}/artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048/region_top_right_2048.png \
  --out-dir ${WSI_CF}/artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_exploration \
  --region-id TCGA-P3-A5QE_top_right_2048 \
  --label 1 \
  --grid-step-px 256 \
  --attention-percentiles 75,80,85,90,95 \
  --sae-percentile 90 \
  --combined-percentile 90 \
  --sae-attention-gate-percentile 75 \
  --attention-weight 0.6 \
  --sae-weight 0.4 \
  --target-importance-mass 0.45 \
  --min-selected-cells 6 \
  --max-selected-cells 24 \
  --default-neighbor-similarity-space feature \
  --neighbor-similarity-threshold 0.90 \
  --neighbor-min-combined-importance 0.08 \
  --max-expanded-cells 48 \
  --connected-support-min-component 2 \
  --connected-support-min-touching-neighbors 1 \
  --gap-fill-min-neighbors 2 \
  --gap-fill-max-iters 4 \
  --expansion-methods seed_only,feature_neighbors,sae_neighbors,connected_support,feature_plus_connected,sae_plus_connected,full_stack_feature,full_stack_sae \
  --device cuda:0
