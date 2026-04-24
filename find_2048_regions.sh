#!/usr/bin/env bash
set -euo pipefail

PACE_PY=/common/users/wq50/envs/pace/bin/python
WSI_CF=/common/users/wq50/wsi_cf

${PACE_PY} ${WSI_CF}/scripts/find_pathology_aware_2048_regions.py \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --split test \
  --features-root /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2 \
  --slides-dir /common/users/wq50/HNSCC/test \
  --out-dir ${WSI_CF}/artifacts/pathology_aware_2048_regions \
  --mil-ckpt /common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt \
  --sae-ckpt /common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt \
  --sae-cfg /common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json \
  --prototype-npz ${WSI_CF}/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz \
  --prototype-key prototype_median \
  --pos-latent 2645 \
  --neg-latent 7036 \
  --target-magnification 20 \
  --region-size 2048 \
  --grid-step-px 256 \
  --window-stride-cells 4 \
  --max-slides-per-label 0 \
  --max-candidates-per-slide 24 \
  --image-qc-topk-per-slide 32 \
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
  --require-label-match \
  --min-label-confidence 0.65 \
  --cache-slide-scores \
  --device cuda:0
