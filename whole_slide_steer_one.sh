#!/usr/bin/env bash
set -euo pipefail

PACE_PY=/common/users/wq50/envs/pace/bin/python
WSI_CF=/common/users/wq50/wsi_cf

${PACE_PY} ${WSI_CF}/scripts/run_whole_slide_attention_steer.py \
  --slide-key TCGA-BB-4225-01Z-00-DX1 \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --features-root /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2 \
  --slides-dir /common/users/wq50/HNSCC/test \
  --out-dir ${WSI_CF}/artifacts/whole_slide_attention_steer_bb4225_p95_allow_missing \
  --mil-ckpt /common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt \
  --attention-percentile 95 \
  --max-edit-tiles 0 \
  --max-windows 0 \
  --direction-mode opposite_label \
  --target-magnification 20 \
  --window-size 1024 \
  --grid-step-px 256 \
  --no-require-full-window-features \
  --device cuda:0 \
  --dtype fp16 \
  --pix-model-id StonyBrook-CVLab/PixCell-1024 \
  --pix-pipeline-id StonyBrook-CVLab/PixCell-pipeline \
  --vae-model-id stabilityai/stable-diffusion-3-medium-diffusers \
  --vae-subfolder vae \
  --steps 30 \
  --guidance 2.0 \
  --patch-batch 256 \
  --prototype-strength 0.8 \
  --steer-blend 1.0 \
  --preserve-edit-strength 0.05 \
  --preserve-visited-strength 0.95 \
  --preserve-fresh-context-strength 0.35 \
  --mid-steer-start-ratio 0.5 \
  --mid-steer-end-ratio 1.0 \
  --mid-steer-alpha-start 0.5 \
  --mid-steer-alpha-end 1.0 \
  --mid-steer-alpha-schedule linear \
  --sae-ckpt /common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt \
  --sae-cfg /common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json \
  --prototype-npz ${WSI_CF}/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz \
  --prototype-key prototype_median \
  --pos-latent 2645 \
  --neg-latent 7036 \
  --local-vis-size 2048 \
  --max-local-vis-areas 0 \
  --save-debug-artifacts
