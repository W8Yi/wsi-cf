#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python latent.py \
  --h5_list tcga_sample_list.txt \
  --ckpt runs/relu_sae_tcga_ld12288_v1/ckpt_best.pt \
  --stage relu \
  --latent_dim 12288 \
  --out_dir results/tcga_latents \
  --select_strategy top_activation \
  --n_latents 32 \
  --topn 32
