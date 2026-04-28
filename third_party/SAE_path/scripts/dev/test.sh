#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=3
WANDB_RUN_NAME="relu_sae_tcga_hnscc"

python -m scripts.latent_mining \
  --mode both \
  --run_name "$WANDB_RUN_NAME" \
  --index_json /common/users/wq50/CLAM/features/HPV_UNI2_features/manifest_index.json \
  --slides_per_project 10 --require_h5_exists \
  --ckpt /common/users/wq50/SAE_path/runs/relu_sae_tcga_hnscc/relu_ckpt_best.pt --stage relu \
  --latent_dim 12288 \
  --topk_nonneg \
  --tiles_per_slide 5096 --chunk_tiles 512 \
  --select_strategy top_activation --n_latents 32 --topn 50 \

python -m scripts.export_latent_tiles \
  --out_root /common/users/wq50/SAE_path/sae_mining \
  --run_name "$WANDB_RUN_NAME" \
  --gdc_client /common/users/wq50/SAE_path/gdc/gdc-client \
  --tile_size 256 --level 0 --vis_size 256 --ncols 10  
