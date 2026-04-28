#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=4

# python -m scripts.latent_mining \
#   --mode both \
#   --run_name relu_sae_tcga_hnscc \
#   --index_json metadata/indexes/manifest_index.json \
#   --slides_per_project 10 --require_h5_exists \
#   --ckpt /common/users/wq50/SAE_path/runs/relu_sae_tcga_hnscc/relu_ckpt_best.pt --stage relu \
#   --latent_dim 12288 \
#   --tiles_per_slide 4096 --chunk_tiles 512 \
#   --select_strategy top_activation --n_latents 64 --topn 64 \

# python -m scripts.latent_mining \
#   --mode both \
#   --run_name tcga_topk64_single \
#   --index_json metadata/indexes/manifest_index.json \
#   --slides_per_project 10 --require_h5_exists \
#   --ckpt /common/users/wq50/SAE_path/runs/topk64_single/topk_ckpt_best.pt --stage topk \
#   --latent_dim 12288 \
#   --topk_nonneg \
#   --tiles_per_slide 4096 --chunk_tiles 512 \
#   --select_strategy top_activation --n_latents 64 --topn 64 \

# python -m scripts.latent_mining \
#   --mode pass2 \
#   --run_name relu_sae_tcga_hnscc \
#   --latent_indices 10228,8465,8986,938,8285,10256,4938,951,8210,1096,9493,794,4842,3148,6678,7237,7390,11496,6717,348,8442,3107,219,11113,1902,7855,2787,1116,567,1270  \
#   --select_strategy manual --n_latents 64 --topn 64 \

python -m scripts.latent_mining \
  --run_name relu_sae_sdf2 \
  --mode both \
  --ckpt /common/users/wq50/SAE_path/runs/relu_sae_sdf2/sdf2_ckpt_last.pt \
  --stage sdf2 \
  --index_json metadata/indexes/manifest_index.json \
  --slides_per_project 200 \
  --require_h5_exists \
  --tiles_per_slide 1024 \
  --chunk_tiles 512 \
  --select_strategy top_activation \
  --n_latents 64 \
  --topn 50
