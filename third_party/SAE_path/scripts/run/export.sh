#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# python -m scripts.export_latent_tiles \
#   --out_root /common/users/wq50/SAE_path/sae_mining \
#   --run_name relu_sae_tcga_hnscc \
#   --pass2_json /common/users/wq50/SAE_path/sae_mining/relu_sae_tcga_hnscc/pass2_top_tiles_manual_n30_top64_20260205-225915.json \
#   --gdc_client /common/users/wq50/SAE_path/gdc/gdc-client \

python -m scripts.export_latent_tiles \
  --out_root /common/users/wq50/SAE_path/sae_mining \
  --run_name relu_sae_sdf2 \
  --pass2_json /common/users/wq50/SAE_path/sae_mining/relu_sae_sdf2/pass2_top_tiles_top_activation_n64_top50_20260212-162321.json \
  --gdc_client /common/users/wq50/SAE_path/gdc/gdc-client
