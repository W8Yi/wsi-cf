#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=4

MANIFEST="/common/users/wq50/UNI2_features/extracted_features/sae_manifests_tcga_patient_train_test_90_10.json"
OUT_DIR="/common/users/wq50/SAE_path/runs/relu_sae_base"

WANDB_PROJECT="SAE_pathology"
WANDB_RUN_NAME="relu_sae_base"
WANDB_TAGS="tcga,uni2,relu_sae,ld12288,pan_cancer"

# python -m scripts.train_sae \
#   --manifest $MANIFEST --out_dir $OUT_DIR \
#   --project "$WANDB_PROJECT" \
#   --run_name "$WANDB_RUN_NAME" \
#   --tags "$WANDB_TAGS" \
#   --stage relu \
#   --tiles_per_slide 4096 \
#   --slide_batch_tiles 4096 \
#   --no_input_layernorm \
  
# python -m scripts.latent_mining \
#   --run_name relu_sae_base \
#   --mode both \
#   --ckpt /common/users/wq50/SAE_path/runs/relu_sae_base/relu_ckpt_last.pt \
#   --stage relu \
#   --index_json metadata/indexes/manifest_index.json \
#   --slides_per_project 200 \
#   --require_h5_exists \
#   --tiles_per_slide 2048 \
#   --chunk_tiles 512 \
#   --select_strategy top_activation \
#   --n_latents 100 \
#   --topn 50

python -m scripts.export_latent_tiles \
  --out_root /common/users/wq50/SAE_path/sae_mining \
  --run_name relu_sae_base \
  --pass2_json /common/users/wq50/SAE_path/sae_mining/relu_sae_base/pass2_top_tiles_top_activation_n100_top50_20260221-052909.json \
  --gdc_client /common/users/wq50/SAE_path/gdc/gdc-client
