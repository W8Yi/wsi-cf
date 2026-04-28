#!/usr/bin/env bash
# run_topk64_sae.sh
# Simple single-stage TopK SAE with k=64

set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=3
export HDF5_USE_FILE_LOCKING=FALSE

MANIFEST="/common/users/wq50/UNI2_features/extracted_features/sae_manifests_tcga_patient_train_test_90_10.json"
OUT_DIR="/common/users/wq50/SAE_path/runs/tcga_topk64_single"
PROJECT="SAE_pathology"
RUN_NAME="tcga_topk64_single"

D_IN=1536
LATENT_DIM=$((8 * 1536))

TILES_PER_SLIDE=2048
SLIDE_BATCH_TILES=2048
BATCH_SIZE=16
NUM_WORKERS=8

LR=3e-4
MAX_STEPS=50000

TOPK_K=64
TOPK_MODE="value"          # or "magnitude"
TOPK_NONNEG="--topk_nonneg" # keep as-is (recommended) or set to "" to disable

EVAL_EVERY=2000
LOG_EVERY=200
PRINT_EVERY=100
EVAL_BATCHES=200

AMP="--amp"                # or "" to disable
MODE="online"              # online/offline/disabled
TIED=""                    # set to "--tied" if you want tied weights
NO_LN=""                   # set to "--no_input_layernorm" if you want to disable input LN

python -m scripts.train_sae \
  --manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --project "$PROJECT" \
  --run_name "$RUN_NAME" \
  --mode "$MODE" \
  --stage topk \
  --d_in "$D_IN" \
  --latent_dim "$LATENT_DIM" \
  $TIED \
  $NO_LN \
  --tiles_per_slide "$TILES_PER_SLIDE" \
  --slide_batch_tiles "$SLIDE_BATCH_TILES" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --max_steps "$MAX_STEPS" \
  --topk_k "$TOPK_K" \
  --topk_mode "$TOPK_MODE" \
  $TOPK_NONNEG \
  --eval_every "$EVAL_EVERY" \
  --eval_batches "$EVAL_BATCHES" \
  --log_every "$LOG_EVERY" \
  --print_every "$PRINT_EVERY" \
  $AMP

python -m scripts.latent_mining \
  --mode both \
  --run_name tcga_topk64_single \
  --index_json metadata/indexes/manifest_index.json \
  --slides_per_project 10 --require_h5_exists \
  --ckpt /common/users/wq50/SAE_path/runs/tcga_topk64_single/topk_ckpt_best.pt --stage topk \
  --latent_dim 12288 \
  --topk_nonneg \
  --tiles_per_slide 2048 --chunk_tiles 512 \
  --select_strategy top_activation --n_latents 64 --topn 50 \

python -m scripts.export_latent_tiles \
  --out_root /common/users/wq50/SAE_path/sae_mining \
  --run_name tcga_topk64_single \
  --manifest /common/users/wq50/SAE_path/metadata/indexes/manifest_index.json \
  --gdc_client /common/users/wq50/SAE_path/gdc/gdc-client \
  --tile_size 256 --level 0 --vis_size 256 --ncols 10  
