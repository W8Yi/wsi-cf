#!/usr/bin/env bash
# run_relu_sparse_sae_tcga_ld12288.sh
#
# Train ReLU Sparse SAE on TCGA with latent_dim = 8 * 1536 = 12288
# Uses corrected epoch handling + AMP + wandb logging
#
# Make executable:
#   chmod +x run_relu_sparse_sae_tcga_ld12288.sh
#
# Run:
#   ./run_relu_sparse_sae_tcga_ld12288.sh

set -e
set -u
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=5
export HDF5_USE_FILE_LOCKING=FALSE

# =========================
# Paths
# =========================
MANIFEST="/common/users/wq50/SAE_path/metadata/manifests/sae_manifest_hpv100_split0_h5.json"
OUT_DIR="/common/users/wq50/SAE_path/runs/relu_sae_tcga_hnscc"

# =========================
# wandb
# =========================
WANDB_PROJECT="SAE_pathology"
WANDB_RUN_NAME="relu_sae_tcga_hnscc"
WANDB_TAGS="tcga,uni2,relu_sae,ld12288,pan_cancer"

# =========================
# Model
# =========================
D_IN=1536
LATENT_DIM=$((8 * 1536))   # 12288

# =========================
# Data / loader (slide fairness)
# =========================
TILES_PER_SLIDE=2048
SLIDE_BATCH_TILES=2048
BATCH_SIZE=16               # chunks per step (reduce if OOM)
NUM_WORKERS=8

# =========================
# Optimization
# =========================
LR=1e-3
WEIGHT_DECAY=0.0
L1_LAMBDA=1e-4
MAX_STEPS=50000
LOG_EVERY=200
EVAL_EVERY=2000
EVAL_BATCHES=200
GRAD_CLIP=1.0

# =========================
# Launch
# =========================
python -m scripts.train_sae \
  --manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  \
  --project "$WANDB_PROJECT" \
  --run_name "$WANDB_RUN_NAME" \
  --tags "$WANDB_TAGS" \
  --stage relu \
  \
  --d_in $D_IN \
  --latent_dim $LATENT_DIM \
  \
  --tiles_per_slide $TILES_PER_SLIDE \
  --slide_batch_tiles $SLIDE_BATCH_TILES \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  \
  --lr $LR \
  --weight_decay $WEIGHT_DECAY \
  --l1_lambda $L1_LAMBDA \
  --max_steps $MAX_STEPS \
  --log_every $LOG_EVERY \
  --eval_every $EVAL_EVERY \
  --eval_batches $EVAL_BATCHES \
  --grad_clip $GRAD_CLIP \
  \
  --log_every 10 \
  --amp

python -m scripts.latent_mining \
  --mode both \
  --run_name "$WANDB_RUN_NAME" \
  --index_json /common/users/wq50/CLAM/features/HPV_UNI2_features/manifest_index.json \
  --slides_per_project 10 --require_h5_exists \
  --ckpt /common/users/wq50/SAE_path/runs/"$WANDB_RUN_NAME"/topk_ckpt_best.pt --stage relu \
  --latent_dim 12288 \
  --topk_nonneg \
  --tiles_per_slide 2048 --chunk_tiles 512 \
  --select_strategy top_activation --n_latents 64 --topn 50 \

python -m scripts.export_latent_tiles \
  --out_root /common/users/wq50/SAE_path/sae_mining \
  --run_name "$WANDB_RUN_NAME" \
  --manifest /common/users/wq50/CLAM/features/HPV_UNI2_features/manifest_index.json \
  --gdc_client /common/users/wq50/SAE_path/gdc/gdc-client \
  --tile_size 256 --level 0 --vis_size 256 --ncols 10  
