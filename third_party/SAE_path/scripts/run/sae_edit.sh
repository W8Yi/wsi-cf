#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# TILE_DIR="/common/users/wq50/SAE_path/cf_imgs/roi3_neg1_tiles"
# SAE_CKPT="/common/users/wq50/SAE_path/runs/tcga_topk64_single/topk_ckpt_best.pt"
# SAE_CFG="/common/users/wq50/SAE_path/runs/tcga_topk64_single/run_config.json"
# OUT_ROOT="/common/users/wq50/SAE_path/outputs/sae_steer_test_topk64_latent607"
# DEVICE="cuda:0"
# LATENT_IDX=607
# DELTA=1.0

# BLEND=0.3
# LATENT_STRENGTH=0.4
# MAX_FEATURE_DELTA_NORM=400

# mkdir -p "${OUT_ROOT}"

# python -m scripts.sae_steer \
#   --image-dir "${TILE_DIR}" \
#   --out-dir "${OUT_ROOT}" \
#   --sae-ckpt "${SAE_CKPT}" \
#   --sae-cfg "${SAE_CFG}" \
#   --device "${DEVICE}" \
#   --tile-px 256 \
#   --grid-step-px 256 \
#   --reference-start-ratio 0.0 \
#   --reference-mix 0.0 \
#   --steer-mode latent_delta \
#   --latent-idx "${LATENT_IDX}" \
#   --delta "${DELTA}" \
#   --blend "${BLEND}" \
#   --latent-strength "${LATENT_STRENGTH}" \
#   --max-feature-delta-norm "${MAX_FEATURE_DELTA_NORM}"

TILE_DIR="/common/users/wq50/SAE_path/cf_imgs/roi3_neg1_tiles"
SAE_CKPT="/common/users/wq50/SAE_path/runs/relu_sae_tcga_ld12288_v1/ckpt_best.pt"
SAE_CFG="/common/users/wq50/SAE_path/runs/relu_sae_tcga_ld12288_v1/run_config.json"
OUT_ROOT="/common/users/wq50/SAE_path/outputs/sae_steer_test_relu_latent607"
DEVICE="cuda:0"
LATENT_IDX=607
DELTA=1.0

python -m scripts.sae_steer \
  --image-dir "${TILE_DIR}" \
  --out-dir "${OUT_ROOT}" \
  --sae-ckpt "${SAE_CKPT}" \
  --sae-cfg "${SAE_CFG}" \
  --device "${DEVICE}" \
  --tile-px 256 \
  --grid-step-px 256 \
  --reference-start-ratio 0.0 \
  --reference-mix 0.0 \
  --steer-mode latent_delta \
  --latent-idx "${LATENT_IDX}" \
  --delta "${DELTA}" \
  --blend 0.2 \
  --max-feature-delta-norm 200.0
