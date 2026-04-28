#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Runtime
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export TCGA_FEATURES_BASE="${TCGA_FEATURES_BASE:-/research/projects/mllab/WSI/TCGA_features}"

# Paths
MANIFEST="${MANIFEST:-/common/users/wq50/SAE_path/metadata/manifests/sae_manifests_tcga_patient_train_test_90_10_tcga_features_existing.json}"
RUN_NAME="${RUN_NAME:-tcga_sae_advanced_v11}"
RUN_ROOT="${RUN_ROOT:-/common/users/wq50/SAE_path/runs/${RUN_NAME}}"
SHAPES_CACHE_JSON="${SHAPES_CACHE_JSON:-/common/users/wq50/SAE_path/cache/sae_shapes_20x_tcga_features_existing.json}"

STAGEA_OUT="${STAGEA_OUT:-${RUN_ROOT}/batch_topk}"
STAGEB_OUT="${STAGEB_OUT:-${RUN_ROOT}/sdf2}"

# Common tracking
PROJECT="${PROJECT:-SAE_pathology}"
MODE="${MODE:-online}"
COMMON_TAGS="${COMMON_TAGS:-tcga,uni2h,sae,advanced_v11,20x,interpretability}"

# Data/loader
TILES_PER_SLIDE="${TILES_PER_SLIDE:-8192}"
SLIDE_BATCH_TILES="${SLIDE_BATCH_TILES:-8192}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"

# Stage A (BatchTopK)
STAGEA_LATENT_DIM="${STAGEA_LATENT_DIM:-12288}"
STAGEA_K="${STAGEA_K:-64}"
STAGEA_LR="${STAGEA_LR:-3e-4}"
STAGEA_MAX_STEPS="${STAGEA_MAX_STEPS:-30000}"

# Stage B (SDF2)
STAGEB_LR="${STAGEB_LR:-1e-4}"
STAGEB_MAX_STEPS="${STAGEB_MAX_STEPS:-60000}"
SDF_N_LEVEL2="${SDF_N_LEVEL2:-256}"
SDF_ALPHA="${SDF_ALPHA:-1.0}"
SDF_LAMBDA_A="${SDF_LAMBDA_A:-1e-4}"
SDF_LAMBDA_U="${SDF_LAMBDA_U:-1e-4}"
SDF_PARENT_BALANCE_LAMBDA="${SDF_PARENT_BALANCE_LAMBDA:-0.05}"

# Controls
DO_PREFLIGHT="${DO_PREFLIGHT:-1}"
DO_STAGE_A="${DO_STAGE_A:-1}"
DO_STAGE_B="${DO_STAGE_B:-1}"

if [[ "${DO_STAGE_A}" == "1" ]]; then
  if [[ "${DO_PREFLIGHT}" == "1" ]]; then
    python -m scripts.train_sae \
      --manifest "${MANIFEST}" \
      --out_dir "${STAGEA_OUT}" \
      --magnification 20x \
      --tiles_per_slide "${TILES_PER_SLIDE}" \
      --slide_batch_tiles "${SLIDE_BATCH_TILES}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --shapes_cache_json "${SHAPES_CACHE_JSON}" \
      --preflight_only
  fi

  python -m scripts.train_sae \
    --manifest "${MANIFEST}" \
    --out_dir "${STAGEA_OUT}" \
    --project "${PROJECT}" \
    --run_name "${RUN_NAME}_batch_topk" \
    --tags "${COMMON_TAGS},batch_topk" \
    --mode "${MODE}" \
    --stage batch_topk \
    --d_in 1536 \
    --latent_dim "${STAGEA_LATENT_DIM}" \
    --magnification 20x \
    --tiles_per_slide "${TILES_PER_SLIDE}" \
    --slide_batch_tiles "${SLIDE_BATCH_TILES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --shapes_cache_json "${SHAPES_CACHE_JSON}" \
    --batch_topk_k "${STAGEA_K}" \
    --topk_mode value \
    --topk_nonneg \
    --lr "${STAGEA_LR}" \
    --max_steps "${STAGEA_MAX_STEPS}" \
    --grad_clip 1.0 \
    --log_every 200 \
    --eval_every 5000 \
    --eval_batches 10 \
    --print_every 100 \
    --amp
fi

if [[ "${DO_STAGE_B}" == "1" ]]; then
  INIT_CKPT="${INIT_CKPT:-${STAGEA_OUT}/batch_topk_final.pt}"
  if [[ ! -f "${INIT_CKPT}" ]]; then
    echo "[error] missing init checkpoint for SDF2: ${INIT_CKPT}" >&2
    exit 1
  fi

  python -m scripts.train_sae \
    --manifest "${MANIFEST}" \
    --out_dir "${STAGEB_OUT}" \
    --project "${PROJECT}" \
    --run_name "${RUN_NAME}_sdf2" \
    --tags "${COMMON_TAGS},sdf2,parent_balanced" \
    --mode "${MODE}" \
    --stage sdf2 \
    --d_in 1536 \
    --latent_dim "${STAGEA_LATENT_DIM}" \
    --magnification 20x \
    --tiles_per_slide "${TILES_PER_SLIDE}" \
    --slide_batch_tiles "${SLIDE_BATCH_TILES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --shapes_cache_json "${SHAPES_CACHE_JSON}" \
    --init_mode batch_topk_to_sdf2 \
    --init_from_ckpt "${INIT_CKPT}" \
    --sdf_n_level2 "${SDF_N_LEVEL2}" \
    --sdf_alpha "${SDF_ALPHA}" \
    --sdf_lambda_a "${SDF_LAMBDA_A}" \
    --sdf_lambda_u "${SDF_LAMBDA_U}" \
    --sdf_active_only \
    --sdf_coeff_simplex \
    --sdf_parent_balance_lambda "${SDF_PARENT_BALANCE_LAMBDA}" \
    --lr "${STAGEB_LR}" \
    --max_steps "${STAGEB_MAX_STEPS}" \
    --grad_clip 1.0 \
    --log_every 200 \
    --eval_every 5000 \
    --eval_batches 10 \
    --print_every 100 \
    --amp
fi

echo "[done] advanced SAE training outputs at ${RUN_ROOT}"
