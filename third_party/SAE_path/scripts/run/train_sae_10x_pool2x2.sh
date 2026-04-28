#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# =========================
# Runtime env
# =========================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# =========================
# Paths
# =========================
MANIFEST="${MANIFEST:-/common/users/wq50/UNI2_features/extracted_features/sae_manifests_tcga_patient_train_test_90_10.json}"
OUT_DIR="${OUT_DIR:-/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_10x_pool2x2}"
POOL_MAP_CACHE_DIR="${POOL_MAP_CACHE_DIR:-/common/users/wq50/SAE_path/cache/pool_maps}"
SHAPES_CACHE_JSON="${SHAPES_CACHE_JSON:-/common/users/wq50/SAE_path/cache/sae_shapes_10x_pool2x2.json}"

# =========================
# W&B
# =========================
PROJECT="${PROJECT:-SAE_pathology}"
RUN_NAME="${RUN_NAME:-tcga_sae_batch_topk_10x_pool2x2}"
TAGS="${TAGS:-tcga,uni2,sae,batch_topk,10x_pool2x2}"
MODE="${MODE:-online}"   # online/offline/disabled

# =========================
# Stage / model
# =========================
STAGE="${STAGE:-batch_topk}"   # relu|topk|batch_topk|sdf2
D_IN="${D_IN:-1536}"
LATENT_DIM="${LATENT_DIM:-12288}"  # 8 * 1536
TIED="${TIED:-}"                  # set to --tied
NO_INPUT_LN="${NO_INPUT_LN:-}"    # set to --no_input_layernorm

# TopK defaults (used if STAGE=topk or batch_topk)
TOPK_K="${TOPK_K:-512}"
TOPK_MODE="${TOPK_MODE:-value}"
TOPK_NONNEG="${TOPK_NONNEG:---topk_nonneg}"   # set to "" to disable
TOPK_LR="${TOPK_LR:-}"                        # optional override; leave empty to use LR

# ReLU defaults (used if STAGE=relu/sdf2)
L1_LAMBDA="${L1_LAMBDA:-1e-4}"

# =========================
# Data / loader
# =========================
MAGNIFICATION="10x_pool2x2"
TILES_PER_SLIDE="${TILES_PER_SLIDE:-2048}"
SLIDE_BATCH_TILES="${SLIDE_BATCH_TILES:-2048}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
POOL_ALLOW_PARTIAL="${POOL_ALLOW_PARTIAL:-}"   # set to --pool_allow_partial only if you want padded boundary groups

# =========================
# Optimization / logging
# =========================
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_STEPS="${MAX_STEPS:-50000}"
LOG_EVERY="${LOG_EVERY:-200}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
EVAL_BATCHES="${EVAL_BATCHES:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
EVAL_SLIDE_BATCH_TILES="${EVAL_SLIDE_BATCH_TILES:-1024}"
PRINT_EVERY="${PRINT_EVERY:-100}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
AMP="${AMP:---amp}"   # set to "" to disable

# =========================
# Run controls
# =========================
DO_PREFLIGHT="${DO_PREFLIGHT:-1}"    # 1=run preflight first
DO_TRAIN="${DO_TRAIN:-1}"            # 1=run training after preflight

echo "[train_sae_10x_pool2x2] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[train_sae_10x_pool2x2] MANIFEST=${MANIFEST}"
echo "[train_sae_10x_pool2x2] OUT_DIR=${OUT_DIR}"
echo "[train_sae_10x_pool2x2] STAGE=${STAGE} MAGNIFICATION=${MAGNIFICATION}"

if [[ "${DO_PREFLIGHT}" == "1" ]]; then
  echo "[train_sae_10x_pool2x2] Running preflight..."
  python -m scripts.train_sae \
    --manifest "${MANIFEST}" \
    --out_dir "${OUT_DIR}" \
    --magnification "${MAGNIFICATION}" \
    --tiles_per_slide "${TILES_PER_SLIDE}" \
    --slide_batch_tiles "${SLIDE_BATCH_TILES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --pool_map_cache_dir "${POOL_MAP_CACHE_DIR}" \
    --shapes_cache_json "${SHAPES_CACHE_JSON}" \
    ${POOL_ALLOW_PARTIAL} \
    --preflight_only
fi

if [[ "${DO_TRAIN}" != "1" ]]; then
  echo "[train_sae_10x_pool2x2] DO_TRAIN=${DO_TRAIN}; stopping after preflight."
  exit 0
fi

echo "[train_sae_10x_pool2x2] Starting training..."

cmd=(
  python -m scripts.train_sae
  --manifest "${MANIFEST}"
  --out_dir "${OUT_DIR}"
  --project "${PROJECT}"
  --run_name "${RUN_NAME}"
  --tags "${TAGS}"
  --mode "${MODE}"
  --stage "${STAGE}"
  --d_in "${D_IN}"
  --latent_dim "${LATENT_DIM}"
  --magnification "${MAGNIFICATION}"
  --tiles_per_slide "${TILES_PER_SLIDE}"
  --slide_batch_tiles "${SLIDE_BATCH_TILES}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --eval_num_workers "${EVAL_NUM_WORKERS}"
  --eval_slide_batch_tiles "${EVAL_SLIDE_BATCH_TILES}"
  --pool_map_cache_dir "${POOL_MAP_CACHE_DIR}"
  --shapes_cache_json "${SHAPES_CACHE_JSON}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_steps "${MAX_STEPS}"
  --log_every "${LOG_EVERY}"
  --eval_every "${EVAL_EVERY}"
  --eval_batches "${EVAL_BATCHES}"
  --print_every "${PRINT_EVERY}"
  --grad_clip "${GRAD_CLIP}"
)

if [[ -n "${TIED}" ]]; then
  cmd+=("${TIED}")
fi
if [[ -n "${NO_INPUT_LN}" ]]; then
  cmd+=("${NO_INPUT_LN}")
fi
if [[ -n "${POOL_ALLOW_PARTIAL}" ]]; then
  cmd+=("${POOL_ALLOW_PARTIAL}")
fi
if [[ -n "${AMP}" ]]; then
  cmd+=("${AMP}")
fi

case "${STAGE}" in
  topk)
    cmd+=(
      --topk_k "${TOPK_K}"
      --topk_mode "${TOPK_MODE}"
    )
    if [[ -n "${TOPK_NONNEG}" ]]; then
      cmd+=("${TOPK_NONNEG}")
    fi
    if [[ -n "${TOPK_LR}" ]]; then
      cmd+=(--topk_lr "${TOPK_LR}")
    fi
    ;;
  batch_topk)
    cmd+=(
      --batch_topk_k "${TOPK_K}"
      --topk_mode "${TOPK_MODE}"
    )
    if [[ -n "${TOPK_NONNEG}" ]]; then
      cmd+=("${TOPK_NONNEG}")
    fi
    if [[ -n "${TOPK_LR}" ]]; then
      cmd+=(--topk_lr "${TOPK_LR}")
    fi
    ;;
  relu)
    cmd+=(--l1_lambda "${L1_LAMBDA}")
    ;;
  sdf2)
    cmd+=(--l1_lambda "${L1_LAMBDA}")
    if [[ -n "${TOPK_LR}" ]]; then
      cmd+=(--topk_lr "${TOPK_LR}")
    fi
    ;;
  *)
    echo "Unsupported STAGE=${STAGE}" >&2
    exit 1
    ;;
esac

"${cmd[@]}"
