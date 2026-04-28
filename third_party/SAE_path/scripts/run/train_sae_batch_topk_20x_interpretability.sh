#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# =========================
# Runtime env
# =========================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# =========================
# Paths
# =========================
MANIFEST="${MANIFEST:-/research/projects/mllab/WSI/extracted_features/sae_manifests_tcga_patient_train_test_90_10.json}"
OUT_DIR="${OUT_DIR:-/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_20x_interp}"
SHAPES_CACHE_JSON="${SHAPES_CACHE_JSON:-/common/users/wq50/SAE_path/cache/sae_shapes_20x.json}"

# =========================
# W&B
# =========================
PROJECT="${PROJECT:-SAE_pathology}"
RUN_NAME="${RUN_NAME:-tcga_sae_batch_topk_20x_interp}"
TAGS="${TAGS:-tcga,uni2,sae,batch_topk,20x,interpretability}"
MODE="${MODE:-online}"   # online/offline/disabled

# =========================
# Model (interpretability-oriented defaults)
# =========================
STAGE="batch_topk"
MAGNIFICATION="20x"
D_IN="${D_IN:-1536}"
LATENT_DIM="${LATENT_DIM:-12288}"      # 8x expansion; bump to 16384/24576 in sweeps if resources allow
TOPK_K="${TOPK_K:-64}"                 # lower k usually improves interpretability/monosemanticity
TOPK_MODE="${TOPK_MODE:-value}"
TOPK_NONNEG="${TOPK_NONNEG:---topk_nonneg}"
TOPK_LR="${TOPK_LR:-}"                 # optional override for topk/batch_topk stage lr
TIED="${TIED:-}"                       # leave empty (untied) for better capacity
NO_INPUT_LN="${NO_INPUT_LN:-}"         # leave empty to keep input layernorm ON

# =========================
# Data / loader
# =========================
TILES_PER_SLIDE="${TILES_PER_SLIDE:-2048}"
SLIDE_BATCH_TILES="${SLIDE_BATCH_TILES:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"          # slide-chunks per step (flattened tiles are the real batch for batch_topk)
NUM_WORKERS="${NUM_WORKERS:-4}"

# =========================
# Optimization / logging
# =========================
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_STEPS="${MAX_STEPS:-100000}"
LOG_EVERY="${LOG_EVERY:-200}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"     # cheap eval by default; increase later for final validation
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
EVAL_SLIDE_BATCH_TILES="${EVAL_SLIDE_BATCH_TILES:-1024}"
PRINT_EVERY="${PRINT_EVERY:-100}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
AMP="${AMP:---amp}"

# =========================
# Run controls
# =========================
DO_PREFLIGHT="${DO_PREFLIGHT:-1}"
DO_TRAIN="${DO_TRAIN:-1}"

echo "[train_sae_batch_topk_20x_interpretability] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[train_sae_batch_topk_20x_interpretability] MANIFEST=${MANIFEST}"
echo "[train_sae_batch_topk_20x_interpretability] OUT_DIR=${OUT_DIR}"
echo "[train_sae_batch_topk_20x_interpretability] STAGE=${STAGE} MAGNIFICATION=${MAGNIFICATION} TOPK_K=${TOPK_K}"

if [[ "${DO_PREFLIGHT}" == "1" ]]; then
  echo "[train_sae_batch_topk_20x_interpretability] Running preflight..."
  python -m scripts.train_sae \
    --manifest "${MANIFEST}" \
    --out_dir "${OUT_DIR}" \
    --magnification "${MAGNIFICATION}" \
    --tiles_per_slide "${TILES_PER_SLIDE}" \
    --slide_batch_tiles "${SLIDE_BATCH_TILES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --shapes_cache_json "${SHAPES_CACHE_JSON}" \
    --preflight_only
fi

if [[ "${DO_TRAIN}" != "1" ]]; then
  echo "[train_sae_batch_topk_20x_interpretability] DO_TRAIN=${DO_TRAIN}; stopping after preflight."
  exit 0
fi

echo "[train_sae_batch_topk_20x_interpretability] Starting training..."

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
  --shapes_cache_json "${SHAPES_CACHE_JSON}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_steps "${MAX_STEPS}"
  --log_every "${LOG_EVERY}"
  --eval_every "${EVAL_EVERY}"
  --eval_batches "${EVAL_BATCHES}"
  --print_every "${PRINT_EVERY}"
  --grad_clip "${GRAD_CLIP}"
  --batch_topk_k "${TOPK_K}"
  --topk_mode "${TOPK_MODE}"
)

if [[ -n "${TIED}" ]]; then
  cmd+=("${TIED}")
fi
if [[ -n "${NO_INPUT_LN}" ]]; then
  cmd+=("${NO_INPUT_LN}")
fi
if [[ -n "${AMP}" ]]; then
  cmd+=("${AMP}")
fi
if [[ -n "${TOPK_NONNEG}" ]]; then
  cmd+=("${TOPK_NONNEG}")
fi
if [[ -n "${TOPK_LR}" ]]; then
  cmd+=(--topk_lr "${TOPK_LR}")
fi

"${cmd[@]}"

