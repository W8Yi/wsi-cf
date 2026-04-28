#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# =========================
# Inputs
# =========================
CKPT="${CKPT:-/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_10x_pool2x2/batch_topk_ckpt_best.pt}"
INDEX_JSON="${INDEX_JSON:-/research/projects/mllab/WSI/extracted_features/manifest_index.json}"
GDC_CLIENT="${GDC_CLIENT:-/common/users/wq50/SAE_path/gdc/gdc-client}"
TOKEN="${TOKEN:-}"   # optional path to GDC token file

# =========================
# Output run (latent mining)
# =========================
OUT_ROOT="${OUT_ROOT:-/common/users/wq50/SAE_path/sae_mining}"
RUN_NAME="${RUN_NAME:-tcga_sae_batch_topk_10x_pool2x2}"

# =========================
# Mining config (10x pooled features)
# =========================
MODE="${MODE:-both}"                    # pass1|pass2|both
STAGE="batch_topk"
MAGNIFICATION="10x_pool2x2"
POOL_ALLOW_PARTIAL="${POOL_ALLOW_PARTIAL:-}"   # set to --pool_allow_partial if desired

SLIDES_PER_PROJECT="${SLIDES_PER_PROJECT:-200}" # -1 for all
REQUIRE_H5_EXISTS="${REQUIRE_H5_EXISTS:---require_h5_exists}"
TILES_PER_SLIDE="${TILES_PER_SLIDE:-2048}"
CHUNK_TILES="${CHUNK_TILES:-512}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1337}"

# Model args for loading the SAE
D_IN="${D_IN:-1536}"
LATENT_DIM="${LATENT_DIM:-12288}"
TOPK_NONNEG="${TOPK_NONNEG:---topk_nonneg}"   # batch_topk model was trained with nonneg in your setup

# Pass2 selection
SELECT_STRATEGY="${SELECT_STRATEGY:-top_activation}"
N_LATENTS="${N_LATENTS:-100}"
TOPN="${TOPN:-70}"
LATENT_INDICES="${LATENT_INDICES:-}"          # only used if SELECT_STRATEGY=manual
MINE_MAX_TILES_PER_SLIDE_PER_LATENT="${MINE_MAX_TILES_PER_SLIDE_PER_LATENT:-3}"
MINE_MIN_DISTANCE_PX_SAME_SLIDE_PER_LATENT="${MINE_MIN_DISTANCE_PX_SAME_SLIDE_PER_LATENT:-512}"
MINE_TOPN_BUFFER_FACTOR="${MINE_TOPN_BUFFER_FACTOR:-4.0}"

# Export config
TILE_SIZE="${TILE_SIZE:-256}"
VIS_SIZE="${VIS_SIZE:-256}"
NCOLS="${NCOLS:-10}"
MAX_LATENTS="${MAX_LATENTS:--1}"
WSI_CACHE="${WSI_CACHE:-/common/users/wq50/SAE_path/wsi_cache}"
KEEP_WSI_CACHE="${KEEP_WSI_CACHE:-}"          # set to --keep_wsi_cache to retain downloads
OVERWRITE="${OVERWRITE:-}"                    # set to --overwrite to re-export tiles
FEATURE_MAG_OVERRIDE="${FEATURE_MAG_OVERRIDE:-}" # optional: 20x or 10x_pool2x2
EXPORT_MAX_TILES_PER_SLIDE_PER_LATENT="${EXPORT_MAX_TILES_PER_SLIDE_PER_LATENT:-3}"
CONTEXT_GRID="${CONTEXT_GRID:-1}"             # mainly for 20x; supported generically
DRAW_CENTER_BOX="${DRAW_CENTER_BOX:-}"        # set to --draw_center_box

# Controls
DO_MINE="${DO_MINE:-1}"
DO_EXPORT="${DO_EXPORT:-1}"

echo "[mine_export_10x_batch_topk] CKPT=${CKPT}"
echo "[mine_export_10x_batch_topk] INDEX_JSON=${INDEX_JSON}"
echo "[mine_export_10x_batch_topk] OUT_ROOT=${OUT_ROOT}"
echo "[mine_export_10x_batch_topk] RUN_NAME=${RUN_NAME}"

if [[ "${DO_MINE}" == "1" ]]; then
  echo "[mine_export_10x_batch_topk] Running latent mining (${MODE})..."
  cmd_mine=(
    python -m scripts.latent_mining
    --out_root "${OUT_ROOT}"
    --run_name "${RUN_NAME}"
    --mode "${MODE}"
    --ckpt "${CKPT}"
    --stage "${STAGE}"
    --index_json "${INDEX_JSON}"
    --magnification "${MAGNIFICATION}"
    --d_in "${D_IN}"
    --latent_dim "${LATENT_DIM}"
    --tiles_per_slide "${TILES_PER_SLIDE}"
    --chunk_tiles "${CHUNK_TILES}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --slides_per_project "${SLIDES_PER_PROJECT}"
    --select_strategy "${SELECT_STRATEGY}"
    --n_latents "${N_LATENTS}"
    --topn "${TOPN}"
    --topn_buffer_factor "${MINE_TOPN_BUFFER_FACTOR}"
    --max_tiles_per_slide_per_latent "${MINE_MAX_TILES_PER_SLIDE_PER_LATENT}"
    --min_distance_px_same_slide_per_latent "${MINE_MIN_DISTANCE_PX_SAME_SLIDE_PER_LATENT}"
  )
  if [[ -n "${REQUIRE_H5_EXISTS}" ]]; then
    cmd_mine+=("${REQUIRE_H5_EXISTS}")
  fi
  if [[ -n "${POOL_ALLOW_PARTIAL}" ]]; then
    cmd_mine+=("${POOL_ALLOW_PARTIAL}")
  fi
  if [[ -n "${TOPK_NONNEG}" ]]; then
    cmd_mine+=("${TOPK_NONNEG}")
  fi
  if [[ "${SELECT_STRATEGY}" == "manual" && -n "${LATENT_INDICES}" ]]; then
    cmd_mine+=(--latent_indices "${LATENT_INDICES}")
  fi
  "${cmd_mine[@]}"
fi

if [[ "${DO_EXPORT}" != "1" ]]; then
  echo "[mine_export_10x_batch_topk] DO_EXPORT=${DO_EXPORT}; stopping after mining."
  exit 0
fi

echo "[mine_export_10x_batch_topk] Exporting latent tiles..."
cmd_export=(
  python -m scripts.export_latent_tiles
  --out_root "${OUT_ROOT}"
  --run_name "${RUN_NAME}"
  --gdc_client "${GDC_CLIENT}"
  --tile_size "${TILE_SIZE}"
  --vis_size "${VIS_SIZE}"
  --ncols "${NCOLS}"
  --max_latents "${MAX_LATENTS}"
  --wsi_cache "${WSI_CACHE}"
  --max_tiles_per_slide_per_latent "${EXPORT_MAX_TILES_PER_SLIDE_PER_LATENT}"
  --context_grid "${CONTEXT_GRID}"
)
if [[ -n "${TOKEN}" ]]; then
  cmd_export+=(--token "${TOKEN}")
fi
if [[ -n "${KEEP_WSI_CACHE}" ]]; then
  cmd_export+=("${KEEP_WSI_CACHE}")
fi
if [[ -n "${OVERWRITE}" ]]; then
  cmd_export+=("${OVERWRITE}")
fi
if [[ -n "${FEATURE_MAG_OVERRIDE}" ]]; then
  cmd_export+=(--feature_magnification_override "${FEATURE_MAG_OVERRIDE}")
fi
if [[ -n "${DRAW_CENTER_BOX}" ]]; then
  cmd_export+=("${DRAW_CENTER_BOX}")
fi
"${cmd_export[@]}"
