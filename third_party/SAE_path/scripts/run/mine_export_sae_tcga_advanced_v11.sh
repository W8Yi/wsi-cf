#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TCGA_FEATURES_BASE="${TCGA_FEATURES_BASE:-/research/projects/mllab/WSI/TCGA_features}"

RUN_NAME="${RUN_NAME:-tcga_sae_advanced_v11}"
RUN_ROOT="${RUN_ROOT:-/common/users/wq50/SAE_path/runs/${RUN_NAME}}"
MINING_ROOT_BASE="${MINING_ROOT_BASE:-/common/users/wq50/SAE_path/sae_mining}"
MINING_ROOT="${MINING_ROOT:-${MINING_ROOT_BASE}/${RUN_NAME}}"

INDEX_JSON="${INDEX_JSON:-metadata/indexes/manifest_index.json}"
SLIDES_PER_PROJECT="${SLIDES_PER_PROJECT:-200}"
TILES_PER_SLIDE="${TILES_PER_SLIDE:-2048}"
CHUNK_TILES="${CHUNK_TILES:-512}"
TOPN="${TOPN:-50}"
N_LATENTS_BATCH="${N_LATENTS_BATCH:-128}"
N_LATENTS_SDF2="${N_LATENTS_SDF2:-128}"

PARENT_MAX_CHILDREN="${PARENT_MAX_CHILDREN:-6}"
PARENT_PREFERRED_CHILDREN="${PARENT_PREFERRED_CHILDREN:-4}"
PARENT_TARGET_COUNT="${PARENT_TARGET_COUNT:--1}"

GDC_CLIENT="${GDC_CLIENT:-/common/users/wq50/SAE_path/gdc/gdc-client}"
DO_BATCH_TOPK="${DO_BATCH_TOPK:-1}"
DO_SDF2="${DO_SDF2:-1}"

mkdir -p "${MINING_ROOT}"

run_and_export () {
  local stage_run_name="$1"
  local pass2_json
  pass2_json="$(ls -t "${MINING_ROOT}/${stage_run_name}"/pass2_top_tiles_*.json 2>/dev/null | head -n 1 || true)"
  if [[ -z "${pass2_json}" ]]; then
    echo "[error] no pass2 json found for ${stage_run_name}" >&2
    exit 1
  fi
  python -m scripts.export_latent_tiles \
    --out_root "${MINING_ROOT}" \
    --run_name "${stage_run_name}" \
    --pass2_json "${pass2_json}" \
    --gdc_client "${GDC_CLIENT}"
}

if [[ "${DO_BATCH_TOPK}" == "1" ]]; then
  python -m scripts.latent_mining \
    --out_root "${MINING_ROOT}" \
    --run_name batch_topk \
    --mode both \
    --ckpt "${RUN_ROOT}/batch_topk/batch_topk_final.pt" \
    --stage batch_topk \
    --d_in 1536 \
    --latent_dim 12288 \
    --topk_k 64 \
    --topk_mode value \
    --topk_nonneg \
    --index_json "${INDEX_JSON}" \
    --slides_per_project "${SLIDES_PER_PROJECT}" \
    --require_h5_exists \
    --tiles_per_slide "${TILES_PER_SLIDE}" \
    --chunk_tiles "${CHUNK_TILES}" \
    --select_strategy top_activation \
    --n_latents "${N_LATENTS_BATCH}" \
    --topn "${TOPN}"

  run_and_export batch_topk
fi

if [[ "${DO_SDF2}" == "1" ]]; then
  python -m scripts.latent_mining \
    --out_root "${MINING_ROOT}" \
    --run_name sdf2 \
    --mode both \
    --ckpt "${RUN_ROOT}/sdf2/sdf2_final.pt" \
    --stage sdf2 \
    --d_in 1536 \
    --latent_dim 12288 \
    --sdf_n_level2 256 \
    --sdf_coeff_simplex \
    --index_json "${INDEX_JSON}" \
    --slides_per_project "${SLIDES_PER_PROJECT}" \
    --require_h5_exists \
    --tiles_per_slide "${TILES_PER_SLIDE}" \
    --chunk_tiles "${CHUNK_TILES}" \
    --select_strategy sdf_parent_balanced \
    --n_latents "${N_LATENTS_SDF2}" \
    --parent_max_children_per_selected_parent "${PARENT_MAX_CHILDREN}" \
    --parent_preferred_children_per_selected_parent "${PARENT_PREFERRED_CHILDREN}" \
    --parent_target_count "${PARENT_TARGET_COUNT}" \
    --topn "${TOPN}"

  run_and_export sdf2
fi

echo "[done] mining/export outputs at ${MINING_ROOT}"
