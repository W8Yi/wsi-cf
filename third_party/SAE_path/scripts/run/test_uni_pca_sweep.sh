#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Sweep UNI PCA directions over a tile folder using scripts.uni_steer.
# Defaults are chosen for quick qualitative inspection.
#
# Override any variable inline, e.g.:
#   PCA_DIR=outputs/uni_pca/tcga_train/directions STRENGTHS="-1 -0.5 0.5 1" \
#   bash scripts/run/test_uni_pca_sweep.sh

TILE_DIR="${TILE_DIR:-/common/users/wq50/SAE_path/cf_imgs/roi3_neg1_tiles}"
PCA_DIR="${PCA_DIR:-/common/users/wq50/SAE_path/outputs/uni_pca/tcga_test/directions}"
OUT_ROOT="${OUT_ROOT:-/common/users/wq50/SAE_path/outputs/uni_steer_pca_sweep}"

MODE="${MODE:-delta}"                  # delta or target
BLEND="${BLEND:-0.5}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

# Include both signs because PCA component sign is arbitrary.
STRENGTHS_STR="${STRENGTHS:--1.0 -0.5 -0.25 0.25 0.5 1.0}"
read -r -a STRENGTHS_ARR <<< "${STRENGTHS_STR}"

# Optional extra flags forwarded to scripts.uni_steer (space-separated).
# Example: UNI_STEER_EXTRA="--steps 20 --guidance 1.5"
UNI_STEER_EXTRA="${UNI_STEER_EXTRA:-}"

slug_float() {
  local x="$1"
  x="${x//-/m}"
  x="${x//./p}"
  printf '%s\n' "$x"
}

if [[ ! -d "${TILE_DIR}" ]]; then
  echo "Missing TILE_DIR: ${TILE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${PCA_DIR}" ]]; then
  echo "Missing PCA_DIR: ${PCA_DIR}" >&2
  exit 1
fi

mapfile -t VECTORS < <(find "${PCA_DIR}" -maxdepth 1 -type f -name 'pc_*.npy' | sort)
if [[ ${#VECTORS[@]} -eq 0 ]]; then
  echo "No PCA direction files found in ${PCA_DIR} (expected pc_*.npy)" >&2
  exit 1
fi
if [[ ${#STRENGTHS_ARR[@]} -eq 0 ]]; then
  echo "No strengths provided." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

TOTAL_RUNS=$(( ${#VECTORS[@]} * ${#STRENGTHS_ARR[@]} ))
RUN_IDX=0

echo "Tile dir:   ${TILE_DIR}"
echo "PCA dir:    ${PCA_DIR}"
echo "Out root:   ${OUT_ROOT}"
echo "Mode:       ${MODE}"
echo "Blend:      ${BLEND}"
echo "Strengths:  ${STRENGTHS_STR}"
echo "Components: ${#VECTORS[@]}"
echo "Total runs: ${TOTAL_RUNS}"

for vec in "${VECTORS[@]}"; do
  pc_name="$(basename "${vec}" .npy)"   # e.g. pc_001
  for strength in "${STRENGTHS_ARR[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    strength_tag="$(slug_float "${strength}")"
    out_dir="${OUT_ROOT}/${pc_name}/${MODE}_strength_${strength_tag}"

    echo
    echo "[${RUN_IDX}/${TOTAL_RUNS}] ${pc_name} | strength=${strength} -> ${out_dir}"

    # shellcheck disable=SC2206
    EXTRA_ARGS_ARR=( ${UNI_STEER_EXTRA} )

    python -m scripts.uni_steer \
      --image-dir "${TILE_DIR}" \
      --out-dir "${out_dir}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --mode "${MODE}" \
      --vector-path "${vec}" \
      --vector-strength "${strength}" \
      --blend "${BLEND}" \
      --reference-mix 0.0 \
      "${EXTRA_ARGS_ARR[@]}"
  done
done

echo
echo "Completed PCA sweep. Outputs in: ${OUT_ROOT}"
