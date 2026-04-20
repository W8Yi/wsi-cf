#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/common/users/wq50/envs/pace/bin/python"
RUNNER="/common/users/wq50/wsi_cf/scripts/run_region_bank_sae_experiments.py"
ROLES_CSV="/common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_1024/region_roles.csv"
OUT_ROOT="/common/users/wq50/wsi_cf/artifacts/selected_cells_sae_sweep"

COMMON_ARGS=(
  --region-roles-csv "${ROLES_CSV}"
  --conditions "baseline,to_hpv_pos_selected_cells"
  --max-sources 1
  --pos-latent 2645
  --neg-latent 7036
  --prototype-key prototype_median
  --prototype-strength 0.8
  --steer-blend 1.0
  --mid-steer-start-ratio 0.5
  --mid-steer-end-ratio 1.0
  --mid-steer-alpha-start 1.0
  --mid-steer-alpha-end 1.0
  --mid-steer-alpha-schedule linear
  --pix_model_id StonyBrook-CVLab/PixCell-1024
  --seed 7
  --device cuda:0
)

run_case() {
  local name="$1"
  shift
  "${PYTHON_BIN}" "${RUNNER}" \
    --out-dir "${OUT_ROOT}/${name}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

# random two
run_case "random_two_00_23" \
  --steer-cell 0,0 \
  --steer-cell 2,3

# neighboring two horizontal
run_case "neighbor_two_h_11_21" \
  --steer-cell 1,1 \
  --steer-cell 2,1

# neighboring two vertical
run_case "neighbor_two_v_11_12" \
  --steer-cell 1,1 \
  --steer-cell 1,2

# neighboring three L-shape
run_case "neighbor_three_L_11_21_22" \
  --steer-cell 1,1 \
  --steer-cell 2,1 \
  --steer-cell 2,2

# neighboring three line
run_case "neighbor_three_line_10_11_12" \
  --steer-cell 1,0 \
  --steer-cell 1,1 \
  --steer-cell 1,2

# 2x2 block
run_case "block_2x2_center" \
  --steer-cell 1,1 \
  --steer-cell 2,1 \
  --steer-cell 1,2 \
  --steer-cell 2,2

# 2x3 block
run_case "block_2x3_center" \
  --steer-cell 1,0 \
  --steer-cell 2,0 \
  --steer-cell 1,1 \
  --steer-cell 2,1 \
  --steer-cell 1,2 \
  --steer-cell 2,2
