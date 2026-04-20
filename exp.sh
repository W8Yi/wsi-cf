#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/common/users/wq50/envs/pace/bin/python"
RUNNER="/common/users/wq50/wsi_cf/scripts/run_region_bank_sae_experiments.py"
ROLES_CSV="/common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_1024/region_roles.csv"
OUT_ROOT="/common/users/wq50/wsi_cf/artifacts/location_sweep_sae_one_cell"

for SPEC in \
  "1 1 center_11" \
  "2 2 center_22" \
  "1 0 top_edge_10" \
  "0 2 left_edge_02" \
  "0 0 corner_00" \
  "3 3 corner_33"
do
  set -- $SPEC
  GX="$1"
  GY="$2"
  NAME="$3"

  "${PYTHON_BIN}" "${RUNNER}" \
    --region-roles-csv "${ROLES_CSV}" \
    --out-dir "${OUT_ROOT}/${NAME}" \
    --conditions baseline,to_hpv_pos_one_cell \
    --max-sources 1 \
    --pos-latent 2645 \
    --prototype-key prototype_median \
    --prototype-strength 0.8 \
    --steer-blend 1.0 \
    --steer-gx "${GX}" \
    --steer-gy "${GY}" \
    --mid-steer-start-ratio 0.5 \
    --mid-steer-end-ratio 1.0 \
    --mid-steer-alpha-start 0.2 \
    --mid-steer-alpha-end 1.0 \
    --mid-steer-alpha-schedule linear \
    --pix_model_id StonyBrook-CVLab/PixCell-1024 \
    --seed 7 \
    --device cuda:0
done
