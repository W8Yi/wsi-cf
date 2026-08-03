#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
N_REGIONS="${N_REGIONS:-1}"
SEED="${SEED:-7}"
OUT_DIR="${OUT_DIR:-paper_outputs/steering_strength_sweeps/hnscc_hpv/hpv_pos_to_hpv_neg}"

extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

"${PYTHON_BIN}" scripts/run_steering_strength_sweep.py \
  --task-name hnscc_hpv \
  --direction hpv_pos_to_hpv_neg \
  --runner-direction hpv_neg \
  --source-label hpv_pos \
  --target-label hpv_neg \
  --label-order hpv_neg,hpv_pos \
  --region-bank-csv artifacts/prediction_transition_region_banks_test_only_unbalanced/hnscc_hpv/region_bank.csv \
  --base-edit-manifest paper_outputs/prediction_transition_benchmark_test_only_unbalanced/manifests/hnscc_hpv/hpv_pos_to_hpv_neg/combined_manifest.json \
  --request-selector attention \
  --request-budget 32 \
  --classifier-ckpt resources/models/classifiers/hnscc_hpv/mil_split0.pt \
  --sae-variant relu_sae_base \
  --n-regions "${N_REGIONS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --out-dir "${OUT_DIR}" \
  --title "HNSCC HPV+ to HPV-: controlled steering strength" \
  "${extra_args[@]}"
