#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-7}"

# 8x8 UNI grid = 64 cells. These are 25%, 50%, 75%, and 100%.
BUDGETS="${BUDGETS:-16,32,48,64}"
RUN_TAG="${RUN_TAG:-policy09_percentage_sweep}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/hnscc_hpv_showcase_smoothed28_${RUN_TAG}}"
MANIFEST_DIR="${MANIFEST_DIR:-artifacts/hnscc_hpv_showcase_policy09_percentage_sweep/manifests}"
EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SAE_CKPT="${SAE_CKPT:-/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt}"
SAE_CFG="${SAE_CFG:-/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json}"

BASE_MANIFEST="${BASE_MANIFEST:-artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/progressive_edit_manifest.json}"
ATTENTION_CELLS_CSV="${ATTENTION_CELLS_CSV:-artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/attention_cells.csv}"
REGION_BANK_CSV="${REGION_BANK_CSV:-artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/region_bank.csv}"
SOURCE_IMAGE="${SOURCE_IMAGE:-paper_example/showcase_smoothed28/source_region_actual.png}"

echo "[source] ${SOURCE_IMAGE}" >&2
echo "[build] policy-09 percentage manifests: ${MANIFEST_DIR}" >&2
"${PYTHON_BIN}" scripts/build_attention_budget_edit_manifests.py \
  --base-manifest "${BASE_MANIFEST}" \
  --attention-cells-csv "${ATTENTION_CELLS_CSV}" \
  --out-dir "${MANIFEST_DIR}" \
  --budgets "${BUDGETS}" \
  --start-mode attention_only \
  --run-suffix "policy09_pct"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] policy: ${EDIT_POLICY}" >&2
  echo "[dry-run] manifest: ${MANIFEST_DIR}/all_budgets_manifest.json" >&2
  echo "[dry-run] output: ${OUT_ROOT}/progressive" >&2
  exit 0
fi

skip_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_args+=(--skip-existing)
fi

echo "[run] HNSCC HPV+ -> HPV- policy-09 percentage sweep" >&2
echo "[run] budgets=${BUDGETS} output=${OUT_ROOT}/progressive" >&2
"${PYTHON_BIN}" scripts/run_progressive_region_edit.py \
  --edit-policy "${EDIT_POLICY}" \
  --region-bank-csv "${REGION_BANK_CSV}" \
  --edit-manifest "${MANIFEST_DIR}/all_budgets_manifest.json" \
  --direction hpv_neg \
  --out-dir "${OUT_ROOT}/progressive" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --sae-ckpt "${SAE_CKPT}" \
  --sae-cfg "${SAE_CFG}" \
  --output-mode "${OUTPUT_MODE}" \
  "${skip_args[@]}"

echo "[ok] wrote policy-09 percentage sweep outputs to ${OUT_ROOT}/progressive" >&2
