#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-7}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
OUT_ROOT="${OUT_ROOT:-paper_example/hnscc_hpv_showcase_smoothed28_attention_budget_sweep_${RUN_TAG}}"
MANIFEST_DIR="${MANIFEST_DIR:-artifacts/hnscc_hpv_showcase_attention_budget_sweep/manifests}"
BUDGETS="${BUDGETS:-28,32,40,48,56,64}"
EDIT_SUPPORT="${EDIT_SUPPORT:-padded_center_2x2}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
SAE_CKPT="${SAE_CKPT:-/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt}"
SAE_CFG="${SAE_CFG:-/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json}"

BASE_MANIFEST="${BASE_MANIFEST:-artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/progressive_edit_manifest.json}"
ATTENTION_CELLS_CSV="${ATTENTION_CELLS_CSV:-artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/attention_cells.csv}"
REGION_BANK_CSV="${REGION_BANK_CSV:-artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/region_bank.csv}"

echo "[build] attention budget manifests: ${MANIFEST_DIR}" >&2
"${PYTHON_BIN}" scripts/build_attention_budget_edit_manifests.py \
  --base-manifest "${BASE_MANIFEST}" \
  --attention-cells-csv "${ATTENTION_CELLS_CSV}" \
  --out-dir "${MANIFEST_DIR}" \
  --budgets "${BUDGETS}" \
  --start-mode base_then_attention

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] manifest: ${MANIFEST_DIR}/all_budgets_manifest.json" >&2
  echo "[dry-run] output: ${OUT_ROOT}/progressive" >&2
  exit 0
fi

echo "[run] HNSCC HPV attention budget sweep" >&2
echo "[run] budgets=${BUDGETS} edit_support=${EDIT_SUPPORT} output=${OUT_ROOT}/progressive" >&2
"${PYTHON_BIN}" scripts/run_progressive_region_edit.py \
  --edit-policy configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json \
  --edit-support "${EDIT_SUPPORT}" \
  --region-bank-csv "${REGION_BANK_CSV}" \
  --edit-manifest "${MANIFEST_DIR}/all_budgets_manifest.json" \
  --direction hpv_neg \
  --out-dir "${OUT_ROOT}/progressive" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --sae-ckpt "${SAE_CKPT}" \
  --sae-cfg "${SAE_CFG}" \
  --output-mode "${OUTPUT_MODE}" \
  --skip-existing

echo "[ok] wrote sweep outputs to ${OUT_ROOT}/progressive" >&2
