#!/usr/bin/env bash
set -euo pipefail

# Comprehensive HNSCC HPV policy benchmark for paper-style comparison.
#
# Defaults compare:
#   1. ours: showcase_best
#   2. naive_no_preserve: same schedule/cells, no preservation
#   3. naive_full_duration: no preservation and full-duration steering
#
# Main outputs:
#   ${OUT_ROOT}/metrics/benchmark_summary_by_method.csv
#   ${OUT_ROOT}/metrics/benchmark_metrics_by_run.csv
#   ${OUT_ROOT}/metrics/benchmark_predictions.csv
#   ${OUT_ROOT}/metrics/benchmark_per_cell_metrics.csv
#   ${OUT_ROOT}/metrics/benchmark_summary.json

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
GENERATION_DEVICE="${GENERATION_DEVICE:-${DEVICE}}"
OUT_ROOT="${OUT_ROOT:-artifacts/hnscc_hpv_showcase_best_paper_policy_benchmark}"

REGION_BANK_CSV="${REGION_BANK_CSV:-artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/region_bank.csv}"
EDIT_MANIFEST="${EDIT_MANIFEST:-artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/progressive_edit_manifest.json}"
MIL_CKPT="${MIL_CKPT:-resources/models/classifiers/hnscc_hpv/mil_split0.pt}"

# The current HNSCC HPV prototype bundle is the legacy ReLU-SAE provenance.
SAE_CKPT="${SAE_CKPT:-/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt}"
SAE_CFG="${SAE_CFG:-/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json}"

MAX_RUNS="${MAX_RUNS:-0}"
RUN_EDITS="${RUN_EDITS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_REENCODE="${FORCE_REENCODE:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"

cmd=(
  "${PYTHON_BIN}" scripts/run_hnscc_hpv_policy_benchmark.py
  --out-dir "${OUT_ROOT}"
  --region-bank-csv "${REGION_BANK_CSV}"
  --edit-manifest "${EDIT_MANIFEST}"
  --mil-ckpt "${MIL_CKPT}"
  --direction hpv_neg
  --device "${DEVICE}"
  --generation-device "${GENERATION_DEVICE}"
  --sae-ckpt "${SAE_CKPT}"
  --sae-cfg "${SAE_CFG}"
  --output-mode "${OUTPUT_MODE}"
  --policy "ours=configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json"
  --policy "naive_no_preserve=configs/edit_policies/naive_no_preserve.json"
  --policy "naive_full_duration=configs/edit_policies/baseline_no_preserve_full_duration.json"
)

if [[ "${MAX_RUNS}" != "0" ]]; then
  cmd+=(--max-runs "${MAX_RUNS}")
fi

if [[ "${RUN_EDITS}" == "1" ]]; then
  cmd+=(--run-edits)
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi

if [[ "${FORCE_REENCODE}" == "1" ]]; then
  cmd+=(--force-reencode)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

printf '[hnscc-hpv benchmark] output root: %s\n' "${OUT_ROOT}"
printf '[hnscc-hpv benchmark] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

if [[ "${DRY_RUN}" != "1" ]]; then
  printf '\n[hnscc-hpv benchmark] wrote metrics:\n'
  printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_summary_by_method.csv"
  printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_metrics_by_run.csv"
  printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_predictions.csv"
  printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_per_cell_metrics.csv"
  printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_summary.json"
fi
