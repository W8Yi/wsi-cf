#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
OUT_ROOT="${OUT_ROOT:-artifacts/hnscc_hpv_showcase_smoothed28_tutorial}"
SAE_CKPT="${SAE_CKPT:-/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt}"
SAE_CFG="${SAE_CFG:-/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json}"

"${PYTHON_BIN}" scripts/run_progressive_region_edit.py \
  --edit-policy configs/edit_policies/original28_flip.json \
  --region-bank-csv artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/region_bank.csv \
  --edit-manifest artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28/progressive_edit_manifest.json \
  --direction hpv_neg \
  --out-dir "${OUT_ROOT}/progressive" \
  --device "${DEVICE}" \
  --sae-ckpt "${SAE_CKPT}" \
  --sae-cfg "${SAE_CFG}" \
  --output-mode debug
