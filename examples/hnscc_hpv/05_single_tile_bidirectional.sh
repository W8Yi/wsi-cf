#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-artifacts/hnscc_single_tile_bidirectional_256}"
REGION_BANK_CSV="${REGION_BANK_CSV:-artifacts/hnscc_region_bank_20x_2048_sample20/region_bank.csv}"

"$PY" scripts/run_hnscc_single_tile_steer.py \
  --region-bank-csv "${REGION_BANK_CSV}" \
  --region-index "${REGION_INDEX:-0}" \
  --cell "${CELL:-auto}" \
  --directions "${DIRECTIONS:-hpv_pos,hpv_neg}" \
  --out-dir "${OUT_DIR}" \
  --prototype-strength "${PROTOTYPE_STRENGTH:-0.8}" \
  --preserve-strength "${PRESERVE_STRENGTH:-0.05}" \
  --mid-steer-start-ratio "${MID_STEER_START_RATIO:-0.5}" \
  --mid-steer-end-ratio "${MID_STEER_END_RATIO:-1.0}" \
  --mid-steer-alpha-start "${MID_STEER_ALPHA_START:-0.5}" \
  --mid-steer-alpha-end "${MID_STEER_ALPHA_END:-1.0}" \
  --sae-variant "${SAE_VARIANT:-relu_sae_base}" \
  --steps "${STEPS:-30}" \
  --guidance "${GUIDANCE:-2.0}" \
  --device "${DEVICE}" \
  ${DRY_RUN:-}

echo "[ok] single-tile HNSCC steer: ${OUT_DIR}"
