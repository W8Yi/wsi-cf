#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-auto}"
OUT_ROOT="${OUT_ROOT:-artifacts/hnscc_hpv_showcase_smoothed28_tutorial}"
RUN_ID="${RUN_ID:-case_clean_sae_expand_attn_seed_p90__attention_only_top23_smooth4_prune28}"

"${PYTHON_BIN}" examples/hnscc_hpv_showcase_smoothed28/compute_metrics.py \
  --progressive-run-dir "${OUT_ROOT}/progressive/${RUN_ID}" \
  --naive-run-dir "${OUT_ROOT}/naive_previous_settings/${RUN_ID}" \
  --out-dir "${OUT_ROOT}/metrics" \
  --stage-mode reencode \
  --device "${DEVICE}"
