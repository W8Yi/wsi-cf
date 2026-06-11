#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
TASK="${TASK:-kirc_normal_tumor}"
ALL="${ALL:-0}"
MAX_FILES="${MAX_FILES:-0}"
INPUT_ROOT="${INPUT_ROOT:-artifacts/normal_tumor_inputs}"
OUT_ROOT="${OUT_ROOT:-artifacts/normal_tumor_slides}"

if [[ "${ALL}" == "1" ]]; then
  MANIFEST="${INPUT_ROOT}/${TASK}/normal_all.gdc_manifest.tsv"
else
  MANIFEST="${INPUT_ROOT}/${TASK}/normal_starter.gdc_manifest.tsv"
fi

"$PY" scripts/download_gdc_manifest_slides.py \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_ROOT}/${TASK}" \
  --max-files "${MAX_FILES}" \
  "$@"

echo "[ok] normal slide directory: ${OUT_ROOT}/${TASK}" >&2
