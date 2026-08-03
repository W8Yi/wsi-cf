#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
OUT_ROOT="${OUT_ROOT:-artifacts/prad_gleason_slides}"
INPUT_ROOT="${INPUT_ROOT:-artifacts/prad_gleason_inputs}"
MAX_FILES="${MAX_FILES:-0}"
VERIFY_MD5="${VERIFY_MD5:-1}"
REFRESH_GDC="${REFRESH_GDC:-0}"
CHUNK_MB="${CHUNK_MB:-4}"
RETRIES="${RETRIES:-8}"
RETRY_SLEEP="${RETRY_SLEEP:-20}"

refresh_args=()
if [[ "${REFRESH_GDC}" == "1" ]]; then
  refresh_args+=(--refresh-gdc)
fi

echo "[manifest] preparing PRAD SVS GDC manifest" >&2
"${PY}" scripts/prepare_prad_slide_download_manifest.py \
  --labels "${INPUT_ROOT}/slide_labels.csv" \
  --out-dir "${INPUT_ROOT}" \
  "${refresh_args[@]}"

verify_arg="--verify-md5"
if [[ "${VERIFY_MD5}" == "0" ]]; then
  verify_arg="--no-verify-md5"
fi

echo "[download] TCGA-PRAD SVS slides -> ${OUT_ROOT}/TCGA-PRAD/slides" >&2
"${PY}" scripts/download_gdc_manifest_slides.py \
  --manifest "${INPUT_ROOT}/prad_all.gdc_manifest.tsv" \
  --out-dir "${OUT_ROOT}/TCGA-PRAD/slides" \
  --max-files "${MAX_FILES}" \
  --chunk-mb "${CHUNK_MB}" \
  --retries "${RETRIES}" \
  --retry-sleep "${RETRY_SLEEP}" \
  "${verify_arg}" \
  "$@"

echo "[ok] PRAD slide root: ${OUT_ROOT}" >&2
