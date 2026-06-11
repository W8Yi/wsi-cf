#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
OUT_ROOT="${OUT_ROOT:-artifacts/normal_tumor_inputs}"
NORMAL_FEATURES_ROOT="${NORMAL_FEATURES_ROOT:-artifacts/normal_tumor_features}"
DOWNLOAD_SUBSET_SIZE="${DOWNLOAD_SUBSET_SIZE:-20}"

"$PY" scripts/prepare_normal_tumor_concept_tasks.py \
  --out-root "${OUT_ROOT}" \
  --normal-features-root "${NORMAL_FEATURES_ROOT}" \
  --download-subset-size "${DOWNLOAD_SUBSET_SIZE}" \
  "$@"

echo "[ok] normal/tumor manifests and concept task JSON files: ${OUT_ROOT}" >&2
echo "[note] concept mining requires UNI2 feature bags for the normal SVS files listed in each manifest." >&2
