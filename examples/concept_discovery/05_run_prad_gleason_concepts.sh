#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_json}"
REFRESH_GDC="${REFRESH_GDC:-0}"

refresh_args=()
if [[ "${REFRESH_GDC}" == "1" ]]; then
  refresh_args+=(--refresh-gdc)
fi

echo "[labels] preparing TCGA-PRAD Gleason labels from GDC clinical metadata" >&2
"${PY}" scripts/prepare_prad_gleason_concept_tasks.py \
  --out-dir artifacts/prad_gleason_inputs \
  "${refresh_args[@]}"

for task_json in \
  configs/concept_tasks/prad_gleason_score.json \
  configs/concept_tasks/prad_low_vs_high_grade.json
do
  echo "[concepts] ${task_json}" >&2
  "${PY}" scripts/find_label_concepts.py \
    --task-json "${task_json}" \
    --out-root "${CONCEPT_OUT}" \
    --device "${DEVICE}" \
    "$@"
done

echo "[ok] wrote PRAD Gleason concept outputs to ${CONCEPT_OUT}" >&2
