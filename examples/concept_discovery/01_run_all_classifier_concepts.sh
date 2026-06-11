#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_json}"
TASK_JSONS="${TASK_JSONS:-configs/concept_tasks/luad_lusc.json,configs/concept_tasks/kirc_low_vs_high_grade.json,configs/concept_tasks/tumor_purity_low_high.json,configs/concept_tasks/hnsc_hpv.json}"

IFS=',' read -r -a TASK_JSON_ARRAY <<< "${TASK_JSONS}"
for task_json in "${TASK_JSON_ARRAY[@]}"; do
  [[ -z "${task_json}" ]] && continue
  echo "[concepts] ${task_json}" >&2
  "$PY" scripts/find_label_concepts.py \
    --task-json "${task_json}" \
    --out-root "${CONCEPT_OUT}" \
    --device "${DEVICE}" \
    --skip-existing
done

echo "[ok] wrote JSON-driven concept discovery outputs to ${CONCEPT_OUT}" >&2
