#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
TASK="prad_morphology_group"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_prad_morphology_group_${SAE_VARIANT}}"
CLASSIFIER_RUN_DIR="${CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_morphology_group}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-artifacts/prad_gleason_inputs/splits/prad_morphology_group_patient_stratified_80_20.json}"
TASK_JSON="${TASK_JSON:-artifacts/prad_gleason_inputs/concept_task.morphology_group.attention_aware.${SAE_VARIANT}.json}"

if [[ ! -f "${CLASSIFIER_RUN_DIR}/best_model.pt" ]]; then
  echo "[error] missing PRAD morphology classifier: ${CLASSIFIER_RUN_DIR}/best_model.pt" >&2
  echo "        Train it first: DEVICE=${DEVICE} examples/classifier_training/14_train_prad_morphology_group.sh" >&2
  exit 2
fi

echo "[labels] refreshing TCGA-PRAD Gleason morphology label table" >&2
"${PY}" scripts/prepare_prad_gleason_concept_tasks.py \
  --out-dir artifacts/prad_gleason_inputs

echo "[task] writing PRAD morphology attention-aware concept task: ${TASK_JSON}" >&2
"${PY}" - "${SAE_VARIANT}" "${CLASSIFIER_RUN_DIR}" "${SPLIT_MANIFEST}" "${TASK_JSON}" <<'PY'
import json
import sys
from pathlib import Path

sae_variant = sys.argv[1]
classifier_run_dir = sys.argv[2]
split_manifest = sys.argv[3]
out_path = Path(sys.argv[4])
payload = json.loads(Path("configs/concept_tasks/prad_morphology_group.json").read_text())
payload["sae_variant"] = sae_variant
payload["ranking_mode"] = "attention_aware_optional"
payload["classifier_run_dir"] = classifier_run_dir
payload["split_manifest"] = split_manifest
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[concepts] refreshing attention-aware PRAD morphology concepts" >&2
"${PY}" scripts/find_label_concepts.py \
  --task-json "${TASK_JSON}" \
  --out-root "${CONCEPT_OUT}" \
  --device "${DEVICE}" \
  "$@"

echo "[ok] PRAD morphology concepts: ${CONCEPT_OUT}/${TASK}" >&2
