#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SAE_VARIANT="${SAE_VARIANT:-tcga_sae_batch_topk_20x_interp}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_json}"
TASK_JSON="${TASK_JSON:-artifacts/prad_gleason_inputs/concept_task.attention_aware.${SAE_VARIANT}.json}"

echo "[labels] refreshing TCGA-PRAD Gleason label table" >&2
"${PY}" scripts/prepare_prad_gleason_concept_tasks.py \
  --out-dir artifacts/prad_gleason_inputs

echo "[task] writing PRAD attention-aware concept task: ${TASK_JSON}" >&2
"${PY}" - "${SAE_VARIANT}" "${TASK_JSON}" <<'PY'
import json
import sys
from pathlib import Path

sae_variant = sys.argv[1]
out_path = Path(sys.argv[2])
payload = json.loads(Path("configs/concept_tasks/prad_low_vs_high_grade.json").read_text())
payload["sae_variant"] = sae_variant
payload["ranking_mode"] = "attention_aware_optional"
payload["classifier_run_dir"] = "artifacts/classifier_training/prad_low_vs_high_grade"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[concepts] refreshing attention-aware PRAD low/high concepts" >&2
"${PY}" scripts/find_label_concepts.py \
  --task-json "${TASK_JSON}" \
  --out-root "${CONCEPT_OUT}" \
  --device "${DEVICE}" \
  "$@"

echo "[ok] PRAD low/high concepts: ${CONCEPT_OUT}/prad_low_vs_high_grade" >&2
