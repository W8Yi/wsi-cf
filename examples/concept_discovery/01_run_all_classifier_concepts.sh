#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
ASSOC_ROOT="${ASSOC_ROOT:-artifacts/concept_label_associations_classifier}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/classifier_label_concepts_all_top10_top50}"
TOP_CONCEPTS="${TOP_CONCEPTS:-10}"
TOP_TILES="${TOP_TILES:-50}"
CANDIDATE_LATENTS="${CANDIDATE_LATENTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_SLIDES="${MAX_SLIDES:-0}"
MAX_SLIDES_PER_CLASS="${MAX_SLIDES_PER_CLASS:-0}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-0}"
SPLIT="${SPLIT:-all}"
TASKS="${TASKS:-kirc_grade,kirc_low_vs_high_grade,msi_coad_stad,luad_lusc,cancer_type_all_tcga}"

labels_for_task() {
  local run_dir="$1"
  "$PY" - "$run_dir/label_mapping.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
for label in payload["label_to_id"].keys():
    print(label)
PY
}

run_one_task() {
  local task="$1"
  local run_dir="artifacts/classifier_training/${task}"
  if [[ ! -f "${run_dir}/task_manifest.csv" ]]; then
    echo "[skip] ${task}: missing ${run_dir}/task_manifest.csv" >&2
    return
  fi
  if [[ ! -f "${run_dir}/best_model.pt" ]]; then
    echo "[skip] ${task}: missing ${run_dir}/best_model.pt" >&2
    return
  fi

  echo "[assoc] ${task}" >&2
  "$PY" scripts/prepare_classifier_concept_associations.py \
    --task-name "${task}" \
    --classifier-run-dir "${run_dir}" \
    --out-root "${ASSOC_ROOT}" \
    --split "${SPLIT}" \
    --max-slides "${MAX_SLIDES}" \
    --max-slides-per-class "${MAX_SLIDES_PER_CLASS}" \
    --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --skip-existing

  while IFS= read -r label; do
    [[ -z "${label}" ]] && continue
    echo "[concepts] ${task} / ${label}" >&2
    "$PY" scripts/find_label_concepts.py \
      --task "${task}" \
      --association-root "${ASSOC_ROOT}" \
      --class-label "${label}" \
      --mode attention_aware \
      --backend mil \
      --classifier-run-dir "${run_dir}" \
      --slides-csv "${ASSOC_ROOT}/${task}/cohort_slides.csv" \
      --out-dir "${CONCEPT_OUT}" \
      --top-concepts "${TOP_CONCEPTS}" \
      --candidate-latents "${CANDIDATE_LATENTS}" \
      --top-tiles-per-concept "${TOP_TILES}" \
      --batch-size "${BATCH_SIZE}" \
      --max-slides "${MAX_SLIDES}" \
      --device "${DEVICE}" \
      --skip-existing
  done < <(labels_for_task "${run_dir}")
}

IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
for task in "${TASK_ARRAY[@]}"; do
  run_one_task "${task}"
done

echo "[ok] wrote associations to ${ASSOC_ROOT}" >&2
echo "[ok] wrote concept cards and representative tiles to ${CONCEPT_OUT}" >&2
