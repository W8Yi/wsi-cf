#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-artifacts/normal_tumor_inputs}"
OUT_DIR="${OUT_DIR:-artifacts/classifier_training_normal_tumor}"
TASKS="${TASKS:-luad_normal_tumor coad_normal_tumor brca_normal_tumor kirc_normal_tumor}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-20}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-4096}"
REQUIRE_COMPLETE_NORMALS="${REQUIRE_COMPLETE_NORMALS:-1}"

for TASK in ${TASKS}; do
  LABELS="${INPUT_ROOT}/${TASK}/slide_labels.csv"
  SPLIT="${INPUT_ROOT}/${TASK}/patient_train_test_90_10.json"
  if [[ ! -f "${LABELS}" || ! -f "${SPLIT}" ]]; then
    echo "[error] missing task inputs for ${TASK}; run examples/concept_discovery/02_prepare_normal_tumor_tasks.sh first" >&2
    exit 2
  fi

  "${PY}" - "${TASK}" "${LABELS}" "${REQUIRE_COMPLETE_NORMALS}" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

task, labels_path, require_complete = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
counts = Counter()
missing = Counter()
with labels_path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        label = row["normal_tumor_label"]
        counts[label] += 1
        if not Path(row["h5_path"]).exists():
            missing[label] += 1
print(f"[check] {task}: labels={dict(counts)} missing_h5={dict(missing)}")
if require_complete and missing["normal"]:
    raise SystemExit(
        f"{task}: {missing['normal']} normal UNI2 bags are missing; "
        "run examples/concept_discovery/04_extract_normal_tumor_uni2_features.sh first"
    )
PY

  echo "[run] classifier ${TASK}"
  "${PY}" scripts/train_attention_classifier.py \
    --task-name "${TASK}" \
    --label-source "${LABELS}" \
    --split-manifest "${SPLIT}" \
    --projects all \
    --label-column normal_tumor_label \
    --include-labels normal,tumor \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
    "$@"
done

echo "[ok] classifier outputs: ${OUT_DIR}" >&2
