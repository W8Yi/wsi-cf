#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
TASKS="${TASKS:-luad_normal_tumor coad_normal_tumor brca_normal_tumor kirc_normal_tumor}"
INPUT_ROOT="${INPUT_ROOT:-artifacts/normal_tumor_inputs}"
SLIDE_ROOT="${SLIDE_ROOT:-artifacts/normal_tumor_slides}"
FEATURE_ROOT="${FEATURE_ROOT:-artifacts/normal_tumor_features}"
OUT_DIR="${OUT_DIR:-artifacts/classifier_training_normal_tumor}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-20}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
FILTER_WORKERS="${FILTER_WORKERS:-4}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"
READER="${READER:-auto}"
REFRESH_INPUTS="${REFRESH_INPUTS:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

TASKS_CSV="${TASKS// /,}"
if [[ "${REFRESH_INPUTS}" == "1" ]]; then
  echo "[stage 1/4] preparing normal/tumor cohort manifests"
  PY="${PY}" OUT_ROOT="${INPUT_ROOT}" NORMAL_FEATURES_ROOT="${FEATURE_ROOT}" \
    bash examples/concept_discovery/02_prepare_normal_tumor_tasks.sh \
    --tasks "${TASKS_CSV}"
else
  echo "[stage 1/4] keeping existing manifests (REFRESH_INPUTS=0)"
fi

echo "[stage 2/4] checking that the requested normal slides are downloaded"
"${PY}" - "${INPUT_ROOT}" "${SLIDE_ROOT}" ${TASKS} <<'PY'
import csv
import sys
from pathlib import Path

input_root = Path(sys.argv[1])
slide_root = Path(sys.argv[2])
tasks = sys.argv[3:]
failed = False
for task in tasks:
    manifest = input_root / task / "normal_all.gdc_manifest.tsv"
    if not manifest.exists():
        raise SystemExit(f"Missing download manifest: {manifest}")
    with manifest.open(newline="") as handle:
        expected = {
            Path(row["filename"]).stem
            for row in csv.DictReader(handle, delimiter="\t")
        }
    downloaded_dir = slide_root / task
    downloaded = {path.stem for path in downloaded_dir.rglob("*.svs")} if downloaded_dir.exists() else set()
    missing = sorted(expected - downloaded)
    print(f"[slides] {task}: downloaded={len(downloaded & expected)}/{len(expected)} missing={len(missing)}")
    if missing:
        print(f"         first missing slide keys: {', '.join(missing[:5])}")
        failed = True
if failed:
    raise SystemExit(
        "Some normal slides are missing under SLIDE_ROOT. "
        "Set SLIDE_ROOT to their actual location or finish the all-slide downloads."
    )
PY

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[ok] slide download preflight complete (CHECK_ONLY=1)" >&2
  exit 0
fi

echo "[stage 3/4] extracting missing UNI2 bags for normal slides"
PY="${PY}" TASKS="${TASKS}" SLIDE_ROOT="${SLIDE_ROOT}" FEATURE_ROOT="${FEATURE_ROOT}" \
DEVICE="${DEVICE}" BATCH_SIZE="${BATCH_SIZE}" FILTER_WORKERS="${FILTER_WORKERS}" \
LOADER_WORKERS="${LOADER_WORKERS}" READER="${READER}" \
bash examples/concept_discovery/04_extract_normal_tumor_uni2_features.sh

echo "[stage 4/4] training normal-versus-tumor classifiers"
PY="${PY}" TASKS="${TASKS}" INPUT_ROOT="${INPUT_ROOT}" OUT_DIR="${OUT_DIR}" \
DEVICE="${DEVICE}" EPOCHS="${EPOCHS}" MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE}" \
bash examples/classifier_training/10_train_all_normal_tumor.sh

echo "[ok] completed UNI2 extraction and classifier training for: ${TASKS}" >&2
echo "[ok] model outputs: ${OUT_DIR}" >&2
