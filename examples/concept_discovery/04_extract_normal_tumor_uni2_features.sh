#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
EXTRACTOR="${EXTRACTOR:-/common/users/wq50/SAE_path/scripts/extract_filtered_uni_from_wsi.py}"
SLIDE_ROOT="${SLIDE_ROOT:-artifacts/normal_tumor_slides}"
FEATURE_ROOT="${FEATURE_ROOT:-artifacts/normal_tumor_features}"
TASKS="${TASKS:-luad_normal_tumor coad_normal_tumor brca_normal_tumor kirc_normal_tumor}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
FILTER_WORKERS="${FILTER_WORKERS:-4}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"
READER="${READER:-auto}"

for TASK in ${TASKS}; do
  case "${TASK}" in
    luad_normal_tumor) PROJECT="TCGA-LUAD" ;;
    coad_normal_tumor) PROJECT="TCGA-COAD" ;;
    brca_normal_tumor) PROJECT="TCGA-BRCA" ;;
    kirc_normal_tumor) PROJECT="TCGA-KIRC" ;;
    lusc_normal_tumor) PROJECT="TCGA-LUSC" ;;
    *) echo "[error] unknown task: ${TASK}" >&2; exit 2 ;;
  esac

  SOURCE_DIR="${SLIDE_ROOT}/${TASK}"
  OUT_DIR="${FEATURE_ROOT}/${PROJECT}/features_uni2"
  COORDS_DIR="${FEATURE_ROOT}/${PROJECT}/coords"
  SUMMARY_DIR="${FEATURE_ROOT}/${PROJECT}/summary"
  VIZ_DIR="${FEATURE_ROOT}/${PROJECT}/vis"
  STAGED_DIR="${FEATURE_ROOT}/${PROJECT}/slides_pending"

  if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "[error] no downloaded slides for ${TASK}: ${SOURCE_DIR}" >&2
    echo "        Download this task first with ALL=1 TASK=${TASK} bash examples/concept_discovery/03_download_normal_tumor_normal_slides.sh" >&2
    exit 2
  fi

  mkdir -p "${OUT_DIR}" "${COORDS_DIR}" "${SUMMARY_DIR}" "${VIZ_DIR}"
  rm -rf "${STAGED_DIR}"
  mkdir -p "${STAGED_DIR}"
  FOUND=0
  PENDING=0
  while IFS= read -r -d '' SLIDE; do
    FOUND=$((FOUND + 1))
    STEM="$(basename "${SLIDE}")"
    STEM="${STEM%.*}"
    if [[ ! -f "${OUT_DIR}/${STEM}.h5" ]]; then
      ln -s "$(realpath "${SLIDE}")" "${STAGED_DIR}/$(basename "${SLIDE}")"
      PENDING=$((PENDING + 1))
    fi
  done < <(find "${SOURCE_DIR}" -type f -iname '*.svs' -print0 | sort -z)

  if [[ "${FOUND}" -eq 0 ]]; then
    echo "[error] no .svs files found for ${TASK}: ${SOURCE_DIR}" >&2
    exit 2
  fi
  if [[ "${PENDING}" -eq 0 ]]; then
    echo "[skip] ${TASK}: all ${FOUND} downloaded normal slides already have UNI2 bags"
    continue
  fi

  echo "[run] ${TASK}: extracting ${PENDING}/${FOUND} normal UNI2 bags -> ${OUT_DIR}"
  "${PY}" "${EXTRACTOR}" \
    --wsi_dir "${STAGED_DIR}" \
    --out_dir "${OUT_DIR}" \
    --coords_dir "${COORDS_DIR}" \
    --summary_dir "${SUMMARY_DIR}" \
    --viz_dir "${VIZ_DIR}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --filter_workers "${FILTER_WORKERS}" \
    --loader_workers "${LOADER_WORKERS}" \
    --reader "${READER}" \
    "$@"
done

echo "[ok] normal UNI2 feature root: ${FEATURE_ROOT}" >&2
