#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
OUT_ROOT="${OUT_ROOT:-artifacts/prediction_transition_region_banks_test_only_unbalanced}"
TASKS="${TASKS:-hnscc_hpv normal_tumor prad_morphology_group}"

REGIONS_PER_SLIDE="${REGIONS_PER_SLIDE:-5}"
MAX_REGIONS_ALL="${MAX_REGIONS_ALL:-100000}"
REFRESH_REGIONS="${REFRESH_REGIONS:-0}"
DRY_RUN="${DRY_RUN:-0}"

NORMAL_TUMOR_TASKS="${NORMAL_TUMOR_TASKS:-luad_normal_tumor coad_normal_tumor kirc_normal_tumor brca_normal_tumor}"
NORMAL_TUMOR_DIRECTIONS="${NORMAL_TUMOR_DIRECTIONS:-normal_to_tumor}"
PRAD_DIRECTIONS="${PRAD_DIRECTIONS:-gg1_to_gg2 gg2_to_gg3 gg3_to_gg4 gg4_to_gg5 gg1_to_gg5 gg5_to_gg1}"
PRAD_MORPH_DIRECTIONS="${PRAD_MORPH_DIRECTIONS:-well_to_p4 p4_to_p5 well_to_p5 p5_to_well}"

NORMAL_CLASSIFIER_ROOT="${NORMAL_CLASSIFIER_ROOT:-artifacts/classifier_training_normal_tumor}"
NORMAL_SLIDES_ROOT="${NORMAL_SLIDES_ROOT:-artifacts/normal_tumor_slides}"
PRAD_CLASSIFIER_RUN_DIR="${PRAD_CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_grade_group}"
PRAD_MORPH_CLASSIFIER_RUN_DIR="${PRAD_MORPH_CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_morphology_group}"
PRAD_SLIDES_ROOT="${PRAD_SLIDES_ROOT:-artifacts/prad_gleason_slides}"

HNSCC_SLIDES_DIR="${HNSCC_SLIDES_DIR:-/common/users/wq50/CLAM/HNSCC_slides}"
HNSCC_CLAM_SPLIT="${HNSCC_CLAM_SPLIT:-test}"
HNSCC_USE_TASK_SPLIT="${HNSCC_USE_TASK_SPLIT:-0}"
HNSCC_SPLIT_TSV="${HNSCC_SPLIT_TSV:-resources/manifests/hnsc_hpv_5fold/split_0.tsv}"

ATTENTION_PERCENTILE="${ATTENTION_PERCENTILE:-50}"
MIN_SELECTED_CELLS="${MIN_SELECTED_CELLS:-16}"
MAX_SELECTED_CELLS="${MAX_SELECTED_CELLS:-32}"
TARGET_IMPORTANCE_MASS="${TARGET_IMPORTANCE_MASS:-0.90}"
MIN_LABEL_CONFIDENCE="${MIN_LABEL_CONFIDENCE:-0.35}"
MAX_LABEL_CONFIDENCE="${MAX_LABEL_CONFIDENCE:-0.9995}"
MIN_TISSUE="${MIN_TISSUE:-0.35}"
MIN_DARK_FRACTION="${MIN_DARK_FRACTION:-0.02}"
MIN_SATURATION_FRACTION="${MIN_SATURATION_FRACTION:-0.02}"

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

has_task() {
  local wanted="$1"
  [[ " ${TASKS} " == *" ${wanted} "* ]]
}

task_project() {
  case "$1" in
    luad_normal_tumor) echo "TCGA-LUAD" ;;
    coad_normal_tumor) echo "TCGA-COAD" ;;
    brca_normal_tumor) echo "TCGA-BRCA" ;;
    kirc_normal_tumor) echo "TCGA-KIRC" ;;
    *) echo "[error] unknown normal/tumor task: $1" >&2; exit 2 ;;
  esac
}

stage_normal_slides() {
  local task="$1"
  local project="$2"
  local source_dir="${NORMAL_SLIDES_ROOT}/${task}"
  local ready_dir="${NORMAL_SLIDES_ROOT}/.tmp/ready_buffer/slides/${project}"
  if [[ ! -d "${source_dir}" ]]; then
    echo "[error] missing normal slides for ${task}: ${source_dir}" >&2
    exit 2
  fi
  rm -rf "${ready_dir}"
  mkdir -p "${ready_dir}"
  while IFS= read -r -d '' slide; do
    ln -s "$(realpath "${slide}")" "${ready_dir}/$(basename "${slide}")"
  done < <(find "${source_dir}" -type f -iname '*.svs' -print0 | sort -z)
}

run_hnscc() {
  local out_dir="${OUT_ROOT}/hnscc_hpv"
  if [[ "${REFRESH_REGIONS}" != "1" && -f "${out_dir}/region_bank.csv" ]]; then
    echo "[skip] HPV test-only bank exists: ${out_dir}" >&2
    return
  fi
  local split_args=(--clam-split "${HNSCC_CLAM_SPLIT}")
  if [[ "${HNSCC_USE_TASK_SPLIT}" == "1" ]]; then
    split_args=(--clam-source-from-task-split --split-tsv "${HNSCC_SPLIT_TSV}" --split test)
  fi
  echo "[regions] HPV test-only unbalanced -> ${out_dir}" >&2
  run_or_print "${PY}" scripts/find_regions.py \
    --mode attention \
    --backend clam \
    "${split_args[@]}" \
    --slides-dir "${HNSCC_SLIDES_DIR}" \
    --target-magnification 20 \
    --region-size 2048 \
    --grid-step-px 256 \
    --final-slides-per-label 0 \
    --final-regions-per-slide "${REGIONS_PER_SLIDE}" \
    --attention-percentile 90 \
    --min-label-confidence 0.65 \
    --min-tissue 0.50 \
    --min-dark-fraction 0.08 \
    --min-saturation-fraction 0.08 \
    --out-dir "${out_dir}" \
    --sae-variant "${SAE_VARIANT}" \
    --device "${DEVICE}"
}

normal_tumor_direction_labels() {
  case "$1" in
    normal_to_tumor) printf 'normal\ttumor\n' ;;
    tumor_to_normal) printf 'tumor\tnormal\n' ;;
    *) echo "[error] unknown normal/tumor direction: $1" >&2; exit 2 ;;
  esac
}

normal_tumor_region_dir() {
  local task="$1"
  local direction="$2"
  if [[ "${direction}" == "normal_to_tumor" ]]; then
    printf '%s\n' "${OUT_ROOT}/normal_tumor/${task}"
  else
    printf '%s\n' "${OUT_ROOT}/normal_tumor/${task}/${direction}"
  fi
}

run_normal_tumor_task() {
  local task="$1"
  local direction="$2"
  local source_label="$3"
  local target_label="$4"
  local project
  project="$(task_project "${task}")"
  local out_dir
  out_dir="$(normal_tumor_region_dir "${task}" "${direction}")"
  if [[ "${REFRESH_REGIONS}" != "1" && -f "${out_dir}/region_bank.csv" ]]; then
    echo "[skip] ${task}/${direction} test-only bank exists: ${out_dir}" >&2
    return
  fi
  stage_normal_slides "${task}" "${project}"
  echo "[regions] ${task}/${direction} test-only source=${source_label} target=${target_label}, unbalanced -> ${out_dir}" >&2
  run_or_print "${PY}" scripts/find_regions.py \
    --mode attention \
    --split test \
    --classifier-run-dir "${NORMAL_CLASSIFIER_ROOT}/${task}" \
    --source-label "${source_label}" \
    --target-label "${target_label}" \
    --slides-root "${NORMAL_SLIDES_ROOT}" \
    --target-magnification 20 \
    --region-size 2048 \
    --grid-step-px 256 \
    --max-regions "${MAX_REGIONS_ALL}" \
    --max-candidates-per-slide "${REGIONS_PER_SLIDE}" \
    --attention-percentile "${ATTENTION_PERCENTILE}" \
    --min-selected-cells "${MIN_SELECTED_CELLS}" \
    --max-selected-cells "${MAX_SELECTED_CELLS}" \
    --target-importance-mass "${TARGET_IMPORTANCE_MASS}" \
    --min-label-confidence "${MIN_LABEL_CONFIDENCE}" \
    --max-label-confidence "${MAX_LABEL_CONFIDENCE}" \
    --min-tissue "${MIN_TISSUE}" \
    --min-dark-fraction "${MIN_DARK_FRACTION}" \
    --min-saturation-fraction "${MIN_SATURATION_FRACTION}" \
    --out-dir "${out_dir}" \
    --sae-variant "${SAE_VARIANT}" \
    --device "${DEVICE}"
}

prad_direction_labels() {
  case "$1" in
    gg1_to_gg2) printf 'GG1\tGG2\n' ;;
    gg2_to_gg3) printf 'GG2\tGG3\n' ;;
    gg3_to_gg4) printf 'GG3\tGG4\n' ;;
    gg4_to_gg5) printf 'GG4\tGG5\n' ;;
    gg1_to_gg5) printf 'GG1\tGG5\n' ;;
    gg5_to_gg1) printf 'GG5\tGG1\n' ;;
    *) echo "[error] unknown PRAD grade direction: $1" >&2; exit 2 ;;
  esac
}

prad_morph_direction_labels() {
  case "$1" in
    well_to_p4) printf 'pattern_1_3_well_formed\tpattern_4_cribriform_poorly_formed_fused\n' ;;
    p4_to_p5) printf 'pattern_4_cribriform_poorly_formed_fused\tpattern_5_solid_single_necrosis\n' ;;
    well_to_p5) printf 'pattern_1_3_well_formed\tpattern_5_solid_single_necrosis\n' ;;
    p5_to_well) printf 'pattern_5_solid_single_necrosis\tpattern_1_3_well_formed\n' ;;
    p4_to_well) printf 'pattern_4_cribriform_poorly_formed_fused\tpattern_1_3_well_formed\n' ;;
    p5_to_p4) printf 'pattern_5_solid_single_necrosis\tpattern_4_cribriform_poorly_formed_fused\n' ;;
    *) echo "[error] unknown PRAD morphology direction: $1" >&2; exit 2 ;;
  esac
}

run_prad_direction() {
  local direction="$1"
  local classifier="$2"
  local family="$3"
  local source_label="$4"
  local target_label="$5"
  local out_dir="${OUT_ROOT}/${family}/${direction}"
  if [[ "${REFRESH_REGIONS}" != "1" && -f "${out_dir}/region_bank.csv" ]]; then
    echo "[skip] ${family}/${direction} test-only bank exists: ${out_dir}" >&2
    return
  fi
  echo "[regions] ${family}/${direction} test-only source=${source_label} target=${target_label} -> ${out_dir}" >&2
  run_or_print "${PY}" scripts/find_regions.py \
    --mode attention \
    --split test \
    --classifier-run-dir "${classifier}" \
    --source-label "${source_label}" \
    --target-label "${target_label}" \
    --slides-root "${PRAD_SLIDES_ROOT}" \
    --target-magnification 20 \
    --region-size 2048 \
    --grid-step-px 256 \
    --max-regions "${MAX_REGIONS_ALL}" \
    --max-candidates-per-slide "${REGIONS_PER_SLIDE}" \
    --attention-percentile "${ATTENTION_PERCENTILE}" \
    --min-selected-cells "${MIN_SELECTED_CELLS}" \
    --max-selected-cells "${MAX_SELECTED_CELLS}" \
    --target-importance-mass "${TARGET_IMPORTANCE_MASS}" \
    --min-label-confidence "${MIN_LABEL_CONFIDENCE}" \
    --max-label-confidence "${MAX_LABEL_CONFIDENCE}" \
    --min-tissue "${MIN_TISSUE}" \
    --min-dark-fraction "${MIN_DARK_FRACTION}" \
    --min-saturation-fraction "${MIN_SATURATION_FRACTION}" \
    --require-label-match \
    --out-dir "${out_dir}" \
    --sae-variant "${SAE_VARIANT}" \
    --device "${DEVICE}"
}

mkdir -p "${OUT_ROOT}"

if has_task hnscc_hpv; then
  run_hnscc
fi

if has_task normal_tumor; then
  for task in ${NORMAL_TUMOR_TASKS}; do
    for direction in ${NORMAL_TUMOR_DIRECTIONS}; do
      IFS=$'\t' read -r source_label target_label < <(normal_tumor_direction_labels "${direction}")
      run_normal_tumor_task "${task}" "${direction}" "${source_label}" "${target_label}"
    done
  done
fi

if has_task prad_grade_group; then
  for direction in ${PRAD_DIRECTIONS}; do
    IFS=$'\t' read -r source_label target_label < <(prad_direction_labels "${direction}")
    run_prad_direction "${direction}" "${PRAD_CLASSIFIER_RUN_DIR}" "prad_grade_group" "${source_label}" "${target_label}"
  done
fi

if has_task prad_morphology_group; then
  for direction in ${PRAD_MORPH_DIRECTIONS}; do
    IFS=$'\t' read -r source_label target_label < <(prad_morph_direction_labels "${direction}")
    run_prad_direction "${direction}" "${PRAD_MORPH_CLASSIFIER_RUN_DIR}" "prad_morphology_group" "${source_label}" "${target_label}"
  done
fi

echo "[ok] test-only unbalanced region banks: ${OUT_ROOT}" >&2
