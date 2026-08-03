#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  RUN_NAIVE=1 examples/paper_metrics/03_run_visual_perturbation_naive_baseline.sh
  RUN_NAIVE=0 RUN_VISUAL_METRICS=1 examples/paper_metrics/03_run_visual_perturbation_naive_baseline.sh

Environment:
  REFERENCE_OUT_ROOT   Existing paper edit benchmark root.
  NAIVE_OUT_ROOT       Temporary/persistent root for bad naive generated images.
  VISUAL_METRICS_ROOT  Output root for visual perturbation CSV and optional plots.
  TASKS                Same task families accepted by 01_run_prediction_transition_benchmark.sh.
  RUN_NAIVE            Generate bad naive baseline when set to 1.
  STREAM_NAIVE         Default 1. Generate/score/delete bad naive images in small chunks.
  STREAM_CHUNK_SIZE    Number of naive runs stored at once. Use 1 for lowest disk use.
  KEEP_NAIVE_IMAGES    Default 0 in streaming mode.
  VISUAL_FORMATS       Plot formats. Default empty in streaming mode; use png,pdf,svg when wanted.
  RUN_VISUAL_METRICS   Compare existing ours/naive generated images when set to 1.
EOF
  exit 0
fi

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
TASKS="${TASKS:-hnscc_hpv normal_tumor prad_morphology_group}"
NORMAL_TUMOR_TASKS="${NORMAL_TUMOR_TASKS:-luad_normal_tumor coad_normal_tumor kirc_normal_tumor brca_normal_tumor}"
PRAD_DIRECTIONS="${PRAD_DIRECTIONS:-gg1_to_gg2 gg2_to_gg3 gg3_to_gg4 gg4_to_gg5 gg1_to_gg5 gg5_to_gg1}"
PRAD_MORPH_DIRECTIONS="${PRAD_MORPH_DIRECTIONS:-well_to_p4 p4_to_p5 well_to_p5 p5_to_well}"

REFERENCE_OUT_ROOT="${REFERENCE_OUT_ROOT:-paper_outputs/prediction_transition_benchmark_test_only_unbalanced}"
NAIVE_OUT_ROOT="${NAIVE_OUT_ROOT:-${REFERENCE_OUT_ROOT}_bad_naive_no_preserve_no_sliding}"
VISUAL_METRICS_ROOT="${VISUAL_METRICS_ROOT:-${REFERENCE_OUT_ROOT}/metrics_visual}"

RUN_NAIVE="${RUN_NAIVE:-0}"
STREAM_NAIVE="${STREAM_NAIVE:-1}"
STREAM_CHUNK_SIZE="${STREAM_CHUNK_SIZE:-8}"
KEEP_NAIVE_IMAGES="${KEEP_NAIVE_IMAGES:-0}"
WRITE_PER_CELL_VISUAL="${WRITE_PER_CELL_VISUAL:-0}"
RUN_VISUAL_METRICS="${RUN_VISUAL_METRICS:-1}"
ALLOW_MISSING_VISUAL="${ALLOW_MISSING_VISUAL:-1}"
if [[ "${STREAM_NAIVE:-1}" == "1" ]]; then
  VISUAL_FORMATS="${VISUAL_FORMATS:-}"
else
  VISUAL_FORMATS="${VISUAL_FORMATS:-png,pdf,svg}"
fi

DEVICE="${DEVICE:-cuda:0}"
EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/bad_naive_no_preserve_no_sliding.json}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
OUTPUT_MODE="${OUTPUT_MODE:-minimal}"
STEPS="${STEPS:-30}"
PATCH_BATCH="${PATCH_BATCH:-256}"
EDIT_SUPPORT="${EDIT_SUPPORT:-padded_center_2x2}"
WINDOW_STRIDE_CELLS="${WINDOW_STRIDE_CELLS:-4}"
WINDOW_SELECTION_MODE="${WINDOW_SELECTION_MODE:-coverage}"
COMMIT_MODE="${COMMIT_MODE:-full_window}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
MAX_CONCEPTS="${MAX_CONCEPTS:-3}"
MAX_RUNS="${MAX_RUNS:-0}"

HNSCC_REGION_BANK_CSV="${HNSCC_REGION_BANK_CSV:-artifacts/prediction_transition_region_banks_test_only_unbalanced/hnscc_hpv/region_bank.csv}"
NORMAL_REGION_ROOT="${NORMAL_REGION_ROOT:-artifacts/prediction_transition_region_banks_test_only_unbalanced/normal_tumor}"
NORMAL_REGION_FALLBACK_ROOTS="${NORMAL_REGION_FALLBACK_ROOTS:-}"
NORMAL_CONCEPT_ROOT="${NORMAL_CONCEPT_ROOT:-artifacts/concept_discovery_normal_tumor_relu_sae_base}"
PRAD_REGION_ROOT="${PRAD_REGION_ROOT:-artifacts/prediction_transition_region_banks_test_only_unbalanced/prad_grade_group}"
PRAD_CONCEPT_ROOT="${PRAD_CONCEPT_ROOT:-artifacts/concept_discovery_prad_grade_group_relu_sae_base/prad_grade_group/labels}"
PRAD_MORPH_REGION_ROOT="${PRAD_MORPH_REGION_ROOT:-artifacts/prediction_transition_region_banks_test_only_unbalanced/prad_morphology_group}"
PRAD_MORPH_CONCEPT_ROOT="${PRAD_MORPH_CONCEPT_ROOT:-artifacts/concept_discovery_prad_morphology_group_relu_sae_base/prad_morphology_group/labels}"

export PY DEVICE TASKS NORMAL_TUMOR_TASKS PRAD_DIRECTIONS PRAD_MORPH_DIRECTIONS
export HNSCC_REGION_BANK_CSV NORMAL_REGION_ROOT NORMAL_REGION_FALLBACK_ROOTS NORMAL_CONCEPT_ROOT
export PRAD_REGION_ROOT PRAD_CONCEPT_ROOT PRAD_MORPH_REGION_ROOT PRAD_MORPH_CONCEPT_ROOT
export SAE_VARIANT OUTPUT_MODE STEPS PATCH_BATCH CONCEPT_TARGET_TOP_K MAX_CONCEPTS

has_task() {
  local wanted="$1"
  local item
  for item in ${TASKS}; do
    [[ "${item}" == "${wanted}" ]] && return 0
  done
  return 1
}

normal_region_bank_path() {
  local task_name="$1"
  local preferred="${NORMAL_REGION_ROOT}/${task_name}/region_bank.csv"
  if [[ -f "${preferred}" ]]; then
    printf '%s\n' "${preferred}"
    return 0
  fi
  local root
  for root in ${NORMAL_REGION_FALLBACK_ROOTS}; do
    local candidate="${root}/${task_name}/region_bank.csv"
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${preferred}"
}

prad_morph_direction_labels() {
  case "$1" in
    well_to_p4) printf '%s\t%s\n' "pattern_1_3_well_formed" "pattern_4_cribriform_poorly_formed_fused" ;;
    p4_to_p5) printf '%s\t%s\n' "pattern_4_cribriform_poorly_formed_fused" "pattern_5_solid_single_necrosis" ;;
    well_to_p5) printf '%s\t%s\n' "pattern_1_3_well_formed" "pattern_5_solid_single_necrosis" ;;
    p5_to_well) printf '%s\t%s\n' "pattern_5_solid_single_necrosis" "pattern_1_3_well_formed" ;;
    p4_to_well) printf '%s\t%s\n' "pattern_4_cribriform_poorly_formed_fused" "pattern_1_3_well_formed" ;;
    p5_to_p4) printf '%s\t%s\n' "pattern_5_solid_single_necrosis" "pattern_4_cribriform_poorly_formed_fused" ;;
    *) echo "[error] unknown PRAD morphology direction: $1" >&2; exit 2 ;;
  esac
}

stream_one() {
  local task_name="$1"
  local direction_name="$2"
  local runner_direction="$3"
  local region_bank="$4"
  local concept_dir="$5"
  local concept_label="$6"

  local manifest="${REFERENCE_OUT_ROOT}/manifests/${task_name}/${direction_name}/combined_manifest.json"
  local ours_root="${REFERENCE_OUT_ROOT}/generated/${task_name}/${direction_name}"
  local naive_root="${NAIVE_OUT_ROOT}/generated/${task_name}/${direction_name}"
  local out_dir="${VISUAL_METRICS_ROOT}/${task_name}/${direction_name}"
  if [[ ! -f "${manifest}" ]]; then
    echo "[skip] missing manifest: ${manifest}" >&2
    return 0
  fi
  if [[ ! -d "${ours_root}" ]]; then
    echo "[skip] missing reference generated dir: ${ours_root}" >&2
    return 0
  fi

  local concept_args=()
  if [[ -n "${concept_dir}" ]]; then
    concept_args+=(
      --concepts-json "${concept_dir}/selected_concepts.json"
      --representative-tiles-csv "${concept_dir}/representative_tiles.csv"
      --concept-class-label "${concept_label}"
      --concept-target-top-k "${CONCEPT_TARGET_TOP_K}"
      --max-concepts "${MAX_CONCEPTS}"
    )
  fi
  local keep_args=()
  if [[ "${KEEP_NAIVE_IMAGES}" == "1" ]]; then
    keep_args+=(--keep-naive-images)
  fi
  if [[ "${WRITE_PER_CELL_VISUAL}" == "1" ]]; then
    keep_args+=(--write-per-cell)
  fi

  echo "[stream] ${task_name}/${direction_name}: chunk=${STREAM_CHUNK_SIZE}, keep_naive=${KEEP_NAIVE_IMAGES}" >&2
  "${PY}" scripts/run_streamed_visual_perturbation_baseline.py \
    --task-name "${task_name}" \
    --direction-name "${direction_name}" \
    --runner-direction "${runner_direction}" \
    --region-bank-csv "${region_bank}" \
    --manifest "${manifest}" \
    --ours-root "${ours_root}" \
    --naive-root "${naive_root}" \
    --out-dir "${out_dir}" \
    --python "${PY}" \
    --device "${DEVICE}" \
    --edit-policy "${EDIT_POLICY}" \
    --sae-variant "${SAE_VARIANT}" \
    --steps "${STEPS}" \
    --patch-batch "${PATCH_BATCH}" \
    --edit-support "${EDIT_SUPPORT}" \
    --window-stride-cells "${WINDOW_STRIDE_CELLS}" \
    --window-selection-mode "${WINDOW_SELECTION_MODE}" \
    --commit-mode "${COMMIT_MODE}" \
    --output-mode "${OUTPUT_MODE}" \
    --chunk-size "${STREAM_CHUNK_SIZE}" \
    --max-runs "${MAX_RUNS}" \
    --formats "${VISUAL_FORMATS}" \
    "${concept_args[@]}" \
    "${keep_args[@]}"
}

if [[ "${RUN_NAIVE}" == "1" ]]; then
  if [[ "${STREAM_NAIVE}" == "1" ]]; then
    if [[ "${BUILD_MANIFESTS:-0}" == "1" ]]; then
      echo "[manifests] refreshing reference manifests before streamed naive scoring" >&2
      TASKS="${TASKS}" OUT_ROOT="${REFERENCE_OUT_ROOT}" RUN_EDITS=0 RUN_EVAL=0 \
        examples/paper_metrics/01_run_prediction_transition_benchmark.sh "$@"
    fi
    if has_task hnscc_hpv; then
      stream_one "hnscc_hpv" "hpv_pos_to_hpv_neg" "hpv_neg" "${HNSCC_REGION_BANK_CSV}" "" ""
      stream_one "hnscc_hpv" "hpv_neg_to_hpv_pos" "hpv_pos" "${HNSCC_REGION_BANK_CSV}" "" ""
    fi
    if has_task normal_tumor; then
      for task_name in ${NORMAL_TUMOR_TASKS}; do
        stream_one "${task_name}" "normal_to_tumor" "hpv_pos" \
          "$(normal_region_bank_path "${task_name}")" \
          "${NORMAL_CONCEPT_ROOT}/${task_name}/labels/tumor" \
          "tumor"
      done
    fi
    if has_task prad_grade_group; then
      for direction_name in ${PRAD_DIRECTIONS}; do
        target_label="$(echo "${direction_name}" | awk -F'_to_' '{print toupper($2)}')"
        stream_one "prad_grade_group" "${direction_name}" "hpv_pos" \
          "${PRAD_REGION_ROOT}/${direction_name}/region_bank.csv" \
          "${PRAD_CONCEPT_ROOT}/${target_label}" \
          "${target_label}"
      done
    fi
    if has_task prad_morphology_group; then
      for direction_name in ${PRAD_MORPH_DIRECTIONS}; do
        IFS=$'\t' read -r _source_label target_label < <(prad_morph_direction_labels "${direction_name}")
        stream_one "prad_morphology_group" "${direction_name}" "hpv_pos" \
          "${PRAD_MORPH_REGION_ROOT}/${direction_name}/region_bank.csv" \
          "${PRAD_MORPH_CONCEPT_ROOT}/${target_label}" \
          "${target_label}"
      done
    fi
  else
    echo "[naive] generating deliberately bad baseline under ${NAIVE_OUT_ROOT}" >&2
    TASKS="${TASKS}" \
    OUT_ROOT="${NAIVE_OUT_ROOT}" \
    BUILD_MANIFESTS="${BUILD_MANIFESTS:-1}" \
    RUN_EDITS=1 \
    RUN_EVAL=0 \
    EDIT_POLICY="${EDIT_POLICY}" \
    EDIT_SUPPORT="${EDIT_SUPPORT}" \
    WINDOW_STRIDE_CELLS="${WINDOW_STRIDE_CELLS}" \
    WINDOW_SELECTION_MODE="${WINDOW_SELECTION_MODE}" \
    COMMIT_MODE="${COMMIT_MODE}" \
    OUTPUT_MODE="${OUTPUT_MODE}" \
    examples/paper_metrics/01_run_prediction_transition_benchmark.sh "$@"
  fi
else
  echo "[skip] naive generation disabled; set RUN_NAIVE=1" >&2
fi

if [[ "${RUN_VISUAL_METRICS}" == "1" && ! ( "${RUN_NAIVE}" == "1" && "${STREAM_NAIVE}" == "1" && "${KEEP_NAIVE_IMAGES}" != "1" ) ]]; then
  echo "[metrics] comparing ${REFERENCE_OUT_ROOT}/generated against ${NAIVE_OUT_ROOT}/generated" >&2
  allow_args=()
  if [[ "${ALLOW_MISSING_VISUAL}" == "1" ]]; then
    allow_args+=(--allow-missing)
  fi
  if [[ -d "${REFERENCE_OUT_ROOT}/generated" ]]; then
    while IFS= read -r ours_dir; do
      rel="${ours_dir#${REFERENCE_OUT_ROOT}/generated/}"
      task_name="${rel%%/*}"
      direction="${rel#*/}"
      naive_dir="${NAIVE_OUT_ROOT}/generated/${rel}"
      manifest="${REFERENCE_OUT_ROOT}/manifests/${task_name}/${direction}/combined_manifest.json"
      if [[ ! -d "${naive_dir}" ]]; then
        echo "[skip] missing naive dir: ${naive_dir}" >&2
        continue
      fi
      if [[ ! -f "${manifest}" ]]; then
        echo "[skip] missing manifest: ${manifest}" >&2
        continue
      fi
      out_dir="${VISUAL_METRICS_ROOT}/${task_name}/${direction}"
      echo "[metrics] ${task_name}/${direction}" >&2
      "${PY}" scripts/compare_edit_visual_perturbation.py \
        --ours-root "${ours_dir}" \
        --naive-root "${naive_dir}" \
        --manifest "${manifest}" \
        --out-dir "${out_dir}" \
        --title "${task_name} ${direction}: visual perturbation" \
        --formats "${VISUAL_FORMATS}" \
        "${allow_args[@]}"
    done < <(find "${REFERENCE_OUT_ROOT}/generated" -mindepth 2 -maxdepth 2 -type d | sort)
  else
    echo "[skip] missing reference generated root: ${REFERENCE_OUT_ROOT}/generated" >&2
  fi
else
  echo "[skip] visual full-folder metrics disabled or already handled by streaming mode" >&2
fi

echo "[ok] visual perturbation comparison outputs under ${VISUAL_METRICS_ROOT}" >&2
