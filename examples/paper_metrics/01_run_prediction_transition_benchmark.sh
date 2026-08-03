#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-configs/metric_benchmarks/prediction_transition_balanced.json}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/prediction_transition_benchmark}"
TASKS="${TASKS:-hnscc_hpv normal_tumor prad_morphology_group}"
NORMAL_TUMOR_TASKS="${NORMAL_TUMOR_TASKS:-luad_normal_tumor coad_normal_tumor kirc_normal_tumor brca_normal_tumor}"
NORMAL_TUMOR_DIRECTIONS="${NORMAL_TUMOR_DIRECTIONS:-normal_to_tumor}"
PRAD_DIRECTIONS="${PRAD_DIRECTIONS:-gg1_to_gg2 gg2_to_gg3 gg3_to_gg4 gg4_to_gg5 gg1_to_gg5 gg5_to_gg1}"
PRAD_MORPH_DIRECTIONS="${PRAD_MORPH_DIRECTIONS:-well_to_p4 p4_to_p5 well_to_p5 p5_to_well}"

BUDGETS="${BUDGETS:-1,2,4,8,16,32,48,64}"
RANDOM_REPEATS="${RANDOM_REPEATS:-5}"
SEED="${SEED:-7}"
INCLUDE_FULL_ENDPOINT="${INCLUDE_FULL_ENDPOINT:-1}"
MAX_SLIDES="${MAX_SLIDES:-20}"
MAX_REGIONS_PER_SLIDE="${MAX_REGIONS_PER_SLIDE:-5}"
MAX_REGIONS="${MAX_REGIONS:-0}"

EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
OUTPUT_MODE="${OUTPUT_MODE:-minimal}"
STEPS="${STEPS:-30}"
PATCH_BATCH="${PATCH_BATCH:-256}"
EDIT_SUPPORT="${EDIT_SUPPORT:-padded_center_2x2}"
WINDOW_STRIDE_CELLS="${WINDOW_STRIDE_CELLS:-1}"
WINDOW_SELECTION_MODE="${WINDOW_SELECTION_MODE:-overlap}"
COMMIT_MODE="${COMMIT_MODE:-full_window}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
MAX_CONCEPTS="${MAX_CONCEPTS:-3}"

BUILD_MANIFESTS="${BUILD_MANIFESTS:-1}"
RUN_EDITS="${RUN_EDITS:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ALLOW_MISSING_EVAL="${ALLOW_MISSING_EVAL:-0}"
FORCE_REENCODE="${FORCE_REENCODE:-0}"
SCORE_SCOPE="${SCORE_SCOPE:-local_region}"
LOCAL_SOURCE="${LOCAL_SOURCE:-source_image}"

HNSCC_CLASSIFIER_RUN_DIR="${HNSCC_CLASSIFIER_RUN_DIR:-}"
HNSCC_CLASSIFIER_CKPT="${HNSCC_CLASSIFIER_CKPT:-resources/models/classifiers/hnscc_hpv/mil_split0.pt}"
HNSCC_REGION_BANK_CSV="${HNSCC_REGION_BANK_CSV:-artifacts/hnscc_hpv_paper_benchmark/regions/region_bank.csv}"
NORMAL_CLASSIFIER_ROOT="${NORMAL_CLASSIFIER_ROOT:-artifacts/classifier_training_normal_tumor}"
NORMAL_CONCEPT_ROOT="${NORMAL_CONCEPT_ROOT:-artifacts/concept_discovery_normal_tumor_relu_sae_base}"
NORMAL_REGION_ROOT="${NORMAL_REGION_ROOT:-artifacts/normal_to_tumor_regions_showcase_sae_cell_fraction_sweep_padded_center_commit_rerun_diverse}"
NORMAL_REGION_FALLBACK_ROOTS="${NORMAL_REGION_FALLBACK_ROOTS:-artifacts/normal_to_tumor_regions_showcase_sae_cell_fraction_sweep_diverse artifacts/normal_to_tumor_regions}"
PRAD_CLASSIFIER_RUN_DIR="${PRAD_CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_grade_group}"
PRAD_CONCEPT_ROOT="${PRAD_CONCEPT_ROOT:-artifacts/concept_discovery_prad_grade_group_relu_sae_base/prad_grade_group/labels}"
PRAD_REGION_ROOT="${PRAD_REGION_ROOT:-paper_outputs/prad_gleason_grade_to_grade_steering/_regions}"
PRAD_MORPH_CLASSIFIER_RUN_DIR="${PRAD_MORPH_CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_morphology_group}"
PRAD_MORPH_CONCEPT_ROOT="${PRAD_MORPH_CONCEPT_ROOT:-artifacts/concept_discovery_prad_morphology_group_relu_sae_base/prad_morphology_group/labels}"
PRAD_MORPH_REGION_ROOT="${PRAD_MORPH_REGION_ROOT:-paper_outputs/prad_morphology_group_steering/_regions}"
PRAD_MORPH_LABEL_ORDER="pattern_1_3_well_formed,pattern_4_cribriform_poorly_formed_fused,pattern_5_solid_single_necrosis"

mkdir -p "${OUT_ROOT}/manifests" "${OUT_ROOT}/generated" "${OUT_ROOT}/metrics" "${OUT_ROOT}/logs"

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing required file: ${path}" >&2
    echo "        ${hint}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local hint="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[error] missing required directory: ${path}" >&2
    echo "        ${hint}" >&2
    exit 2
  fi
}

normal_region_bank_path() {
  local task_name="$1"
  local direction_name="${2:-normal_to_tumor}"
  local preferred
  if [[ "${direction_name}" == "normal_to_tumor" ]]; then
    preferred="${NORMAL_REGION_ROOT}/${task_name}/region_bank.csv"
  else
    preferred="${NORMAL_REGION_ROOT}/${task_name}/${direction_name}/region_bank.csv"
  fi
  if [[ -f "${preferred}" ]]; then
    printf '%s\n' "${preferred}"
    return 0
  fi
  local root
  for root in ${NORMAL_REGION_FALLBACK_ROOTS}; do
    local candidate
    if [[ "${direction_name}" == "normal_to_tumor" ]]; then
      candidate="${root}/${task_name}/region_bank.csv"
    else
      candidate="${root}/${task_name}/${direction_name}/region_bank.csv"
    fi
    if [[ -f "${candidate}" ]]; then
      echo "[warn] ${task_name}/${direction_name}: using fallback region bank ${candidate}" >&2
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${preferred}"
  return 1
}

normal_tumor_direction_labels() {
  case "$1" in
    normal_to_tumor) printf '%s\t%s\t%s\n' "normal" "tumor" "hpv_pos" ;;
    tumor_to_normal) printf '%s\t%s\t%s\n' "tumor" "normal" "hpv_neg" ;;
    *) echo "[error] unknown normal/tumor direction: $1" >&2; exit 2 ;;
  esac
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

run_progressive() {
  local task_name="$1"
  local direction_name="$2"
  local runner_direction="$3"
  local region_bank="$4"
  local manifest="$5"
  local generated_root="$6"
  local concept_dir="$7"
  local concept_label="$8"

  local skip_args=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi

  local concept_args=()
  if [[ -n "${concept_dir}" ]]; then
    concept_args+=(
      --concepts-json "${concept_dir}/selected_concepts.json"
      --representative-tiles-csv "${concept_dir}/representative_tiles.csv"
      --concept-class-label "${concept_label}"
      --concept-ranking-method attention_weighted
      --concept-target-stat median
      --concept-target-top-k "${CONCEPT_TARGET_TOP_K}"
      --concept-steering-mode prototype_vector
      --max-concepts "${MAX_CONCEPTS}"
    )
  fi

  echo "[generate] ${task_name}/${direction_name} -> ${generated_root}" >&2
  "${PY}" scripts/run_progressive_region_edit.py \
    --task "${task_name}" \
    --region-bank-csv "${region_bank}" \
    --edit-manifest "${manifest}" \
    --out-dir "${generated_root}" \
    --edit-policy "${EDIT_POLICY}" \
    --direction "${runner_direction}" \
    --target-magnification 20 \
    --sae-variant "${SAE_VARIANT}" \
    --steps "${STEPS}" \
    --patch-batch "${PATCH_BATCH}" \
    --edit-support "${EDIT_SUPPORT}" \
    --window-stride-cells "${WINDOW_STRIDE_CELLS}" \
    --window-selection-mode "${WINDOW_SELECTION_MODE}" \
    --commit-mode "${COMMIT_MODE}" \
    --output-mode "${OUTPUT_MODE}" \
    --device "${DEVICE}" \
    "${concept_args[@]}" \
    "${skip_args[@]}"
}

build_manifest() {
  local task_name="$1"
  local direction_name="$2"
  local source_label="$3"
  local target_label="$4"
  local region_bank="$5"
  local classifier_run_dir="$6"
  local classifier_ckpt="$7"
  local manifest_dir="$8"
  local classifier_args=()
  if [[ -n "${classifier_ckpt}" ]]; then
    classifier_args+=(--classifier-ckpt "${classifier_ckpt}")
  else
    classifier_args+=(--classifier-run-dir "${classifier_run_dir}")
  fi
  local full_endpoint_args=()
  if [[ "${INCLUDE_FULL_ENDPOINT}" == "0" ]]; then
    full_endpoint_args+=(--no-include-full-endpoint)
  else
    full_endpoint_args+=(--include-full-endpoint)
  fi

  echo "[manifests] ${task_name}/${direction_name}" >&2
  "${PY}" scripts/build_prediction_transition_manifests.py \
    --task-name "${task_name}" \
    --direction "${direction_name}" \
    --source-label "${source_label}" \
    --target-label "${target_label}" \
    --region-bank-csv "${region_bank}" \
    --out-dir "${manifest_dir}" \
    --budgets "${BUDGETS}" \
    --random-repeats "${RANDOM_REPEATS}" \
    --seed "${SEED}" \
    --max-slides "${MAX_SLIDES}" \
    --max-regions-per-slide "${MAX_REGIONS_PER_SLIDE}" \
    --max-regions "${MAX_REGIONS}" \
    --run-prefix "predtrans" \
    --device "${DEVICE}" \
    "${full_endpoint_args[@]}" \
    "${classifier_args[@]}"
}

score_edits() {
  local task_name="$1"
  local direction_name="$2"
  local target_label="$3"
  local region_bank="$4"
  local classifier_run_dir="$5"
  local classifier_ckpt="$6"
  local manifest="$7"
  local generated_root="$8"
  local label_order="$9"
  local out_dir="${10}"

  local missing_args=()
  if [[ "${ALLOW_MISSING_EVAL}" == "1" ]]; then
    missing_args+=(--allow-missing)
  fi
  if [[ "${FORCE_REENCODE}" == "1" ]]; then
    missing_args+=(--force-reencode)
  fi
  local classifier_args=()
  if [[ -n "${classifier_ckpt}" ]]; then
    classifier_args+=(--classifier-ckpt "${classifier_ckpt}")
  else
    classifier_args+=(--classifier-run-dir "${classifier_run_dir}")
  fi

  echo "[evaluate] ${task_name}/${direction_name}" >&2
  "${PY}" scripts/evaluate_prediction_transition_edits.py \
    --task-name "${task_name}" \
    --direction "${direction_name}" \
    --target-label "${target_label}" \
    --region-bank-csv "${region_bank}" \
    --edit-manifest "${manifest}" \
    --generated-root "${generated_root}" \
    --out-dir "${out_dir}" \
    --label-order "${label_order}" \
    --score-scope "${SCORE_SCOPE}" \
    --local-source "${LOCAL_SOURCE}" \
    --device "${DEVICE}" \
    "${classifier_args[@]}" \
    "${missing_args[@]}"
}

run_one() {
  local task_name="$1"
  local direction_name="$2"
  local source_label="$3"
  local target_label="$4"
  local runner_direction="$5"
  local region_bank="$6"
  local classifier_run_dir="$7"
  local classifier_ckpt="$8"
  local concept_dir="$9"
  local concept_label="${10}"
  local label_order="${11}"

  require_file "${region_bank}" "Create candidate regions first with scripts/find_regions.py or the task wrapper."
  if [[ -n "${classifier_ckpt}" ]]; then
    require_file "${classifier_ckpt}" "Provide a classifier checkpoint with HNSCC_CLASSIFIER_CKPT or the task-specific override."
  else
    require_file "${classifier_run_dir}/best_model.pt" "Train the classifier first."
  fi
  if [[ -n "${concept_dir}" ]]; then
    require_file "${concept_dir}/selected_concepts.json" "Refresh concept discovery for this target label."
    require_file "${concept_dir}/representative_tiles.csv" "Refresh representative concept tiles for this target label."
  fi

  local manifest_dir="${OUT_ROOT}/manifests/${task_name}/${direction_name}"
  local generated_root="${OUT_ROOT}/generated/${task_name}/${direction_name}"
  local metrics_dir="${OUT_ROOT}/metrics/${task_name}/${direction_name}"
  if [[ "${BUILD_MANIFESTS}" == "1" ]]; then
    build_manifest "${task_name}" "${direction_name}" "${source_label}" "${target_label}" "${region_bank}" "${classifier_run_dir}" "${classifier_ckpt}" "${manifest_dir}"
  else
    require_file "${manifest_dir}/combined_manifest.json" "Set BUILD_MANIFESTS=1 or provide existing benchmark manifests."
  fi
  if [[ "${RUN_EDITS}" == "1" ]]; then
    run_progressive "${task_name}" "${direction_name}" "${runner_direction}" "${region_bank}" "${manifest_dir}/combined_manifest.json" "${generated_root}" "${concept_dir}" "${concept_label}"
  else
    echo "[skip] generation disabled for ${task_name}/${direction_name}; set RUN_EDITS=1" >&2
  fi
  if [[ "${RUN_EVAL}" == "1" ]]; then
    score_edits "${task_name}" "${direction_name}" "${target_label}" "${region_bank}" "${classifier_run_dir}" "${classifier_ckpt}" "${manifest_dir}/combined_manifest.json" "${generated_root}" "${label_order}" "${metrics_dir}"
  else
    echo "[skip] evaluation disabled for ${task_name}/${direction_name}; set RUN_EVAL=1" >&2
  fi
}

for family in ${TASKS}; do
  case "${family}" in
    hnscc_hpv)
      run_one "hnscc_hpv" "hpv_pos_to_hpv_neg" "hpv_pos" "hpv_neg" "hpv_neg" "${HNSCC_REGION_BANK_CSV}" "${HNSCC_CLASSIFIER_RUN_DIR}" "${HNSCC_CLASSIFIER_CKPT}" "" "" "hpv_neg,hpv_pos"
      run_one "hnscc_hpv" "hpv_neg_to_hpv_pos" "hpv_neg" "hpv_pos" "hpv_pos" "${HNSCC_REGION_BANK_CSV}" "${HNSCC_CLASSIFIER_RUN_DIR}" "${HNSCC_CLASSIFIER_CKPT}" "" "" "hpv_neg,hpv_pos"
      ;;
    normal_tumor)
      for task_name in ${NORMAL_TUMOR_TASKS}; do
        for direction_name in ${NORMAL_TUMOR_DIRECTIONS}; do
          IFS=$'\t' read -r source_label target_label runner_direction < <(normal_tumor_direction_labels "${direction_name}")
          region_bank="$(normal_region_bank_path "${task_name}" "${direction_name}")" || true
          run_one "${task_name}" "${direction_name}" "${source_label}" "${target_label}" "${runner_direction}" \
            "${region_bank}" \
            "${NORMAL_CLASSIFIER_ROOT}/${task_name}" \
            "" \
            "${NORMAL_CONCEPT_ROOT}/${task_name}/labels/${target_label}" \
            "${target_label}" \
            "normal,tumor"
        done
      done
      ;;
    prad_grade_group)
      for direction_name in ${PRAD_DIRECTIONS}; do
        source_label="$(echo "${direction_name}" | awk -F'_to_' '{print toupper($1)}')"
        target_label="$(echo "${direction_name}" | awk -F'_to_' '{print toupper($2)}')"
        run_one "prad_grade_group" "${direction_name}" "${source_label}" "${target_label}" "hpv_pos" \
          "${PRAD_REGION_ROOT}/${direction_name}/region_bank.csv" \
          "${PRAD_CLASSIFIER_RUN_DIR}" \
          "" \
          "${PRAD_CONCEPT_ROOT}/${target_label}" \
          "${target_label}" \
          "GG1,GG2,GG3,GG4,GG5"
      done
      ;;
    prad_morphology_group)
      for direction_name in ${PRAD_MORPH_DIRECTIONS}; do
        IFS=$'\t' read -r source_label target_label < <(prad_morph_direction_labels "${direction_name}")
        run_one "prad_morphology_group" "${direction_name}" "${source_label}" "${target_label}" "hpv_pos" \
          "${PRAD_MORPH_REGION_ROOT}/${direction_name}/region_bank.csv" \
          "${PRAD_MORPH_CLASSIFIER_RUN_DIR}" \
          "" \
          "${PRAD_MORPH_CONCEPT_ROOT}/${target_label}" \
          "${target_label}" \
          "${PRAD_MORPH_LABEL_ORDER}"
      done
      ;;
    *)
      echo "[error] unknown benchmark family: ${family}" >&2
      exit 2
      ;;
  esac
done

echo "[ok] prediction-transition benchmark prepared under ${OUT_ROOT}" >&2
