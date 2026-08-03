#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 \
  TASKS="hnscc_hpv normal_tumor prad_morphology_group" \
  examples/paper_metrics/08_run_full_test_streaming_benchmark.sh

This builds exact-budget cumulative-prefix manifests and streams generation through
both prediction-transition and visual/border metrics. Generated images and encoded
grids are deleted by default after metrics are written.

Important defaults:
  OUT_ROOT=paper_outputs/full_test_streaming_benchmark_v1
  BUDGETS=1,8,16,32,48,64
  RANDOM_REPEATS=5
  RUN_PLOTS=1
  MAX_SLIDES=0
  MAX_REGIONS_PER_SLIDE=5
  INCLUDE_FULL_ENDPOINT=0
  KEEP_IMAGES=0
  KEEP_ENCODED_GRIDS=0
  LEGACY_OURS_ROOT / LEGACY_NAIVE_ROOT can point to an old generated/<task>/<direction>
  tree with matching run_ids; matching run folders are copied and scored instead
  of regenerated.
EOF
  exit 0
fi

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/full_test_streaming_benchmark_v1}"
BANK_ROOT="${BANK_ROOT:-artifacts/prediction_transition_region_banks_test_only_unbalanced}"

TASKS="${TASKS:-hnscc_hpv normal_tumor prad_morphology_group}"
NORMAL_TUMOR_TASKS="${NORMAL_TUMOR_TASKS:-luad_normal_tumor coad_normal_tumor kirc_normal_tumor brca_normal_tumor}"
PRAD_MORPH_DIRECTIONS="${PRAD_MORPH_DIRECTIONS:-well_to_p4 p4_to_p5}"

BUDGETS="${BUDGETS:-1,8,16,32,48,64}"
RANDOM_REPEATS="${RANDOM_REPEATS:-5}"
SEED="${SEED:-7}"
MAX_SLIDES="${MAX_SLIDES:-0}"
MAX_REGIONS_PER_SLIDE="${MAX_REGIONS_PER_SLIDE:-5}"
MAX_REGIONS="${MAX_REGIONS:-0}"
INCLUDE_FULL_ENDPOINT="${INCLUDE_FULL_ENDPOINT:-0}"

BUILD_AUDIT="${BUILD_AUDIT:-1}"
BUILD_MANIFESTS="${BUILD_MANIFESTS:-1}"
RUN_STREAM="${RUN_STREAM:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

OURS_POLICY="${OURS_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"
NAIVE_POLICY="${NAIVE_POLICY:-configs/edit_policies/bad_naive_no_preserve_no_sliding.json}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
STEPS="${STEPS:-30}"
PATCH_BATCH="${PATCH_BATCH:-256}"
EDIT_SUPPORT="${EDIT_SUPPORT:-padded_center_2x2}"
WINDOW_STRIDE_CELLS="${WINDOW_STRIDE_CELLS:-1}"
WINDOW_SELECTION_MODE="${WINDOW_SELECTION_MODE:-overlap}"
COMMIT_MODE="${COMMIT_MODE:-full_window}"
OUTPUT_MODE="${OUTPUT_MODE:-minimal}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
MAX_RUNS="${MAX_RUNS:-0}"
SCORE_SCOPE="${SCORE_SCOPE:-local_region}"
LOCAL_SOURCE="${LOCAL_SOURCE:-source_image}"
KEEP_IMAGES="${KEEP_IMAGES:-0}"
KEEP_ENCODED_GRIDS="${KEEP_ENCODED_GRIDS:-0}"
KEEP_GALLERY="${KEEP_GALLERY:-1}"
GALLERY_REGIONS_PER_DIRECTION="${GALLERY_REGIONS_PER_DIRECTION:-2}"
GALLERY_BUDGETS="${GALLERY_BUDGETS:-1,32,64}"
WRITE_PER_CELL="${WRITE_PER_CELL:-0}"
LEGACY_OURS_ROOT="${LEGACY_OURS_ROOT:-}"
LEGACY_NAIVE_ROOT="${LEGACY_NAIVE_ROOT:-}"
PLOT_FORMATS="${PLOT_FORMATS:-png,pdf,svg}"
PLOT_REPORT_BUDGET="${PLOT_REPORT_BUDGET:-64}"

NORMAL_CLASSIFIER_ROOT="${NORMAL_CLASSIFIER_ROOT:-artifacts/classifier_training_normal_tumor}"
NORMAL_CONCEPT_ROOT="${NORMAL_CONCEPT_ROOT:-artifacts/concept_discovery_normal_tumor_relu_sae_base}"
PRAD_MORPH_CLASSIFIER_RUN_DIR="${PRAD_MORPH_CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_morphology_group}"
PRAD_MORPH_CONCEPT_ROOT="${PRAD_MORPH_CONCEPT_ROOT:-artifacts/concept_discovery_prad_morphology_group_relu_sae_base/prad_morphology_group/labels}"
HNSCC_CLASSIFIER_CKPT="${HNSCC_CLASSIFIER_CKPT:-resources/models/classifiers/hnscc_hpv/mil_split0.pt}"
PRAD_MORPH_LABEL_ORDER="pattern_1_3_well_formed,pattern_4_cribriform_poorly_formed_fused,pattern_5_solid_single_necrosis"

mkdir -p "${OUT_ROOT}/manifests" "${OUT_ROOT}/logs"

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing required file: ${path}" >&2
    echo "        ${hint}" >&2
    exit 2
  fi
}

include_full_args=()
if [[ "${INCLUDE_FULL_ENDPOINT}" == "1" ]]; then
  include_full_args+=(--include-full-endpoint)
else
  include_full_args+=(--no-include-full-endpoint)
fi

keep_args=()
if [[ "${KEEP_IMAGES}" == "1" ]]; then keep_args+=(--keep-images); fi
if [[ "${KEEP_ENCODED_GRIDS}" == "1" ]]; then keep_args+=(--keep-encoded-grids); fi
if [[ "${KEEP_GALLERY}" == "1" ]]; then keep_args+=(--keep-gallery); else keep_args+=(--no-keep-gallery); fi
if [[ "${WRITE_PER_CELL}" == "1" ]]; then keep_args+=(--write-per-cell); fi

if [[ "${BUILD_AUDIT}" == "1" ]]; then
  echo "[audit] ${OUT_ROOT}/audit" >&2
  "${PY}" scripts/build_full_test_benchmark_audit.py --out-dir "${OUT_ROOT}/audit"
fi

build_manifest() {
  local task_name="$1"; local direction="$2"; local source_label="$3"; local target_label="$4"
  local region_bank="$5"; local classifier_run_dir="$6"; local classifier_ckpt="$7"; local manifest_dir="$8"
  local classifier_args=()
  if [[ -n "${classifier_ckpt}" ]]; then classifier_args+=(--classifier-ckpt "${classifier_ckpt}"); else classifier_args+=(--classifier-run-dir "${classifier_run_dir}"); fi
  echo "[manifest] ${task_name}/${direction}" >&2
  "${PY}" scripts/build_prediction_transition_manifests.py \
    --task-name "${task_name}" \
    --direction "${direction}" \
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
    "${include_full_args[@]}" \
    "${classifier_args[@]}"
}

stream_one() {
  local task_name="$1"; local direction="$2"; local source_label="$3"; local target_label="$4"; local runner_direction="$5"
  local region_bank="$6"; local classifier_run_dir="$7"; local classifier_ckpt="$8"; local label_order="$9"
  local concept_json="${10}"; local rep_tiles="${11}"; local concept_label="${12}"
  local manifest_dir="${OUT_ROOT}/manifests/${task_name}/${direction}"
  require_file "${region_bank}" "Build/refresh full-test region banks first."
  if [[ -n "${classifier_ckpt}" ]]; then require_file "${classifier_ckpt}" "Missing classifier checkpoint."; else require_file "${classifier_run_dir}/best_model.pt" "Missing trained classifier."; fi
  if [[ "${BUILD_MANIFESTS}" == "1" ]]; then
    build_manifest "${task_name}" "${direction}" "${source_label}" "${target_label}" "${region_bank}" "${classifier_run_dir}" "${classifier_ckpt}" "${manifest_dir}"
  fi
  if [[ "${RUN_STREAM}" != "1" ]]; then
    echo "[skip] stream disabled for ${task_name}/${direction}" >&2
    return 0
  fi
  local concept_args=()
  if [[ -n "${concept_json}" ]]; then
    require_file "${concept_json}" "Missing target concept JSON."
    require_file "${rep_tiles}" "Missing representative concept tiles."
    concept_args+=(--concepts-json "${concept_json}" --representative-tiles-csv "${rep_tiles}" --concept-class-label "${concept_label}")
  fi
  local legacy_args=()
  if [[ -n "${LEGACY_OURS_ROOT}" ]]; then
    legacy_args+=(--legacy-ours-root "${LEGACY_OURS_ROOT}/${task_name}/${direction}")
  fi
  if [[ -n "${LEGACY_NAIVE_ROOT}" ]]; then
    legacy_args+=(--legacy-naive-root "${LEGACY_NAIVE_ROOT}/${task_name}/${direction}")
  fi
  echo "[stream] ${task_name}/${direction}" >&2
  "${PY}" scripts/run_streamed_full_test_benchmark.py \
    --task-name "${task_name}" \
    --direction-name "${direction}" \
    --runner-direction "${runner_direction}" \
    --target-label "${target_label}" \
    --label-order "${label_order}" \
    --region-bank-csv "${region_bank}" \
    --manifest "${manifest_dir}/combined_manifest.json" \
    --out-root "${OUT_ROOT}" \
    --classifier-run-dir "${classifier_run_dir}" \
    ${classifier_ckpt:+--classifier-ckpt "${classifier_ckpt}"} \
    --python "${PY}" \
    --device "${DEVICE}" \
    --ours-policy "${OURS_POLICY}" \
    --naive-policy "${NAIVE_POLICY}" \
    --sae-variant "${SAE_VARIANT}" \
    --steps "${STEPS}" \
    --patch-batch "${PATCH_BATCH}" \
    --edit-support "${EDIT_SUPPORT}" \
    --window-stride-cells "${WINDOW_STRIDE_CELLS}" \
    --window-selection-mode "${WINDOW_SELECTION_MODE}" \
    --commit-mode "${COMMIT_MODE}" \
    --output-mode "${OUTPUT_MODE}" \
    --chunk-size "${CHUNK_SIZE}" \
    --max-runs "${MAX_RUNS}" \
    --random-repeats "${RANDOM_REPEATS}" \
    --score-scope "${SCORE_SCOPE}" \
    --local-source "${LOCAL_SOURCE}" \
    --gallery-regions-per-direction "${GALLERY_REGIONS_PER_DIRECTION}" \
    --gallery-budgets "${GALLERY_BUDGETS}" \
    "${keep_args[@]}" \
    "${legacy_args[@]}" \
    "${concept_args[@]}"
}

prad_morph_labels() {
  case "$1" in
    well_to_p4) printf '%s\t%s\n' "pattern_1_3_well_formed" "pattern_4_cribriform_poorly_formed_fused" ;;
    p4_to_p5) printf '%s\t%s\n' "pattern_4_cribriform_poorly_formed_fused" "pattern_5_solid_single_necrosis" ;;
    *) echo "[error] unsupported PRAD morphology direction for paper default: $1" >&2; exit 2 ;;
  esac
}

for family in ${TASKS}; do
  case "${family}" in
    hnscc_hpv)
      stream_one "hnscc_hpv" "hpv_pos_to_hpv_neg" "hpv_pos" "hpv_neg" "hpv_neg" \
        "${BANK_ROOT}/hnscc_hpv/region_bank.csv" "" "${HNSCC_CLASSIFIER_CKPT}" "hpv_neg,hpv_pos" "" "" ""
      stream_one "hnscc_hpv" "hpv_neg_to_hpv_pos" "hpv_neg" "hpv_pos" "hpv_pos" \
        "${BANK_ROOT}/hnscc_hpv/region_bank.csv" "" "${HNSCC_CLASSIFIER_CKPT}" "hpv_neg,hpv_pos" "" "" ""
      ;;
    normal_tumor)
      for task_name in ${NORMAL_TUMOR_TASKS}; do
        stream_one "${task_name}" "normal_to_tumor" "normal" "tumor" "hpv_pos" \
          "${BANK_ROOT}/normal_tumor/${task_name}/region_bank.csv" \
          "${NORMAL_CLASSIFIER_ROOT}/${task_name}" "" "normal,tumor" \
          "${NORMAL_CONCEPT_ROOT}/${task_name}/labels/tumor/selected_concepts.json" \
          "${NORMAL_CONCEPT_ROOT}/${task_name}/labels/tumor/representative_tiles.csv" \
          "tumor"
      done
      ;;
    prad_morphology_group)
      for direction in ${PRAD_MORPH_DIRECTIONS}; do
        IFS=$'\t' read -r source_label target_label < <(prad_morph_labels "${direction}")
        stream_one "prad_morphology_group" "${direction}" "${source_label}" "${target_label}" "hpv_pos" \
          "${BANK_ROOT}/prad_morphology_group/${direction}/region_bank.csv" \
          "${PRAD_MORPH_CLASSIFIER_RUN_DIR}" "" "${PRAD_MORPH_LABEL_ORDER}" \
          "${PRAD_MORPH_CONCEPT_ROOT}/${target_label}/selected_concepts.json" \
          "${PRAD_MORPH_CONCEPT_ROOT}/${target_label}/representative_tiles.csv" \
          "${target_label}"
      done
      ;;
    *)
      echo "[error] unknown family: ${family}" >&2
      exit 2
      ;;
  esac
done

if [[ "${RUN_PLOTS}" == "1" ]]; then
  echo "[plots] ${OUT_ROOT}/plots" >&2
  "${PY}" scripts/plot_full_test_streaming_benchmark.py \
    --out-root "${OUT_ROOT}" \
    --report-budget "${PLOT_REPORT_BUDGET}" \
    --formats "${PLOT_FORMATS}"
fi

echo "[ok] full-test streaming benchmark root: ${OUT_ROOT}" >&2
