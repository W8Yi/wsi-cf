#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
RUN_TAG="${RUN_TAG:-showcase_sae_cell_fraction_sweep}"
TASKS="${TASKS:-brca_normal_tumor kirc_normal_tumor luad_normal_tumor coad_normal_tumor}"
INCLUDE_LUSC="${INCLUDE_LUSC:-0}"

INPUT_ROOT="${INPUT_ROOT:-artifacts/normal_tumor_inputs}"
SLIDE_ROOT="${SLIDE_ROOT:-artifacts/normal_tumor_slides}"
FEATURE_ROOT="${FEATURE_ROOT:-artifacts/normal_tumor_features}"
CLASSIFIER_ROOT="${CLASSIFIER_ROOT:-artifacts/classifier_training_normal_tumor}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_normal_tumor_${SAE_VARIANT}}"
REGION_ROOT="${REGION_ROOT:-artifacts/normal_to_tumor_regions_${RUN_TAG}_diverse}"
MANIFEST_ROOT="${MANIFEST_ROOT:-artifacts/normal_to_tumor_cell_fraction_manifests_${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/normal_to_tumor_after_images_${RUN_TAG}}"

REFRESH_INPUTS="${REFRESH_INPUTS:-0}"
REFRESH_REGIONS="${REFRESH_REGIONS:-1}"
REFRESH_MANIFESTS="${REFRESH_MANIFESTS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
UPDATE_PAPER_OUTPUTS="${UPDATE_PAPER_OUTPUTS:-1}"

FRACTIONS="${FRACTIONS:-0.5,0.65,0.8,1.0}"
MAX_RUNS="${MAX_RUNS:-0}"
MAX_REGIONS="${MAX_REGIONS:-5}"
MAX_CANDIDATES_PER_SLIDE="${MAX_CANDIDATES_PER_SLIDE:-1}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
MAX_CONCEPTS="${MAX_CONCEPTS:-3}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"
EDIT_SUPPORT="${EDIT_SUPPORT:-padded_center_2x2}"
WINDOW_STRIDE_CELLS="${WINDOW_STRIDE_CELLS:-1}"
WINDOW_SELECTION_MODE="${WINDOW_SELECTION_MODE:-overlap}"
STEER_FULL_SUPPORT="${STEER_FULL_SUPPORT:-0}"
COMMIT_MODE="${COMMIT_MODE:-full_window}"
COMMIT_FEATHER_PX="${COMMIT_FEATHER_PX:-0}"
COMMIT_HALO_CELLS="${COMMIT_HALO_CELLS:-0}"
COMMIT_HALO_ALPHA="${COMMIT_HALO_ALPHA:-0.0}"
ATTENTION_PERCENTILE="${ATTENTION_PERCENTILE:-50}"
MIN_SELECTED_CELLS="${MIN_SELECTED_CELLS:-16}"
MAX_SELECTED_CELLS="${MAX_SELECTED_CELLS:-32}"
TARGET_IMPORTANCE_MASS="${TARGET_IMPORTANCE_MASS:-0.90}"
MIN_LABEL_CONFIDENCE="${MIN_LABEL_CONFIDENCE:-0.50}"
MAX_LABEL_CONFIDENCE="${MAX_LABEL_CONFIDENCE:-0.9995}"
MIN_TISSUE="${MIN_TISSUE:-0.35}"
MIN_DARK_FRACTION="${MIN_DARK_FRACTION:-0.02}"
MIN_SATURATION_FRACTION="${MIN_SATURATION_FRACTION:-0.02}"

task_project() {
  case "$1" in
    luad_normal_tumor) echo "TCGA-LUAD" ;;
    coad_normal_tumor) echo "TCGA-COAD" ;;
    brca_normal_tumor) echo "TCGA-BRCA" ;;
    kirc_normal_tumor) echo "TCGA-KIRC" ;;
    lusc_normal_tumor) echo "TCGA-LUSC" ;;
    *) echo "[error] unknown normal/tumor task: $1" >&2; exit 2 ;;
  esac
}

has_h5() {
  local project="$1"
  find "${FEATURE_ROOT}/${project}/features_uni2" -maxdepth 1 -type f -name '*.h5' -print -quit 2>/dev/null | grep -q .
}

task_ready_for_sweep() {
  local task="$1"
  local project="$2"
  [[ -f "${INPUT_ROOT}/${task}/concept_task.json" ]] || return 1
  [[ -d "${SLIDE_ROOT}/${task}" ]] || return 1
  has_h5 "${project}" || return 1
  [[ -f "${CLASSIFIER_ROOT}/${task}/best_model.pt" ]] || return 1
  [[ -f "${CONCEPT_OUT}/${task}/labels/tumor/selected_concepts.json" ]] || return 1
  [[ -f "${CONCEPT_OUT}/${task}/labels/tumor/representative_tiles.csv" ]] || return 1
}

stage_normal_slides_for_find_regions() {
  local task="$1"
  local project="$2"
  local source_dir="${SLIDE_ROOT}/${task}"
  local ready_dir="${SLIDE_ROOT}/.tmp/ready_buffer/slides/${project}"

  if [[ ! -d "${source_dir}" ]]; then
    echo "[error] missing downloaded normal slides for ${task}: ${source_dir}" >&2
    exit 2
  fi

  rm -rf "${ready_dir}"
  mkdir -p "${ready_dir}"
  local count=0
  while IFS= read -r -d '' slide; do
    ln -s "$(realpath "${slide}")" "${ready_dir}/$(basename "${slide}")"
    count=$((count + 1))
  done < <(find "${source_dir}" -type f -iname '*.svs' -print0 | sort -z)
  if [[ "${count}" -eq 0 ]]; then
    echo "[error] no .svs files found under ${source_dir}" >&2
    exit 2
  fi
  echo "[slides] ${task}: staged ${count} normal SVS symlinks for ${project}" >&2
}

maybe_prepare_inputs() {
  if [[ "${REFRESH_INPUTS}" != "1" ]]; then
    return
  fi
  local tasks_csv="${TASKS// /,}"
  echo "[prep] refreshing normal/tumor inputs for: ${TASKS}" >&2
  PY="${PY}" OUT_ROOT="${INPUT_ROOT}" NORMAL_FEATURES_ROOT="${FEATURE_ROOT}" \
    bash examples/concept_discovery/02_prepare_normal_tumor_tasks.sh \
    --tasks "${tasks_csv}"
}

find_regions_for_task() {
  local task="$1"
  local project="$2"
  local region_dir="${REGION_ROOT}/${task}"
  if [[ "${REFRESH_REGIONS}" != "1" && -f "${region_dir}/region_bank.csv" ]]; then
    echo "[skip] regions exist: ${region_dir}" >&2
    return
  fi
  stage_normal_slides_for_find_regions "${task}" "${project}"
  echo "[regions] ${task}: diverse normal regions, max ${MAX_CANDIDATES_PER_SLIDE}/slide" >&2
  "${PY}" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir "${CLASSIFIER_ROOT}/${task}" \
    --source-label normal \
    --target-label tumor \
    --slides-root "${SLIDE_ROOT}" \
    --target-magnification 20 \
    --region-size 2048 \
    --grid-step-px 256 \
    --max-regions "${MAX_REGIONS}" \
    --max-candidates-per-slide "${MAX_CANDIDATES_PER_SLIDE}" \
    --attention-percentile "${ATTENTION_PERCENTILE}" \
    --min-selected-cells "${MIN_SELECTED_CELLS}" \
    --max-selected-cells "${MAX_SELECTED_CELLS}" \
    --target-importance-mass "${TARGET_IMPORTANCE_MASS}" \
    --min-label-confidence "${MIN_LABEL_CONFIDENCE}" \
    --max-label-confidence "${MAX_LABEL_CONFIDENCE}" \
    --min-tissue "${MIN_TISSUE}" \
    --min-dark-fraction "${MIN_DARK_FRACTION}" \
    --min-saturation-fraction "${MIN_SATURATION_FRACTION}" \
    --out-dir "${region_dir}" \
    --sae-variant "${SAE_VARIANT}" \
    --device "${DEVICE}"
}

build_fraction_manifests() {
  local task="$1"
  local manifest_dir="${MANIFEST_ROOT}/${task}"
  if [[ "${REFRESH_MANIFESTS}" != "1" && -f "${manifest_dir}/all_fractions_manifest.json" ]]; then
    echo "[skip] fraction manifests exist: ${manifest_dir}" >&2
    return
  fi
  echo "[manifests] ${task}: fractions ${FRACTIONS}" >&2
  "${PY}" scripts/build_normal_tumor_cell_fraction_manifests.py \
    --region-bank-csv "${REGION_ROOT}/${task}/region_bank.csv" \
    --classifier-run-dir "${CLASSIFIER_ROOT}/${task}" \
    --out-dir "${manifest_dir}" \
    --fractions "${FRACTIONS}" \
    --device "${DEVICE}"
}

run_sweep_for_task() {
  local task="$1"
  local manifest_dir="${MANIFEST_ROOT}/${task}"
  local concept_dir="${CONCEPT_OUT}/${task}/labels/tumor"
  local out_dir="${OUT_ROOT}/${task}"
  local skip_args=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi
  echo "[generate] ${task}: cell-fraction sweep -> ${out_dir}" >&2
  local steer_support_args=()
  if [[ "${STEER_FULL_SUPPORT}" == "1" ]]; then
    steer_support_args+=(--steer-full-support)
  fi
  "${PY}" scripts/run_progressive_region_edit.py \
    --task "${task}" \
    --region-bank-csv "${REGION_ROOT}/${task}/region_bank.csv" \
    --edit-manifest "${manifest_dir}/all_fractions_manifest.json" \
    --out-dir "${out_dir}" \
    --edit-policy "${EDIT_POLICY}" \
    --edit-support "${EDIT_SUPPORT}" \
    --window-stride-cells "${WINDOW_STRIDE_CELLS}" \
    --window-selection-mode "${WINDOW_SELECTION_MODE}" \
    --commit-mode "${COMMIT_MODE}" \
    --commit-feather-px "${COMMIT_FEATHER_PX}" \
    --commit-halo-cells "${COMMIT_HALO_CELLS}" \
    --commit-halo-alpha "${COMMIT_HALO_ALPHA}" \
    --concepts-json "${concept_dir}/selected_concepts.json" \
    --representative-tiles-csv "${concept_dir}/representative_tiles.csv" \
    --concept-class-label tumor \
    --concept-ranking-method attention_weighted \
    --concept-target-stat median \
    --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
    --concept-steering-mode prototype_vector \
    --max-concepts "${MAX_CONCEPTS}" \
    --max-runs "${MAX_RUNS}" \
    --target-magnification 20 \
    --sae-variant "${SAE_VARIANT}" \
    --output-mode "${OUTPUT_MODE}" \
    "${steer_support_args[@]}" \
    "${skip_args[@]}" \
    --device "${DEVICE}"
}

if [[ "${INCLUDE_LUSC}" == "1" && " ${TASKS} " != *" lusc_normal_tumor "* ]]; then
  TASKS="${TASKS} lusc_normal_tumor"
fi

maybe_prepare_inputs
mkdir -p "${OUT_ROOT}"

for task in ${TASKS}; do
  project="$(task_project "${task}")"
  if ! task_ready_for_sweep "${task}" "${project}"; then
    if [[ "${task}" == "lusc_normal_tumor" ]]; then
      echo "[skip] lusc_normal_tumor is not ready; needs inputs, normal slides, UNI2 H5, classifier, and tumor concepts." >&2
      continue
    fi
    echo "[error] ${task} is not ready for sweep." >&2
    echo "        Expected inputs under ${INPUT_ROOT}/${task}, ${SLIDE_ROOT}/${task}, ${FEATURE_ROOT}/${project}, ${CLASSIFIER_ROOT}/${task}, and ${CONCEPT_OUT}/${task}/labels/tumor." >&2
    exit 2
  fi
  find_regions_for_task "${task}" "${project}"
  build_fraction_manifests "${task}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${task}: would run ${MANIFEST_ROOT}/${task}/all_fractions_manifest.json -> ${OUT_ROOT}/${task}" >&2
  else
    run_sweep_for_task "${task}"
  fi
done

if [[ "${UPDATE_PAPER_OUTPUTS}" == "1" ]]; then
  echo "[paper] refreshing paper_outputs/current symlink view" >&2
  "${PY}" paper/artifacts.py scan
  "${PY}" paper/artifacts.py make-paper-view
fi

echo "[ok] normal-to-tumor cell-fraction sweep output root: ${OUT_ROOT}" >&2
