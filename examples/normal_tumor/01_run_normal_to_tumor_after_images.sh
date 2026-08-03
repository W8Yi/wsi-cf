#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
RUN_TAG="${RUN_TAG:-showcase_sae_more_cells_less_confident}"

# Good first pathologist-facing set. Add brca_normal_tumor if you want all four.
TASKS="${TASKS:-luad_normal_tumor coad_normal_tumor kirc_normal_tumor}"

INPUT_ROOT="${INPUT_ROOT:-artifacts/normal_tumor_inputs}"
SLIDE_ROOT="${SLIDE_ROOT:-artifacts/normal_tumor_slides}"
FEATURE_ROOT="${FEATURE_ROOT:-artifacts/normal_tumor_features}"
CLASSIFIER_ROOT="${CLASSIFIER_ROOT:-artifacts/classifier_training_normal_tumor}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_normal_tumor_${SAE_VARIANT}}"
REGION_ROOT="${REGION_ROOT:-artifacts/normal_to_tumor_regions_${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-artifacts/normal_to_tumor_after_images_${RUN_TAG}}"

REFRESH_INPUTS="${REFRESH_INPUTS:-1}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
MINE_CONCEPTS_IF_MISSING="${MINE_CONCEPTS_IF_MISSING:-1}"
REFRESH_REGIONS="${REFRESH_REGIONS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
UPDATE_PAPER_OUTPUTS="${UPDATE_PAPER_OUTPUTS:-1}"

MAX_RUNS="${MAX_RUNS:-5}"
MAX_REGIONS="${MAX_REGIONS:-${MAX_RUNS}}"
MAX_CANDIDATES_PER_SLIDE="${MAX_CANDIDATES_PER_SLIDE:-2}"
EPOCHS="${EPOCHS:-20}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-4096}"
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

stage_normal_slides_for_find_regions() {
  local task="$1"
  local project="$2"
  local source_dir="${SLIDE_ROOT}/${task}"
  local ready_dir="${SLIDE_ROOT}/.tmp/ready_buffer/slides/${project}"

  if [[ ! -d "${source_dir}" ]]; then
    echo "[error] missing downloaded normal slides for ${task}: ${source_dir}" >&2
    echo "        Download first, e.g. ALL=1 TASK=${task} bash examples/concept_discovery/03_download_normal_tumor_normal_slides.sh" >&2
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

write_attention_aware_concept_task() {
  local task="$1"
  local classifier_dir="$2"
  local src_json="${INPUT_ROOT}/${task}/concept_task.json"
  local out_json="${INPUT_ROOT}/${task}/concept_task.attention_aware.${SAE_VARIANT}.json"

  "${PY}" - "${src_json}" "${out_json}" "${classifier_dir}" "${SAE_VARIANT}" <<'PY'
import json
import sys
from pathlib import Path

src, out, classifier, sae_variant = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
payload = json.loads(src.read_text())
payload["ranking_mode"] = "attention_aware_optional"
payload["classifier_run_dir"] = classifier
payload["sae_variant"] = sae_variant
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
PY
}

maybe_refresh_inputs() {
  if [[ "${REFRESH_INPUTS}" != "1" ]]; then
    return
  fi
  local tasks_csv="${TASKS// /,}"
  echo "[prep] refreshing normal/tumor inputs for: ${TASKS}" >&2
  PY="${PY}" OUT_ROOT="${INPUT_ROOT}" NORMAL_FEATURES_ROOT="${FEATURE_ROOT}" \
    bash examples/concept_discovery/02_prepare_normal_tumor_tasks.sh \
    --tasks "${tasks_csv}"
}

maybe_train_classifier() {
  local task="$1"
  local classifier_dir="${CLASSIFIER_ROOT}/${task}"
  if [[ -f "${classifier_dir}/best_model.pt" ]]; then
    echo "[skip] classifier exists: ${classifier_dir}/best_model.pt" >&2
    return
  fi
  if [[ "${TRAIN_IF_MISSING}" != "1" ]]; then
    echo "[error] missing classifier: ${classifier_dir}/best_model.pt" >&2
    echo "        Set TRAIN_IF_MISSING=1 or train it first." >&2
    exit 2
  fi
  echo "[train] normal/tumor classifier: ${task}" >&2
  PY="${PY}" TASKS="${task}" INPUT_ROOT="${INPUT_ROOT}" OUT_DIR="${CLASSIFIER_ROOT}" \
    DEVICE="${DEVICE}" EPOCHS="${EPOCHS}" MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE}" \
    bash examples/classifier_training/10_train_all_normal_tumor.sh
}

maybe_mine_concepts() {
  local task="$1"
  local classifier_dir="${CLASSIFIER_ROOT}/${task}"
  local concept_dir="${CONCEPT_OUT}/${task}/labels/tumor"
  if [[ -f "${concept_dir}/selected_concepts.json" && -f "${concept_dir}/representative_tiles.csv" ]]; then
    echo "[skip] tumor concepts exist: ${concept_dir}" >&2
    return
  fi
  if [[ "${MINE_CONCEPTS_IF_MISSING}" != "1" ]]; then
    echo "[error] missing tumor concept bundle: ${concept_dir}" >&2
    echo "        Set MINE_CONCEPTS_IF_MISSING=1 or mine concepts first." >&2
    exit 2
  fi
  local task_json
  task_json="$(write_attention_aware_concept_task "${task}" "${classifier_dir}")"
  echo "[concepts] mining tumor concepts: ${task}" >&2
  "${PY}" scripts/find_label_concepts.py \
    --task-json "${task_json}" \
    --out-root "${CONCEPT_OUT}" \
    --device "${DEVICE}" \
    --skip-existing
}

find_normal_regions() {
  local task="$1"
  local project="$2"
  local classifier_dir="${CLASSIFIER_ROOT}/${task}"
  local region_dir="${REGION_ROOT}/${task}"

  if [[ "${REFRESH_REGIONS}" != "1" && -f "${region_dir}/region_bank.csv" && -f "${region_dir}/progressive_edit_manifest.json" ]]; then
    echo "[skip] regions exist: ${region_dir}" >&2
    return
  fi

  stage_normal_slides_for_find_regions "${task}" "${project}"
  echo "[regions] normal -> tumor source regions: ${task}" >&2
  "${PY}" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir "${classifier_dir}" \
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

run_after_images() {
  local task="$1"
  local region_dir="${REGION_ROOT}/${task}"
  local concept_dir="${CONCEPT_OUT}/${task}/labels/tumor"
  local out_dir="${OUT_ROOT}/${task}"
  local skip_args=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi

  echo "[generate] ${task}: normal -> tumor after-images" >&2
  local steer_support_args=()
  if [[ "${STEER_FULL_SUPPORT}" == "1" ]]; then
    steer_support_args+=(--steer-full-support)
  fi
  "${PY}" scripts/run_progressive_region_edit.py \
    --task "${task}" \
    --region-bank-csv "${region_dir}/region_bank.csv" \
    --edit-manifest "${region_dir}/progressive_edit_manifest.json" \
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

maybe_refresh_inputs
for task in ${TASKS}; do
  project="$(task_project "${task}")"
  maybe_train_classifier "${task}"
  maybe_mine_concepts "${task}"
  find_normal_regions "${task}" "${project}"
  run_after_images "${task}"
done

if [[ "${UPDATE_PAPER_OUTPUTS}" == "1" ]]; then
  echo "[paper] refreshing paper_outputs/current symlink view" >&2
  "${PY}" paper/artifacts.py scan
  "${PY}" paper/artifacts.py make-paper-view
fi

echo "[ok] normal -> tumor after-images: ${OUT_ROOT}" >&2
