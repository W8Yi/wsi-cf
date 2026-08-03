#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  # Build the small paper subset only, no GPU generation.
  RUN_BUILD=1 RUN_OURS=0 RUN_NAIVE=0 RUN_METRICS=0 \
    examples/paper_metrics/06_run_visual_perturbation_paper_subset.sh

  # Generate ours + bad naive + visual/border metrics for the recommended paper subset.
  CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 RUN_BUILD=1 RUN_OURS=1 RUN_NAIVE=1 RUN_METRICS=1 \
    examples/paper_metrics/06_run_visual_perturbation_paper_subset.sh

Defaults:
  Tasks: hnscc_hpv/hpv_pos_to_hpv_neg, luad_normal_tumor/normal_to_tumor,
         prad_morphology_group/p4_to_p5
  Budgets: 1,8,16,32
  Regions per direction: 4
  Slide-balanced mode: set SLIDES_PER_DIRECTION=5 REGIONS_PER_SLIDE=4
  Our policy: configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json
  Naive policy: configs/edit_policies/bad_naive_no_preserve_no_sliding.json
  Output: paper_outputs/visual_perturbation_paper_subset

Environment:
  TASK_DIRECTIONS        Space-separated task/direction keys.
  BUDGETS                Comma-separated budgets.
  REGIONS_PER_DIRECTION  Number of complete regions per direction.
  SLIDES_PER_DIRECTION   Optional number of slides per direction.
  REGIONS_PER_SLIDE      Optional number of regions per selected slide.
  REGION_SORT            first or best_delta.
  RUN_BUILD              Build subset manifests.
  RUN_OURS               Regenerate our method images.
  RUN_NAIVE              Regenerate bad naive baseline images.
  RUN_METRICS            Compute visual perturbation and border metrics.
  KEEP_IMAGES            Keep generated images after metrics. Default 1 for auditability.
EOF
  exit 0
fi

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
REFERENCE_ROOT="${REFERENCE_ROOT:-paper_outputs/prediction_transition_benchmark_test_only_unbalanced}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/visual_perturbation_paper_subset}"
TASK_DIRECTIONS="${TASK_DIRECTIONS:-hnscc_hpv/hpv_pos_to_hpv_neg luad_normal_tumor/normal_to_tumor prad_morphology_group/p4_to_p5}"
BUDGETS="${BUDGETS:-1,8,16,32}"
REGIONS_PER_DIRECTION="${REGIONS_PER_DIRECTION:-4}"
SLIDES_PER_DIRECTION="${SLIDES_PER_DIRECTION:-0}"
REGIONS_PER_SLIDE="${REGIONS_PER_SLIDE:-0}"
REGION_SORT="${REGION_SORT:-first}"
SELECTOR="${SELECTOR:-attention}"

RUN_BUILD="${RUN_BUILD:-1}"
RUN_OURS="${RUN_OURS:-0}"
RUN_NAIVE="${RUN_NAIVE:-0}"
if [[ -z "${RUN_METRICS:-}" ]]; then
  if [[ "${RUN_OURS}" == "1" && "${RUN_NAIVE}" == "1" ]]; then
    RUN_METRICS=1
  else
    RUN_METRICS=0
  fi
fi
KEEP_IMAGES="${KEEP_IMAGES:-1}"

DEVICE="${DEVICE:-cuda:0}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
OUTPUT_MODE="${OUTPUT_MODE:-minimal}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
MAX_CONCEPTS="${MAX_CONCEPTS:-3}"
VISUAL_FORMATS="${VISUAL_FORMATS:-png,pdf,svg}"
NO_PER_CELL="${NO_PER_CELL:-1}"

OURS_POLICY="${OURS_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"
NAIVE_POLICY="${NAIVE_POLICY:-configs/edit_policies/bad_naive_no_preserve_no_sliding.json}"

HNSCC_REGION_BANK_CSV="${HNSCC_REGION_BANK_CSV:-artifacts/prediction_transition_region_banks_test_only_unbalanced/hnscc_hpv/region_bank.csv}"
NORMAL_REGION_ROOT="${NORMAL_REGION_ROOT:-artifacts/prediction_transition_region_banks_test_only_unbalanced/normal_tumor}"
NORMAL_CONCEPT_ROOT="${NORMAL_CONCEPT_ROOT:-artifacts/concept_discovery_normal_tumor_relu_sae_base}"
PRAD_MORPH_REGION_ROOT="${PRAD_MORPH_REGION_ROOT:-artifacts/prediction_transition_region_banks_test_only_unbalanced/prad_morphology_group}"
PRAD_MORPH_CONCEPT_ROOT="${PRAD_MORPH_CONCEPT_ROOT:-artifacts/concept_discovery_prad_morphology_group_relu_sae_base/prad_morphology_group/labels}"

task_dir_args=()
for task_direction in ${TASK_DIRECTIONS}; do
  task_dir_args+=(--task-direction "${task_direction}")
done

if [[ "${RUN_BUILD}" == "1" ]]; then
  echo "[build] visual subset manifests -> ${OUT_ROOT}/manifests" >&2
  "${PY}" scripts/build_visual_perturbation_subset_manifests.py \
    --reference-root "${REFERENCE_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --budgets "${BUDGETS}" \
    --regions-per-direction "${REGIONS_PER_DIRECTION}" \
    --slides-per-direction "${SLIDES_PER_DIRECTION}" \
    --regions-per-slide "${REGIONS_PER_SLIDE}" \
    --selector "${SELECTOR}" \
    --region-sort "${REGION_SORT}" \
    "${task_dir_args[@]}"
fi

region_bank_for() {
  local task_name="$1"
  local direction="$2"
  case "${task_name}/${direction}" in
    hnscc_hpv/*) printf '%s\n' "${HNSCC_REGION_BANK_CSV}" ;;
    *_normal_tumor/normal_to_tumor) printf '%s\n' "${NORMAL_REGION_ROOT}/${task_name}/region_bank.csv" ;;
    *_normal_tumor/tumor_to_normal) printf '%s\n' "${NORMAL_REGION_ROOT}/${task_name}/tumor_to_normal/region_bank.csv" ;;
    prad_morphology_group/*) printf '%s\n' "${PRAD_MORPH_REGION_ROOT}/${direction}/region_bank.csv" ;;
    *) echo "[error] no region-bank rule for ${task_name}/${direction}" >&2; return 2 ;;
  esac
}

runner_direction_for() {
  local task_name="$1"
  local direction="$2"
  case "${task_name}/${direction}" in
    hnscc_hpv/hpv_pos_to_hpv_neg) printf '%s\n' "hpv_neg" ;;
    hnscc_hpv/hpv_neg_to_hpv_pos) printf '%s\n' "hpv_pos" ;;
    *_normal_tumor/normal_to_tumor) printf '%s\n' "hpv_pos" ;;
    *_normal_tumor/tumor_to_normal) printf '%s\n' "hpv_neg" ;;
    prad_morphology_group/*) printf '%s\n' "hpv_pos" ;;
    *) echo "[error] no direction rule for ${task_name}/${direction}" >&2; return 2 ;;
  esac
}

concept_args_for() {
  local task_name="$1"
  local direction="$2"
  case "${task_name}/${direction}" in
    hnscc_hpv/*)
      return 0
      ;;
    *_normal_tumor/normal_to_tumor)
      printf '%s\0%s\0%s\0' \
        "${NORMAL_CONCEPT_ROOT}/${task_name}/labels/tumor/selected_concepts.json" \
        "${NORMAL_CONCEPT_ROOT}/${task_name}/labels/tumor/representative_tiles.csv" \
        "tumor"
      ;;
    *_normal_tumor/tumor_to_normal)
      printf '%s\0%s\0%s\0' \
        "${NORMAL_CONCEPT_ROOT}/${task_name}/labels/normal/selected_concepts.json" \
        "${NORMAL_CONCEPT_ROOT}/${task_name}/labels/normal/representative_tiles.csv" \
        "normal"
      ;;
    prad_morphology_group/p4_to_p5)
      printf '%s\0%s\0%s\0' \
        "${PRAD_MORPH_CONCEPT_ROOT}/pattern_5_solid_single_necrosis/selected_concepts.json" \
        "${PRAD_MORPH_CONCEPT_ROOT}/pattern_5_solid_single_necrosis/representative_tiles.csv" \
        "pattern_5_solid_single_necrosis"
      ;;
    prad_morphology_group/well_to_p4)
      printf '%s\0%s\0%s\0' \
        "${PRAD_MORPH_CONCEPT_ROOT}/pattern_4_cribriform_poorly_formed_fused/selected_concepts.json" \
        "${PRAD_MORPH_CONCEPT_ROOT}/pattern_4_cribriform_poorly_formed_fused/representative_tiles.csv" \
        "pattern_4_cribriform_poorly_formed_fused"
      ;;
    prad_morphology_group/p4_to_well)
      printf '%s\0%s\0%s\0' \
        "${PRAD_MORPH_CONCEPT_ROOT}/pattern_1_3_well_formed/selected_concepts.json" \
        "${PRAD_MORPH_CONCEPT_ROOT}/pattern_1_3_well_formed/representative_tiles.csv" \
        "pattern_1_3_well_formed"
      ;;
    *) echo "[error] no concept rule for ${task_name}/${direction}" >&2; return 2 ;;
  esac
}

generate_method() {
  local method="$1"
  local policy="$2"
  local task_name="$3"
  local direction="$4"
  local manifest="${OUT_ROOT}/manifests/${task_name}/${direction}/visual_subset_manifest.json"
  local out_dir="${OUT_ROOT}/generated_${method}/${task_name}/${direction}"
  local region_bank
  region_bank="$(region_bank_for "${task_name}" "${direction}")"
  local runner_direction
  runner_direction="$(runner_direction_for "${task_name}" "${direction}")"

  if [[ ! -f "${manifest}" ]]; then
    echo "[error] missing subset manifest: ${manifest}" >&2
    return 2
  fi
  if [[ ! -f "${region_bank}" ]]; then
    echo "[error] missing region bank: ${region_bank}" >&2
    return 2
  fi

  local cmd=(
    "${PY}" scripts/run_progressive_region_edit.py
    --task "${task_name}"
    --region-bank-csv "${region_bank}"
    --edit-manifest "${manifest}"
    --out-dir "${out_dir}"
    --edit-policy "${policy}"
    --direction "${runner_direction}"
    --target-magnification 20
    --sae-variant "${SAE_VARIANT}"
    --output-mode "${OUTPUT_MODE}"
    --device "${DEVICE}"
  )
  if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  local concept_payload=()
  if mapfile -d '' -t concept_payload < <(concept_args_for "${task_name}" "${direction}"); then
    if [[ "${#concept_payload[@]}" -gt 0 ]]; then
      cmd+=(
        --concepts-json "${concept_payload[0]}"
        --representative-tiles-csv "${concept_payload[1]}"
        --concept-class-label "${concept_payload[2]}"
        --concept-ranking-method attention_weighted
        --concept-target-stat median
        --concept-target-top-k "${CONCEPT_TARGET_TOP_K}"
        --concept-steering-mode prototype_vector
        --max-concepts "${MAX_CONCEPTS}"
      )
    fi
  fi

  echo "[generate:${method}] ${task_name}/${direction} -> ${out_dir}" >&2
  "${cmd[@]}"
}

run_metrics() {
  local task_name="$1"
  local direction="$2"
  local manifest="${OUT_ROOT}/manifests/${task_name}/${direction}/visual_subset_manifest.json"
  local ours_root="${OUT_ROOT}/generated_ours/${task_name}/${direction}"
  local naive_root="${OUT_ROOT}/generated_bad_naive/${task_name}/${direction}"
  local out_dir="${OUT_ROOT}/metrics_visual/${task_name}/${direction}"
  local per_cell_arg=()
  if [[ "${NO_PER_CELL}" == "1" ]]; then
    per_cell_arg+=(--no-per-cell)
  fi
  echo "[metrics] ${task_name}/${direction} -> ${out_dir}" >&2
  "${PY}" scripts/compare_edit_visual_perturbation.py \
    --ours-root "${ours_root}" \
    --naive-root "${naive_root}" \
    --manifest "${manifest}" \
    --out-dir "${out_dir}" \
    --title "${task_name} ${direction}: paper edit vs bad naive" \
    --formats "${VISUAL_FORMATS}" \
    "${per_cell_arg[@]}"
}

for task_direction in ${TASK_DIRECTIONS}; do
  task_name="${task_direction%%/*}"
  direction="${task_direction#*/}"
  if [[ "${RUN_OURS}" == "1" ]]; then
    generate_method "ours" "${OURS_POLICY}" "${task_name}" "${direction}"
  else
    echo "[skip] ours generation disabled for ${task_name}/${direction}" >&2
  fi
  if [[ "${RUN_NAIVE}" == "1" ]]; then
    generate_method "bad_naive" "${NAIVE_POLICY}" "${task_name}" "${direction}"
  else
    echo "[skip] bad naive generation disabled for ${task_name}/${direction}" >&2
  fi
  if [[ "${RUN_METRICS}" == "1" ]]; then
    run_metrics "${task_name}" "${direction}"
  else
    echo "[skip] metrics disabled for ${task_name}/${direction}" >&2
  fi
  if [[ "${KEEP_IMAGES}" != "1" && "${RUN_METRICS}" == "1" ]]; then
    echo "[cleanup] removing generated images for ${task_name}/${direction}" >&2
    rm -rf "${OUT_ROOT}/generated_ours/${task_name}/${direction}" "${OUT_ROOT}/generated_bad_naive/${task_name}/${direction}"
  fi
done

echo "[ok] visual perturbation paper subset root: ${OUT_ROOT}" >&2
