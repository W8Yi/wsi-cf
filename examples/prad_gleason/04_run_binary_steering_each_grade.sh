#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-prad_low_vs_high_grade}"
CLASSIFIER_RUN_DIR="${CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_low_vs_high_grade}"
CONCEPT_SAE_VARIANT="${CONCEPT_SAE_VARIANT:-relu_sae_base}"
STEERING_SAE_VARIANT="${STEERING_SAE_VARIANT:-${SAE_VARIANT:-relu_sae_base}}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_prad_gleason_${CONCEPT_SAE_VARIANT}}"
CONCEPT_ROOT="${CONCEPT_ROOT:-${CONCEPT_OUT}/prad_low_vs_high_grade/labels}"
SLIDES_ROOT="${SLIDES_ROOT:-artifacts/prad_gleason_slides}"
LABEL_SOURCE="${LABEL_SOURCE:-artifacts/prad_gleason_inputs/slide_labels.csv}"
GRADE_MANIFEST_ROOT="${GRADE_MANIFEST_ROOT:-artifacts/prad_gleason_inputs/binary_grade_manifests}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/prad_gleason_binary_by_grade_steering}"
REGION_ROOT="${REGION_ROOT:-${OUT_ROOT}/_regions}"
EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"

GRADE_GROUPS="${GRADE_GROUPS:-GG1,GG2,GG3,GG4,GG5}"
SLIDES_PER_GRADE="${SLIDES_PER_GRADE:-5}"
CANDIDATE_REGIONS_PER_GRADE="${CANDIDATE_REGIONS_PER_GRADE:-60}"
REFRESH_LABELS="${REFRESH_LABELS:-0}"
REFRESH_GRADE_MANIFESTS="${REFRESH_GRADE_MANIFESTS:-1}"
REFRESH_CONCEPTS="${REFRESH_CONCEPTS:-0}"
REFRESH_REGIONS="${REFRESH_REGIONS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_LABEL_MATCH="${REQUIRE_LABEL_MATCH:-1}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
CONCEPT_RANKING_METHOD="${CONCEPT_RANKING_METHOD:-attention_weighted}"
MAX_CONCEPTS="${MAX_CONCEPTS:-3}"
EDIT_SUPPORT="${EDIT_SUPPORT:-padded_center_2x2}"
COMMIT_MODE="${COMMIT_MODE:-full_window}"
COMMIT_FEATHER_PX="${COMMIT_FEATHER_PX:-0}"
ATTENTION_PERCENTILE="${ATTENTION_PERCENTILE:-50}"
MIN_SELECTED_CELLS="${MIN_SELECTED_CELLS:-16}"
MAX_SELECTED_CELLS="${MAX_SELECTED_CELLS:-32}"
TARGET_IMPORTANCE_MASS="${TARGET_IMPORTANCE_MASS:-0.90}"
MIN_LABEL_CONFIDENCE="${MIN_LABEL_CONFIDENCE:-0.50}"
MAX_LABEL_CONFIDENCE="${MAX_LABEL_CONFIDENCE:-0.9995}"
MIN_TISSUE="${MIN_TISSUE:-0.35}"
MIN_DARK_FRACTION="${MIN_DARK_FRACTION:-0.02}"
MIN_SATURATION_FRACTION="${MIN_SATURATION_FRACTION:-0.02}"
UPDATE_PAPER_OUTPUTS="${UPDATE_PAPER_OUTPUTS:-1}"

grade_source_label() {
  case "$1" in
    GG1|GG2) echo "low" ;;
    GG3|GG4|GG5) echo "high" ;;
    *) echo "[error] unsupported PRAD grade group '$1'; expected GG1-GG5" >&2; exit 2 ;;
  esac
}

target_for_source_label() {
  case "$1" in
    low) echo "high" ;;
    high) echo "low" ;;
    *) echo "[error] unsupported source label '$1'" >&2; exit 2 ;;
  esac
}

slug() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

policy_manifest_has_enough_requests() {
  local manifest_path="$1"
  [[ -f "${manifest_path}" ]] || return 1
  "${PY}" - "${manifest_path}" "${SLIDES_PER_GRADE}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if isinstance(payload, list) and len(payload) >= int(sys.argv[2]) else 1)
PY
}

assert_binary_classifier_ready() {
  local missing=0
  for path in \
    "${CLASSIFIER_RUN_DIR}/best_model.pt" \
    "${CLASSIFIER_RUN_DIR}/label_mapping.json" \
    "${CLASSIFIER_RUN_DIR}/task_manifest.csv"; do
    if [[ ! -f "${path}" ]]; then
      echo "[error] missing PRAD binary classifier file: ${path}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" == "1" ]]; then
    echo "        Train it with: DEVICE=${DEVICE} examples/classifier_training/12_train_prad_low_vs_high_grade.sh" >&2
    exit 2
  fi
}

assert_slides_ready() {
  local project_dir="${SLIDES_ROOT}/TCGA-PRAD/slides"
  local n
  n="$(find "${project_dir}" -type f -iname '*.svs' 2>/dev/null | wc -l || true)"
  if [[ "${n}" -eq 0 ]]; then
    echo "[error] no PRAD SVS slides found under ${project_dir}" >&2
    echo "        Run: MAX_FILES=0 examples/prad_gleason/00_download_prad_slides.sh" >&2
    exit 2
  fi
  echo "[slides] PRAD SVS available: ${n}" >&2
}

assert_concepts_ready() {
  local missing=0
  for label in low high; do
    for path in \
      "${CONCEPT_ROOT}/${label}/selected_concepts.json" \
      "${CONCEPT_ROOT}/${label}/representative_tiles.csv" \
      "${CONCEPT_ROOT}/${label}/summary.json"; do
      if [[ ! -f "${path}" ]]; then
        echo "[error] missing PRAD ${label} concept file: ${path}" >&2
        missing=1
      fi
    done
  done
  if [[ "${missing}" == "1" ]]; then
    echo "        Refresh concepts with: DEVICE=${DEVICE} SAE_VARIANT=${CONCEPT_SAE_VARIANT} CONCEPT_OUT=${CONCEPT_OUT} examples/prad_gleason/01_refresh_attention_concepts.sh" >&2
    exit 2
  fi
}

maybe_refresh_labels() {
  if [[ "${REFRESH_LABELS}" != "1" ]]; then
    return
  fi
  echo "[labels] refreshing PRAD Gleason labels" >&2
  "${PY}" scripts/prepare_prad_gleason_concept_tasks.py \
    --out-dir "$(dirname "${LABEL_SOURCE}")"
}

maybe_refresh_concepts() {
  if [[ "${REFRESH_CONCEPTS}" != "1" ]]; then
    return
  fi
  echo "[concepts] refreshing binary PRAD concepts with ${CONCEPT_SAE_VARIANT}" >&2
  DEVICE="${DEVICE}" \
    SAE_VARIANT="${CONCEPT_SAE_VARIANT}" \
    CONCEPT_OUT="${CONCEPT_OUT}" \
    examples/prad_gleason/01_refresh_attention_concepts.sh
}

build_grade_manifests() {
  if [[ "${REFRESH_GRADE_MANIFESTS}" != "1" && -f "${GRADE_MANIFEST_ROOT}/summary.json" ]]; then
    echo "[skip] grade manifests exist: ${GRADE_MANIFEST_ROOT}" >&2
    return
  fi
  echo "[manifests] building binary classifier manifests for grades: ${GRADE_GROUPS}" >&2
  "${PY}" scripts/build_prad_binary_grade_manifests.py \
    --classifier-task-manifest "${CLASSIFIER_RUN_DIR}/task_manifest.csv" \
    --slide-labels "${LABEL_SOURCE}" \
    --out-dir "${GRADE_MANIFEST_ROOT}" \
    --grade-column grade_group \
    --grades "${GRADE_GROUPS}"
}

write_policy_manifest() {
  local region_dir="$1"
  local out_manifest="$2"

  PYTHONPATH=src "${PY}" - "${region_dir}/region_bank.csv" "${region_dir}/progressive_edit_manifest.json" "${out_manifest}" "${SLIDES_PER_GRADE}" "${EDIT_SUPPORT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np

from wsi_cf.steering.progressive import split_cells_by_edit_support

bank_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
n_needed = int(sys.argv[4])
edit_support = str(sys.argv[5])

bank_rows = {row["region_id"]: row for row in csv.DictReader(bank_path.open())}
requests = json.loads(manifest_path.read_text())
selected = []
seen_slides = set()
for request in requests:
    row = bank_rows.get(str(request["region_id"]))
    if row is None:
        continue
    slide_key = str(row.get("slide_key", ""))
    if slide_key in seen_slides:
        continue
    grid = np.load(row["feature_grid_path"])
    grid_h, grid_w = int(grid.shape[0]), int(grid.shape[1])
    cells = [(int(cell["gx"]), int(cell["gy"])) for cell in request["target_cells"]]
    supported, dropped = split_cells_by_edit_support(
        target_cells=cells,
        grid_w=grid_w,
        grid_h=grid_h,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=int(row["grid_step_px"]),
        edit_support=edit_support,
    )
    if not supported:
        continue
    kept = dict(request)
    kept["target_cells"] = [{"gx": int(gx), "gy": int(gy)} for gx, gy in supported]
    kept["dropped_unsupported_cells"] = [{"gx": int(gx), "gy": int(gy)} for gx, gy in dropped]
    kept["selector"] = f"{kept.get('selector', 'classifier_attention')}__{edit_support}_policy"
    selected.append(kept)
    seen_slides.add(slide_key)
    if len(selected) >= n_needed:
        break

if len(selected) < n_needed:
    raise SystemExit(
        f"Only {len(selected)} distinct-slide requests are compatible with {edit_support}; "
        f"needed {n_needed}. Increase CANDIDATE_REGIONS_PER_GRADE or relax selection thresholds."
    )
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(selected, indent=2) + "\n")
print(f"[ok] policy-compatible requests: {len(selected)} -> {out_path}")
PY
}

prepare_grade_regions() {
  local grade="$1"
  local source_label="$2"
  local target_label="$3"
  local direction="$4"
  local grade_manifest="${GRADE_MANIFEST_ROOT}/prad_binary_grade_group_${grade}.csv"
  local region_dir="${REGION_ROOT}/${direction}"
  local policy_manifest="${region_dir}/progressive_edit_manifest_showcase_best.json"

  if [[ ! -f "${grade_manifest}" ]]; then
    echo "[error] missing grade manifest for ${grade}: ${grade_manifest}" >&2
    exit 2
  fi

  if [[ "${REFRESH_REGIONS}" != "1" ]] && policy_manifest_has_enough_requests "${policy_manifest}"; then
    printf '%s\t%s\n' "${region_dir}" "${policy_manifest}"
    return
  fi

  local label_match_flag="--require-label-match"
  if [[ "${REQUIRE_LABEL_MATCH}" == "0" ]]; then
    label_match_flag="--no-require-label-match"
  fi

  echo "[regions] ${grade}: ${source_label} -> ${target_label}: ${region_dir}" >&2
  "${PY}" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
    --task-manifest-csv "${grade_manifest}" \
    --source-label "${source_label}" \
    --target-label "${target_label}" \
    --slides-root "${SLIDES_ROOT}" \
    --target-magnification 20 \
    --region-size 2048 \
    --grid-step-px 256 \
    --max-regions "${CANDIDATE_REGIONS_PER_GRADE}" \
    --max-candidates-per-slide 1 \
    --attention-percentile "${ATTENTION_PERCENTILE}" \
    --min-selected-cells "${MIN_SELECTED_CELLS}" \
    --max-selected-cells "${MAX_SELECTED_CELLS}" \
    --target-importance-mass "${TARGET_IMPORTANCE_MASS}" \
    --min-label-confidence "${MIN_LABEL_CONFIDENCE}" \
    --max-label-confidence "${MAX_LABEL_CONFIDENCE}" \
    --min-tissue "${MIN_TISSUE}" \
    --min-dark-fraction "${MIN_DARK_FRACTION}" \
    --min-saturation-fraction "${MIN_SATURATION_FRACTION}" \
    "${label_match_flag}" \
    --out-dir "${region_dir}" \
    --sae-variant "${CONCEPT_SAE_VARIANT}" \
    --device "${DEVICE}" >&2
  write_policy_manifest "${region_dir}" "${policy_manifest}" >&2
  printf '%s\t%s\n' "${region_dir}" "${policy_manifest}"
}

run_grade_direction() {
  local grade="$1"
  local source_label="$2"
  local target_label="$3"
  local direction="$4"
  local region_dir="$5"
  local edit_manifest="$6"
  local concept_dir="${CONCEPT_ROOT}/${target_label}"
  local out_dir="${OUT_ROOT}/${direction}"
  local skip_args=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi

  echo "[generate] ${grade}: ${source_label} -> ${target_label}: ${SLIDES_PER_GRADE} regions" >&2
  "${PY}" scripts/run_progressive_region_edit.py \
    --task "${TASK}" \
    --region-bank-csv "${region_dir}/region_bank.csv" \
    --edit-manifest "${edit_manifest}" \
    --out-dir "${out_dir}" \
    --edit-policy "${EDIT_POLICY}" \
    --concepts-json "${concept_dir}/selected_concepts.json" \
    --representative-tiles-csv "${concept_dir}/representative_tiles.csv" \
    --concept-class-label "${target_label}" \
    --concept-ranking-method "${CONCEPT_RANKING_METHOD}" \
    --concept-target-stat median \
    --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
    --concept-steering-mode prototype_vector \
    --max-concepts "${MAX_CONCEPTS}" \
    --max-runs "${SLIDES_PER_GRADE}" \
    --target-magnification 20 \
    --sae-variant "${STEERING_SAE_VARIANT}" \
    --edit-support "${EDIT_SUPPORT}" \
    --commit-mode "${COMMIT_MODE}" \
    --commit-feather-px "${COMMIT_FEATHER_PX}" \
    --output-mode "${OUTPUT_MODE}" \
    "${skip_args[@]}" \
    --device "${DEVICE}"
}

maybe_refresh_labels
assert_binary_classifier_ready
maybe_refresh_concepts
assert_concepts_ready
assert_slides_ready
build_grade_manifests

mkdir -p "${OUT_ROOT}"
IFS=',' read -r -a GRADES <<< "${GRADE_GROUPS}"
for raw_grade in "${GRADES[@]}"; do
  grade="$(echo "${raw_grade}" | xargs)"
  [[ -n "${grade}" ]] || continue
  source_label="$(grade_source_label "${grade}")"
  target_label="$(target_for_source_label "${source_label}")"
  direction="$(slug "${grade}")_${source_label}_to_${target_label}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${grade}: ${source_label} -> ${target_label}; manifest=${GRADE_MANIFEST_ROOT}/prad_binary_grade_group_${grade}.csv; out=${OUT_ROOT}/${direction}" >&2
    continue
  fi
  IFS=$'\t' read -r region_dir edit_manifest < <(
    prepare_grade_regions "${grade}" "${source_label}" "${target_label}" "${direction}"
  )
  run_grade_direction "${grade}" "${source_label}" "${target_label}" "${direction}" "${region_dir}" "${edit_manifest}"
done

if [[ "${UPDATE_PAPER_OUTPUTS}" == "1" ]]; then
  echo "[paper] refreshing paper_outputs/current symlink view" >&2
  "${PY}" paper/artifacts.py scan
  "${PY}" paper/artifacts.py make-paper-view
fi

echo "[ok] PRAD binary-by-grade steering outputs: ${OUT_ROOT}" >&2
