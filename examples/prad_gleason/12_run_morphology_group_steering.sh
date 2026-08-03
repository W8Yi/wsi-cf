#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-prad_morphology_group}"
CLASSIFIER_RUN_DIR="${CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_morphology_group}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
CONCEPT_SAE_VARIANT="${CONCEPT_SAE_VARIANT:-${SAE_VARIANT}}"
STEERING_SAE_VARIANT="${STEERING_SAE_VARIANT:-${SAE_VARIANT}}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_prad_morphology_group_${CONCEPT_SAE_VARIANT}}"
CONCEPT_ROOT="${CONCEPT_ROOT:-${CONCEPT_OUT}/${TASK}/labels}"
SLIDES_ROOT="${SLIDES_ROOT:-artifacts/prad_gleason_slides}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/prad_morphology_group_steering}"
REGION_ROOT="${REGION_ROOT:-${OUT_ROOT}/_regions}"
EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"

WELL="pattern_1_3_well_formed"
P4="pattern_4_cribriform_poorly_formed_fused"
P5="pattern_5_solid_single_necrosis"
DIRECTIONS="${DIRECTIONS:-well_to_p4,p4_to_p5,well_to_p5,p5_to_well}"
SLIDES_PER_PAIR="${SLIDES_PER_PAIR:-5}"
CANDIDATE_REGIONS_PER_PAIR="${CANDIDATE_REGIONS_PER_PAIR:-80}"
REFRESH_REGIONS="${REFRESH_REGIONS:-1}"
RUN_EDITS="${RUN_EDITS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
CONCEPT_RANKING_METHOD="${CONCEPT_RANKING_METHOD:-attention_weighted}"
MAX_CONCEPTS="${MAX_CONCEPTS:-3}"
EDIT_SUPPORT="${EDIT_SUPPORT:-padded_center_2x2}"
COMMIT_MODE="${COMMIT_MODE:-full_window}"
WINDOW_STRIDE_CELLS="${WINDOW_STRIDE_CELLS:-1}"
WINDOW_SELECTION_MODE="${WINDOW_SELECTION_MODE:-overlap}"
ATTENTION_PERCENTILE="${ATTENTION_PERCENTILE:-50}"
MIN_SELECTED_CELLS="${MIN_SELECTED_CELLS:-16}"
MAX_SELECTED_CELLS="${MAX_SELECTED_CELLS:-32}"
TARGET_IMPORTANCE_MASS="${TARGET_IMPORTANCE_MASS:-0.90}"
MIN_LABEL_CONFIDENCE="${MIN_LABEL_CONFIDENCE:-0.35}"
MAX_LABEL_CONFIDENCE="${MAX_LABEL_CONFIDENCE:-0.9995}"
MIN_TISSUE="${MIN_TISSUE:-0.35}"
MIN_DARK_FRACTION="${MIN_DARK_FRACTION:-0.02}"
MIN_SATURATION_FRACTION="${MIN_SATURATION_FRACTION:-0.02}"
UPDATE_PAPER_OUTPUTS="${UPDATE_PAPER_OUTPUTS:-1}"

split_csv() {
  local text="$1"
  IFS=',' read -r -a _tokens <<< "${text}"
  for token in "${_tokens[@]}"; do
    echo "${token}" | xargs
  done
}

direction_labels() {
  case "$1" in
    well_to_p4) printf '%s\t%s\n' "${WELL}" "${P4}" ;;
    p4_to_p5) printf '%s\t%s\n' "${P4}" "${P5}" ;;
    well_to_p5) printf '%s\t%s\n' "${WELL}" "${P5}" ;;
    p5_to_well) printf '%s\t%s\n' "${P5}" "${WELL}" ;;
    p4_to_well) printf '%s\t%s\n' "${P4}" "${WELL}" ;;
    p5_to_p4) printf '%s\t%s\n' "${P5}" "${P4}" ;;
    *)
      echo "[error] unsupported morphology direction '$1'" >&2
      echo "        Supported: well_to_p4,p4_to_p5,well_to_p5,p5_to_well,p4_to_well,p5_to_p4" >&2
      exit 2
      ;;
  esac
}

policy_manifest_has_enough_requests() {
  local manifest_path="$1"
  [[ -f "${manifest_path}" ]] || return 1
  "${PY}" - "${manifest_path}" "${SLIDES_PER_PAIR}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if isinstance(payload, list) and len(payload) >= int(sys.argv[2]) else 1)
PY
}

assert_ready() {
  for path in \
    "${CLASSIFIER_RUN_DIR}/best_model.pt" \
    "${CLASSIFIER_RUN_DIR}/label_mapping.json" \
    "${CLASSIFIER_RUN_DIR}/task_manifest.csv"; do
    if [[ ! -f "${path}" ]]; then
      echo "[error] missing PRAD morphology classifier file: ${path}" >&2
      echo "        Train it with: DEVICE=${DEVICE} examples/classifier_training/14_train_prad_morphology_group.sh" >&2
      exit 2
    fi
  done
  for label in "${WELL}" "${P4}" "${P5}"; do
    for path in \
      "${CONCEPT_ROOT}/${label}/selected_concepts.json" \
      "${CONCEPT_ROOT}/${label}/representative_tiles.csv"; do
      if [[ ! -f "${path}" ]]; then
        echo "[error] missing PRAD morphology concept file: ${path}" >&2
        echo "        Refresh concepts with: DEVICE=${DEVICE} examples/prad_gleason/11_refresh_morphology_group_concepts.sh" >&2
        exit 2
      fi
    done
  done
}

write_policy_manifest() {
  local region_dir="$1"
  local out_manifest="$2"

  PYTHONPATH=src "${PY}" - "${region_dir}/region_bank.csv" "${region_dir}/progressive_edit_manifest.json" "${out_manifest}" "${SLIDES_PER_PAIR}" "${EDIT_SUPPORT}" <<'PY'
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
    cells = [(int(cell["gx"]), int(cell["gy"])) for cell in request["target_cells"]]
    supported, dropped = split_cells_by_edit_support(
        target_cells=cells,
        grid_w=int(grid.shape[1]),
        grid_h=int(grid.shape[0]),
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
        f"needed {n_needed}. Increase CANDIDATE_REGIONS_PER_PAIR or relax selection thresholds."
    )
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(selected, indent=2) + "\n")
print(f"[ok] policy-compatible requests: {len(selected)} -> {out_path}")
PY
}

prepare_pair_regions() {
  local source_label="$1"
  local target_label="$2"
  local direction="$3"
  local region_dir="${REGION_ROOT}/${direction}"
  local policy_manifest="${region_dir}/progressive_edit_manifest_showcase_best.json"

  if [[ "${REFRESH_REGIONS}" != "1" ]] && policy_manifest_has_enough_requests "${policy_manifest}"; then
    printf '%s\t%s\n' "${region_dir}" "${policy_manifest}"
    return
  fi

  echo "[regions] ${source_label} -> ${target_label}: ${region_dir}" >&2
  "${PY}" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
    --source-label "${source_label}" \
    --target-label "${target_label}" \
    --slides-root "${SLIDES_ROOT}" \
    --target-magnification 20 \
    --region-size 2048 \
    --grid-step-px 256 \
    --max-regions "${CANDIDATE_REGIONS_PER_PAIR}" \
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
    --require-label-match \
    --out-dir "${region_dir}" \
    --sae-variant "${CONCEPT_SAE_VARIANT}" \
    --device "${DEVICE}" >&2
  write_policy_manifest "${region_dir}" "${policy_manifest}" >&2
  printf '%s\t%s\n' "${region_dir}" "${policy_manifest}"
}

run_pair() {
  local source_label="$1"
  local target_label="$2"
  local direction="$3"
  local region_dir="$4"
  local edit_manifest="$5"
  local concept_dir="${CONCEPT_ROOT}/${target_label}"
  local out_dir="${OUT_ROOT}/${direction}"
  local skip_args=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi

  echo "[generate] ${source_label} -> ${target_label}: ${SLIDES_PER_PAIR} regions" >&2
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
    --max-runs "${SLIDES_PER_PAIR}" \
    --target-magnification 20 \
    --sae-variant "${STEERING_SAE_VARIANT}" \
    --edit-support "${EDIT_SUPPORT}" \
    --window-stride-cells "${WINDOW_STRIDE_CELLS}" \
    --window-selection-mode "${WINDOW_SELECTION_MODE}" \
    --commit-mode "${COMMIT_MODE}" \
    --output-mode "${OUTPUT_MODE}" \
    "${skip_args[@]}" \
    --device "${DEVICE}"
}

assert_ready
mkdir -p "${OUT_ROOT}"

for direction in $(split_csv "${DIRECTIONS}"); do
  IFS=$'\t' read -r source_label target_label < <(direction_labels "${direction}")
  IFS=$'\t' read -r region_dir edit_manifest < <(
    prepare_pair_regions "${source_label}" "${target_label}" "${direction}"
  )
  if [[ "${RUN_EDITS}" == "1" ]]; then
    run_pair "${source_label}" "${target_label}" "${direction}" "${region_dir}" "${edit_manifest}"
  else
    echo "[skip] generation disabled for ${direction}; set RUN_EDITS=1" >&2
  fi
done

if [[ "${UPDATE_PAPER_OUTPUTS}" == "1" ]]; then
  echo "[paper] refreshing paper_outputs/current symlink view" >&2
  "${PY}" paper/artifacts.py scan
  "${PY}" paper/artifacts.py make-paper-view
fi

echo "[ok] PRAD morphology steering outputs: ${OUT_ROOT}" >&2
