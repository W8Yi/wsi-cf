#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-prad_low_vs_high_grade}"
CLASSIFIER_RUN_DIR="${CLASSIFIER_RUN_DIR:-artifacts/classifier_training/prad_low_vs_high_grade}"
CONCEPT_SAE_VARIANT="${CONCEPT_SAE_VARIANT:-tcga_sae_batch_topk_20x_interp}"
STEERING_SAE_VARIANT="${STEERING_SAE_VARIANT:-${SAE_VARIANT:-relu_sae_base}}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_json}"
CONCEPT_ROOT="${CONCEPT_ROOT:-${CONCEPT_OUT}/prad_low_vs_high_grade/labels}"
SLIDES_ROOT="${SLIDES_ROOT:-artifacts/prad_gleason_slides}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/prad_gleason_low_high_steering}"
REGION_ROOT="${REGION_ROOT:-${OUT_ROOT}/_regions}"
EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json}"
SLIDES_PER_DIRECTION="${SLIDES_PER_DIRECTION:-10}"
CANDIDATE_REGIONS_PER_DIRECTION="${CANDIDATE_REGIONS_PER_DIRECTION:-30}"
REFRESH_REGIONS="${REFRESH_REGIONS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
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

policy_manifest_has_enough_requests() {
  local manifest_path="$1"
  [[ -f "${manifest_path}" ]] || return 1
  "${PY}" - "${manifest_path}" "${SLIDES_PER_DIRECTION}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if isinstance(payload, list) and len(payload) >= int(sys.argv[2]) else 1)
PY
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
  "${PY}" - "${CONCEPT_ROOT}" "${CONCEPT_SAE_VARIANT}" "${STEERING_SAE_VARIANT}" "${CONCEPT_RANKING_METHOD}" <<'PY'
import json
import sys
from pathlib import Path

concept_root = Path(sys.argv[1])
expected_concept_sae = str(sys.argv[2])
steering_sae = str(sys.argv[3])
ranking_method = str(sys.argv[4])
errors = []
for label in ("low", "high"):
    label_dir = concept_root / label
    summary_path = label_dir / "summary.json"
    concepts_path = label_dir / "selected_concepts.json"
    reps_path = label_dir / "representative_tiles.csv"
    for path in (summary_path, concepts_path, reps_path):
        if not path.exists():
            errors.append(f"missing {path}")
    if not summary_path.exists():
        continue
    summary = json.loads(summary_path.read_text())
    args = summary.get("args", {}) if isinstance(summary.get("args"), dict) else {}
    actual_sae = str(args.get("sae_variant") or summary.get("sae_variant") or "")
    effective_mode = str(summary.get("effective_mode") or "")
    if actual_sae != expected_concept_sae:
        errors.append(f"{summary_path}: concept sae_variant={actual_sae!r}, expected {expected_concept_sae!r}")
    if ranking_method == "attention_weighted" and effective_mode != "attention_aware":
        errors.append(
            f"{summary_path}: effective_mode={effective_mode!r}; run examples/prad_gleason/01_refresh_attention_concepts.sh"
        )
if errors:
    print("[error] PRAD concept bundle is not ready for this steering run:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    raise SystemExit(2)
print(
    f"[concepts] ready: {concept_root} concept_sae={expected_concept_sae} "
    f"steering_sae={steering_sae} ranking={ranking_method}",
    file=sys.stderr,
)
PY
}

write_policy_manifest() {
  local region_dir="$1"
  local out_manifest="$2"

  PYTHONPATH=src "${PY}" - "${region_dir}/region_bank.csv" "${region_dir}/progressive_edit_manifest.json" "${out_manifest}" "${SLIDES_PER_DIRECTION}" "${EDIT_SUPPORT}" <<'PY'
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
        f"Only {len(selected)} distinct-slide PRAD requests are compatible with {edit_support}; "
        f"needed {n_needed}. Increase CANDIDATE_REGIONS_PER_DIRECTION."
    )
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(selected, indent=2) + "\n")
print(f"[ok] policy-compatible requests: {len(selected)} -> {out_path}")
PY
}

prepare_direction_regions() {
  local source_label="$1"
  local target_label="$2"
  local direction="$3"
  local region_dir="${REGION_ROOT}/${direction}"
  local policy_manifest="${region_dir}/progressive_edit_manifest_showcase_best.json"

  if [[ "${REFRESH_REGIONS}" != "1" ]] && policy_manifest_has_enough_requests "${policy_manifest}"; then
    printf '%s\t%s\n' "${region_dir}" "${policy_manifest}"
    return
  fi

  local label_match_flag="--require-label-match"
  if [[ "${REQUIRE_LABEL_MATCH}" == "0" ]]; then
    label_match_flag="--no-require-label-match"
  fi

  echo "[regions] ${source_label} -> ${target_label}: ${region_dir}" >&2
  echo "[regions] selection: attention>=p${ATTENTION_PERCENTILE}, cells=${MIN_SELECTED_CELLS}-${MAX_SELECTED_CELLS}, mass=${TARGET_IMPORTANCE_MASS}" >&2
  "${PY}" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
    --source-label "${source_label}" \
    --target-label "${target_label}" \
    --slides-root "${SLIDES_ROOT}" \
    --target-magnification 20 \
    --region-size 2048 \
    --grid-step-px 256 \
    --max-regions "${CANDIDATE_REGIONS_PER_DIRECTION}" \
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

run_direction() {
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

  echo "[generate] ${source_label} -> ${target_label}: ${SLIDES_PER_DIRECTION} PRAD regions" >&2
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
    --max-runs "${SLIDES_PER_DIRECTION}" \
    --target-magnification 20 \
    --sae-variant "${STEERING_SAE_VARIANT}" \
    --edit-support "${EDIT_SUPPORT}" \
    --commit-mode "${COMMIT_MODE}" \
    --commit-feather-px "${COMMIT_FEATHER_PX}" \
    --output-mode "${OUTPUT_MODE}" \
    "${skip_args[@]}" \
    --device "${DEVICE}"
}

assert_slides_ready
assert_concepts_ready

IFS=$'\t' read -r LOW_TO_HIGH_DIR LOW_TO_HIGH_MANIFEST < <(
  prepare_direction_regions "low" "high" "low_to_high"
)
IFS=$'\t' read -r HIGH_TO_LOW_DIR HIGH_TO_LOW_MANIFEST < <(
  prepare_direction_regions "high" "low" "high_to_low"
)

run_direction "low" "high" "low_to_high" "${LOW_TO_HIGH_DIR}" "${LOW_TO_HIGH_MANIFEST}"
run_direction "high" "low" "high_to_low" "${HIGH_TO_LOW_DIR}" "${HIGH_TO_LOW_MANIFEST}"

echo "[ok] PRAD grade steering outputs: ${OUT_ROOT}" >&2
