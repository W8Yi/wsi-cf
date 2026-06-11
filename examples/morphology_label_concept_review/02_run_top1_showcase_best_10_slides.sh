#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EDIT_POLICY="${EDIT_POLICY:-configs/edit_policies/showcase_best.json}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
SLIDES_PER_LABEL="${SLIDES_PER_LABEL:-10}"
CANDIDATE_REGIONS_PER_LABEL="${CANDIDATE_REGIONS_PER_LABEL:-20}"
DOWNLOAD_SLIDES="${DOWNLOAD_SLIDES:-1}"
DOWNLOAD_SLIDE_COUNT="${DOWNLOAD_SLIDE_COUNT:-30}"
REQUIRE_LABEL_MATCH="${REQUIRE_LABEL_MATCH:-1}"
REFRESH_REGIONS="${REFRESH_REGIONS:-0}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"

SLIDES_ROOT="${SLIDES_ROOT:-/research/projects/mllab/WSI/TCGA_features}"
REVIEW_ROOT="${REVIEW_ROOT:-artifacts/morphology_label_concept_review/selected}"
OUT_ROOT="${OUT_ROOT:-artifacts/morphology_label_concept_review_top1_showcase_best_10slides}"
REGION_ROOT="${REGION_ROOT:-${OUT_ROOT}/_regions}"

policy_manifest_has_enough_requests() {
  local manifest_path="$1"
  [[ -f "${manifest_path}" ]] || return 1
  "$PY" - "${manifest_path}" "${SLIDES_PER_LABEL}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if isinstance(payload, list) and len(payload) >= int(sys.argv[2]) else 1)
PY
}

stage_source_slides() {
  local classifier_run_dir="$1"
  local source_label="$2"
  local project="$3"

  if [[ "${DOWNLOAD_SLIDES}" != "1" ]]; then
    return
  fi

  "$PY" - "${classifier_run_dir}" "${source_label}" "${project}" "${SLIDES_ROOT}" "${DOWNLOAD_SLIDE_COUNT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

import requests

classifier_dir = Path(sys.argv[1])
source_label = sys.argv[2]
project = sys.argv[3]
slides_root = Path(sys.argv[4])
needed = int(sys.argv[5])
ready_root = slides_root.parent / ".tmp/ready_buffer/slides" / project
rows = [
    row for row in csv.DictReader((classifier_dir / "task_manifest.csv").open())
    if row.get("label_name") == source_label
]
ready_root.mkdir(parents=True, exist_ok=True)

def local_slide(slide_key: str) -> Path | None:
    for root in (slides_root / project / "slides", ready_root):
        hits = list(root.glob(f"{slide_key}*.svs")) + list(root.glob(f"{slide_key}*/*.svs"))
        if hits:
            return hits[0]
    return None

def gdc_slide(case_id: str, slide_key: str) -> tuple[str, str] | None:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": [case_id]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
        ],
    }
    response = requests.get(
        "https://api.gdc.cancer.gov/files",
        params={
            "filters": json.dumps(filters),
            "fields": "file_id,file_name",
            "format": "JSON",
            "size": "100",
        },
        timeout=60,
    )
    response.raise_for_status()
    for hit in response.json()["data"]["hits"]:
        name = str(hit["file_name"])
        if name.startswith(slide_key) and name.endswith(".svs"):
            return str(hit["file_id"]), name
    return None

n_ready = 0
for row in rows:
    slide_key = row["slide_key"]
    if local_slide(slide_key) is not None:
        n_ready += 1
    else:
        found = gdc_slide(row["case_id"], slide_key)
        if found is None:
            continue
        file_id, file_name = found
        out_dir = ready_root / slide_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / file_name
        print(f"[download] {source_label} {slide_key} -> {out_path}", flush=True)
        with requests.get(f"https://api.gdc.cancer.gov/data/{file_id}", stream=True, timeout=120) as response:
            response.raise_for_status()
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            tmp_path.rename(out_path)
        n_ready += 1
    if n_ready >= needed:
        break

if n_ready == 0:
    raise SystemExit(f"No local or downloadable slides found for {source_label}")
print(f"[ok] source slides ready for {source_label}: {n_ready}")
PY
}

write_policy_manifest() {
  local region_dir="$1"
  local out_manifest="$2"

  PYTHONPATH=src "$PY" - "${region_dir}/region_bank.csv" "${region_dir}/progressive_edit_manifest.json" "${out_manifest}" "${SLIDES_PER_LABEL}" <<'PY'
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
    supported, _ = split_cells_by_edit_support(
        target_cells=cells,
        grid_w=grid_w,
        grid_h=grid_h,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=int(row["grid_step_px"]),
        edit_support="center_2x2",
    )
    if not supported:
        continue
    kept = dict(request)
    kept["target_cells"] = [{"gx": int(gx), "gy": int(gy)} for gx, gy in supported]
    kept["selector"] = f"{kept.get('selector', 'classifier_attention')}__center_2x2_policy"
    selected.append(kept)
    seen_slides.add(slide_key)
    if len(selected) >= n_needed:
        break

if len(selected) < n_needed:
    raise SystemExit(
        f"Only {len(selected)} distinct-slide edit requests are compatible with center_2x2; "
        f"needed {n_needed}. Increase CANDIDATE_REGIONS_PER_LABEL and DOWNLOAD_SLIDE_COUNT."
    )
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(selected, indent=2) + "\n")
print(f"[ok] policy-compatible requests: {len(selected)} -> {out_path}")
PY
}

prepare_direction_regions() {
  local classifier_run_dir="$1"
  local project="$2"
  local source_label="$3"
  local target_label="$4"
  local direction_token="$5"
  local region_dir="${REGION_ROOT}/${direction_token}"
  local policy_manifest="${region_dir}/progressive_edit_manifest_showcase_best.json"

  if [[ "${REFRESH_REGIONS}" != "1" ]] && policy_manifest_has_enough_requests "${policy_manifest}"; then
    printf '%s\t%s\n' "${region_dir}" "${policy_manifest}"
    return
  fi

  stage_source_slides "${classifier_run_dir}" "${source_label}" "${project}" >&2
  local label_match_flag="--require-label-match"
  if [[ "${REQUIRE_LABEL_MATCH}" == "0" ]]; then
    label_match_flag="--no-require-label-match"
  fi
  "$PY" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir "${classifier_run_dir}" \
    --source-label "${source_label}" \
    --target-label "${target_label}" \
    --slides-root "${SLIDES_ROOT}" \
    --target-magnification 20 \
    --region-size 2048 \
    --grid-step-px 256 \
    --max-regions "${CANDIDATE_REGIONS_PER_LABEL}" \
    --max-candidates-per-slide 1 \
    --attention-percentile 85 \
    --min-selected-cells 4 \
    --max-selected-cells 12 \
    --target-importance-mass 0.45 \
    --min-tissue 0.45 \
    --min-dark-fraction 0.04 \
    --min-saturation-fraction 0.04 \
    "${label_match_flag}" \
    --out-dir "${region_dir}" \
    --sae-variant "${SAE_VARIANT}" \
    --device "${DEVICE}" >&2
  write_policy_manifest "${region_dir}" "${policy_manifest}" >&2
  printf '%s\t%s\n' "${region_dir}" "${policy_manifest}"
}

run_direction() {
  local task="$1"
  local source_label="$2"
  local target_label="$3"
  local concept_dir="$4"
  local direction_token="$5"
  local region_dir="$6"
  local edit_manifest="$7"
  local out_dir="${OUT_ROOT}/${direction_token}"
  local skip_args=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi

  echo "[run] ${source_label} -> ${target_label}: ${SLIDES_PER_LABEL} slides, top-1 ${target_label} concept, policy=${EDIT_POLICY}"
  "$PY" scripts/run_progressive_region_edit.py \
    --task "${task}" \
    --region-bank-csv "${region_dir}/region_bank.csv" \
    --edit-manifest "${edit_manifest}" \
    --out-dir "${out_dir}" \
    --edit-policy "${EDIT_POLICY}" \
    --concepts-json "${concept_dir}/selected_concepts.json" \
    --representative-tiles-csv "${concept_dir}/representative_tiles.csv" \
    --concept-class-label "${target_label}" \
    --concept-ranking-method attention_weighted \
    --concept-target-stat median \
    --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
    --concept-steering-mode prototype_vector \
    --max-concepts 1 \
    --max-runs "${SLIDES_PER_LABEL}" \
    --target-magnification 20 \
    --sae-variant "${SAE_VARIANT}" \
    --output-mode "${OUTPUT_MODE}" \
    "${skip_args[@]}" \
    --device "${DEVICE}"
}

IFS=$'\t' read -r LUAD_TO_LUSC_DIR LUAD_TO_LUSC_MANIFEST < <(
  prepare_direction_regions "artifacts/classifier_training/luad_lusc" "TCGA-LUAD" "LUAD" "LUSC" "01_luad_to_lusc"
)
IFS=$'\t' read -r LUSC_TO_LUAD_DIR LUSC_TO_LUAD_MANIFEST < <(
  prepare_direction_regions "artifacts/classifier_training/luad_lusc" "TCGA-LUSC" "LUSC" "LUAD" "01_lusc_to_luad"
)
IFS=$'\t' read -r KIRC_LOW_TO_HIGH_DIR KIRC_LOW_TO_HIGH_MANIFEST < <(
  prepare_direction_regions "artifacts/classifier_training/kirc_low_vs_high_grade" "TCGA-KIRC" "low" "high" "03_kirc_low_to_high"
)
IFS=$'\t' read -r KIRC_HIGH_TO_LOW_DIR KIRC_HIGH_TO_LOW_MANIFEST < <(
  prepare_direction_regions "artifacts/classifier_training/kirc_low_vs_high_grade" "TCGA-KIRC" "high" "low" "03_kirc_high_to_low"
)

# Priority 1: toward LUSC latent 3103 and toward LUAD latent 701.
run_direction "luad_lusc" "LUAD" "LUSC" \
  "${REVIEW_ROOT}/01_luad_lusc__LUSC" "01_luad_to_lusc" "${LUAD_TO_LUSC_DIR}" "${LUAD_TO_LUSC_MANIFEST}"
run_direction "luad_lusc" "LUSC" "LUAD" \
  "${REVIEW_ROOT}/01_luad_lusc__LUAD" "01_lusc_to_luad" "${LUSC_TO_LUAD_DIR}" "${LUSC_TO_LUAD_MANIFEST}"

# Priority 3: toward high latent 4766 and toward low latent 2957.
run_direction "kirc_low_vs_high_grade" "low" "high" \
  "${REVIEW_ROOT}/03_kirc_low_vs_high_grade__high" "03_kirc_low_to_high" "${KIRC_LOW_TO_HIGH_DIR}" "${KIRC_LOW_TO_HIGH_MANIFEST}"
run_direction "kirc_low_vs_high_grade" "high" "low" \
  "${REVIEW_ROOT}/03_kirc_low_vs_high_grade__low" "03_kirc_high_to_low" "${KIRC_HIGH_TO_LOW_DIR}" "${KIRC_HIGH_TO_LOW_MANIFEST}"

echo "[ok] showcase_best top-1 10-slide steering outputs: ${OUT_ROOT}"
