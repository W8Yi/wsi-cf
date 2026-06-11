#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
MAX_RUNS="${MAX_RUNS:-1}"
STEPS="${STEPS:-30}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
PROTOTYPE_STRENGTH="${PROTOTYPE_STRENGTH:-0.8}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DOWNLOAD_SLIDES="${DOWNLOAD_SLIDES:-1}"
DOWNLOAD_SLIDE_COUNT="${DOWNLOAD_SLIDE_COUNT:-3}"
REQUIRE_LABEL_MATCH="${REQUIRE_LABEL_MATCH:-1}"

REVIEW_ROOT="${REVIEW_ROOT:-artifacts/morphology_label_concept_review/selected}"
OUT_ROOT="${OUT_ROOT:-artifacts/morphology_label_concept_review_top1_steers}"
SLIDES_ROOT="${SLIDES_ROOT:-/research/projects/mllab/WSI/TCGA_features}"

# These existing region banks avoid re-mining regions or downloading slides.
# Override either root to steer a newly prepared compatible region bank.
LUAD_LUSC_REGION_ROOT="${LUAD_LUSC_REGION_ROOT:-artifacts/luad_lusc_1024_concept_remine_sweep_old_relu_sae_base_strength_0p4_0p8_1p0/regions}"
KIRC_REGION_ROOT="${KIRC_REGION_ROOT:-artifacts/kirc_low_high_20x_2048_each_concept_strength_sweep_prototype_retest}"

manifest_has_requests() {
  local manifest_path="$1"
  [[ -f "${manifest_path}" ]] || return 1
  "$PY" - "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if isinstance(payload, list) and len(payload) > 0 else 1)
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

classifier_run_dir = Path(sys.argv[1])
source_label = sys.argv[2]
project = sys.argv[3]
slides_root = Path(sys.argv[4])
max_download = int(sys.argv[5])
ready_root = slides_root.parent / ".tmp/ready_buffer/slides" / project
manifest = classifier_run_dir / "task_manifest.csv"
rows = [row for row in csv.DictReader(manifest.open()) if row.get("label_name") == source_label]
ready_root.mkdir(parents=True, exist_ok=True)

def existing_slide(slide_key: str) -> Path | None:
    roots = [slides_root / project / "slides", ready_root]
    for root in roots:
        hits = list(root.glob(f"{slide_key}*.svs")) + list(root.glob(f"{slide_key}*/*.svs"))
        if hits:
            return hits[0]
    return None

def query_dx_file(case_id: str, slide_key: str) -> tuple[str, str] | None:
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
            "fields": "file_id,file_name,file_size",
            "format": "JSON",
            "size": "100",
        },
        timeout=60,
    )
    response.raise_for_status()
    for hit in response.json()["data"]["hits"]:
        name = str(hit["file_name"])
        if name.startswith(slide_key) and ".svs" in name:
            return str(hit["file_id"]), name
    return None

n_ready = 0
for row in rows:
    slide_key = row["slide_key"]
    if existing_slide(slide_key):
        n_ready += 1
        if n_ready >= max_download:
            break
        continue
    found = query_dx_file(row["case_id"], slide_key)
    if found is None:
        print(f"[warn] no GDC DX file for {slide_key}", file=sys.stderr)
        continue
    file_id, file_name = found
    slide_dir = ready_root / slide_key
    slide_dir.mkdir(parents=True, exist_ok=True)
    out_path = slide_dir / file_name
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
    if n_ready >= max_download:
        break

if n_ready == 0:
    raise SystemExit(
        f"No staged or downloadable source slides found for {source_label}. "
        "Provide a populated region bank or a suitable SLIDES_ROOT."
    )
print(f"[ok] source slides available for {source_label}: {n_ready}")
PY
}

prepare_region_bank() {
  local task="$1"
  local classifier_run_dir="$2"
  local project="$3"
  local source_label="$4"
  local target_label="$5"
  local preferred_dir="$6"
  local fallback_dir="$7"
  local region_size="$8"
  local min_selected_cells="$9"
  local max_selected_cells="${10}"

  if manifest_has_requests "${preferred_dir}/progressive_edit_manifest.json"; then
    printf '%s\n' "${preferred_dir}"
    return
  fi
  if manifest_has_requests "${fallback_dir}/progressive_edit_manifest.json"; then
    printf '%s\n' "${fallback_dir}"
    return
  fi

  echo "[prepare] ${source_label} -> ${target_label}: source region manifest is empty; mining replacement regions" >&2
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
    --region-size "${region_size}" \
    --grid-step-px 256 \
    --max-regions "${MAX_RUNS}" \
    --max-candidates-per-slide 4 \
    --attention-percentile 85 \
    --min-selected-cells "${min_selected_cells}" \
    --max-selected-cells "${max_selected_cells}" \
    --target-importance-mass 0.45 \
    --min-tissue 0.45 \
    --min-dark-fraction 0.04 \
    --min-saturation-fraction 0.04 \
    "${label_match_flag}" \
    --out-dir "${fallback_dir}" \
    --sae-variant "${SAE_VARIANT}" \
    --device "${DEVICE}" >&2
  if ! manifest_has_requests "${fallback_dir}/progressive_edit_manifest.json"; then
    echo "[error] No regions selected for ${source_label} -> ${target_label}. Try REQUIRE_LABEL_MATCH=0." >&2
    exit 1
  fi
  printf '%s\n' "${fallback_dir}"
}

direction_outputs_complete() {
  local region_dir="$1"
  local direction_out_dir="$2"
  "$PY" - "${region_dir}/progressive_edit_manifest.json" "${direction_out_dir}" "${MAX_RUNS}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
max_runs = int(sys.argv[3])
requests = json.loads(manifest_path.read_text())
if max_runs > 0:
    requests = requests[:max_runs]
complete = bool(requests) and all((out_dir / str(row["run_id"]) / "generated.png").exists() for row in requests)
raise SystemExit(0 if complete else 1)
PY
}

run_top1_steer() {
  local task="$1"
  local source_label="$2"
  local target_label="$3"
  local concept_dir="$4"
  local region_dir="$5"
  local run_name="$6"

  for required_path in \
    "${concept_dir}/selected_concepts.json" \
    "${concept_dir}/representative_tiles.csv" \
    "${region_dir}/region_bank.csv" \
    "${region_dir}/progressive_edit_manifest.json"; do
    if [[ ! -f "${required_path}" ]]; then
      echo "[error] Missing required input: ${required_path}" >&2
      exit 1
    fi
  done

  if [[ "${SKIP_EXISTING}" == "1" ]] && direction_outputs_complete "${region_dir}" "${OUT_ROOT}/${run_name}"; then
    echo "[reuse] ${source_label} -> ${target_label}; output already complete"
    return
  fi

  echo "[run] ${source_label} -> ${target_label}; top-1 ${target_label} concept"
  local skip_args=()
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi
  "$PY" scripts/run_progressive_region_edit.py \
    --task "${task}" \
    --region-bank-csv "${region_dir}/region_bank.csv" \
    --edit-manifest "${region_dir}/progressive_edit_manifest.json" \
    --out-dir "${OUT_ROOT}/${run_name}" \
    --concepts-json "${concept_dir}/selected_concepts.json" \
    --representative-tiles-csv "${concept_dir}/representative_tiles.csv" \
    --concept-class-label "${target_label}" \
    --concept-ranking-method attention_weighted \
    --concept-target-stat median \
    --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
    --concept-steering-mode prototype_vector \
    --max-concepts 1 \
    --max-runs "${MAX_RUNS}" \
    --target-magnification 20 \
    --edit-support border_relaxed \
    --sae-variant "${SAE_VARIANT}" \
    --prototype-strength "${PROTOTYPE_STRENGTH}" \
    --steer-blend 1.0 \
    --preserve-edit-strength 0.05 \
    --preserve-visited-strength 0.95 \
    --preserve-fresh-context-strength 0.35 \
    --mid-steer-start-ratio 0.5 \
    --mid-steer-end-ratio 1.0 \
    --mid-steer-alpha-start 0.5 \
    --mid-steer-alpha-end 1.0 \
    --mid-steer-alpha-schedule linear \
    --steps "${STEPS}" \
    --guidance 2.0 \
    --patch-batch 256 \
    --output-mode "${OUTPUT_MODE}" \
    "${skip_args[@]}" \
    --device "${DEVICE}"
}

LUAD_TO_LUSC_REGIONS="$(prepare_region_bank \
  "luad_lusc" \
  "artifacts/classifier_training/luad_lusc" \
  "TCGA-LUAD" \
  "LUAD" \
  "LUSC" \
  "${LUAD_LUSC_REGION_ROOT}/luad_to_lusc_1024_regions10" \
  "${OUT_ROOT}/_regions/luad_to_lusc_1024_regions${MAX_RUNS}" \
  "1024" \
  "2" \
  "6")"

LUSC_TO_LUAD_REGIONS="$(prepare_region_bank \
  "luad_lusc" \
  "artifacts/classifier_training/luad_lusc" \
  "TCGA-LUSC" \
  "LUSC" \
  "LUAD" \
  "${LUAD_LUSC_REGION_ROOT}/lusc_to_luad_1024_regions10" \
  "${OUT_ROOT}/_regions/lusc_to_luad_1024_regions${MAX_RUNS}" \
  "1024" \
  "2" \
  "6")"

KIRC_LOW_TO_HIGH_REGIONS="$(prepare_region_bank \
  "kirc_low_vs_high_grade" \
  "artifacts/classifier_training/kirc_low_vs_high_grade" \
  "TCGA-KIRC" \
  "low" \
  "high" \
  "${KIRC_REGION_ROOT}/low_to_high_regions3" \
  "${OUT_ROOT}/_regions/kirc_low_to_high_2048_regions${MAX_RUNS}" \
  "2048" \
  "4" \
  "12")"

KIRC_HIGH_TO_LOW_REGIONS="$(prepare_region_bank \
  "kirc_low_vs_high_grade" \
  "artifacts/classifier_training/kirc_low_vs_high_grade" \
  "TCGA-KIRC" \
  "high" \
  "low" \
  "${KIRC_REGION_ROOT}/high_to_low_regions3" \
  "${OUT_ROOT}/_regions/kirc_high_to_low_2048_regions${MAX_RUNS}" \
  "2048" \
  "4" \
  "12")"

# Priority 1: LUAD/LUSC. Top concepts: toward LUAD latent 701, toward LUSC latent 3103.
run_top1_steer \
  "luad_lusc" \
  "LUAD" \
  "LUSC" \
  "${REVIEW_ROOT}/01_luad_lusc__LUSC" \
  "${LUAD_TO_LUSC_REGIONS}" \
  "01_luad_to_lusc_top1"

run_top1_steer \
  "luad_lusc" \
  "LUSC" \
  "LUAD" \
  "${REVIEW_ROOT}/01_luad_lusc__LUAD" \
  "${LUSC_TO_LUAD_REGIONS}" \
  "01_lusc_to_luad_top1"

# Priority 3: KIRC grade. Top concepts: toward low latent 2957, toward high latent 4766.
run_top1_steer \
  "kirc_low_vs_high_grade" \
  "low" \
  "high" \
  "${REVIEW_ROOT}/03_kirc_low_vs_high_grade__high" \
  "${KIRC_LOW_TO_HIGH_REGIONS}" \
  "03_kirc_low_to_high_top1"

run_top1_steer \
  "kirc_low_vs_high_grade" \
  "high" \
  "low" \
  "${REVIEW_ROOT}/03_kirc_low_vs_high_grade__low" \
  "${KIRC_HIGH_TO_LOW_REGIONS}" \
  "03_kirc_high_to_low_top1"

echo "[ok] top-1 morphology review steering outputs: ${OUT_ROOT}"
