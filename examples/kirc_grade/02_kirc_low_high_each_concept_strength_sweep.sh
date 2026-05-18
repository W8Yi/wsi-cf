#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

# Number of source regions per direction. Keep small for exploration because this runs:
# directions x strengths x concepts x regions.
MAX_RUNS="${MAX_RUNS:-3}"
STEPS="${STEPS:-30}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
STRENGTHS_CSV="${STRENGTHS_CSV:-0.4,0.8,1.0}"
CONCEPT_TARGET_STAT="${CONCEPT_TARGET_STAT:-median}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
CONCEPT_STEERING_MODE="${CONCEPT_STEERING_MODE:-prototype_vector}"
STEER_BLEND="${STEER_BLEND:-1.0}"
MAX_CONCEPTS_PER_LABEL="${MAX_CONCEPTS_PER_LABEL:-0}"

SLIDES_ROOT="${SLIDES_ROOT:-/research/projects/mllab/WSI/TCGA_features}"
READY_SLIDE_ROOT="${READY_SLIDE_ROOT:-/research/projects/mllab/WSI/.tmp/ready_buffer/slides/TCGA-KIRC}"
DOWNLOAD_SLIDES="${DOWNLOAD_SLIDES:-1}"
DOWNLOAD_SLIDE_COUNT="${DOWNLOAD_SLIDE_COUNT:-4}"
REQUIRE_LABEL_MATCH="${REQUIRE_LABEL_MATCH:-1}"

OUT_ROOT="${OUT_ROOT:-artifacts/kirc_low_high_20x_2048_each_concept_strength_sweep}"
SPLIT_CONCEPT_ROOT="${SPLIT_CONCEPT_ROOT:-${OUT_ROOT}/_single_concept_jsons}"

download_kirc_slides_for_label() {
  local source_label="$1"
  if [[ "${DOWNLOAD_SLIDES}" != "1" ]]; then
    return
  fi
  "$PY" - <<PY
import csv
import json
import sys
from pathlib import Path

import requests

manifest = Path("artifacts/classifier_training/kirc_low_vs_high_grade/task_manifest.csv")
ready_root = Path("${READY_SLIDE_ROOT}")
source_label = "${source_label}"
max_download = int("${DOWNLOAD_SLIDE_COUNT}")

rows = [r for r in csv.DictReader(manifest.open()) if r.get("label_name") == source_label]
ready_root.mkdir(parents=True, exist_ok=True)

def existing_slide(slide_key: str) -> Path | None:
    slide_dir = ready_root / slide_key
    hits = list(slide_dir.glob(f"{slide_key}*.svs")) + list(ready_root.glob(f"{slide_key}*.svs"))
    return hits[0] if hits else None

def query_dx_file(case_id: str, slide_key: str) -> tuple[str, str] | None:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": [case_id]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size",
        "format": "JSON",
        "size": "100",
    }
    resp = requests.get("https://api.gdc.cancer.gov/files", params=params, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]
    for hit in hits:
        name = str(hit["file_name"])
        if name.startswith(slide_key) and ".svs" in name:
            return str(hit["file_id"]), name
    return None

downloaded_or_present = 0
for row in rows:
    slide_key = row["slide_key"]
    case_id = row["case_id"]
    slide_dir = ready_root / slide_key
    slide_dir.mkdir(parents=True, exist_ok=True)
    if existing_slide(slide_key):
        downloaded_or_present += 1
        if downloaded_or_present >= max_download:
            break
        continue
    found = query_dx_file(case_id, slide_key)
    if found is None:
        print(f"[warn] no GDC DX file for {slide_key}", file=sys.stderr)
        continue
    file_id, file_name = found
    out_path = slide_dir / file_name
    print(f"[download] {source_label} {slide_key} -> {out_path}", flush=True)
    with requests.get(f"https://api.gdc.cancer.gov/data/{file_id}", stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with tmp_path.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp_path.rename(out_path)
    downloaded_or_present += 1
    if downloaded_or_present >= max_download:
        break

if downloaded_or_present == 0:
    raise SystemExit(f"No source KIRC slides are staged or downloaded for label={source_label}.")
print(f"[ok] KIRC source slides available for {source_label}: {downloaded_or_present}")
PY
}

find_regions_for_direction() {
  local source_label="$1"
  local target_label="$2"
  local region_dir="$3"
  if [[ -f "${region_dir}/region_bank.csv" && -f "${region_dir}/progressive_edit_manifest.json" ]]; then
    if "$PY" - <<PY
import json
from pathlib import Path
p = Path("${region_dir}/summary.json")
raise SystemExit(0 if p.exists() and int(json.loads(p.read_text()).get("n_selected_rows", 0)) > 0 else 1)
PY
    then
      echo "[reuse] regions ${region_dir}"
      return
    fi
    echo "[refresh] existing region directory has no selected rows: ${region_dir}"
    rm -f "${region_dir}/region_bank.csv" "${region_dir}/progressive_edit_manifest.json" "${region_dir}/selected_regions.csv" "${region_dir}/summary.json"
  fi
  download_kirc_slides_for_label "${source_label}"
  local label_match_flag="--require-label-match"
  if [[ "${REQUIRE_LABEL_MATCH}" == "0" ]]; then
    label_match_flag="--no-require-label-match"
  fi
  "$PY" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir artifacts/classifier_training/kirc_low_vs_high_grade \
    --source-label "${source_label}" \
    --target-label "${target_label}" \
    --slides-root "${SLIDES_ROOT}" \
    --target-magnification 20 \
    --region-size 2048 \
    --grid-step-px 256 \
    --max-regions "${MAX_RUNS}" \
    --max-candidates-per-slide 4 \
    --attention-percentile 85 \
    --min-selected-cells 4 \
    --max-selected-cells 12 \
    --target-importance-mass 0.45 \
    --min-tissue 0.45 \
    --min-dark-fraction 0.04 \
    --min-saturation-fraction 0.04 \
    "${label_match_flag}" \
    --out-dir "${region_dir}" \
    --device "${DEVICE}"
  "$PY" - <<PY
import json
from pathlib import Path

summary_path = Path("${region_dir}/summary.json")
summary = json.loads(summary_path.read_text())
n_selected = int(summary.get("n_selected_rows", 0))
if n_selected <= 0:
    raise SystemExit(
        "No eligible regions found for ${source_label}->${target_label}. "
        f"See {summary_path} and {Path('${region_dir}') / 'candidate_scan.csv'}. "
        "Try increasing DOWNLOAD_SLIDE_COUNT or set REQUIRE_LABEL_MATCH=0 for exploratory runs."
    )
PY
}

split_concepts_for_label() {
  local concept_label="$1"
  local concept_dir="artifacts/classifier_label_concepts_all_top10_top50/kirc_low_vs_high_grade/${concept_label}"
  local split_dir="${SPLIT_CONCEPT_ROOT}/${concept_label}"
  mkdir -p "${split_dir}"
  "$PY" - <<PY
import json
from pathlib import Path

concept_label = "${concept_label}"
concept_dir = Path("${concept_dir}")
concepts_path = concept_dir / "selected_concepts.json"
representative_tiles_csv = concept_dir / "representative_tiles.csv"
out_dir = Path("${split_dir}")
data = json.loads(concepts_path.read_text())
concepts = sorted(data.get("concepts", []), key=lambda c: int(c.get("concept_rank", 10**9)))
max_concepts = int("${MAX_CONCEPTS_PER_LABEL}")
if max_concepts > 0:
    concepts = concepts[:max_concepts]
for concept in concepts:
    rank = int(concept.get("concept_rank", 0))
    latent = int(concept["latent_idx"])
    single = dict(data)
    single["concepts"] = [concept]
    single["single_concept_rank"] = rank
    single["single_concept_latent_idx"] = latent
    out_path = out_dir / f"{concept_label}_rank_{rank:02d}_latent_{latent}.json"
    out_path.write_text(json.dumps(single, indent=2) + "\\n")
    print(f"{rank}\\t{latent}\\t{out_path}\\t{representative_tiles_csv}")
PY
}

run_direction_sweep() {
  local source_label="$1"
  local target_label="$2"
  local concept_label="$3"
  local direction_name="${source_label}_to_${target_label}"
  local region_dir="${OUT_ROOT}/${direction_name}_regions${MAX_RUNS}"

  find_regions_for_direction "${source_label}" "${target_label}" "${region_dir}"

  mapfile -t CONCEPT_LINES < <(split_concepts_for_label "${concept_label}")
  IFS=',' read -ra STRENGTHS <<< "${STRENGTHS_CSV}"

  for strength in "${STRENGTHS[@]}"; do
    strength_clean="${strength//./p}"
    for line in "${CONCEPT_LINES[@]}"; do
      IFS=$'\t' read -r rank latent single_json representative_tiles_csv <<< "${line}"
      concept_out="${OUT_ROOT}/${direction_name}/strength_${strength_clean}/${concept_label}_rank_$(printf '%02d' "${rank}")_latent_${latent}"
      echo "[run] ${direction_name} strength=${strength} concept=${concept_label} rank=${rank} latent=${latent}"
      "$PY" scripts/run_progressive_region_edit.py \
        --task kirc_low_vs_high_grade \
        --region-bank-csv "${region_dir}/region_bank.csv" \
        --edit-manifest "${region_dir}/progressive_edit_manifest.json" \
        --out-dir "${concept_out}" \
        --concepts-json "${single_json}" \
        --representative-tiles-csv "${representative_tiles_csv}" \
        --concept-class-label "${concept_label}" \
        --concept-ranking-method attention_weighted \
        --concept-target-stat "${CONCEPT_TARGET_STAT}" \
        --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
        --concept-steering-mode "${CONCEPT_STEERING_MODE}" \
        --max-concepts 0 \
        --max-runs "${MAX_RUNS}" \
        --target-magnification 20 \
        --edit-support border_relaxed \
        --prototype-strength "${strength}" \
        --steer-blend "${STEER_BLEND}" \
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
        --device "${DEVICE}"
    done
  done
}

run_direction_sweep "low" "high" "high"
run_direction_sweep "high" "low" "low"

echo "[ok] KIRC bidirectional per-concept strength sweep: ${OUT_ROOT}"
echo "[ok] edit-box overlays are saved as source_targets_overlay.png and generated_targets_overlay.png in each run folder."
