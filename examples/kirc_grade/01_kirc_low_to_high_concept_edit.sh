#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

# KIRC low-grade -> high-grade showcase defaults.
SOURCE_LABEL="${SOURCE_LABEL:-low}"
TARGET_LABEL="${TARGET_LABEL:-high}"
CONCEPT_LABEL="${CONCEPT_LABEL:-high}"

MAX_RUNS="${MAX_RUNS:-5}"
STEPS="${STEPS:-30}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
CONCEPT_STEERING_MODE="${CONCEPT_STEERING_MODE:-prototype_vector}"

REGION_DIR="${REGION_DIR:-artifacts/kirc_low_to_high_20x_2048_regions${MAX_RUNS}}"
OUT_DIR="${OUT_DIR:-artifacts/kirc_low_to_high_20x_2048_regions${MAX_RUNS}_concept_edit}"
CONCEPT_DIR="${CONCEPT_DIR:-artifacts/classifier_label_concepts_all_top10_top50/kirc_low_vs_high_grade/${CONCEPT_LABEL}}"

SLIDES_ROOT="${SLIDES_ROOT:-/research/projects/mllab/WSI/TCGA_features}"
READY_SLIDE_ROOT="${READY_SLIDE_ROOT:-/research/projects/mllab/WSI/.tmp/ready_buffer/slides/TCGA-KIRC}"

# Download just enough KIRC diagnostic DX slides for region cropping if they are not staged.
# Set DOWNLOAD_SLIDES=0 if you already staged matching KIRC .svs files.
DOWNLOAD_SLIDES="${DOWNLOAD_SLIDES:-1}"
DOWNLOAD_SLIDE_COUNT="${DOWNLOAD_SLIDE_COUNT:-3}"

if [[ "${DOWNLOAD_SLIDES}" == "1" ]]; then
  "$PY" - <<PY
import csv
import json
import sys
from pathlib import Path
from urllib.parse import quote

import requests

manifest = Path("artifacts/classifier_training/kirc_low_vs_high_grade/task_manifest.csv")
ready_root = Path("${READY_SLIDE_ROOT}")
source_label = "${SOURCE_LABEL}"
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
    print(f"[download] {slide_key} -> {out_path}", flush=True)
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
    raise SystemExit("No source KIRC slides are staged or downloaded. Cannot crop image regions.")
print(f"[ok] KIRC source slides available: {downloaded_or_present}")
PY
fi

"$PY" scripts/find_regions.py \
  --mode attention \
  --classifier-run-dir artifacts/classifier_training/kirc_low_vs_high_grade \
  --source-label "${SOURCE_LABEL}" \
  --target-label "${TARGET_LABEL}" \
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
  --out-dir "${REGION_DIR}" \
  --device "${DEVICE}"

"$PY" scripts/run_progressive_region_edit.py \
  --task kirc_low_vs_high_grade \
  --region-bank-csv "${REGION_DIR}/region_bank.csv" \
  --edit-manifest "${REGION_DIR}/progressive_edit_manifest.json" \
  --out-dir "${OUT_DIR}" \
  --concepts-json "${CONCEPT_DIR}/selected_concepts.json" \
  --representative-tiles-csv "${CONCEPT_DIR}/representative_tiles.csv" \
  --concept-class-label "${CONCEPT_LABEL}" \
  --concept-ranking-method attention_weighted \
  --concept-target-stat median \
  --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
  --concept-steering-mode "${CONCEPT_STEERING_MODE}" \
  --max-concepts 0 \
  --max-runs "${MAX_RUNS}" \
  --target-magnification 20 \
  --edit-support border_relaxed \
  --prototype-strength 0.8 \
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
  --device "${DEVICE}"

echo "[ok] regions: ${REGION_DIR}"
echo "[ok] edits:   ${OUT_DIR}"
echo "[ok] edit-box overlays are saved as source_targets_overlay.png and generated_targets_overlay.png in each run folder."
