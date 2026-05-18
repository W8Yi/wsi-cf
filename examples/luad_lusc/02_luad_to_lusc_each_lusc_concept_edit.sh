#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

# First stage: choose 10 LUAD source regions at 20x equivalent.
REGION_DIR="${REGION_DIR:-artifacts/luad_to_lusc_20x_2048_regions10}"
MAX_RUNS="${MAX_RUNS:-10}"

# LUSC concepts discovered by the concept-finder workflow.
CONCEPT_DIR="${CONCEPT_DIR:-artifacts/classifier_label_concepts_all_top10_top50/luad_lusc/LUSC}"
CONCEPTS_JSON="${CONCEPTS_JSON:-${CONCEPT_DIR}/selected_concepts.json}"
REPRESENTATIVE_TILES_CSV="${REPRESENTATIVE_TILES_CSV:-${CONCEPT_DIR}/representative_tiles.csv}"

# One output folder per concept. Each concept folder contains edits for the same 10 regions.
OUT_ROOT="${OUT_ROOT:-artifacts/luad_to_lusc_20x_2048_each_lusc_concept}"
SPLIT_CONCEPT_DIR="${SPLIT_CONCEPT_DIR:-${OUT_ROOT}/_single_concept_jsons}"

STEPS="${STEPS:-30}"
OUTPUT_MODE="${OUTPUT_MODE:-minimal}"
CONCEPT_TARGET_STAT="${CONCEPT_TARGET_STAT:-median}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
PROTOTYPE_STRENGTH="${PROTOTYPE_STRENGTH:-0.8}"
STEER_BLEND="${STEER_BLEND:-1.0}"
PRESERVE_EDIT_STRENGTH="${PRESERVE_EDIT_STRENGTH:-0.05}"
PRESERVE_VISITED_STRENGTH="${PRESERVE_VISITED_STRENGTH:-0.95}"
PRESERVE_FRESH_CONTEXT_STRENGTH="${PRESERVE_FRESH_CONTEXT_STRENGTH:-0.35}"
MID_STEER_ALPHA_START="${MID_STEER_ALPHA_START:-0.5}"
MID_STEER_ALPHA_END="${MID_STEER_ALPHA_END:-1.0}"

if [[ ! -f "${REGION_DIR}/region_bank.csv" || ! -f "${REGION_DIR}/progressive_edit_manifest.json" ]]; then
  "$PY" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir artifacts/classifier_training/luad_lusc \
    --source-label LUAD \
    --target-label LUSC \
    --slides-root /research/projects/mllab/WSI/TCGA_features \
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
fi

mkdir -p "${SPLIT_CONCEPT_DIR}"

mapfile -t CONCEPT_LINES < <("$PY" - <<PY
import json
from pathlib import Path

concepts_path = Path("${CONCEPTS_JSON}")
out_dir = Path("${SPLIT_CONCEPT_DIR}")
data = json.loads(concepts_path.read_text())
concepts = sorted(data.get("concepts", []), key=lambda c: int(c.get("concept_rank", 10**9)))

for concept in concepts:
    rank = int(concept.get("concept_rank", 0))
    latent = int(concept["latent_idx"])
    single = dict(data)
    single["concepts"] = [concept]
    single["single_concept_rank"] = rank
    single["single_concept_latent_idx"] = latent
    out_path = out_dir / f"lusc_rank_{rank:02d}_latent_{latent}.json"
    out_path.write_text(json.dumps(single, indent=2) + "\\n")
    print(f"{rank}\\t{latent}\\t{out_path}")
PY
)

for line in "${CONCEPT_LINES[@]}"; do
  IFS=$'\t' read -r rank latent single_json <<< "${line}"
  concept_out="${OUT_ROOT}/lusc_rank_$(printf '%02d' "${rank}")_latent_${latent}"

  echo "[run] LUSC concept rank=${rank} latent=${latent}"
  "$PY" scripts/run_progressive_region_edit.py \
    --task luad_lusc \
    --region-bank-csv "${REGION_DIR}/region_bank.csv" \
    --edit-manifest "${REGION_DIR}/progressive_edit_manifest.json" \
    --out-dir "${concept_out}" \
    --concepts-json "${single_json}" \
    --representative-tiles-csv "${REPRESENTATIVE_TILES_CSV}" \
    --concept-class-label LUSC \
    --concept-ranking-method attention_weighted \
    --concept-target-stat "${CONCEPT_TARGET_STAT}" \
    --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
    --max-concepts 0 \
    --max-runs "${MAX_RUNS}" \
    --target-magnification 20 \
    --edit-support border_relaxed \
    --prototype-strength "${PROTOTYPE_STRENGTH}" \
    --steer-blend "${STEER_BLEND}" \
    --preserve-edit-strength "${PRESERVE_EDIT_STRENGTH}" \
    --preserve-visited-strength "${PRESERVE_VISITED_STRENGTH}" \
    --preserve-fresh-context-strength "${PRESERVE_FRESH_CONTEXT_STRENGTH}" \
    --mid-steer-start-ratio 0.5 \
    --mid-steer-end-ratio 1.0 \
    --mid-steer-alpha-start "${MID_STEER_ALPHA_START}" \
    --mid-steer-alpha-end "${MID_STEER_ALPHA_END}" \
    --mid-steer-alpha-schedule linear \
    --steps "${STEPS}" \
    --guidance 2.0 \
    --patch-batch 256 \
    --output-mode "${OUTPUT_MODE}" \
    --device "${DEVICE}"
done

echo "[ok] source regions: ${REGION_DIR}"
echo "[ok] per-concept edits: ${OUT_ROOT}"
