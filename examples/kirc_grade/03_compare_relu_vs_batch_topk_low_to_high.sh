#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

# This is intentionally a small comparison test. Increase these once the smoke
# run looks sane.
MAX_RUNS="${MAX_RUNS:-1}"
TOP_CONCEPTS="${TOP_CONCEPTS:-1}"
TOP_TILES_PER_CONCEPT="${TOP_TILES_PER_CONCEPT:-50}"
STRENGTHS_CSV="${STRENGTHS_CSV:-0.4}"
STEPS="${STEPS:-12}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
CONCEPT_STEERING_MODE="${CONCEPT_STEERING_MODE:-prototype_vector}"

# Keep concept discovery bounded for the first pass. Set these to 0 for a full
# association scan.
ASSOC_MAX_SLIDES_PER_CLASS="${ASSOC_MAX_SLIDES_PER_CLASS:-24}"
ASSOC_MAX_TILES_PER_SLIDE="${ASSOC_MAX_TILES_PER_SLIDE:-2048}"
FIND_CONCEPT_MAX_SLIDES="${FIND_CONCEPT_MAX_SLIDES:-24}"
TOPK_TO_RELU_MAP_TOP_K="${TOPK_TO_RELU_MAP_TOP_K:-5}"

TASK="kirc_low_vs_high_grade"
CLASSIFIER_RUN_DIR="${CLASSIFIER_RUN_DIR:-artifacts/classifier_training/kirc_low_vs_high_grade}"

# Reuse the low->high regions from the main KIRC sweep if present. This keeps
# the SAE comparison fair because both branches edit the same source region(s).
REGION_DIR="${REGION_DIR:-artifacts/kirc_low_high_20x_2048_each_concept_strength_sweep/low_to_high_regions3}"

OUT_ROOT="${OUT_ROOT:-artifacts/kirc_low_high_sae_model_compare_low_to_high}"
ASSOC_ROOT="${ASSOC_ROOT:-${OUT_ROOT}/concept_associations}"
CONCEPT_ROOT="${CONCEPT_ROOT:-${OUT_ROOT}/concept_cards}"

RELU_SAE_CKPT="${RELU_SAE_CKPT:-resources/models/sae/tcga_uni2_sae_relu_v1/relu_final.pt}"
RELU_SAE_CFG="${RELU_SAE_CFG:-resources/models/sae/tcga_uni2_sae_relu_v1/run_config.json}"

BATCH_TOPK_SAE_CKPT="${BATCH_TOPK_SAE_CKPT:-/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_20x_interp/batch_topk_final.pt}"
BATCH_TOPK_SAE_CFG="${BATCH_TOPK_SAE_CFG:-/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_20x_interp/run_config.json}"

if [[ ! -f "${CLASSIFIER_RUN_DIR}/task_manifest.csv" || ! -f "${CLASSIFIER_RUN_DIR}/best_model.pt" ]]; then
  echo "[error] missing KIRC low/high classifier bundle: ${CLASSIFIER_RUN_DIR}" >&2
  exit 1
fi

if [[ ! -f "${REGION_DIR}/region_bank.csv" || ! -f "${REGION_DIR}/progressive_edit_manifest.json" ]]; then
  echo "[info] ${REGION_DIR} is missing, finding one low->high test region first"
  "$PY" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
    --source-label low \
    --target-label high \
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
    --require-label-match \
    --out-dir "${REGION_DIR}" \
    --device "${DEVICE}"
fi

prepare_concepts_for_sae() {
  local tag="$1"
  local sae_ckpt="$2"
  local sae_cfg="$3"
  local assoc_task="${TASK}_${tag}"
  local concept_out="${CONCEPT_ROOT}/${tag}"

  echo "[concepts] ${tag}: preparing concept-label associations"
  "$PY" scripts/prepare_classifier_concept_associations.py \
    --task-name "${assoc_task}" \
    --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
    --out-root "${ASSOC_ROOT}" \
    --include-labels low,high \
    --max-slides-per-class "${ASSOC_MAX_SLIDES_PER_CLASS}" \
    --max-tiles-per-slide "${ASSOC_MAX_TILES_PER_SLIDE}" \
    --batch-size 4096 \
    --sae-ckpt "${sae_ckpt}" \
    --sae-cfg "${sae_cfg}" \
    --device "${DEVICE}"

  echo "[concepts] ${tag}: finding high-grade concepts"
  "$PY" scripts/find_label_concepts.py \
    --task "${TASK}" \
    --association-task "${assoc_task}" \
    --association-root "${ASSOC_ROOT}" \
    --class-label high \
    --out-dir "${concept_out}" \
    --mode attention_aware \
    --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
    --top-concepts "${TOP_CONCEPTS}" \
    --top-tiles-per-concept "${TOP_TILES_PER_CONCEPT}" \
    --max-slides "${FIND_CONCEPT_MAX_SLIDES}" \
    --sae-ckpt "${sae_ckpt}" \
    --sae-cfg "${sae_cfg}" \
    --device "${DEVICE}"
}

map_batch_topk_concepts_to_relu_steering_space() {
  local source_tag="batch_topk_20x"
  local mapped_tag="batch_topk_20x_find_relu_steer"
  local source_dir="${CONCEPT_ROOT}/${source_tag}/${TASK}/high"
  local mapped_dir="${CONCEPT_ROOT}/${mapped_tag}/${TASK}/high"

  if [[ ! -f "${source_dir}/selected_concepts.json" || ! -f "${source_dir}/representative_tiles.csv" ]]; then
    echo "[error] missing batch-TopK concept outputs: ${source_dir}" >&2
    exit 1
  fi

  echo "[map] batch-TopK concepts -> ReLU steering latents using top ${TOPK_TO_RELU_MAP_TOP_K} representative tiles"
  "$PY" - <<PY
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/common/users/wq50/wsi_cf")
sys.path.insert(0, str(ROOT / "src"))

from wsi_cf.common.runtime import resolve_device
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features

source_dir = Path("${source_dir}")
mapped_dir = Path("${mapped_dir}")
mapped_dir.mkdir(parents=True, exist_ok=True)
map_top_k = int("${TOPK_TO_RELU_MAP_TOP_K}")
device = resolve_device("${DEVICE}")

payload = json.loads((source_dir / "selected_concepts.json").read_text())
concepts = sorted(payload.get("concepts", []), key=lambda c: int(c.get("concept_rank", 10**9)))
if int("${TOP_CONCEPTS}") > 0:
    concepts = concepts[: int("${TOP_CONCEPTS}")]

with (source_dir / "representative_tiles.csv").open("r", newline="") as handle:
    rep_rows = list(csv.DictReader(handle))

rows_by_latent = defaultdict(list)
for row in rep_rows:
    if row.get("ranking_method") != "attention_weighted":
        continue
    rows_by_latent[int(row["latent_idx"])].append(row)
for rows in rows_by_latent.values():
    rows.sort(key=lambda r: int(r.get("tile_rank", 10**9)))

sae_model, _, d_latent = load_sae_from_config(
    Path("${RELU_SAE_CKPT}"),
    Path("${RELU_SAE_CFG}"),
    device=str(device),
)
sae_model.eval()

def read_one_feature(row):
    h5_path = Path(row["h5_path"])
    tile_index = int(row["tile_index"])
    with h5py.File(h5_path, "r") as handle:
        feats = handle["features"]
        if feats.ndim == 3 and feats.shape[0] == 1:
            arr = np.asarray(feats[0, tile_index], dtype=np.float32)
        else:
            arr = np.asarray(feats[tile_index], dtype=np.float32)
    return arr

mapped_concepts = []
mapped_rep_rows = []
mapping_rows = []
used_relu_latents = set()

for out_rank, concept in enumerate(concepts, start=1):
    source_latent = int(concept["latent_idx"])
    seed_rows = rows_by_latent.get(source_latent, [])[:map_top_k]
    if not seed_rows:
        raise SystemExit(f"No attention_weighted representative rows for batch-TopK latent {source_latent}")

    features = np.stack([read_one_feature(row) for row in seed_rows], axis=0).astype(np.float32)
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        z = sae_encode_features(sae_model, x).detach().cpu().numpy().astype(np.float32)
    mean_z = z.mean(axis=0)
    candidate_order = np.argsort(-mean_z)
    relu_latent = None
    for cand in candidate_order:
        cand = int(cand)
        if cand not in used_relu_latents and float(mean_z[cand]) > 0:
            relu_latent = cand
            break
    if relu_latent is None:
        relu_latent = int(candidate_order[0])
    used_relu_latents.add(relu_latent)

    mapped = dict(concept)
    mapped["latent_idx"] = int(relu_latent)
    mapped["concept_rank"] = int(out_rank)
    mapped["source_concept_sae"] = "batch_topk_20x"
    mapped["source_latent_idx"] = int(source_latent)
    mapped["steering_sae"] = "tcga_uni2_sae_relu_v1"
    mapped["topk_to_relu_map_top_k"] = int(map_top_k)
    mapped["relu_mean_activation_on_topk_tiles"] = float(mean_z[relu_latent])
    mapped_concepts.append(mapped)

    for tile_rank, row in enumerate(seed_rows, start=1):
        out = dict(row)
        out["latent_idx"] = str(relu_latent)
        out["source_latent_idx"] = str(source_latent)
        out["source_concept_sae"] = "batch_topk_20x"
        out["steering_sae"] = "tcga_uni2_sae_relu_v1"
        out["ranking_method"] = "attention_weighted"
        out["tile_rank"] = str(tile_rank)
        out["activation"] = str(float(z[tile_rank - 1, relu_latent]))
        attn = float(out.get("attention", 0.0) or 0.0)
        out["attention_weighted_activation"] = str(float(z[tile_rank - 1, relu_latent]) * attn)
        mapped_rep_rows.append(out)

    mapping_rows.append({
        "concept_rank": out_rank,
        "source_batch_topk_latent_idx": source_latent,
        "mapped_relu_latent_idx": relu_latent,
        "relu_mean_activation_on_topk_tiles": float(mean_z[relu_latent]),
        "n_tiles_used": len(seed_rows),
    })

mapped_payload = dict(payload)
mapped_payload["mode"] = "batch_topk_find_relu_steer"
mapped_payload["source_concept_sae"] = "batch_topk_20x"
mapped_payload["steering_sae"] = "tcga_uni2_sae_relu_v1"
mapped_payload["concepts"] = mapped_concepts
(mapped_dir / "selected_concepts.json").write_text(json.dumps(mapped_payload, indent=2) + "\\n")

fieldnames = []
for row in mapped_rep_rows:
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
with (mapped_dir / "representative_tiles.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mapped_rep_rows)

with (mapped_dir / "batch_topk_to_relu_mapping.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0].keys()))
    writer.writeheader()
    writer.writerows(mapping_rows)

(mapped_dir / "summary.json").write_text(json.dumps({
    "source_dir": str(source_dir),
    "mapped_dir": str(mapped_dir),
    "source_concept_sae": "batch_topk_20x",
    "steering_sae": "tcga_uni2_sae_relu_v1",
    "map_top_k": map_top_k,
    "n_concepts": len(mapped_concepts),
    "mapping_csv": str(mapped_dir / "batch_topk_to_relu_mapping.csv"),
}, indent=2) + "\\n")
print(json.dumps({"mapped_dir": str(mapped_dir), "n_concepts": len(mapped_concepts)}, indent=2))
PY
}

run_edit_for_sae() {
  local tag="$1"
  local sae_ckpt="$2"
  local sae_cfg="$3"
  local concept_dir="${CONCEPT_ROOT}/${tag}/${TASK}/high"

  if [[ ! -f "${concept_dir}/selected_concepts.json" || ! -f "${concept_dir}/representative_tiles.csv" ]]; then
    echo "[error] missing concept outputs for ${tag}: ${concept_dir}" >&2
    exit 1
  fi

  IFS=',' read -ra STRENGTHS <<< "${STRENGTHS_CSV}"
  for strength in "${STRENGTHS[@]}"; do
    local strength_clean="${strength//./p}"
    local out_dir="${OUT_ROOT}/edits/${tag}/low_to_high/strength_${strength_clean}"
    echo "[edit] ${tag}: low->high strength=${strength}"
    "$PY" scripts/run_progressive_region_edit.py \
      --task "${TASK}" \
      --region-bank-csv "${REGION_DIR}/region_bank.csv" \
      --edit-manifest "${REGION_DIR}/progressive_edit_manifest.json" \
      --out-dir "${out_dir}" \
      --concepts-json "${concept_dir}/selected_concepts.json" \
      --representative-tiles-csv "${concept_dir}/representative_tiles.csv" \
      --concept-class-label high \
      --concept-ranking-method attention_weighted \
      --concept-target-stat median \
      --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
      --concept-steering-mode "${CONCEPT_STEERING_MODE}" \
      --max-concepts "${TOP_CONCEPTS}" \
      --max-runs "${MAX_RUNS}" \
      --target-magnification 20 \
      --edit-support border_relaxed \
      --prototype-strength "${strength}" \
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
      --sae-ckpt "${sae_ckpt}" \
      --sae-cfg "${sae_cfg}" \
      --output-mode "${OUTPUT_MODE}" \
      --device "${DEVICE}"
  done
}

prepare_concepts_for_sae "relu" "${RELU_SAE_CKPT}" "${RELU_SAE_CFG}"
prepare_concepts_for_sae "batch_topk_20x" "${BATCH_TOPK_SAE_CKPT}" "${BATCH_TOPK_SAE_CFG}"
map_batch_topk_concepts_to_relu_steering_space

run_edit_for_sae "relu" "${RELU_SAE_CKPT}" "${RELU_SAE_CFG}"
run_edit_for_sae "batch_topk_20x_find_relu_steer" "${RELU_SAE_CKPT}" "${RELU_SAE_CFG}"

echo "[ok] SAE comparison written to ${OUT_ROOT}"
echo "     ReLU concept finding + ReLU steering:       ${OUT_ROOT}/edits/relu"
echo "     batch-TopK concept finding + ReLU steering: ${OUT_ROOT}/edits/batch_topk_20x_find_relu_steer"
