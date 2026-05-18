#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SAE_VARIANT="${SAE_VARIANT:-tcga_uni2_sae_relu_v1}"

CLASSIFIER_RUN_DIR="${CLASSIFIER_RUN_DIR:-artifacts/classifier_training/luad_lusc}"
EXP_ROOT="${EXP_ROOT:-artifacts/luad_lusc_1024_concept_remine_sweep}"
ASSOC_ROOT="${ASSOC_ROOT:-${EXP_ROOT}/concept_associations}"
CONCEPT_ROOT="${CONCEPT_ROOT:-${EXP_ROOT}/concepts}"
REGION_ROOT="${REGION_ROOT:-${EXP_ROOT}/regions}"
SWEEP_ROOT="${SWEEP_ROOT:-${EXP_ROOT}/sweeps}"

TOP_CONCEPTS="${TOP_CONCEPTS:-5}"
REP_TILES="${REP_TILES:-50}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
MAX_REGIONS="${MAX_REGIONS:-10}"
STEPS="${STEPS:-30}"
SETTINGS="${SETTINGS:-default,low_strength,high_strength,very_high_strength,stronger_edit_preserve,early_steer,border_relaxed}"
OUTPUT_MODE="${OUTPUT_MODE:-minimal}"

# Smoke knobs:
#   MAX_REGIONS=1 TOP_CONCEPTS=1 SETTINGS=default STEPS=5 bash examples/luad_lusc/03_luad_lusc_remine_and_1024_sweep.sh

mkdir -p "${EXP_ROOT}" "${ASSOC_ROOT}" "${CONCEPT_ROOT}" "${REGION_ROOT}" "${SWEEP_ROOT}"

echo "[1/5] Prepare LUAD/LUSC concept-label associations with ${SAE_VARIANT}"
"$PY" scripts/prepare_classifier_concept_associations.py \
  --task-name luad_lusc \
  --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
  --out-root "${ASSOC_ROOT}" \
  --include-labels LUAD,LUSC \
  --sae-variant "${SAE_VARIANT}" \
  --batch-size 4096 \
  --device "${DEVICE}" \
  --skip-existing

echo "[2/5] Re-mine LUAD/LUSC concepts: labels-only and attention-aware"
for mode in labels_only attention_aware; do
  for label in LUAD LUSC; do
    extra_args=()
    if [[ "${mode}" == "attention_aware" ]]; then
      extra_args+=(--classifier-run-dir "${CLASSIFIER_RUN_DIR}" --slides-csv "${ASSOC_ROOT}/luad_lusc/cohort_slides.csv")
    fi
    "$PY" scripts/find_label_concepts.py \
      --task luad_lusc \
      --association-root "${ASSOC_ROOT}" \
      --class-label "${label}" \
      --mode "${mode}" \
      --backend mil \
      --concept-quality-mode morphology \
      --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
      "${extra_args[@]}" \
      --out-dir "${CONCEPT_ROOT}/${mode}" \
      --top-concepts "${TOP_CONCEPTS}" \
      --candidate-latents 150 \
      --top-tiles-per-concept "${REP_TILES}" \
      --batch-size 4096 \
      --sae-variant "${SAE_VARIANT}" \
      --device "${DEVICE}" \
      --skip-existing
  done
done

"$PY" - <<PY
import json
from pathlib import Path
root = Path("${EXP_ROOT}")
concept_root = Path("${CONCEPT_ROOT}")
summary = {}
for mode in ["labels_only", "attention_aware"]:
    for label in ["LUAD", "LUSC"]:
        p = concept_root / mode / "luad_lusc" / label / "summary.json"
        if p.exists():
            data = json.loads(p.read_text())
            summary[f"{mode}/{label}"] = {
                "concept_cards": data.get("concept_cards"),
                "representative_tiles": data.get("representative_tiles"),
                "effective_mode": data.get("effective_mode"),
                "concept_quality_mode": data.get("concept_quality_mode"),
                "sae_variant": data.get("args", {}).get("sae_variant"),
            }
out = root / "concept_mining_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\\n")
print(f"[ok] wrote {out}")
PY

echo "[3/5] Select 10 source regions per direction at 1024 / 20x"
declare -A REGION_DIRS
REGION_DIRS[LUAD_to_LUSC]="${REGION_ROOT}/luad_to_lusc_1024_regions${MAX_REGIONS}"
REGION_DIRS[LUSC_to_LUAD]="${REGION_ROOT}/lusc_to_luad_1024_regions${MAX_REGIONS}"

for direction in LUAD_to_LUSC LUSC_to_LUAD; do
  if [[ "${direction}" == "LUAD_to_LUSC" ]]; then
    source_label="LUAD"
    target_label="LUSC"
  else
    source_label="LUSC"
    target_label="LUAD"
  fi
  out_dir="${REGION_DIRS[${direction}]}"
  "$PY" scripts/find_regions.py \
    --mode attention \
    --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
    --source-label "${source_label}" \
    --target-label "${target_label}" \
    --slides-root /research/projects/mllab/WSI/TCGA_features \
    --target-magnification 20 \
    --region-size 1024 \
    --grid-step-px 256 \
    --max-regions "${MAX_REGIONS}" \
    --max-candidates-per-slide 8 \
    --attention-percentile 85 \
    --min-selected-cells 2 \
    --max-selected-cells 6 \
    --target-importance-mass 0.45 \
    --min-tissue 0.45 \
    --min-dark-fraction 0.04 \
    --min-saturation-fraction 0.04 \
    --out-dir "${out_dir}" \
    --sae-variant "${SAE_VARIANT}" \
    --device "${DEVICE}"
done

"$PY" - <<PY
import json
from pathlib import Path
root = Path("${EXP_ROOT}")
summary = {}
for name, path in {
    "LUAD_to_LUSC": Path("${REGION_DIRS[LUAD_to_LUSC]}"),
    "LUSC_to_LUAD": Path("${REGION_DIRS[LUSC_to_LUAD]}"),
}.items():
    p = path / "summary.json"
    if p.exists():
        summary[name] = json.loads(p.read_text())
out = root / "region_selection_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\\n")
print(f"[ok] wrote {out}")
PY

echo "[4/5] Run concept steering sweeps in both directions"
"$PY" scripts/run_concept_steering_sweep.py \
  --source-region-bank-csv "${REGION_DIRS[LUAD_to_LUSC]}/region_bank.csv" \
  --source-edit-manifest "${REGION_DIRS[LUAD_to_LUSC]}/progressive_edit_manifest.json" \
  --source-label LUAD \
  --target-label LUSC \
  --concept-root "${CONCEPT_ROOT}" \
  --out-dir "${SWEEP_ROOT}/luad_to_lusc" \
  --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
  --settings "${SETTINGS}" \
  --top-concepts "${TOP_CONCEPTS}" \
  --max-regions "${MAX_REGIONS}" \
  --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
  --steps "${STEPS}" \
  --output-mode "${OUTPUT_MODE}" \
  --sae-variant "${SAE_VARIANT}" \
  --device "${DEVICE}"

"$PY" scripts/run_concept_steering_sweep.py \
  --source-region-bank-csv "${REGION_DIRS[LUSC_to_LUAD]}/region_bank.csv" \
  --source-edit-manifest "${REGION_DIRS[LUSC_to_LUAD]}/progressive_edit_manifest.json" \
  --source-label LUSC \
  --target-label LUAD \
  --concept-root "${CONCEPT_ROOT}" \
  --out-dir "${SWEEP_ROOT}/lusc_to_luad" \
  --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
  --settings "${SETTINGS}" \
  --top-concepts "${TOP_CONCEPTS}" \
  --max-regions "${MAX_REGIONS}" \
  --concept-target-top-k "${CONCEPT_TARGET_TOP_K}" \
  --steps "${STEPS}" \
  --output-mode "${OUTPUT_MODE}" \
  --sae-variant "${SAE_VARIANT}" \
  --device "${DEVICE}"

"$PY" - <<PY
import csv
import json
from pathlib import Path
import numpy as np

root = Path("${EXP_ROOT}")
sweep_root = Path("${SWEEP_ROOT}")

def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

manifest_rows = []
result_rows = []
for direction in ["luad_to_lusc", "lusc_to_luad"]:
    manifest_rows.extend(read_csv(sweep_root / direction / "sweep_manifest.csv"))
    result_rows.extend(read_csv(sweep_root / direction / "sweep_results.csv"))
write_csv(root / "sweep_manifest.csv", manifest_rows)
write_csv(root / "sweep_results.csv", result_rows)

groups = {}
bucket = {}
for row in result_rows:
    key = "|".join([row.get("direction", ""), row.get("concept_mode", ""), row.get("concept_rank", ""), row.get("setting", "")])
    bucket.setdefault(key, []).append(row)
for key, rows in bucket.items():
    vals = np.asarray([float(r.get("generated_target_delta", 0.0)) for r in rows], dtype=np.float32)
    flips = np.asarray([float(r.get("generated_flip_to_target", 0.0)) for r in rows], dtype=np.float32)
    groups[key] = {
        "n": int(len(rows)),
        "mean_generated_target_delta": float(vals.mean()) if vals.size else 0.0,
        "median_generated_target_delta": float(np.median(vals)) if vals.size else 0.0,
        "flip_rate": float(flips.mean()) if flips.size else 0.0,
    }
summary = {"n_runs_planned": len(manifest_rows), "n_runs_completed": len(result_rows), "groups": groups}
(root / "summary_by_direction_mode_concept_setting.json").write_text(json.dumps(summary, indent=2) + "\\n")
print(f"[ok] wrote combined sweep outputs under {root}")
PY

echo "[5/5] Done"
echo "[ok] concept summary: ${EXP_ROOT}/concept_mining_summary.json"
echo "[ok] region summary:  ${EXP_ROOT}/region_selection_summary.json"
echo "[ok] sweep manifest:  ${EXP_ROOT}/sweep_manifest.csv"
echo "[ok] sweep results:   ${EXP_ROOT}/sweep_results.csv"
echo "[ok] sweeps:          ${SWEEP_ROOT}"
