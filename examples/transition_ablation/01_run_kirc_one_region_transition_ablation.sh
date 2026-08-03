#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-kirc_normal_tumor}"
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"
MAX_CONCEPTS="${MAX_CONCEPTS:-3}"
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAKE_SHEET="${MAKE_SHEET:-1}"

RUN_MANIFEST="${RUN_MANIFEST:-paper_outputs/normal_to_tumor_after_images_showcase_sae_cell_fraction_sweep_center_commit/kirc_normal_tumor/TCGA-B2-3924-11A-01-TS1__normal_to_tumor__mag_20p0__gx_78__gy_3__to_tumor_concepts__attn_frac_1p00/run_manifest.json}"
SOURCE_REGION_BANK="${SOURCE_REGION_BANK:-artifacts/normal_to_tumor_regions_showcase_sae_cell_fraction_sweep_diverse/kirc_normal_tumor/region_bank.csv}"
POLICY_DIR="${POLICY_DIR:-configs/edit_policies/transition_ablation}"
WORK_ROOT="${WORK_ROOT:-artifacts/transition_ablation/kirc_one_region}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/transition_ablation/kirc_one_region}"
CONCEPT_DIR="${CONCEPT_DIR:-artifacts/concept_discovery_normal_tumor_${SAE_VARIANT}/kirc_normal_tumor/labels/tumor}"

mkdir -p "${WORK_ROOT}" "${OUT_ROOT}"

echo "[setup] building one-region inputs from ${RUN_MANIFEST}" >&2
readarray -t input_lines < <("${PY}" - <<'PY' "${RUN_MANIFEST}" "${SOURCE_REGION_BANK}" "${WORK_ROOT}"
import csv
import json
import sys
from pathlib import Path

run_manifest = Path(sys.argv[1])
source_region_bank = Path(sys.argv[2])
work_root = Path(sys.argv[3])
work_root.mkdir(parents=True, exist_ok=True)

run = json.loads(run_manifest.read_text())
region_id = str(run["region_id"])
target_cells = run.get("target_cells") or run.get("runtime_target_cells")
if not isinstance(target_cells, list) or not target_cells:
    raise SystemExit(f"No target_cells in {run_manifest}")

candidate_banks = [source_region_bank]
candidate_banks.extend(sorted(Path("artifacts").glob("normal_to_tumor_regions*/kirc_normal_tumor/region_bank.csv")))
match = None
fieldnames = None
matched_bank = None
for bank in candidate_banks:
    if not bank.exists():
        continue
    with bank.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        for row in rows:
            if str(row.get("region_id", "")) == region_id:
                match = row
                fieldnames = reader.fieldnames
                matched_bank = bank
                break
    if match is not None:
        break
if match is None or fieldnames is None:
    raise SystemExit(f"Could not find region_id={region_id} in {source_region_bank} or artifacts/normal_to_tumor_regions*")

region_bank_out = work_root / "region_bank.csv"
with region_bank_out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(match)

run_id = f"{region_id}__to_tumor_concepts__transition_ablation"
manifest_out = work_root / "edit_manifest.json"
manifest_out.write_text(
    json.dumps(
        [
            {
                "run_id": run_id,
                "region_id": region_id,
                "target_cells": target_cells,
                "source_run_manifest": str(run_manifest),
                "source_region_bank": str(matched_bank),
                "source_run_id": str(run.get("run_id", "")),
            }
        ],
        indent=2,
    )
)
source_image = Path(str(match["image_path"]))
print(f"REGION_ID={region_id}")
print(f"RUN_ID={run_id}")
print(f"REGION_BANK={region_bank_out}")
print(f"EDIT_MANIFEST={manifest_out}")
print(f"SOURCE_IMAGE={source_image}")
print(f"N_TARGET_CELLS={len(target_cells)}")
PY
)

for line in "${input_lines[@]}"; do
  echo "[setup] ${line}" >&2
  eval "${line}"
done

if [[ -z "${POLICIES:-}" ]]; then
  mapfile -t policy_paths < <(find "${POLICY_DIR}" -maxdepth 1 -type f -name '*.json' | sort)
else
  policy_paths=()
  for name in ${POLICIES}; do
    if [[ "${name}" == *.json ]]; then
      policy_paths+=("${name}")
    else
      policy_paths+=("${POLICY_DIR}/${name}.json")
    fi
  done
fi

if [[ "${#policy_paths[@]}" -eq 0 ]]; then
  echo "[error] no policies found in ${POLICY_DIR}" >&2
  exit 2
fi

skip_args=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  skip_args+=(--skip-existing)
fi

for policy in "${policy_paths[@]}"; do
  if [[ ! -f "${policy}" ]]; then
    echo "[error] missing policy: ${policy}" >&2
    exit 2
  fi
  name="$(basename "${policy}" .json)"
  out_dir="${OUT_ROOT}/${name}"
  echo "[run] ${name} -> ${out_dir}" >&2
  cmd=(
    "${PY}" scripts/run_progressive_region_edit.py
    --task "${TASK}"
    --region-bank-csv "${REGION_BANK}"
    --edit-manifest "${EDIT_MANIFEST}"
    --out-dir "${out_dir}"
    --edit-policy "${policy}"
    --concepts-json "${CONCEPT_DIR}/selected_concepts.json"
    --representative-tiles-csv "${CONCEPT_DIR}/representative_tiles.csv"
    --concept-class-label tumor
    --concept-ranking-method attention_weighted
    --concept-target-stat median
    --concept-target-top-k "${CONCEPT_TARGET_TOP_K}"
    --concept-steering-mode prototype_vector
    --max-concepts "${MAX_CONCEPTS}"
    --max-runs 1
    --target-magnification 20
    --sae-variant "${SAE_VARIANT}"
    --output-mode "${OUTPUT_MODE}"
    --device "${DEVICE}"
    "${skip_args[@]}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
done

if [[ "${DRY_RUN}" != "1" && "${MAKE_SHEET}" == "1" ]]; then
  "${PY}" scripts/make_transition_ablation_contact_sheet.py \
    --out-root "${OUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --source "${SOURCE_IMAGE}" \
    --output "${OUT_ROOT}/transition_ablation_contact_sheet.png"
fi

echo "[done] outputs: ${OUT_ROOT}" >&2

