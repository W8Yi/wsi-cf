#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-7}"
OUT_DIR="${OUT_DIR:-paper_example/random_sae_top_concepts_center2x2}"
EXPORT_DIR="${EXPORT_DIR:-/common/users/wq50/wsi-sae/exports/tcga_uni2_sae_relu_v1}"

# One sampled 1024x1024 tissue region, ten distinct random concepts from the
# exported SAE top-concept pool, and the same editable center 2x2 for each run.
cmd=(
  "${PYTHON_BIN}" scripts/run_export_random_concept_steer.py
  --export-dir "${EXPORT_DIR}"
  --out-dir "${OUT_DIR}"
  --n-slides 1
  --concepts-per-slide 10
  --settings default
  --prototype-strengths "${PROTOTYPE_STRENGTH:-0.9}"
  --prototype-top-k "${PROTOTYPE_TOP_K:-5}"
  --steps "${STEPS:-30}"
  --output-mode "${OUTPUT_MODE:-debug}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  cmd+=(--skip-existing)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi

printf '[random SAE concept example] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

printf '\n[random SAE concept example] outputs:\n'
printf '  %s\n' "${OUT_DIR}/region_bank.csv"
printf '  %s\n' "${OUT_DIR}/sweep_manifest.csv"
printf '  %s\n' "${OUT_DIR}/browse/slide_01"
