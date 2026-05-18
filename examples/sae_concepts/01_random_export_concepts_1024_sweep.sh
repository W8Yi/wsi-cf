#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-artifacts/random_tcga_export_concepts_1024_sweep}"
EXPORT_DIR="${EXPORT_DIR:-/common/users/wq50/wsi-sae/exports/tcga_uni2_sae_relu_v1}"

# Smoke example:
#   N_SLIDES=1 CONCEPTS_PER_SLIDE=1 SETTINGS=default STEPS=5 DRY_RUN=--dry-run bash examples/sae_concepts/01_random_export_concepts_1024_sweep.sh
#
# Strength sweep example:
#   PROTOTYPE_STRENGTHS=0.4,0.8,1.0 SETTINGS=default bash examples/sae_concepts/01_random_export_concepts_1024_sweep.sh

"$PY" scripts/run_export_random_concept_steer.py \
  --export-dir "${EXPORT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --n-slides "${N_SLIDES:-10}" \
  --concepts-per-slide "${CONCEPTS_PER_SLIDE:-5}" \
  --settings "${SETTINGS:-default,loose_context,stronger_preserve,early_steer,border_relaxed}" \
  --prototype-strengths "${PROTOTYPE_STRENGTHS:-0.8}" \
  --prototype-top-k "${PROTOTYPE_TOP_K:-5}" \
  --steps "${STEPS:-30}" \
  --device "${DEVICE}" \
  ${DRY_RUN:-}

echo "[ok] random export-concept sweep: ${OUT_DIR}"
