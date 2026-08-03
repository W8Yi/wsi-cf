#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

# Stronger normal -> tumor rerun for pathologist-facing after-images:
# - use the same ReLU SAE family as the HNSCC showcase,
# - remine tumor concepts in that SAE space,
# - choose less-confident normal source slides,
# - edit a larger fraction of each 2048 region.

RUN_TAG="${RUN_TAG:-showcase_sae_more_cells_less_confident}" \
SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}" \
REFRESH_INPUTS="${REFRESH_INPUTS:-0}" \
REFRESH_REGIONS="${REFRESH_REGIONS:-1}" \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
TASKS="${TASKS:-luad_normal_tumor coad_normal_tumor kirc_normal_tumor brca_normal_tumor}" \
MAX_RUNS="${MAX_RUNS:-5}" \
MAX_REGIONS="${MAX_REGIONS:-${MAX_RUNS:-5}}" \
MAX_CANDIDATES_PER_SLIDE="${MAX_CANDIDATES_PER_SLIDE:-4}" \
ATTENTION_PERCENTILE="${ATTENTION_PERCENTILE:-50}" \
MIN_SELECTED_CELLS="${MIN_SELECTED_CELLS:-16}" \
MAX_SELECTED_CELLS="${MAX_SELECTED_CELLS:-32}" \
TARGET_IMPORTANCE_MASS="${TARGET_IMPORTANCE_MASS:-0.90}" \
MIN_LABEL_CONFIDENCE="${MIN_LABEL_CONFIDENCE:-0.50}" \
MAX_LABEL_CONFIDENCE="${MAX_LABEL_CONFIDENCE:-0.9995}" \
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/concept_discovery_normal_tumor_relu_sae_base}" \
REGION_ROOT="${REGION_ROOT:-artifacts/normal_to_tumor_regions_${RUN_TAG:-showcase_sae_more_cells_less_confident}}" \
OUT_ROOT="${OUT_ROOT:-artifacts/normal_to_tumor_after_images_${RUN_TAG:-showcase_sae_more_cells_less_confident}}" \
examples/normal_tumor/01_run_normal_to_tumor_after_images.sh
