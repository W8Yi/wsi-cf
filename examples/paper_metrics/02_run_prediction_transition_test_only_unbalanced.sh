#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

BANK_ROOT="${BANK_ROOT:-artifacts/prediction_transition_region_banks_test_only_unbalanced}"

export HNSCC_REGION_BANK_CSV="${HNSCC_REGION_BANK_CSV:-${BANK_ROOT}/hnscc_hpv/region_bank.csv}"
export NORMAL_REGION_ROOT="${NORMAL_REGION_ROOT:-${BANK_ROOT}/normal_tumor}"
export NORMAL_REGION_FALLBACK_ROOTS="${NORMAL_REGION_FALLBACK_ROOTS:-}"
export PRAD_REGION_ROOT="${PRAD_REGION_ROOT:-${BANK_ROOT}/prad_grade_group}"
export PRAD_MORPH_REGION_ROOT="${PRAD_MORPH_REGION_ROOT:-${BANK_ROOT}/prad_morphology_group}"

export OUT_ROOT="${OUT_ROOT:-paper_outputs/prediction_transition_benchmark_test_only_unbalanced}"
export MAX_SLIDES="${MAX_SLIDES:-0}"
export MAX_REGIONS_PER_SLIDE="${MAX_REGIONS_PER_SLIDE:-5}"
export MAX_REGIONS="${MAX_REGIONS:-0}"
export SCORE_SCOPE="${SCORE_SCOPE:-local_region}"
export LOCAL_SOURCE="${LOCAL_SOURCE:-source_image}"

examples/paper_metrics/01_run_prediction_transition_benchmark.sh "$@"
