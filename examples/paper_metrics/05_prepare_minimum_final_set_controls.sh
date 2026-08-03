#!/usr/bin/env bash
set -euo pipefail

# Prepare the recommended minimum final control set:
#   1. COAD tumor -> normal
#   2. PRAD morphology pattern 4 -> well-formed
#
# By default this only builds region banks and transition manifests. Set
# RUN_EDITS=1 and RUN_EVAL=1 to generate/evaluate after the current long PRAD
# p4->p5 job finishes or when a GPU is available.

cd /common/users/wq50/wsi_cf

DEVICE="${DEVICE:-cuda:3}"
OUT_ROOT="${OUT_ROOT:-paper_outputs/prediction_transition_benchmark_test_only_unbalanced}"
BANK_ROOT="${BANK_ROOT:-artifacts/prediction_transition_region_banks_test_only_unbalanced}"
MAX_REGIONS="${MAX_REGIONS:-0}"
MAX_REGIONS_PER_SLIDE="${MAX_REGIONS_PER_SLIDE:-5}"
RUN_EDITS="${RUN_EDITS:-0}"
RUN_EVAL="${RUN_EVAL:-0}"

echo "[prepare] COAD tumor -> normal region bank" >&2
TASKS=normal_tumor \
NORMAL_TUMOR_TASKS=coad_normal_tumor \
NORMAL_TUMOR_DIRECTIONS=tumor_to_normal \
OUT_ROOT="${BANK_ROOT}" \
DEVICE="${DEVICE}" \
examples/paper_metrics/00_prepare_test_only_unbalanced_region_banks.sh

echo "[prepare] PRAD morphology pattern 4 -> well-formed region bank" >&2
TASKS=prad_morphology_group \
PRAD_MORPH_DIRECTIONS=p4_to_well \
OUT_ROOT="${BANK_ROOT}" \
DEVICE="${DEVICE}" \
examples/paper_metrics/00_prepare_test_only_unbalanced_region_banks.sh

echo "[benchmark] build manifests; generation/eval controlled by RUN_EDITS/RUN_EVAL" >&2
TASKS="normal_tumor prad_morphology_group" \
NORMAL_TUMOR_TASKS=coad_normal_tumor \
NORMAL_TUMOR_DIRECTIONS=tumor_to_normal \
NORMAL_REGION_ROOT="${BANK_ROOT}/normal_tumor" \
PRAD_MORPH_DIRECTIONS=p4_to_well \
PRAD_MORPH_REGION_ROOT="${BANK_ROOT}/prad_morphology_group" \
OUT_ROOT="${OUT_ROOT}" \
MAX_REGIONS="${MAX_REGIONS}" \
MAX_REGIONS_PER_SLIDE="${MAX_REGIONS_PER_SLIDE}" \
DEVICE="${DEVICE}" \
RUN_EDITS="${RUN_EDITS}" \
RUN_EVAL="${RUN_EVAL}" \
examples/paper_metrics/02_run_prediction_transition_test_only_unbalanced.sh

