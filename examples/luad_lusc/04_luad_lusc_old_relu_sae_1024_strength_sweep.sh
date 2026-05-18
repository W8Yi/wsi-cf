#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

# Same LUAD/LUSC 1024 concept-remine and steering workflow as
# 03_luad_lusc_remine_and_1024_sweep.sh, but pinned to the older ReLU SAE.
#
# Strengths:
#   low_strength = 0.4
#   default      = 0.8
#   strength_1   = 1.0

SAE_VARIANT="${SAE_VARIANT:-relu_sae_base}" \
EXP_ROOT="${EXP_ROOT:-artifacts/luad_lusc_1024_concept_remine_sweep_old_relu_sae_base_strength_0p4_0p8_1p0}" \
SETTINGS="${SETTINGS:-low_strength,default,strength_1}" \
TOP_CONCEPTS="${TOP_CONCEPTS:-5}" \
MAX_REGIONS="${MAX_REGIONS:-10}" \
CONCEPT_TARGET_TOP_K="${CONCEPT_TARGET_TOP_K:-5}" \
STEPS="${STEPS:-30}" \
OUTPUT_MODE="${OUTPUT_MODE:-minimal}" \
bash examples/luad_lusc/03_luad_lusc_remine_and_1024_sweep.sh
