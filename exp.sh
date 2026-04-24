#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/common/users/wq50/envs/pace/bin/python"
RUNNER="/common/users/wq50/wsi_cf/scripts/export_hnscc_region_bank_10x.py"

"${PYTHON_BIN}" "${RUNNER}" \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --features-dir /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2 \
  --slides-dir /common/users/wq50/HNSCC/HNSCC_slides \
  --out-dir /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_1024_sample4 \
  --target-magnification 10 \
  --region-size 1024 \
  --grid-step-px 256 \
  --regions-total 40 \
  --regions-per-slide 1 \
  --seed 7 \
  --device cuda:0
