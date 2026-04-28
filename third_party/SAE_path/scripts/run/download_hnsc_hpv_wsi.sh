#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python metadata/labels/scripts/build_hnsc_hpv_wsi_list.py

python utils/wsi_downloader.py \
  --slides-file metadata/manifests/hnsc_hpv_wsi/slide_keys.txt \
  --index metadata/indexes/manifest_index.json \
  --out wsi/hnsc_hpv
