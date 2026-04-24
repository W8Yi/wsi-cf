mkdir -p /common/users/wq50/wsi_cf/artifacts/edit_manifests

cat > /common/users/wq50/wsi_cf/artifacts/edit_manifests/sample_region_complex_targets.json <<'JSON'
[
  {
    "run_id": "cv7263_complex_multi_target_to_hpv_pos",
    "region_id": "TCGA-CV-7263-01Z-00-DX1__mag_10p0__x_3868__y_33550",
    "target_cells": [
      {"gx": 2, "gy": 2},
      {"gx": 3, "gy": 2},
      {"gx": 4, "gy": 2},
      {"gx": 5, "gy": 2},
      {"gx": 5, "gy": 3},
      {"gx": 5, "gy": 4},
      {"gx": 4, "gy": 4},
      {"gx": 3, "gy": 4},
      {"gx": 2, "gy": 5},
      {"gx": 6, "gy": 5}
    ],
    "source_method": "manual",
    "note": "complex multi-window sample region test"
  }
]
JSON

/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/run_progressive_region_edit.py \
  --region-bank-csv /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_2048_sample4/region_bank.csv \
  --edit-manifest /common/users/wq50/wsi_cf/artifacts/edit_manifests/sample_region_complex_targets.json \
  --out-dir /common/users/wq50/wsi_cf/artifacts/sample_region_progressive_edit_cv7263_complex \
  --direction hpv_pos \
  --prototype-strength 0.8 \
  --steer-blend 1.0 \
  --preserve-edit-strength 0.05 \
  --preserve-visited-strength 0.95 \
  --preserve-fresh-context-strength 0.35 \
  --mid-steer-start-ratio 0.5 \
  --mid-steer-end-ratio 1.0 \
  --mid-steer-alpha-start 0.5 \
  --mid-steer-alpha-end 1.0 \
  --mid-steer-alpha-schedule linear \
  --output-mode debug \
  --device cuda:0
