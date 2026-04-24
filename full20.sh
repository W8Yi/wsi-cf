mkdir -p /common/users/wq50/wsi_cf/artifacts/edit_manifests

/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/export_hnscc_region_bank_10x.py \
  --out-dir /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_20x_2048_sample2 \
  --target-magnification 20 \
  --region-size 2048 \
  --grid-step-px 256 \
  --regions-total 2 \
  --regions-per-slide 1 \
  --min-tissue 0.35 \
  --max-region-tries 64 \
  --seed 7 \
  --device cuda:0 \
  --dtype fp16

REGION_ID=$(
/common/users/wq50/envs/pace/bin/python - <<'PY'
import csv
from pathlib import Path
csv_path = Path("/common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_20x_2048_sample2/region_bank.csv")
with csv_path.open() as f:
    rows = list(csv.DictReader(f))
print(rows[0]["region_id"])
PY
)

cat > /common/users/wq50/wsi_cf/artifacts/edit_manifests/sample_region_20x_complex.json <<JSON
[
  {
    "run_id": "sample_20x_progressive_test",
    "region_id": "${REGION_ID}",
    "target_cells": [
      {"gx": 2, "gy": 2},
      {"gx": 3, "gy": 2},
      {"gx": 5, "gy": 3},
      {"gx": 2, "gy": 5},
      {"gx": 6, "gy": 5}
    ],
    "source_method": "manual",
    "note": "20x progressive sample-region test"
  }
]
JSON

/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/run_progressive_region_edit.py \
  --region-bank-csv /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_20x_2048_sample2/region_bank.csv \
  --edit-manifest /common/users/wq50/wsi_cf/artifacts/edit_manifests/sample_region_20x_complex.json \
  --out-dir /common/users/wq50/wsi_cf/artifacts/sample_region_progressive_edit_20x \
  --direction hpv_pos \
  --device cuda:0
