cd /common/users/wq50/wsi_cf

python scripts/find_regions.py \
  --mode manual \
  --region-image artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048/region_top_right_2048.png \
  --region-id showcase_manual_2048 \
  --out-dir artifacts/test_manual_region_export \
  --device cuda:0

python scripts/run_progressive_region_edit.py \
  --task hnscc_hpv \
  --region-bank-csv artifacts/test_manual_region_export/region_bank.csv \
  --edit-manifest artifacts/test_manual_region_export/progressive_edit_manifest.json \
  --direction hpv_neg \
  --output-mode debug \
  --device cuda:0 \
  --out-dir artifacts/test_manual_region_progressive_hpvneg
