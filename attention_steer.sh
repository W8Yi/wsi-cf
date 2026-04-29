cd /common/users/wq50/wsi_cf

python scripts/visualize_attention.py \
  --task hnscc_hpv \
  --backend clam \
  --max-slides 1 \
  --thumbnail-max-side 4096 \
  --out-dir artifacts/test_attention_vis_clam \
  --device cuda:0


python scripts/find_regions.py \
  --mode attention \
  --backend clam \
  --target-magnification 20 \
  --region-size 2048 \
  --final-regions-per-label 1 \
  --max-slides-per-label 1 \
  --out-dir artifacts/test_attention_regions_20x \
  --device cuda:0

python scripts/run_progressive_region_edit.py \
  --task hnscc_hpv \
  --region-bank-csv artifacts/test_attention_regions_20x/region_bank.csv \
  --edit-manifest artifacts/test_attention_regions_20x/progressive_edit_manifest.json \
  --direction hpv_neg \
  --max-runs 1 \
  --output-mode debug \
  --device cuda:0 \
  --out-dir artifacts/test_attention_region_steer_hpvneg
