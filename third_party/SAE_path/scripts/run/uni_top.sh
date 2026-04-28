cd /common/users/wq50/SAE_path

for start in $(seq 0 128 1535); do
  end=$((start + 128))
  if [ "$end" -gt 1536 ]; then end=1536; fi
  python -m scripts.uni_dim_top_tiles \
    --manifest metadata/manifests/sae_manifests_tcga_patient_train_test_90_10.json \
    --split test \
    --axes-range ${start}:${end} \
    --signs pos,neg \
    --top-n 50 \
    --tiles-per-slide 256 \
    --max-slides 300 \
    --zscore \
    --per-slide-cap 3 \
    --min-coord-dist-px 512 \
    --out-json outputs/uni_axis_top_tiles/test_axes_${start}_${end}.json
done

python -m scripts.aggregate_uni_dim_top_tiles \
  --inputs "outputs/uni_axis_top_tiles/test_axes_*.json" \
  --out-json outputs/uni_axis_top_tiles/test_merged.json \
  --out-csv outputs/uni_axis_top_tiles/test_ranking.csv

python -m scripts.export_uni_dim_top_tiles \
  --input-json outputs/uni_axis_top_tiles/test_merged.json \
  --out-dir outputs/uni_axis_top_tiles_export/test_top20 \
  --index-json metadata/indexes/manifest_index.json \
  --gdc-client /common/users/wq50/SAE_path/gdc/gdc-client \
  --sort-by diversity \
  --max-axes 50 \
  --signs pos,neg \
  --top-n 50 \
  --tile-size 256 \
  --vis-size 256 \
  --ncols 10 \
  --skip-if-exists
