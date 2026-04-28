cd /common/users/wq50/SAE_path
export TCGA_FEATURES_BASE=/research/projects/mllab/WSI/TCGA_features

RUN_NAME="tcga_sae_advanced_v11"
RUN_ROOT="/common/users/wq50/SAE_path/runs/${RUN_NAME}"
MINING_ROOT="/common/users/wq50/SAE_path/sae_mining/${RUN_NAME}"
INDEX_JSON="/common/users/wq50/SAE_path/metadata/indexes/manifest_index.json"
GDC_CLIENT="/common/users/wq50/SAE_path/gdc/gdc-client"

rm -f "${MINING_ROOT}/sdf2/pass1_stats.json"
rm -f "${MINING_ROOT}/sdf2"/pass2_top_tiles_*.json

python -m scripts.latent_mining \
  --out_root "${MINING_ROOT}" \
  --run_name sdf2 \
  --mode both \
  --ckpt "${RUN_ROOT}/sdf2/sdf2_final.pt" \
  --stage sdf2 \
  --d_in 1536 \
  --latent_dim 12288 \
  --sdf_n_level2 256 \
  --sdf_coeff_simplex \
  --index_json "${INDEX_JSON}" \
  --slides_per_project 200 \
  --require_h5_exists \
  --tiles_per_slide 2048 \
  --chunk_tiles 512 \
  --select_strategy sdf_parent_balanced \
  --n_latents 128 \
  --parent_max_children_per_selected_parent 6 \
  --parent_preferred_children_per_selected_parent 4 \
  --parent_target_count -1 \
  --topn 50

PASS2_JSON="$(ls -t "${MINING_ROOT}/sdf2"/pass2_top_tiles_*.json | head -n 1)"

python -m scripts.export_latent_tiles \
  --out_root "${MINING_ROOT}" \
  --run_name sdf2 \
  --pass2_json "${PASS2_JSON}" \
  --gdc_client "${GDC_CLIENT}"
