cd /common/users/wq50/wsi_cf

python scripts/generate_representative_tiles.py \
  --association-root artifacts/concept_label_associations_all \
  --tasks kirc_grade,msi_coad_stad \
  --out-dir artifacts/representative_tiles_grade_msi \
  --top-latents-per-class 20 \
  --top-tiles-per-latent 25 \
  --batch-size 4096 \
  --device cpu \
  --write-per-task-files
