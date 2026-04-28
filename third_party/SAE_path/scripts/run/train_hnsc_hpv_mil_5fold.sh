#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="/common/users/wq50/envs/pace/bin/python"
SPLITS_DIR="${SPLITS_DIR:-metadata/manifests/hnsc_hpv_5fold}"
H5_DIR="${H5_DIR:-extracted_features/hnsc_hpv_filtered_uni2h}"
OUT_DIR="${OUT_DIR:-runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h}"

"$PYTHON_BIN" scripts/train_hnsc_hpv_mil_5fold.py \
  --splits_dir "$SPLITS_DIR" \
  --h5_dir "$H5_DIR" \
  --out_dir "$OUT_DIR" \
  --model attention \
  --embed_dim 1536 \
  --hidden_dim 512 \
  --attn_dim 256 \
  --dropout 0.25 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --epochs 20 \
  --max_tiles_train 2048 \
  --max_tiles_eval 4096 \
  --normalize none \
  --device auto
