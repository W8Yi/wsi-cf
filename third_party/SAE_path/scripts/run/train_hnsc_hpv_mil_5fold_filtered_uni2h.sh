#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="/common/users/wq50/envs/pace/bin/python"

"$PYTHON_BIN" scripts/train_hnsc_hpv_mil_5fold.py \
  --splits_dir metadata/manifests/hnsc_hpv_5fold \
  --h5_dir extracted_features/hnsc_hpv_filtered_uni2h \
  --out_dir runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80 \
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
