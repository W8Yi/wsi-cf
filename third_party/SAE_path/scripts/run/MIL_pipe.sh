#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/SAE_path

export CUDA_VISIBLE_DEVICES=3
PYTHON_BIN="/common/users/wq50/envs/pace/bin/python"

RUN_DIR="/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80"
SPLIT="split_0"
ROWS_CSV="${RUN_DIR}/${SPLIT}/test_predictions.csv"

SAE_DIR="/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_20x_interp"
SAE_CKPT="${SAE_DIR}/batch_topk_final.pt"
SAE_CFG="${SAE_DIR}/run_config.json"

OUT_DIR="${RUN_DIR}/sae_neuron_pipeline_batch_topk/${SPLIT}*"

# 1) MIL attention + SAE neuron summary/prototypes
"$PYTHON_BIN" concept_steer/run_hnsc_hpv_sae_neuron_pipeline.py \
  --run_dir "$RUN_DIR" \
  --split_name "$SPLIT" \
  --checkpoint_name final.pt \
  --rows_csv "$ROWS_CSV" \
  --data_split test \
  --h5_dir /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features \
  --sae_ckpt "$SAE_CKPT" \
  --sae_cfg "$SAE_CFG" \
  --out_dir "$OUT_DIR" \
  --batch_size 4096 \
  --top_k_neurons 20 \
  --top_tiles_per_neuron 50 \
  --local_top_tiles_per_slide 128 \
  --device auto

# 2) Tile visualization (individual + sheets)
"$PYTHON_BIN" concept_steer/visualize_hnsc_hpv_sae_neuron_tiles.py \
  --tiles_csv "${OUT_DIR}/top_neuron_tiles.csv" \
  --wsi_dir /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/slides \
  --out_dir "${OUT_DIR}/neuron_tile_visualizations" \
  --tile_size_20x 256 \
  --out_tile_size 256 \
  --sheet_ncols 10 \
  --sheet_nrows 8 \
  --max_open_slides 8
