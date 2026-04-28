#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="/common/users/wq50/envs/pace/bin/python"

"$PYTHON_BIN" concept_steer/visualize_hnsc_hpv_sae_neuron_tiles.py \
  --tiles_csv runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h/sae_neuron_pipeline/split_0/top_neuron_tiles.csv \
  --wsi_dir wsi/hnsc_hpv \
  --out_dir runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h/sae_neuron_pipeline/split_0/neuron_tile_visualizations \
  --tile_size_20x 256 \
  --out_tile_size 256 \
  --sheet_ncols 10 \
  --sheet_nrows 8 \
  --max_open_slides 8
