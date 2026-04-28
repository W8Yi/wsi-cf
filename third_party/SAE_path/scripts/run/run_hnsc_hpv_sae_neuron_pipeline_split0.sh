#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="/common/users/wq50/envs/pace/bin/python"

"$PYTHON_BIN" concept_steer/run_hnsc_hpv_sae_neuron_pipeline.py \
  --run_dir runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h \
  --split_name split_0 \
  --checkpoint_name final.pt \
  --rows_csv runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h/split_0/test_predictions.csv \
  --data_split test \
  --h5_dir extracted_features/hnsc_hpv_filtered_uni2h \
  --sae_ckpt runs/relu_sae_tcga_hnscc/relu_final.pt \
  --sae_cfg runs/relu_sae_tcga_hnscc/run_config.json \
  --out_dir runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h/sae_neuron_pipeline/split_0 \
  --batch_size 4096 \
  --top_k_neurons 20 \
  --top_tiles_per_neuron 50 \
  --local_top_tiles_per_slide 128 \
  --device auto
