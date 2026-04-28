#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m scripts.train_diffusion \
  --train_mode flow \
  --out_dir runs/pixcell_student_flow \
  --manifest /common/users/wq50/UNI2_features/extracted_features/sae_manifests_tcga_patient_train_test_90_10.json \
  --manifest_key train \
  --batch_size 16 \
  --max_steps 30000 \
  --teacher_steps 28 \
  --guidance_teacher 3.0 \
  --guidance_student 1.0 \
  --device cuda:4 \
  --dtype fp16 \
  --init_student_from_teacher \
  --ema \
  --log_every 10 \
  --preview_every 500 \
  --preview_steps 6
