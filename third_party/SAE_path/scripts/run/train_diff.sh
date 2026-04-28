#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m scripts.train_diffusion \
  --train_mode traj \
  --out_dir runs/pixcell_student_traj \
  --manifest /common/users/wq50/UNI2_features/extracted_features/sae_manifests_tcga_patient_train_test_90_10.json \
  --manifest_key train \
  --batch_size 8 \
  --max_steps 30000 \
  --teacher_steps 28 \
  --traj_min_steps 4 \
  --traj_max_steps 8 \
  --guidance_teacher 3.0 \
  --guidance_student 1.0 \
  --preview_every 500 --preview_steps 6 --log_every 10 \
  --device cuda:5 \
  --init_student_from_teacher \
  --ema
