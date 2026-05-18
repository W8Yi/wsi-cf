#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

# bash examples/classifier_training/01_train_kirc_grade.sh
# bash examples/classifier_training/02_train_msi_coad_stad.sh
# bash examples/classifier_training/03_train_luad_lusc.sh
bash examples/classifier_training/04_train_cancer_type_all_tcga.sh
bash examples/classifier_training/05_train_kirc_low_vs_high_grade.sh
