/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/run_region_attention_classifier_eval.py \
  --region-bank-csv /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_20x_2048_sample20/region_bank.csv \
  --out-dir /common/users/wq50/wsi_cf/artifacts/region_attention_classifier_eval_20x_2048_comprehensive_pilot10 \
  --final-regions-total 10 \
  --final-regions-per-label 5 \
  --skip-existing \
  --device cuda:0

