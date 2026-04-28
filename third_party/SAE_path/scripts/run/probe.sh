#!/usr/bin/env bash
# run_linear_probe_hpv.sh
#
# Run linear probing on SAE latents to predict HPV status
# using TRAIN slides only.
#
# Assumes:
# - linear_probe_hpv_from_sae_latents.py is in the current directory
# - sae_models.py is importable (same directory or PYTHONPATH set)
#
# Edit paths if needed.

set -euo pipefail
cd "$(dirname "$0")/../.."

MANIFEST="/common/users/wq50/SAE_path/metadata/manifests/sae_manifest_hpv100_split0_h5.json"
HNSCC_CSV="/common/users/wq50/CLAM/dataset_csv/HNSCC.csv"

SAE_CKPT="/common/users/wq50/SAE_path/runs/relu_sae_tcga_hnscc/relu_ckpt_best.pt"
SAE_CFG="/common/users/wq50/SAE_path/runs/relu_sae_tcga_hnscc/run_config.json"
STAGE="relu"

OUT_DIR="/common/users/wq50/SAE_path/counterfactual/sae_probe_hpv_s0"

DEVICE="cuda:0"
TILES_PER_SLIDE=4096
BATCH_SIZE=2048
SEED=66

# Logistic regression hyperparameters
C=1.0
PENALTY="l2"
SOLVER="lbfgs"

echo "Running SAE linear probe for HPV..."
echo "Output dir: ${OUT_DIR}"

python -m scripts.linear_probe \
  --manifest "${MANIFEST}" \
  --hnscc_csv "${HNSCC_CSV}" \
  --sae_ckpt "${SAE_CKPT}" \
  --sae_cfg "${SAE_CFG}" \
  --stage "${STAGE}" \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --tiles_per_slide "${TILES_PER_SLIDE}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --C "${C}" \
  --penalty "${PENALTY}" \
  --solver "${SOLVER}"

echo "Done."
