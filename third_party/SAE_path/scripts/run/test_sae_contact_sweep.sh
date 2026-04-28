#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

TILE_DIR="${TILE_DIR:-/common/users/wq50/SAE_path/cf_imgs/roi3_neg1_tiles}"
CONTACT_DIR="${CONTACT_DIR:-/common/users/wq50/SAE_path/sae_mining/relu_sae_base/latent_tiles_pass2_top_tiles_top_activation_n100_top50_20260221-052909/contact_sheets}"
SAE_CKPT="${SAE_CKPT:-/common/users/wq50/SAE_path/runs/relu_sae_base/relu_ckpt_best.pt}"
SAE_CFG="${SAE_CFG:-/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json}"

OUT_ROOT="${OUT_ROOT:-/common/users/wq50/SAE_path/outputs/sae_contact_sweep_relu_sae_base}"
SUMMARY_OUT="${SUMMARY_OUT:-/common/users/wq50/SAE_path/outputs/sae_contact_sweep_relu_sae_base_summary}"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
NAIVE="${NAIVE:-0}"

# Symmetric delta sweep around zero for quick semantic checks.
DELTAS="${DELTAS:--4,-2,-1,-0.5,0.5,1,2,4}"

# Edit stabilizers. Increase BLEND / MAX_FEATURE_DELTA_NORM if edits are too weak.
BLEND="${BLEND:-0.35}"
LATENT_STRENGTH="${LATENT_STRENGTH:-0.4}"
MAX_FEATURE_DELTA_NORM="${MAX_FEATURE_DELTA_NORM:-6.0}"
REFERENCE_START_RATIO="${REFERENCE_START_RATIO:-0.0}"
REFERENCE_MIX="${REFERENCE_MIX:-0.0}"

# NAIVE=1 disables edit damping so you can verify the sweep has visible effects.
# Notes:
# - BLEND controls how much of the decoded SAE edit is applied to the UNI feature.
# - LATENT_STRENGTH is ignored for latent_delta mode (kept for compatibility).
# - MAX_FEATURE_DELTA_NORM=none disables feature-delta clamping.
if [[ "${NAIVE}" == "1" ]]; then
  BLEND="1.0"
  LATENT_STRENGTH="1.0"
  MAX_FEATURE_DELTA_NORM="none"
  REFERENCE_START_RATIO="0.0"
  REFERENCE_MIX="0.0"
fi

# Optional caps for faster smoke tests. Set to 0 to disable.
LATENT_LIMIT="${LATENT_LIMIT:-0}"
TILE_LIMIT="${TILE_LIMIT:-0}"

cmd=(
  python -m concept_steer.sae_steer_sweep
  --image-dir "${TILE_DIR}"
  --out-dir "${OUT_ROOT}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --latent-contact-dir "${CONTACT_DIR}"
  --deltas="${DELTAS}"
  --latent-limit "${LATENT_LIMIT}"
  --tile-limit "${TILE_LIMIT}"
  --sae-ckpt "${SAE_CKPT}"
  --sae-cfg "${SAE_CFG}"
  --tile-px 256
  --grid-step-px 256
  --reference-start-ratio "${REFERENCE_START_RATIO}"
  --reference-mix "${REFERENCE_MIX}"
  --blend "${BLEND}"
  --latent-strength "${LATENT_STRENGTH}"
  --make-summary
  --summary-out-dir "${SUMMARY_OUT}"
  --summary-tile-limit 24
  --summary-include-diff
)

if [[ "${MAX_FEATURE_DELTA_NORM}" != "none" && "${MAX_FEATURE_DELTA_NORM}" != "NONE" ]]; then
  cmd+=(--max-feature-delta-norm "${MAX_FEATURE_DELTA_NORM}")
fi

"${cmd[@]}"

echo
echo "Sweep outputs:   ${OUT_ROOT}"
echo "Summary outputs: ${SUMMARY_OUT}"
