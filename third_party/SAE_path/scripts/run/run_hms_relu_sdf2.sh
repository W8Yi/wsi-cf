#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
# Simple runner: always compute DATA-DRIVEN HMS (locator + WSI mode).

PYTHON_BIN="${PYTHON_BIN:-/common/users/wq50/envs/pace/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/relu_sae_sdf2}"
CKPT_PATH="${CKPT_PATH:-$RUN_DIR/sdf2_final.pt}"
OUT_JSON="${OUT_JSON:-$RUN_DIR/hms_report_data.json}"
DEVICE="${DEVICE:-auto}"

# Required inputs
LOCATOR_CSV_DIR="${LOCATOR_CSV_DIR:-$ROOT_DIR/outputs/hms_locators/locators}"

# Feature source mode
ENCODE_ON_THE_FLY_UNI="${ENCODE_ON_THE_FLY_UNI:-1}"  # 1 => infer UNI2-h from WSI patches; 0 => use H5_ROOT
H5_ROOT="${H5_ROOT:-/research/projects/mllab/WSI/extracted_features}"

# Optional (recommended) WSI settings
WSI_DIR="${WSI_DIR:-}"
WSI_READER="${WSI_READER:-cucim}"        # auto|cucim|openslide
WSI_PATCH_SIZE_20X="${WSI_PATCH_SIZE_20X:-256}"

# Runtime controls
GIGAPATH_MODEL="${GIGAPATH_MODEL:-hf_hub:prov-gigapath/prov-gigapath}"
HMS_MAX_SLIDES="${HMS_MAX_SLIDES:-0}"    # 0 => all
HMS_MAX_TILES="${HMS_MAX_TILES:-0}"      # 0 => all
HMS_TILE_BATCH="${HMS_TILE_BATCH:-512}"
HMS_GIGAPATH_BATCH="${HMS_GIGAPATH_BATCH:-64}"
HMS_PROGRESS_EVERY="${HMS_PROGRESS_EVERY:-20}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/hms_data_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] $LOG_FILE"

if [[ ! -d "$LOCATOR_CSV_DIR" ]]; then
  echo "[error] LOCATOR_CSV_DIR not found: $LOCATOR_CSV_DIR" >&2
  exit 1
fi
if [[ "$ENCODE_ON_THE_FLY_UNI" != "1" ]]; then
  if [[ ! -d "$H5_ROOT" ]]; then
    echo "[error] H5_ROOT not found: $H5_ROOT" >&2
    exit 1
  fi
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/compute_hms_sdf2.py"
  --run_dir "$RUN_DIR"
  --ckpt_path "$CKPT_PATH"
  --out_json "$OUT_JSON"
  --device "$DEVICE"
  --run_data_hms
  --locator_csv_dir "$LOCATOR_CSV_DIR"
  --wsi_reader "$WSI_READER"
  --wsi_patch_size_20x "$WSI_PATCH_SIZE_20X"
  --gigapath_model "$GIGAPATH_MODEL"
  --max_slides "$HMS_MAX_SLIDES"
  --max_tiles "$HMS_MAX_TILES"
  --tile_batch "$HMS_TILE_BATCH"
  --gigapath_batch "$HMS_GIGAPATH_BATCH"
  --progress_every_batches "$HMS_PROGRESS_EVERY"
)
if [[ "$ENCODE_ON_THE_FLY_UNI" == "1" ]]; then
  CMD+=(--encode_uni_on_the_fly)
else
  CMD+=(--h5_root "$H5_ROOT")
fi

if [[ -n "$WSI_DIR" ]]; then
  CMD+=(--wsi_dir "$WSI_DIR")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"
