#!/usr/bin/env bash
set -euo pipefail

# Upload one project to Hugging Face: H5 features + vis only.
# No coord CSV and no summary upload.

DATASET_ID="${DATASET_ID:-w8yi/tcga-wsi-uni2h-features}"
PROJECT="${PROJECT:-TCGA-HNSC}" # e.g. TCGA-HNSC, TCGA-CESC
BASE_DATA_DIR="${BASE_DATA_DIR:-/research/projects/mllab/WSI/TCGA_features}"
UPLOAD_MODE="${UPLOAD_MODE:-standard}" # standard | large
NUM_WORKERS="${NUM_WORKERS:-8}"
HF_BIN="${HF_BIN:-/common/users/wq50/envs/pace/bin/hf}"
TMP_WORK_ROOT="${TMP_WORK_ROOT:-$BASE_DATA_DIR/.tmp_hf_upload}"

if [[ ! -x "$HF_BIN" ]]; then
  HF_BIN="$(command -v hf || true)"
fi
if [[ -z "$HF_BIN" ]]; then
  echo "hf CLI not found. Set HF_BIN=/path/to/hf" >&2
  exit 1
fi

FEATURE_DIR="$BASE_DATA_DIR/$PROJECT/features"
VIS_DIR="$BASE_DATA_DIR/$PROJECT/vis"

if [[ ! -d "$FEATURE_DIR" || ! -d "$VIS_DIR" ]]; then
  echo "Missing project folders under $BASE_DATA_DIR/$PROJECT" >&2
  echo "Expected: features/ and vis/" >&2
  exit 1
fi

mkdir -p "$TMP_WORK_ROOT"
H5_STAGE=""
STAGING_DIR=""
cleanup_tmp() {
  [[ -n "$STAGING_DIR" && -d "$STAGING_DIR" ]] && rm -rf "$STAGING_DIR"
  [[ -n "$H5_STAGE" && -d "$H5_STAGE" ]] && rm -rf "$H5_STAGE"
}
trap cleanup_tmp EXIT

H5_STAGE="$(mktemp -d "$TMP_WORK_ROOT/hf_h5_only_${PROJECT}.XXXXXX")"
rsync -a --prune-empty-dirs --include='*/' --include='*.h5' --exclude='*' "$FEATURE_DIR"/ "$H5_STAGE"/

if [[ "$UPLOAD_MODE" == "large" ]]; then
  STAGING_DIR="$(mktemp -d "$TMP_WORK_ROOT/hf_staging_${PROJECT}.XXXXXX")"
  mkdir -p "$STAGING_DIR/$PROJECT/features" "$STAGING_DIR/$PROJECT/vis"
  rsync -a "$H5_STAGE"/ "$STAGING_DIR/$PROJECT/features"/
  rsync -a "$VIS_DIR"/ "$STAGING_DIR/$PROJECT/vis"/
  "$HF_BIN" upload-large-folder "$DATASET_ID" "$STAGING_DIR" --repo-type dataset --num-workers "$NUM_WORKERS"
else
  "$HF_BIN" upload "$DATASET_ID" "$H5_STAGE" "$PROJECT/features" --repo-type dataset
  "$HF_BIN" upload "$DATASET_ID" "$VIS_DIR" "$PROJECT/vis" --repo-type dataset
fi

echo "[done] uploaded $PROJECT (features .h5 + vis)"
