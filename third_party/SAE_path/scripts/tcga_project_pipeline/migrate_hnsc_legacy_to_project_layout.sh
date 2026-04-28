#!/usr/bin/env bash
set -euo pipefail

# Simple migration for existing HNSC outputs into the new layout.
#
# Source (legacy):
#   /common/users/wq50/SAE_path/extracted_features/hnsc_hpv_filtered_uni2h
#   /common/users/wq50/SAE_path/extracted_features/hnsc_hpv_filtered_uni2h_coords
#   /common/users/wq50/SAE_path/extracted_features/hnsc_hpv_filtered_uni2h_viz
#   /common/users/wq50/SAE_path/extracted_features/hnsc_hpv_filtered_uni2h_summary
#
# Target (new):
#   /research/projects/mllab/WSI/TCGA-HNSC/features   (.h5 only)
#   /research/projects/mllab/WSI/TCGA-HNSC/coords     (.coords.csv, local only)
#   /research/projects/mllab/WSI/TCGA-HNSC/vis
#   /research/projects/mllab/WSI/TCGA-HNSC/shard_*.log

SRC_ROOT="${SRC_ROOT:-/common/users/wq50/SAE_path/extracted_features}"
DST_PROJECT_DIR="${DST_PROJECT_DIR:-/research/projects/mllab/WSI/TCGA-HNSC}"
MODE="${MODE:-copy}" # copy | move

SRC_FEATURES="$SRC_ROOT/hnsc_hpv_filtered_uni2h"
SRC_COORDS="$SRC_ROOT/hnsc_hpv_filtered_uni2h_coords"
SRC_VIZ="$SRC_ROOT/hnsc_hpv_filtered_uni2h_viz"
SRC_SUMMARY="$SRC_ROOT/hnsc_hpv_filtered_uni2h_summary"

DST_FEATURES="$DST_PROJECT_DIR/features"
DST_COORDS="$DST_PROJECT_DIR/coords"
DST_VIS="$DST_PROJECT_DIR/vis"

if [[ ! -d "$SRC_FEATURES" || ! -d "$SRC_VIZ" ]]; then
  echo "Missing one or more legacy source folders under: $SRC_ROOT" >&2
  exit 1
fi

mkdir -p "$DST_FEATURES" "$DST_COORDS" "$DST_VIS" "$DST_PROJECT_DIR"

echo "[1/3] Sync features (.h5 only)"
rsync -a --prune-empty-dirs --include='*/' --include='*.h5' --exclude='*' "$SRC_FEATURES"/ "$DST_FEATURES"/

echo "[2/4] Sync coordinate CSVs (if available)"
if [[ -d "$SRC_COORDS" ]]; then
  rsync -a --prune-empty-dirs --include='*/' --include='*.coords.csv' --exclude='*' "$SRC_COORDS"/ "$DST_COORDS"/
fi

echo "[3/4] Sync visualization overlays"
rsync -a "$SRC_VIZ"/ "$DST_VIS"/

echo "[4/4] Sync shard logs to project root (if available)"
if [[ -d "$SRC_SUMMARY" ]]; then
  for f in "$SRC_SUMMARY"/shard_*.log; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "$DST_PROJECT_DIR/$(basename "$f")"
  done
fi
rm -f "$DST_PROJECT_DIR/run_summary.json"
find "$DST_PROJECT_DIR" -maxdepth 1 -type f -name '*.summary.json' -delete

if [[ "$MODE" == "move" ]]; then
  echo "[cleanup] MODE=move -> removing legacy source folders"
  rm -rf "$SRC_FEATURES" "$SRC_COORDS" "$SRC_VIZ" "$SRC_SUMMARY"
fi

echo
echo "[done] HNSC migration complete."
echo "Target: $DST_PROJECT_DIR"
echo "  features count: $(find "$DST_FEATURES" -maxdepth 1 -type f | wc -l)"
echo "  coords count:   $(find "$DST_COORDS" -maxdepth 1 -type f -name '*.coords.csv' | wc -l)"
echo "  vis count:      $(find "$DST_VIS" -maxdepth 1 -type f | wc -l)"
