#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/common/users/wq50/SAE_path"
cd "${REPO_ROOT}"

echo "[rebuild_labels] starting"
python metadata/labels/scripts/rebuild_labels_pipeline.py "$@"
echo "[rebuild_labels] done"
