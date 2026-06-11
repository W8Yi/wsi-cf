#!/usr/bin/env bash
set -euo pipefail

# Bidirectional HNSCC HPV paper benchmark:
#   5 HPV+ slides x 5 regions, edited toward HPV-
#   5 HPV- slides x 5 regions, edited toward HPV+

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
GENERATION_DEVICE="${GENERATION_DEVICE:-${DEVICE}}"
SEED="${SEED:-7}"
SLIDES_PER_LABEL="${SLIDES_PER_LABEL:-5}"
REGIONS_PER_SLIDE="${REGIONS_PER_SLIDE:-5}"
IMAGE_QC_TOPK_PER_SLIDE="${IMAGE_QC_TOPK_PER_SLIDE:-96}"
OUT_ROOT="${OUT_ROOT:-paper_example/hnscc_hpv_paper_benchmark}"

RUN_REGION_DISCOVERY="${RUN_REGION_DISCOVERY:-1}"
RUN_EDITS="${RUN_EDITS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_REENCODE="${FORCE_REENCODE:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"

MIL_CKPT="${MIL_CKPT:-resources/models/classifiers/hnscc_hpv/mil_split0.pt}"
SPLIT_TSV="${SPLIT_TSV:-resources/manifests/hnsc_hpv_5fold/split_0.tsv}"
SLIDES_DIR="${SLIDES_DIR:-/common/users/wq50/CLAM/HNSCC_slides}"

# The current HNSCC HPV prototype bundle is the legacy ReLU-SAE provenance.
SAE_CKPT="${SAE_CKPT:-/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt}"
SAE_CFG="${SAE_CFG:-/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json}"

REGIONS_DIR="${OUT_ROOT}/regions"
REGION_BANK_CSV="${REGION_BANK_CSV:-${REGIONS_DIR}/region_bank.csv}"
EDIT_MANIFEST="${EDIT_MANIFEST:-${REGIONS_DIR}/progressive_edit_manifest.json}"

region_cmd=(
  "${PYTHON_BIN}" scripts/find_regions.py
  --mode attention
  --backend clam
  --split-tsv "${SPLIT_TSV}"
  --slides-dir "${SLIDES_DIR}"
  --clam-source-from-task-split
  --split test
  --target-magnification 20
  --region-size 2048
  --final-slides-per-label "${SLIDES_PER_LABEL}"
  --final-regions-per-slide "${REGIONS_PER_SLIDE}"
  --max-candidates-per-slide "${REGIONS_PER_SLIDE}"
  --image-qc-topk-per-slide "${IMAGE_QC_TOPK_PER_SLIDE}"
  --region-selection-mode borderline
  --edit-cell-selection-mode attention_percentile_smooth
  --edit-cell-attention-percentile 64.0625
  --out-dir "${REGIONS_DIR}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --sae-ckpt "${SAE_CKPT}"
  --sae-cfg "${SAE_CFG}"
)

benchmark_cmd=(
  "${PYTHON_BIN}" scripts/run_hnscc_hpv_policy_benchmark.py
  --out-dir "${OUT_ROOT}"
  --region-bank-csv "${REGION_BANK_CSV}"
  --edit-manifest "${EDIT_MANIFEST}"
  --split-tsv "${SPLIT_TSV}"
  --mil-ckpt "${MIL_CKPT}"
  --bidirectional
  --device "${DEVICE}"
  --generation-device "${GENERATION_DEVICE}"
  --sae-ckpt "${SAE_CKPT}"
  --sae-cfg "${SAE_CFG}"
  --seed "${SEED}"
  --output-mode "${OUTPUT_MODE}"
  --prepare-associations
  --require-paper-deps
  --policy "ours=configs/edit_policies/showcase_best.json"
  --policy "naive_no_preserve=configs/edit_policies/naive_no_preserve.json"
  --policy "naive_full_duration=configs/edit_policies/baseline_no_preserve_full_duration.json"
)

if [[ "${MAX_RUNS}" != "0" ]]; then
  benchmark_cmd+=(--max-runs "${MAX_RUNS}")
fi

if [[ "${RUN_EDITS}" == "1" ]]; then
  benchmark_cmd+=(--run-edits)
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  benchmark_cmd+=(--skip-existing)
fi

if [[ "${FORCE_REENCODE}" == "1" ]]; then
  benchmark_cmd+=(--force-reencode)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  mkdir -p "${OUT_ROOT}"
  "${PYTHON_BIN}" - "${OUT_ROOT}" "${region_cmd[@]}" --MARK-- "${benchmark_cmd[@]}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
argv = sys.argv[2:]
mark = argv.index("--MARK--")
region_cmd = argv[:mark]
benchmark_cmd = argv[mark + 1 :]
payload = {
    "dry_run": True,
    "commands": [
        {"stage": "region_discovery", "command": " ".join(shlex.quote(x) for x in region_cmd)},
        {"stage": "bidirectional_policy_benchmark", "command": " ".join(shlex.quote(x) for x in benchmark_cmd + ["--dry-run"])},
    ],
}
(out_dir / "run_commands.json").write_text(json.dumps(payload, indent=2) + "\n")
(out_dir / "run_config.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
  exit 0
fi

"${PYTHON_BIN}" - <<'PY'
import importlib

missing = []
for name in ("sklearn", "skimage", "lpips"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing required paper benchmark dependencies:\n" + "\n".join(missing))
PY

printf '[hnscc-hpv paper benchmark] output root: %s\n' "${OUT_ROOT}"

if [[ "${RUN_REGION_DISCOVERY}" == "1" ]]; then
  printf '[hnscc-hpv paper benchmark] region command:'
  printf ' %q' "${region_cmd[@]}"
  printf '\n'
  "${region_cmd[@]}"
fi

printf '[hnscc-hpv paper benchmark] benchmark command:'
printf ' %q' "${benchmark_cmd[@]}"
printf '\n'
"${benchmark_cmd[@]}"

printf '\n[hnscc-hpv paper benchmark] wrote metrics:\n'
printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_summary_by_method.csv"
printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_metrics_by_run.csv"
printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_predictions.csv"
printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_concept_fidelity.csv"
printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_window_consistency.csv"
printf '  %s\n' "${OUT_ROOT}/metrics/benchmark_summary.json"
