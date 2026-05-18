#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-tumor_purity_low_high}"

TARGET_CASE_TSV="${TARGET_CASE_TSV:-resources/labels/targets/tumor_purity_case.tsv}"
MASTER_SLIDE_TSV="${MASTER_SLIDE_TSV:-resources/labels/master/slide_labels_master.tsv}"
LABEL_SOURCE="${LABEL_SOURCE:-artifacts/label_sources/tumor_purity_low_high_slide_labels.tsv}"
LABEL_SUMMARY="${LABEL_SUMMARY:-artifacts/label_sources/tumor_purity_low_high_summary.json}"

# By default use the bottom and top quartiles of case-level tumor_purity.
# Override with explicit thresholds, for example:
#   LOW_THRESHOLD=0.40 HIGH_THRESHOLD=0.80 bash examples/tumor_purity/01_find_low_high_purity_concepts.sh
LOW_QUANTILE="${LOW_QUANTILE:-0.25}"
HIGH_QUANTILE="${HIGH_QUANTILE:-0.75}"
LOW_THRESHOLD="${LOW_THRESHOLD:-}"
HIGH_THRESHOLD="${HIGH_THRESHOLD:-}"

CLASSIFIER_ROOT="${CLASSIFIER_ROOT:-artifacts/classifier_training}"
CLASSIFIER_RUN_DIR="${CLASSIFIER_RUN_DIR:-${CLASSIFIER_ROOT}/${TASK}}"
ASSOC_ROOT="${ASSOC_ROOT:-artifacts/concept_label_associations_classifier}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/classifier_label_concepts_all_top10_top50}"

PROJECTS="${PROJECTS:-all}"
EPOCHS="${EPOCHS:-8}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-2048}"
MAX_SLIDES_PER_CLASS="${MAX_SLIDES_PER_CLASS:-0}"
MAX_TRAIN_SLIDES="${MAX_TRAIN_SLIDES:-0}"
MAX_TEST_SLIDES="${MAX_TEST_SLIDES:-0}"
MIN_SLIDES_PER_CLASS="${MIN_SLIDES_PER_CLASS:-10}"

TOP_CONCEPTS="${TOP_CONCEPTS:-10}"
TOP_TILES="${TOP_TILES:-50}"
CANDIDATE_LATENTS="${CANDIDATE_LATENTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
CONCEPT_MAX_SLIDES="${CONCEPT_MAX_SLIDES:-0}"
CONCEPT_MAX_SLIDES_PER_CLASS="${CONCEPT_MAX_SLIDES_PER_CLASS:-0}"
CONCEPT_MAX_TILES_PER_SLIDE="${CONCEPT_MAX_TILES_PER_SLIDE:-0}"
SPLIT="${SPLIT:-all}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_ASSOC="${SKIP_ASSOC:-0}"
SKIP_CONCEPTS="${SKIP_CONCEPTS:-0}"
BUILD_LABELS_ONLY="${BUILD_LABELS_ONLY:-0}"

echo "[labels] building ${LABEL_SOURCE}" >&2
LABEL_ARGS=(
  --target-case-tsv "${TARGET_CASE_TSV}" \
  --master-slide-tsv "${MASTER_SLIDE_TSV}" \
  --label-source "${LABEL_SOURCE}" \
  --summary-json "${LABEL_SUMMARY}" \
  --low-quantile "${LOW_QUANTILE}" \
  --high-quantile "${HIGH_QUANTILE}"
)
if [[ -n "${LOW_THRESHOLD}" ]]; then
  LABEL_ARGS+=(--low-threshold "${LOW_THRESHOLD}")
fi
if [[ -n "${HIGH_THRESHOLD}" ]]; then
  LABEL_ARGS+=(--high-threshold "${HIGH_THRESHOLD}")
fi
"$PY" - "${LABEL_ARGS[@]}" <<'PY'
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def quantile(values, q):
    vals = sorted(values)
    if not vals:
        raise ValueError("No tumor_purity values available")
    pos = (len(vals) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


parser = argparse.ArgumentParser()
parser.add_argument("--target-case-tsv", type=Path, required=True)
parser.add_argument("--master-slide-tsv", type=Path, required=True)
parser.add_argument("--label-source", type=Path, required=True)
parser.add_argument("--summary-json", type=Path, required=True)
parser.add_argument("--low-quantile", type=float, required=True)
parser.add_argument("--high-quantile", type=float, required=True)
parser.add_argument("--low-threshold", type=float, default=None)
parser.add_argument("--high-threshold", type=float, default=None)
args = parser.parse_args()

case_purity = {}
case_project = {}
with args.target_case_tsv.open("r", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        value = str(row.get("tumor_purity", "")).strip()
        if not value:
            continue
        case_id = str(row["case_id"])
        case_purity[case_id] = float(value)
        case_project[case_id] = str(row.get("project_dir", ""))

values = list(case_purity.values())
low_threshold = float(args.low_threshold) if args.low_threshold is not None else quantile(values, args.low_quantile)
high_threshold = float(args.high_threshold) if args.high_threshold is not None else quantile(values, args.high_quantile)
if low_threshold >= high_threshold:
    raise ValueError(f"low threshold must be < high threshold, got {low_threshold} >= {high_threshold}")

out_rows = []
with args.master_slide_tsv.open("r", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fieldnames = list(reader.fieldnames or [])
    extra_fields = ["purity_group", "purity_case_project_dir", "purity_low_threshold", "purity_high_threshold"]
    for row in reader:
        case_id = str(row.get("case_id", ""))
        purity = case_purity.get(case_id)
        if purity is None:
            group = "missing"
            purity_text = ""
        else:
            group = "low" if purity <= low_threshold else "high" if purity >= high_threshold else "mid"
            purity_text = f"{purity:.6f}"
        row["tumor_purity"] = purity_text
        row["purity_group"] = group
        row["purity_case_project_dir"] = case_project.get(case_id, "")
        row["purity_low_threshold"] = f"{low_threshold:.6f}"
        row["purity_high_threshold"] = f"{high_threshold:.6f}"
        out_rows.append(row)

args.label_source.parent.mkdir(parents=True, exist_ok=True)
with args.label_source.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames + extra_fields)
    writer.writeheader()
    writer.writerows(out_rows)

slide_counts = Counter(row["purity_group"] for row in out_rows)
case_counts = Counter(
    "low" if v <= low_threshold else "high" if v >= high_threshold else "mid"
    for v in case_purity.values()
)
summary = {
    "target_case_tsv": str(args.target_case_tsv),
    "master_slide_tsv": str(args.master_slide_tsv),
    "label_source": str(args.label_source),
    "low_threshold": low_threshold,
    "high_threshold": high_threshold,
    "low_quantile": args.low_quantile,
    "high_quantile": args.high_quantile,
    "case_counts": dict(case_counts),
    "slide_counts": dict(slide_counts),
}
args.summary_json.parent.mkdir(parents=True, exist_ok=True)
args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

if [[ "${BUILD_LABELS_ONLY}" == "1" ]]; then
  echo "[ok] built label source only" >&2
  exit 0
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[train] ${TASK}" >&2
  "$PY" scripts/train_attention_classifier.py \
    --task-name "${TASK}" \
    --label-source "${LABEL_SOURCE}" \
    --projects "${PROJECTS}" \
    --label-column purity_group \
    --include-labels low,high \
    --out-dir "${CLASSIFIER_ROOT}" \
    --epochs "${EPOCHS}" \
    --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
    --max-slides-per-class "${MAX_SLIDES_PER_CLASS}" \
    --max-train-slides "${MAX_TRAIN_SLIDES}" \
    --max-test-slides "${MAX_TEST_SLIDES}" \
    --min-slides-per-class "${MIN_SLIDES_PER_CLASS}" \
    --device "${DEVICE}"
fi

if [[ "${SKIP_ASSOC}" != "1" ]]; then
  echo "[assoc] ${TASK}" >&2
  "$PY" scripts/prepare_classifier_concept_associations.py \
    --task-name "${TASK}" \
    --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
    --out-root "${ASSOC_ROOT}" \
    --split "${SPLIT}" \
    --max-slides "${CONCEPT_MAX_SLIDES}" \
    --max-slides-per-class "${CONCEPT_MAX_SLIDES_PER_CLASS}" \
    --max-tiles-per-slide "${CONCEPT_MAX_TILES_PER_SLIDE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --skip-existing
fi

if [[ "${SKIP_CONCEPTS}" != "1" ]]; then
  for label in low high; do
    echo "[concepts] ${TASK} / ${label}" >&2
    "$PY" scripts/find_label_concepts.py \
      --task "${TASK}" \
      --association-root "${ASSOC_ROOT}" \
      --class-label "${label}" \
      --mode attention_aware \
      --backend mil \
      --classifier-run-dir "${CLASSIFIER_RUN_DIR}" \
      --slides-csv "${ASSOC_ROOT}/${TASK}/cohort_slides.csv" \
      --out-dir "${CONCEPT_OUT}" \
      --top-concepts "${TOP_CONCEPTS}" \
      --candidate-latents "${CANDIDATE_LATENTS}" \
      --top-tiles-per-concept "${TOP_TILES}" \
      --batch-size "${BATCH_SIZE}" \
      --max-slides "${CONCEPT_MAX_SLIDES}" \
      --device "${DEVICE}" \
      --skip-existing
  done
fi

echo "[ok] classifier: ${CLASSIFIER_RUN_DIR}" >&2
echo "[ok] associations: ${ASSOC_ROOT}/${TASK}" >&2
echo "[ok] concepts: ${CONCEPT_OUT}/${TASK}/{low,high}" >&2
