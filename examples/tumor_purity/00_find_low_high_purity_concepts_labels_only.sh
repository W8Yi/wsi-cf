#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-tumor_purity_low_high}"

TARGET_CASE_TSV="${TARGET_CASE_TSV:-resources/labels/targets/tumor_purity_case.tsv}"
MASTER_SLIDE_TSV="${MASTER_SLIDE_TSV:-resources/labels/master/slide_labels_master.tsv}"
FEATURES_ROOT="${FEATURES_ROOT:-/research/projects/mllab/WSI/TCGA_features}"
LABEL_SOURCE="${LABEL_SOURCE:-artifacts/label_sources/tumor_purity_low_high_slide_labels.tsv}"
MANIFEST_CSV="${MANIFEST_CSV:-artifacts/label_sources/tumor_purity_low_high_manifest.csv}"
LABEL_SUMMARY="${LABEL_SUMMARY:-artifacts/label_sources/tumor_purity_low_high_summary.json}"

# By default use the bottom and top quartiles of case-level tumor_purity.
# Override with explicit thresholds, for example:
#   LOW_THRESHOLD=0.40 HIGH_THRESHOLD=0.80 bash examples/tumor_purity/00_find_low_high_purity_concepts_labels_only.sh
LOW_QUANTILE="${LOW_QUANTILE:-0.25}"
HIGH_QUANTILE="${HIGH_QUANTILE:-0.75}"
LOW_THRESHOLD="${LOW_THRESHOLD:-}"
HIGH_THRESHOLD="${HIGH_THRESHOLD:-}"

ASSOC_ROOT="${ASSOC_ROOT:-artifacts/concept_label_associations_labels_only}"
CONCEPT_OUT="${CONCEPT_OUT:-artifacts/label_only_concepts_top10_top50}"

PROJECTS="${PROJECTS:-all}"
TOP_CONCEPTS="${TOP_CONCEPTS:-10}"
TOP_TILES="${TOP_TILES:-50}"
CANDIDATE_LATENTS="${CANDIDATE_LATENTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_SLIDES="${MAX_SLIDES:-0}"
MAX_SLIDES_PER_CLASS="${MAX_SLIDES_PER_CLASS:-0}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-0}"
CONCEPT_MAX_SLIDES="${CONCEPT_MAX_SLIDES:-0}"
BUILD_MANIFEST_ONLY="${BUILD_MANIFEST_ONLY:-0}"
SKIP_ASSOC="${SKIP_ASSOC:-0}"
SKIP_CONCEPTS="${SKIP_CONCEPTS:-0}"

echo "[manifest] building ${MANIFEST_CSV}" >&2
LABEL_ARGS=(
  --target-case-tsv "${TARGET_CASE_TSV}"
  --master-slide-tsv "${MASTER_SLIDE_TSV}"
  --features-root "${FEATURES_ROOT}"
  --label-source "${LABEL_SOURCE}"
  --manifest-csv "${MANIFEST_CSV}"
  --summary-json "${LABEL_SUMMARY}"
  --low-quantile "${LOW_QUANTILE}"
  --high-quantile "${HIGH_QUANTILE}"
  --projects "${PROJECTS}"
  --max-slides "${MAX_SLIDES}"
  --max-slides-per-class "${MAX_SLIDES_PER_CLASS}"
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
from collections import Counter, defaultdict
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


def parse_projects(value):
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return set() if len(items) == 1 and items[0].lower() == "all" else set(items)


parser = argparse.ArgumentParser()
parser.add_argument("--target-case-tsv", type=Path, required=True)
parser.add_argument("--master-slide-tsv", type=Path, required=True)
parser.add_argument("--features-root", type=Path, required=True)
parser.add_argument("--label-source", type=Path, required=True)
parser.add_argument("--manifest-csv", type=Path, required=True)
parser.add_argument("--summary-json", type=Path, required=True)
parser.add_argument("--low-quantile", type=float, required=True)
parser.add_argument("--high-quantile", type=float, required=True)
parser.add_argument("--low-threshold", type=float, default=None)
parser.add_argument("--high-threshold", type=float, default=None)
parser.add_argument("--projects", type=str, default="all")
parser.add_argument("--max-slides", type=int, default=0)
parser.add_argument("--max-slides-per-class", type=int, default=0)
args = parser.parse_args()

project_filter = parse_projects(args.projects)

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

label_rows = []
manifest_rows = []
seen_slides = set()
by_label_seen = defaultdict(int)
with args.master_slide_tsv.open("r", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    label_fieldnames = list(reader.fieldnames or [])
    extra_fields = ["purity_group", "purity_case_project_dir", "purity_low_threshold", "purity_high_threshold"]
    for row in reader:
        case_id = str(row.get("case_id", ""))
        slide_key = str(row.get("slide_key", ""))
        project = str(row.get("project_dir", ""))
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
        label_rows.append(row)

        if group not in {"low", "high"}:
            continue
        if project_filter and project not in project_filter:
            continue
        if slide_key in seen_slides:
            continue
        if int(args.max_slides_per_class) > 0 and by_label_seen[group] >= int(args.max_slides_per_class):
            continue
        h5_path = args.features_root / project / "features_uni2" / f"{slide_key}.h5"
        manifest_rows.append(
            {
                "case_id": case_id,
                "slide_key": slide_key,
                "sample_id": str(row.get("sample_id", "")),
                "sample_code": slide_key.split("-")[3][:2] if len(slide_key.split("-")) >= 4 else "",
                "project_dir": project,
                "label_name": group,
                "label": group,
                "purity_group": group,
                "tumor_purity": purity_text,
                "split": "all",
                "h5_path": str(h5_path),
            }
        )
        seen_slides.add(slide_key)
        by_label_seen[group] += 1
        if int(args.max_slides) > 0 and len(manifest_rows) >= int(args.max_slides):
            break

label_rows.sort(key=lambda r: (str(r.get("case_id", "")), str(r.get("slide_key", ""))))
manifest_rows.sort(key=lambda r: (str(r["label"]), str(r["project_dir"]), str(r["case_id"]), str(r["slide_key"])))

args.label_source.parent.mkdir(parents=True, exist_ok=True)
with args.label_source.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=label_fieldnames + extra_fields)
    writer.writeheader()
    writer.writerows(label_rows)

args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
manifest_fields = [
    "case_id",
    "slide_key",
    "sample_id",
    "sample_code",
    "project_dir",
    "label_name",
    "label",
    "purity_group",
    "tumor_purity",
    "split",
    "h5_path",
]
with args.manifest_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=manifest_fields)
    writer.writeheader()
    writer.writerows(manifest_rows)

case_counts = Counter(
    "low" if v <= low_threshold else "high" if v >= high_threshold else "mid"
    for v in case_purity.values()
)
summary = {
    "target_case_tsv": str(args.target_case_tsv),
    "master_slide_tsv": str(args.master_slide_tsv),
    "features_root": str(args.features_root),
    "label_source": str(args.label_source),
    "manifest_csv": str(args.manifest_csv),
    "low_threshold": low_threshold,
    "high_threshold": high_threshold,
    "low_quantile": args.low_quantile,
    "high_quantile": args.high_quantile,
    "case_counts": dict(case_counts),
    "slide_counts_all": dict(Counter(row["purity_group"] for row in label_rows)),
    "manifest_counts": dict(Counter(row["label"] for row in manifest_rows)),
    "projects": "all" if not project_filter else sorted(project_filter),
}
args.summary_json.parent.mkdir(parents=True, exist_ok=True)
args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

if [[ "${BUILD_MANIFEST_ONLY}" == "1" ]]; then
  echo "[ok] built labels and manifest only" >&2
  exit 0
fi

if [[ "${SKIP_ASSOC}" != "1" ]]; then
  echo "[assoc labels-only] ${TASK}" >&2
  "$PY" scripts/prepare_classifier_concept_associations.py \
    --task-name "${TASK}" \
    --classifier-run-dir . \
    --manifest-csv "${MANIFEST_CSV}" \
    --label-column label \
    --out-root "${ASSOC_ROOT}" \
    --split all \
    --max-slides "${MAX_SLIDES}" \
    --max-slides-per-class "${MAX_SLIDES_PER_CLASS}" \
    --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --skip-existing
fi

if [[ "${SKIP_CONCEPTS}" != "1" ]]; then
  for label in low high; do
    echo "[concepts labels-only] ${TASK} / ${label}" >&2
    "$PY" scripts/find_label_concepts.py \
      --task "${TASK}" \
      --association-root "${ASSOC_ROOT}" \
      --class-label "${label}" \
      --mode labels_only \
      --slides-csv "${ASSOC_ROOT}/${TASK}/cohort_slides.csv" \
      --slide-label-column label \
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

echo "[ok] manifest: ${MANIFEST_CSV}" >&2
echo "[ok] associations: ${ASSOC_ROOT}/${TASK}" >&2
echo "[ok] concepts: ${CONCEPT_OUT}/${TASK}/{low,high}" >&2
