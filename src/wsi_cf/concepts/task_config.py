from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wsi_cf.common.paths import WSI_CF_ROOT, resource_path


DEFAULT_LABEL_SOURCE = WSI_CF_ROOT / "resources/labels/master/slide_labels_master.tsv"
DEFAULT_SPLIT_MANIFEST = WSI_CF_ROOT / "resources/manifests/sae_manifests_tcga_patient_train_test_90_10.json"
DEFAULT_FEATURES_ROOT = Path("/research/projects/mllab/WSI/TCGA_features")
UNKNOWN_VALUES = {"", "UNKNOWN", "UNK", "NA", "N/A", "NONE", "NULL", "GX"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def read_table_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def parse_csv_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip()]
    raise TypeError(f"Expected string/list value, got {type(value).__name__}")


def normalize_label_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("task JSON field 'label_map' must be an object")
    out: dict[str, str] = {}
    for raw, mapped in value.items():
        out[str(raw).strip()] = str(mapped).strip()
    return out


def slide_key_from_manifest_path(path_value: str) -> str:
    return Path(str(path_value)).name.split(".")[0]


def case_id_from_slide_key(slide_key: str) -> str:
    parts = str(slide_key).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(slide_key)


def sample_code_from_slide_key(slide_key: str) -> str:
    parts = str(slide_key).split("-")
    if len(parts) >= 4 and len(parts[3]) >= 2:
        return parts[3][:2]
    return ""


def load_split_cases(path: Path) -> tuple[set[str], set[str]]:
    payload = read_json(path)
    train_cases = {case_id_from_slide_key(slide_key_from_manifest_path(p)) for p in payload.get("train", [])}
    test_cases = {case_id_from_slide_key(slide_key_from_manifest_path(p)) for p in payload.get("test", [])}
    overlap = train_cases & test_cases
    if overlap:
        raise ValueError(f"{path}: split manifest has {len(overlap)} overlapping case IDs; examples={sorted(overlap)[:5]}")
    return train_cases, test_cases


def split_matches(row_split: str, requested: str) -> bool:
    requested = str(requested or "all")
    return requested == "all" or str(row_split) == requested


def label_slug(label: str) -> str:
    return str(label).replace("/", "_").replace(" ", "_")


def resolve_task_config(path: Path) -> dict[str, Any]:
    task_path = resource_path(path)
    cfg = read_json(task_path)
    task_name = str(cfg.get("task_name", "")).strip()
    if not task_name:
        raise ValueError(f"{task_path}: missing required field 'task_name'")

    resolved = dict(cfg)
    resolved["task_json"] = str(task_path)
    resolved["task_name"] = task_name
    resolved["label_source"] = str(resource_path(cfg.get("label_source", DEFAULT_LABEL_SOURCE)))
    resolved["split_manifest"] = str(resource_path(cfg.get("split_manifest", DEFAULT_SPLIT_MANIFEST)))
    resolved["features_root"] = str(resource_path(cfg.get("features_root", DEFAULT_FEATURES_ROOT)))
    resolved["projects"] = parse_csv_list(cfg.get("projects", "all"))
    if not resolved["projects"]:
        resolved["projects"] = ["all"]
    resolved["label_column"] = str(cfg.get("label_column", "")).strip()
    if not resolved["label_column"]:
        raise ValueError(f"{task_path}: missing required field 'label_column'")
    resolved["label_map"] = normalize_label_map(cfg.get("label_map", {}))
    resolved["numeric_bins"] = cfg.get("numeric_bins", {})
    resolved["include_labels"] = parse_csv_list(cfg.get("include_labels", []))
    resolved["concept_labels"] = parse_csv_list(cfg.get("concept_labels", resolved["include_labels"]))
    if not resolved["concept_labels"]:
        raise ValueError(f"{task_path}: concept_labels is empty")
    resolved["exclude_raw_labels"] = parse_csv_list(cfg.get("exclude_raw_labels", []))
    resolved["association_split"] = str(cfg.get("association_split", "train"))
    resolved["representative_split"] = str(cfg.get("representative_split", "all"))
    resolved["sae_variant"] = str(cfg.get("sae_variant", "tcga_sae_batch_topk_20x_interp"))
    resolved["ranking_mode"] = str(cfg.get("ranking_mode", "attention_aware_optional"))
    resolved["classifier_run_dir"] = str(resource_path(cfg["classifier_run_dir"])) if cfg.get("classifier_run_dir") else ""
    resolved["classifier_ckpt"] = str(resource_path(cfg["classifier_ckpt"])) if cfg.get("classifier_ckpt") else ""
    resolved["top_concepts"] = int(cfg.get("top_concepts", 20))
    resolved["candidate_latents"] = int(cfg.get("candidate_latents", 150))
    resolved["top_tiles_per_concept"] = int(cfg.get("top_tiles_per_concept", 50))
    resolved["batch_size"] = int(cfg.get("batch_size", 4096))
    resolved["max_slides"] = int(cfg.get("max_slides", 0))
    resolved["max_slides_per_class"] = int(cfg.get("max_slides_per_class", 0))
    resolved["max_tiles_per_slide"] = int(cfg.get("max_tiles_per_slide", 0))
    resolved["min_slides_per_class"] = int(cfg.get("min_slides_per_class", 1))
    resolved["metric"] = str(cfg.get("metric", "fraction"))
    resolved["min_cohen_d"] = float(cfg.get("min_cohen_d", 0.0))
    resolved["concept_quality_mode"] = str(cfg.get("concept_quality_mode", "morphology"))
    resolved["association_weight"] = float(cfg.get("association_weight", 0.7))
    resolved["attention_weight"] = float(cfg.get("attention_weight", 0.3))
    resolved["morphology_target_prevalence"] = float(cfg.get("morphology_target_prevalence", 0.03))
    resolved["morphology_prevalence_sigma"] = float(cfg.get("morphology_prevalence_sigma", 0.75))
    resolved["morphology_coherence_top_k"] = int(cfg.get("morphology_coherence_top_k", 10))
    resolved["attn_class"] = str(cfg.get("attn_class", "pred"))
    resolved["concept_export_target_magnification"] = float(cfg.get("concept_export_target_magnification", 20.0))
    resolved["concept_export_tile_size_px"] = int(cfg.get("concept_export_tile_size_px", 256))
    return resolved


def build_task_cohort(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    label_source = Path(str(cfg["label_source"]))
    split_manifest = Path(str(cfg["split_manifest"]))
    features_root = Path(str(cfg.get("features_root", DEFAULT_FEATURES_ROOT)))
    rows, fieldnames = read_table_rows(label_source)
    label_column = str(cfg["label_column"])
    if label_column not in fieldnames:
        raise ValueError(f"{label_source}: label column {label_column!r} not found")
    if "project_dir" not in fieldnames or "slide_key" not in fieldnames:
        raise ValueError(f"{label_source}: expected project_dir and slide_key columns")

    train_cases, test_cases = load_split_cases(split_manifest)
    projects = set(parse_csv_list(cfg.get("projects", ["all"])))
    all_projects = projects == {"all"}
    label_map = normalize_label_map(cfg.get("label_map", {}))
    include_labels = set(parse_csv_list(cfg.get("include_labels", [])))
    exclude_raw = set(parse_csv_list(cfg.get("exclude_raw_labels", [])))
    numeric_bins = cfg.get("numeric_bins", {})
    if numeric_bins is None:
        numeric_bins = {}
    if not isinstance(numeric_bins, dict):
        raise TypeError("task JSON field 'numeric_bins' must be an object")
    unknown_values = {v.upper() for v in UNKNOWN_VALUES}

    cohort: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source in rows:
        project = str(source.get("project_dir", ""))
        if not all_projects and project not in projects:
            continue
        slide_key = str(source.get("slide_key", ""))
        case_id = str(source.get("case_id", "")) or case_id_from_slide_key(slide_key)
        split = "train" if case_id in train_cases else "test" if case_id in test_cases else ""
        if not split:
            skipped.append({"case_id": case_id, "slide_key": slide_key, "project_dir": project, "reason": "missing_split"})
            continue
        raw_label = project if label_column == "project_dir" else str(source.get(label_column, ""))
        raw_label = raw_label.strip()
        if raw_label in exclude_raw or raw_label.upper() in unknown_values:
            skipped.append({"case_id": case_id, "slide_key": slide_key, "project_dir": project, "raw_label": raw_label, "reason": "excluded_or_unknown_label"})
            continue
        if raw_label in label_map:
            label = label_map[raw_label].strip()
        elif numeric_bins:
            try:
                raw_number = float(raw_label)
            except ValueError:
                skipped.append({"case_id": case_id, "slide_key": slide_key, "project_dir": project, "raw_label": raw_label, "reason": "non_numeric_label"})
                continue
            label = ""
            for bin_label, spec in numeric_bins.items():
                if not isinstance(spec, dict):
                    raise TypeError(f"numeric_bins[{bin_label!r}] must be an object")
                lo = spec.get("min", None)
                hi = spec.get("max", None)
                if lo is not None and raw_number < float(lo):
                    continue
                if hi is not None and raw_number > float(hi):
                    continue
                label = str(bin_label)
                break
            if not label:
                skipped.append({"case_id": case_id, "slide_key": slide_key, "project_dir": project, "raw_label": raw_label, "reason": "numeric_bin_unassigned"})
                continue
        else:
            label = raw_label
        if label.upper() in unknown_values:
            skipped.append({"case_id": case_id, "slide_key": slide_key, "project_dir": project, "raw_label": raw_label, "label": label, "reason": "mapped_to_unknown_label"})
            continue
        if include_labels and label not in include_labels:
            continue
        h5_path = str(source.get("h5_path", "")).strip()
        fallback_h5 = features_root / project / "features_uni2" / f"{slide_key}.h5"
        if not h5_path or (not Path(h5_path).exists() and fallback_h5.exists()):
            h5_path = str(fallback_h5)
        if not Path(h5_path).exists():
            skipped.append(
                {
                    "case_id": case_id,
                    "slide_key": slide_key,
                    "project_dir": project,
                    "raw_label": raw_label,
                    "label": label,
                    "split": split,
                    "h5_path": h5_path,
                    "reason": "missing_h5",
                }
            )
            continue
        cohort.append(
            {
                "task": str(cfg["task_name"]),
                "case_id": case_id,
                "slide_key": slide_key,
                "sample_id": str(source.get("sample_id", "")),
                "sample_code": sample_code_from_slide_key(slide_key),
                "project_dir": project,
                "label": label,
                "label_name": label,
                "raw_label": raw_label,
                "split": split,
                "h5_path": h5_path,
            }
        )

    cohort.sort(key=lambda r: (str(r["label"]), str(r["split"]), str(r["project_dir"]), str(r["case_id"]), str(r["slide_key"])))
    cohort = enforce_class_limits(
        cohort,
        min_slides_per_class=int(cfg.get("min_slides_per_class", 1)),
        max_slides_per_class=int(cfg.get("max_slides_per_class", 0)),
        max_slides=int(cfg.get("max_slides", 0)),
    )
    label_counts = Counter(str(row["label"]) for row in cohort)
    missing_labels = [label for label in parse_csv_list(cfg.get("concept_labels", [])) if label_counts.get(label, 0) == 0]
    if missing_labels:
        raise RuntimeError(f"Concept labels have no cohort rows after filtering: {missing_labels}; counts={dict(label_counts)}")
    summary = {
        "label_counts": dict(label_counts),
        "split_counts": {label: dict(Counter(str(row["split"]) for row in rows_for_label)) for label, rows_for_label in group_by_label(cohort).items()},
        "n_cohort_rows": int(len(cohort)),
        "n_skipped_rows": int(len(skipped)),
    }
    return cohort, skipped, summary


def group_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)
    return dict(grouped)


def enforce_class_limits(
    rows: list[dict[str, Any]],
    *,
    min_slides_per_class: int,
    max_slides_per_class: int,
    max_slides: int,
) -> list[dict[str, Any]]:
    counts = Counter(str(row["label"]) for row in rows)
    keep_labels = {label for label, count in counts.items() if int(count) >= int(min_slides_per_class)}
    rows = [row for row in rows if str(row["label"]) in keep_labels]
    if int(max_slides_per_class) > 0:
        kept: list[dict[str, Any]] = []
        seen: Counter[str] = Counter()
        for row in rows:
            label = str(row["label"])
            if seen[label] >= int(max_slides_per_class):
                continue
            kept.append(row)
            seen[label] += 1
        rows = kept
    if int(max_slides) > 0:
        rows = rows[: int(max_slides)]
    return rows


def select_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if split_matches(str(row.get("split", "")), split)]
