#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import requests


WSI_CF_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABEL_SOURCE = WSI_CF_ROOT / "resources/labels/master/slide_labels_master.tsv"
DEFAULT_FEATURES_ROOT = Path("/research/projects/mllab/WSI/TCGA_features")
DEFAULT_NORMAL_FEATURES_ROOT = WSI_CF_ROOT / "artifacts/normal_tumor_features"
DEFAULT_OUT_ROOT = WSI_CF_ROOT / "artifacts/normal_tumor_inputs"
TASKS = {
    "luad_normal_tumor": {"gdc_project": "TCGA-LUAD", "local_projects": ["TCGA-LUAD"]},
    "coad_normal_tumor": {"gdc_project": "TCGA-COAD", "local_projects": ["TCGA-COAD"]},
    "brca_normal_tumor": {"gdc_project": "TCGA-BRCA", "local_projects": ["TCGA-BRCA_IDC", "TCGA-BRCA_OTHERS"]},
    "kirc_normal_tumor": {"gdc_project": "TCGA-KIRC", "local_projects": ["TCGA-KIRC"]},
    "lusc_normal_tumor": {"gdc_project": "TCGA-LUSC", "local_projects": ["TCGA-LUSC"]},
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare normal-versus-tumor concept tasks from GDC normal SVS inventory and local tumor UNI2 bags."
    )
    parser.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated task names.")
    parser.add_argument("--label-source", type=Path, default=DEFAULT_LABEL_SOURCE)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument(
        "--normal-features-root",
        type=Path,
        default=DEFAULT_NORMAL_FEATURES_ROOT,
        help="Destination root for newly extracted normal-slide UNI2 bags.",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument(
        "--download-subset-size",
        type=int,
        default=20,
        help="Number of normal GDC files to include in the small starter download manifest for each task; 0 disables it.",
    )
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def case_id_from_slide_key(slide_key: str) -> str:
    return "-".join(str(slide_key).split("-")[:3])


def sample_id_from_slide_key(slide_key: str) -> str:
    return "-".join(str(slide_key).split("-")[:4])


def sample_code_from_slide_key(slide_key: str) -> str:
    parts = str(slide_key).split("-")
    return parts[3][:2] if len(parts) >= 4 else ""


def resolve_feature_h5(features_dir: Path, slide_key: str) -> Path:
    exact = features_dir / f"{slide_key}.h5"
    if exact.exists():
        return exact
    matches = sorted(features_dir.glob(f"{slide_key}*.h5"))
    return matches[0] if matches else exact


def query_gdc_normal_slides(gdc_project: str) -> list[dict[str, Any]]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [gdc_project]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Solid Tissue Normal"]}},
        ],
    }
    response = requests.get(
        "https://api.gdc.cancer.gov/files",
        params={
            "filters": json.dumps(filters),
            "fields": (
                "file_id,file_name,file_size,md5sum,state,"
                "cases.submitter_id,cases.samples.sample_type,cases.project.project_id"
            ),
            "format": "JSON",
            "size": "5000",
        },
        timeout=120,
    )
    response.raise_for_status()
    rows: list[dict[str, Any]] = []
    for hit in response.json()["data"]["hits"]:
        file_name = str(hit["file_name"])
        slide_key = file_name.split(".")[0]
        cases = hit.get("cases") or [{}]
        case_id = str(cases[0].get("submitter_id", "")) or case_id_from_slide_key(slide_key)
        rows.append(
            {
                "case_id": case_id,
                "slide_key": slide_key,
                "sample_id": sample_id_from_slide_key(slide_key),
                "sample_code": sample_code_from_slide_key(slide_key),
                "label": "normal",
                "normal_tumor_label": "normal",
                "gdc_sample_type": "Solid Tissue Normal",
                "gdc_file_id": str(hit["file_id"]),
                "gdc_file_name": file_name,
                "gdc_file_size": int(hit.get("file_size", 0)),
                "gdc_md5sum": str(hit.get("md5sum", "")),
                "gdc_state": str(hit.get("state", "")),
                "download_url": f"https://api.gdc.cancer.gov/data/{hit['file_id']}",
                "source": "gdc_normal_inventory",
            }
        )
    return sorted(rows, key=lambda row: (str(row["case_id"]), str(row["slide_key"])))


def load_local_tumor_rows(
    label_source: Path,
    *,
    local_projects: list[str],
    gdc_project: str,
    features_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with label_source.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for source in reader:
            local_project = str(source.get("project_dir", ""))
            if local_project not in local_projects:
                continue
            slide_key = str(source["slide_key"])
            if sample_code_from_slide_key(slide_key) != "01":
                continue
            h5_path = resolve_feature_h5(features_root / local_project / "features_uni2", slide_key)
            if not h5_path.exists():
                continue
            rows.append(
                {
                    "case_id": str(source.get("case_id", "")) or case_id_from_slide_key(slide_key),
                    "slide_key": slide_key,
                    "sample_id": str(source.get("sample_id", "")) or sample_id_from_slide_key(slide_key),
                    "sample_code": sample_code_from_slide_key(slide_key),
                    "label": "tumor",
                    "normal_tumor_label": "tumor",
                    "gdc_sample_type": "Primary Tumor",
                    "gdc_file_id": "",
                    "gdc_file_name": "",
                    "gdc_file_size": "",
                    "gdc_md5sum": "",
                    "gdc_state": "",
                    "download_url": "",
                    "source": "existing_tumor_uni2_bag",
                    "project_dir": gdc_project,
                    "local_feature_project": local_project,
                    "h5_path": str(h5_path),
                    "feature_ready": 1,
                }
            )
    return sorted(rows, key=lambda row: (str(row["case_id"]), str(row["slide_key"])))


def assign_patient_split(rows: list[dict[str, Any]], *, seed: int, test_fraction: float) -> dict[str, str]:
    cases = sorted({str(row["case_id"]) for row in rows})
    random.Random(int(seed)).shuffle(cases)
    n_test = max(1, int(round(len(cases) * float(test_fraction))))
    test_cases = set(cases[:n_test])
    return {case_id: ("test" if case_id in test_cases else "train") for case_id in cases}


def write_gdc_manifest(path: Path, normal_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "filename", "md5", "size", "state"])
        for row in normal_rows:
            writer.writerow(
                [row["gdc_file_id"], row["gdc_file_name"], row["gdc_md5sum"], row["gdc_file_size"], row["gdc_state"]]
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    if not 0.0 < float(args.test_fraction) < 1.0:
        raise ValueError("--test-fraction must be in (0, 1)")
    requested = [token.strip() for token in str(args.tasks).split(",") if token.strip()]
    unknown = sorted(set(requested) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}; expected one of {sorted(TASKS)}")

    all_summary: dict[str, Any] = {}
    for offset, task_name in enumerate(requested):
        spec = TASKS[task_name]
        gdc_project = str(spec["gdc_project"])
        task_dir = args.out_root / task_name
        normal_rows = query_gdc_normal_slides(gdc_project)
        for row in normal_rows:
            row["project_dir"] = gdc_project
            row["local_feature_project"] = gdc_project
            row["h5_path"] = str(resolve_feature_h5(args.normal_features_root / gdc_project / "features_uni2", str(row["slide_key"])))
            row["feature_ready"] = int(Path(str(row["h5_path"])).exists())
        tumor_rows = load_local_tumor_rows(
            args.label_source,
            local_projects=list(spec["local_projects"]),
            gdc_project=gdc_project,
            features_root=args.features_root,
        )
        rows = tumor_rows + normal_rows
        split_by_case = assign_patient_split(rows, seed=int(args.seed) + offset, test_fraction=float(args.test_fraction))
        for row in rows:
            row["split"] = split_by_case[str(row["case_id"])]
        fieldnames = [
            "case_id",
            "slide_key",
            "sample_id",
            "sample_code",
            "project_dir",
            "local_feature_project",
            "normal_tumor_label",
            "label",
            "split",
            "gdc_sample_type",
            "h5_path",
            "feature_ready",
            "gdc_file_id",
            "gdc_file_name",
            "gdc_file_size",
            "gdc_md5sum",
            "gdc_state",
            "download_url",
            "source",
        ]
        write_csv(task_dir / "slide_labels.csv", rows, fieldnames)
        write_csv(task_dir / "normal_gdc_files.csv", normal_rows, fieldnames)
        write_gdc_manifest(task_dir / "normal_all.gdc_manifest.tsv", normal_rows)
        if int(args.download_subset_size) > 0:
            write_gdc_manifest(task_dir / "normal_starter.gdc_manifest.tsv", normal_rows[: int(args.download_subset_size)])
        split_manifest = {
            "train": [row["slide_key"] for row in rows if row["split"] == "train"],
            "test": [row["slide_key"] for row in rows if row["split"] == "test"],
        }
        write_json(task_dir / "patient_train_test_90_10.json", split_manifest)
        write_json(
            task_dir / "concept_task.json",
            {
                "task_name": task_name,
                "label_source": str(task_dir / "slide_labels.csv"),
                "split_manifest": str(task_dir / "patient_train_test_90_10.json"),
                "projects": [gdc_project],
                "label_column": "normal_tumor_label",
                "label_map": {"normal": "normal", "tumor": "tumor"},
                "include_labels": ["normal", "tumor"],
                "concept_labels": ["normal", "tumor"],
                "association_split": "train",
                "representative_split": "all",
                "sae_variant": "tcga_sae_batch_topk_20x_interp",
                "ranking_mode": "attention_aware_optional",
                "classifier_run_dir": None,
                "top_concepts": 20,
                "candidate_latents": 150,
                "top_tiles_per_concept": 50,
            },
        )
        feature_ready = Counter((row["label"], int(row["feature_ready"])) for row in rows)
        summary = {
            "task_name": task_name,
            "gdc_project": gdc_project,
            "steering_direction": "normal -> tumor",
            "source_concept_label": "normal",
            "target_concept_label": "tumor",
            "normal_svs_available_in_gdc": len(normal_rows),
            "tumor_uni2_bags_available_locally": len(tumor_rows),
            "normal_features_root": str(args.normal_features_root),
            "normal_uni2_bags_available_locally": int(feature_ready[("normal", 1)]),
            "normal_uni2_bags_needed_before_concept_discovery": int(feature_ready[("normal", 0)]),
            "outputs": {
                "slide_labels": str(task_dir / "slide_labels.csv"),
                "concept_task": str(task_dir / "concept_task.json"),
                "normal_download_manifest": str(task_dir / "normal_all.gdc_manifest.tsv"),
                "normal_starter_download_manifest": str(task_dir / "normal_starter.gdc_manifest.tsv"),
            },
        }
        write_json(task_dir / "summary.json", summary)
        all_summary[task_name] = summary
        print(
            f"[prepared] {task_name}: tumor UNI2={len(tumor_rows)} normal GDC SVS={len(normal_rows)} "
            f"normal UNI2 ready={feature_ready[('normal', 1)]} target_label=tumor",
            flush=True,
        )
    write_json(args.out_root / "summary.json", all_summary)


if __name__ == "__main__":
    main()
