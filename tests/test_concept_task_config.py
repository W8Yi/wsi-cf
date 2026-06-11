from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wsi_cf.concepts.task_config import build_task_cohort, resolve_task_config


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def touch_h5(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not opened by cohort builder")
    return str(path)


def base_split(tmp_path: Path) -> Path:
    split = tmp_path / "split.json"
    write_json(
        split,
        {
            "train": [
                "/x/TCGA-AA-0001-01Z-00-DX1.h5",
                "/x/TCGA-BB-0002-01Z-00-DX1.h5",
                "/x/TCGA-CC-0003-01Z-00-DX1.h5",
            ],
            "test": [
                "/x/TCGA-DD-0004-01Z-00-DX1.h5",
            ],
        },
    )
    return split


def test_project_labels_map_luad_lusc(tmp_path: Path) -> None:
    label_source = tmp_path / "labels.tsv"
    write_tsv(
        label_source,
        [
            {
                "slide_key": "TCGA-AA-0001-01Z-00-DX1",
                "case_id": "TCGA-AA-0001",
                "sample_id": "TCGA-AA-0001-01",
                "project_dir": "TCGA-LUAD",
                "h5_path": touch_h5(tmp_path / "luad.h5"),
            },
            {
                "slide_key": "TCGA-BB-0002-01Z-00-DX1",
                "case_id": "TCGA-BB-0002",
                "sample_id": "TCGA-BB-0002-01",
                "project_dir": "TCGA-LUSC",
                "h5_path": touch_h5(tmp_path / "lusc.h5"),
            },
        ],
    )
    cfg = {
        "task_name": "luad_lusc",
        "label_source": str(label_source),
        "split_manifest": str(base_split(tmp_path)),
        "projects": ["TCGA-LUAD", "TCGA-LUSC"],
        "label_column": "project_dir",
        "label_map": {"TCGA-LUAD": "LUAD", "TCGA-LUSC": "LUSC"},
        "include_labels": ["LUAD", "LUSC"],
        "concept_labels": ["LUAD", "LUSC"],
    }

    cohort, skipped, summary = build_task_cohort(cfg)

    assert skipped == []
    assert [row["label"] for row in cohort] == ["LUAD", "LUSC"]
    assert summary["label_counts"] == {"LUAD": 1, "LUSC": 1}


def test_kirc_grade_mapping_skips_unknown_and_gx(tmp_path: Path) -> None:
    label_source = tmp_path / "labels.tsv"
    rows = []
    for idx, grade in enumerate(["G1", "G2", "G3", "G4", "GX", "Unknown"], start=1):
        case = f"TCGA-AA-000{idx}"
        rows.append(
            {
                "slide_key": f"{case}-01Z-00-DX1",
                "case_id": case,
                "sample_id": f"{case}-01",
                "project_dir": "TCGA-KIRC",
                "tumor_grade": grade,
                "h5_path": touch_h5(tmp_path / f"{grade}.h5"),
            }
        )
    write_tsv(label_source, rows)
    write_json(
        tmp_path / "split.json",
        {
            "train": [f"/x/TCGA-AA-000{idx}-01Z-00-DX1.h5" for idx in range(1, 7)],
            "test": [],
        },
    )
    cfg = {
        "task_name": "kirc_low_vs_high_grade",
        "label_source": str(label_source),
        "split_manifest": str(tmp_path / "split.json"),
        "projects": ["TCGA-KIRC"],
        "label_column": "tumor_grade",
        "label_map": {"G1": "low", "G2": "low", "G3": "high", "G4": "high"},
        "exclude_raw_labels": ["GX", "Unknown"],
        "include_labels": ["low", "high"],
        "concept_labels": ["low", "high"],
    }

    cohort, skipped, summary = build_task_cohort(cfg)

    assert [row["label"] for row in cohort] == ["high", "high", "low", "low"]
    assert summary["label_counts"] == {"high": 2, "low": 2}
    assert {row["reason"] for row in skipped} == {"excluded_or_unknown_label"}


def test_missing_label_column_fails_clearly(tmp_path: Path) -> None:
    label_source = tmp_path / "labels.tsv"
    write_tsv(
        label_source,
        [
            {
                "slide_key": "TCGA-AA-0001-01Z-00-DX1",
                "case_id": "TCGA-AA-0001",
                "project_dir": "TCGA-LUAD",
                "h5_path": touch_h5(tmp_path / "x.h5"),
            }
        ],
    )
    cfg = {
        "task_name": "bad",
        "label_source": str(label_source),
        "split_manifest": str(base_split(tmp_path)),
        "projects": ["TCGA-LUAD"],
        "label_column": "tumor_grade",
        "concept_labels": ["G1"],
    }

    with pytest.raises(ValueError, match="label column 'tumor_grade' not found"):
        build_task_cohort(cfg)


def test_resolve_task_config_accepts_numeric_bins(tmp_path: Path) -> None:
    task_json = tmp_path / "task.json"
    write_json(
        task_json,
        {
            "task_name": "purity",
            "label_column": "tumor_purity",
            "concept_labels": ["low", "high"],
            "numeric_bins": {"low": {"max": 0.35}, "high": {"min": 0.75}},
        },
    )

    cfg = resolve_task_config(task_json)

    assert cfg["sae_variant"] == "tcga_sae_batch_topk_20x_interp"
    assert cfg["numeric_bins"] == {"low": {"max": 0.35}, "high": {"min": 0.75}}
