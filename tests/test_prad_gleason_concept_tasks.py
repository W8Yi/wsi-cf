from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts/prepare_prad_gleason_concept_tasks.py"
    spec = importlib.util.spec_from_file_location("prepare_prad_gleason_concept_tasks", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_isup_grade_group_mapping() -> None:
    script = load_script()

    assert script.isup_grade_group(3, 3) == 1
    assert script.isup_grade_group(2, 4) == 1
    assert script.isup_grade_group(3, 4) == 2
    assert script.isup_grade_group(4, 3) == 3
    assert script.isup_grade_group(4, 4) == 4
    assert script.isup_grade_group(4, 5) == 5


def test_build_case_labels_derives_score_and_binary_target() -> None:
    script = load_script()
    cases = [
        {
            "submitter_id": "TCGA-AA-0001",
            "diagnoses": [
                {
                    "tissue_or_organ_of_origin": "Prostate gland",
                    "primary_gleason_grade": "Pattern 3",
                    "secondary_gleason_grade": "Pattern 4",
                }
            ],
        },
        {
            "submitter_id": "TCGA-AA-0002",
            "diagnoses": [
                {
                    "tissue_or_organ_of_origin": "Prostate gland",
                    "primary_gleason_grade": "Pattern 4",
                    "secondary_gleason_grade": "Pattern 3",
                }
            ],
        },
    ]

    labels = script.build_case_labels(cases)

    assert labels["TCGA-AA-0001"]["gleason_score_label"] == "GS7"
    assert labels["TCGA-AA-0001"]["grade_group"] == "GG2"
    assert labels["TCGA-AA-0001"]["low_high_grade"] == "low"
    assert labels["TCGA-AA-0002"]["grade_group"] == "GG3"
    assert labels["TCGA-AA-0002"]["low_high_grade"] == "high"
