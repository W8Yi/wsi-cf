from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_artifacts_module():
    script_path = REPO_ROOT / "paper" / "artifacts.py"
    spec = importlib.util.spec_from_file_location("paper_artifacts", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_rules(path: Path) -> None:
    rules = {
        "paper_view_dirs": ["inputs", "models", "concepts", "regions", "generated", "metrics"],
        "rules": [
            {
                "pattern": "artifacts/normal_tumor_features",
                "artifact_class": "feature_cache",
                "keep_level": "must_keep",
                "paper_status": "paper_input",
                "paper_view": "inputs",
                "notes": "features",
            },
            {
                "pattern": "artifacts/classifier_training",
                "artifact_class": "model",
                "keep_level": "must_keep",
                "paper_status": "paper_input",
                "paper_view": "models",
                "notes": "models",
            },
            {
                "pattern": "artifacts/hnscc_hpv_paper_benchmark",
                "artifact_class": "metrics",
                "keep_level": "paper_keep",
                "paper_status": "paper_result",
                "paper_view": "metrics",
                "notes": "paper benchmark",
            },
            {
                "pattern": "artifacts/*dryrun*",
                "artifact_class": "scratch",
                "keep_level": "scratch",
                "paper_status": "scratch",
                "paper_view": None,
                "notes": "dry run",
            },
            {
                "pattern": "artifacts/test_*",
                "artifact_class": "scratch",
                "keep_level": "scratch",
                "paper_status": "scratch",
                "paper_view": None,
                "notes": "test",
            },
            {
                "pattern": "artifacts/*strength_sweep*",
                "artifact_class": "legacy",
                "keep_level": "archive_candidate",
                "paper_status": "legacy",
                "paper_view": None,
                "notes": "sweep",
            },
        ],
        "default_rule": {
            "artifact_class": "generated",
            "keep_level": "archive_candidate",
            "paper_status": "unclassified",
            "paper_view": None,
            "notes": "review",
        },
    }
    path.write_text(json.dumps(rules))


def make_artifact(root: Path, name: str, content: str = "x") -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "payload.txt").write_text(content)
    return path


def test_scan_classifies_and_writes_index(tmp_path: Path) -> None:
    module = load_artifacts_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rules = tmp_path / "rules.json"
    write_rules(rules)

    make_artifact(artifacts, "normal_tumor_features", "feature")
    make_artifact(artifacts, "classifier_training", "model")
    make_artifact(artifacts, "grade_risk_training_dryrun", "scratch")
    make_artifact(artifacts, "kirc_strength_sweep_old", "legacy")

    scanned = module.scan_artifacts(artifacts, rules)
    by_name = {item.path.name: item for item in scanned}
    assert by_name["normal_tumor_features"].artifact_class == "feature_cache"
    assert by_name["normal_tumor_features"].keep_level == "must_keep"
    assert by_name["classifier_training"].artifact_class == "model"
    assert by_name["grade_risk_training_dryrun"].keep_level == "scratch"
    assert by_name["kirc_strength_sweep_old"].keep_level == "archive_candidate"

    index = tmp_path / "artifact_index.csv"
    module.write_index_csv(scanned, index)
    with index.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0].keys() == set(module.INDEX_COLUMNS)
    assert {row["path"] for row in rows} == {
        "artifacts/classifier_training",
        "artifacts/grade_risk_training_dryrun",
        "artifacts/kirc_strength_sweep_old",
        "artifacts/normal_tumor_features",
    }


def test_make_paper_view_links_selected_artifacts(tmp_path: Path, capsys) -> None:
    module = load_artifacts_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rules = tmp_path / "rules.json"
    write_rules(rules)

    make_artifact(artifacts, "normal_tumor_features", "feature")
    make_artifact(artifacts, "classifier_training", "model")
    make_artifact(artifacts, "test_output", "scratch")

    view_root = tmp_path / "paper_outputs" / "current"
    rc = module.main(
        [
            "make-paper-view",
            "--artifacts-root",
            str(artifacts),
            "--rules",
            str(rules),
            "--view-root",
            str(view_root),
        ]
    )
    assert rc == 0
    assert (view_root / "inputs" / "normal_tumor_features").is_symlink()
    assert (view_root / "models" / "classifier_training").is_symlink()
    assert not (view_root / "generated" / "test_output").exists()

    target = os.readlink(view_root / "inputs" / "normal_tumor_features")
    assert not target.startswith("/")

    rc = module.main(
        [
            "make-paper-view",
            "--artifacts-root",
            str(artifacts),
            "--rules",
            str(rules),
            "--view-root",
            str(view_root),
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    assert "[skip] existing link" in output


def test_make_paper_view_does_not_overwrite_real_directory(tmp_path: Path, capsys) -> None:
    module = load_artifacts_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rules = tmp_path / "rules.json"
    write_rules(rules)

    make_artifact(artifacts, "normal_tumor_features", "feature")
    real_dir = tmp_path / "paper_outputs" / "current" / "inputs" / "normal_tumor_features"
    real_dir.mkdir(parents=True)
    (real_dir / "keep.txt").write_text("do not overwrite")

    rc = module.main(
        [
            "make-paper-view",
            "--artifacts-root",
            str(artifacts),
            "--rules",
            str(rules),
            "--view-root",
            str(tmp_path / "paper_outputs" / "current"),
        ]
    )
    assert rc == 0
    assert not real_dir.is_symlink()
    assert (real_dir / "keep.txt").read_text() == "do not overwrite"
    assert "will not be overwritten" in capsys.readouterr().out


def test_plan_archive_prints_without_mutating(tmp_path: Path, capsys) -> None:
    module = load_artifacts_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rules = tmp_path / "rules.json"
    write_rules(rules)

    legacy = make_artifact(artifacts, "old_strength_sweep", "legacy")
    scratch = make_artifact(artifacts, "test_output", "scratch")

    rc = module.main(
        [
            "plan-archive",
            "--artifacts-root",
            str(artifacts),
            "--rules",
            str(rules),
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    assert "mv artifacts/old_strength_sweep" in output
    assert "mv artifacts/test_output" in output
    assert legacy.exists()
    assert scratch.exists()
