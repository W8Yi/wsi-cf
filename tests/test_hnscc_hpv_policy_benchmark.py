import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import load_script_module


def test_rgb_diff_metrics_splits_target_and_context(tmp_path: Path) -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    generated = np.zeros((4, 4, 3), dtype=np.uint8)
    generated[:2, :2, :] = 30
    generated[2:, 2:, :] = 10
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(generated).save(generated_path)

    metrics = script.rgb_diff_metrics(
        source_path,
        generated_path,
        {"target_cells": [{"gx": 0, "gy": 0}], "grid_step_px": 2},
        lpips_model=None,
        device=None,
        require_deps=False,
    )

    assert metrics["rgb_abs_target_mean"] == 30.0
    assert math.isclose(metrics["rgb_abs_context_mean"], 10.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(metrics["masked_edit_ratio"], 9.0, rel_tol=1e-6)

    cell_rows = script.rgb_cell_metrics(source_path, generated_path, 2, [{"gx": 0, "gy": 0}])
    target_cell = next(row for row in cell_rows if row["cell_gx"] == 0 and row["cell_gy"] == 0)
    assert target_cell["is_target_cell"] is True
    assert target_cell["rgb_abs_mean"] == 30.0


def test_uni_cell_metrics_reports_target_and_context() -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")
    source = np.zeros((2, 2, 2), dtype=np.float32)
    generated = np.zeros((2, 2, 2), dtype=np.float32)
    generated[0, 0, :] = np.asarray([3.0, 4.0], dtype=np.float32)
    generated[1, 1, :] = np.asarray([0.0, 2.0], dtype=np.float32)

    summary, rows = script.uni_cell_metrics(source, generated, [{"gx": 0, "gy": 0}])

    assert summary["uni_l2_target_mean"] == 5.0
    assert math.isclose(summary["uni_l2_context_mean"], 2.0 / 3.0, rel_tol=1e-6)
    assert len(rows) == 4
    assert rows[0]["is_target_cell"] is True


def test_target_probability_uses_requested_direction() -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")
    row = {"prob_hpv_pos": 0.25, "prob_hpv_neg": 0.75}

    assert script.target_probability(row, "hpv_neg") == 0.75
    assert script.target_probability(row, "hpv_pos") == 0.25
    assert script.target_label("hpv_neg") == "HPV-"


def test_default_ours_policy_uses_showcase_best() -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")

    specs = script.parse_method_specs(None)
    ours = next(spec for spec in specs if spec.name == "ours")

    assert ours.policy_path.name == "showcase_best.json"


def test_split_manifest_by_direction_sends_source_labels_to_opposite_targets() -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")
    manifest_rows = [{"run_id": "pos_source", "region_id": "r1"}, {"run_id": "neg_source", "region_id": "r2"}]
    region_rows = [{"region_id": "r1", "label": "1"}, {"region_id": "r2", "label": "0"}]

    grouped = script.split_manifest_by_direction(manifest_rows=manifest_rows, region_rows=region_rows)

    assert [row["run_id"] for row in grouped["hpv_neg"]] == ["pos_source"]
    assert [row["run_id"] for row in grouped["hpv_pos"]] == ["neg_source"]
    assert grouped["hpv_neg"][0]["target_label"] == "HPV-"
    assert grouped["hpv_pos"][0]["target_label"] == "HPV+"


def test_classification_summary_handles_bidirectional_metrics() -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")
    rows = [
        {"method": "ours", "target_label_id": 0, "pred_label_id": 0, "prob_hpv_pos": 0.1},
        {"method": "ours", "target_label_id": 1, "pred_label_id": 1, "prob_hpv_pos": 0.9},
        {"method": "source", "source_label_id": 0, "pred_label_id": 0, "prob_hpv_pos": 0.2},
        {"method": "source", "source_label_id": 1, "pred_label_id": 1, "prob_hpv_pos": 0.8},
    ]

    summary = {row["method"]: row for row in script.classification_summary(rows)}

    assert summary["ours"]["target_accuracy"] == 1.0
    assert summary["ours"]["target_f1"] == 1.0
    assert summary["ours"]["target_auroc"] == 1.0
    assert summary["source_original_label"]["target_accuracy"] == 1.0


def test_preflight_dependency_reports_missing_lpips(monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")
    real_import = __import__

    def fake_import(name: str, *args, **kwargs):
        if name == "lpips":
            raise ModuleNotFoundError("No module named 'lpips'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="lpips"):
        script.preflight_dependencies(require_paper_deps=True)


def test_window_consistency_and_seam_score_on_synthetic_windows(tmp_path: Path) -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    generated = np.zeros((4, 4, 3), dtype=np.uint8)
    generated[:, 1:, :] = 20
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(generated).save(generated_path)

    steps = tmp_path / "steps"
    (steps / "step_01").mkdir(parents=True)
    (steps / "step_02").mkdir(parents=True)
    first = np.zeros((4, 3, 3), dtype=np.uint8)
    first[:, 1:, :] = 5
    second = np.full((4, 3, 3), 10, dtype=np.uint8)
    Image.fromarray(first).save(steps / "step_01" / "steered_window.png")
    Image.fromarray(second).save(steps / "step_02" / "steered_window.png")

    metrics = script.window_consistency_metrics(
        source_path,
        generated_path,
        {
            "window_history": [
                {"commit_bounds_global": {"x0": 0, "y0": 0, "x1": 3, "y1": 4}},
                {"commit_bounds_global": {"x0": 1, "y0": 0, "x1": 4, "y1": 4}},
            ]
        },
        tmp_path,
    )

    assert metrics["overlap_consistency_count"] == 1
    assert metrics["overlap_consistency_rgb_abs_mean"] > 0
    assert metrics["seam_score_rgb_abs"] > metrics["source_seam_score_rgb_abs"]


def test_concept_fidelity_from_latents_moves_toward_target() -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")
    source = np.asarray([[1.0, 0.0, 0.2], [1.0, 0.0, 0.1]], dtype=np.float32)
    generated = np.asarray([[0.0, 1.0, 0.8], [0.0, 1.0, 0.7]], dtype=np.float32)

    metrics = script.concept_fidelity_from_latents(
        source,
        generated,
        np.asarray([True, True]),
        target_proto=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        source_proto=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        target_latent=2,
        source_latent=0,
        target_assoc_latents=[2],
        source_assoc_latents=[0],
    )

    assert metrics["delta_target_proto_cos"] > 0
    assert metrics["delta_source_proto_cos"] < 0
    assert metrics["delta_target_latent_activation"] > 0
    assert metrics["delta_source_latent_activation"] < 0


def test_resolve_split_h5_path_falls_back_to_tcga_features_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script_module("run_hnscc_hpv_policy_benchmark.py")
    fallback = tmp_path / "TCGA_features" / "TCGA-HNSC" / "features_uni2" / "TCGA-XX-0000-01Z-00-DX1.h5"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"fake")
    monkeypatch.setattr(script, "DEFAULT_TCGA_FEATURES_ROOT", tmp_path / "TCGA_features")

    resolved = script.resolve_split_h5_path(
        {
            "h5_path": "/missing/extracted_features/TCGA-XX-0000-01Z-00-DX1.uuid.h5",
            "project_dir": "TCGA-HNSC",
            "slide_key": "TCGA-XX-0000-01Z-00-DX1",
        }
    )

    assert resolved == fallback


def test_balanced_region_selection_enforces_slide_region_shape() -> None:
    from wsi_cf.data.region_selection import select_final_region_candidates

    eligible_by_label: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for label in (0, 1):
        for slide_idx in range(3):
            for rank in range(3):
                eligible_by_label[label].append(
                    {
                        "label": label,
                        "slide_key": f"label{label}_slide{slide_idx}",
                        "candidate_rank_in_slide": rank + 1,
                    }
                )

    rows = select_final_region_candidates(
        eligible_by_label,
        final_regions_per_label=99,
        final_slides_per_label=2,
        final_regions_per_slide=2,
    )

    assert len(rows) == 8
    assert {row["selection_region_rank_in_slide"] for row in rows} == {1, 2}
    for label in (0, 1):
        label_rows = [row for row in rows if row["label"] == label]
        assert len({row["slide_key"] for row in label_rows}) == 2
        assert len(label_rows) == 4
