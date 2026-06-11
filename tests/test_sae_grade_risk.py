from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wsi_cf.eval.sae_grade_risk import (
    aggregate_summaries_by_case,
    aggregate_activation_array,
    apply_feature_scaler,
    build_feature_matrix,
    fit_feature_scaler,
    load_sae_grade_risk_model,
    select_top_latents,
    select_stable_top_latents,
)
from wsi_cf.models.concept_risk import LinearProportionalOddsRisk


def test_aggregate_activation_array_reports_mean_fraction_and_top_fraction() -> None:
    activations = np.asarray([[0.0, 1.0], [2.0, 0.0], [4.0, 3.0], [0.0, 5.0]], dtype=np.float32)
    summary = aggregate_activation_array(activations, active_threshold=1e-6, top_fraction=0.5)

    assert np.allclose(summary["mean_activation"], [1.5, 2.25])
    assert np.allclose(summary["fraction_active"], [0.5, 0.75])
    assert np.allclose(summary["top_fraction_mean"], [3.0, 4.0])


def test_training_only_selection_and_scaler_use_supplied_training_arrays() -> None:
    train_targets = [0.0, 0.33, 0.66, 1.0]
    train = {
        "mean_activation": np.asarray([[0, 0], [1, 0], [2, 0], [3, 100]], dtype=np.float32),
        "fraction_active": np.zeros((4, 2), dtype=np.float32),
        "top_fraction_mean": np.zeros((4, 2), dtype=np.float32),
    }
    selected, rows = select_top_latents(train, train_targets, latent_ids=np.asarray([7, 9]), top_n=1)

    assert selected.tolist() == [7]
    assert rows[0]["winning_statistic"] == "mean_activation"

    full = {
        "mean_activation": np.asarray([[0, 0], [1, 0], [2, 0], [3, 100], [300, 0]], dtype=np.float32),
        "fraction_active": np.zeros((5, 2), dtype=np.float32),
        "top_fraction_mean": np.zeros((5, 2), dtype=np.float32),
    }
    matrix, _ = build_feature_matrix(full, selected_latents=selected, latent_ids=np.asarray([7, 9]))
    mean, scale = fit_feature_scaler(matrix[:4])
    scaled = apply_feature_scaler(matrix, mean, scale)

    assert mean[0] == 1.5
    assert scaled[4, 0] > 100


def test_case_aggregation_gives_each_patient_one_summary() -> None:
    rows = [
        {"case_id": "A", "slide_key": "A-1", "raw_grade": "G1", "risk_target": 0.0, "split": "train"},
        {"case_id": "A", "slide_key": "A-2", "raw_grade": "G1", "risk_target": 0.0, "split": "train"},
        {"case_id": "B", "slide_key": "B-1", "raw_grade": "G3", "risk_target": 0.66, "split": "test"},
    ]
    arrays = {
        "mean_activation": np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32),
        "fraction_active": np.asarray([[0.0], [1.0], [1.0]], dtype=np.float32),
        "top_fraction_mean": np.asarray([[1.0], [3.0], [5.0]], dtype=np.float32),
        "latent_ids": np.asarray([17], dtype=np.int64),
    }

    case_rows, case_arrays = aggregate_summaries_by_case(rows, arrays)

    assert [row["case_id"] for row in case_rows] == ["A", "B"]
    assert case_rows[0]["n_slides"] == 2
    assert np.allclose(case_arrays["mean_activation"][:, 0], [1.0, 4.0])
    assert np.allclose(case_arrays["top_fraction_mean"][:, 0], [2.0, 5.0])


def test_stable_selection_is_deterministic_and_records_fold_support() -> None:
    targets = [0.0, 0.0, 0.33, 0.33, 0.66, 0.66, 1.0, 1.0]
    trend = np.asarray(targets, dtype=np.float32)
    train = {
        "mean_activation": np.stack([trend, np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)], axis=1),
        "fraction_active": np.zeros((8, 2), dtype=np.float32),
        "top_fraction_mean": np.zeros((8, 2), dtype=np.float32),
    }

    selected, rows = select_stable_top_latents(
        train,
        targets,
        strata=["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"],
        latent_ids=np.asarray([5, 6]),
        top_n=1,
        n_folds=2,
        seed=7,
    )

    assert selected.tolist() == [5]
    assert rows[0]["selection_fold_count"] == 2
    assert rows[0]["selection_fold_fraction"] == 1.0


def test_linear_proportional_odds_outputs_ordered_threshold_probabilities() -> None:
    model = LinearProportionalOddsRisk(n_features=2, n_thresholds=3)
    risk, probabilities, _ = model(torch.zeros((2, 2)))

    assert risk.shape == (2,)
    assert torch.all((risk >= 0.0) & (risk <= 1.0))
    assert torch.all(model.thresholds[1:] > model.thresholds[:-1])
    assert torch.all(probabilities[:, 0] >= probabilities[:, 1])
    assert torch.all(probabilities[:, 1] >= probabilities[:, 2])


def test_sae_concept_checkpoint_load_reproduces_prediction(tmp_path: Path) -> None:
    model = LinearProportionalOddsRisk(n_features=3, n_thresholds=3)
    features = torch.asarray([[0.2, -0.1, 1.0]], dtype=torch.float32)
    expected = model(features)[0].detach()
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_config": {"n_features": 3, "n_thresholds": 3},
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    loaded, _ = load_sae_grade_risk_model(checkpoint_path, device=torch.device("cpu"))

    assert torch.allclose(loaded(features)[0], expected)
