from __future__ import annotations

import numpy as np
import torch

from wsi_cf.eval.grade_risk import grade_risk_metrics, map_region_cells_to_bag, replace_region_features
from wsi_cf.models.mil import (
    AttentionMILOrdinalRegressor,
    AttentionMILRegressor,
    GatedAttentionMILOrdinalRegressor,
    GatedAttentionMILRegressor,
)


def test_grade_risk_regressors_return_bounded_slide_score() -> None:
    features = torch.randn(7, 4)
    for model in (
        AttentionMILRegressor(embed_dim=4, hidden_dim=5, attn_dim=3),
        GatedAttentionMILRegressor(embed_dim=4, hidden_dim=5, attn_dim=3),
    ):
        score, attention, results = model(features)
        assert score.shape == (1, 1)
        assert 0.0 <= float(score.item()) <= 1.0
        assert attention.shape == (1, 7)
        assert "risk_logit" in results


def test_ordinal_grade_risk_models_report_threshold_probabilities() -> None:
    features = torch.randn(7, 4)
    for model in (
        AttentionMILOrdinalRegressor(embed_dim=4, hidden_dim=5, attn_dim=3),
        GatedAttentionMILOrdinalRegressor(embed_dim=4, hidden_dim=5, attn_dim=3),
    ):
        score, attention, results = model(features)
        assert score.shape == (1, 1)
        assert 0.0 <= float(score.item()) <= 1.0
        assert attention.shape == (1, 7)
        assert results["ordinal_logits"].shape == (1, 3)
        assert results["ordinal_probs"].shape == (1, 3)


def test_grade_risk_metrics_reports_monotonic_predictions() -> None:
    metrics = grade_risk_metrics([0.0, 0.33, 0.66, 1.0], [0.05, 0.30, 0.70, 0.95])
    assert metrics["mae"] < 0.05
    assert metrics["pearson"] > 0.99
    assert metrics["spearman"] == 1.0


def test_replace_region_features_maps_local_cells_back_to_full_bag() -> None:
    coords = np.asarray([[0, 0], [256, 0], [0, 256], [256, 256], [512, 512]])
    original = np.zeros((5, 2), dtype=np.float32)
    generated = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    mapping = map_region_cells_to_bag(coords, region_gx0=0, region_gy0=0, grid_shape=(2, 2))

    full, n_full = replace_region_features(original, generated, mapping)
    target, n_target = replace_region_features(original, generated, mapping, cells={(1, 0)})

    assert n_full == 4
    assert np.array_equal(full[3], np.asarray([7, 8], dtype=np.float32))
    assert n_target == 1
    assert np.array_equal(target[1], np.asarray([3, 4], dtype=np.float32))
    assert np.array_equal(target[0], np.asarray([0, 0], dtype=np.float32))
