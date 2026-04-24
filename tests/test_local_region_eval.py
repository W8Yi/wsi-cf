from __future__ import annotations

import numpy as np

from wsi_cf.eval.local_region import (
    count_high_attention_cells,
    flatten_region_zgrid,
    passes_label_confidence,
    select_attention_mass_cells,
    select_balanced_region_rows,
    select_attention_cells,
    summarize_region_classifier_runs,
)


def test_flatten_region_zgrid_preserves_shape_and_cell_order() -> None:
    z_grid = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    bag, rows = flatten_region_zgrid(z_grid)
    assert bag.shape == (6, 4)
    assert rows[0]["cell_gx"] == 0
    assert rows[0]["cell_gy"] == 0
    assert rows[4]["cell_gx"] == 1
    assert rows[4]["cell_gy"] == 1
    np.testing.assert_allclose(bag[0], z_grid[0, 0])
    np.testing.assert_allclose(bag[5], z_grid[1, 2])


def test_select_attention_cells_topk_maps_to_grid_cells() -> None:
    attention = np.asarray([0.1, 0.9, 0.2, 0.3], dtype=np.float32)
    cells = select_attention_cells(
        attention=attention,
        grid_w=2,
        grid_h=2,
        mode="topk",
        top_k=2,
        percentile=90.0,
        min_cells=1,
        max_cells=4,
    )
    assert cells == [(1, 0), (1, 1)]


def test_select_attention_cells_percentile_respects_min_max() -> None:
    attention = np.asarray([0.91, 0.90, 0.89, 0.10], dtype=np.float32)
    cells = select_attention_cells(
        attention=attention,
        grid_w=2,
        grid_h=2,
        mode="percentile",
        top_k=1,
        percentile=95.0,
        min_cells=2,
        max_cells=2,
    )
    assert cells == [(0, 0), (1, 0)]


def test_select_attention_cells_respects_allowed_cells() -> None:
    attention = np.asarray([0.1, 0.9, 0.8, 0.7], dtype=np.float32)
    cells = select_attention_cells(
        attention=attention,
        grid_w=2,
        grid_h=2,
        mode="topk",
        top_k=2,
        percentile=90.0,
        min_cells=1,
        max_cells=4,
        allowed_cells={(0, 0), (0, 1)},
    )
    assert cells == [(0, 0), (0, 1)]


def test_label_confidence_filter_accepts_and_rejects_by_label() -> None:
    ok, reason, conf = passes_label_confidence(
        label=0,
        pred=0,
        prob_pos=0.10,
        min_confidence=0.75,
        require_label_match=True,
    )
    assert ok is True
    assert reason == "eligible_label_confidence"
    assert conf == 0.9

    ok, reason, conf = passes_label_confidence(
        label=1,
        pred=0,
        prob_pos=0.90,
        min_confidence=0.75,
        require_label_match=True,
    )
    assert ok is False
    assert reason == "pred_label_mismatch"
    assert conf == 0.9


def test_count_high_attention_cells_uses_percentile_and_allowed_cells() -> None:
    attention = np.asarray([0.1, 0.2, 0.9, 0.8], dtype=np.float32)
    count, threshold = count_high_attention_cells(
        attention=attention,
        grid_w=2,
        grid_h=2,
        percentile=75.0,
        allowed_cells={(0, 0), (1, 0), (0, 1)},
    )
    assert threshold > 0.6
    assert count == 1


def test_select_attention_mass_cells_varies_count_and_respects_bounds() -> None:
    attention = np.asarray([0.20, 0.18, 0.10, 0.08, 0.04], dtype=np.float32)
    cells, mass, ok, reason = select_attention_mass_cells(
        attention=attention,
        grid_w=5,
        grid_h=1,
        target_mass=0.35,
        min_cells=2,
        max_cells=4,
        allowed_cells={(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)},
    )
    assert cells == [(0, 0), (1, 0)]
    assert mass >= 0.35
    assert ok is True
    assert reason == "eligible_attention_mass"


def test_select_attention_mass_cells_rejects_when_mass_unreachable() -> None:
    attention = np.asarray([0.10, 0.09, 0.08, 0.07], dtype=np.float32)
    cells, mass, ok, reason = select_attention_mass_cells(
        attention=attention,
        grid_w=4,
        grid_h=1,
        target_mass=0.35,
        min_cells=2,
        max_cells=3,
        allowed_cells={(0, 0), (1, 0), (2, 0), (3, 0)},
    )
    assert cells == [(0, 0), (1, 0), (2, 0)]
    assert mass < 0.35
    assert ok is False
    assert reason == "insufficient_attention_mass"


def test_balanced_region_selection_requires_per_label_counts() -> None:
    rows = [
        {"region_id": "a", "slide_key": "s1", "label": 0, "eligible": True},
        {"region_id": "b", "slide_key": "s2", "label": 0, "eligible": True},
        {"region_id": "c", "slide_key": "s3", "label": 1, "eligible": True},
        {"region_id": "d", "slide_key": "s4", "label": 1, "eligible": False},
    ]
    selected, counts = select_balanced_region_rows(rows, per_label=1)
    assert [row["region_id"] for row in selected] == ["a", "c"]
    assert counts == {0: 2, 1: 1}


def test_summarize_region_classifier_runs_reports_expected_metrics() -> None:
    summary = summarize_region_classifier_runs(
        [
            {
                "pred_before": 0,
                "pred_after": 1,
                "source_label": 0,
                "target_label": 1,
                "target_prob_before": 0.2,
                "target_prob_after": 0.7,
            },
            {
                "pred_before": 1,
                "pred_after": 1,
                "source_label": 1,
                "target_label": 0,
                "target_prob_before": 0.1,
                "target_prob_after": 0.05,
            },
        ]
    )
    assert summary["n_regions"] == 2
    assert summary["accuracy_before"] == 1.0
    assert summary["accuracy_after"] == 0.5
    assert summary["target_pred_rate_after"] == 0.5
    assert summary["target_shift_success_rate"] == 0.5
