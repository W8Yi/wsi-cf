from __future__ import annotations

import numpy as np

from conftest import load_script_module


def test_rank_cells_by_attention_orders_eligible_cells_only() -> None:
    script = load_script_module("evaluate_luad_lusc_cell_budget_edits.py")
    attention = np.asarray([0.1, 0.8, 0.5, 0.9], dtype=np.float32)
    mapping = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}

    cells = script.rank_cells_by_attention(attention, mapping, [(1, 0), (0, 1), (0, 0)])

    assert cells == [(1, 0), (0, 1), (0, 0)]


def test_cell_budget_summary_reports_lusc_probability_movement() -> None:
    script = load_script_module("evaluate_luad_lusc_cell_budget_edits.py")
    summary = script.summarize_results(
        [
            {
                "cell_count": 4,
                "source_lusc_probability": 0.2,
                "edited_lusc_probability": 0.4,
                "lusc_probability_delta": 0.2,
                "edited_pred_is_lusc": 0,
                "flip_luad_to_lusc": 0,
            },
            {
                "cell_count": 4,
                "source_lusc_probability": 0.3,
                "edited_lusc_probability": 0.7,
                "lusc_probability_delta": 0.4,
                "edited_pred_is_lusc": 1,
                "flip_luad_to_lusc": 1,
            },
        ]
    )

    assert summary[0]["region_fraction_percent"] == 6.25
    assert summary[0]["mean_cells_replaced"] == 4.0
    assert np.isclose(summary[0]["mean_lusc_probability_delta"], 0.3)
    assert summary[0]["positive_delta_rate"] == 1.0
    assert summary[0]["flip_luad_to_lusc_rate"] == 0.5
