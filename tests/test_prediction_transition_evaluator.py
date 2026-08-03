from __future__ import annotations

import numpy as np

from conftest import load_script_module


def test_expected_grade_uses_label_order() -> None:
    script = load_script_module("evaluate_prediction_transition_edits.py")
    probs = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    id_to_label = {0: "GG1", 1: "GG2", 2: "GG3", 3: "GG4"}

    assert np.isclose(script.expected_grade(probs, id_to_label, "GG1,GG2,GG3,GG4"), 3.0)


def test_expected_grade_returns_none_for_non_grade_labels() -> None:
    script = load_script_module("evaluate_prediction_transition_edits.py")

    assert script.expected_grade(np.asarray([0.3, 0.7]), {0: "normal", 1: "tumor"}, "GG1,GG2") is None


def test_summary_reports_transition_metrics() -> None:
    script = load_script_module("evaluate_prediction_transition_edits.py")
    rows = [
        {
            "task_name": "task",
            "direction": "a_to_b",
            "selector": "attention",
            "budget": 4,
            "source_target_probability": 0.2,
            "edited_target_probability": 0.5,
            "target_probability_delta": 0.3,
            "flip_to_target": 1,
            "edited_pred_is_target": 1,
        },
        {
            "task_name": "task",
            "direction": "a_to_b",
            "selector": "attention",
            "budget": 4,
            "source_target_probability": 0.3,
            "edited_target_probability": 0.25,
            "target_probability_delta": -0.05,
            "flip_to_target": 0,
            "edited_pred_is_target": 0,
        },
    ]

    summary = script.summarize_group(rows, ["task_name", "direction", "selector", "budget"])

    assert len(summary) == 1
    assert np.isclose(summary[0]["mean_target_probability_delta"], 0.125)
    assert summary[0]["positive_target_delta_rate"] == 0.5
    assert summary[0]["flip_to_target_rate"] == 0.5


def test_random_vs_attention_summary_compares_means() -> None:
    script = load_script_module("evaluate_prediction_transition_edits.py")
    rows = [
        {"task_name": "task", "direction": "d", "budget": 1, "selector": "attention", "target_probability_delta": 0.4},
        {"task_name": "task", "direction": "d", "budget": 1, "selector": "random", "target_probability_delta": 0.1},
        {"task_name": "task", "direction": "d", "budget": 1, "selector": "random", "target_probability_delta": 0.3},
    ]

    summary = script.random_vs_attention_summary(rows)

    assert len(summary) == 1
    assert np.isclose(summary[0]["attention_minus_random_delta"], 0.2)


def test_local_grid_replacement_only_changes_selected_cells() -> None:
    script = load_script_module("evaluate_prediction_transition_edits.py")
    source = np.zeros((2, 2, 3), dtype=np.float32)
    generated = np.ones((2, 2, 3), dtype=np.float32)

    edited, replaced = script.replace_local_grid_cells(source, generated, {(1, 0), (9, 9)})
    full, full_replaced = script.replace_local_grid_all(source, generated)

    assert replaced == 1
    assert np.allclose(edited[0, 1], 1.0)
    assert np.allclose(edited[0, 0], 0.0)
    assert full_replaced == 4
    assert script.flatten_grid_bag(full).shape == (4, 3)
