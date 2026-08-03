from __future__ import annotations

from conftest import load_script_module


def test_effective_budgets_clamps_and_includes_full_endpoint() -> None:
    script = load_script_module("build_prediction_transition_manifests.py")

    assert script.effective_budgets([1, 2, 4, 8, 64], valid_count=5) == [1, 2, 4, 5]
    assert script.effective_budgets([1, 2, 4], valid_count=5, include_full_endpoint=False) == [1, 2, 4]


def test_region_requests_are_cumulative_and_skip_random_full_budget() -> None:
    script = load_script_module("build_prediction_transition_manifests.py")
    row = {"region_id": "R1", "slide_key": "S1"}
    ranked = [
        {"gx": 0, "gy": 0, "attention": 0.9, "attention_rank": 1},
        {"gx": 1, "gy": 0, "attention": 0.8, "attention_rank": 2},
        {"gx": 2, "gy": 0, "attention": 0.7, "attention_rank": 3},
    ]

    requests, selected, summary = script.build_region_requests(
        row=row,
        ranked_cells=ranked,
        requested_budgets=[1, 2, 64],
        task_name="task",
        direction="a_to_b",
        source_label="a",
        target_label="b",
        random_repeats=2,
        seed=7,
        run_prefix="bench",
    )

    attention = [req for req in requests if req["selector"] == "attention"]
    random = [req for req in requests if req["selector"] == "random"]
    assert [req["budget"] for req in attention] == [1, 2, 3]
    assert [req["budget"] for req in random] == [1, 2, 1, 2]
    assert attention[1]["target_cells"] == [{"gx": 0, "gy": 0}, {"gx": 1, "gy": 0}]
    assert all(not req["budget_is_full"] for req in random)
    assert len(selected) == sum(len(req["target_cells"]) for req in requests)
    assert len(summary) == len(requests)


def test_region_requests_can_skip_full_endpoint_for_probes() -> None:
    script = load_script_module("build_prediction_transition_manifests.py")
    row = {"region_id": "R1", "slide_key": "S1"}
    ranked = [
        {"gx": idx, "gy": 0, "attention": float(10 - idx), "attention_rank": idx + 1}
        for idx in range(6)
    ]

    requests, _, _ = script.build_region_requests(
        row=row,
        ranked_cells=ranked,
        requested_budgets=[1, 2, 4],
        task_name="task",
        direction="a_to_b",
        source_label="a",
        target_label="b",
        random_repeats=0,
        seed=7,
        run_prefix="bench",
        include_full_endpoint=False,
    )

    assert [req["budget"] for req in requests] == [1, 2, 4]
    assert all(not req["budget_is_full"] for req in requests)


def test_random_requests_are_deterministic_for_same_seed() -> None:
    script = load_script_module("build_prediction_transition_manifests.py")
    row = {"region_id": "R1", "slide_key": "S1"}
    ranked = [
        {"gx": idx, "gy": 0, "attention": float(10 - idx), "attention_rank": idx + 1}
        for idx in range(6)
    ]
    kwargs = dict(
        row=row,
        ranked_cells=ranked,
        requested_budgets=[4],
        task_name="task",
        direction="a_to_b",
        source_label="a",
        target_label="b",
        random_repeats=1,
        seed=11,
        run_prefix="bench",
    )

    first, _, _ = script.build_region_requests(**kwargs)
    second, _, _ = script.build_region_requests(**kwargs)

    first_random = [req for req in first if req["selector"] == "random"][0]["target_cells"]
    second_random = [req for req in second if req["selector"] == "random"][0]["target_cells"]
    assert first_random == second_random
    assert first_random != [{"gx": idx, "gy": 0} for idx in range(4)]
