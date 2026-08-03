from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts/build_attention_budget_edit_manifests.py"
    spec = importlib.util.spec_from_file_location("build_attention_budget_edit_manifests", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ordered_budget_cells_keeps_base_then_adds_attention() -> None:
    script = load_script()
    base = [(1, 1), (3, 3)]
    attention = [
        {"gx": 0, "gy": 0, "attention_rank": 1},
        {"gx": 1, "gy": 1, "attention_rank": 2},
        {"gx": 2, "gy": 2, "attention_rank": 3},
        {"gx": 3, "gy": 3, "attention_rank": 4},
    ]

    cells = script.ordered_budget_cells(
        base_cells=base,
        attention_rows=attention,
        budget=4,
        start_mode="base_then_attention",
    )

    assert cells == [(1, 1), (3, 3), (0, 0), (2, 2)]


def test_ordered_budget_cells_attention_only() -> None:
    script = load_script()
    attention = [
        {"gx": 0, "gy": 0, "attention_rank": 1},
        {"gx": 1, "gy": 1, "attention_rank": 2},
    ]

    cells = script.ordered_budget_cells(
        base_cells=[(1, 1)],
        attention_rows=attention,
        budget=2,
        start_mode="attention_only",
    )

    assert cells == [(0, 0), (1, 1)]
