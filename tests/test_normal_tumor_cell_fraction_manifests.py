from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts/build_normal_tumor_cell_fraction_manifests.py"
    spec = importlib.util.spec_from_file_location("build_normal_tumor_cell_fraction_manifests", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_fractions_sorts_unique_values() -> None:
    script = load_script()

    assert script.parse_fractions("1,0.5,0.65,0.5") == [0.5, 0.65, 1.0]


def test_choose_fraction_cells_uses_ceil_and_keeps_attention_order() -> None:
    script = load_script()
    ranked = [
        {"gx": 2, "gy": 0, "attention": 0.9},
        {"gx": 0, "gy": 1, "attention": 0.8},
        {"gx": 1, "gy": 1, "attention": 0.7},
    ]

    assert script.choose_fraction_cells(ranked, 0.5) == [(2, 0), (0, 1)]
    assert script.choose_fraction_cells(ranked, 1.0) == [(2, 0), (0, 1), (1, 1)]


def test_rank_local_cells_by_attention_respects_valid_mask(tmp_path: Path) -> None:
    script = load_script()
    mask = np.array(
        [
            [1, 1],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    mask_path = tmp_path / "valid_feature_mask.npy"
    np.save(mask_path, mask)
    row = {
        "region_id": "r0",
        "region_gx0": "10",
        "region_gy0": "20",
        "valid_mask_path": str(mask_path),
    }
    cell_to_index = {
        (10, 20): 0,
        (11, 20): 1,
        (10, 21): 2,
        (11, 21): 3,
    }
    attention = np.asarray([0.2, 0.7, 0.9, 0.5], dtype=np.float32)

    ranked = script.rank_local_cells_by_attention(
        row=row,
        attention=attention,
        cell_to_index=cell_to_index,
        region_side=2,
        use_existing_valid_mask=True,
    )

    assert [(item["gx"], item["gy"]) for item in ranked] == [(1, 0), (1, 1), (0, 0)]
    assert [item["attention_rank"] for item in ranked] == [1, 2, 3]


def test_empty_valid_mask_path_falls_back_to_h5_cells() -> None:
    script = load_script()
    row = {
        "region_id": "r0",
        "region_gx0": "0",
        "region_gy0": "0",
        "valid_mask_path": "",
    }
    cell_to_index = {
        (0, 0): 0,
        (1, 0): 1,
    }
    attention = np.asarray([0.2, 0.7], dtype=np.float32)

    ranked = script.rank_local_cells_by_attention(
        row=row,
        attention=attention,
        cell_to_index=cell_to_index,
        region_side=2,
        use_existing_valid_mask=True,
    )

    assert [(item["gx"], item["gy"]) for item in ranked] == [(1, 0), (0, 0)]
