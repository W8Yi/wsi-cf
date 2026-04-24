from __future__ import annotations

import numpy as np

from conftest import load_script_module


def test_infer_region_grid_shape_accepts_divisible_sizes() -> None:
    script = load_script_module("run_region_image_tile_selector.py")
    assert script.infer_grid_shape_from_image_size(width=2048, height=2048, grid_step_px=256) == (8, 8)
    assert script.infer_grid_shape_from_image_size(width=4096, height=2048, grid_step_px=256) == (8, 16)


def test_infer_region_grid_shape_rejects_non_divisible_sizes() -> None:
    script = load_script_module("run_region_image_tile_selector.py")
    try:
        script.infer_grid_shape_from_image_size(width=2050, height=2048, grid_step_px=256)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected non-divisible region size to raise ValueError")


def test_connected_support_expansion_adds_touching_component() -> None:
    script = load_script_module("run_region_image_tile_selector.py")
    selected = [(1, 1)]
    high_cells = [(1, 1), (2, 1), (2, 2), (4, 4)]
    importance = {(1, 1): 0.9, (2, 1): 0.8, (2, 2): 0.7, (4, 4): 0.95}
    added = script.expand_by_connected_support(
        selected_cells=selected,
        high_cells=high_cells,
        importance_by_cell=importance,
        min_component_size=2,
        min_touching_neighbors=1,
    )
    assert set(added) == {(2, 1), (2, 2)}


def test_fill_selection_gaps_bridges_hole_without_leaving_valid_cells() -> None:
    script = load_script_module("run_region_image_tile_selector.py")
    valid = {(0, 0), (1, 0), (2, 0)}
    added = script.fill_selection_gaps(
        selected_cells=[(0, 0), (2, 0)],
        valid_cells=valid,
        min_neighbors=3,
        max_iters=2,
    )
    assert added == [(1, 0)]


def test_feature_neighbor_expansion_adds_similar_neighbor() -> None:
    script = load_script_module("run_region_image_tile_selector.py")
    selected = script.expand_selected_cells_by_feature_neighbors(
        seed_cells=[(0, 0)],
        valid_cells=[(0, 0), (1, 0), (2, 0)],
        region_features=np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        combined_by_cell={(0, 0): 0.9, (1, 0): 0.4, (2, 0): 0.4},
        similarity_threshold=0.95,
        min_combined_importance=0.1,
        max_cells=2,
    )
    assert selected == [(0, 0), (1, 0)]
