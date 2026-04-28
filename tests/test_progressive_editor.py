from __future__ import annotations

from pathlib import Path

import torch

from wsi_cf.common.io import write_json
from wsi_cf.steering.progressive import (
    CENTER_2X2_LOCAL_CELLS,
    advance_progressive_state,
    build_history_aware_preserve_map,
    load_progressive_edit_manifest,
    make_initial_progressive_state,
    plan_progressive_steps,
    window_local_cells,
)


def test_progressive_edit_manifest_loads_global_cells(tmp_path: Path) -> None:
    manifest_path = tmp_path / "edit_manifest.json"
    write_json(
        manifest_path,
        [
            {
                "region_id": "region_a",
                "target_cells": [{"gx": 1, "gy": 1}, [2, 1], "3,1"],
                "source_method": "attention",
            }
        ],
    )
    requests = load_progressive_edit_manifest(manifest_path)
    assert len(requests) == 1
    assert requests[0].region_id == "region_a"
    assert requests[0].target_cells == ((1, 1), (2, 1), (3, 1))
    assert requests[0].metadata["source_method"] == "attention"


def test_progressive_planner_covers_requested_target_cells_and_is_deterministic() -> None:
    target_cells = [(1, 1), (5, 1), (6, 1), (6, 5)]
    steps1 = plan_progressive_steps(
        target_cells=target_cells,
        grid_w=8,
        grid_h=8,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=256,
    )
    steps2 = plan_progressive_steps(
        target_cells=target_cells,
        grid_w=8,
        grid_h=8,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=256,
    )
    covered1 = sorted({cell for step in steps1 for cell in step.edit_cells_global}, key=lambda item: (item[1], item[0]))
    covered2 = sorted({cell for step in steps2 for cell in step.edit_cells_global}, key=lambda item: (item[1], item[0]))
    assert covered1 == sorted(target_cells, key=lambda item: (item[1], item[0]))
    assert covered2 == covered1
    assert [step.window.window_id for step in steps1] == [step.window.window_id for step in steps2]
    for step in steps1:
        local_cells = window_local_cells(window=step.window, global_cells=step.edit_cells_global)
        assert all(cell in CENTER_2X2_LOCAL_CELLS for cell in local_cells)


def test_progressive_planner_rejects_targets_outside_center_support() -> None:
    try:
        plan_progressive_steps(
            target_cells=[(0, 0)],
            grid_w=8,
            grid_h=8,
            window_grid_side=4,
            stride_cells=2,
            grid_step_px=256,
    )
    except RuntimeError as exc:
        assert "edit_support=center_2x2" in str(exc)
    else:
        raise AssertionError("Expected planner to reject a target outside the center 2x2 support")


def test_progressive_planner_border_relaxed_covers_region_edge_targets() -> None:
    target_cells = [(0, 0), (7, 0), (7, 4), (6, 7)]
    steps = plan_progressive_steps(
        target_cells=target_cells,
        grid_w=8,
        grid_h=8,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=256,
        edit_support="border_relaxed",
    )
    covered = sorted({cell for step in steps for cell in step.edit_cells_global}, key=lambda item: (item[1], item[0]))
    assert covered == sorted(target_cells, key=lambda item: (item[1], item[0]))


def test_progressive_state_tracks_edited_visited_and_history() -> None:
    steps = plan_progressive_steps(
        target_cells=[(1, 1), (5, 1)],
        grid_w=8,
        grid_h=8,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=256,
    )
    state = make_initial_progressive_state(target_cells=[(1, 1), (5, 1)])
    state = advance_progressive_state(state, window=steps[0].window, edit_cells_global=list(steps[0].edit_cells_global))
    assert state.edited_cells == ((1, 1),)
    assert steps[0].window.window_id in state.window_history
    assert len(state.visited_cells) == 16

    state = advance_progressive_state(state, window=steps[1].window, edit_cells_global=list(steps[1].edit_cells_global))
    assert state.edited_cells == ((1, 1), (5, 1))
    assert len(state.window_history) == 2
    assert len(state.visited_cells) > 16


def test_history_aware_preserve_map_prefers_visited_context_over_fresh_context() -> None:
    step = plan_progressive_steps(
        target_cells=[(5, 1)],
        grid_w=8,
        grid_h=8,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=256,
    )[0]
    preserve_map = build_history_aware_preserve_map(
        width=1024,
        height=1024,
        grid_step_px=256,
        window=step.window,
        edit_cells_global=list(step.edit_cells_global),
        visited_cells_global=[(4, 1)],
        preserve_edit_strength=0.5,
        preserve_visited_strength=0.9,
        preserve_fresh_context_strength=0.2,
    )
    assert preserve_map.shape == (1, 1, 1024, 1024)
    visited_patch = preserve_map[0, 0, 300, 50]
    edit_patch = preserve_map[0, 0, 300, 300]
    fresh_patch = preserve_map[0, 0, 800, 800]
    assert torch.isclose(visited_patch, torch.tensor(0.9))
    assert torch.isclose(edit_patch, torch.tensor(0.5))
    assert torch.isclose(fresh_patch, torch.tensor(0.2))
