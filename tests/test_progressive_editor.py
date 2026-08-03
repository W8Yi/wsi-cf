from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from wsi_cf.common.io import write_json
from wsi_cf.steering.progressive import (
    CENTER_2X2_LOCAL_CELLS,
    advance_progressive_state,
    build_history_aware_preserve_map,
    draw_cells_overlay,
    draw_step_edit_area_zoom_4x4,
    draw_step_region_overlay,
    load_progressive_edit_manifest,
    make_initial_progressive_state,
    plan_progressive_steps,
    ProgressiveWindow,
    preserve_map_preview,
    split_cells_by_edit_support,
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


def test_progressive_planner_overlap_mode_moves_one_tile_and_reuses_center_support() -> None:
    target_cells = [(2, y) for y in range(1, 5)] + [(3, y) for y in range(1, 5)]

    coverage_steps = plan_progressive_steps(
        target_cells=target_cells,
        grid_w=6,
        grid_h=6,
        window_grid_side=4,
        stride_cells=1,
        grid_step_px=256,
        selection_mode="coverage",
    )
    overlap_steps = plan_progressive_steps(
        target_cells=target_cells,
        grid_w=6,
        grid_h=6,
        window_grid_side=4,
        stride_cells=1,
        grid_step_px=256,
        selection_mode="overlap",
    )

    assert coverage_steps[1].window.gy0 == 2
    assert overlap_steps[1].window.gy0 == 1
    assert overlap_steps[1].edit_cells_global == ((2, 3), (3, 3))


def test_split_cells_by_edit_support_separates_unsupported_cells() -> None:
    supported, unsupported = split_cells_by_edit_support(
        target_cells=[(0, 0), (1, 1), (6, 6), (7, 7)],
        grid_w=8,
        grid_h=8,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=256,
        edit_support="center_2x2",
    )

    assert supported == ((1, 1), (6, 6))
    assert unsupported == ((0, 0), (7, 7))


def test_split_cells_by_padded_center_support_covers_region_border_cells() -> None:
    supported, unsupported = split_cells_by_edit_support(
        target_cells=[(0, 0), (1, 1), (6, 6), (7, 7)],
        grid_w=8,
        grid_h=8,
        window_grid_side=4,
        stride_cells=2,
        grid_step_px=256,
        edit_support="padded_center_2x2",
    )

    assert supported == ((0, 0), (1, 1), (6, 6), (7, 7))
    assert unsupported == ()


def test_split_cells_by_full_window_support_covers_all_window_cells_without_sliding() -> None:
    supported, unsupported = split_cells_by_edit_support(
        target_cells=[(0, 0), (3, 3), (5, 4), (7, 7)],
        grid_w=8,
        grid_h=8,
        window_grid_side=4,
        stride_cells=4,
        grid_step_px=256,
        edit_support="full_window",
    )

    assert supported == ((0, 0), (3, 3), (5, 4), (7, 7))
    assert unsupported == ()


def test_draw_cells_overlay_draws_one_clean_shared_boundary_between_selected_cells() -> None:
    base = Image.new("RGB", (96, 48), color=(0, 0, 0))
    overlay = draw_cells_overlay(base, cells=[(0, 0), (1, 0)], grid_step_px=32)

    assert overlay.getpixel((0, 24)) == (255, 0, 0)
    assert overlay.getpixel((64, 24)) == (255, 0, 0)
    assert overlay.getpixel((32, 24)) == (255, 0, 0)
    assert overlay.getpixel((27, 24)) == (0, 0, 0)
    assert overlay.getpixel((37, 24)) == (0, 0, 0)


def test_draw_step_region_overlay_shows_grid_window_and_transparent_edit_cells() -> None:
    base = Image.new("RGB", (192, 192), color=(255, 255, 255))
    window = ProgressiveWindow(
        window_id="r1_c1",
        row_index=1,
        col_index=1,
        gx0=1,
        gy0=1,
        grid_w=4,
        grid_h=4,
        left=32,
        top=32,
    )
    overlay = draw_step_region_overlay(
        base,
        window=window,
        edit_cells_global=[(2, 2), (3, 2)],
        support_cells_global=[(2, 2), (3, 2), (2, 3), (3, 3)],
        grid_step_px=32,
    )

    assert overlay.getpixel((0, 16))[0] < 100
    assert overlay.getpixel((32, 32)) == (255, 0, 0)
    selected_center = overlay.getpixel((80, 80))
    assert selected_center[0] > selected_center[1]
    assert selected_center[1] > 100
    support_center = overlay.getpixel((80, 112))
    assert support_center[0] > support_center[1] > selected_center[1]
    assert overlay.getpixel((96, 80)) == (255, 0, 0)
    context_center = overlay.getpixel((48, 48))
    assert context_center[0] == context_center[1] == context_center[2]
    assert context_center[0] > 245


def test_preserve_map_preview_uses_red_for_editable_region_and_grey_elsewhere() -> None:
    preserve_map = torch.tensor([[0.0, 0.22, 0.84]], dtype=torch.float32)
    preview = preserve_map_preview(preserve_map)

    editable = preview.getpixel((0, 0))
    fresh_context = preview.getpixel((1, 0))
    visited = preview.getpixel((2, 0))
    assert editable[0] > editable[1] == editable[2]
    assert fresh_context[0] == fresh_context[1] == fresh_context[2]
    assert visited[0] == visited[1] == visited[2]
    assert fresh_context[0] < visited[0]


def test_draw_step_edit_area_zoom_crops_full_window() -> None:
    base = Image.new("RGB", (192, 192), color=(255, 255, 255))
    window = ProgressiveWindow(
        window_id="r1_c1",
        row_index=1,
        col_index=1,
        gx0=1,
        gy0=1,
        grid_w=4,
        grid_h=4,
        left=32,
        top=32,
    )
    zoom = draw_step_edit_area_zoom_4x4(
        base,
        window=window,
        edit_cells_global=[(2, 2)],
        support_cells_global=[(2, 2), (3, 2), (2, 3), (3, 3)],
        grid_step_px=32,
    )

    assert zoom.size == (128, 128)
    assert zoom.getpixel((16, 16))[0] == zoom.getpixel((16, 16))[1]
    assert zoom.getpixel((48, 48))[0] > zoom.getpixel((48, 48))[1]
    assert zoom.getpixel((80, 48))[1] > zoom.getpixel((48, 48))[1]
    assert zoom.getpixel((64, 48)) == (255, 0, 0)


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
