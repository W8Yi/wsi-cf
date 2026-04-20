from __future__ import annotations

import numpy as np
from PIL import Image

from wsi_cf.data.attention_proposals import (
    build_region_candidate,
    compute_attention_threshold,
    map_center_to_region_cell,
    select_attention_region,
)
from wsi_cf.data.slides import quick_region_quality_metrics
from wsi_cf.eval.local_region import replace_selected_cells_in_local_bag


def make_grid_coords(*, nx: int, ny: int, step: int = 256) -> np.ndarray:
    rows = []
    for gy in range(int(ny)):
        for gx in range(int(nx)):
            rows.append((gx * int(step), gy * int(step)))
    return np.asarray(rows, dtype=np.int64)


def test_map_center_to_region_cell_maps_expected_quadrants() -> None:
    gx, gy = map_center_to_region_cell(
        center_x=640.0,
        center_y=640.0,
        region_x=0,
        region_y=0,
        crop_w_level0=2048,
        crop_h_level0=2048,
        grid_side=4,
    )
    assert (gx, gy) == (1, 1)


def test_build_region_candidate_collects_selected_cells() -> None:
    coords = make_grid_coords(nx=8, ny=8, step=256)
    attention = np.zeros((coords.shape[0],), dtype=np.float32)
    high_tiles = [27, 28, 35, 36]
    for idx in high_tiles:
        attention[idx] = 1.0
    candidate = build_region_candidate(
        slide_key="SLIDE1",
        case_id="CASE1",
        label=1,
        coords=coords,
        attention=attention,
        anchor_tile_index=27,
        anchor_rank=1,
        high_attention_threshold=0.5,
        slide_w=4096,
        slide_h=4096,
        crop_w_level0=2048,
        crop_h_level0=2048,
        tile_size_level0=256,
        grid_side=4,
    )
    assert candidate["anchor_tile_index"] == 27
    assert candidate["selected_cell_count"] >= 1
    assert all(len(cell) == 2 for cell in candidate["selected_cells"])


def test_select_attention_region_prefers_candidate_with_half_or_less_cells() -> None:
    coords = make_grid_coords(nx=12, ny=12, step=256)
    attention = np.zeros((coords.shape[0],), dtype=np.float32)
    block_many = [
        gy * 12 + gx
        for gy in range(2, 10)
        for gx in range(2, 10)
        if (gx + gy) % 2 == 0
    ]
    block_few = [8 * 12 + 8, 8 * 12 + 9, 9 * 12 + 8, 9 * 12 + 9]
    for idx in block_many:
        attention[idx] = 0.95
    for idx in block_few:
        attention[idx] = 0.90
    attention[5 * 12 + 5] = 1.0
    proposal = select_attention_region(
        slide_key="SLIDE1",
        case_id="CASE1",
        label=1,
        coords=coords,
        attention=attention,
        slide_w=4096,
        slide_h=4096,
        crop_w_level0=2048,
        crop_h_level0=2048,
        tile_size_level0=256,
        grid_side=4,
        attention_percentile=90.0,
        min_high_attention_cells=1,
        max_high_attention_cells=8,
        candidate_anchor_limit=32,
    )
    assert proposal["selected_cell_count"] <= 8
    assert proposal["selection_fallback"] is False


def test_select_attention_region_falls_back_when_no_candidate_qualifies() -> None:
    coords = make_grid_coords(nx=12, ny=12, step=256)
    attention = np.ones((coords.shape[0],), dtype=np.float32)
    proposal = select_attention_region(
        slide_key="SLIDE1",
        case_id="CASE1",
        label=1,
        coords=coords,
        attention=attention,
        slide_w=4096,
        slide_h=4096,
        crop_w_level0=2048,
        crop_h_level0=2048,
        tile_size_level0=256,
        grid_side=4,
        attention_percentile=90.0,
        min_high_attention_cells=1,
        max_high_attention_cells=8,
        candidate_anchor_limit=8,
    )
    assert proposal["selection_fallback"] is True
    assert proposal["selected_cell_count"] >= 8


def test_replace_selected_cells_in_local_bag_only_changes_target_cells() -> None:
    features_local = np.zeros((4, 3), dtype=np.float32)
    tile_rows = [
        {"tile_index": 0, "coord_x": 0, "coord_y": 0, "cell_gx": 0, "cell_gy": 0},
        {"tile_index": 1, "coord_x": 256, "coord_y": 0, "cell_gx": 1, "cell_gy": 0},
        {"tile_index": 2, "coord_x": 0, "coord_y": 256, "cell_gx": 0, "cell_gy": 1},
        {"tile_index": 3, "coord_x": 256, "coord_y": 256, "cell_gx": 1, "cell_gy": 1},
    ]
    replacement_grid = np.zeros((4, 4, 3), dtype=np.float32)
    replacement_grid[1, 1] = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    out, manifest = replace_selected_cells_in_local_bag(
        features_local=features_local,
        tile_rows=tile_rows,
        replacement_grid=replacement_grid,
        selected_cells=[(1, 1)],
    )
    assert np.allclose(out[0], np.zeros((3,), dtype=np.float32))
    assert np.allclose(out[3], np.array([7.0, 8.0, 9.0], dtype=np.float32))
    assert sum(1 for row in manifest if row["replaced"]) == 1


def test_compute_attention_threshold_uses_percentile() -> None:
    attention = np.asarray([0.0, 0.1, 0.2, 0.5, 1.0], dtype=np.float32)
    threshold = compute_attention_threshold(attention, percentile=80.0)
    assert 0.2 < threshold <= 1.0


def test_quick_region_quality_metrics_penalizes_blank_regions() -> None:
    dense = np.zeros((64, 64, 3), dtype=np.uint8)
    dense[..., 0] = 140
    dense[..., 1] = 80
    dense[..., 2] = 170
    blank = np.full((64, 64, 3), 245, dtype=np.uint8)
    dense_metrics = quick_region_quality_metrics(Image.fromarray(dense))
    blank_metrics = quick_region_quality_metrics(Image.fromarray(blank))
    assert dense_metrics["tissue_score"] > blank_metrics["tissue_score"]
    assert dense_metrics["dark_fraction"] > blank_metrics["dark_fraction"]
    assert dense_metrics["saturation_fraction"] > blank_metrics["saturation_fraction"]
