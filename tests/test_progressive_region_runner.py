from __future__ import annotations

import numpy as np
from PIL import Image

from conftest import load_script_module


def test_infer_grid_shape_supports_8x8() -> None:
    script = load_script_module("run_region_bank_10x_sae_cases.py")
    z_grid = np.zeros((8, 8, 1536), dtype=np.float32)
    assert script.infer_grid_shape(z_grid) == (8, 8)


def test_enumerate_progressive_windows_for_2048_stride_512() -> None:
    script = load_script_module("run_region_bank_10x_sae_cases.py")
    windows = script.enumerate_progressive_windows(
        region_w_px=2048,
        region_h_px=2048,
        window_size=1024,
        window_stride=512,
        grid_step_px=256,
    )
    assert len(windows) == 9
    assert windows[0]["left"] == 0
    assert windows[1]["left"] == 512
    assert windows[2]["left"] == 1024
    assert windows[0]["gx0"] == 0
    assert windows[1]["gx0"] == 2
    assert windows[2]["gx0"] == 4


def test_progressive_shape_cells_returns_expected_center_sets() -> None:
    script = load_script_module("run_region_bank_10x_sae_cases.py")
    assert script.progressive_shape_cells("center_one") == [(1, 1)]
    assert script.progressive_shape_cells("center_two_h") == [(1, 1), (2, 1)]
    assert script.progressive_shape_cells("center_2x2") == [(1, 1), (2, 1), (1, 2), (2, 2)]


def test_update_full_zgrid_selected_cells_only_overwrites_selected() -> None:
    script = load_script_module("run_region_bank_10x_sae_cases.py")
    full = np.zeros((8, 8, 2), dtype=np.float32)
    edited_local = np.ones((4, 4, 2), dtype=np.float32)
    out = script.update_full_zgrid_selected_cells(
        full_zgrid=full,
        edited_local_zgrid=edited_local,
        gx0=2,
        gy0=4,
        selected_cells=[(1, 1), (2, 1)],
    )
    assert np.allclose(out[5, 3], 1.0)
    assert np.allclose(out[5, 4], 1.0)
    assert np.allclose(out[4, 2], 0.0)
    assert np.allclose(out[7, 7], 0.0)


def test_build_commit_alpha_mask_support_cells_with_soft_halo() -> None:
    script = load_script_module("run_progressive_region_edit.py")

    alpha = script.build_commit_alpha_mask(
        width=128,
        height=128,
        grid_step_px=32,
        cells_local=[(1, 1), (2, 1), (1, 2), (2, 2)],
        feather_px=0,
        halo_cells=1,
        halo_alpha=0.35,
    )

    assert alpha.shape == (128, 128, 1)
    assert np.isclose(alpha[48, 48, 0], 1.0)
    assert np.isclose(alpha[80, 80, 0], 1.0)
    assert np.isclose(alpha[16, 16, 0], 0.35)
    assert np.isclose(alpha[112, 112, 0], 0.35)


def test_composite_invalid_feature_cells_restores_blank_tiles() -> None:
    script = load_script_module("run_progressive_region_edit.py")
    source = Image.fromarray(np.full((64, 64, 3), 245, dtype=np.uint8))
    generated = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
    valid_mask = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)

    composited, alpha = script.composite_invalid_feature_cells(
        generated_img=generated,
        source_img=source,
        valid_feature_mask=valid_mask,
        grid_step_px=32,
        feather_px=0,
    )
    out = np.asarray(composited)

    assert alpha.shape == (64, 64, 1)
    assert np.all(out[16, 16] == 120)
    assert np.all(out[16, 48] == 245)
    assert np.all(out[48, 48] == 120)


def test_valid_feature_alpha_mask_feathers_tile_boundary() -> None:
    script = load_script_module("run_progressive_region_edit.py")
    valid_mask = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)

    alpha = script.build_valid_feature_alpha_mask(
        valid_feature_mask=valid_mask,
        width=64,
        height=64,
        grid_step_px=32,
        feather_px=4,
    )

    assert alpha[16, 16, 0] > 0.99
    assert 0.0 < alpha[16, 31, 0] < 1.0
    assert 0.0 < alpha[16, 32, 0] < 1.0
    assert alpha[16, 48, 0] < 0.01


def test_commit_progressive_window_update_uses_full_window_latest_wins() -> None:
    script = load_script_module("run_region_bank_10x_sae_cases.py")
    canvas = np.zeros((2048, 2048, 3), dtype=np.float32)
    first_img = Image.fromarray(np.full((1024, 1024, 3), 255, dtype=np.uint8))
    second_img = Image.fromarray(np.full((1024, 1024, 3), 128, dtype=np.uint8))

    out1, box1, mode1 = script.commit_progressive_window_update(
        current_canvas=canvas,
        steered_img=first_img,
        left=0,
        top=512,
        window_size=1024,
        window_stride=512,
        step_index=0,
        commit_mode="full_window_latest_wins",
    )
    assert mode1 == "full_window"
    assert box1 == (0, 512, 1024, 1536)
    assert np.allclose(out1[700, 100], 1.0)

    out2, box2, mode2 = script.commit_progressive_window_update(
        current_canvas=out1,
        steered_img=second_img,
        left=512,
        top=512,
        window_size=1024,
        window_stride=512,
        step_index=1,
        commit_mode="full_window_latest_wins",
    )
    assert mode2 == "full_window_latest_wins"
    assert box2 == (512, 512, 1536, 1536)
    assert np.allclose(out2[700, 900], 128.0 / 255.0)
    assert np.allclose(out2[700, 1200], 128.0 / 255.0)


def test_commit_progressive_window_update_supports_new_right_band_only() -> None:
    script = load_script_module("run_region_bank_10x_sae_cases.py")
    canvas = np.zeros((2048, 2048, 3), dtype=np.float32)
    first_img = Image.fromarray(np.full((1024, 1024, 3), 255, dtype=np.uint8))
    second_img = Image.fromarray(np.full((1024, 1024, 3), 128, dtype=np.uint8))

    out1, _, _ = script.commit_progressive_window_update(
        current_canvas=canvas,
        steered_img=first_img,
        left=0,
        top=512,
        window_size=1024,
        window_stride=512,
        step_index=0,
        commit_mode="new_right_band_only",
    )
    out2, box2, mode2 = script.commit_progressive_window_update(
        current_canvas=out1,
        steered_img=second_img,
        left=512,
        top=512,
        window_size=1024,
        window_stride=512,
        step_index=1,
        commit_mode="new_right_band_only",
    )
    assert mode2 == "new_right_band_only"
    assert box2 == (1024, 512, 1536, 1536)
    assert np.allclose(out2[700, 900], 1.0)
    assert np.allclose(out2[700, 1200], 128.0 / 255.0)
