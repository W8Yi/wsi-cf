from __future__ import annotations

import torch

from wsi_cf.generation.pixcell import (
    compute_condition_blend_for_step,
    select_condition_grid_for_step,
    validate_condition_schedule_ratios,
)


def test_validate_condition_schedule_ratios_accepts_valid_bounds() -> None:
    start, end = validate_condition_schedule_ratios(start_ratio=0.5, end_ratio=1.0)
    assert start == 0.5
    assert end == 1.0


def test_validate_condition_schedule_ratios_rejects_invalid_bounds() -> None:
    for start, end in [(-0.1, 1.0), (0.0, 1.1), (0.8, 0.7)]:
        try:
            validate_condition_schedule_ratios(start_ratio=start, end_ratio=end)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid schedule ratios start={start} end={end}")


def test_select_condition_grid_for_step_uses_base_then_scheduled_then_base() -> None:
    base = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    scheduled = torch.ones((1, 4, 4, 3), dtype=torch.float32)

    out0 = select_condition_grid_for_step(
        base_z_grid=base,
        scheduled_z_grid=scheduled,
        step_idx=0,
        num_steps=5,
        start_ratio=0.5,
        end_ratio=0.75,
        alpha_start=0.0,
        alpha_end=1.0,
        schedule="linear",
    )
    out2 = select_condition_grid_for_step(
        base_z_grid=base,
        scheduled_z_grid=scheduled,
        step_idx=2,
        num_steps=5,
        start_ratio=0.5,
        end_ratio=0.75,
        alpha_start=0.0,
        alpha_end=1.0,
        schedule="linear",
    )
    out4 = select_condition_grid_for_step(
        base_z_grid=base,
        scheduled_z_grid=scheduled,
        step_idx=4,
        num_steps=5,
        start_ratio=0.5,
        end_ratio=0.75,
        alpha_start=0.0,
        alpha_end=1.0,
        schedule="linear",
    )

    assert torch.equal(out0, base)
    assert torch.equal(out2, scheduled)
    assert torch.equal(out4, scheduled)


def test_compute_condition_blend_for_step_ramps_low_to_high() -> None:
    alphas = [
        compute_condition_blend_for_step(
            step_idx=idx,
            num_steps=5,
            start_ratio=0.0,
            end_ratio=1.0,
            alpha_start=0.2,
            alpha_end=1.0,
            schedule="linear",
        )
        for idx in range(5)
    ]
    assert alphas[0] == 0.2
    assert round(alphas[-1], 6) == 1.0
    assert alphas[0] < alphas[1] < alphas[2] < alphas[3] < alphas[4]
