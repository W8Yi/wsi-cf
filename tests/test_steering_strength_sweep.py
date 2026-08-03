from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import load_script_module
from wsi_cf.paper.strength_sweep import (
    expand_requests_to_all_region_cells,
    expand_requests_to_random_valid_blocks,
    expand_requests_to_valid_feature_cells,
    monotonicity_record,
    parse_strengths,
    select_random_requests,
    strength_slug,
    summarize_monotonicity,
)


def test_expand_requests_to_all_region_cells_uses_full_2048_grid() -> None:
    requests = [{"run_id": "run_1", "region_id": "region_1", "selector": "attention", "target_cells": [{"gx": 3, "gy": 4}]}]
    regions = {
        "region_1": {
            "region_id": "region_1",
            "region_w": "2048",
            "region_h": "2048",
            "grid_step_px": "256",
        }
    }
    expanded = expand_requests_to_all_region_cells(requests, regions)
    assert len(expanded) == 1
    assert expanded[0]["selector"] == "all_region_cells"
    assert expanded[0]["base_selector"] == "attention"
    assert expanded[0]["valid_cell_count"] == 64
    assert expanded[0]["target_cells"][0] == {"gx": 0, "gy": 0}
    assert expanded[0]["target_cells"][-1] == {"gx": 7, "gy": 7}


def test_expand_requests_to_valid_feature_cells_supports_center_block(tmp_path: Path) -> None:
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[:3, 6:] = 0
    feature_grid_path = tmp_path / "region_zgrid.npy"
    np.save(feature_grid_path, np.zeros((8, 8, 2), dtype=np.float32))
    np.save(tmp_path / "valid_feature_mask.npy", mask)
    requests = [{"run_id": "run_1", "region_id": "region_1", "selector": "attention", "target_cells": [{"gx": 3, "gy": 4}]}]
    regions = {"region_1": {"region_id": "region_1", "feature_grid_path": str(feature_grid_path)}}

    all_valid = expand_requests_to_valid_feature_cells(requests, regions)
    center = expand_requests_to_valid_feature_cells(requests, regions, center_margin_cells=1)

    assert all_valid[0]["selector"] == "all_valid_feature_cells"
    assert len(all_valid[0]["target_cells"]) == 58
    assert center[0]["selector"] == "center_valid_feature_cells"
    assert center[0]["center_margin_cells"] == 1
    assert len(center[0]["target_cells"]) == 34
    assert all(1 <= cell["gx"] <= 6 and 1 <= cell["gy"] <= 6 for cell in center[0]["target_cells"])


def test_expand_requests_to_random_valid_blocks_is_seeded_and_contiguous(tmp_path: Path) -> None:
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[:3, 6:] = 0
    feature_grid_path = tmp_path / "region_zgrid.npy"
    np.save(feature_grid_path, np.zeros((8, 8, 2), dtype=np.float32))
    np.save(tmp_path / "valid_feature_mask.npy", mask)
    requests = [{"run_id": "run_1", "region_id": "region_1", "selector": "attention", "target_cells": [{"gx": 3, "gy": 4}]}]
    regions = {"region_1": {"region_id": "region_1", "feature_grid_path": str(feature_grid_path)}}

    first = expand_requests_to_random_valid_blocks(requests, regions, block_side_cells=4, seed=7)
    second = expand_requests_to_random_valid_blocks(requests, regions, block_side_cells=4, seed=7)

    assert first == second
    assert first[0]["selector"] == "random_valid_block"
    assert first[0]["valid_cell_count"] == 16
    cells = {(cell["gx"], cell["gy"]) for cell in first[0]["target_cells"]}
    assert len({gx for gx, _ in cells}) == 4
    assert len({gy for _, gy in cells}) == 4
    assert all(mask[gy, gx] for gx, gy in cells)


def test_parse_strengths_and_slug() -> None:
    assert parse_strengths("1,0.4,0,0.4") == (0.0, 0.4, 1.0)
    assert strength_slug(0.2) == "strength_020"
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_strengths("0,1.2")


def test_cell_fraction_sweep_uses_nested_deterministic_valid_cells() -> None:
    script = load_script_module("run_cell_fraction_sweep.py")
    full = [{
        "run_id": "run",
        "region_id": "region",
        "target_cells": [{"gx": x, "gy": 0} for x in range(5)],
    }]
    by_fraction = script._fraction_requests(full, (0.0, 0.4, 1.0), seed=7)
    assert by_fraction[0.0][0]["target_cells"] == []
    assert len(by_fraction[0.4][0]["target_cells"]) == 2
    assert by_fraction[1.0][0]["target_cells"][:2] == by_fraction[0.4][0]["target_cells"]
    assert len(by_fraction[1.0][0]["target_cells"]) == 5


def test_seeded_random_region_selection_is_stable() -> None:
    requests = [
        {"run_id": f"run_{index}", "region_id": f"region_{index}", "target_cells": [{"gx": 1, "gy": 1}]}
        for index in range(6)
    ]
    regions = {
        f"region_{index}": {"region_id": f"region_{index}", "region_w": "2048", "region_h": "2048", "label_name": "source"}
        for index in range(6)
    }
    first = select_random_requests(requests, regions, seed=19, n_regions=3, source_label="source")
    second = select_random_requests(list(reversed(requests)), regions, seed=19, n_regions=3, source_label="source")
    assert [row["run_id"] for row in first] == [row["run_id"] for row in second]


def test_monotonicity_reports_reversal() -> None:
    record = monotonicity_record([0.0, 0.2, 0.4, 0.6], [0.1, 0.3, 0.25, 0.8])
    assert record["n_decreases"] == 1
    assert record["strictly_monotonic_nondecreasing"] is False
    assert record["adjacent_nondecreasing_fraction"] == pytest.approx(2.0 / 3.0)

    rows = [
        {
            "task_name": "demo",
            "direction": "a_to_b",
            "region_id": "r1",
            "seed": 7,
            "steering_strength": strength,
            "target_probability": probability,
            "concept_activation": activation,
        }
        for strength, probability, activation in zip(
            (0.0, 0.5, 1.0),
            (0.1, 0.5, 0.9),
            (0.2, 0.4, 0.8),
        )
    ]
    summary = summarize_monotonicity(rows)
    assert {row["metric"] for row in summary} == {"target_probability", "concept_activation"}
    assert all(row["n_decreases"] == 0 for row in summary)


def test_plotter_writes_vector_and_raster_outputs(tmp_path: Path) -> None:
    script = load_script_module("plot_steering_strength_sweep.py")
    rows = []
    for index, strength in enumerate((0.0, 0.5, 1.0)):
        image_path = tmp_path / f"image_{index}.png"
        Image.fromarray(np.full((32, 32, 3), 100 + index * 40, dtype=np.uint8)).save(image_path)
        rows.append(
            {
                "task_name": "demo",
                "direction": "source_to_target",
                "region_id": "region_1",
                "seed": 7,
                "steering_strength": strength,
                "target_probability": 0.1 + 0.7 * strength,
                "concept_activation": 0.2 + 0.6 * strength,
                "concept_score": 0.2 + 0.6 * strength,
                "concept_score_name": "Target SAE activation (z5)",
                "image_path": str(image_path),
            }
        )
    args = argparse.Namespace(
        out_dir=tmp_path / "figures",
        region_id="",
        dpi=80,
        title="Demo",
        prefix="controlled_sweep",
    )
    paths = script.make_figure(rows, args)
    assert {path.suffix for path in paths} == {".png", ".pdf", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_runner_cli_defaults_to_controlled_six_point_sweep(tmp_path: Path) -> None:
    script = load_script_module("run_steering_strength_sweep.py")
    args = script.build_arg_parser().parse_args(
        [
            "--task-name",
            "demo",
            "--direction",
            "source_to_target",
            "--runner-direction",
            "hpv_pos",
            "--target-label",
            "target",
            "--label-order",
            "source,target",
            "--region-bank-csv",
            str(tmp_path / "region_bank.csv"),
            "--base-edit-manifest",
            str(tmp_path / "manifest.json"),
            "--out-dir",
            str(tmp_path / "out"),
            "--classifier-ckpt",
            str(tmp_path / "classifier.pt"),
            "--concept-latent",
            "5",
        ]
    )
    assert parse_strengths(args.strengths) == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert args.region_size == 2048
    assert args.seed == 7
    assert args.request_selector == ""
    assert args.request_budget == 0
    assert args.all_region_cells is False
    assert args.all_valid_feature_cells is False
    assert args.center_valid_feature_cells is False
    assert args.random_valid_block_side_cells == 0
    assert args.center_margin_cells == 1
    assert args.probability_source == "selected_cells"
