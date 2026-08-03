from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from conftest import load_script_module


script = load_script_module("compare_edit_visual_perturbation.py")
stream_script = load_script_module("run_streamed_visual_perturbation_baseline.py")


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def test_summarize_diff_uses_mean_abs_rgb_per_pixel() -> None:
    source = np.zeros((2, 2, 3), dtype=np.float32)
    generated = np.zeros((2, 2, 3), dtype=np.float32)
    generated[0, 0] = [30, 60, 90]
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True

    out = script.summarize_diff(source, generated, mask)

    assert out["pixel_count"] == 1
    assert out["mean_abs_rgb"] == 60.0
    assert out["median_abs_rgb"] == 60.0
    assert math.isclose(out["rmse_rgb"], math.sqrt((30**2 + 60**2 + 90**2) / 3.0), rel_tol=1e-6)


def test_cell_mask_marks_selected_grid_cells() -> None:
    mask = script.cell_mask((4, 4, 3), [{"gx": 1, "gy": 0}, {"gx": 0, "gy": 1}], 2)

    expected = np.array(
        [
            [False, False, True, True],
            [False, False, True, True],
            [True, True, False, False],
            [True, True, False, False],
        ],
        dtype=bool,
    )
    assert np.array_equal(mask, expected)


def test_window_history_ignores_integer_diffusion_steps() -> None:
    run_manifest = {
        "steps": 30,
        "window_history": [{"commit_bounds_global": {"x0": 0, "y0": 0, "x1": 2, "y1": 2}}],
    }

    history = script.window_history_from_manifest(run_manifest)
    mask = script.commit_mask_from_steps((4, 4, 3), history)

    assert history == [{"commit_bounds_global": {"x0": 0, "y0": 0, "x1": 2, "y1": 2}}]
    assert int(mask.sum()) == 4


def test_tile_border_seam_excess_detects_generated_boundary() -> None:
    source = np.zeros((4, 4, 3), dtype=np.float32)
    generated = np.zeros((4, 4, 3), dtype=np.float32)
    generated[:, 2:, :] = 30.0

    rows = script.seam_rows_for_method(
        run_id="run_001",
        method="bad_naive",
        source=source,
        generated=generated,
        grid_step_px=2,
        selected_cells={(0, 0)},
        edited_cells={(0, 0), (1, 0)},
        visited_cells={(0, 0), (1, 0), (0, 1), (1, 1)},
        steps=[{"commit_bounds_global": {"x0": 0, "y0": 0, "x1": 2, "y1": 4}}],
        run_meta={"task_name": "toy"},
    )

    by_area = {row["seam_area"]: row for row in rows}
    assert by_area["all_tile_boundaries"]["seam_excess_mean_abs_rgb"] > 0
    assert by_area["selected_cell_perimeter"]["seam_excess_mean_abs_rgb"] > 0
    assert by_area["committed_window_edges"]["seam_excess_mean_abs_rgb"] > 0


def test_compare_writes_paired_visual_metrics(tmp_path: Path) -> None:
    ours_root = tmp_path / "ours"
    naive_root = tmp_path / "naive"
    run_id = "run_001"
    for root, value in [(ours_root, 10), (naive_root, 30)]:
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        save_rgb(run_dir / "source_region_actual.png", np.zeros((4, 4, 3), dtype=np.uint8))
        generated = np.zeros((4, 4, 3), dtype=np.uint8)
        generated[:2, :2] = value
        save_rgb(run_dir / "generated.png", generated)
    (ours_root / run_id / "run_manifest.json").write_text(
        json.dumps(
            {
                "grid_step_px": 2,
                "target_cells": [{"gx": 0, "gy": 0}],
                "steps": [{"commit_bounds_global": {"x0": 0, "y0": 0, "x1": 2, "y1": 2}}],
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "run_id": run_id,
                    "task_name": "toy",
                    "direction": "a_to_b",
                    "selector": "attention",
                    "budget": 1,
                    "target_cells": [{"gx": 0, "gy": 0}],
                }
            ]
        )
    )

    args = script.build_arg_parser().parse_args(
        [
            "--ours-root",
            str(ours_root),
            "--naive-root",
            str(naive_root),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(tmp_path / "metrics"),
            "--formats",
            "png",
        ]
    )
    payload = script.compare(args)

    assert payload["n_paired_runs"] == 1
    paired = (tmp_path / "metrics" / "visual_perturbation_paired_by_run.csv").read_text()
    assert "naive_minus_ours_mean_abs_rgb" in paired
    assert "20.0" in paired
    seam = (tmp_path / "metrics" / "border_inconsistency_paired_summary.csv").read_text()
    assert "naive_minus_ours_seam_excess" in seam


def test_compare_can_skip_per_cell_output(tmp_path: Path) -> None:
    ours_root = tmp_path / "ours"
    naive_root = tmp_path / "naive"
    run_id = "run_001"
    for root, value in [(ours_root, 5), (naive_root, 7)]:
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        save_rgb(run_dir / "source_region_actual.png", np.zeros((2, 2, 3), dtype=np.uint8))
        save_rgb(run_dir / "generated.png", np.full((2, 2, 3), value, dtype=np.uint8))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"run_id": run_id, "target_cells": [{"gx": 0, "gy": 0}]}]))

    args = script.build_arg_parser().parse_args(
        [
            "--ours-root",
            str(ours_root),
            "--naive-root",
            str(naive_root),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(tmp_path / "metrics"),
            "--no-per-cell",
        ]
    )
    script.compare(args)

    assert not (tmp_path / "metrics" / "visual_perturbation_per_cell.csv").exists()
    assert (tmp_path / "metrics" / "visual_perturbation_by_run.csv").exists()


def test_streaming_helpers_filter_and_chunk_requests(tmp_path: Path) -> None:
    ours_root = tmp_path / "ours"
    for run_id in ["run_a", "run_c"]:
        run_dir = ours_root / run_id
        run_dir.mkdir(parents=True)
        save_rgb(run_dir / "generated.png", np.zeros((2, 2, 3), dtype=np.uint8))
    requests = [{"run_id": "run_a"}, {"run_id": "run_b"}, {"run_id": "run_c"}]

    kept, missing = stream_script.eligible_requests(
        requests,
        ours_root=ours_root,
        generated_image_name="generated.png",
        max_runs=0,
        allow_missing_ours=False,
    )

    assert [row["run_id"] for row in kept] == ["run_a", "run_c"]
    assert missing == ["run_b"]
    assert [[row["run_id"] for row in chunk] for chunk in stream_script.chunks(kept, 1)] == [["run_a"], ["run_c"]]
