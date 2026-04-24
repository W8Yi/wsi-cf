from __future__ import annotations

from pathlib import Path

from conftest import load_script_module


def test_export_region_bank_10x_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("export_hnscc_region_bank_10x.py")
    args = script.build_arg_parser().parse_args(
        [
            "--split-tsv",
            str(tmp_path / "split.tsv"),
            "--out-dir",
            str(tmp_path / "region_bank_10x"),
            "--target-magnification",
            "10",
            "--regions-total",
            "8",
        ]
    )
    assert str(args.out_dir).endswith("region_bank_10x")
    assert float(args.target_magnification) == 10.0
    assert int(args.regions_total) == 8


def test_region_bank_10x_sae_cases_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_region_bank_10x_sae_cases.py")
    args = script.build_arg_parser().parse_args(
        [
            "--region-bank-csv",
            str(tmp_path / "region_bank.csv"),
            "--out-dir",
            str(tmp_path / "tenx_cases"),
            "--cases",
            "random_two,block_2x2",
            "--manual-cell",
            "1,1",
            "--manual-cell",
            "2,2",
            "--preserve-outside-latents",
            "--preserve-outside-strength",
            "0.95",
            "--preserve-edit-strength",
            "0.5",
        ]
    )
    assert str(args.out_dir).endswith("tenx_cases")
    assert args.cases == "random_two,block_2x2"
    assert args.manual_cell == ["1,1", "2,2"]
    assert args.preserve_outside_latents is True
    assert float(args.preserve_outside_strength) == 0.95
    assert float(args.preserve_edit_strength) == 0.5


def test_region_bank_10x_sae_cases_progressive_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_region_bank_10x_sae_cases.py")
    args = script.build_arg_parser().parse_args(
        [
            "--region-bank-csv",
            str(tmp_path / "region_bank.csv"),
            "--out-dir",
            str(tmp_path / "progressive"),
            "--progressive-mode",
            "overlap_truth_right_strip",
            "--progressive-commit-mode",
            "full_window_latest_wins",
            "--window-size",
            "1024",
            "--window-stride",
            "512",
            "--max-span",
            "3",
            "--edit-shapes",
            "center_one,center_2x2",
            "--target-row-index",
            "1",
            "--start-col-index",
            "0",
            "--save-progress-steps",
        ]
    )
    assert args.progressive_mode == "overlap_truth_right_strip"
    assert args.progressive_commit_mode == "full_window_latest_wins"
    assert int(args.window_size) == 1024
    assert int(args.window_stride) == 512
    assert int(args.max_span) == 3
    assert args.edit_shapes == "center_one,center_2x2"
    assert int(args.target_row_index) == 1
    assert int(args.start_col_index) == 0
    assert args.save_progress_steps is True


def test_progressive_region_edit_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_progressive_region_edit.py")
    args = script.build_arg_parser().parse_args(
        [
            "--region-bank-csv",
            str(tmp_path / "region_bank.csv"),
            "--edit-manifest",
            str(tmp_path / "edit_manifest.json"),
            "--out-dir",
            str(tmp_path / "progressive_edit"),
            "--direction",
            "hpv_pos",
            "--preserve-edit-strength",
            "0.5",
            "--preserve-visited-strength",
            "0.95",
            "--preserve-fresh-context-strength",
            "0.35",
            "--output-mode",
            "debug",
        ]
    )
    assert str(args.out_dir).endswith("progressive_edit")
    assert str(args.edit_manifest).endswith("edit_manifest.json")
    assert float(args.preserve_edit_strength) == 0.5
    assert float(args.preserve_visited_strength) == 0.95
    assert float(args.preserve_fresh_context_strength) == 0.35
    assert args.output_mode == "debug"


def test_region_attention_classifier_eval_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_region_attention_classifier_eval.py")
    args = script.build_arg_parser().parse_args(
        [
            "--region-bank-csv",
            str(tmp_path / "region_bank.csv"),
            "--out-dir",
            str(tmp_path / "region_attention_eval"),
            "--selection-mode",
            "percentile",
            "--attention-percentile",
            "85",
            "--direction-mode",
            "opposite_label",
            "--max-cells",
            "3",
        ]
    )
    assert str(args.out_dir).endswith("region_attention_eval")
    assert args.experiment_mode == "comprehensive"
    assert args.selection_mode == "percentile"
    assert float(args.attention_percentile) == 85.0
    assert args.direction_mode == "opposite_label"
    assert int(args.max_cells) == 3
    assert args.editor_mode == "progressive"
    assert args.prototype_strengths == "0.4,0.8,1.2"
    assert bool(args.require_label_match) is True


def test_region_attention_classifier_eval_comprehensive_defaults(tmp_path: Path) -> None:
    script = load_script_module("run_region_attention_classifier_eval.py")
    args = script.build_arg_parser().parse_args(
        [
            "--region-bank-csv",
            str(tmp_path / "region_bank.csv"),
            "--out-dir",
            str(tmp_path / "region_attention_eval"),
        ]
    )
    assert args.experiment_mode == "comprehensive"
    assert int(args.final_regions_total) == 20
    assert int(args.final_regions_per_label) == 10
    assert args.selection_mode == "attention_mass"
    assert float(args.target_attention_mass) == 0.35
    assert int(args.min_high_attention_cells) == 2
    assert int(args.max_high_attention_cells) == 6
    assert float(args.min_label_confidence) == 0.75
    assert args.prototype_strengths == "0.4,0.8,1.2"


def test_region_image_tile_selector_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_region_image_tile_selector.py")
    args = script.build_arg_parser().parse_args(
        [
            "--region-image",
            str(tmp_path / "region.png"),
            "--out-dir",
            str(tmp_path / "region_selector"),
            "--label",
            "1",
            "--model-backend",
            "both",
            "--target-importance-mass",
            "0.5",
            "--max-expanded-cells",
            "32",
            "--fill-gaps",
        ]
    )
    assert str(args.region_image).endswith("region.png")
    assert str(args.out_dir).endswith("region_selector")
    assert int(args.label) == 1
    assert args.model_backend == "both"
    assert float(args.target_importance_mass) == 0.5
    assert int(args.max_expanded_cells) == 32
    assert args.fill_gaps is True


def test_region_image_tile_selector_exploration_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_region_image_tile_selector_exploration.py")
    args = script.build_arg_parser().parse_args(
        [
            "--region-image",
            str(tmp_path / "region.png"),
            "--out-dir",
            str(tmp_path / "region_explore"),
            "--label",
            "1",
            "--attention-percentiles",
            "80,90,95",
            "--expansion-methods",
            "seed_only,full_stack",
        ]
    )
    assert str(args.region_image).endswith("region.png")
    assert str(args.out_dir).endswith("region_explore")
    assert int(args.label) == 1
    assert args.attention_percentiles == "80,90,95"
    assert args.expansion_methods == "seed_only,full_stack"
