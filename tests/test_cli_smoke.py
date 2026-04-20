from __future__ import annotations

from pathlib import Path

from conftest import load_script_module


def test_cli_parsers_load_without_running_heavy_work(tmp_path: Path) -> None:
    export_feature = load_script_module("export_h5_tile_feature.py")
    args = export_feature.build_arg_parser().parse_args(
        ["--h5", str(tmp_path / "in.h5"), "--out", str(tmp_path / "out.npy")]
    )
    assert str(args.h5).endswith("in.h5")

    multidiff = load_script_module("multidiff_img2img_wsi.py")
    args2 = multidiff.build_arg_parser().parse_args(
        [
            "--input_image",
            str(tmp_path / "img.png"),
            "--out_dir",
            str(tmp_path / "out_dir"),
        ]
    )
    assert str(args2.out_dir).endswith("out_dir")


def test_naive_manifest_cli_generates_outputs(tmp_path: Path) -> None:
    script = load_script_module("build_pixcell_1024_naive_manifest.py")
    csv_path = tmp_path / "tile_pool.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,label,hpv_status,case_id,slide_key,tile_index,coord_x,coord_y,feature_path,image_path",
            ]
            + [
                f"test,1,positive,CASE{i},SLIDE{i},{i},{i * 10},{i * 10},{tmp_path / f'f{i}.npy'},{tmp_path / f'i{i}.png'}"
                for i in range(16)
            ]
        )
    )
    for i in range(16):
        (tmp_path / f"f{i}.npy").write_bytes(b"x")
    out_json = tmp_path / "manifest.json"
    script.main(["--tile-pool-csv", str(csv_path), "--out-json", str(out_json)])
    assert out_json.exists()
    assert out_json.with_suffix(".csv").exists()


def test_region_bank_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("export_hnscc_region_bank.py")
    args = script.build_arg_parser().parse_args(
        [
            "--split-tsv",
            str(tmp_path / "split.tsv"),
            "--out-dir",
            str(tmp_path / "region_bank"),
        ]
    )
    assert str(args.out_dir).endswith("region_bank")


def test_region_bank_runner_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_region_bank_experiments.py")
    args = script.build_arg_parser().parse_args(
        [
            "--region-roles-csv",
            str(tmp_path / "region_roles.csv"),
            "--out-dir",
            str(tmp_path / "runs"),
            "--mid-steer-start-ratio",
            "0.5",
            "--mid-steer-alpha-end",
            "0.9",
        ]
    )
    assert str(args.out_dir).endswith("runs")
    assert float(args.mid_steer_start_ratio) == 0.5
    assert float(args.mid_steer_alpha_end) == 0.9


def test_region_bank_sae_runner_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_region_bank_sae_experiments.py")
    args = script.build_arg_parser().parse_args(
        [
            "--region-roles-csv",
            str(tmp_path / "region_roles.csv"),
            "--out-dir",
            str(tmp_path / "sae_runs"),
            "--mid-steer-end-ratio",
            "0.75",
            "--mid-steer-alpha-start",
            "0.2",
            "--steer-cell",
            "1,1",
            "--steer-cell",
            "2,1",
        ]
    )
    assert str(args.out_dir).endswith("sae_runs")
    assert float(args.mid_steer_end_ratio) == 0.75
    assert float(args.mid_steer_alpha_start) == 0.2
    assert args.steer_cell == ["1,1", "2,1"]


def test_region_bank_sae_selected_cells_manifest_builds() -> None:
    script = load_script_module("run_region_bank_sae_experiments.py")
    rows = [
        {
            "region_id": "R0",
            "label": 1,
            "slide_key": "S0",
            "role_rank": 0,
            "image_path": "/tmp/r0.png",
            "feature_grid_path": "/tmp/r0.npy",
        }
    ]
    manifest = script.build_sae_manifest(
        rows,
        conditions=["baseline", "to_hpv_pos_selected_cells"],
        steer_gx=1,
        steer_gy=1,
        selected_cells=[(0, 0), (1, 1), (2, 1)],
        pos_latent=2645,
        neg_latent=7036,
    )
    assert len(manifest) == 2
    selected = manifest[1]
    assert selected["steer_mode"] == "selected_cells"
    assert selected["steer_cells"] == "0,0;1,1;2,1"
    assert selected["steer_cell_count"] == 3


def test_oneoff_10x_sae_runner_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_sae_10x_selected_cells_test.py")
    args = script.build_arg_parser().parse_args(
        [
            "--input-svs",
            str(tmp_path / "slide.svs"),
            "--out-dir",
            str(tmp_path / "tenx"),
            "--selection-mode",
            "block",
            "--anchor-gx",
            "1",
            "--anchor-gy",
            "0",
            "--block-w",
            "2",
            "--block-h",
            "3",
        ]
    )
    assert str(args.out_dir).endswith("tenx")
    assert args.selection_mode == "block"


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


def test_export_10x_concept_regions_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("export_10x_concept_regions_from_20x_representatives.py")
    args = script.build_arg_parser().parse_args(
        [
            "--prototype-json",
            str(tmp_path / "prototype.json"),
            "--tiles-csv",
            str(tmp_path / "top_tiles.csv"),
            "--out-dir",
            str(tmp_path / "concept_bank_10x"),
            "--selected-direction",
            "hpv_pos",
            "--examples-per-latent",
            "2",
        ]
    )
    assert str(args.out_dir).endswith("concept_bank_10x")
    assert str(args.selected_direction) == "hpv_pos"
    assert int(args.examples_per_latent) == 2


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
        ]
    )
    assert str(args.out_dir).endswith("tenx_cases")
    assert args.cases == "random_two,block_2x2"
    assert args.manual_cell == ["1,1", "2,2"]
    assert args.preserve_outside_latents is True
    assert float(args.preserve_outside_strength) == 0.95


def test_slide_seam_stress_4096_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_slide_seam_stress_test_4096.py")
    args = script.build_arg_parser().parse_args(
        [
            "--input-svs",
            str(tmp_path / "slide.svs"),
            "--x",
            "1024",
            "--y",
            "2048",
            "--patterns",
            "edge_2cell,corner_L",
            "--out-dir",
            str(tmp_path / "seam_stress"),
        ]
    )
    assert str(args.input_svs).endswith("slide.svs")
    assert int(args.x) == 1024
    assert str(args.patterns) == "edge_2cell,corner_L"


def test_seam_repair_pixcell256_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_seam_repair_pixcell256.py")
    args = script.build_arg_parser().parse_args(
        [
            "--stress-run-dir",
            str(tmp_path / "stress"),
            "--stitch-mode",
            "center_weighted_blend",
            "--repair-method",
            "seam_inpaint",
            "--repair-scope",
            "edited_boundaries",
            "--feature-source",
            "blend",
            "--feature-blend-alpha",
            "0.25",
        ]
    )
    assert str(args.stress_run_dir).endswith("stress")
    assert str(args.stitch_mode) == "center_weighted_blend"
    assert str(args.repair_method) == "seam_inpaint"
    assert str(args.repair_scope) == "edited_boundaries"
    assert float(args.feature_blend_alpha) == 0.25


def test_attention_guided_local_1024_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("run_attention_guided_local_1024.py")
    args = script.build_arg_parser().parse_args(
        [
            "--split-json",
            str(tmp_path / "split.json"),
            "--split-tsv",
            str(tmp_path / "split.tsv"),
            "--out-dir",
            str(tmp_path / "attention_local"),
            "--direction",
            "hpv_neg",
            "--attention-percentile",
            "85",
            "--max-high-attention-cells",
            "8",
        ]
    )
    assert str(args.out_dir).endswith("attention_local")
    assert args.direction == "hpv_neg"
    assert float(args.attention_percentile) == 85.0
    assert int(args.max_high_attention_cells) == 8


def test_export_attention_guided_region_bank_10x_cli_parser_loads(tmp_path: Path) -> None:
    script = load_script_module("export_attention_guided_region_bank_10x.py")
    args = script.build_arg_parser().parse_args(
        [
            "--split-json",
            str(tmp_path / "split.json"),
            "--split-tsv",
            str(tmp_path / "split.tsv"),
            "--out-dir",
            str(tmp_path / "attn_bank_10x"),
            "--attention-percentile",
            "92",
            "--min-high-attention-cells",
            "1",
            "--max-high-attention-cells",
            "10",
        ]
    )
    assert str(args.out_dir).endswith("attn_bank_10x")
    assert float(args.attention_percentile) == 92.0
    assert int(args.min_high_attention_cells) == 1
    assert int(args.max_high_attention_cells) == 10
