from __future__ import annotations

import csv
import json
from pathlib import Path

from conftest import load_script_module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def make_concept_dir(tmp_path: Path) -> Path:
    concept_dir = tmp_path / "tumor_purity_low_high" / "high"
    concepts = [
        {
            "concept_rank": 1,
            "task": "tumor_purity_low_high",
            "class_label": "high",
            "latent_idx": 11188,
            "steering_direction": "toward_high",
            "final_score": 0.94,
            "association_score": 0.94,
            "cohen_d": 0.36,
            "mean_class": 0.39,
            "mean_rest": 0.29,
            "diff_class_minus_rest": 0.10,
            "activation_prevalence": 0.41,
            "top_activation_slide_key": "TCGA-A-01Z-00-DX1",
            "top_activation_coord_x": 1024,
            "top_activation_coord_y": 2048,
        },
        {
            "concept_rank": 2,
            "task": "tumor_purity_low_high",
            "class_label": "high",
            "latent_idx": 1435,
            "steering_direction": "toward_high",
            "final_score": 0.83,
            "association_score": 0.83,
            "cohen_d": 0.34,
            "mean_class": 0.69,
            "mean_rest": 0.61,
            "diff_class_minus_rest": 0.08,
            "activation_prevalence": 0.69,
            "top_activation_slide_key": "TCGA-B-01Z-00-DX1",
            "top_activation_coord_x": 4096,
            "top_activation_coord_y": 8192,
        },
    ]
    write_json(
        concept_dir / "selected_concepts.json",
        {
            "task": "tumor_purity_low_high",
            "class_label": "high",
            "mode": "labels_only",
            "concept_quality_mode": "association",
            "concepts": concepts,
        },
    )
    write_json(
        concept_dir / "summary.json",
        {
            "args": {
                "task": "tumor_purity_low_high",
                "class_label": "high",
                "sae_variant": "tcga_uni2_sae_relu_v1",
                "sae_ckpt": "/repo/resources/models/sae/relu_final.pt",
                "sae_cfg": "/repo/resources/models/sae/run_config.json",
            },
            "command": "python scripts/find_label_concepts.py --task tumor_purity_low_high",
            "effective_mode": "labels_only",
            "concept_quality_mode": "association",
            "sae_d_in": 1536,
            "sae_d_latent": 12288,
        },
    )
    write_csv(
        concept_dir / "concept_cards.csv",
        concepts,
        [
            "concept_rank",
            "task",
            "class_label",
            "latent_idx",
            "steering_direction",
            "final_score",
            "association_score",
            "cohen_d",
            "mean_class",
            "mean_rest",
            "diff_class_minus_rest",
            "activation_prevalence",
            "top_activation_slide_key",
            "top_activation_coord_x",
            "top_activation_coord_y",
        ],
    )
    write_csv(
        concept_dir / "representative_tiles.csv",
        [
            {
                "task": "tumor_purity_low_high",
                "class_label": "high",
                "latent_idx": 1435,
                "ranking_method": "activation",
                "tile_rank": 1,
                "activation": 154.0,
                "case_id": "TCGA-B",
                "slide_key": "TCGA-B-01Z-00-DX1",
                "project_dir": "TCGA-GBM",
                "label": "high",
                "h5_path": "/server/features/TCGA-B-01Z-00-DX1.h5",
                "tile_index": 22,
                "coord_x": 4096,
                "coord_y": 8192,
            },
            {
                "task": "tumor_purity_low_high",
                "class_label": "high",
                "latent_idx": 11188,
                "ranking_method": "activation",
                "tile_rank": 1,
                "activation": 21.0,
                "case_id": "TCGA-A",
                "slide_key": "TCGA-A-01Z-00-DX1",
                "project_dir": "TCGA-LGG",
                "label": "high",
                "h5_path": "/server/features/TCGA-A-01Z-00-DX1.h5",
                "tile_index": 11,
                "coord_x": 1024,
                "coord_y": 2048,
            },
            {
                "task": "tumor_purity_low_high",
                "class_label": "high",
                "latent_idx": 11188,
                "ranking_method": "activation",
                "tile_rank": 2,
                "activation": 20.5,
                "case_id": "TCGA-A",
                "slide_key": "TCGA-A-01Z-00-DX1",
                "project_dir": "TCGA-LGG",
                "label": "high",
                "h5_path": "/server/features/TCGA-A-01Z-00-DX1.h5",
                "tile_index": 12,
                "coord_x": 1280,
                "coord_y": 2048,
            },
        ],
        [
            "task",
            "class_label",
            "latent_idx",
            "ranking_method",
            "tile_rank",
            "activation",
            "attention",
            "attention_norm",
            "attention_weighted_activation",
            "case_id",
            "slide_key",
            "project_dir",
            "label",
            "h5_path",
            "tile_index",
            "coord_x",
            "coord_y",
        ],
    )
    return concept_dir


def test_export_concept_package_writes_portable_files(tmp_path: Path) -> None:
    script = load_script_module("export_concept_package.py")
    concept_dir = make_concept_dir(tmp_path)
    out_dir = tmp_path / "concept_export"

    args = script.build_arg_parser().parse_args(["--concept-dir", str(concept_dir), "--out-dir", str(out_dir)])
    outputs = script.export_concept_package(args)

    for output in outputs.values():
        assert Path(output).exists()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "concept_export.v1"
    assert manifest["task"] == "tumor_purity_low_high"
    assert manifest["class_label"] == "high"
    assert manifest["feature_space"] == {"name": "UNI2", "feature_dim": 1536}
    assert manifest["sae"]["variant"] == "tcga_uni2_sae_relu_v1"
    assert manifest["tile_extraction"] == {
        "target_magnification": 20.0,
        "tile_size_px": 256,
        "coord_space": "level0_h5_coords",
    }

    concepts = read_csv(out_dir / "concepts.csv")
    assert [row["latent_idx"] for row in concepts] == ["11188", "1435"]

    reps = read_csv(out_dir / "representative_tiles.csv")
    assert [row["concept_rank"] for row in reps] == ["1", "1", "2"]
    assert set(reps[0]) >= {"concept_rank", "latent_idx", "coord_x", "coord_y", "h5_path"}

    slide_keys = read_csv(out_dir / "slide_keys.csv")
    assert slide_keys == [
        {
            "slide_key": "TCGA-A-01Z-00-DX1",
            "case_id": "TCGA-A",
            "project_dir": "TCGA-LGG",
            "n_tiles_requested": "2",
        },
        {
            "slide_key": "TCGA-B-01Z-00-DX1",
            "case_id": "TCGA-B",
            "project_dir": "TCGA-GBM",
            "n_tiles_requested": "1",
        },
    ]

    path_template = read_csv(out_dir / "slide_path_map.template.csv")
    assert [row["slide_key"] for row in path_template] == [row["slide_key"] for row in slide_keys]
    assert path_template[0]["svs_path"].endswith("TCGA-A-01Z-00-DX1.svs")

    format_md = (out_dir / "FORMAT.md").read_text()
    assert "20x-equivalent `256 x 256` tile" in format_md
    assert "level-0 slide coordinates" in format_md


def test_find_label_concepts_source_wires_portable_export() -> None:
    # Keep this test lightweight: importing find_label_concepts imports torch, which
    # may fail in non-ML test environments missing CUDA shared libraries.
    source = (Path(__file__).resolve().parents[1] / "scripts/find_label_concepts.py").read_text()

    assert "from export_concept_package import export_concept_package" in source
    assert "--no-export-concept-package" in source
    assert "maybe_export_concept_package(args, out_dir)" in source
