from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import load_script_module


overlay = load_script_module("overlay_sae_concepts_on_wsi.py")


def _write_concept_export(root: Path, task: str, label: str, latent_ids: list[int]) -> Path:
    export_dir = root / "selected" / f"01_{task}__{label}" / "concept_export"
    export_dir.mkdir(parents=True)
    (export_dir / "concepts.csv").write_text(
        "concept_rank,latent_idx,class_label,steering_direction,final_score,association_score,cohen_d\n"
        + "\n".join(
            f"{rank},{latent},{label},toward_{label},1.0,0.9,0.5"
            for rank, latent in enumerate(latent_ids, start=1)
        )
        + "\n"
    )
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "concept_export.v1",
                "task": task,
                "class_label": label,
                "feature_space": {"name": "UNI2", "feature_dim": 1536},
                "sae": {
                    "checkpoint": "/tmp/fake_sae.pt",
                    "config": "/tmp/fake_sae.json",
                    "latent_dim": 12288,
                },
                "tile_extraction": {
                    "target_magnification": 20.0,
                    "tile_size_px": 256,
                    "coord_space": "level0_h5_coords",
                },
            }
        )
    )
    return export_dir


def test_validate_feature_coord_shapes_enforces_matching_lengths() -> None:
    features = np.zeros((3, 4), dtype=np.float32)
    coords = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="length mismatch"):
        overlay.validate_feature_coord_shapes(features, coords)


def test_read_h5_features_coords_accepts_feature_and_coord_datasets(tmp_path: Path) -> None:
    h5_path = tmp_path / "sample.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("features", data=np.zeros((1, 4, 1536), dtype=np.float32))
        handle.create_dataset("coords", data=np.array([[[0, 0], [256, 0], [0, 256], [256, 256]]], dtype=np.int64))

    features, coords = overlay.read_h5_features_coords(h5_path)
    assert features.shape == (4, 1536)
    assert coords.shape == (4, 2)


def test_infer_coord_tile_size_prefers_smallest_grid_step() -> None:
    coords = np.array([[0, 0], [256, 0], [512, 0], [0, 512]], dtype=np.int64)
    assert overlay.infer_coord_tile_size(coords, fallback=1024) == 256


def test_normalize_scores_handles_constant_values() -> None:
    norm = overlay.normalize_scores(np.array([3.0, 3.0, 3.0]), vmin_percentile=50, vmax_percentile=99)
    assert np.allclose(norm, 0.0)


def test_tile_bounds_on_thumbnail_stay_inside_canvas() -> None:
    bounds = overlay.tile_bounds_on_thumbnail(
        (2000, 2000),
        tile_size_level0=512,
        scale_x=0.1,
        scale_y=0.1,
        width=200,
        height=200,
    )
    x0, y0, x1, y1 = bounds
    assert 0 <= x0 < x1 <= 200
    assert 0 <= y0 < y1 <= 200


def test_load_selected_concepts_uses_curated_csv_and_top_n(tmp_path: Path) -> None:
    export_dir = _write_concept_export(tmp_path, "luad_lusc", "LUAD", [701, 11683, 12043])
    (tmp_path / "selected_morphology_labels.csv").write_text(
        "task,class_label,priority,verdict,available,concept_export_dir\n"
        f"luad_lusc,LUAD,1,best,True,{export_dir}\n"
    )

    labels, manifests = overlay.load_selected_concepts(
        concept_review_root=tmp_path,
        concept_set="all_curated",
        top_concepts_per_label=2,
        label_filters=[],
    )

    assert len(labels) == 1
    assert [int(row["latent_idx"]) for row in labels[0]["concepts"]] == [701, 11683]
    assert manifests["luad_lusc/LUAD"]["sae"]["latent_dim"] == 12288


def test_find_matched_slide_h5_pairs_uses_tcga_store_layout(tmp_path: Path) -> None:
    slide_root = tmp_path / "store"
    feature_root = tmp_path / "TCGA_features"
    slide_root.mkdir()
    (slide_root / "TCGA-AB-1234-01Z-00-DX1.svs").write_bytes(b"slide")
    h5_dir = feature_root / "TCGA-LUAD" / "features_uni2"
    h5_dir.mkdir(parents=True)
    (h5_dir / "TCGA-AB-1234-01Z-00-DX1.h5").write_bytes(b"h5")
    (h5_dir / "TCGA-NO-MATCH-01Z-00-DX1.h5").write_bytes(b"h5")

    pairs = overlay.find_matched_slide_h5_pairs(slide_store_root=slide_root, features_root=feature_root)

    assert len(pairs) == 1
    assert pairs[0]["slide_key"] == "TCGA-AB-1234-01Z-00-DX1"
    assert pairs[0]["project"] == "TCGA-LUAD"


def test_winner_concept_by_tile_uses_highest_normalized_concept() -> None:
    labels = [
        {"task": "task", "class_label": "A", "sae_group_index": 0, "concepts": [{"concept_rank": "1", "latent_idx": "10"}]},
        {"task": "task", "class_label": "B", "sae_group_index": 0, "concepts": [{"concept_rank": "1", "latent_idx": "11"}]},
    ]
    concepts = overlay.assign_concept_colors(labels)
    groups = [
        {
            "latent_to_col": {10: 0, 11: 1},
            "activations": np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
        }
    ]

    winners, scores, raw = overlay.winner_concept_by_tile(
        concept_records=concepts,
        sae_groups=groups,
        vmin_percentile=0,
        vmax_percentile=100,
        score_mode="raw",
    )

    assert [concepts[int(idx)]["class_label"] for idx in winners] == ["B", "A"]
    assert np.allclose(scores, 1.0)
    assert np.allclose(raw, [0.9, 0.8])


def test_load_all_sae_concepts_builds_every_latent(tmp_path: Path) -> None:
    cfg = tmp_path / "run_config.json"
    ckpt = tmp_path / "relu_final.pt"
    cfg.write_text(json.dumps({"latent_dim": 4}))
    ckpt.write_bytes(b"fake")

    labels, manifests = overlay.load_all_sae_concepts(sae_ckpt=ckpt, sae_cfg=cfg)

    assert len(labels) == 1
    assert len(labels[0]["concepts"]) == 4
    assert [int(row["latent_idx"]) for row in labels[0]["concepts"]] == [0, 1, 2, 3]
    assert manifests["all_sae/all_latents"]["sae"]["latent_dim"] == 4


def test_group_labels_by_sae_keeps_mixed_saes_separate(tmp_path: Path) -> None:
    labels = [
        {"task": "a", "class_label": "x", "manifest": {"sae": {"checkpoint": "a.pt", "config": "cfg.json", "latent_dim": 4}}},
        {"task": "b", "class_label": "y", "manifest": {"sae": {"checkpoint": "b.pt", "config": "cfg.json", "latent_dim": 4}}},
    ]
    groups = overlay.group_labels_by_sae(labels)
    assert len(groups) == 2
    assert labels[0]["sae_group_index"] == 0
    assert labels[1]["sae_group_index"] == 1
