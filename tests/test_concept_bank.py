from __future__ import annotations

import json
from pathlib import Path

from wsi_cf.data.concept_bank import load_representative_tiles


def test_load_representative_tiles_matches_prototype_json_and_csv(tmp_path: Path) -> None:
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    (slides_dir / "SLIDE1.sample.svs").write_bytes(b"")

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    (features_dir / "SLIDE1.h5").write_bytes(b"")

    split_tsv = tmp_path / "split.tsv"
    split_tsv.write_text(
        "split\tlabel\thpv_status\tcase_id\tslide_key\n"
        "test\t1\tpositive\tCASE1\tSLIDE1\n"
    )

    tiles_csv = tmp_path / "top_tiles.csv"
    tiles_csv.write_text(
        "\n".join(
            [
                "latent_idx,selected_direction,prototype_rank,label,pred,prob_pos,case_id,slide_key,tile_index,attention,sae_activation,attention_weighted_activation,coord_x,coord_y,h5_path",
                "2645,hpv_pos,1,1,1,0.9,CASE1,SLIDE1,123,0.8,5.0,4.0,1024,2048,/old/features/SLIDE1.h5",
            ]
        )
    )

    prototype_json = tmp_path / "prototype.json"
    prototype_json.write_text(
        json.dumps(
            {
                "per_latent": {
                    "2645": {
                        "latent_idx": 2645,
                        "selected_direction": "hpv_pos",
                        "source_examples": [
                            {
                                "prototype_rank": 1,
                                "case_id": "CASE1",
                                "slide_key": "SLIDE1",
                                "tile_index": 123,
                                "label": 1,
                                "pred": 1,
                                "prob_pos": 0.9,
                                "attention": 0.8,
                                "sae_activation": 5.0,
                                "attention_weighted_activation": 4.0,
                                "h5_path": "/old/features/SLIDE1.h5",
                            }
                        ],
                    }
                }
            }
        )
    )

    rows = load_representative_tiles(
        prototype_json=prototype_json,
        tiles_csv=tiles_csv,
        split_tsv=split_tsv,
        slides_dir=slides_dir,
        features_dir=features_dir,
        selected_direction="hpv_pos",
        examples_per_latent=1,
        latent_ids=None,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.latent_idx == 2645
    assert row.slide_key == "SLIDE1"
    assert row.coord_x == 1024
    assert row.coord_y == 2048
    assert row.slide_path.endswith(".svs")
    assert row.canonical_h5_path.endswith("SLIDE1.h5")
