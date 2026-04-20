from __future__ import annotations

from pathlib import Path

from wsi_cf.data.donor_pool import parse_donor_pool_csv


def test_parse_donor_pool_preserves_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "tile_pool.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,label,hpv_status,case_id,slide_key,tile_index,coord_x,coord_y,feature_path,image_path",
                "test,1,positive,TCGA-XX,TCGA-XX-0001,12,100,200,/tmp/a.npy,/tmp/a.png",
            ]
        )
    )
    rows = parse_donor_pool_csv(csv_path)
    assert len(rows) == 1
    row = rows[0]
    assert row.label == 1
    assert row.slide_key == "TCGA-XX-0001"
    assert row.tile_index == 12
    assert row.coord_x == 100
    assert row.coord_y == 200
    assert row.feature_path == "/tmp/a.npy"
    assert row.image_path == "/tmp/a.png"
