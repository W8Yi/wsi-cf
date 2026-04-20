from __future__ import annotations

import random

from wsi_cf.steering.cell_selection import (
    block_cells,
    decode_cells,
    encode_cells,
    parse_cell_specs,
    random_cells,
    random_connected_cells,
)


def test_parse_encode_decode_cells_round_trip() -> None:
    cells = parse_cell_specs(["1,1", "2,1", "2,2"])
    assert cells == [(1, 1), (2, 1), (2, 2)]
    enc = encode_cells(cells)
    assert enc == "1,1;2,1;2,2"
    assert decode_cells(enc) == cells


def test_random_connected_cells_is_connected() -> None:
    rng = random.Random(7)
    cells = random_connected_cells(grid_w=4, grid_h=4, count=3, rng=rng, start=(1, 1))
    assert len(cells) == 3
    cell_set = set(cells)
    assert (1, 1) in cell_set
    # every non-anchor cell should touch at least one other selected cell
    for gx, gy in cells:
        neighbors = {(gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)}
        assert any(n in cell_set for n in neighbors if n != (gx, gy))


def test_block_cells_builds_expected_shapes() -> None:
    assert block_cells(origin_gx=1, origin_gy=1, width=2, height=2, grid_w=4, grid_h=4) == [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
    ]
    assert block_cells(origin_gx=1, origin_gy=0, width=2, height=3, grid_w=4, grid_h=4) == [
        (1, 0),
        (2, 0),
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
    ]


def test_random_cells_count() -> None:
    rng = random.Random(3)
    cells = random_cells(grid_w=4, grid_h=4, count=2, rng=rng)
    assert len(cells) == 2
    assert len(set(cells)) == 2
