from __future__ import annotations

import numpy as np
import pytest

from wsi_cf.steering.manifest import build_real_grid_rows


def test_real_grid_builder_fails_on_incomplete_grid() -> None:
    coords = np.asarray(
        [
            [0, 0],
            [256, 0],
            [0, 256],
        ],
        dtype=np.int64,
    )
    with pytest.raises(RuntimeError):
        build_real_grid_rows(coords=coords, tile_index=0, grid_side=2, anchor_gx=0, anchor_gy=0)
