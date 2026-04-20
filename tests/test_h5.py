from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from wsi_cf.data.h5 import load_h5_features_coords, resolve_tile_index


def write_h5(path: Path, *, features: np.ndarray, coords: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)


def test_h5_loader_handles_2d_and_3d(tmp_path: Path) -> None:
    feats_2d = np.arange(12, dtype=np.float32).reshape(3, 4)
    coords_2d = np.asarray([[0, 0], [10, 20], [30, 40]], dtype=np.int64)
    path_2d = tmp_path / "two_d.h5"
    write_h5(path_2d, features=feats_2d, coords=coords_2d)

    x2, c2 = load_h5_features_coords(path_2d)
    assert x2.shape == (3, 4)
    assert c2 is not None
    assert c2.shape == (3, 2)

    feats_3d = feats_2d[None, ...]
    coords_3d = coords_2d[None, ...]
    path_3d = tmp_path / "three_d.h5"
    write_h5(path_3d, features=feats_3d, coords=coords_3d)

    x3, c3 = load_h5_features_coords(path_3d)
    assert x3.shape == (3, 4)
    assert c3 is not None
    assert c3.shape == (3, 2)


def test_resolve_tile_index_by_index_and_coord(tmp_path: Path) -> None:
    feats = np.arange(20, dtype=np.float32).reshape(5, 4)
    coords = np.asarray([[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.int64)
    path = tmp_path / "tiles.h5"
    write_h5(path, features=feats, coords=coords)
    x, c = load_h5_features_coords(path)

    assert resolve_tile_index(feats=x, coords=c, tile_index=3, coord=None) == 3
    assert resolve_tile_index(feats=x, coords=c, tile_index=None, coord=(20, 20)) == 2
