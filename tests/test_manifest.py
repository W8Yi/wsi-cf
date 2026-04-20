from __future__ import annotations

from pathlib import Path

import pytest

from wsi_cf.common.io import write_json
from wsi_cf.steering.manifest import load_steer_manifest, parse_steer_spec


def test_parse_steer_spec_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        parse_steer_spec("1,2")


def test_load_steer_manifest_rejects_malformed_items(tmp_path: Path) -> None:
    bad_manifest = tmp_path / "bad_manifest.json"
    write_json(bad_manifest, [{"gx": 0, "path": "foo.npy"}])
    with pytest.raises(ValueError):
        load_steer_manifest(bad_manifest)
