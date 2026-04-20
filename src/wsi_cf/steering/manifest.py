from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from wsi_cf.common.io import read_json, write_json


def parse_steer_spec(spec: str) -> tuple[int, int, Path]:
    parts = [part.strip() for part in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Bad steer spec '{spec}'. Expected format: gx,gy,path.npy")
    return int(parts[0]), int(parts[1]), Path(parts[2])


def load_steer_manifest(manifest_path: Path) -> list[str]:
    payload = read_json(manifest_path)
    if not isinstance(payload, list):
        raise ValueError(f"Steer manifest must be a JSON list: {manifest_path}")
    specs: list[str] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Steer manifest item {idx} must be an object")
        if "gx" not in item or "gy" not in item or "path" not in item:
            raise ValueError(f"Steer manifest item {idx} must contain gx, gy, path")
        gx = int(item["gx"])
        gy = int(item["gy"])
        path = Path(str(item["path"]))
        specs.append(f"{gx},{gy},{path}")
    return specs


def write_manifest_json(path: Path, manifest: list[dict[str, object]]) -> None:
    write_json(path, manifest)


def write_manifest_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def build_naive_manifest_preview(
    *,
    rows: Sequence[object],
    count: int,
    grid_side: int = 4,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if count > len(rows):
        raise ValueError(f"Need {count} donor rows, found {len(rows)}")
    manifest: list[dict[str, object]] = []
    preview: list[dict[str, object]] = []
    chosen = list(rows[:count])
    for idx, row in enumerate(chosen):
        gy = idx // grid_side
        gx = idx % grid_side
        manifest.append({"gx": int(gx), "gy": int(gy), "path": str(row.feature_path)})
        preview.append(
            {
                "gx": int(gx),
                "gy": int(gy),
                "label": int(row.label),
                "slide_key": str(row.slide_key),
                "tile_index": int(row.tile_index),
                "coord_x": int(row.coord_x),
                "coord_y": int(row.coord_y),
                "feature_path": str(row.feature_path),
                "image_path": str(row.image_path),
            }
        )
    return manifest, preview


def infer_coord_step(coords: np.ndarray) -> int:
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    dx = np.diff(xs)
    dy = np.diff(ys)
    vals = np.concatenate([dx[dx > 0], dy[dy > 0]])
    if vals.size == 0:
        raise RuntimeError("Could not infer coordinate step from coords")
    return int(np.min(vals))


def build_real_grid_rows(
    *,
    coords: np.ndarray,
    tile_index: int,
    grid_side: int,
    anchor_gx: int,
    anchor_gy: int,
) -> list[dict[str, int]]:
    if grid_side <= 0:
        raise ValueError("grid_side must be > 0")
    if not (0 <= anchor_gx < grid_side and 0 <= anchor_gy < grid_side):
        raise ValueError("anchor_gx/anchor_gy must be within the grid")
    if tile_index < 0 or tile_index >= int(coords.shape[0]):
        raise ValueError(f"tile_index {tile_index} out of range for coords with {coords.shape[0]} rows")

    step = infer_coord_step(coords)
    anchor_x = int(coords[int(tile_index), 0])
    anchor_y = int(coords[int(tile_index), 1])
    top_left_x = int(anchor_x - anchor_gx * step)
    top_left_y = int(anchor_y - anchor_gy * step)
    coord_to_idx = {tuple(map(int, coord)): idx for idx, coord in enumerate(coords.tolist())}

    rows: list[dict[str, int]] = []
    for gy in range(int(grid_side)):
        for gx in range(int(grid_side)):
            coord_x = int(top_left_x + gx * step)
            coord_y = int(top_left_y + gy * step)
            idx = coord_to_idx.get((coord_x, coord_y))
            if idx is None:
                raise RuntimeError(
                    f"Requested real grid is incomplete. Missing coord {(coord_x, coord_y)} for gx={gx}, gy={gy}."
                )
            rows.append(
                {
                    "gx": int(gx),
                    "gy": int(gy),
                    "tile_index": int(idx),
                    "coord_x": int(coord_x),
                    "coord_y": int(coord_y),
                }
            )
    return rows
