from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class DonorPoolRow:
    label: int
    slide_key: str
    tile_index: int
    coord_x: int
    coord_y: int
    feature_path: str
    image_path: str
    hpv_status: str = ""
    case_id: str = ""
    split: str = ""


def load_split_rows(split_tsv: Path, split_filter: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with split_tsv.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if split_filter != "all" and str(row.get("split", "")) != split_filter:
                continue
            try:
                label = int(row.get("label", -1))
            except Exception:
                continue
            if label not in (0, 1):
                continue
            slide_key = str(row.get("slide_key", "")).strip()
            if not slide_key:
                continue
            rows.append(
                {
                    "split": str(row.get("split", "")),
                    "label": int(label),
                    "hpv_status": str(row.get("hpv_status", "")),
                    "case_id": str(row.get("case_id", "")),
                    "slide_key": slide_key,
                }
            )
    rows.sort(key=lambda row: (int(row["label"]), str(row["slide_key"])))
    return rows


def canonical_h5_path(features_dir: Path, slide_key: str) -> Path:
    return features_dir / f"{slide_key}.h5"


def choose_tile_indices(coords: np.ndarray, *, mode: str, k: int, rng: random.Random) -> list[int]:
    n = int(coords.shape[0])
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if mode == "first":
        return list(range(k))
    if mode == "random":
        return rng.sample(range(n), k=k)
    if mode == "center":
        center = coords.mean(axis=0, keepdims=True)
        d2 = ((coords.astype(np.float64) - center) ** 2).sum(axis=1)
        order = np.argsort(d2)
        return [int(i) for i in order[:k].tolist()]
    raise ValueError(f"Unsupported tile select mode: {mode}")


def parse_donor_pool_csv(csv_path: Path, *, label: int | None = None) -> list[DonorPoolRow]:
    out: list[DonorPoolRow] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = DonorPoolRow(
                label=int(row["label"]),
                slide_key=str(row["slide_key"]),
                tile_index=int(row["tile_index"]),
                coord_x=int(row["coord_x"]),
                coord_y=int(row["coord_y"]),
                feature_path=str(row["feature_path"]),
                image_path=str(row["image_path"]),
                hpv_status=str(row.get("hpv_status", "")),
                case_id=str(row.get("case_id", "")),
                split=str(row.get("split", "")),
            )
            if label is not None and parsed.label != int(label):
                continue
            out.append(parsed)
    return out


def make_contact_sheet(paths: list[Path], thumb_size: int = 128, ncols: int = 5, pad: int = 4) -> Image.Image:
    if not paths:
        return Image.new("RGB", (thumb_size, thumb_size), (245, 245, 245))
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(paths) / float(ncols)))
    width = pad + ncols * (thumb_size + pad)
    height = pad + nrows * (thumb_size + pad)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(paths):
        row = idx // ncols
        col = idx % ncols
        x0 = pad + col * (thumb_size + pad)
        y0 = pad + row * (thumb_size + pad)
        img = Image.open(path).convert("RGB").resize((thumb_size, thumb_size), resample=Image.BILINEAR)
        canvas.paste(img, (x0, y0))
        draw.rectangle([x0, y0, x0 + thumb_size - 1, y0 + thumb_size - 1], outline=(200, 200, 200), width=1)
    return canvas
