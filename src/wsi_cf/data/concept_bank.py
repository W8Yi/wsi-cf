from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wsi_cf.data.donor_pool import canonical_h5_path, load_split_rows
from wsi_cf.data.slides import find_slide_path


@dataclass(frozen=True)
class ConceptRepresentativeRow:
    latent_idx: int
    selected_direction: str
    prototype_rank: int
    label: int
    hpv_status: str
    split: str
    case_id: str
    slide_key: str
    slide_path: str
    canonical_h5_path: str
    tile_index: int
    coord_x: int
    coord_y: int
    old_h5_path: str
    attention: float
    sae_activation: float
    attention_weighted_activation: float
    pred: int
    prob_pos: float


def parse_top_tiles_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "latent_idx": int(row["latent_idx"]),
                    "selected_direction": str(row["selected_direction"]),
                    "prototype_rank": int(row["prototype_rank"]),
                    "label": int(row["label"]),
                    "pred": int(row["pred"]),
                    "prob_pos": float(row["prob_pos"]),
                    "case_id": str(row["case_id"]),
                    "slide_key": str(row["slide_key"]),
                    "tile_index": int(row["tile_index"]),
                    "attention": float(row["attention"]),
                    "sae_activation": float(row["sae_activation"]),
                    "attention_weighted_activation": float(row["attention_weighted_activation"]),
                    "coord_x": int(row["coord_x"]),
                    "coord_y": int(row["coord_y"]),
                    "h5_path": str(row["h5_path"]),
                }
            )
    return rows


def _load_split_map(split_tsv: Path) -> dict[str, dict[str, Any]]:
    return {str(row["slide_key"]): dict(row) for row in load_split_rows(split_tsv, split_filter="all")}


def _make_tiles_index(rows: list[dict[str, Any]]) -> dict[tuple[int, int, str, int], dict[str, Any]]:
    out: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["latent_idx"]),
            int(row["prototype_rank"]),
            str(row["slide_key"]),
            int(row["tile_index"]),
        )
        out[key] = dict(row)
    return out


def load_representative_tiles(
    *,
    prototype_json: Path,
    tiles_csv: Path,
    split_tsv: Path,
    slides_dir: Path,
    features_dir: Path,
    selected_direction: str,
    examples_per_latent: int,
    latent_ids: set[int] | None = None,
) -> list[ConceptRepresentativeRow]:
    import json

    bundle = json.loads(prototype_json.read_text())
    per_latent = bundle.get("per_latent", {})
    tiles_index = _make_tiles_index(parse_top_tiles_csv(tiles_csv))
    split_map = _load_split_map(split_tsv)

    requested_direction = str(selected_direction).strip().lower()
    out: list[ConceptRepresentativeRow] = []
    for latent_key, payload in per_latent.items():
        latent_idx = int(payload.get("latent_idx", latent_key))
        if latent_ids and latent_idx not in latent_ids:
            continue
        direction = str(payload.get("selected_direction", "")).strip().lower()
        if requested_direction != "all" and direction != requested_direction:
            continue
        examples = list(payload.get("source_examples", []))[: max(1, int(examples_per_latent))]
        for example in examples:
            slide_key = str(example["slide_key"])
            split_row = split_map.get(slide_key)
            slide_path = find_slide_path(slides_dir, slide_key)
            h5_path = canonical_h5_path(features_dir, slide_key)
            if split_row is None or slide_path is None or not h5_path.exists():
                continue
            key = (
                int(latent_idx),
                int(example["prototype_rank"]),
                slide_key,
                int(example["tile_index"]),
            )
            csv_row = tiles_index.get(key)
            if csv_row is None:
                continue
            out.append(
                ConceptRepresentativeRow(
                    latent_idx=int(latent_idx),
                    selected_direction=direction,
                    prototype_rank=int(example["prototype_rank"]),
                    label=int(split_row["label"]),
                    hpv_status=str(split_row["hpv_status"]),
                    split=str(split_row["split"]),
                    case_id=str(split_row["case_id"]),
                    slide_key=slide_key,
                    slide_path=str(slide_path),
                    canonical_h5_path=str(h5_path),
                    tile_index=int(example["tile_index"]),
                    coord_x=int(csv_row["coord_x"]),
                    coord_y=int(csv_row["coord_y"]),
                    old_h5_path=str(csv_row["h5_path"]),
                    attention=float(example.get("attention", csv_row["attention"])),
                    sae_activation=float(example.get("sae_activation", csv_row["sae_activation"])),
                    attention_weighted_activation=float(
                        example.get("attention_weighted_activation", csv_row["attention_weighted_activation"])
                    ),
                    pred=int(example.get("pred", csv_row["pred"])),
                    prob_pos=float(example.get("prob_pos", csv_row["prob_pos"])),
                )
            )
    out.sort(key=lambda row: (row.selected_direction, row.latent_idx, row.prototype_rank, row.slide_key))
    return out
