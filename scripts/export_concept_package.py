#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json


SCHEMA_VERSION = "concept_export.v1"
CONCEPT_FIELDS = [
    "concept_rank",
    "latent_idx",
    "class_label",
    "steering_direction",
    "final_score",
    "association_score",
    "cohen_d",
    "mean_class",
    "mean_rest",
    "diff_class_minus_rest",
    "activation_prevalence",
    "top_activation_slide_key",
    "top_activation_coord_x",
    "top_activation_coord_y",
]
REPRESENTATIVE_TILE_FIELDS = [
    "concept_rank",
    "latent_idx",
    "tile_rank",
    "ranking_method",
    "activation",
    "case_id",
    "slide_key",
    "project_dir",
    "label",
    "tile_index",
    "coord_x",
    "coord_y",
    "h5_path",
]
SLIDE_KEY_FIELDS = ["slide_key", "case_id", "project_dir", "n_tiles_requested"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export discovered SAE label concepts into a portable package for local-PC SVS tile review. "
            "The package contains stable CSV/JSON manifests and FORMAT.md, but does not extract image tiles."
        )
    )
    parser.add_argument(
        "--concept-dir",
        type=Path,
        required=True,
        help="Directory containing selected_concepts.json, concept_cards.csv, representative_tiles.csv, and summary.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output package directory. Defaults to <concept-dir>/concept_export.",
    )
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument("--tile-size-px", type=int, default=256)
    parser.add_argument("--coord-space", type=str, default="level0_h5_coords")
    parser.add_argument("--feature-name", type=str, default="UNI2")
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required concept artifact: {path}")
    return path


def safe_get(row: dict[str, Any], key: str) -> Any:
    value = row.get(key, "")
    return "" if value is None else value


def normalize_int_text(value: Any) -> str:
    text = str(value)
    if text == "":
        return ""
    return str(int(float(text)))


def build_concepts_rows(concepts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    rows: list[dict[str, Any]] = []
    rank_by_latent: dict[int, int] = {}
    for idx, concept in enumerate(concepts, start=1):
        latent_idx = int(concept["latent_idx"])
        concept_rank = int(concept.get("concept_rank") or idx)
        rank_by_latent[latent_idx] = concept_rank
        rows.append(
            {
                field: safe_get(concept, field)
                for field in CONCEPT_FIELDS
            }
        )
        rows[-1]["concept_rank"] = concept_rank
        rows[-1]["latent_idx"] = latent_idx
    rows.sort(key=lambda row: (int(row["concept_rank"]), int(row["latent_idx"])))
    return rows, rank_by_latent


def build_representative_rows(
    representative_rows: list[dict[str, str]],
    *,
    rank_by_latent: dict[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_latents = set(rank_by_latent)
    for row in representative_rows:
        latent_idx = int(row["latent_idx"])
        if latent_idx not in selected_latents:
            continue
        out = {field: safe_get(row, field) for field in REPRESENTATIVE_TILE_FIELDS}
        out["concept_rank"] = int(rank_by_latent[latent_idx])
        out["latent_idx"] = latent_idx
        out["tile_rank"] = normalize_int_text(out["tile_rank"])
        out["tile_index"] = normalize_int_text(out["tile_index"])
        out["coord_x"] = normalize_int_text(out["coord_x"])
        out["coord_y"] = normalize_int_text(out["coord_y"])
        rows.append(out)
    rows.sort(
        key=lambda row: (
            int(row["concept_rank"]),
            int(row["latent_idx"]),
            str(row["ranking_method"]),
            int(row["tile_rank"] or 0),
            str(row["slide_key"]),
            int(row["tile_index"] or 0),
        )
    )
    return rows


def build_slide_key_rows(representative_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row["slide_key"]) for row in representative_rows)
    first_by_slide: dict[str, dict[str, Any]] = {}
    for row in representative_rows:
        slide_key = str(row["slide_key"])
        if slide_key not in first_by_slide:
            first_by_slide[slide_key] = row
    rows = [
        {
            "slide_key": slide_key,
            "case_id": safe_get(first_by_slide[slide_key], "case_id"),
            "project_dir": safe_get(first_by_slide[slide_key], "project_dir"),
            "n_tiles_requested": int(counts[slide_key]),
        }
        for slide_key in sorted(first_by_slide)
    ]
    return rows


def build_slide_path_template_rows(slide_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "slide_key": str(row["slide_key"]),
            "svs_path": f"/path/on/local/pc/{row['slide_key']}.svs",
        }
        for row in slide_rows
    ]


def format_markdown(*, manifest: dict[str, Any], high_level_example: dict[str, Any] | None) -> str:
    task = manifest.get("task", "")
    class_label = manifest.get("class_label", "")
    example_text = ""
    if high_level_example:
        example_text = (
            "\nExample representative row:\n\n"
            "```csv\n"
            "latent_idx,slide_key,coord_x,coord_y\n"
            f"{high_level_example.get('latent_idx', '')},"
            f"{high_level_example.get('slide_key', '')},"
            f"{high_level_example.get('coord_x', '')},"
            f"{high_level_example.get('coord_y', '')}\n"
            "```\n"
        )
    return f"""# Portable Concept Export Format

This directory is a portable export of discovered SAE concepts for `task={task}` and `class_label={class_label}`.
It is designed for reviewing actual image tiles on a local PC that stores the SVS files.

## Files

- `manifest.json`: package metadata, source artifact paths, SAE/feature provenance, and the default tile extraction contract.
- `concepts.csv`: one row per selected concept latent with association and ranking evidence.
- `representative_tiles.csv`: top activated feature tiles for the selected concepts.
- `slide_keys.csv`: unique slides needed for local extraction.
- `slide_path_map.template.csv`: copy this to `slide_path_map.csv` and fill in local SVS paths.

## Original Discovery Artifacts

The source discovery run produced these repo-native files:

- `selected_concepts.json`: ranked concept records used by steering and downstream scripts.
- `concept_cards.csv`: table version of concept-level evidence.
- `representative_tiles.csv`: top activated tiles selected from feature bags by SAE activation, and optionally attention-weighted activation.
- `summary.json`: command, arguments, SAE metadata, and scan provenance.

## Local PC Tile Extraction

1. Copy `slide_path_map.template.csv` to `slide_path_map.csv`.
2. Fill `slide_path_map.csv` with absolute SVS paths on the local PC.
3. Match `representative_tiles.csv.slide_key` to `slide_path_map.csv.slide_key`.
4. Open the matching SVS file.
5. Read a 20x-equivalent `256 x 256` tile anchored at `coord_x, coord_y`.

## Coordinate Contract

- `coord_x` and `coord_y` come from the H5 `coords` dataset used with the UNI2 feature bag.
- Coordinates are treated as level-0 slide coordinates.
- The default visual extraction target is 20x magnification with a 256-pixel output tile.
- Extraction code must read the SVS objective power and convert the 20x request to the correct level-0 crop size.

## Verification Notes

- Concept discovery used top activated feature tiles, not raw image patches.
- Local extracted tiles are for visual and pathology review of those feature-selected coordinates.
- `h5_path` in `representative_tiles.csv` is provenance only; local extraction should use `slide_key` plus `slide_path_map.csv`.
{example_text}"""


def build_manifest(
    *,
    selected_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    concept_dir: Path,
    args: argparse.Namespace,
    n_concepts: int,
    n_representative_tiles: int,
    n_slides: int,
) -> dict[str, Any]:
    summary_args = summary_payload.get("args", {}) if isinstance(summary_payload, dict) else {}
    source_artifacts = {
        "selected_concepts_json": str(concept_dir / "selected_concepts.json"),
        "concept_cards_csv": str(concept_dir / "concept_cards.csv"),
        "representative_tiles_csv": str(concept_dir / "representative_tiles.csv"),
        "summary_json": str(concept_dir / "summary.json"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": selected_payload.get("task", summary_args.get("task", "")),
        "class_label": selected_payload.get("class_label", summary_args.get("class_label", "")),
        "mode": selected_payload.get("mode", summary_payload.get("effective_mode", "")),
        "concept_quality_mode": selected_payload.get(
            "concept_quality_mode",
            summary_payload.get("concept_quality_mode", ""),
        ),
        "source_artifacts": source_artifacts,
        "feature_space": {
            "name": str(args.feature_name),
            "feature_dim": int(summary_payload.get("sae_d_in", summary_args.get("feature_dim", 1536))),
        },
        "sae": {
            "variant": str(summary_args.get("sae_variant", "")),
            "checkpoint": str(summary_args.get("sae_ckpt", "")),
            "config": str(summary_args.get("sae_cfg", "")),
            "latent_dim": int(summary_payload.get("sae_d_latent", summary_args.get("sae_d_latent", 12288))),
        },
        "tile_extraction": {
            "target_magnification": float(args.target_magnification),
            "tile_size_px": int(args.tile_size_px),
            "coord_space": str(args.coord_space),
        },
        "created_from": {
            "command": str(summary_payload.get("command", "")),
            "args": summary_args,
        },
        "counts": {
            "concepts": int(n_concepts),
            "representative_tiles": int(n_representative_tiles),
            "slides": int(n_slides),
        },
    }


def export_concept_package(args: argparse.Namespace) -> dict[str, str]:
    concept_dir = args.concept_dir.resolve()
    out_dir = (args.out_dir or (concept_dir / "concept_export")).resolve()
    selected_path = require_file(concept_dir / "selected_concepts.json")
    concept_cards_path = require_file(concept_dir / "concept_cards.csv")
    representative_path = require_file(concept_dir / "representative_tiles.csv")
    summary_path = require_file(concept_dir / "summary.json")

    selected_payload = read_json(selected_path)
    summary_payload = read_json(summary_path)
    concepts = list(selected_payload.get("concepts", []))
    if not concepts:
        raise ValueError(f"{selected_path}: no concepts found")

    # Read concept_cards.csv as a consistency check and to fail early on malformed source packages.
    concept_card_rows = read_csv_rows(concept_cards_path)
    concept_card_latents = {int(row["latent_idx"]) for row in concept_card_rows if row.get("latent_idx")}
    missing_card_latents = [int(row["latent_idx"]) for row in concepts if int(row["latent_idx"]) not in concept_card_latents]
    if missing_card_latents:
        raise ValueError(f"{concept_cards_path}: missing selected latents {missing_card_latents}")

    concept_rows, rank_by_latent = build_concepts_rows(concepts)
    representative_rows = build_representative_rows(read_csv_rows(representative_path), rank_by_latent=rank_by_latent)
    slide_rows = build_slide_key_rows(representative_rows)
    slide_template_rows = build_slide_path_template_rows(slide_rows)
    manifest = build_manifest(
        selected_payload=selected_payload,
        summary_payload=summary_payload,
        concept_dir=concept_dir,
        args=args,
        n_concepts=len(concept_rows),
        n_representative_tiles=len(representative_rows),
        n_slides=len(slide_rows),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", manifest)
    write_csv(out_dir / "concepts.csv", concept_rows, CONCEPT_FIELDS)
    write_csv(out_dir / "representative_tiles.csv", representative_rows, REPRESENTATIVE_TILE_FIELDS)
    write_csv(out_dir / "slide_keys.csv", slide_rows, SLIDE_KEY_FIELDS)
    write_csv(out_dir / "slide_path_map.template.csv", slide_template_rows, ["slide_key", "svs_path"])
    (out_dir / "FORMAT.md").write_text(
        format_markdown(
            manifest=manifest,
            high_level_example=representative_rows[0] if representative_rows else None,
        )
    )
    return {
        "manifest_json": str(out_dir / "manifest.json"),
        "concepts_csv": str(out_dir / "concepts.csv"),
        "representative_tiles_csv": str(out_dir / "representative_tiles.csv"),
        "slide_keys_csv": str(out_dir / "slide_keys.csv"),
        "slide_path_map_template_csv": str(out_dir / "slide_path_map.template.csv"),
        "format_md": str(out_dir / "FORMAT.md"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    outputs = export_concept_package(args)
    print(json.dumps({"command": " ".join(shlex.quote(part) for part in sys.argv), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
