#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png, write_json
from wsi_cf.common.paths import DEFAULT_SAE_VARIANT, SAE_VARIANTS
from wsi_cf.data.region_bank import make_region_cells_preview
from wsi_cf.data.slides import (
    find_slide_path,
    infer_objective_power,
    open_slide,
    quick_region_quality_metrics,
    read_region_rgb_at_magnification,
)
from wsi_cf.steering.progressive import draw_cells_overlay


SETTINGS: dict[str, dict[str, Any]] = {
    "default": {
        "prototype_strength": 0.9,
        "preserve_edit_strength": 0.0,
        "preserve_visited_strength": 0.84,
        "preserve_fresh_context_strength": 0.22,
        "mid_steer_start_ratio": 0.55,
        "mid_steer_end_ratio": 1.0,
        "mid_steer_alpha_start": 0.4,
        "mid_steer_alpha_end": 1.0,
        "edit_support": "center_2x2",
    },
    "loose_context": {
        "preserve_fresh_context_strength": 0.15,
    },
    "stronger_preserve": {
        "preserve_edit_strength": 0.30,
        "preserve_fresh_context_strength": 0.50,
    },
    "early_steer": {
        "mid_steer_start_ratio": 0.25,
        "mid_steer_alpha_start": 0.7,
    },
    "late_steer": {
        "mid_steer_start_ratio": 0.70,
        "mid_steer_alpha_start": 0.5,
    },
    "border_relaxed": {
        "edit_support": "border_relaxed",
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explore random TCGA 1024 regions steered by top exported SAE concepts. "
            "Each sampled slide receives a fresh set of concepts."
        )
    )
    parser.add_argument("--export-dir", type=Path, default=Path("/common/users/wq50/wsi-sae/exports/tcga_uni2_sae_relu_v1"))
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/random_tcga_export_concepts_1024_sweep")
    parser.add_argument("--wsi-root", type=Path, default=Path("/research/projects/mllab/WSI"))
    parser.add_argument("--slides-root", type=Path, default=Path("/research/projects/mllab/WSI/.tmp/ready_buffer/slides"))
    parser.add_argument("--n-slides", type=int, default=10)
    parser.add_argument("--concepts-per-slide", type=int, default=5)
    parser.add_argument("--latent-strategies", type=str, default="top_activation,top_variance,top_sparsity")
    parser.add_argument("--representative-method", type=str, default="max_activation")
    parser.add_argument("--prototype-top-k", type=int, default=5)
    parser.add_argument(
        "--prefer-distinct-prototype-slides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer top representative tiles from different slides when building each concept prototype.",
    )
    parser.add_argument("--settings", type=str, default="default,loose_context,stronger_preserve,early_steer,border_relaxed")
    parser.add_argument(
        "--prototype-strengths",
        type=str,
        default="0.8",
        help="Comma-separated concept prototype strengths to sweep, e.g. 0.4,0.8,1.0.",
    )
    parser.add_argument("--region-size", type=int, default=1024)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument("--min-tissue", type=float, default=0.45)
    parser.add_argument("--min-dark-fraction", type=float, default=0.04)
    parser.add_argument("--min-saturation-fraction", type=float, default=0.04)
    parser.add_argument("--max-slide-tries", type=int, default=400)
    parser.add_argument("--max-region-tries-per-slide", type=int, default=300)
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--output-mode", type=str, default="minimal", choices=["minimal", "debug"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_csv_list(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def parse_float_csv(value: str) -> list[float]:
    vals = [float(token) for token in parse_csv_list(value)]
    if not vals:
        raise ValueError("Expected at least one float value")
    return vals


def strength_token(value: float) -> str:
    return f"s{float(value):g}".replace(".", "p").replace("-", "m")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(str(key))
                    fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_token(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p").replace("-", "_")


def read_h5_features_coords(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        feats = np.asarray(handle["features"][:], dtype=np.float32)
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if feats.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unsupported H5 shapes for {h5_path}: features={feats.shape}, coords={coords.shape}")
    return feats, coords


def infer_coord_tile_size(coords: np.ndarray, fallback: int = 512) -> int:
    arr = np.asarray(coords, dtype=np.int64)
    candidates: list[int] = []
    for axis in (0, 1):
        vals = np.unique(arr[:, axis])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    return int(min(candidates)) if candidates else int(fallback)


def build_cell_maps(coords: np.ndarray, tile_size_level0: int) -> tuple[dict[tuple[int, int], int], dict[int, tuple[int, int]]]:
    cell_to_index: dict[tuple[int, int], int] = {}
    index_to_cell: dict[int, tuple[int, int]] = {}
    for idx, (x, y) in enumerate(np.asarray(coords, dtype=np.int64).tolist()):
        cell = (int(round(int(x) / float(tile_size_level0))), int(round(int(y) / float(tile_size_level0))))
        cell_to_index[cell] = int(idx)
        index_to_cell[int(idx)] = cell
    return cell_to_index, index_to_cell


def effective_grid_step_at_target_magnification(*, tile_size_level0: int, objective_power: float, target_magnification: float) -> float:
    return float(tile_size_level0) * float(target_magnification) / max(float(objective_power), 1e-8)


def project_from_feature_relpath(feature_relpath: str) -> str:
    parts = Path(str(feature_relpath)).parts
    for part in parts:
        if str(part).startswith("TCGA-"):
            return str(part)
    raise ValueError(f"Could not infer TCGA project from feature_relpath={feature_relpath!r}")


def unique_existing_dirs(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        out.append(path)
    return out


def slide_search_dirs(args: argparse.Namespace, project: str) -> list[Path]:
    """Return the local slide roots we commonly use for TCGA experiments.

    The SAE export points at feature H5s, but raw slides may live in several
    local caches.  Keep this search explicit and shallow so random exploration
    does not accidentally recurse through huge archives.
    """
    project_short = str(project).replace("TCGA-", "")
    return unique_existing_dirs(
        [
            Path(args.slides_root) / project,
            Path(args.slides_root) / project / "slides",
            Path(args.wsi_root) / "TCGA_features" / project / "slides",
            Path(args.wsi_root) / "TCGA" / "store",
            Path(args.wsi_root) / "wsi_slides",
            Path("/common/users/wq50/CLAM") / f"{project_short}_slides",
            Path("/common/users/wq50/CLAM") / f"{project_short}_slides_high_conf",
        ]
    )


def find_slide_for_project(args: argparse.Namespace, project: str, slide_key: str) -> Path | None:
    for search_dir in slide_search_dirs(args, project):
        slide_path = find_slide_path(search_dir, slide_key)
        if slide_path is not None:
            return slide_path
    return None


def resolve_feature_path(wsi_root: Path, row: dict[str, str]) -> Path:
    rel = str(row.get("feature_relpath", "")).strip()
    if rel:
        return wsi_root / rel
    legacy = Path(str(row.get("legacy_h5_path", "")))
    return legacy


def load_concept_pool(args: argparse.Namespace) -> tuple[list[int], dict[int, list[dict[str, str]]], dict[int, dict[str, str]]]:
    rep_dir = args.export_dir / "representatives_test"
    latent_rows = read_csv_rows(rep_dir / "latent_summary.csv")
    support_rows = read_csv_rows(rep_dir / "representative_support_tiles.csv")
    strategies = set(parse_csv_list(args.latent_strategies))

    latent_meta: dict[int, dict[str, str]] = {}
    for row in latent_rows:
        if str(row.get("latent_strategy", "")) not in strategies:
            continue
        latent = int(row["latent_idx"])
        if latent not in latent_meta:
            latent_meta[latent] = row

    rows_by_latent: dict[int, list[dict[str, str]]] = {latent: [] for latent in latent_meta}
    for row in support_rows:
        latent = int(row["latent_idx"])
        if latent not in rows_by_latent:
            continue
        if str(row.get("latent_strategy", "")) != str(latent_meta[latent].get("latent_strategy", "")):
            continue
        if str(row.get("representative_method", "")) != str(args.representative_method):
            continue
        h5_path = resolve_feature_path(args.wsi_root, row)
        if h5_path.exists():
            rows_by_latent[latent].append(row)

    usable: list[int] = []
    for latent, rows in rows_by_latent.items():
        rows.sort(key=lambda row: (int(row.get("method_rank", 10**9)), -float(row.get("activation", 0.0))))
        if len(select_prototype_rows(args, rows)) >= int(args.prototype_top_k):
            usable.append(int(latent))
    if not usable:
        raise RuntimeError(f"No usable exported concepts found in {rep_dir}")
    return usable, rows_by_latent, latent_meta


def select_prototype_rows(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Pick representative rows for prototype construction.

    The export can contain the same tile through multiple latent-strategy views.
    For visual steering, a prototype built from duplicated tiles is brittle, so
    we first deduplicate exact tiles and then prefer different source slides.
    """
    top_k = int(args.prototype_top_k)
    deduped: list[dict[str, str]] = []
    seen_tiles: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(resolve_feature_path(args.wsi_root, row)), str(row.get("tile_index", "")))
        if key in seen_tiles:
            continue
        seen_tiles.add(key)
        deduped.append(row)

    if not bool(args.prefer_distinct_prototype_slides):
        return deduped[:top_k]

    chosen: list[dict[str, str]] = []
    seen_slides: set[str] = set()
    for row in deduped:
        slide_key = str(row.get("slide_key", ""))
        if slide_key in seen_slides:
            continue
        chosen.append(row)
        seen_slides.add(slide_key)
        if len(chosen) >= top_k:
            return chosen
    for row in deduped:
        if row in chosen:
            continue
        chosen.append(row)
        if len(chosen) >= top_k:
            break
    return chosen


def build_slide_pool(args: argparse.Namespace, support_rows_by_latent: dict[int, list[dict[str, str]]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    pool: list[dict[str, str]] = []
    for rows in support_rows_by_latent.values():
        for row in rows:
            h5_path = resolve_feature_path(args.wsi_root, row)
            if not h5_path.exists():
                continue
            slide_key = str(row["slide_key"])
            if slide_key in seen:
                continue
            seen.add(slide_key)
            project = project_from_feature_relpath(str(row.get("feature_relpath", "")))
            slide_path = find_slide_for_project(args, project, slide_key)
            if slide_path is None:
                continue
            pool.append(
                {
                    "slide_key": slide_key,
                    "case_id": str(row.get("case_id", slide_key[:12])),
                    "project_dir": project,
                    "h5_path": str(h5_path),
                    "slide_path": str(slide_path),
                }
            )
    return pool


def find_valid_4x4_starts(cell_to_index: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    cells = set(cell_to_index)
    starts: list[tuple[int, int]] = []
    for gx, gy in cells:
        ok = True
        for dy in range(4):
            for dx in range(4):
                if (gx + dx, gy + dy) not in cells:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            starts.append((int(gx), int(gy)))
    return starts


def sample_source_region(args: argparse.Namespace, slide_row: dict[str, str], out_dir: Path, rng: random.Random) -> dict[str, Any] | None:
    h5_path = Path(str(slide_row["h5_path"]))
    slide_path = Path(str(slide_row["slide_path"]))
    features, coords = read_h5_features_coords(h5_path)
    if int(features.shape[1]) != 1536:
        return None
    tile_size_level0 = infer_coord_tile_size(coords)
    cell_to_index, _ = build_cell_maps(coords, tile_size_level0)
    starts = find_valid_4x4_starts(cell_to_index)
    rng.shuffle(starts)
    slide = open_slide(slide_path)
    try:
        objective = infer_objective_power(slide)
        effective = effective_grid_step_at_target_magnification(
            tile_size_level0=int(tile_size_level0),
            objective_power=float(objective),
            target_magnification=float(args.target_magnification),
        )
        if abs(float(effective) - float(args.grid_step_px)) > 16.0:
            return None
        for gx0, gy0 in starts[: int(args.max_region_tries_per_slide)]:
            x0 = int(gx0) * int(tile_size_level0)
            y0 = int(gy0) * int(tile_size_level0)
            region_img, crop_w0, crop_h0 = read_region_rgb_at_magnification(
                slide,
                x0=x0,
                y0=y0,
                out_w=int(args.region_size),
                out_h=int(args.region_size),
                target_magnification=float(args.target_magnification),
            )
            quality = quick_region_quality_metrics(region_img)
            if quality["tissue_score"] < float(args.min_tissue):
                continue
            if quality["dark_fraction"] < float(args.min_dark_fraction):
                continue
            if quality["saturation_fraction"] < float(args.min_saturation_fraction):
                continue

            zgrid = np.zeros((4, 4, int(features.shape[1])), dtype=np.float32)
            for ly in range(4):
                for lx in range(4):
                    zgrid[ly, lx] = features[cell_to_index[(gx0 + lx, gy0 + ly)]]

            region_id = (
                f"{slide_row['slide_key']}__random1024__mag_{str(args.target_magnification).replace('.', 'p')}"
                f"__gx_{gx0}__gy_{gy0}"
            )
            region_dir = out_dir / "regions" / region_id
            region_dir.mkdir(parents=True, exist_ok=True)
            image_path = region_dir / "region.png"
            zgrid_path = region_dir / "region_zgrid.npy"
            cells_path = region_dir / "region_cells.png"
            overlay_path = region_dir / "edit_overlay.png"
            save_png(region_img, image_path)
            np.save(zgrid_path, zgrid)
            save_png(make_region_cells_preview(region_img, grid_step_px=int(args.grid_step_px)), cells_path)
            target_cells = [(1, 1), (2, 1), (1, 2), (2, 2)]
            save_png(draw_cells_overlay(region_img, cells=target_cells, grid_step_px=int(args.grid_step_px)), overlay_path)
            row = {
                "region_id": region_id,
                "split": "random_export",
                "label": 0,
                "hpv_status": "random_tcga",
                "case_id": str(slide_row["case_id"]),
                "slide_key": str(slide_row["slide_key"]),
                "slide_path": str(slide_path),
                "canonical_h5_path": str(h5_path),
                "region_x": int(x0),
                "region_y": int(y0),
                "region_w": int(args.region_size),
                "region_h": int(args.region_size),
                "grid_step_px": int(args.grid_step_px),
                "feature_dim": int(features.shape[1]),
                "tissue_score": float(quality["tissue_score"]),
                "seed": int(args.seed),
                "image_path": str(image_path),
                "feature_grid_path": str(zgrid_path),
                "cell_preview_path": str(cells_path),
                "region_dir": str(region_dir),
                "project_dir": str(slide_row["project_dir"]),
                "region_gx0": int(gx0),
                "region_gy0": int(gy0),
                "crop_w_level0": int(crop_w0),
                "crop_h_level0": int(crop_h0),
                "objective_power": float(objective),
                "tile_size_level0": int(tile_size_level0),
                "effective_grid_step_at_target_mag": float(effective),
                "edit_overlay_path": str(overlay_path),
            }
            write_json(region_dir / "region_meta.json", row)
            return row
    finally:
        slide.close()
    return None


def write_concept_files(
    *,
    args: argparse.Namespace,
    latent: int,
    concept_rank: int,
    rows_by_latent: dict[int, list[dict[str, str]]],
    latent_meta: dict[int, dict[str, str]],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_json = out_dir / f"concept_latent_{int(latent)}.json"
    reps_csv = out_dir / f"representatives_latent_{int(latent)}.csv"
    meta = latent_meta[int(latent)]
    concept = {
        "task": "export_random_concepts",
        "class_label": "export_concept",
        "latent_idx": int(latent),
        "concept_rank": int(concept_rank),
        "final_score": float(meta.get("activation_mean", meta.get("activation_max", 1.0)) or 1.0),
        "association_score": 1.0,
        "attention_support_score": 0.0,
        "steering_direction": "toward_export_concept",
        "latent_strategy": str(meta.get("latent_strategy", "")),
        "activation_max": float(meta.get("activation_max", 0.0) or 0.0),
        "activation_mean": float(meta.get("activation_mean", 0.0) or 0.0),
        "support_tile_count": int(float(meta.get("support_tile_count", 0) or 0)),
        "unique_slide_count": int(float(meta.get("unique_slide_count", 0) or 0)),
    }
    write_json(
        concept_json,
        {
            "task": "export_random_concepts",
            "class_label": "export_concept",
            "mode": "export_top_concepts",
            "concepts": [concept],
        },
    )
    rep_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(select_prototype_rows(args, rows_by_latent[int(latent)]), start=1):
        rep_rows.append(
            {
                "task": "export_random_concepts",
                "class_label": "export_concept",
                "latent_idx": int(latent),
                "ranking_method": "activation",
                "tile_rank": int(rank),
                "activation": float(row.get("activation", 0.0)),
                "attention": "",
                "attention_norm": "",
                "attention_weighted_activation": "",
                "case_id": str(row.get("case_id", "")),
                "slide_key": str(row.get("slide_key", "")),
                "project_dir": project_from_feature_relpath(str(row.get("feature_relpath", ""))),
                "label": "export_concept",
                "h5_path": str(resolve_feature_path(args.wsi_root, row)),
                "tile_index": int(row["tile_index"]),
                "coord_x": int(row["coord_x"]),
                "coord_y": int(row["coord_y"]),
                "latent_strategy": str(row.get("latent_strategy", "")),
                "representative_method": str(row.get("representative_method", "")),
            }
        )
    write_csv(reps_csv, rep_rows)
    return concept_json, reps_csv


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rng = random.Random(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "experiment_args.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:]))),
        },
    )
    settings = parse_csv_list(args.settings)
    for setting in settings:
        if setting not in SETTINGS:
            raise ValueError(f"Unknown setting {setting!r}. Available: {sorted(SETTINGS)}")
    prototype_strengths = parse_float_csv(args.prototype_strengths)

    latent_ids, rows_by_latent, latent_meta = load_concept_pool(args)
    rng.shuffle(latent_ids)
    needed = int(args.n_slides) * int(args.concepts_per_slide)
    if len(latent_ids) < needed:
        raise RuntimeError(f"Need {needed} unique concepts, found {len(latent_ids)}")

    slide_pool = build_slide_pool(args, rows_by_latent)
    rng.shuffle(slide_pool)
    selected_regions: list[dict[str, Any]] = []
    for slide_row in slide_pool[: int(args.max_slide_tries)]:
        if len(selected_regions) >= int(args.n_slides):
            break
        try:
            region = sample_source_region(args, slide_row, args.out_dir, rng)
        except Exception as exc:
            region = None
            print(f"[skip] {slide_row.get('slide_key')} {exc}", file=sys.stderr)
        if region is not None:
            region["slide_index"] = int(len(selected_regions) + 1)
            selected_regions.append(region)
            print(f"[region] {len(selected_regions)}/{args.n_slides}: {region['region_id']}", flush=True)
    if len(selected_regions) < int(args.n_slides):
        raise RuntimeError(f"Only found {len(selected_regions)} valid regions out of requested {args.n_slides}")

    write_csv(args.out_dir / "region_bank.csv", selected_regions)
    write_json(args.out_dir / "selected_regions.json", selected_regions)

    manifest_rows: list[dict[str, Any]] = []
    target_cells = [{"gx": 1, "gy": 1}, {"gx": 2, "gy": 1}, {"gx": 1, "gy": 2}, {"gx": 2, "gy": 2}]
    concept_cursor = 0
    for slide_idx, region in enumerate(selected_regions, start=1):
        assigned = latent_ids[concept_cursor : concept_cursor + int(args.concepts_per_slide)]
        concept_cursor += int(args.concepts_per_slide)
        for concept_offset, latent in enumerate(assigned, start=1):
            concept_rank = int(concept_cursor - int(args.concepts_per_slide) + concept_offset)
            concept_json, reps_csv = write_concept_files(
                args=args,
                latent=int(latent),
                concept_rank=concept_rank,
                rows_by_latent=rows_by_latent,
                latent_meta=latent_meta,
                out_dir=args.out_dir / "_concepts" / f"latent_{int(latent)}",
            )
            for setting_name in settings:
                preset = dict(SETTINGS["default"])
                preset.update(SETTINGS[setting_name])
                for prototype_strength in prototype_strengths:
                    strength_id = strength_token(float(prototype_strength))
                    run_id = (
                        f"slide_{slide_idx:02d}__concept_{concept_offset:02d}"
                        f"__latent_{int(latent)}__{setting_name}__{strength_id}"
                    )
                    run_dir = args.out_dir / "runs" / run_id
                    input_dir = run_dir / "_inputs"
                    input_dir.mkdir(parents=True, exist_ok=True)
                    bank_path = input_dir / "region_bank.csv"
                    edit_manifest_path = input_dir / "progressive_edit_manifest.json"
                    write_csv(bank_path, [region])
                    write_json(
                        edit_manifest_path,
                        [
                            {
                                "run_id": run_id,
                                "region_id": str(region["region_id"]),
                                "target_cells": target_cells,
                                "selector": "fixed_center_2x2",
                                "latent_idx": int(latent),
                                "latent_strategy": str(latent_meta[int(latent)].get("latent_strategy", "")),
                                "setting": setting_name,
                                "prototype_strength": float(prototype_strength),
                            }
                        ],
                    )
                    generated_path = run_dir / run_id / "generated.png"
                    cmd = [
                        sys.executable,
                        str(SCRIPT_DIR / "run_progressive_region_edit.py"),
                        "--task",
                        "export_random_concepts",
                        "--region-bank-csv",
                        str(bank_path),
                        "--edit-manifest",
                        str(edit_manifest_path),
                        "--out-dir",
                        str(run_dir),
                        "--concepts-json",
                        str(concept_json),
                        "--representative-tiles-csv",
                        str(reps_csv),
                        "--concept-class-label",
                        "export_concept",
                        "--concept-ranking-method",
                        "activation",
                        "--concept-target-stat",
                        "median",
                        "--concept-target-top-k",
                        str(args.prototype_top_k),
                        "--target-magnification",
                        str(args.target_magnification),
                        "--sae-variant",
                        str(args.sae_variant),
                        "--prototype-strength",
                        str(float(prototype_strength)),
                        "--steer-blend",
                        "1.0",
                        "--preserve-edit-strength",
                        str(preset["preserve_edit_strength"]),
                        "--preserve-visited-strength",
                        str(preset["preserve_visited_strength"]),
                        "--preserve-fresh-context-strength",
                        str(preset["preserve_fresh_context_strength"]),
                        "--mid-steer-start-ratio",
                        str(preset["mid_steer_start_ratio"]),
                        "--mid-steer-end-ratio",
                        str(preset["mid_steer_end_ratio"]),
                        "--mid-steer-alpha-start",
                        str(preset["mid_steer_alpha_start"]),
                        "--mid-steer-alpha-end",
                        str(preset["mid_steer_alpha_end"]),
                        "--mid-steer-alpha-schedule",
                        "linear",
                        "--edit-support",
                        str(preset["edit_support"]),
                        "--steps",
                        str(args.steps),
                        "--guidance",
                        str(args.guidance),
                        "--patch-batch",
                        str(args.patch_batch),
                        "--output-mode",
                        str(args.output_mode),
                        "--seed",
                        str(args.seed),
                        "--device",
                        str(args.device),
                    ]
                    row = {
                        "run_id": run_id,
                        "slide_index": int(slide_idx),
                        "concept_index_for_slide": int(concept_offset),
                        "region_id": str(region["region_id"]),
                        "slide_key": str(region["slide_key"]),
                        "project_dir": str(region["project_dir"]),
                        "latent_idx": int(latent),
                        "latent_strategy": str(latent_meta[int(latent)].get("latent_strategy", "")),
                        "setting": setting_name,
                        "prototype_strength": float(prototype_strength),
                        "preserve_edit_strength": float(preset["preserve_edit_strength"]),
                        "preserve_visited_strength": float(preset["preserve_visited_strength"]),
                        "preserve_fresh_context_strength": float(preset["preserve_fresh_context_strength"]),
                        "mid_steer_start_ratio": float(preset["mid_steer_start_ratio"]),
                        "mid_steer_end_ratio": float(preset["mid_steer_end_ratio"]),
                        "mid_steer_alpha_start": float(preset["mid_steer_alpha_start"]),
                        "mid_steer_alpha_end": float(preset["mid_steer_alpha_end"]),
                        "edit_support": str(preset["edit_support"]),
                        "source_region": str(region["image_path"]),
                        "edit_overlay": str(region["edit_overlay_path"]),
                        "generated_path": str(generated_path),
                        "concept_json": str(concept_json),
                        "representative_tiles_csv": str(reps_csv),
                        "command": " ".join(shlex.quote(part) for part in cmd),
                    }
                    manifest_rows.append(row)
                    write_csv(args.out_dir / "sweep_manifest.csv", manifest_rows)
                    if bool(args.dry_run):
                        continue
                    if not (bool(args.skip_existing) and generated_path.exists()):
                        print(f"[run] {run_id}", flush=True)
                        subprocess.run(cmd, check=True, cwd=str(WSI_CF_ROOT))
                    flat_dir = args.out_dir / "browse" / f"slide_{slide_idx:02d}" / f"latent_{int(latent)}" / setting_name / strength_id
                    flat_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(region["image_path"], flat_dir / "source_region_actual.png")
                    shutil.copy2(region["edit_overlay_path"], flat_dir / "edit_overlay.png")
                    if generated_path.exists():
                        shutil.copy2(generated_path, flat_dir / "generated.png")

    write_csv(args.out_dir / "sweep_manifest.csv", manifest_rows)
    write_json(
        args.out_dir / "summary.json",
        {
            "n_slides": int(len(selected_regions)),
            "concepts_per_slide": int(args.concepts_per_slide),
            "settings": settings,
            "prototype_strengths": prototype_strengths,
            "runs_planned": int(len(manifest_rows)),
            "dry_run": bool(args.dry_run),
            "export_dir": str(args.export_dir),
            "sae_variant": str(args.sae_variant),
            "region_bank_csv": str(args.out_dir / "region_bank.csv"),
            "sweep_manifest_csv": str(args.out_dir / "sweep_manifest.csv"),
        },
    )
    print(json.dumps(json.loads((args.out_dir / "summary.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
