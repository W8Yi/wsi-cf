from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from wsi_cf.common.io import save_png, write_json
from wsi_cf.data.donor_pool import canonical_h5_path, load_split_rows, make_contact_sheet
from wsi_cf.data.slides import find_slide_path, quick_tissue_score, read_region_rgb, read_region_rgb_at_magnification


@dataclass(frozen=True)
class RegionBankRow:
    region_id: str
    split: str
    label: int
    hpv_status: str
    case_id: str
    slide_key: str
    slide_path: str
    canonical_h5_path: str
    region_x: int
    region_y: int
    region_w: int
    region_h: int
    grid_step_px: int
    feature_dim: int
    tissue_score: float
    seed: int
    image_path: str
    feature_grid_path: str
    cell_preview_path: str
    region_dir: str = ""


@dataclass(frozen=True)
class RegionRoleRow(RegionBankRow):
    role: str = ""
    role_rank: int = -1


def load_local_labeled_rows(*, split_tsv: Path, slides_dir: Path, features_dir: Path) -> list[dict[str, Any]]:
    rows = load_split_rows(split_tsv, split_filter="all")
    out: list[dict[str, Any]] = []
    for row in rows:
        slide_key = str(row["slide_key"])
        slide_path = find_slide_path(slides_dir, slide_key)
        h5_path = canonical_h5_path(features_dir, slide_key)
        if slide_path is None or not h5_path.exists():
            continue
        item = dict(row)
        item["slide_path"] = str(slide_path)
        item["canonical_h5_path"] = str(h5_path)
        out.append(item)
    out.sort(key=lambda r: (int(r["label"]), str(r["slide_key"])))
    return out


def sample_random_tissue_region(
    slide: Any,
    *,
    region_size: int,
    n_tries: int,
    min_tissue: float,
    rng: random.Random,
) -> tuple[int, int, float]:
    width, height = slide.dimensions
    if width <= region_size or height <= region_size:
        x0 = 0
        y0 = 0
        img = read_region_rgb(slide, x0, y0, int(region_size), int(region_size))
        return x0, y0, quick_tissue_score(img)

    best = (0, 0, -1.0)
    for _ in range(max(1, n_tries)):
        x0 = rng.randint(0, max(0, width - region_size))
        y0 = rng.randint(0, max(0, height - region_size))
        img = read_region_rgb(slide, x0, y0, int(region_size), int(region_size))
        score = quick_tissue_score(img)
        if score > best[2]:
            best = (x0, y0, score)
        if score >= min_tissue:
            return x0, y0, score
    return best


def sample_random_tissue_region_at_magnification(
    slide: Any,
    *,
    out_size: int,
    target_magnification: float,
    n_tries: int,
    min_tissue: float,
    rng: random.Random,
) -> tuple[int, int, float, int, int]:
    width, height = slide.dimensions
    img0, crop_w0, crop_h0 = read_region_rgb_at_magnification(
        slide,
        x0=0,
        y0=0,
        out_w=int(out_size),
        out_h=int(out_size),
        target_magnification=float(target_magnification),
    )
    if width <= crop_w0 or height <= crop_h0:
        return 0, 0, quick_tissue_score(img0), int(crop_w0), int(crop_h0)

    best = (0, 0, -1.0, int(crop_w0), int(crop_h0))
    for _ in range(max(1, n_tries)):
        x0 = rng.randint(0, max(0, width - int(crop_w0)))
        y0 = rng.randint(0, max(0, height - int(crop_h0)))
        img, crop_w, crop_h = read_region_rgb_at_magnification(
            slide,
            x0=int(x0),
            y0=int(y0),
            out_w=int(out_size),
            out_h=int(out_size),
            target_magnification=float(target_magnification),
        )
        score = quick_tissue_score(img)
        if score > best[2]:
            best = (x0, y0, score, int(crop_w), int(crop_h))
        if score >= float(min_tissue):
            return x0, y0, score, int(crop_w), int(crop_h)
    return best


def make_region_cells_preview(img: Image.Image, *, grid_step_px: int) -> Image.Image:
    width, height = img.size
    grid_w = math.ceil(width / float(grid_step_px))
    grid_h = math.ceil(height / float(grid_step_px))
    canvas = Image.new("RGB", (grid_w * grid_step_px, grid_h * grid_step_px), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    idx = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            x0 = gx * grid_step_px
            y0 = gy * grid_step_px
            patch = img.crop((x0, y0, min(width, x0 + grid_step_px), min(height, y0 + grid_step_px)))
            if patch.size != (grid_step_px, grid_step_px):
                bg = Image.new("RGB", (grid_step_px, grid_step_px), (255, 255, 255))
                bg.paste(patch, (0, 0))
                patch = bg
            canvas.paste(patch, (x0, y0))
            draw.rectangle([x0, y0, x0 + grid_step_px - 1, y0 + grid_step_px - 1], outline=(180, 180, 180), width=1)
            draw.text((x0 + 6, y0 + 6), str(idx), fill=(255, 255, 0))
            idx += 1
    return canvas


def region_feature_grid_shape(*, region_size: int, grid_step_px: int, feature_dim: int) -> tuple[int, int, int]:
    side = int(math.ceil(region_size / float(grid_step_px)))
    return side, side, int(feature_dim)


def export_region_bundle(
    *,
    region_img: Image.Image,
    out_dir: Path,
    region_id: str,
    row: dict[str, Any],
    region_x: int,
    region_y: int,
    region_size: int,
    grid_step_px: int,
    tissue_score: float,
    seed: int,
    build_feature_grid: Callable[[Image.Image], np.ndarray],
) -> dict[str, Any]:
    region_dir = out_dir / f"label_{int(row['label'])}_{'hpv_pos' if int(row['label']) == 1 else 'hpv_neg'}" / region_id
    region_dir.mkdir(parents=True, exist_ok=True)

    image_path = region_dir / "region.png"
    feature_grid_path = region_dir / "region_zgrid.npy"
    cell_preview_path = region_dir / "region_cells.png"
    meta_path = region_dir / "region_meta.json"

    save_png(region_img, image_path)
    z_grid = np.asarray(build_feature_grid(region_img), dtype=np.float32)
    np.save(feature_grid_path, z_grid)
    cell_preview = make_region_cells_preview(region_img, grid_step_px=int(grid_step_px))
    cell_preview.save(cell_preview_path)

    feature_dim = int(z_grid.shape[-1]) if z_grid.ndim == 3 else -1
    meta = {
        "region_id": str(region_id),
        "split": str(row["split"]),
        "label": int(row["label"]),
        "hpv_status": str(row["hpv_status"]),
        "case_id": str(row["case_id"]),
        "slide_key": str(row["slide_key"]),
        "slide_path": str(row["slide_path"]),
        "canonical_h5_path": str(row["canonical_h5_path"]),
        "region_x": int(region_x),
        "region_y": int(region_y),
        "region_w": int(region_size),
        "region_h": int(region_size),
        "grid_step_px": int(grid_step_px),
        "feature_dim": int(feature_dim),
        "tissue_score": float(tissue_score),
        "seed": int(seed),
        "image_path": str(image_path),
        "feature_grid_path": str(feature_grid_path),
        "cell_preview_path": str(cell_preview_path),
    }
    write_json(meta_path, meta)
    meta["region_dir"] = str(region_dir)
    return meta


def write_region_bank_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def parse_region_bank_csv(csv_path: Path) -> list[RegionBankRow]:
    out: list[RegionBankRow] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out.append(
                RegionBankRow(
                    region_id=str(row["region_id"]),
                    split=str(row["split"]),
                    label=int(row["label"]),
                    hpv_status=str(row["hpv_status"]),
                    case_id=str(row["case_id"]),
                    slide_key=str(row["slide_key"]),
                    slide_path=str(row["slide_path"]),
                    canonical_h5_path=str(row["canonical_h5_path"]),
                    region_x=int(row["region_x"]),
                    region_y=int(row["region_y"]),
                    region_w=int(row["region_w"]),
                    region_h=int(row["region_h"]),
                    grid_step_px=int(row["grid_step_px"]),
                    feature_dim=int(row["feature_dim"]),
                    tissue_score=float(row["tissue_score"]),
                    seed=int(row["seed"]),
                    image_path=str(row["image_path"]),
                    feature_grid_path=str(row["feature_grid_path"]),
                    cell_preview_path=str(row["cell_preview_path"]),
                    region_dir=str(row.get("region_dir", "")),
                )
            )
    return out


def parse_region_roles_csv(csv_path: Path) -> list[RegionRoleRow]:
    out: list[RegionRoleRow] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out.append(
                RegionRoleRow(
                    region_id=str(row["region_id"]),
                    split=str(row["split"]),
                    label=int(row["label"]),
                    hpv_status=str(row["hpv_status"]),
                    case_id=str(row["case_id"]),
                    slide_key=str(row["slide_key"]),
                    slide_path=str(row["slide_path"]),
                    canonical_h5_path=str(row["canonical_h5_path"]),
                    region_x=int(row["region_x"]),
                    region_y=int(row["region_y"]),
                    region_w=int(row["region_w"]),
                    region_h=int(row["region_h"]),
                    grid_step_px=int(row["grid_step_px"]),
                    feature_dim=int(row["feature_dim"]),
                    tissue_score=float(row["tissue_score"]),
                    seed=int(row["seed"]),
                    image_path=str(row["image_path"]),
                    feature_grid_path=str(row["feature_grid_path"]),
                    cell_preview_path=str(row["cell_preview_path"]),
                    region_dir=str(row.get("region_dir", "")),
                    role=str(row["role"]),
                    role_rank=int(row["role_rank"]),
                )
            )
    return out


def assign_region_roles(
    rows: list[dict[str, Any]],
    *,
    sources_per_label: int,
    donors_per_label: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label"])].append(dict(row))
    out: list[dict[str, Any]] = []
    for label in (0, 1):
        label_rows = sorted(grouped[label], key=lambda r: (str(r["slide_key"]), str(r["region_id"])))
        need = int(sources_per_label) + int(donors_per_label)
        if len(label_rows) < need:
            raise ValueError(f"Need {need} regions for label={label}, found {len(label_rows)}")
        for idx, row in enumerate(label_rows):
            role = ""
            if idx < int(sources_per_label):
                role = "source"
            elif idx < need:
                role = "donor"
            if role:
                rec = dict(row)
                rec["role"] = role
                rec["role_rank"] = idx if role == "source" else idx - int(sources_per_label)
                out.append(rec)
    out.sort(key=lambda r: (int(r["label"]), str(r["role"]), int(r["role_rank"]), str(r["slide_key"])))
    return out


def build_region_pairs(rows: list[RegionRoleRow]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, list[RegionRoleRow]]] = {
        0: {"source": [], "donor": []},
        1: {"source": [], "donor": []},
    }
    for row in rows:
        if row.role not in {"source", "donor"}:
            continue
        grouped[int(row.label)][str(row.role)].append(row)

    for label in (0, 1):
        for role in ("source", "donor"):
            grouped[label][role] = sorted(grouped[label][role], key=lambda r: (int(r.role_rank), str(r.slide_key), str(r.region_id)))

    pairs: list[dict[str, Any]] = []
    for label in (0, 1):
        sources = grouped[label]["source"]
        donors_same = grouped[label]["donor"]
        donors_cross = grouped[1 - label]["donor"]
        if len(sources) != len(donors_same) or len(sources) != len(donors_cross):
            raise ValueError(
                f"Role counts must match for deterministic pairing: label={label} "
                f"sources={len(sources)} same_donors={len(donors_same)} cross_donors={len(donors_cross)}"
            )
        for idx, source in enumerate(sources):
            same = donors_same[idx]
            cross = donors_cross[idx]
            pairs.append(
                {
                    "source_label": int(source.label),
                    "source_region_id": str(source.region_id),
                    "source_slide_key": str(source.slide_key),
                    "source_role_rank": int(source.role_rank),
                    "source_image_path": str(source.image_path),
                    "source_feature_grid_path": str(source.feature_grid_path),
                    "same_label_donor_region_id": str(same.region_id),
                    "same_label_donor_slide_key": str(same.slide_key),
                    "same_label_donor_image_path": str(same.image_path),
                    "same_label_donor_feature_grid_path": str(same.feature_grid_path),
                    "cross_label_donor_region_id": str(cross.region_id),
                    "cross_label_donor_slide_key": str(cross.slide_key),
                    "cross_label_donor_image_path": str(cross.image_path),
                    "cross_label_donor_feature_grid_path": str(cross.feature_grid_path),
                }
            )
    pairs.sort(key=lambda r: (int(r["source_label"]), int(r["source_role_rank"]), str(r["source_slide_key"])))
    return pairs


def build_experiment_manifest(
    pairs: list[dict[str, Any]],
    *,
    conditions: list[str] | None = None,
    one_cell_gx: int = 1,
    one_cell_gy: int = 1,
) -> list[dict[str, Any]]:
    requested = conditions or [
        "baseline",
        "same_label_one_cell",
        "cross_label_one_cell",
        "same_label_full_grid",
        "cross_label_full_grid",
    ]
    out: list[dict[str, Any]] = []
    for pair in pairs:
        for condition in requested:
            row = dict(pair)
            row["condition"] = condition
            row["steer_mode"] = "none"
            row["donor_region_id"] = ""
            row["donor_feature_grid_path"] = ""
            row["donor_cell_gx"] = ""
            row["donor_cell_gy"] = ""
            if condition == "baseline":
                pass
            elif condition == "same_label_one_cell":
                row["steer_mode"] = "one_cell"
                row["donor_region_id"] = str(pair["same_label_donor_region_id"])
                row["donor_feature_grid_path"] = str(pair["same_label_donor_feature_grid_path"])
                row["donor_cell_gx"] = int(one_cell_gx)
                row["donor_cell_gy"] = int(one_cell_gy)
            elif condition == "cross_label_one_cell":
                row["steer_mode"] = "one_cell"
                row["donor_region_id"] = str(pair["cross_label_donor_region_id"])
                row["donor_feature_grid_path"] = str(pair["cross_label_donor_feature_grid_path"])
                row["donor_cell_gx"] = int(one_cell_gx)
                row["donor_cell_gy"] = int(one_cell_gy)
            elif condition == "same_label_full_grid":
                row["steer_mode"] = "full_grid"
                row["donor_region_id"] = str(pair["same_label_donor_region_id"])
                row["donor_feature_grid_path"] = str(pair["same_label_donor_feature_grid_path"])
            elif condition == "cross_label_full_grid":
                row["steer_mode"] = "full_grid"
                row["donor_region_id"] = str(pair["cross_label_donor_region_id"])
                row["donor_feature_grid_path"] = str(pair["cross_label_donor_feature_grid_path"])
            else:
                raise ValueError(f"Unsupported condition: {condition}")
            out.append(row)
    return out


def validate_balanced_request(*, eligible_rows: list[dict[str, Any]], total_regions: int, regions_per_slide: int) -> None:
    if int(total_regions) % 2 != 0:
        raise ValueError("total_regions must be even for balanced export")
    need_per_label = int(total_regions) // 2
    grouped = {0: 0, 1: 0}
    for row in eligible_rows:
        grouped[int(row["label"])] += 1
    for label in (0, 1):
        max_regions = grouped[label] * int(regions_per_slide)
        if max_regions < need_per_label:
            raise ValueError(
                f"Cannot satisfy balanced export for label={label}: need {need_per_label}, "
                f"have capacity {max_regions} from {grouped[label]} eligible slides."
            )


def make_region_bank_summary(
    rows: list[dict[str, Any]],
    *,
    roles: list[dict[str, Any]],
    out_csv: Path,
    out_roles_csv: Path,
) -> dict[str, Any]:
    label_counts = {str(label): sum(1 for row in rows if int(row["label"]) == label) for label in (0, 1)}
    role_counts = {
        "source_label_0": sum(1 for row in roles if row["role"] == "source" and int(row["label"]) == 0),
        "source_label_1": sum(1 for row in roles if row["role"] == "source" and int(row["label"]) == 1),
        "donor_label_0": sum(1 for row in roles if row["role"] == "donor" and int(row["label"]) == 0),
        "donor_label_1": sum(1 for row in roles if row["role"] == "donor" and int(row["label"]) == 1),
    }
    return {
        "n_regions": len(rows),
        "label_counts": label_counts,
        "role_counts": role_counts,
        "region_bank_csv": str(out_csv),
        "region_roles_csv": str(out_roles_csv),
    }


def save_label_contact_sheets(*, rows: list[dict[str, Any]], out_dir: Path) -> None:
    paths_by_label: dict[int, list[Path]] = {0: [], 1: []}
    for row in rows:
        paths_by_label[int(row["label"])].append(Path(str(row["image_path"])))
    for label, paths in paths_by_label.items():
        sheet = make_contact_sheet(paths)
        sheet.save(out_dir / f"label_{label}_contact_sheet.png")
