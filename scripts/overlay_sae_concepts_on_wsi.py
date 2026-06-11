#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import random
import shlex
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png, write_json


PRIORITY_LABELS: tuple[tuple[str, str], ...] = (
    ("luad_lusc", "LUAD"),
    ("luad_lusc", "LUSC"),
    ("tumor_purity_low_high", "high"),
    ("tumor_purity_low_high", "low"),
    ("kirc_low_vs_high_grade", "high"),
    ("kirc_low_vs_high_grade", "low"),
)

LABEL_COLORS: dict[tuple[str, str], str] = {
    ("luad_lusc", "LUAD"): "#2f6bff",
    ("luad_lusc", "LUSC"): "#e24a33",
    ("tumor_purity_low_high", "high"): "#d936d0",
    ("tumor_purity_low_high", "low"): "#00a6c8",
    ("kirc_low_vs_high_grade", "high"): "#7b4cc2",
    ("kirc_low_vs_high_grade", "low"): "#2ca25f",
    ("cancer_type_all_tcga", "TCGA-LUAD"): "#4e79a7",
    ("cancer_type_all_tcga", "TCGA-LUSC"): "#f28e2b",
    ("cancer_type_all_tcga", "TCGA-KIRC"): "#59a14f",
    ("msi_coad_stad", "MSI"): "#edc948",
}

DEFAULT_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
)

CONCEPT_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#e6550d",
    "#31a354",
    "#756bb1",
    "#636363",
    "#9edae5",
    "#c7e9c0",
    "#fdd0a2",
    "#dadaeb",
    "#f7b6d2",
)

DEFAULT_SLIDE_STORE_ROOT = Path("/research/projects/mllab/WSI/TCGA/store")
DEFAULT_FEATURES_ROOT = Path("/research/projects/mllab/WSI/TCGA_features")
DEFAULT_ALL_SAE_CKPT = WSI_CF_ROOT / "resources/models/sae/tcga_uni2_sae_relu_v1/relu_final.pt"
DEFAULT_ALL_SAE_CFG = WSI_CF_ROOT / "resources/models/sae/tcga_uni2_sae_relu_v1/run_config.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay SAE concept activations on WSI thumbnails. By default, every latent in one SAE "
            "is eligible and each feature-backed tile is colored by its most activated latent."
        )
    )
    parser.add_argument("--slide-path", type=Path, default=None, help="Path to one sample WSI, for example .svs.")
    parser.add_argument("--h5-path", type=Path, default=None, help="Matching UNI2 feature H5 with features and coords.")
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "paper_example/sae_concept_wsi_overlay")
    parser.add_argument("--slide-store-root", type=Path, default=DEFAULT_SLIDE_STORE_ROOT)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument(
        "--sample-random-slides",
        type=int,
        default=10,
        help="Sample this many matched slide/H5 pairs when --slide-path/--h5-path are not both supplied.",
    )
    parser.add_argument(
        "--concept-source",
        type=str,
        default="all_sae",
        choices=["all_sae", "curated"],
        help="Use every latent in one SAE, or the curated morphology concept exports.",
    )
    parser.add_argument("--sae-ckpt", type=Path, default=DEFAULT_ALL_SAE_CKPT)
    parser.add_argument("--sae-cfg", type=Path, default=DEFAULT_ALL_SAE_CFG)
    parser.add_argument("--max-legend-concepts", type=int, default=80)
    parser.add_argument("--concept-review-root", type=Path, default=WSI_CF_ROOT / "artifacts/morphology_label_concept_review")
    parser.add_argument("--concept-set", type=str, default="all_curated", choices=["all_curated", "priority"])
    parser.add_argument(
        "--label-filter",
        action="append",
        default=[],
        help="Optional task/label filter. Can be supplied multiple times, e.g. luad_lusc/LUAD.",
    )
    parser.add_argument("--top-concepts-per-label", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--thumbnail-max-side", type=int, default=4096)
    parser.add_argument("--vmin-percentile", type=float, default=50.0)
    parser.add_argument("--vmax-percentile", type=float, default=99.0)
    parser.add_argument("--winner-score-mode", type=str, default="raw", choices=["raw", "normalized"])
    parser.add_argument("--overlay-alpha", type=int, default=180)
    parser.add_argument("--min-normalized-activation", type=float, default=0.05)
    parser.add_argument("--max-tiles-per-concept", type=int, default=40)
    parser.add_argument("--save-top-tiles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-tile-size", type=int, default=256)
    parser.add_argument(
        "--tile-size-level0",
        type=int,
        default=0,
        help="Override H5 coordinate tile size in level-0 pixels. By default it is inferred from coords.",
    )
    parser.add_argument(
        "--save-vector-panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save PDF/SVG wrappers for combined and per-label overlays.",
    )
    return parser


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve_path(path: str | Path, *, base: Path = WSI_CF_ROOT) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if (base / p).exists():
        return base / p
    return p


def sanitize_component(value: str) -> str:
    cleaned = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    return out or "label"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = str(value).strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {value!r}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def infer_coord_tile_size(coords: np.ndarray, *, fallback: int) -> int:
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return int(fallback)
    candidates: list[int] = []
    for axis in (0, 1):
        vals = np.unique(arr[:, axis])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    return int(min(candidates)) if candidates else int(fallback)


def scaled_thumbnail_size(width: int, height: int, max_side: int) -> tuple[int, int, float, float]:
    scale = min(float(max_side) / float(max(1, width)), float(max_side) / float(max(1, height)), 1.0)
    out_w = max(1, int(round(float(width) * scale)))
    out_h = max(1, int(round(float(height) * scale)))
    return out_w, out_h, float(out_w) / float(max(1, width)), float(out_h) / float(max(1, height))


def tile_bounds_on_thumbnail(
    coord: np.ndarray | tuple[int, int] | list[int],
    *,
    tile_size_level0: int,
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y = int(coord[0]), int(coord[1])
    w = max(1, int(width))
    h = max(1, int(height))
    x0 = max(0, min(w - 1, int(round(float(x) * float(scale_x)))))
    y0 = max(0, min(h - 1, int(round(float(y) * float(scale_y)))))
    x1 = max(0, min(w, int(round(float(x + int(tile_size_level0)) * float(scale_x)))))
    y1 = max(0, min(h, int(round(float(y + int(tile_size_level0)) * float(scale_y)))))
    return x0, y0, min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))


def normalize_scores(values: np.ndarray, *, vmin_percentile: float, vmax_percentile: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(finite, float(vmin_percentile)))
    hi = float(np.percentile(finite, float(vmax_percentile)))
    if hi <= lo:
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def validate_feature_coord_shapes(features: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feat = np.asarray(features)
    coord = np.asarray(coords)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    if coord.ndim == 3 and coord.shape[0] == 1:
        coord = coord[0]
    if feat.ndim != 2:
        raise ValueError(f"features must be 2D [n_tiles, d], got shape {feat.shape}")
    if coord.ndim != 2 or coord.shape[1] < 2:
        raise ValueError(f"coords must be 2D [n_tiles, 2+], got shape {coord.shape}")
    coord = coord[:, :2]
    if feat.shape[0] != coord.shape[0]:
        raise ValueError(f"features/coords length mismatch: {feat.shape[0]} vs {coord.shape[0]}")
    return feat.astype(np.float32, copy=False), coord.astype(np.int64, copy=False)


def read_h5_features_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{path} is missing dataset 'features'")
        if "coords" not in handle:
            raise KeyError(f"{path} is missing dataset 'coords'")
        features = np.asarray(handle["features"])
        coords = np.asarray(handle["coords"])
    return validate_feature_coord_shapes(features, coords)


def color_for_label(task: str, label: str, index: int) -> str:
    key = (str(task), str(label))
    if key in LABEL_COLORS:
        return LABEL_COLORS[key]
    return DEFAULT_COLORS[int(index) % len(DEFAULT_COLORS)]


def color_for_concept_index(index: int) -> str:
    if int(index) < len(CONCEPT_COLORS):
        return CONCEPT_COLORS[int(index)]
    hue = (0.618033988749895 * float(index)) % 1.0
    sat = 0.62 + 0.24 * (((int(index) // 17) % 3) / 2.0)
    val = 0.72 + 0.20 * (((int(index) // 43) % 3) / 2.0)
    rgb = colorsys.hsv_to_rgb(hue, min(float(sat), 0.9), min(float(val), 0.95))
    return "#" + "".join(f"{int(round(channel * 255)):02x}" for channel in rgb)


def load_sae_latent_dim(sae_cfg: Path) -> int:
    cfg = read_json(resolve_path(sae_cfg))
    if "latent_dim" not in cfg:
        raise KeyError(f"SAE config is missing latent_dim: {sae_cfg}")
    return int(cfg["latent_dim"])


def load_all_sae_concepts(*, sae_ckpt: Path, sae_cfg: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ckpt = resolve_path(sae_ckpt)
    cfg = resolve_path(sae_cfg)
    latent_dim = load_sae_latent_dim(cfg)
    manifest = {
        "schema_version": "all_sae_latents.v1",
        "task": "all_sae",
        "class_label": "all_latents",
        "feature_space": {"name": "UNI2"},
        "sae": {
            "checkpoint": str(ckpt),
            "config": str(cfg),
            "latent_dim": int(latent_dim),
        },
        "tile_extraction": {
            "target_magnification": 20.0,
            "tile_size_px": 256,
            "coord_space": "level0_h5_coords",
        },
    }
    concepts = [
        {
            "concept_rank": str(latent_idx + 1),
            "latent_idx": str(latent_idx),
            "class_label": "all_latents",
            "steering_direction": "activation",
            "final_score": "",
            "association_score": "",
            "cohen_d": "",
        }
        for latent_idx in range(int(latent_dim))
    ]
    labels = [
        {
            "task": "all_sae",
            "class_label": "all_latents",
            "priority": "",
            "verdict": "all_sae",
            "reason": "Every latent in the selected SAE is eligible as a per-tile winner.",
            "concept_export_dir": "",
            "manifest_path": "",
            "manifest": manifest,
            "color": "#1f77b4",
            "concepts": concepts,
        }
    ]
    return labels, {"all_sae/all_latents": manifest}


def slide_key_from_svs(path: Path) -> str:
    return path.name.split(".", 1)[0]


def h5_project_name(path: Path) -> str:
    if path.parent.name == "features_uni2":
        return path.parent.parent.name
    return ""


def find_matched_slide_h5_pairs(*, slide_store_root: Path, features_root: Path) -> list[dict[str, Any]]:
    slide_paths = sorted(Path(slide_store_root).glob("*.svs"))
    slide_by_key = {slide_key_from_svs(path): path for path in slide_paths}
    pairs: list[dict[str, Any]] = []
    for h5_path in sorted(Path(features_root).glob("*/features_uni2/*.h5")):
        slide_key = h5_path.stem
        slide_path = slide_by_key.get(slide_key)
        if slide_path is None:
            continue
        pairs.append(
            {
                "slide_key": slide_key,
                "project": h5_project_name(h5_path),
                "slide_path": slide_path,
                "h5_path": h5_path,
            }
        )
    return pairs


def choose_slide_h5_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.slide_path is not None or args.h5_path is not None:
        if args.slide_path is None or args.h5_path is None:
            raise ValueError("--slide-path and --h5-path must be supplied together.")
        return [
            {
                "slide_key": slide_key_from_svs(Path(args.slide_path)),
                "project": h5_project_name(Path(args.h5_path)),
                "slide_path": Path(args.slide_path),
                "h5_path": Path(args.h5_path),
            }
        ]

    pairs = find_matched_slide_h5_pairs(slide_store_root=Path(args.slide_store_root), features_root=Path(args.features_root))
    if not pairs:
        raise RuntimeError(
            "No matched TCGA slide/H5 pairs found. Expected slides at "
            f"{args.slide_store_root}/<slide_key>.svs and features at "
            f"{args.features_root}/<project>/features_uni2/<slide_key>.h5."
        )
    rng = random.Random(int(args.seed))
    rng.shuffle(pairs)
    return pairs[: max(1, int(args.sample_random_slides))]


def assign_concept_colors(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label in labels:
        for concept in label["concepts"]:
            records.append(
                {
                    "concept_id": "",
                    "task": label["task"],
                    "class_label": label["class_label"],
                    "concept_rank": int(float(concept.get("concept_rank") or 0)),
                    "latent_idx": int(float(concept["latent_idx"])),
                    "sae_group_index": int(label["sae_group_index"]),
                    "label_color": label.get("color", color_for_label(str(label["task"]), str(label["class_label"]), len(records))),
                    "concept_color": "",
                    "concept": concept,
                    "label": label,
                }
            )
    records.sort(key=lambda row: (str(row["task"]), str(row["class_label"]), int(row["concept_rank"]), int(row["latent_idx"])))
    for idx, record in enumerate(records):
        record["concept_id"] = (
            f"{sanitize_component(str(record['task']))}__"
            f"{sanitize_component(str(record['class_label']))}__latent_{int(record['latent_idx'])}"
        )
        record["concept_color"] = color_for_concept_index(idx)
    return records


def load_selected_concepts(
    *,
    concept_review_root: Path,
    concept_set: str,
    top_concepts_per_label: int,
    label_filters: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_csv = concept_review_root / "selected_morphology_labels.csv"
    if not selected_csv.exists():
        raise FileNotFoundError(f"Missing curated label table: {selected_csv}")

    rows = read_csv_rows(selected_csv)
    wanted = {tuple(item.split("/", 1)) for item in label_filters if "/" in item}
    labels: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}

    for row in rows:
        task = str(row.get("task", "")).strip()
        class_label = str(row.get("class_label", "")).strip()
        if not task or not class_label:
            continue
        if row.get("available", "") and not parse_bool(row.get("available")):
            continue
        if concept_set == "priority" and (task, class_label) not in PRIORITY_LABELS:
            continue
        if wanted and (task, class_label) not in wanted:
            continue

        export_dir_raw = row.get("concept_export_dir") or row.get("review_dir", "")
        export_dir = resolve_path(export_dir_raw, base=WSI_CF_ROOT)
        if export_dir.name != "concept_export":
            export_dir = export_dir / "concept_export"
        concepts_csv = export_dir / "concepts.csv"
        manifest_path = export_dir / "manifest.json"
        if not concepts_csv.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Concept export is incomplete for {task}/{class_label}: {export_dir}")

        concept_rows = read_csv_rows(concepts_csv)
        concept_rows = sorted(concept_rows, key=lambda r: int(float(r.get("concept_rank") or 10**9)))
        selected = concept_rows[: max(1, int(top_concepts_per_label))]
        manifest = read_json(manifest_path)
        manifest_key = f"{task}/{class_label}"
        manifests[manifest_key] = manifest
        label_idx = len(labels)
        labels.append(
            {
                "task": task,
                "class_label": class_label,
                "priority": row.get("priority", ""),
                "verdict": row.get("verdict", ""),
                "reason": row.get("reason", ""),
                "concept_export_dir": str(export_dir),
                "manifest_path": str(manifest_path),
                "manifest": manifest,
                "color": color_for_label(task, class_label, label_idx),
                "concepts": selected,
            }
        )

    if not labels:
        raise RuntimeError("No curated concepts were selected. Check --concept-set and --label-filter.")
    return labels, manifests


def group_labels_by_sae(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    key_to_index: dict[tuple[str, str, str], int] = {}
    for label in labels:
        sae = dict(label["manifest"].get("sae", {}))
        key = (
            str(sae.get("checkpoint", "")),
            str(sae.get("config", "")),
            str(sae.get("latent_dim", "")),
        )
        if not key[0] or not key[1]:
            raise RuntimeError(f"Missing SAE checkpoint/config in manifest for {label['task']}/{label['class_label']}.")
        if key not in key_to_index:
            key_to_index[key] = len(groups)
            groups.append({"sae": sae, "labels": [], "group_index": len(groups)})
        group_index = key_to_index[key]
        label["sae_group_index"] = int(group_index)
        groups[group_index]["labels"].append(label)
    if not groups:
        raise RuntimeError("No SAE provenance found.")
    return groups


def compute_latent_activations(
    *,
    features: np.ndarray,
    latent_ids: list[int],
    sae_ckpt: Path,
    sae_cfg: Path,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    from wsi_cf.common.runtime import resolve_device
    from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features

    resolved_device = resolve_device(str(device))
    cache_key = (str(Path(sae_ckpt).resolve()), str(Path(sae_cfg).resolve()), str(resolved_device))
    cache = getattr(compute_latent_activations, "_model_cache", {})
    if cache_key in cache:
        model, d_in, d_latent = cache[cache_key]
    else:
        model, d_in, d_latent = load_sae_from_config(sae_ckpt, sae_cfg, device=str(resolved_device))
        cache[cache_key] = (model, d_in, d_latent)
        setattr(compute_latent_activations, "_model_cache", cache)
    if int(features.shape[1]) != int(d_in):
        raise ValueError(f"Feature dimension {features.shape[1]} does not match SAE d_in={d_in}")
    bad = [latent for latent in latent_ids if int(latent) < 0 or int(latent) >= int(d_latent)]
    if bad:
        raise ValueError(f"Latent IDs outside SAE latent dimension {d_latent}: {bad[:10]}")

    out = np.empty((features.shape[0], len(latent_ids)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], int(batch_size)):
            end = min(features.shape[0], start + int(batch_size))
            x = torch.as_tensor(features[start:end], dtype=torch.float32, device=resolved_device)
            z = sae_encode_features(model, x)
            out[start:end] = z[:, latent_ids].detach().cpu().numpy().astype(np.float32, copy=False)
    provenance = {
        "device_requested": str(device),
        "device_resolved": str(resolved_device),
        "d_in": int(d_in),
        "d_latent": int(d_latent),
    }
    return out, provenance


def make_thumbnail(slide: Any, *, max_side: int) -> tuple[Image.Image, float, float]:
    width, height = slide.dimensions
    out_w, out_h, scale_x, scale_y = scaled_thumbnail_size(int(width), int(height), int(max_side))
    thumb = slide.get_thumbnail((out_w, out_h)).convert("RGB")
    if thumb.size != (out_w, out_h):
        thumb = thumb.resize((out_w, out_h), resample=Image.BILINEAR)
    return thumb, float(scale_x), float(scale_y)


def draw_score_overlay(
    base: Image.Image,
    *,
    coords: np.ndarray,
    scores_norm: np.ndarray,
    tile_size_level0: int,
    scale_x: float,
    scale_y: float,
    color: str,
    alpha: int,
    min_score: float,
) -> Image.Image:
    out = base.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rgb = hex_to_rgb(color)
    width, height = out.size
    coords_i = np.asarray(coords, dtype=np.int64)
    scores = np.asarray(scores_norm, dtype=np.float32).reshape(-1)
    for idx, coord in enumerate(coords_i):
        score = float(scores[idx])
        if score < float(min_score):
            continue
        x0, y0, x1, y1 = tile_bounds_on_thumbnail(
            coord,
            tile_size_level0=int(tile_size_level0),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            width=int(width),
            height=int(height),
        )
        fill_alpha = max(8, min(255, int(round(float(alpha) * score))))
        draw.rectangle([x0, y0, x1, y1], fill=(*rgb, fill_alpha))
        if score >= 0.98:
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 220), width=1)
    return Image.alpha_composite(out, overlay).convert("RGB")


def draw_combined_overlay(
    base: Image.Image,
    *,
    coords: np.ndarray,
    label_scores: list[tuple[dict[str, Any], np.ndarray]],
    tile_size_level0: int,
    scale_x: float,
    scale_y: float,
    alpha: int,
    min_score: float,
) -> Image.Image:
    out = base.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = out.size
    best_scores = np.full(coords.shape[0], -np.inf, dtype=np.float32)
    best_colors: list[str | None] = [None] * int(coords.shape[0])
    for label, scores in label_scores:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        mask = scores > best_scores
        best_scores[mask] = scores[mask]
        for idx in np.flatnonzero(mask):
            best_colors[int(idx)] = str(label["color"])

    for idx, coord in enumerate(np.asarray(coords, dtype=np.int64)):
        score = float(best_scores[idx])
        color = best_colors[idx]
        if color is None or score < float(min_score):
            continue
        x0, y0, x1, y1 = tile_bounds_on_thumbnail(
            coord,
            tile_size_level0=int(tile_size_level0),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            width=int(width),
            height=int(height),
        )
        fill_alpha = max(8, min(255, int(round(float(alpha) * score))))
        draw.rectangle([x0, y0, x1, y1], fill=(*hex_to_rgb(color), fill_alpha))
    return Image.alpha_composite(out, overlay).convert("RGB")


def draw_winner_concept_overlay(
    base: Image.Image,
    *,
    coords: np.ndarray,
    winner_indices: np.ndarray,
    winner_scores: np.ndarray,
    concept_records: list[dict[str, Any]],
    tile_size_level0: int,
    scale_x: float,
    scale_y: float,
    alpha: int,
    min_score: float,
) -> Image.Image:
    out = base.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = out.size
    coords_i = np.asarray(coords, dtype=np.int64)
    winners = np.asarray(winner_indices, dtype=np.int64).reshape(-1)
    scores = np.asarray(winner_scores, dtype=np.float32).reshape(-1)
    for idx, coord in enumerate(coords_i):
        concept_idx = int(winners[idx])
        score = float(scores[idx])
        if concept_idx < 0 or score < float(min_score):
            continue
        color = str(concept_records[concept_idx]["concept_color"])
        x0, y0, x1, y1 = tile_bounds_on_thumbnail(
            coord,
            tile_size_level0=int(tile_size_level0),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            width=int(width),
            height=int(height),
        )
        fill_alpha = max(10, min(255, int(round(float(alpha) * score))))
        draw.rectangle([x0, y0, x1, y1], fill=(*hex_to_rgb(color), fill_alpha))
    return Image.alpha_composite(out, overlay).convert("RGB")


def winner_concept_by_tile(
    *,
    concept_records: list[dict[str, Any]],
    sae_groups: list[dict[str, Any]],
    vmin_percentile: float,
    vmax_percentile: float,
    score_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_columns: list[np.ndarray] = []
    raw_columns: list[np.ndarray] = []
    for record in concept_records:
        group = sae_groups[int(record["sae_group_index"])]
        latent_idx = int(record["latent_idx"])
        col = group["latent_to_col"][latent_idx]
        raw = np.asarray(group["activations"][:, col], dtype=np.float32)
        norm = normalize_scores(raw, vmin_percentile=float(vmin_percentile), vmax_percentile=float(vmax_percentile))
        raw_columns.append(raw)
        score_columns.append(norm)
    if not score_columns:
        raise RuntimeError("No selected concepts were available for winner overlay.")
    norm_matrix = np.stack(score_columns, axis=1)
    raw_matrix = np.stack(raw_columns, axis=1)
    if str(score_mode) == "normalized":
        selector = norm_matrix
    elif str(score_mode) == "raw":
        selector = raw_matrix
    else:
        raise ValueError(f"Unsupported winner score mode: {score_mode}")
    winner_indices = np.argmax(selector, axis=1).astype(np.int64)
    tile_indices = np.arange(norm_matrix.shape[0], dtype=np.int64)
    winner_scores = norm_matrix[tile_indices, winner_indices].astype(np.float32)
    winner_raw = raw_matrix[tile_indices, winner_indices].astype(np.float32)
    return winner_indices, winner_scores, winner_raw


def save_concept_legend(concept_records: list[dict[str, Any]], path: Path, *, max_records: int = 80) -> None:
    records = concept_records[: max(0, int(max_records))]
    row_h = 24
    width = 980
    height = max(row_h, row_h * (len(records) + 2))
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    suffix = "" if len(records) == len(concept_records) else f" (top {len(records)} of {len(concept_records)})"
    draw.text((8, 4), f"Color legend: winner concept per feature tile{suffix}", fill=(0, 0, 0))
    for idx, record in enumerate(records, start=1):
        y = idx * row_h + 2
        color = hex_to_rgb(str(record["concept_color"]))
        draw.rectangle([8, y, 26, y + 16], fill=color, outline=(0, 0, 0))
        text = (
            f"{record['task']}/{record['class_label']} "
            f"rank {record['concept_rank']} latent {record['latent_idx']}"
        )
        draw.text((34, y), text, fill=(0, 0, 0))
    save_png(img, path)


def summarize_winner_concepts(
    *,
    winner_indices: np.ndarray,
    winner_scores: np.ndarray,
    winner_raw: np.ndarray,
    concept_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    winners = np.asarray(winner_indices, dtype=np.int64).reshape(-1)
    scores = np.asarray(winner_scores, dtype=np.float32).reshape(-1)
    raw = np.asarray(winner_raw, dtype=np.float32).reshape(-1)
    rows: list[dict[str, Any]] = []
    for concept_idx in sorted(set(int(x) for x in winners.tolist())):
        mask = winners == concept_idx
        record = concept_records[int(concept_idx)]
        rows.append(
            {
                "winner_concept_index": int(concept_idx),
                "winner_concept_id": record["concept_id"],
                "task": record["task"],
                "class_label": record["class_label"],
                "concept_rank": int(record["concept_rank"]),
                "latent_idx": int(record["latent_idx"]),
                "sae_group_index": int(record["sae_group_index"]),
                "color": record["concept_color"],
                "tile_count": int(mask.sum()),
                "tile_fraction": float(mask.mean()) if mask.size else 0.0,
                "mean_raw_activation": float(raw[mask].mean()) if mask.any() else 0.0,
                "max_raw_activation": float(raw[mask].max()) if mask.any() else 0.0,
                "mean_normalized_activation": float(scores[mask].mean()) if mask.any() else 0.0,
                "max_normalized_activation": float(scores[mask].max()) if mask.any() else 0.0,
            }
        )
    rows.sort(key=lambda row: (-int(row["tile_count"]), -float(row["max_raw_activation"]), int(row["latent_idx"])))
    return rows


def save_vector_wrappers(img: Image.Image, base_path_without_suffix: Path, *, enabled: bool) -> list[str]:
    if not enabled:
        return []
    written: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return written

    for suffix in (".pdf", ".svg"):
        out_path = base_path_without_suffix.with_suffix(suffix)
        fig_w = max(1.0, float(img.size[0]) / 300.0)
        fig_h = max(1.0, float(img.size[1]) / 300.0)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.set_axis_off()
        fig.savefig(out_path)
        plt.close(fig)
        written.append(str(out_path))
    return written


def save_top_tiles(
    *,
    slide: Any,
    coords: np.ndarray,
    raw_scores: np.ndarray,
    norm_scores: np.ndarray,
    latent_idx: int,
    out_dir: Path,
    tile_size_px: int,
    max_tiles: int,
    top_tile_size: int,
) -> list[dict[str, Any]]:
    from wsi_cf.data.slides import crop_tile_rgb

    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(np.asarray(raw_scores, dtype=np.float32))[::-1]
    rows: list[dict[str, Any]] = []
    tiles: list[Image.Image] = []
    for rank, tile_index in enumerate(order[: max(0, int(max_tiles))], start=1):
        coord = np.asarray(coords[int(tile_index)], dtype=np.int64)
        img, crop_px = crop_tile_rgb(
            slide,
            x=int(coord[0]),
            y=int(coord[1]),
            tile_size_20x=int(tile_size_px),
            out_tile_size=int(top_tile_size),
        )
        out_path = out_dir / f"rank_{rank:03d}_latent_{int(latent_idx)}_x_{int(coord[0])}_y_{int(coord[1])}.png"
        save_png(img, out_path)
        tiles.append(img)
        rows.append(
            {
                "rank_within_latent": int(rank),
                "tile_index": int(tile_index),
                "coord_x": int(coord[0]),
                "coord_y": int(coord[1]),
                "raw_activation": float(raw_scores[int(tile_index)]),
                "normalized_activation": float(norm_scores[int(tile_index)]),
                "crop_px_level0": int(crop_px),
                "output_path": str(out_path),
            }
        )

    if tiles:
        cols = min(8, len(tiles))
        rows_n = int(math.ceil(len(tiles) / float(cols)))
        sheet = Image.new("RGB", (cols * int(top_tile_size), rows_n * int(top_tile_size)), (255, 255, 255))
        for idx, img in enumerate(tiles):
            sheet.paste(img, ((idx % cols) * int(top_tile_size), (idx // cols) * int(top_tile_size)))
        save_png(sheet, out_dir / "contact_sheet.png")
    return rows


def concept_output_dir(out_dir: Path, task: str, class_label: str) -> Path:
    return out_dir / "labels" / f"{sanitize_component(task)}__{sanitize_component(class_label)}"


def compute_slide_sae_activations(
    *,
    features: np.ndarray,
    sae_groups: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    for group in sae_groups:
        sae = dict(group["sae"])
        sae_ckpt = resolve_path(str(sae.get("checkpoint", "")), base=WSI_CF_ROOT)
        sae_cfg = resolve_path(str(sae.get("config", "")), base=WSI_CF_ROOT)
        if not sae_ckpt.exists():
            raise FileNotFoundError(f"SAE checkpoint does not exist: {sae_ckpt}")
        if not sae_cfg.exists():
            raise FileNotFoundError(f"SAE config does not exist: {sae_cfg}")
        latent_ids = sorted(
            {
                int(float(concept["latent_idx"]))
                for label in group["labels"]
                for concept in label["concepts"]
                if str(concept.get("latent_idx", "")).strip()
            }
        )
        activations, activation_provenance = compute_latent_activations(
            features=features,
            latent_ids=latent_ids,
            sae_ckpt=sae_ckpt,
            sae_cfg=sae_cfg,
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        group["checkpoint_resolved"] = str(sae_ckpt)
        group["config_resolved"] = str(sae_cfg)
        group["latent_ids"] = latent_ids
        group["latent_to_col"] = {latent: idx for idx, latent in enumerate(latent_ids)}
        group["activations"] = activations
        group["activation_provenance"] = activation_provenance


def run_overlay_for_pair(
    *,
    args: argparse.Namespace,
    pair: dict[str, Any],
    labels: list[dict[str, Any]],
    manifests: dict[str, Any],
    sae_groups: list[dict[str, Any]],
    concept_records: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slide_path = Path(pair["slide_path"])
    h5_path = Path(pair["h5_path"])
    slide_key = str(pair["slide_key"])

    features, coords = read_h5_features_coords(h5_path)
    compute_slide_sae_activations(features=features, sae_groups=sae_groups, args=args)

    from wsi_cf.data.slides import open_slide

    slide = open_slide(slide_path)
    thumbnail, scale_x, scale_y = make_thumbnail(slide, max_side=int(args.thumbnail_max_side))
    sample_thumb_path = out_dir / "sample_thumbnail.png"
    save_png(thumbnail, sample_thumb_path)

    first_manifest = labels[0]["manifest"]
    tile_info = dict(first_manifest.get("tile_extraction", {}))
    tile_size_px = int(tile_info.get("tile_size_px") or 256)
    inferred_tile_size = infer_coord_tile_size(coords, fallback=tile_size_px)
    tile_size_level0 = int(args.tile_size_level0) if int(args.tile_size_level0) > 0 else int(inferred_tile_size)

    overlay_rows: list[dict[str, Any]] = []
    concept_config_rows: list[dict[str, Any]] = []
    label_scores_for_combined: list[tuple[dict[str, Any], np.ndarray]] = []
    overlay_rows.append(
        {
            "slide_key": slide_key,
            "task": "sample",
            "class_label": "thumbnail",
            "concept_rank": "",
            "latent_idx": "",
            "overlay_type": "sample_thumbnail",
            "coord_x": "",
            "coord_y": "",
            "tile_index": "",
            "raw_activation": "",
            "normalized_activation": "",
            "raw_min": "",
            "raw_median": "",
            "raw_p95": "",
            "raw_max": "",
            "rank_within_latent": "",
            "thumbnail_x0": "",
            "thumbnail_y0": "",
            "thumbnail_x1": "",
            "thumbnail_y1": "",
            "output_path": str(sample_thumb_path),
        }
    )

    concept_legend_rows: list[dict[str, Any]] = []
    for concept_idx, record in enumerate(concept_records):
        concept_legend_rows.append(
            {
                "concept_index": int(concept_idx),
                "concept_id": record["concept_id"],
                "task": record["task"],
                "class_label": record["class_label"],
                "concept_rank": record["concept_rank"],
                "latent_idx": record["latent_idx"],
                "sae_group_index": record["sae_group_index"],
                "concept_color": record["concept_color"],
            }
        )
    write_csv(
        out_dir / "concept_legend.csv",
        concept_legend_rows,
        ["concept_index", "concept_id", "task", "class_label", "concept_rank", "latent_idx", "sae_group_index", "concept_color"],
    )

    winner_indices, winner_scores, winner_raw = winner_concept_by_tile(
        concept_records=concept_records,
        sae_groups=sae_groups,
        vmin_percentile=float(args.vmin_percentile),
        vmax_percentile=float(args.vmax_percentile),
        score_mode=str(args.winner_score_mode),
    )
    winner_img = draw_winner_concept_overlay(
        thumbnail,
        coords=coords,
        winner_indices=winner_indices,
        winner_scores=winner_scores,
        concept_records=concept_records,
        tile_size_level0=int(tile_size_level0),
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        alpha=int(args.overlay_alpha),
        min_score=float(args.min_normalized_activation),
    )
    winner_path = out_dir / "winner_concept_per_tile_overlay.png"
    save_png(winner_img, winner_path)
    save_png(winner_img, out_dir / "all_selected_concepts_overlay.png")
    winner_vector_paths = save_vector_wrappers(winner_img, out_dir / "winner_concept_per_tile_overlay", enabled=bool(args.save_vector_panels))

    winner_rows: list[dict[str, Any]] = []
    for tile_index, coord in enumerate(np.asarray(coords, dtype=np.int64)):
        concept_idx = int(winner_indices[int(tile_index)])
        record = concept_records[concept_idx]
        bounds = tile_bounds_on_thumbnail(
            coord,
            tile_size_level0=int(tile_size_level0),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            width=int(thumbnail.size[0]),
            height=int(thumbnail.size[1]),
        )
        winner_rows.append(
            {
                "slide_key": slide_key,
                "project": pair.get("project", ""),
                "tile_index": int(tile_index),
                "coord_x": int(coord[0]),
                "coord_y": int(coord[1]),
                "winner_concept_index": concept_idx,
                "winner_concept_id": record["concept_id"],
                "winner_task": record["task"],
                "winner_class_label": record["class_label"],
                "winner_concept_rank": int(record["concept_rank"]),
                "winner_latent_idx": int(record["latent_idx"]),
                "winner_sae_group_index": int(record["sae_group_index"]),
                "winner_color": record["concept_color"],
                "winner_score_mode": str(args.winner_score_mode),
                "winner_raw_activation": float(winner_raw[int(tile_index)]),
                "winner_normalized_activation": float(winner_scores[int(tile_index)]),
                "thumbnail_x0": bounds[0],
                "thumbnail_y0": bounds[1],
                "thumbnail_x1": bounds[2],
                "thumbnail_y1": bounds[3],
            }
        )
    write_csv(
        out_dir / "tile_winner_concepts.csv",
        winner_rows,
        [
            "slide_key",
            "project",
            "tile_index",
            "coord_x",
            "coord_y",
            "winner_concept_index",
            "winner_concept_id",
            "winner_task",
            "winner_class_label",
            "winner_concept_rank",
            "winner_latent_idx",
            "winner_sae_group_index",
            "winner_color",
            "winner_score_mode",
            "winner_raw_activation",
            "winner_normalized_activation",
            "thumbnail_x0",
            "thumbnail_y0",
            "thumbnail_x1",
            "thumbnail_y1",
        ],
    )
    winner_summary_rows = summarize_winner_concepts(
        winner_indices=winner_indices,
        winner_scores=winner_scores,
        winner_raw=winner_raw,
        concept_records=concept_records,
    )
    write_csv(
        out_dir / "winning_concepts_summary.csv",
        winner_summary_rows,
        [
            "winner_concept_index",
            "winner_concept_id",
            "task",
            "class_label",
            "concept_rank",
            "latent_idx",
            "sae_group_index",
            "color",
            "tile_count",
            "tile_fraction",
            "mean_raw_activation",
            "max_raw_activation",
            "mean_normalized_activation",
            "max_normalized_activation",
        ],
    )
    top_winner_indices = [int(row["winner_concept_index"]) for row in winner_summary_rows[: max(1, int(args.max_legend_concepts))]]
    save_concept_legend(
        [concept_records[idx] for idx in top_winner_indices],
        out_dir / "concept_legend.png",
        max_records=int(args.max_legend_concepts),
    )
    overlay_rows.append(
        {
            "slide_key": slide_key,
            "task": "winner",
            "class_label": "per_tile",
            "concept_rank": "",
            "latent_idx": "",
            "overlay_type": "winner_concept_per_tile_overlay",
            "coord_x": "",
            "coord_y": "",
            "tile_index": "",
            "raw_activation": "",
            "normalized_activation": "",
            "raw_min": "",
            "raw_median": "",
            "raw_p95": "",
            "raw_max": "",
            "rank_within_latent": "",
            "thumbnail_x0": "",
            "thumbnail_y0": "",
            "thumbnail_x1": "",
            "thumbnail_y1": "",
            "output_path": str(winner_path),
        }
    )
    for vector_path in winner_vector_paths:
        overlay_rows.append(
            {
                "slide_key": slide_key,
                "task": "winner",
                "class_label": "per_tile",
                "concept_rank": "",
                "latent_idx": "",
                "overlay_type": "winner_concept_per_tile_overlay_vector",
                "coord_x": "",
                "coord_y": "",
                "tile_index": "",
                "raw_activation": "",
                "normalized_activation": "",
                "raw_min": "",
                "raw_median": "",
                "raw_p95": "",
                "raw_max": "",
                "rank_within_latent": "",
                "thumbnail_x0": "",
                "thumbnail_y0": "",
                "thumbnail_x1": "",
                "thumbnail_y1": "",
                "output_path": vector_path,
            }
        )

    detail_labels = labels if str(args.concept_source) == "curated" else []
    for label in detail_labels:
        label_dir = concept_output_dir(out_dir, str(label["task"]), str(label["class_label"]))
        label_dir.mkdir(parents=True, exist_ok=True)

        per_latent_norms: list[np.ndarray] = []
        for concept in label["concepts"]:
            latent_idx = int(float(concept["latent_idx"]))
            concept_rank = int(float(concept.get("concept_rank") or 0))
            sae_group = sae_groups[int(label["sae_group_index"])]
            col = sae_group["latent_to_col"][latent_idx]
            raw = sae_group["activations"][:, col]
            norm = normalize_scores(
                raw,
                vmin_percentile=float(args.vmin_percentile),
                vmax_percentile=float(args.vmax_percentile),
            )
            per_latent_norms.append(norm)
            latent_img = draw_score_overlay(
                thumbnail,
                coords=coords,
                scores_norm=norm,
                tile_size_level0=int(tile_size_level0),
                scale_x=float(scale_x),
                scale_y=float(scale_y),
                color=str(next(r["concept_color"] for r in concept_records if r["task"] == label["task"] and r["class_label"] == label["class_label"] and int(r["latent_idx"]) == int(latent_idx))),
                alpha=int(args.overlay_alpha),
                min_score=float(args.min_normalized_activation),
            )
            latent_path = label_dir / f"latent_{latent_idx}_overlay.png"
            save_png(latent_img, latent_path)

            top_tile_dir = label_dir / f"latent_{latent_idx}_top_tiles"
            crop_rows = save_top_tiles(
                slide=slide,
                coords=coords,
                raw_scores=raw,
                norm_scores=norm,
                latent_idx=latent_idx,
                out_dir=top_tile_dir,
                tile_size_px=int(tile_size_px),
                max_tiles=int(args.max_tiles_per_concept),
                top_tile_size=int(args.top_tile_size),
            ) if bool(args.save_top_tiles) else []
            for crop_row in crop_rows:
                bounds = tile_bounds_on_thumbnail(
                    (int(crop_row["coord_x"]), int(crop_row["coord_y"])),
                    tile_size_level0=int(tile_size_level0),
                    scale_x=float(scale_x),
                    scale_y=float(scale_y),
                    width=int(thumbnail.size[0]),
                    height=int(thumbnail.size[1]),
                )
                overlay_rows.append(
                    {
                        "slide_key": slide_key,
                        "task": label["task"],
                        "class_label": label["class_label"],
                        "concept_rank": concept_rank,
                        "latent_idx": latent_idx,
                        "overlay_type": "tile_crop",
                        "coord_x": crop_row["coord_x"],
                        "coord_y": crop_row["coord_y"],
                        "tile_index": crop_row["tile_index"],
                        "raw_activation": crop_row["raw_activation"],
                        "normalized_activation": crop_row["normalized_activation"],
                        "raw_min": "",
                        "raw_median": "",
                        "raw_p95": "",
                        "raw_max": "",
                        "rank_within_latent": crop_row["rank_within_latent"],
                        "thumbnail_x0": bounds[0],
                        "thumbnail_y0": bounds[1],
                        "thumbnail_x1": bounds[2],
                        "thumbnail_y1": bounds[3],
                        "output_path": crop_row["output_path"],
                    }
                )

            overlay_rows.append(
                {
                    "slide_key": slide_key,
                    "task": label["task"],
                    "class_label": label["class_label"],
                    "concept_rank": concept_rank,
                    "latent_idx": latent_idx,
                    "overlay_type": "latent_overlay",
                    "coord_x": "",
                    "coord_y": "",
                    "tile_index": "",
                    "raw_activation": float(np.max(raw)) if raw.size else 0.0,
                    "normalized_activation": float(np.max(norm)) if norm.size else 0.0,
                    "raw_min": float(np.min(raw)) if raw.size else 0.0,
                    "raw_median": float(np.median(raw)) if raw.size else 0.0,
                    "raw_p95": float(np.percentile(raw, 95.0)) if raw.size else 0.0,
                    "raw_max": float(np.max(raw)) if raw.size else 0.0,
                    "rank_within_latent": "",
                    "thumbnail_x0": "",
                    "thumbnail_y0": "",
                    "thumbnail_x1": "",
                    "thumbnail_y1": "",
                    "output_path": str(latent_path),
                }
            )
            concept_config_rows.append(
                {
                    "task": label["task"],
                    "class_label": label["class_label"],
                    "concept_rank": concept_rank,
                    "latent_idx": latent_idx,
                    "final_score": concept.get("final_score", ""),
                    "association_score": concept.get("association_score", ""),
                    "cohen_d": concept.get("cohen_d", ""),
                    "color": next(r["concept_color"] for r in concept_records if r["task"] == label["task"] and r["class_label"] == label["class_label"] and int(r["latent_idx"]) == int(latent_idx)),
                    "latent_overlay_path": str(latent_path),
                    "raw_min": float(np.min(raw)) if raw.size else 0.0,
                    "raw_median": float(np.median(raw)) if raw.size else 0.0,
                    "raw_p95": float(np.percentile(raw, 95.0)) if raw.size else 0.0,
                    "raw_max": float(np.max(raw)) if raw.size else 0.0,
                }
            )

        for contact_sheet_path in sorted(label_dir.glob("latent_*_top_tiles/contact_sheet.png")):
            latent_text = contact_sheet_path.parent.name.replace("latent_", "").replace("_top_tiles", "")
            overlay_rows.append(
                {
                    "slide_key": slide_key,
                    "task": label["task"],
                    "class_label": label["class_label"],
                    "concept_rank": "",
                    "latent_idx": latent_text,
                    "overlay_type": "top_tiles_contact_sheet",
                    "coord_x": "",
                    "coord_y": "",
                    "tile_index": "",
                    "raw_activation": "",
                    "normalized_activation": "",
                    "raw_min": "",
                    "raw_median": "",
                    "raw_p95": "",
                    "raw_max": "",
                    "rank_within_latent": "",
                    "thumbnail_x0": "",
                    "thumbnail_y0": "",
                    "thumbnail_x1": "",
                    "thumbnail_y1": "",
                    "output_path": str(contact_sheet_path),
                }
            )

        if per_latent_norms:
            label_score = np.max(np.stack(per_latent_norms, axis=1), axis=1)
        else:
            label_score = np.zeros((coords.shape[0],), dtype=np.float32)
        label_scores_for_combined.append((label, label_score))

        label_img = draw_score_overlay(
            thumbnail,
            coords=coords,
            scores_norm=label_score,
            tile_size_level0=int(tile_size_level0),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            color=str(label["color"]),
            alpha=int(args.overlay_alpha),
            min_score=float(args.min_normalized_activation),
        )
        label_overlay_path = label_dir / "overlay.png"
        save_png(label_img, label_overlay_path)
        label_vector_paths = save_vector_wrappers(label_img, label_dir / "overlay", enabled=bool(args.save_vector_panels))
        overlay_rows.append(
            {
                "slide_key": slide_key,
                "task": label["task"],
                "class_label": label["class_label"],
                "concept_rank": "",
                "latent_idx": "",
                "overlay_type": "label_overlay",
                "coord_x": "",
                "coord_y": "",
                "tile_index": "",
                "raw_activation": "",
                "normalized_activation": float(np.max(label_score)) if label_score.size else 0.0,
                "raw_min": "",
                "raw_median": "",
                "raw_p95": "",
                "raw_max": "",
                "rank_within_latent": "",
                "thumbnail_x0": "",
                "thumbnail_y0": "",
                "thumbnail_x1": "",
                "thumbnail_y1": "",
                "output_path": str(label_overlay_path),
            }
        )
        for vector_path in label_vector_paths:
            overlay_rows.append(
                {
                    "slide_key": slide_key,
                    "task": label["task"],
                    "class_label": label["class_label"],
                    "concept_rank": "",
                    "latent_idx": "",
                    "overlay_type": "label_overlay_vector",
                    "coord_x": "",
                    "coord_y": "",
                    "tile_index": "",
                    "raw_activation": "",
                    "normalized_activation": float(np.max(label_score)) if label_score.size else 0.0,
                    "raw_min": "",
                    "raw_median": "",
                    "raw_p95": "",
                    "raw_max": "",
                    "rank_within_latent": "",
                    "thumbnail_x0": "",
                    "thumbnail_y0": "",
                    "thumbnail_x1": "",
                    "thumbnail_y1": "",
                    "output_path": vector_path,
                }
            )

    overlay_rows.append(
        {
            "slide_key": slide_key,
            "task": "all_selected",
            "class_label": "all_selected",
            "concept_rank": "",
            "latent_idx": "",
            "overlay_type": "combined_overlay",
            "coord_x": "",
            "coord_y": "",
            "tile_index": "",
            "raw_activation": "",
            "normalized_activation": "",
            "raw_min": "",
            "raw_median": "",
            "raw_p95": "",
            "raw_max": "",
            "rank_within_latent": "",
            "thumbnail_x0": "",
            "thumbnail_y0": "",
            "thumbnail_x1": "",
            "thumbnail_y1": "",
            "output_path": str(out_dir / "all_selected_concepts_overlay.png"),
        }
    )

    overlay_fields = [
        "slide_key",
        "task",
        "class_label",
        "concept_rank",
        "latent_idx",
        "overlay_type",
        "coord_x",
        "coord_y",
        "tile_index",
        "raw_activation",
        "normalized_activation",
        "raw_min",
        "raw_median",
        "raw_p95",
        "raw_max",
        "rank_within_latent",
        "thumbnail_x0",
        "thumbnail_y0",
        "thumbnail_x1",
        "thumbnail_y1",
        "output_path",
    ]
    write_csv(out_dir / "overlay_table.csv", overlay_rows, overlay_fields)
    write_csv(
        out_dir / "selected_concepts_table.csv",
        concept_config_rows,
        [
            "task",
            "class_label",
            "concept_rank",
            "latent_idx",
            "final_score",
            "association_score",
            "cohen_d",
            "color",
            "latent_overlay_path",
            "raw_min",
            "raw_median",
            "raw_p95",
            "raw_max",
        ],
    )

    run_config = {
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "slide_key": slide_key,
        "project": pair.get("project", ""),
        "slide_path": str(slide_path),
        "h5_path": str(h5_path),
        "out_dir": str(out_dir),
        "concept_review_root": str(args.concept_review_root),
        "concept_set": str(args.concept_set),
        "top_concepts_per_label": int(args.top_concepts_per_label),
        "selected_labels": [
            {
                "task": label["task"],
                "class_label": label["class_label"],
                "color": label["color"],
                "sae_group_index": int(label["sae_group_index"]),
                "concept_export_dir": label["concept_export_dir"],
                "n_concepts": len(label["concepts"]),
                "concepts": [] if str(args.concept_source) == "all_sae" else [
                    {
                        "concept_rank": int(float(concept.get("concept_rank") or 0)),
                        "latent_idx": int(float(concept["latent_idx"])),
                        "final_score": concept.get("final_score", ""),
                        "association_score": concept.get("association_score", ""),
                        "cohen_d": concept.get("cohen_d", ""),
                    }
                    for concept in label["concepts"]
                ],
            }
            for label in labels
        ],
        "sae_groups": [
            {
                "group_index": int(group["group_index"]),
                "sae": group["sae"],
                "checkpoint_resolved": group["checkpoint_resolved"],
                "config_resolved": group["config_resolved"],
                "latent_ids": [int(latent) for latent in group["latent_ids"]],
                "labels": [f"{label['task']}/{label['class_label']}" for label in group["labels"]],
                **group["activation_provenance"],
            }
            for group in sae_groups
        ],
        "h5": {
            "features_shape": list(features.shape),
            "coords_shape": list(coords.shape),
            "coord_space_assumption": "level0_h5_coords",
            "inferred_tile_size_level0": int(inferred_tile_size),
            "tile_size_level0_used": int(tile_size_level0),
        },
        "thumbnail": {
            "path": str(sample_thumb_path),
            "width": int(thumbnail.size[0]),
            "height": int(thumbnail.size[1]),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "max_side": int(args.thumbnail_max_side),
        },
        "normalization": {
            "vmin_percentile": float(args.vmin_percentile),
            "vmax_percentile": float(args.vmax_percentile),
            "min_normalized_activation": float(args.min_normalized_activation),
            "combined_overlay": "winner_concept_per_feature_tile",
            "winner_score_mode": str(args.winner_score_mode),
            "concept_source": str(args.concept_source),
        },
        "manifests": manifests,
    }
    write_json(out_dir / "run_config.json", run_config)
    return overlay_rows, run_config


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.concept_source) == "all_sae":
        labels, manifests = load_all_sae_concepts(sae_ckpt=args.sae_ckpt, sae_cfg=args.sae_cfg)
    else:
        labels, manifests = load_selected_concepts(
            concept_review_root=args.concept_review_root,
            concept_set=str(args.concept_set),
            top_concepts_per_label=int(args.top_concepts_per_label),
            label_filters=list(args.label_filter or []),
        )
    sae_groups = group_labels_by_sae(labels)
    concept_records = assign_concept_colors(labels)
    pairs = choose_slide_h5_pairs(args)

    sampled_rows: list[dict[str, Any]] = []
    aggregate_overlay_rows: list[dict[str, Any]] = []
    slide_configs: list[dict[str, Any]] = []
    explicit_single = args.slide_path is not None and args.h5_path is not None and len(pairs) == 1
    for idx, pair in enumerate(pairs, start=1):
        slide_out_dir = out_dir if explicit_single else out_dir / "slides" / sanitize_component(str(pair["slide_key"]))
        print(f"[overlay-sae-concepts] slide {idx}/{len(pairs)}: {pair['slide_key']}")
        overlay_rows, run_config = run_overlay_for_pair(
            args=args,
            pair=pair,
            labels=labels,
            manifests=manifests,
            sae_groups=sae_groups,
            concept_records=concept_records,
            out_dir=slide_out_dir,
        )
        aggregate_overlay_rows.extend(overlay_rows)
        slide_configs.append(run_config)
        sampled_rows.append(
            {
                "sample_index": int(idx),
                "slide_key": pair["slide_key"],
                "project": pair.get("project", ""),
                "slide_path": str(pair["slide_path"]),
                "h5_path": str(pair["h5_path"]),
                "out_dir": str(slide_out_dir),
            }
        )

    write_csv(out_dir / "sampled_slides.csv", sampled_rows, ["sample_index", "slide_key", "project", "slide_path", "h5_path", "out_dir"])
    if not explicit_single:
        write_csv(out_dir / "overlay_table.csv", aggregate_overlay_rows, [
            "slide_key",
            "task",
            "class_label",
            "concept_rank",
            "latent_idx",
            "overlay_type",
            "coord_x",
            "coord_y",
            "tile_index",
            "raw_activation",
            "normalized_activation",
            "raw_min",
            "raw_median",
            "raw_p95",
            "raw_max",
            "rank_within_latent",
            "thumbnail_x0",
            "thumbnail_y0",
            "thumbnail_x1",
            "thumbnail_y1",
            "output_path",
        ])
        write_json(
            out_dir / "run_config.json",
            {
                "command": " ".join(shlex.quote(arg) for arg in sys.argv),
                "out_dir": str(out_dir),
                "slide_store_root": str(args.slide_store_root),
                "features_root": str(args.features_root),
                "sample_random_slides": int(args.sample_random_slides),
                "seed": int(args.seed),
                "concept_source": str(args.concept_source),
                "sae_ckpt": str(args.sae_ckpt),
                "sae_cfg": str(args.sae_cfg),
                "n_slides": len(pairs),
                "slides": slide_configs,
            },
        )
    print(f"[overlay-sae-concepts] wrote {out_dir}")


if __name__ == "__main__":
    main()
