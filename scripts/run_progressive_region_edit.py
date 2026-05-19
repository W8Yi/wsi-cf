#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png, write_json
from wsi_cf.common.paths import (
    DEFAULT_HNSCC_PROTOTYPE_NPZ,
    DEFAULT_SAE_CFG,
    DEFAULT_SAE_CKPT,
    DEFAULT_SAE_VARIANT,
    DEFAULT_SHOWCASE_EDIT_MANIFEST,
    DEFAULT_SHOWCASE_OUT_DIR,
    DEFAULT_SHOWCASE_REGION_IMAGE,
    DEFAULT_TASK,
    SAE_VARIANTS,
    resolve_sae_paths,
)
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import RegionBankRow, parse_region_bank_csv
from wsi_cf.eval.hnsc_hpv import load_prototypes, pick_prototype_latent
from wsi_cf.generation.pixcell import (
    build_uni_grid_from_image,
    load_uni2,
    resolve_pixcell_window_config,
    sample_large_pixcell_multidiffusion,
    vae_encode_auto,
)
from wsi_cf.steering.sae_edit import edit_uni_z_grid_with_sae
from wsi_cf.steering.edit_policy import add_edit_policy_args, apply_edit_policy
from wsi_cf.steering.progressive import (
    CENTER_2X2_LOCAL_CELLS,
    EDIT_SUPPORT_CHOICES,
    advance_progressive_state,
    build_history_aware_preserve_map,
    draw_cells_overlay,
    load_progressive_edit_manifest,
    local_edit_support_global_cells,
    make_initial_progressive_state,
    plan_progressive_steps,
    preserve_map_preview,
    window_local_cells,
)
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_decode_latents, sae_encode_features


def _jsonify(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _serialize_args(args: argparse.Namespace) -> dict[str, object]:
    return {str(k): _jsonify(v) for k, v in vars(args).items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical manifest-driven progressive region editor. "
            "The editor automatically plans overlapping PixCell-1024 windows from target grid cells, "
            "tracks edit/visit history, and uses history-aware preservation during diffusion."
        )
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--region-image", type=Path, default=DEFAULT_SHOWCASE_REGION_IMAGE)
    parser.add_argument("--region-bank-csv", type=Path, default=None)
    parser.add_argument("--edit-manifest", type=Path, default=DEFAULT_SHOWCASE_EDIT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SHOWCASE_OUT_DIR)
    add_edit_policy_args(parser)
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument("--direction", type=str, default="hpv_pos", choices=["hpv_pos", "hpv_neg"])
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap on number of manifest runs to execute")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--pix-model-id", type=str, default="StonyBrook-CVLab/PixCell-1024")
    parser.add_argument("--pix-pipeline-id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    parser.add_argument("--vae-model-id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae-subfolder", type=str, default="vae")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--prototype-strength", type=float, default=0.9)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--preserve-edit-strength", type=float, default=0.0)
    parser.add_argument("--preserve-visited-strength", type=float, default=0.84)
    parser.add_argument("--preserve-fresh-context-strength", type=float, default=0.22)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.55)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=0.4)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--sae-ckpt", type=Path, default=None, help=f"Explicit SAE checkpoint override. Defaults to --sae-variant ({DEFAULT_SAE_CKPT}).")
    parser.add_argument("--sae-cfg", type=Path, default=None, help=f"Explicit SAE config override. Defaults to --sae-variant ({DEFAULT_SAE_CFG}).")
    parser.add_argument(
        "--prototype-npz",
        type=Path,
        default=DEFAULT_HNSCC_PROTOTYPE_NPZ,
    )
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--concepts-json", type=Path, default=None, help="Optional selected_concepts.json for generic task concept steering.")
    parser.add_argument("--representative-tiles-csv", type=Path, default=None, help="Representative tile CSV used to infer target activation values for --concepts-json.")
    parser.add_argument("--concept-class-label", type=str, default="", help="Optional concept class label to keep from selected_concepts.json.")
    parser.add_argument("--concept-ranking-method", type=str, default="attention_weighted", choices=["attention_weighted", "activation"])
    parser.add_argument("--concept-target-stat", type=str, default="median", choices=["median", "mean", "q75", "max"])
    parser.add_argument("--concept-target-top-k", type=int, default=5, help="Use only the top K representative tiles per concept to estimate steering target activation; 0 uses all rows.")
    parser.add_argument(
        "--concept-steering-mode",
        type=str,
        default="prototype_vector",
        choices=["prototype_vector", "latent_target"],
        help=(
            "prototype_vector builds a full SAE-code prototype from representative tiles and steers selected cells toward it. "
            "latent_target only clamps the listed concept latent activation and is kept for ablations."
        ),
    )
    parser.add_argument("--max-concepts", type=int, default=0, help="0 means use all concepts in --concepts-json.")
    parser.add_argument(
        "--edit-support",
        type=str,
        default="center_2x2",
        choices=list(EDIT_SUPPORT_CHOICES),
        help="Use border_relaxed to allow region-edge target cells outside the local center 2x2.",
    )
    parser.add_argument("--output-mode", type=str, default="debug", choices=["minimal", "debug"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def infer_grid_shape(z_grid: np.ndarray) -> tuple[int, int]:
    if z_grid.ndim != 3:
        raise ValueError(f"Expected z_grid to have shape [H,W,D], got {z_grid.shape}")
    return int(z_grid.shape[0]), int(z_grid.shape[1])


def build_image_first_region_row(
    *,
    region_image: Path,
    request_region_id: str,
    out_dir: Path,
    grid_step_px: int,
    device: torch.device,
) -> RegionBankRow:
    """Encode a standalone region image and expose it through the RegionBankRow interface."""
    source_img = load_image(str(region_image))
    width, height = source_img.size
    if width % int(grid_step_px) != 0 or height % int(grid_step_px) != 0:
        raise ValueError(
            f"Image-first progressive editing requires image dimensions divisible by grid_step_px={grid_step_px}; "
            f"got {width}x{height} for {region_image}"
        )
    source_dir = out_dir / "_image_first_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    image_path = source_dir / "region.png"
    zgrid_path = source_dir / "region_zgrid.npy"
    save_png(source_img, image_path)
    if not zgrid_path.exists():
        uni_model, uni_transform = load_uni2(device)
        z_grid = build_uni_grid_from_image(
            source_img,
            uni_model=uni_model,
            uni_transform=uni_transform,
            grid_step_px=int(grid_step_px),
            device=device,
            out_dtype=torch.float32,
        )
        np.save(zgrid_path, z_grid.detach().cpu().numpy().astype(np.float32))
        del uni_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    z_grid_np = np.asarray(np.load(zgrid_path), dtype=np.float32)
    return RegionBankRow(
        region_id=str(request_region_id),
        split="showcase",
        label=1,
        hpv_status="HPV+",
        case_id=Path(region_image).stem,
        slide_key=Path(region_image).stem,
        slide_path=str(region_image),
        canonical_h5_path="",
        region_x=0,
        region_y=0,
        region_w=int(width),
        region_h=int(height),
        grid_step_px=int(grid_step_px),
        feature_dim=int(z_grid_np.shape[-1]),
        tissue_score=1.0,
        seed=0,
        image_path=str(image_path),
        feature_grid_path=str(zgrid_path),
        cell_preview_path="",
        region_dir=str(source_dir),
    )


def commit_full_window(
    *,
    current_canvas: np.ndarray,
    steered_img: Image.Image,
    left: int,
    top: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = np.asarray(current_canvas, dtype=np.float32).copy()
    img = np.asarray(steered_img, dtype=np.float32) / 255.0
    height, width = img.shape[:2]
    dst_x0 = int(left)
    dst_y0 = int(top)
    dst_x1 = int(left) + int(width)
    dst_y1 = int(top) + int(height)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img
    return out, (dst_x0, dst_y0, dst_x1, dst_y1)


def update_full_zgrid_selected_cells(
    *,
    full_zgrid: np.ndarray,
    edited_local_zgrid: np.ndarray,
    gx0: int,
    gy0: int,
    selected_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(full_zgrid, dtype=np.float32).copy()
    for lx, ly in selected_cells:
        out[int(gy0) + int(ly), int(gx0) + int(lx), :] = edited_local_zgrid[int(ly), int(lx), :]
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def load_concept_targets(
    *,
    concepts_json: Path,
    representative_tiles_csv: Path | None,
    class_label: str,
    ranking_method: str,
    target_stat: str,
    target_top_k: int,
    max_concepts: int,
) -> tuple[list[int], dict[int, float], dict[str, object]]:
    payload = json.loads(concepts_json.read_text())
    concepts = list(payload.get("concepts", []))
    if class_label:
        concepts = [row for row in concepts if str(row.get("class_label", "")) == str(class_label)]
    concepts.sort(key=lambda row: (int(row.get("concept_rank", 10**9)), -float(row.get("final_score", 0.0)), int(row["latent_idx"])))
    if int(max_concepts) > 0:
        concepts = concepts[: int(max_concepts)]
    if not concepts:
        raise ValueError(f"No concepts found in {concepts_json} for class_label={class_label!r}")
    latent_ids = [int(row["latent_idx"]) for row in concepts]
    values_by_latent: dict[int, list[float]] = {latent: [] for latent in latent_ids}
    if representative_tiles_csv is not None and representative_tiles_csv.exists():
        rows_by_latent: dict[int, list[dict[str, str]]] = {latent: [] for latent in latent_ids}
        for row in read_csv_rows(representative_tiles_csv):
            latent = int(row.get("latent_idx", -1))
            if latent not in rows_by_latent:
                continue
            if str(row.get("ranking_method", "")) != str(ranking_method):
                continue
            rows_by_latent[latent].append(row)
        for latent, rows in rows_by_latent.items():
            rows.sort(key=lambda row: int(row.get("tile_rank", 10**9)))
            if int(target_top_k) > 0:
                rows = rows[: int(target_top_k)]
            for row in rows:
                activation = row.get("activation", "")
                if str(activation).strip():
                    values_by_latent[latent].append(float(activation))
    target_values: dict[int, float] = {}
    for concept in concepts:
        latent = int(concept["latent_idx"])
        vals = np.asarray(values_by_latent.get(latent, []), dtype=np.float32)
        if vals.size:
            if target_stat == "median":
                target = float(np.median(vals))
            elif target_stat == "mean":
                target = float(np.mean(vals))
            elif target_stat == "q75":
                target = float(np.percentile(vals, 75.0))
            else:
                target = float(np.max(vals))
        else:
            # Fallback: association summaries from fraction metrics can be tiny,
            # but this keeps the run defined if representative rows are absent.
            target = float(concept.get("mean_class", concept.get("top_activation", 1.0)))
        target_values[latent] = target
    meta = {
        "concepts_json": str(concepts_json),
        "representative_tiles_csv": "" if representative_tiles_csv is None else str(representative_tiles_csv),
        "class_label": class_label or str(payload.get("class_label", "")),
        "ranking_method": str(ranking_method),
        "target_stat": str(target_stat),
        "target_top_k": int(target_top_k),
        "latent_ids": latent_ids,
        "target_tile_counts": {str(k): int(len(v)) for k, v in values_by_latent.items()},
        "target_values": {str(k): float(v) for k, v in target_values.items()},
    }
    return latent_ids, target_values, meta


def _read_h5_feature(path: Path, tile_index: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{path}: missing dataset 'features'")
        feats = handle["features"]
        if feats.ndim == 3 and feats.shape[0] == 1:
            arr = np.asarray(feats[0, int(tile_index)], dtype=np.float32)
        elif feats.ndim == 2:
            arr = np.asarray(feats[int(tile_index)], dtype=np.float32)
        else:
            raise ValueError(f"{path}: unsupported features shape {tuple(feats.shape)}")
    return arr.astype(np.float32, copy=False)


@torch.no_grad()
def load_concept_prototype_vector(
    *,
    sae_model: torch.nn.Module,
    concepts_json: Path,
    representative_tiles_csv: Path | None,
    class_label: str,
    ranking_method: str,
    target_stat: str,
    target_top_k: int,
    max_concepts: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    if representative_tiles_csv is None or not representative_tiles_csv.exists():
        raise FileNotFoundError(
            "Full concept-prototype steering requires --representative-tiles-csv with source H5 tile references."
        )
    payload = json.loads(concepts_json.read_text())
    concepts = list(payload.get("concepts", []))
    if class_label:
        concepts = [row for row in concepts if str(row.get("class_label", "")) == str(class_label)]
    concepts.sort(key=lambda row: (int(row.get("concept_rank", 10**9)), -float(row.get("final_score", 0.0)), int(row["latent_idx"])))
    if int(max_concepts) > 0:
        concepts = concepts[: int(max_concepts)]
    if not concepts:
        raise ValueError(f"No concepts found in {concepts_json} for class_label={class_label!r}")

    latent_ids = [int(row["latent_idx"]) for row in concepts]
    rows_by_latent: dict[int, list[dict[str, str]]] = {latent: [] for latent in latent_ids}
    for row in read_csv_rows(representative_tiles_csv):
        latent = int(row.get("latent_idx", -1))
        if latent not in rows_by_latent:
            continue
        if str(row.get("ranking_method", "")) != str(ranking_method):
            continue
        rows_by_latent[latent].append(row)

    device = next(sae_model.parameters()).device
    concept_vectors: list[torch.Tensor] = []
    per_concept_meta: list[dict[str, object]] = []
    for concept in concepts:
        latent = int(concept["latent_idx"])
        rows = rows_by_latent.get(latent, [])
        rows.sort(key=lambda row: int(row.get("tile_rank", 10**9)))
        if int(target_top_k) > 0:
            rows = rows[: int(target_top_k)]
        if not rows:
            raise ValueError(
                f"No representative rows for concept latent_idx={latent}, ranking_method={ranking_method!r}; "
                "cannot build full-code concept prototype."
            )

        features = np.stack([_read_h5_feature(Path(str(row["h5_path"])), int(row["tile_index"])) for row in rows], axis=0)
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        z = sae_encode_features(sae_model, x).float()
        if target_stat == "median":
            proto = torch.median(z, dim=0).values
        elif target_stat == "mean":
            proto = torch.mean(z, dim=0)
        elif target_stat == "q75":
            proto = torch.quantile(z, q=0.75, dim=0)
        else:
            proto = torch.max(z, dim=0).values
        concept_vectors.append(proto)
        per_concept_meta.append(
            {
                "concept_rank": int(concept.get("concept_rank", len(per_concept_meta) + 1)),
                "latent_idx": int(latent),
                "source_latent_idx": int(concept.get("source_latent_idx", latent)),
                "representative_tile_count": int(len(rows)),
                "prototype_norm": float(proto.norm().detach().cpu()),
                "prototype_target_activation_at_latent": float(proto[int(latent)].detach().cpu()),
            }
        )

    stacked = torch.stack(concept_vectors, dim=0)
    # Multiple concept cards in one run become a single target SAE-code
    # prototype. Individual-concept scripts pass one concept at a time.
    prototype = torch.mean(stacked, dim=0)
    meta = {
        "concepts_json": str(concepts_json),
        "representative_tiles_csv": str(representative_tiles_csv),
        "class_label": class_label or str(payload.get("class_label", "")),
        "ranking_method": str(ranking_method),
        "target_stat": str(target_stat),
        "target_top_k": int(target_top_k),
        "steering_mode": "prototype_vector",
        "latent_ids": latent_ids,
        "n_concepts": int(len(concepts)),
        "prototype_aggregation": "mean_across_concepts",
        "prototype_norm": float(prototype.norm().detach().cpu()),
        "concept_prototypes": per_concept_meta,
    }
    return prototype.detach(), meta


@torch.no_grad()
def edit_uni_z_grid_with_concept_targets(
    *,
    sae_model: torch.nn.Module,
    z_grid: torch.Tensor,
    latent_target_values: dict[int, float],
    target_strength: float,
    tile_mask: np.ndarray,
    blend: float,
) -> torch.Tensor:
    if z_grid.dim() != 3:
        raise ValueError(f"Expected local z_grid [Gh,Gw,D], got {tuple(z_grid.shape)}")
    gh, gw, d = z_grid.shape
    device = next(sae_model.parameters()).device
    x = z_grid.reshape(gh * gw, d).to(device=device, dtype=torch.float32)
    z_lat = sae_encode_features(sae_model, x)
    mask = np.asarray(tile_mask, dtype=np.float32)
    if mask.shape != (gh, gw):
        raise ValueError(f"tile_mask must be shape {(gh, gw)}, got {mask.shape}")
    sel = np.flatnonzero(mask.reshape(-1) > 0.0)
    if sel.size == 0:
        return z_grid
    sel_t = torch.as_tensor(sel, device=z_lat.device, dtype=torch.long)
    z_edit = z_lat.clone()
    for latent_idx, target_value in latent_target_values.items():
        latent = int(latent_idx)
        if latent < 0 or latent >= z_edit.shape[1]:
            raise ValueError(f"Concept latent_idx={latent} outside SAE latent dim={z_edit.shape[1]}")
        cur = z_edit[sel_t, latent]
        tgt = torch.full_like(cur, float(target_value))
        z_edit[sel_t, latent] = (1.0 - float(target_strength)) * cur + float(target_strength) * tgt
    x_rec = sae_decode_latents(sae_model, z_edit)
    w = torch.from_numpy(mask.reshape(gh * gw, 1)).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    x_new = x * (1.0 - float(blend) * w) + x_rec * (float(blend) * w)
    return x_new.reshape(gh, gw, d).to(device=z_grid.device, dtype=z_grid.dtype)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)
    args = apply_edit_policy(args, parser=parser, argv=raw_argv, root=WSI_CF_ROOT)
    args.sae_ckpt, args.sae_cfg = resolve_sae_paths(args.sae_variant, args.sae_ckpt, args.sae_cfg)
    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    set_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "cli_args": _serialize_args(args),
        "argv": raw_argv,
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + raw_argv)),
        "cwd": str(Path.cwd()),
    }
    write_json(args.out_dir / "experiment_args.json", args_payload)

    edit_requests = load_progressive_edit_manifest(args.edit_manifest)
    if int(args.max_runs) > 0:
        edit_requests = edit_requests[: int(args.max_runs)]
    if not edit_requests:
        raise ValueError("No edit requests found in edit manifest")
    run_id_counts: dict[str, int] = {}
    for request in edit_requests:
        run_id_counts[str(request.run_id)] = run_id_counts.get(str(request.run_id), 0) + 1
    duplicate_run_ids = {run_id for run_id, count in run_id_counts.items() if count > 1}
    if duplicate_run_ids:
        raise ValueError(f"Duplicate run_id values in edit manifest: {sorted(duplicate_run_ids)}")

    if args.region_bank_csv is not None:
        region_rows = parse_region_bank_csv(args.region_bank_csv)
    else:
        first_region_id = str(edit_requests[0].region_id)
        region_rows = [
            build_image_first_region_row(
                region_image=args.region_image,
                request_region_id=first_region_id,
                out_dir=args.out_dir,
                grid_step_px=256,
                device=device,
            )
        ]
    region_by_id = {str(row.region_id): row for row in region_rows}

    sae_model, _, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    concept_latent_targets: dict[int, float] | None = None
    concept_prototype_vector: torch.Tensor | None = None
    concept_meta: dict[str, object] | None = None
    chosen_latent = -1
    if args.concepts_json is not None:
        if str(args.concept_steering_mode) == "prototype_vector":
            concept_prototype_vector, concept_meta = load_concept_prototype_vector(
                sae_model=sae_model,
                concepts_json=args.concepts_json,
                representative_tiles_csv=args.representative_tiles_csv,
                class_label=str(args.concept_class_label),
                ranking_method=str(args.concept_ranking_method),
                target_stat=str(args.concept_target_stat),
                target_top_k=int(args.concept_target_top_k),
                max_concepts=int(args.max_concepts),
            )
        else:
            _, concept_latent_targets, concept_meta = load_concept_targets(
                concepts_json=args.concepts_json,
                representative_tiles_csv=args.representative_tiles_csv,
                class_label=str(args.concept_class_label),
                ranking_method=str(args.concept_ranking_method),
                target_stat=str(args.concept_target_stat),
                target_top_k=int(args.concept_target_top_k),
                max_concepts=int(args.max_concepts),
            )
            if concept_meta is not None:
                concept_meta["steering_mode"] = "latent_target"
    else:
        proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)
        pos_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.pos_latent), direction="hpv_pos")
        neg_latent = pick_prototype_latent(proto_by_latent, direction_by_latent, preferred=int(args.neg_latent), direction="hpv_neg")
        chosen_latent = int(pos_latent if str(args.direction) == "hpv_pos" else neg_latent)

    pipeline = DiffusionPipeline.from_pretrained(
        args.pix_model_id,
        vae=AutoencoderKL.from_pretrained(args.vae_model_id, subfolder=args.vae_subfolder, torch_dtype=dtype),
        custom_pipeline=args.pix_pipeline_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_px, stride_px, cond_grid_side = resolve_pixcell_window_config(
        pix_model_id=args.pix_model_id,
        patch_px=0,
        stride_px=0,
    )

    summary_rows: list[dict[str, object]] = []
    for request in edit_requests:
        row = region_by_id.get(str(request.region_id))
        if row is None:
            raise ValueError(f"Manifest references unknown region_id '{request.region_id}'")

        source_img = load_image(str(row.image_path))
        source_zgrid = np.asarray(np.load(str(row.feature_grid_path)), dtype=np.float32)
        grid_h, grid_w = infer_grid_shape(source_zgrid)
        run_dir = args.out_dir / str(request.run_id)
        final_out_path = run_dir / "generated.png"
        if bool(args.skip_existing) and final_out_path.exists():
            summary_rows.append(
                {
                    "run_id": str(request.run_id),
                    "region_id": str(request.region_id),
                    "output_path": str(final_out_path),
                    "num_targets": int(len(request.target_cells)),
                }
            )
            continue

        planned_steps = plan_progressive_steps(
            target_cells=list(request.target_cells),
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            window_grid_side=4,
            stride_cells=2,
            grid_step_px=int(row.grid_step_px),
            edit_support=str(args.edit_support),
        )
        current_canvas = np.asarray(source_img, dtype=np.float32) / 255.0
        current_zgrid = np.asarray(source_zgrid, dtype=np.float32).copy()
        state = make_initial_progressive_state(target_cells=list(request.target_cells))
        run_dir.mkdir(parents=True, exist_ok=True)
        save_png(source_img, run_dir / "source_region_actual.png")

        if str(args.output_mode) == "debug":
            save_png(
                draw_cells_overlay(source_img, cells=list(request.target_cells), grid_step_px=int(row.grid_step_px)),
                run_dir / "source_targets_overlay.png",
            )

        step_records: list[dict[str, object]] = []
        for step in planned_steps:
            window = step.window
            window_px_w = int(window.grid_w) * int(row.grid_step_px)
            window_px_h = int(window.grid_h) * int(row.grid_step_px)
            local_source_img = Image.fromarray((np.clip(current_canvas, 0.0, 1.0) * 255.0).astype(np.uint8)).crop(
                (
                    int(window.left),
                    int(window.top),
                    int(window.left) + int(window_px_w),
                    int(window.top) + int(window_px_h),
                )
            ).convert("RGB")
            gx0 = int(window.gx0)
            gy0 = int(window.gy0)
            local_base_zgrid = np.asarray(current_zgrid[gy0 : gy0 + 4, gx0 : gx0 + 4, :], dtype=np.float32)
            local_base_t = torch.from_numpy(local_base_zgrid).to(device=device, dtype=torch.float32)
            local_edit_t = local_base_t.clone()

            local_edit_cells = window_local_cells(window=window, global_cells=step.edit_cells_global)
            allowed_local_cells = window_local_cells(
                window=window,
                global_cells=local_edit_support_global_cells(
                    window,
                    grid_w=int(grid_w),
                    grid_h=int(grid_h),
                    edit_support=str(args.edit_support),
                ),
            )
            bad_local_cells = [cell for cell in local_edit_cells if cell not in set(allowed_local_cells)]
            if bad_local_cells:
                raise RuntimeError(
                    f"Planned local edit cells are outside edit_support={args.edit_support}, got {bad_local_cells} "
                    f"for window {window.window_id}. Center support is {CENTER_2X2_LOCAL_CELLS}."
                )
            tile_mask = np.zeros(local_base_zgrid.shape[:2], dtype=np.float32)
            for lx, ly in local_edit_cells:
                tile_mask[int(ly), int(lx)] = 1.0
            if concept_prototype_vector is not None:
                local_edit_t, _ = edit_uni_z_grid_with_sae(
                    sae_model=sae_model,
                    z_grid=local_edit_t,
                    target_latent_vector=concept_prototype_vector,
                    target_latent_vector_strength=float(args.prototype_strength),
                    tile_mask=tile_mask,
                    blend=float(args.steer_blend),
                    keep_non_selected=True,
                    return_debug=False,
                )
            elif concept_latent_targets is not None:
                local_edit_t = edit_uni_z_grid_with_concept_targets(
                    sae_model=sae_model,
                    z_grid=local_edit_t,
                    latent_target_values=concept_latent_targets,
                    target_strength=float(args.prototype_strength),
                    tile_mask=tile_mask,
                    blend=float(args.steer_blend),
                )
            else:
                local_edit_t, _ = edit_uni_z_grid_with_sae(
                    sae_model=sae_model,
                    z_grid=local_edit_t,
                    target_latent_vector=proto_by_latent[int(chosen_latent)],
                    target_latent_vector_strength=float(args.prototype_strength),
                    tile_mask=tile_mask,
                    blend=float(args.steer_blend),
                    keep_non_selected=True,
                    return_debug=False,
                )

            source_np = np.asarray(local_source_img, dtype=np.float32) / 255.0
            source_img_t = torch.from_numpy(source_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
            preserve_source_latents = vae_encode_auto(
                pipeline.vae,
                source_img_t,
                use_tiled=False,
                tile_img=0,
                overlap_img=0,
            )
            preserve_map = build_history_aware_preserve_map(
                width=int(local_source_img.size[0]),
                height=int(local_source_img.size[1]),
                grid_step_px=int(row.grid_step_px),
                window=window,
                edit_cells_global=list(step.edit_cells_global),
                visited_cells_global=list(state.visited_cells),
                preserve_edit_strength=float(args.preserve_edit_strength),
                preserve_visited_strength=float(args.preserve_visited_strength),
                preserve_fresh_context_strength=float(args.preserve_fresh_context_strength),
            ).to(device=device)

            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.seed) + int(step.step_index))
            use_autocast = device.type == "cuda" and dtype == torch.float16
            ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad()
            with torch.inference_mode(), ctx:
                img_t = sample_large_pixcell_multidiffusion(
                    pipeline=pipeline,
                    z_grid=local_base_t.to(device=device, dtype=dtype),
                    scheduled_z_grid=local_edit_t.to(device=device, dtype=dtype),
                    condition_start_ratio=float(args.mid_steer_start_ratio),
                    condition_end_ratio=float(args.mid_steer_end_ratio),
                    condition_alpha_start=float(args.mid_steer_alpha_start),
                    condition_alpha_end=float(args.mid_steer_alpha_end),
                    condition_alpha_schedule=str(args.mid_steer_alpha_schedule),
                    out_h=int(local_source_img.size[1]),
                    out_w=int(local_source_img.size[0]),
                    patch_px=patch_px,
                    stride_px=stride_px,
                    cond_grid_side=cond_grid_side,
                    guidance_scale=float(args.guidance),
                    num_steps=int(args.steps),
                    patch_batch=int(args.patch_batch),
                    strength=0.0,
                    init_latents=None,
                    preserve_source_latents=preserve_source_latents,
                    preserve_strength_map=preserve_map,
                    preserve_outside_strength=float(args.preserve_fresh_context_strength),
                    preserve_edit_strength=float(args.preserve_edit_strength),
                    use_tiled_vae_decode=False,
                    decode_tile_lat=128,
                    decode_overlap_lat=16,
                    generator=generator,
                )

            img_np = (img_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            steered_img = Image.fromarray(img_np)
            current_canvas, commit_box = commit_full_window(
                current_canvas=current_canvas,
                steered_img=steered_img,
                left=int(window.left),
                top=int(window.top),
            )
            current_zgrid = update_full_zgrid_selected_cells(
                full_zgrid=current_zgrid,
                edited_local_zgrid=local_edit_t.detach().cpu().numpy().astype(np.float32),
                gx0=int(window.gx0),
                gy0=int(window.gy0),
                selected_cells=local_edit_cells,
            )
            step_record = {
                "step_index": int(step.step_index),
                "window_id": str(window.window_id),
                "row_index": int(window.row_index),
                "col_index": int(window.col_index),
                "gx0": int(window.gx0),
                "gy0": int(window.gy0),
                "left": int(window.left),
                "top": int(window.top),
                "edit_cells_global": [{"gx": int(gx), "gy": int(gy)} for gx, gy in step.edit_cells_global],
                "edit_cells_local": [{"gx": int(gx), "gy": int(gy)} for gx, gy in local_edit_cells],
                "visited_cells_local_before_step": [
                    {"gx": int(gx), "gy": int(gy)}
                    for gx, gy in window_local_cells(window=window, global_cells=state.visited_cells)
                ],
                "commit_bounds_global": {
                    "x0": int(commit_box[0]),
                    "y0": int(commit_box[1]),
                    "x1": int(commit_box[2]),
                    "y1": int(commit_box[3]),
                },
            }

            if str(args.output_mode) == "debug":
                step_dir = run_dir / "steps" / f"step_{int(step.step_index) + 1:02d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                save_png(local_source_img, step_dir / "source_window.png")
                save_png(draw_cells_overlay(local_source_img, cells=local_edit_cells, grid_step_px=int(row.grid_step_px)), step_dir / "selected_cells_overlay.png")
                save_png(preserve_map_preview(preserve_map), step_dir / "preserve_map.png")
                save_png(steered_img, step_dir / "steered_window.png")
                step_record["source_window_path"] = str(step_dir / "source_window.png")
                step_record["selected_overlay_path"] = str(step_dir / "selected_cells_overlay.png")
                step_record["preserve_map_path"] = str(step_dir / "preserve_map.png")
                step_record["steered_window_path"] = str(step_dir / "steered_window.png")

            step_records.append(step_record)
            state = advance_progressive_state(
                state,
                window=window,
                edit_cells_global=list(step.edit_cells_global),
            )

        final_img = Image.fromarray((np.clip(current_canvas, 0.0, 1.0) * 255.0).astype(np.uint8))
        save_png(final_img, final_out_path)
        if str(args.output_mode) == "debug":
            save_png(
                draw_cells_overlay(final_img, cells=list(request.target_cells), grid_step_px=int(row.grid_step_px)),
                run_dir / "generated_targets_overlay.png",
            )

        run_manifest = {
            "run_id": str(request.run_id),
            "region_id": str(request.region_id),
            "source_image_path": str(row.image_path),
            "source_feature_grid_path": str(row.feature_grid_path),
            "region_size": [int(source_img.size[0]), int(source_img.size[1])],
            "grid_shape": [int(grid_h), int(grid_w)],
            "grid_step_px": int(row.grid_step_px),
            "target_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in request.target_cells],
            "edited_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in state.edited_cells],
            "visited_cells": [{"gx": int(gx), "gy": int(gy)} for gx, gy in state.visited_cells],
            "window_history": list(step_records),
            "target_metadata": request.metadata,
            "prototype_direction": str(args.direction),
            "prototype_latent": int(chosen_latent),
            "prototype_key": str(args.prototype_key),
            "concept_steering": concept_meta,
            "prototype_strength": float(args.prototype_strength),
            "steer_blend": float(args.steer_blend),
            "preserve_edit_strength": float(args.preserve_edit_strength),
            "preserve_visited_strength": float(args.preserve_visited_strength),
            "preserve_fresh_context_strength": float(args.preserve_fresh_context_strength),
            "mid_steer_start_ratio": float(args.mid_steer_start_ratio),
            "mid_steer_end_ratio": float(args.mid_steer_end_ratio),
            "mid_steer_alpha_start": float(args.mid_steer_alpha_start),
            "mid_steer_alpha_end": float(args.mid_steer_alpha_end),
            "mid_steer_alpha_schedule": str(args.mid_steer_alpha_schedule),
            "pix_model_id": str(args.pix_model_id),
            "pix_pipeline_id": str(args.pix_pipeline_id),
            "steps": int(args.steps),
            "guidance": float(args.guidance),
            "seed": int(args.seed),
            "edit_support": str(args.edit_support),
            "output_mode": str(args.output_mode),
            "output_path": str(final_out_path),
            "experiment_args_path": str(args.out_dir / "experiment_args.json"),
            "cli_args": args_payload["cli_args"],
            "command": args_payload["command"],
        }
        write_json(run_dir / "run_manifest.json", run_manifest)
        summary_rows.append(
            {
                "run_id": str(request.run_id),
                "region_id": str(request.region_id),
                "output_path": str(final_out_path),
                "num_targets": int(len(request.target_cells)),
                "num_windows": int(len(step_records)),
            }
        )

    write_json(args.out_dir / "run_summary.json", {"runs": summary_rows})


if __name__ == "__main__":
    main()
