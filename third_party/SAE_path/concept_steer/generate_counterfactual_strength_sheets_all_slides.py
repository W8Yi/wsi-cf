#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, DiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_steer.run_hnsc_hpv_sae_neuron_pipeline import build_mil_from_checkpoint, run_mil_attention
from utils.diffusion import sample_multidiffusion_from_zgrid, sample_multidiffusion_from_zgrid_with_midref
from utils.sae import load_sae_from_config
from utils.sae_edit import edit_uni_z_grid_with_sae

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide are required for tile extraction.") from e


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "For all test slides: pick top/bottom attention tiles, classify each tile, "
            "steer with prototype vectors across strengths, generate PixCell images, and "
            "save one strength-sweep sheet per tile."
        )
    )
    ap.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.json",
    )
    ap.add_argument(
        "--split-tsv",
        type=Path,
        default=REPO_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.tsv",
    )
    ap.add_argument(
        "--features-root",
        type=Path,
        default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features"),
    )
    ap.add_argument(
        "--slides-root",
        type=Path,
        default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/slides"),
    )
    ap.add_argument(
        "--mil-ckpt",
        type=Path,
        default=REPO_ROOT / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt",
    )
    ap.add_argument(
        "--sae-ckpt",
        type=Path,
        default=REPO_ROOT / "runs/relu_sae_base/relu_final.pt",
    )
    ap.add_argument(
        "--sae-cfg",
        type=Path,
        default=REPO_ROOT / "runs/relu_sae_base/run_config.json",
    )
    ap.add_argument(
        "--prototype-npz",
        type=Path,
        default=REPO_ROOT
        / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/notebook_counterfactual_single_tile_and_area/prototype_vectors_for_selected_sae.npz",
    )
    ap.add_argument(
        "--prototype-key",
        type=str,
        default="prototype_median",
        choices=["prototype_mean", "prototype_median"],
    )
    ap.add_argument("--pos-latent", type=int, default=2645)
    ap.add_argument("--neg-latent", type=int, default=7036)

    ap.add_argument("--top-k", type=int, default=5, help="Top attention tiles per slide.")
    ap.add_argument("--bottom-k", type=int, default=5, help="Bottom attention tiles per slide.")
    ap.add_argument(
        "--strengths",
        type=str,
        default="0.00,0.10,0.20,0.35,0.50,0.70,1.00",
        help="Comma-separated steering strengths.",
    )
    ap.add_argument(
        "--directions",
        type=str,
        default="to_hpv_pos,to_hpv_neg",
        help="Comma-separated subset of: to_hpv_pos,to_hpv_neg",
    )
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--identity-at-zero", action="store_true")
    ap.add_argument(
        "--norm-match-to-orig",
        action="store_true",
        help="Rescale steered feature to original L2 norm (recommended).",
    )
    ap.add_argument(
        "--max-slides",
        type=int,
        default=0,
        help="0 means all test slides.",
    )

    ap.add_argument("--no-generation", action="store_true", help="Skip PixCell generation; only run predictions.")
    ap.add_argument("--tile-size-20x", type=int, default=256)
    ap.add_argument("--out-tile-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--patch-px", type=int, default=256)
    ap.add_argument("--stride-px", type=int, default=128)
    ap.add_argument("--patch-batch", type=int, default=64)
    ap.add_argument(
        "--reference-start-ratio",
        type=float,
        default=0.0,
        help="Used only when --reference-mix > 0 (anchored generation).",
    )
    ap.add_argument(
        "--reference-mix",
        type=float,
        default=0.0,
        help="0.0 = naive generation (no anchor). >0 enables anchored generation.",
    )
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--pixcell-model", type=str, default="StonyBrook-CVLab/PixCell-256")
    ap.add_argument("--pixcell-custom-pipeline", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    ap.add_argument("--vae-model", type=str, default="stabilityai/stable-diffusion-3.5-large")

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/cf_strength_sheets_all_slides",
    )
    return ap.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _parse_float_csv(v: str) -> list[float]:
    out = [float(x.strip()) for x in str(v).split(",") if x.strip()]
    if not out:
        raise ValueError("No values parsed from --strengths")
    return out


def _parse_str_csv(v: str) -> list[str]:
    out = [str(x.strip()) for x in str(v).split(",") if x.strip()]
    if not out:
        raise ValueError("No values parsed from --directions")
    return out


def _parse_slide_key_from_h5_path(h5_path: str) -> str:
    return Path(h5_path).name.split(".")[0]


def _load_split_json_test_map(split_json: Path) -> dict[str, str]:
    payload = json.loads(split_json.read_text())
    out: dict[str, str] = {}
    for p in payload.get("test", []):
        key = _parse_slide_key_from_h5_path(str(p))
        out.setdefault(key, str(p))
    return out


def _resolve_h5_for_slide(slide_key: str, tsv_h5: str, json_h5: str | None, features_root: Path) -> Path | None:
    cands = [
        features_root / f"{slide_key}.h5",
        Path(tsv_h5) if tsv_h5 else None,
        Path(json_h5) if json_h5 else None,
    ]
    for c in cands:
        if c is not None and c.exists():
            return c
    return None


def _load_test_rows(split_json: Path, split_tsv: Path, features_root: Path) -> list[dict[str, Any]]:
    test_map = _load_split_json_test_map(split_json)
    out: list[dict[str, Any]] = []
    with split_tsv.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if str(row.get("split", "")) != "test":
                continue
            slide_key = str(row.get("slide_key", ""))
            if slide_key not in test_map:
                continue
            label = int(row.get("label", -1))
            if label not in (0, 1):
                continue
            h5_path = _resolve_h5_for_slide(
                slide_key=slide_key,
                tsv_h5=str(row.get("h5_path", "")),
                json_h5=test_map.get(slide_key),
                features_root=features_root,
            )
            if h5_path is None:
                continue
            out.append(
                {
                    "slide_key": slide_key,
                    "case_id": str(row.get("case_id", slide_key)),
                    "label": int(label),
                    "h5_path": str(h5_path),
                }
            )
    out.sort(key=lambda r: (int(r["label"]), str(r["slide_key"])))
    return out


def _read_h5_features_coords(h5_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]
        if feats.ndim == 2:
            x = feats[:]
        elif feats.ndim == 3 and int(feats.shape[0]) == 1:
            x = feats[0]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")

        coords = None
        if "coords" in f:
            c = f["coords"]
            if c.ndim == 2 and c.shape[1] == 2:
                coords = c[:]
            elif c.ndim == 3 and int(c.shape[0]) == 1:
                coords = c[0]
    return np.asarray(x, dtype=np.float32), (np.asarray(coords) if coords is not None else None)


def _load_prototypes(npz_path: Path, key: str) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    with np.load(npz_path, allow_pickle=False) as d:
        latent_ids = np.asarray(d["latent_ids"], dtype=np.int64).reshape(-1)
        vecs = np.asarray(d[key], dtype=np.float32)
        dirs = np.asarray(d["selected_direction"]).astype(str)
    table = {int(latent_ids[i]): vecs[i] for i in range(latent_ids.shape[0])}
    dir_table = {int(latent_ids[i]): str(dirs[i]) for i in range(latent_ids.shape[0])}
    return table, dir_table


def _pick_latent(dir_table: dict[int, str], table: dict[int, np.ndarray], preferred: int, direction: str) -> int:
    if preferred in table:
        return int(preferred)
    cands = sorted([lid for lid, d in dir_table.items() if d == direction])
    if not cands:
        raise RuntimeError(f"No prototype latent for direction={direction}")
    return int(cands[0])


def _steer_feature(
    x_feat: np.ndarray,
    *,
    sae_model: torch.nn.Module,
    proto_vec: np.ndarray,
    strength: float,
    blend: float,
    device: torch.device,
    norm_match_to_orig: bool,
) -> np.ndarray:
    z_grid = torch.from_numpy(np.asarray(x_feat, dtype=np.float32)).to(device=device).view(1, 1, -1)
    z_edit, _ = edit_uni_z_grid_with_sae(
        sae_model=sae_model,
        z_grid=z_grid,
        latent_idx=None,
        target_latent_vector=proto_vec,
        target_latent_vector_strength=float(strength),
        blend=float(blend),
        latent_strength=1.0,
        keep_non_selected=True,
        return_debug=True,
    )
    out = z_edit.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    if norm_match_to_orig:
        n0 = float(np.linalg.norm(x_feat))
        n1 = float(np.linalg.norm(out))
        if n0 > 0.0 and n1 > 1e-8:
            out = out * (n0 / n1)
    return out


def _infer_objective_power(slide: "openslide.OpenSlide") -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            try:
                v = float(props.get(key))
            except Exception:
                v = -1.0
            if v > 0:
                return v
    try:
        mpp_x = float(props.get("openslide.mpp-x", -1.0))
    except Exception:
        mpp_x = -1.0
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def _level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(tile_size_20x * (objective_power / 20.0))))


def _crop_tile_rgb(
    slide: "openslide.OpenSlide",
    x: int,
    y: int,
    tile_size_20x: int,
    out_tile_size: int,
) -> Image.Image:
    objective = _infer_objective_power(slide)
    crop_px = _level0_tile_size(tile_size_20x=tile_size_20x, objective_power=objective)
    rgba = slide.read_region((int(x), int(y)), 0, (crop_px, crop_px)).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    if crop_px != out_tile_size:
        rgb = rgb.resize((out_tile_size, out_tile_size), resample=Image.BILINEAR)
    return rgb


def _find_slide_path(slides_root: Path, slide_key: str, cache: dict[str, Path | None]) -> Path | None:
    if slide_key in cache:
        return cache[slide_key]
    for ext in (".svs", ".tif", ".tiff", ".ndpi", ".mrxs"):
        p = slides_root / f"{slide_key}{ext}"
        if p.exists():
            cache[slide_key] = p
            return p
    for ext in ("*.svs", "*.tif", "*.tiff", "*.ndpi", "*.mrxs"):
        hits = sorted(slides_root.glob(f"{slide_key}*{ext[1:]}"))
        if hits:
            cache[slide_key] = hits[0]
            return hits[0]
    cache[slide_key] = None
    return None


def _pick_top_bottom_indices(attn: np.ndarray, top_k: int, bottom_k: int) -> list[tuple[str, int, int]]:
    n = int(attn.shape[0])
    if n == 0:
        return []
    top_k = max(0, int(top_k))
    bottom_k = max(0, int(bottom_k))
    order_desc = np.argsort(attn)[::-1]
    rank_desc = np.empty(n, dtype=np.int64)
    rank_desc[order_desc] = np.arange(1, n + 1)

    top_idx = [int(i) for i in order_desc[: min(top_k, n)]]
    top_set = set(top_idx)

    bottom_idx: list[int] = []
    if bottom_k > 0:
        order_asc = np.argsort(attn)
        for i in order_asc.tolist():
            ii = int(i)
            if ii in top_set:
                continue
            bottom_idx.append(ii)
            if len(bottom_idx) >= bottom_k:
                break

    out: list[tuple[str, int, int]] = []
    for i in top_idx:
        out.append(("top", i, int(rank_desc[i])))
    for i in bottom_idx:
        out.append(("bottom", i, int(rank_desc[i])))
    return out


def _to_uint8_image(img_t: torch.Tensor) -> Image.Image:
    arr = (img_t[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _strength_tag(s: float) -> str:
    return f"{s:.2f}".replace(".", "p")


def _render_strength_sheet(
    *,
    raw_tile: Image.Image,
    directions: list[str],
    strengths: list[float],
    generated: dict[tuple[str, float], Image.Image],
    metrics: dict[tuple[str, float], dict[str, float | int]],
    slide_key: str,
    slide_label: int,
    tile_idx: int,
    tile_rank: int,
    tile_group: str,
    tile_attention: float,
    tile_pred_orig: int,
    tile_prob_orig: float,
    slide_pred_orig: int,
    slide_prob_orig: float,
    out_path: Path,
) -> None:
    tile_px = raw_tile.width
    pad = 8
    caption_h = 46
    header_h = 58
    cols = 1 + len(strengths)
    rows = max(1, len(directions))

    cell_w = tile_px
    cell_h = caption_h + tile_px
    canvas_w = pad + cols * (cell_w + pad)
    canvas_h = header_h + pad + rows * (cell_h + pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (242, 242, 242))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, canvas_w, header_h], fill=(225, 225, 225))
    title1 = (
        f"{slide_key} | slide_label={slide_label} | tile={tile_idx} ({tile_group}, rank={tile_rank}) "
        f"| attn={tile_attention:.6f}"
    )
    title2 = (
        f"tile_orig pred={tile_pred_orig} p_pos={tile_prob_orig:.4f} | "
        f"slide_orig pred={slide_pred_orig} p_pos={slide_prob_orig:.4f}"
    )
    draw.text((8, 6), title1, fill=(0, 0, 0))
    draw.text((8, 30), title2, fill=(0, 0, 0))

    for r, direction in enumerate(directions):
        y_base = header_h + pad + r * (cell_h + pad)
        for c in range(cols):
            x_base = pad + c * (cell_w + pad)
            draw.rectangle([x_base, y_base, x_base + cell_w, y_base + caption_h], fill=(255, 255, 255))
            if c == 0:
                caption = (
                    f"{direction}\n"
                    f"orig pred={tile_pred_orig} p={tile_prob_orig:.3f}"
                )
                img = raw_tile
            else:
                s = float(strengths[c - 1])
                key = (direction, s)
                img = generated[key]
                m = metrics[key]
                caption = (
                    f"{direction} s={s:.2f}\n"
                    f"pred={int(m['pred_cf'])} p={float(m['prob_cf']):.3f} "
                    f"d={float(m['delta_prob']):+.3f}"
                )
            draw.multiline_text((x_base + 4, y_base + 3), caption, fill=(0, 0, 0), spacing=2)
            canvas.paste(img, (x_base, y_base + caption_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    strengths = _parse_float_csv(args.strengths)
    directions = _parse_str_csv(args.directions)

    valid_dirs = {"to_hpv_pos", "to_hpv_neg"}
    for d in directions:
        if d not in valid_dirs:
            raise ValueError(f"Unsupported direction {d}. Allowed: {sorted(valid_dirs)}")

    required = [
        args.split_json,
        args.split_tsv,
        args.features_root,
        args.slides_root,
        args.mil_ckpt,
        args.sae_ckpt,
        args.sae_cfg,
        args.prototype_npz,
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))

    print(f"[setup] device={device}")
    print(f"[setup] loading rows from {args.split_tsv}")
    rows = _load_test_rows(args.split_json, args.split_tsv, args.features_root)
    if args.max_slides > 0:
        rows = rows[: int(args.max_slides)]
    if not rows:
        raise SystemExit("No test slides found.")
    n0 = sum(1 for r in rows if int(r["label"]) == 0)
    n1 = sum(1 for r in rows if int(r["label"]) == 1)
    print(f"[setup] test slides: {len(rows)} (label0={n0}, label1={n1})")

    print(f"[setup] loading MIL: {args.mil_ckpt}")
    mil_model = build_mil_from_checkpoint(args.mil_ckpt, device)
    print(f"[setup] loading SAE: {args.sae_ckpt}")
    sae_model, d_in, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))

    proto_table, dir_table = _load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = _pick_latent(dir_table, proto_table, args.pos_latent, "hpv_pos")
    neg_latent = _pick_latent(dir_table, proto_table, args.neg_latent, "hpv_neg")
    proto_pos = np.asarray(proto_table[pos_latent], dtype=np.float32)
    proto_neg = np.asarray(proto_table[neg_latent], dtype=np.float32)
    print(f"[setup] prototype latents: hpv_pos={pos_latent}, hpv_neg={neg_latent}")

    do_generation = not bool(args.no_generation)
    pipe = None
    if do_generation:
        if device.type != "cuda":
            raise RuntimeError("PixCell generation requires CUDA in this script. Use --no-generation to disable.")
        print("[setup] loading PixCell pipeline")
        sd3_vae = AutoencoderKL.from_pretrained(args.vae_model, subfolder="vae")
        pipe = DiffusionPipeline.from_pretrained(
            args.pixcell_model,
            vae=sd3_vae,
            custom_pipeline=args.pixcell_custom_pipeline,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        pipe.to(str(device))
        pipe_dtype = next(pipe.transformer.parameters()).dtype
    else:
        pipe_dtype = torch.float32

    all_records: list[dict[str, Any]] = []
    selected_tiles_rows: list[dict[str, Any]] = []
    failed_slides: list[dict[str, Any]] = []
    slide_path_cache: dict[str, Path | None] = {}

    for si, row in enumerate(rows, start=1):
        slide_key = str(row["slide_key"])
        slide_label = int(row["label"])
        case_id = str(row["case_id"])
        h5_path = str(row["h5_path"])
        print(f"[slide {si}/{len(rows)}] {slide_key} label={slide_label}")

        try:
            x, coords = _read_h5_features_coords(h5_path)
            if x.ndim != 2 or x.shape[1] != d_in:
                raise RuntimeError(f"Feature shape {x.shape} incompatible with SAE d_in={d_in}")

            attn, slide_pred_orig, slide_prob_orig = run_mil_attention(mil_model, x, device=device)
            picks = _pick_top_bottom_indices(attn, top_k=int(args.top_k), bottom_k=int(args.bottom_k))
            if not picks:
                raise RuntimeError("No tile indices selected.")

            slide_dir = args.out_dir / slide_key
            slide_dir.mkdir(parents=True, exist_ok=True)

            slide_path = _find_slide_path(args.slides_root, slide_key, slide_path_cache)
            slide_obj = openslide.OpenSlide(str(slide_path)) if (slide_path is not None and coords is not None) else None
            try:
                for local_i, (tile_group, tile_idx, tile_rank) in enumerate(picks, start=1):
                    x_orig = np.asarray(x[tile_idx], dtype=np.float32)
                    _, tile_pred_orig, tile_prob_orig = run_mil_attention(mil_model, x_orig.reshape(1, -1), device=device)
                    coord_x = int(coords[tile_idx, 0]) if coords is not None else -1
                    coord_y = int(coords[tile_idx, 1]) if coords is not None else -1
                    tile_attn = float(attn[tile_idx])

                    tile_tag = f"{tile_group}_rank{tile_rank:05d}_tile{tile_idx:06d}"
                    tile_dir = slide_dir / tile_tag
                    tile_dir.mkdir(parents=True, exist_ok=True)

                    if slide_obj is not None and coord_x >= 0 and coord_y >= 0:
                        raw_tile = _crop_tile_rgb(
                            slide_obj,
                            x=coord_x,
                            y=coord_y,
                            tile_size_20x=int(args.tile_size_20x),
                            out_tile_size=int(args.out_tile_size),
                        )
                    else:
                        raw_tile = Image.new("RGB", (int(args.out_tile_size), int(args.out_tile_size)), (255, 255, 255))
                    raw_tile_path = tile_dir / "raw_tile.png"
                    raw_tile.save(raw_tile_path)

                    selected_tiles_rows.append(
                        {
                            "slide_key": slide_key,
                            "case_id": case_id,
                            "slide_label": int(slide_label),
                            "slide_pred_orig": int(slide_pred_orig),
                            "slide_prob_orig": float(slide_prob_orig),
                            "tile_group": tile_group,
                            "tile_index": int(tile_idx),
                            "tile_attention_rank": int(tile_rank),
                            "tile_attention": float(tile_attn),
                            "coord_x": int(coord_x),
                            "coord_y": int(coord_y),
                            "tile_pred_orig": int(tile_pred_orig),
                            "tile_prob_orig": float(tile_prob_orig),
                            "raw_tile_path": str(raw_tile_path),
                        }
                    )

                    generated_imgs: dict[tuple[str, float], Image.Image] = {}
                    generated_metrics: dict[tuple[str, float], dict[str, float | int]] = {}

                    for di, direction in enumerate(directions):
                        proto_vec = proto_pos if direction == "to_hpv_pos" else proto_neg
                        for sj, strength in enumerate(strengths):
                            s = float(strength)
                            if bool(args.identity_at_zero) and abs(s) < 1e-12:
                                x_cf = x_orig.copy()
                            else:
                                x_cf = _steer_feature(
                                    x_orig,
                                    sae_model=sae_model,
                                    proto_vec=proto_vec,
                                    strength=s,
                                    blend=float(args.blend),
                                    device=device,
                                    norm_match_to_orig=bool(args.norm_match_to_orig),
                                )

                            _, tile_pred_cf, tile_prob_cf = run_mil_attention(mil_model, x_cf.reshape(1, -1), device=device)
                            delta_prob = float(tile_prob_cf - tile_prob_orig)
                            gen_path = ""
                            if do_generation and pipe is not None and slide_obj is not None and coord_x >= 0 and coord_y >= 0:
                                ref_np = np.asarray(raw_tile).astype(np.float32) / 255.0
                                ref_t = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
                                z_uni = torch.from_numpy(np.asarray(x_cf, dtype=np.float32)).to(device=device, dtype=pipe_dtype).view(1, 1, -1)
                                seed = int(args.seed + si * 100000 + local_i * 1000 + di * 100 + sj)
                                g = torch.Generator(device=device).manual_seed(seed)
                                ac = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                                with torch.inference_mode(), ac:
                                    if float(args.reference_mix) <= 0.0:
                                        img_t = sample_multidiffusion_from_zgrid(
                                            pipeline=pipe,
                                            z_grid=z_uni,
                                            out_h=int(args.out_tile_size),
                                            out_w=int(args.out_tile_size),
                                            patch_px=int(args.patch_px),
                                            stride_px=int(args.stride_px),
                                            steps=int(args.steps),
                                            guidance=float(args.guidance),
                                            patch_batch=int(args.patch_batch),
                                            generator=g,
                                        )
                                    else:
                                        img_t = sample_multidiffusion_from_zgrid_with_midref(
                                            pipeline=pipe,
                                            z_grid=z_uni,
                                            original_image=ref_t,
                                            out_h=int(args.out_tile_size),
                                            out_w=int(args.out_tile_size),
                                            patch_px=int(args.patch_px),
                                            stride_px=int(args.stride_px),
                                            steps=int(args.steps),
                                            guidance=float(args.guidance),
                                            patch_batch=int(args.patch_batch),
                                            reference_start_ratio=float(args.reference_start_ratio),
                                            reference_mix=float(args.reference_mix),
                                            generator=g,
                                        )
                                img = _to_uint8_image(img_t)
                                gen_name = f"{direction}__s_{_strength_tag(s)}__pred_{int(tile_pred_cf)}__p_{_strength_tag(float(tile_prob_cf))}.png"
                                gen_path = str(tile_dir / gen_name)
                                img.save(gen_path)
                            else:
                                img = raw_tile.copy()

                            generated_imgs[(direction, s)] = img
                            generated_metrics[(direction, s)] = {
                                "pred_cf": int(tile_pred_cf),
                                "prob_cf": float(tile_prob_cf),
                                "delta_prob": float(delta_prob),
                            }

                            all_records.append(
                                {
                                    "slide_key": slide_key,
                                    "case_id": case_id,
                                    "slide_label": int(slide_label),
                                    "h5_path": h5_path,
                                    "slide_pred_orig": int(slide_pred_orig),
                                    "slide_prob_orig": float(slide_prob_orig),
                                    "tile_group": tile_group,
                                    "tile_index": int(tile_idx),
                                    "tile_attention_rank": int(tile_rank),
                                    "tile_attention": float(tile_attn),
                                    "coord_x": int(coord_x),
                                    "coord_y": int(coord_y),
                                    "direction": direction,
                                    "strength": float(s),
                                    "tile_pred_orig": int(tile_pred_orig),
                                    "tile_prob_orig": float(tile_prob_orig),
                                    "tile_pred_cf": int(tile_pred_cf),
                                    "tile_prob_cf": float(tile_prob_cf),
                                    "tile_delta_prob_cf_minus_orig": float(delta_prob),
                                    "prototype_latent_pos": int(pos_latent),
                                    "prototype_latent_neg": int(neg_latent),
                                    "raw_tile_path": str(raw_tile_path),
                                    "generated_image_path": gen_path,
                                }
                            )

                    sheet_path = tile_dir / "strength_sweep_sheet.png"
                    _render_strength_sheet(
                        raw_tile=raw_tile,
                        directions=directions,
                        strengths=strengths,
                        generated=generated_imgs,
                        metrics=generated_metrics,
                        slide_key=slide_key,
                        slide_label=slide_label,
                        tile_idx=int(tile_idx),
                        tile_rank=int(tile_rank),
                        tile_group=tile_group,
                        tile_attention=float(tile_attn),
                        tile_pred_orig=int(tile_pred_orig),
                        tile_prob_orig=float(tile_prob_orig),
                        slide_pred_orig=int(slide_pred_orig),
                        slide_prob_orig=float(slide_prob_orig),
                        out_path=sheet_path,
                    )
            finally:
                if slide_obj is not None:
                    slide_obj.close()
        except Exception as e:
            failed_slides.append({"slide_key": slide_key, "error": str(e)})
            print(f"[warn] failed slide {slide_key}: {e}")
            continue

    records_csv = args.out_dir / "tile_counterfactual_strength_sweep_records.csv"
    selected_csv = args.out_dir / "selected_tiles_top_bottom_attention.csv"
    _write_csv(records_csv, all_records)
    _write_csv(selected_csv, selected_tiles_rows)

    summary = {
        "n_test_slides_requested": int(len(rows)),
        "n_failed_slides": int(len(failed_slides)),
        "n_success_slides": int(len(rows) - len(failed_slides)),
        "n_selected_tiles": int(len(selected_tiles_rows)),
        "n_records": int(len(all_records)),
        "top_k": int(args.top_k),
        "bottom_k": int(args.bottom_k),
        "strengths": [float(s) for s in strengths],
        "directions": directions,
        "prototype_key": str(args.prototype_key),
        "prototype_latent_pos": int(pos_latent),
        "prototype_latent_neg": int(neg_latent),
        "identity_at_zero": bool(args.identity_at_zero),
        "norm_match_to_orig": bool(args.norm_match_to_orig),
        "generation_enabled": bool(do_generation),
        "device": str(device),
        "records_csv": str(records_csv),
        "selected_tiles_csv": str(selected_csv),
        "failed_slides": failed_slides,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[done] wrote:")
    print(f"  {records_csv}")
    print(f"  {selected_csv}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
