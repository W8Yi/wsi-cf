#!/usr/bin/env python3
"""
Render full-slide attention heatmaps for downloaded HNSC HPV WSIs.

This script:
- finds WSIs already downloaded in wsi/hnsc_hpv
- matches them to H5 feature files via slide_labels_master.tsv
- runs the MIL model on every tile in the H5
- builds a dense thumbnail heatmap from all attention weights
- saves flat outputs into one folder (no per-slide subfolders)

Per slide it saves:
- <slide_key>__original.png
- <slide_key>__heatmap.png
- <slide_key>__overlay.png
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from e

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.classifier import AttentionMIL, GatedAttentionMIL


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(x: object, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def infer_objective_power(slide: "openslide.OpenSlide") -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            val = safe_float(props.get(key), -1.0)
            if val > 0:
                return val
    mpp_x = safe_float(props.get("openslide.mpp-x"), -1.0)
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(tile_size_20x * (objective_power / 20.0))))


def choose_render_level(slide: "openslide.OpenSlide", max_dim: int) -> int:
    w0, h0 = slide.dimensions
    best_level = 0
    best_error = float("inf")
    target = float(max_dim)
    for level, (w, h) in enumerate(slide.level_dimensions):
        current = float(max(w, h))
        error = abs(current - target)
        if current <= target:
            return level
        if error < best_error:
            best_error = error
            best_level = level
    return best_level


def read_h5_features_and_coords(h5_path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        feats = handle["features"]
        coords = handle["coords"]
        if feats.ndim == 3 and feats.shape[0] == 1:
            x = feats[0]
        elif feats.ndim == 2:
            x = feats[:]
        else:
            raise ValueError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")

        if coords.ndim == 3 and coords.shape[0] == 1:
            c = coords[0]
        elif coords.ndim == 2 and coords.shape[1] == 2:
            c = coords[:]
        else:
            raise ValueError(f"{h5_path}: unsupported coords shape {tuple(coords.shape)}")

    x = np.asarray(x, dtype=np.float32)
    c = np.asarray(c)
    if x.shape[0] != c.shape[0]:
        raise ValueError(f"{h5_path}: features N={x.shape[0]} != coords N={c.shape[0]}")
    return x, c


def build_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})

    common = {
        "embed_dim": int(saved_args.get("embed_dim", 1536)),
        "hidden_dim": int(saved_args.get("hidden_dim", 512)),
        "attn_dim": int(saved_args.get("attn_dim", 256)),
        "n_classes": 2,
        "dropout": float(saved_args.get("dropout", 0.25)),
    }
    model_type = saved_args.get("model", "attention")

    if model_type == "gated":
        model = GatedAttentionMIL(
            **common,
            learnable_temperature=(not bool(saved_args.get("fixed_temperature", False))),
            init_temperature=float(saved_args.get("init_temperature", 1.0)),
        )
    else:
        model = AttentionMIL(**common)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def normalize_heat(heat: np.ndarray, n_tiles: int) -> np.ndarray:
    # Scale by uniform-attention baseline, then use a robust high percentile.
    baseline = max(1.0 / max(1, n_tiles), 1e-12)
    scaled = heat / baseline
    hi = float(np.percentile(scaled, 99.5))
    if hi <= 0:
        return np.zeros_like(heat, dtype=np.float32)
    out = np.clip(scaled / hi, 0.0, 1.0).astype(np.float32, copy=False)
    return out


def heat_to_rgb(heat_01: np.ndarray) -> np.ndarray:
    # Simple black -> blue -> cyan -> yellow -> red map without matplotlib.
    h = np.clip(heat_01, 0.0, 1.0)
    r = np.clip(3.0 * h - 1.0, 0.0, 1.0)
    g = np.clip(3.0 * h, 0.0, 1.0) - np.clip(3.0 * h - 2.0, 0.0, 1.0)
    b = np.clip(1.5 - 3.0 * h, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def save_images(
    *,
    original_rgb: np.ndarray,
    heat_rgb: np.ndarray,
    heat_01: np.ndarray,
    out_prefix: Path,
) -> None:
    orig_img = Image.fromarray(original_rgb)
    heat_img = Image.fromarray(heat_rgb)

    alpha = (np.clip(heat_01, 0.0, 1.0) * 180.0).astype(np.uint8)
    overlay_rgba = np.dstack([heat_rgb, alpha])
    overlay = Image.alpha_composite(orig_img.convert("RGBA"), Image.fromarray(overlay_rgba).convert("RGBA")).convert("RGB")

    orig_img.save(out_prefix.with_name(out_prefix.name + "__original.png"))
    heat_img.save(out_prefix.with_name(out_prefix.name + "__heatmap.png"))
    overlay.save(out_prefix.with_name(out_prefix.name + "__overlay.png"))


def render_one(
    *,
    model: torch.nn.Module,
    device: torch.device,
    slide_key: str,
    h5_path: str,
    wsi_path: Path,
    out_dir: Path,
    thumb_max_dim: int,
    tile_size_20x: int,
) -> dict:
    x, coords = read_h5_features_and_coords(h5_path)
    xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits, y_prob, y_hat, a_raw, _ = model(xt)
        attn = F.softmax(a_raw, dim=1).detach().cpu().numpy().reshape(-1)

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        w0, h0 = slide.dimensions
        objective_power = infer_objective_power(slide)
        tile_px = level0_tile_size(tile_size_20x=tile_size_20x, objective_power=objective_power)

        render_level = choose_render_level(slide, thumb_max_dim)
        level_w, level_h = slide.level_dimensions[render_level]
        level_downsample = float(slide.level_downsamples[render_level])
        level_img = slide.read_region((0, 0), render_level, (level_w, level_h)).convert("RGB")
        thumb_np = np.asarray(level_img)
        thumb_w, thumb_h = level_img.size
        scale = 1.0 / max(level_downsample, 1e-12)

        heat = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        for (x0, y0), a in zip(coords, attn):
            x1 = int(round(float(x0) * scale))
            y1 = int(round(float(y0) * scale))
            x2 = int(round((float(x0) + tile_px) * scale))
            y2 = int(round((float(y0) + tile_px) * scale))

            x1 = max(0, min(x1, thumb_w - 1))
            y1 = max(0, min(y1, thumb_h - 1))
            x2 = max(x1 + 1, min(x2, thumb_w))
            y2 = max(y1 + 1, min(y2, thumb_h))

            heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], float(a))

        heat_01 = normalize_heat(heat, n_tiles=attn.shape[0])
        heat_rgb = heat_to_rgb(heat_01)

        out_prefix = out_dir / slide_key
        save_images(
            original_rgb=thumb_np,
            heat_rgb=heat_rgb,
            heat_01=heat_01,
            out_prefix=out_prefix,
        )
    finally:
        slide.close()

    return {
        "slide_key": slide_key,
        "h5_path": h5_path,
        "wsi_path": str(wsi_path),
        "n_tiles": int(attn.shape[0]),
        "render_level": int(render_level),
        "render_downsample": float(level_downsample),
        "pred": int(y_hat.detach().cpu()[0, 0].item()),
        "prob_pos": float(y_prob.detach().cpu()[0, 1].item()),
        "attn_min": float(attn.min()),
        "attn_median": float(np.median(attn)),
        "attn_p95": float(np.percentile(attn, 95)),
        "attn_max": float(attn.max()),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run_dir",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold"),
        help="MIL training run directory.",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="split_0/final.pt",
        help="Checkpoint path relative to run_dir.",
    )
    ap.add_argument(
        "--wsi_dir",
        type=Path,
        default=Path("wsi/hnsc_hpv"),
        help="Directory with downloaded .svs files.",
    )
    ap.add_argument(
        "--labels_tsv",
        type=Path,
        default=Path("metadata/labels/master/slide_labels_master.tsv"),
        help="Slide labels table for slide_key -> h5 mapping.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/hnsc_hpv_attention_mil_5fold/full_attention_heatmaps_split0"),
        help="Flat output directory for all PNGs.",
    )
    ap.add_argument(
        "--num_slides",
        type=int,
        default=10,
        help="Random number of downloaded slides to render.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for slide sampling.",
    )
    ap.add_argument(
        "--thumb_max_dim",
        type=int,
        default=2048,
        help="Max width/height of rendered thumbnails.",
    )
    ap.add_argument(
        "--tile_size_20x",
        type=int,
        default=256,
        help="Tile size used at 20x extraction.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda:0, cpu, or auto.",
    )
    return ap.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt_path = args.run_dir / args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    label_rows = read_tsv(args.labels_tsv)
    slide_to_h5: Dict[str, str] = {}
    for row in label_rows:
        if row["project_dir"] != "TCGA-HNSC":
            continue
        if row["hpv_status"] not in {"HPV+", "HPV-"}:
            continue
        if row["hpv_conflict"] == "1":
            continue
        slide_to_h5[row["slide_key"]] = row["h5_path"]

    downloaded = sorted(p for p in args.wsi_dir.iterdir() if p.is_file() and p.suffix.lower() == ".svs")
    candidates = []
    for p in downloaded:
        slide_key = p.stem
        h5_path = slide_to_h5.get(slide_key)
        if h5_path:
            candidates.append((slide_key, h5_path, p))

    if not candidates:
        raise SystemExit(f"No matching downloaded HNSC HPV slides found in {args.wsi_dir}")

    rng = random.Random(args.seed)
    if len(candidates) > args.num_slides:
        chosen = rng.sample(candidates, args.num_slides)
    else:
        chosen = candidates
    chosen.sort(key=lambda t: t[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model_from_checkpoint(ckpt_path, device=device)

    summary_rows = []
    for slide_key, h5_path, wsi_path in chosen:
        print(f"[render] {slide_key}", flush=True)
        info = render_one(
            model=model,
            device=device,
            slide_key=slide_key,
            h5_path=h5_path,
            wsi_path=wsi_path,
            out_dir=args.out_dir,
            thumb_max_dim=args.thumb_max_dim,
            tile_size_20x=args.tile_size_20x,
        )
        summary_rows.append(info)

    write_csv(
        args.out_dir / "rendered_slides.csv",
        [
            "slide_key",
            "h5_path",
            "wsi_path",
            "n_tiles",
            "render_level",
            "render_downsample",
            "pred",
            "prob_pos",
            "attn_min",
            "attn_median",
            "attn_p95",
            "attn_max",
        ],
        summary_rows,
    )
    print(f"[ok] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
