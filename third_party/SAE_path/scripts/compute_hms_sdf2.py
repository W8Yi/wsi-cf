#!/usr/bin/env python3
"""
Compute HMS-style diagnostics for a trained SDFSAE2Level checkpoint.

Outputs:
- Dictionary approximation metrics (W vs U @ A^T)
- Dictionary HMS (within-parent pairwise cosine of L1 decoder atoms)
- Optional data-driven HMS using external GigaPath embeddings from tile images
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.sae import InputNormWrapper, SDFSAE2Level  # noqa: E402
from utils.uni import get_uni  # noqa: E402


COORD_RE = re.compile(r"_x(\d+)_y(\d+)", re.IGNORECASE)
IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
WSI_SUFFIXES = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")
UNI_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class HMSSource:
    slide_name: str
    h5_path: Optional[Path] = None
    tile_dir: Optional[Path] = None
    wsi_path: Optional[Path] = None
    locator_coords: Optional[np.ndarray] = None
    locator_level0_tile_size: Optional[int] = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run_dir",
        type=Path,
        default=Path("runs/relu_sae_sdf2"),
        help="Run directory containing sdf2 checkpoints.",
    )
    ap.add_argument(
        "--ckpt_path",
        type=Path,
        default=None,
        help="Explicit checkpoint path. If omitted, uses <run_dir>/sdf2_final.pt.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Compute device for SAE/GigaPath inference ("auto", "cpu", "cuda:0", ...).',
    )
    ap.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Output JSON path. Default: <run_dir>/hms_report.json",
    )
    ap.add_argument("--min_children_per_parent", type=int, default=2)
    ap.add_argument("--alive_eps", type=float, default=1e-4)
    ap.add_argument("--near_zero_eps", type=float, default=1e-4)

    # Data-driven HMS options
    ap.add_argument("--run_data_hms", action="store_true", help="Enable data-driven HMS.")
    ap.add_argument(
        "--tile_root",
        type=Path,
        default=None,
        help="Directory of exported tile images. Expected layout: <tile_root>/<slide_name>/*.jpg",
    )
    ap.add_argument(
        "--h5_root",
        type=Path,
        default=None,
        help="Directory of H5 feature files matching tile slide names.",
    )
    ap.add_argument(
        "--encode_uni_on_the_fly",
        action="store_true",
        help="For locator mode, encode UNI2-h features directly from WSI patches instead of loading from H5.",
    )
    ap.add_argument(
        "--locator_csv_dir",
        type=Path,
        default=None,
        help="Directory of locator CSVs (from sample_wsi_patches_20x_coverage.py locators/).",
    )
    ap.add_argument(
        "--wsi_dir",
        type=Path,
        default='/common/users/wq50/SAE_path/wsi/tcga_balanced_core_subset10',
        help="Directory of WSIs for locator mode (fallback if CSV wsi_path is missing).",
    )
    ap.add_argument(
        "--wsi_reader",
        type=str,
        default="auto",
        choices=["auto", "cucim", "openslide"],
        help="WSI backend for on-the-fly patch reading in locator mode.",
    )
    ap.add_argument(
        "--wsi_patch_size_20x",
        type=int,
        default=256,
        help="Target 20x tile size. Used if locator CSV has no level0_tile_size.",
    )
    ap.add_argument(
        "--gigapath_model",
        type=str,
        default="hf_hub:prov-gigapath/prov-gigapath",
        help="timm model id for GigaPath tile encoder.",
    )
    ap.add_argument("--gigapath_batch", type=int, default=64)
    ap.add_argument("--tile_batch", type=int, default=512, help="Tile batch size for SAE pass.")
    ap.add_argument("--max_slides", type=int, default=0, help="0 means no cap.")
    ap.add_argument("--max_tiles", type=int, default=0, help="0 means no cap.")
    ap.add_argument(
        "--progress_every_batches",
        type=int,
        default=20,
        help="Print progress every N batches inside first/second data passes. Set <=0 to disable periodic progress.",
    )

    return ap.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def infer_coeff_flags(run_dir: Path) -> Tuple[bool, bool]:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return True, False
    try:
        cfg = json.loads(cfg_path.read_text())
        return bool(cfg.get("sdf_coeff_nonneg", True)), bool(cfg.get("sdf_coeff_simplex", False))
    except Exception:
        return True, False


def load_sdf2_model(ckpt_path: Path, run_dir: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' state_dict: {ckpt_path}")
    state = ckpt["model"]

    d_in = int(state["enc.weight"].shape[1])
    d_latent = int(state["enc.weight"].shape[0])
    d_level2 = int(state["U"].shape[1])
    tied = "dec.weight" not in state
    coeff_nonneg, coeff_simplex = infer_coeff_flags(run_dir)

    model = SDFSAE2Level(
        d_in=d_in,
        d_latent=d_latent,
        d_level2=d_level2,
        tied=tied,
        use_pre_bias=("b_pre" in state),
        coeff_nonneg=coeff_nonneg,
        coeff_simplex=coeff_simplex,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    wrapper_ln = bool(ckpt.get("wrapper_layernorm", False))
    eval_model = InputNormWrapper(d_in, model).to(device).eval() if wrapper_ln else model

    meta = {
        "ckpt_path": str(ckpt_path.resolve()),
        "d_in": d_in,
        "d_latent": d_latent,
        "d_level2": d_level2,
        "tied": bool(tied),
        "coeff_nonneg": bool(coeff_nonneg),
        "coeff_simplex": bool(coeff_simplex),
        "wrapper_layernorm": bool(wrapper_ln),
        "step": ckpt.get("step"),
        "epoch": ckpt.get("epoch"),
        "best_test_mse": ckpt.get("best_test_mse"),
    }
    return model, eval_model, meta


def mean_pairwise_cos(x: torch.Tensor) -> Optional[float]:
    if x.ndim != 2 or x.shape[0] < 2:
        return None
    x = F.normalize(x, p=2, dim=1)
    sim = x @ x.t()
    idx = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1, device=sim.device)
    if idx.shape[1] == 0:
        return None
    return float(sim[idx[0], idx[1]].mean().item())


def summarize_scores(scores: Sequence[float]) -> Dict[str, float | int]:
    if not scores:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_parent_assignment(model: SDFSAE2Level) -> torch.Tensor:
    with torch.no_grad():
        return model.parent_assignment().detach().cpu()


def parse_coords_from_filename(name: str) -> Optional[Tuple[int, int]]:
    m = COORD_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def infer_level0_scale_20x_from_wsi(wsi_path: Path) -> float:
    try:
        import openslide
    except Exception:
        return 1.0
    slide = openslide.OpenSlide(str(wsi_path))
    obj_raw = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if obj_raw:
        try:
            obj = float(obj_raw)
            if obj > 0:
                return max(0.25, obj / 20.0)
        except ValueError:
            pass
    mpp_raw = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp_raw:
        try:
            mpp = float(mpp_raw)
            if mpp > 0:
                return max(0.25, 0.5 / mpp)
        except ValueError:
            pass
    return 1.0


def resolve_h5_for_slide(h5_root: Path, slide_name: str) -> Optional[Path]:
    exact = h5_root / f"{slide_name}.h5"
    if exact.exists():
        return exact
    cands = sorted(h5_root.glob(f"{slide_name}*.h5"))
    if cands:
        return cands[0]
    cands = sorted(h5_root.rglob(f"{slide_name}*.h5"))
    if cands:
        return cands[0]
    return None


def resolve_wsi_path(slide_name: str, csv_wsi_path: str, wsi_dir: Optional[Path]) -> Optional[Path]:
    if csv_wsi_path:
        p = Path(csv_wsi_path)
        if p.exists():
            return p
    if wsi_dir is None:
        return None
    for ext in WSI_SUFFIXES:
        p = wsi_dir / f"{slide_name}{ext}"
        if p.exists():
            return p
    cands: List[Path] = []
    for ext in WSI_SUFFIXES:
        cands.extend(sorted(wsi_dir.glob(f"{slide_name}*{ext}")))
    return cands[0] if cands else None


def discover_tile_sources(tile_root: Path, h5_root: Path, max_slides: int) -> List[HMSSource]:
    h5_by_prefix: Dict[str, Path] = {}
    for h5 in sorted(h5_root.rglob("*.h5")):
        prefix = h5.stem.split(".")[0]
        if prefix not in h5_by_prefix:
            h5_by_prefix[prefix] = h5

    out: List[HMSSource] = []
    for d in sorted(tile_root.iterdir()):
        if not d.is_dir():
            continue
        h5 = h5_by_prefix.get(d.name)
        if h5 is None:
            cands = sorted(h5_root.glob(f"{d.name}*.h5"))
            if cands:
                h5 = cands[0]
        if h5 is None:
            continue
        out.append(HMSSource(slide_name=d.name, tile_dir=d, h5_path=h5))
        if max_slides > 0 and len(out) >= max_slides:
            break
    return out


def discover_locator_sources(
    locator_csv_dir: Path,
    h5_root: Optional[Path],
    wsi_dir: Optional[Path],
    max_slides: int,
    wsi_patch_size_20x: int,
    require_h5: bool = True,
) -> List[HMSSource]:
    out: List[HMSSource] = []
    csv_files = sorted(locator_csv_dir.glob("*.csv"))
    for csv_path in csv_files:
        slide_name = csv_path.stem
        h5_path: Optional[Path] = None
        if h5_root is not None:
            h5_path = resolve_h5_for_slide(h5_root, slide_name)
        if require_h5 and h5_path is None:
            continue

        coords: List[Tuple[int, int]] = []
        level0_sizes: List[int] = []
        csv_wsi = ""
        with csv_path.open("r", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                x_raw = (row.get("x") or "").strip()
                y_raw = (row.get("y") or "").strip()
                if not x_raw or not y_raw:
                    continue
                try:
                    x = int(float(x_raw))
                    y = int(float(y_raw))
                except ValueError:
                    continue
                coords.append((x, y))
                l0 = (row.get("level0_tile_size") or "").strip()
                if l0:
                    try:
                        level0_sizes.append(int(float(l0)))
                    except ValueError:
                        pass
                if not csv_wsi:
                    csv_wsi = (row.get("wsi_path") or "").strip()
        if not coords:
            continue

        wsi_path = resolve_wsi_path(slide_name=slide_name, csv_wsi_path=csv_wsi, wsi_dir=wsi_dir)
        if wsi_path is None:
            continue

        if level0_sizes:
            level0_tile_size = int(round(float(np.median(np.asarray(level0_sizes, dtype=np.float32)))))
        else:
            scale = infer_level0_scale_20x_from_wsi(wsi_path)
            level0_tile_size = max(1, int(round(float(wsi_patch_size_20x) * float(scale))))

        uniq = np.asarray(sorted(set(coords)), dtype=np.int64)
        out.append(
            HMSSource(
                slide_name=slide_name,
                h5_path=h5_path,
                wsi_path=wsi_path,
                locator_coords=uniq,
                locator_level0_tile_size=level0_tile_size,
            )
        )
        if max_slides > 0 and len(out) >= max_slides:
            break
    return out


def load_h5_coords_feats(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("h5py is required for --run_data_hms mode.") from exc

    with h5py.File(h5_path, "r") as hf:
        if "coords" in hf:
            coords = hf["coords"][:]
        elif "coords_patching" in hf:
            coords = hf["coords_patching"][:]
        else:
            raise KeyError(f"No coords or coords_patching in {h5_path}")
        feats = hf["features"][:]
    if coords.ndim == 3:
        coords = coords[0]
    if feats.ndim == 3:
        feats = feats[0]
    coords = np.asarray(coords, dtype=np.int64)
    feats = np.asarray(feats, dtype=np.float32)
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(f"Coords/features row mismatch in {h5_path}: {coords.shape} vs {feats.shape}")
    return coords, feats


def iter_tile_batches(
    tile_dir: Path,
    h5_path: Path,
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, List[Path]]]:
    tile_map: Dict[Tuple[int, int], Path] = {}
    for p in sorted(tile_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_SUFFIXES:
            continue
        c = parse_coords_from_filename(p.name)
        if c is None:
            continue
        tile_map[c] = p
    if not tile_map:
        return

    coords, feats = load_h5_coords_feats(h5_path)
    h5_idx = {(int(c[0]), int(c[1])): i for i, c in enumerate(coords)}

    matched: List[Tuple[int, Path]] = []
    for c, p in tile_map.items():
        i = h5_idx.get(c)
        if i is None:
            continue
        matched.append((i, p))
    if not matched:
        return

    matched.sort(key=lambda x: x[0])
    idx_arr = np.asarray([m[0] for m in matched], dtype=np.int64)
    paths = [m[1] for m in matched]

    for start in range(0, len(idx_arr), batch_size):
        sel = idx_arr[start:start + batch_size]
        batch_feats = torch.from_numpy(feats[sel]).float()
        batch_paths = paths[start:start + batch_size]
        yield batch_feats, batch_paths


def iter_locator_batches(
    locator_coords: np.ndarray,
    h5_path: Path,
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, np.ndarray]]:
    coords, feats = load_h5_coords_feats(h5_path)
    h5_idx = {(int(c[0]), int(c[1])): i for i, c in enumerate(coords)}

    matched: List[Tuple[int, Tuple[int, int]]] = []
    for c in locator_coords.tolist():
        cc = (int(c[0]), int(c[1]))
        i = h5_idx.get(cc)
        if i is None:
            continue
        matched.append((i, cc))
    if not matched:
        return

    matched.sort(key=lambda x: x[0])
    idx_arr = np.asarray([m[0] for m in matched], dtype=np.int64)
    coord_arr = np.asarray([m[1] for m in matched], dtype=np.int64)
    for start in range(0, len(idx_arr), batch_size):
        sel = idx_arr[start:start + batch_size]
        batch_feats = torch.from_numpy(feats[sel]).float()
        batch_coords = coord_arr[start:start + batch_size]
        yield batch_feats, batch_coords


def preprocess_batch_for_uni(
    batch_np: np.ndarray,
    *,
    tile_size_20x: int,
    device: str,
) -> torch.Tensor:
    x = torch.from_numpy(batch_np).to(device=device, dtype=torch.uint8, non_blocking=True)
    x = x.permute(0, 3, 1, 2).contiguous()
    x = x.to(dtype=torch.float32)
    if x.shape[-2] != tile_size_20x or x.shape[-1] != tile_size_20x:
        x = F.interpolate(x, size=(tile_size_20x, tile_size_20x), mode="bilinear", align_corners=False)

    if x.shape[-2] >= UNI_INPUT_SIZE and x.shape[-1] >= UNI_INPUT_SIZE:
        top = max(0, (x.shape[-2] - UNI_INPUT_SIZE) // 2)
        left = max(0, (x.shape[-1] - UNI_INPUT_SIZE) // 2)
        x = x[:, :, top:top + UNI_INPUT_SIZE, left:left + UNI_INPUT_SIZE]
    else:
        x = F.interpolate(x, size=(UNI_INPUT_SIZE, UNI_INPUT_SIZE), mode="bilinear", align_corners=False)

    x = x / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class OnTheFlyUNIEncoder:
    def __init__(self, device: str, tile_size_20x: int):
        self.device = device
        self.tile_size_20x = int(tile_size_20x)
        self.model, _ = get_uni(device=device)
        self.model = self.model.to(device).eval()

    @torch.inference_mode()
    def encode_np_batch(self, batch_np: np.ndarray) -> torch.Tensor:
        x = preprocess_batch_for_uni(
            batch_np=batch_np,
            tile_size_20x=self.tile_size_20x,
            device=self.device,
        )
        return self.model(x).detach().float().cpu()


def iter_locator_batches_on_the_fly(
    src: HMSSource,
    batch_size: int,
    wsi_reader: "OnTheFlyWSIReader",
    uni_encoder: OnTheFlyUNIEncoder,
) -> Iterable[Tuple[torch.Tensor, np.ndarray]]:
    if src.locator_coords is None or src.wsi_path is None or src.locator_level0_tile_size is None:
        return
    coords = np.asarray(src.locator_coords, dtype=np.int64)
    if coords.size == 0:
        return

    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        imgs = []
        kept = []
        for c in chunk:
            x0, y0 = int(c[0]), int(c[1])
            try:
                arr = wsi_reader.read_patch(src.wsi_path, x=x0, y=y0, level0_size=int(src.locator_level0_tile_size))
            except Exception:
                continue
            if arr.ndim != 3 or arr.shape[-1] != 3:
                continue
            imgs.append(arr)
            kept.append((x0, y0))
        if not imgs:
            continue
        batch_np = np.stack(imgs, axis=0)
        feats = uni_encoder.encode_np_batch(batch_np)
        ref_coords = np.asarray(kept, dtype=np.int64)
        yield feats, ref_coords


def iter_source_batches(
    src: HMSSource,
    batch_size: int,
    *,
    on_the_fly_uni: Optional[OnTheFlyUNIEncoder] = None,
    on_the_fly_reader: Optional["OnTheFlyWSIReader"] = None,
) -> Iterable[Tuple[torch.Tensor, Sequence[object]]]:
    if src.tile_dir is not None:
        if src.h5_path is None:
            return
        yield from iter_tile_batches(src.tile_dir, src.h5_path, batch_size=batch_size)
        return
    if src.locator_coords is not None:
        if src.h5_path is not None:
            yield from iter_locator_batches(src.locator_coords, src.h5_path, batch_size=batch_size)
            return
        if on_the_fly_uni is not None and on_the_fly_reader is not None:
            yield from iter_locator_batches_on_the_fly(
                src=src,
                batch_size=batch_size,
                wsi_reader=on_the_fly_reader,
                uni_encoder=on_the_fly_uni,
            )
        return


def first_pass_activation_stats(
    eval_model: torch.nn.Module,
    sources: Sequence[HMSSource],
    d_latent: int,
    device: str,
    tile_batch: int,
    max_tiles: int,
    on_the_fly_uni: Optional[OnTheFlyUNIEncoder] = None,
    on_the_fly_reader: Optional["OnTheFlyWSIReader"] = None,
    progress_every_batches: int = 20,
) -> Dict[str, object]:
    act_sum = torch.zeros(d_latent, dtype=torch.float32)
    act_min = torch.full((d_latent,), float("inf"), dtype=torch.float32)
    act_max = torch.full((d_latent,), float("-inf"), dtype=torch.float32)

    n_tiles = 0
    n_slides_used = 0
    total_slides = len(sources)
    global_batches = 0

    with torch.no_grad():
        for slide_i, src in enumerate(sources, start=1):
            slide_used = False
            slide_tiles = 0
            for feats_batch, _ in iter_source_batches(
                src,
                batch_size=tile_batch,
                on_the_fly_uni=on_the_fly_uni,
                on_the_fly_reader=on_the_fly_reader,
            ):
                if max_tiles > 0 and n_tiles >= max_tiles:
                    break
                if max_tiles > 0:
                    remain = int(max_tiles - n_tiles)
                    if remain <= 0:
                        break
                    if int(feats_batch.shape[0]) > remain:
                        feats_batch = feats_batch[:remain]
                global_batches += 1
                x = feats_batch.to(device, non_blocking=True)
                _, z, _ = eval_model(x)
                zc = z.detach().float().cpu()
                if zc.numel() == 0:
                    continue
                slide_used = True
                act_sum += zc.sum(dim=0)
                act_min = torch.minimum(act_min, zc.min(dim=0).values)
                act_max = torch.maximum(act_max, zc.max(dim=0).values)
                n_tiles += int(zc.shape[0])
                slide_tiles += int(zc.shape[0])
                if progress_every_batches > 0 and (global_batches % progress_every_batches == 0):
                    print(
                        f"[data:first:progress] slide={slide_i}/{total_slides} "
                        f"slide_name={src.slide_name} batches={global_batches} tiles_total={n_tiles}",
                        flush=True,
                    )
                if max_tiles > 0 and n_tiles >= max_tiles:
                    break
            if slide_used:
                n_slides_used += 1
            print(
                f"[data:first:slide] {slide_i}/{total_slides} {src.slide_name} "
                f"slide_tiles={slide_tiles} tiles_total={n_tiles}",
                flush=True,
            )
            if max_tiles > 0 and n_tiles >= max_tiles:
                break

    if n_tiles == 0:
        raise RuntimeError("No matched tiles found for data-driven HMS.")

    act_mean = act_sum / float(max(1, n_tiles))
    return {
        "n_tiles": int(n_tiles),
        "n_slides_used": int(n_slides_used),
        "act_mean": act_mean,
        "act_min": act_min,
        "act_max": act_max,
    }


def load_gigapath_encoder(model_id: str, device: str):
    try:
        import timm
        from torchvision import transforms
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Data HMS requires timm, torchvision and pillow. Install them in your environment."
        ) from exc

    model = timm.create_model(model_id, pretrained=True).to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, transform, Image


class OnTheFlyWSIReader:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self._openslide_key: Optional[str] = None
        self._openslide_obj: Optional[object] = None
        self._cucim_key: Optional[str] = None
        self._cucim_obj: Optional[object] = None
        self._has_cucim = False
        self._cucim_mod = None
        if mode in {"auto", "cucim"}:
            try:
                from cucim import CuImage  # type: ignore

                self._cucim_mod = CuImage
                self._has_cucim = True
            except Exception:
                self._has_cucim = False
                self._cucim_mod = None

    def close(self) -> None:
        if self._openslide_obj is not None:
            try:
                self._openslide_obj.close()
            except Exception:
                pass
        self._openslide_key = None
        self._openslide_obj = None
        self._cucim_key = None
        self._cucim_obj = None

    def _read_cucim(self, wsi_path: Path, x: int, y: int, size: int) -> np.ndarray:
        if not self._has_cucim or self._cucim_mod is None:
            raise RuntimeError("cucim unavailable")
        key = str(wsi_path)
        cu = self._cucim_obj
        if cu is None or self._cucim_key != key:
            cu = self._cucim_mod(key)
            self._cucim_obj = cu
            self._cucim_key = key
        arr = cu.read_region(location=(int(x), int(y)), size=(int(size), int(size)), level=0)
        if hasattr(arr, "get"):
            arr = arr.get()
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _read_openslide(self, wsi_path: Path, x: int, y: int, size: int) -> np.ndarray:
        import openslide

        key = str(wsi_path)
        slide = self._openslide_obj
        if slide is None or self._openslide_key != key:
            if self._openslide_obj is not None:
                try:
                    self._openslide_obj.close()
                except Exception:
                    pass
            slide = openslide.OpenSlide(key)
            self._openslide_obj = slide
            self._openslide_key = key
        img = slide.read_region((int(x), int(y)), 0, (int(size), int(size))).convert("RGB")
        return np.asarray(img)

    def read_patch(self, wsi_path: Path, x: int, y: int, level0_size: int) -> np.ndarray:
        if self.mode == "cucim":
            return self._read_cucim(wsi_path, x, y, level0_size)
        if self.mode == "openslide":
            return self._read_openslide(wsi_path, x, y, level0_size)
        # auto
        if self._has_cucim:
            try:
                return self._read_cucim(wsi_path, x, y, level0_size)
            except Exception:
                pass
        return self._read_openslide(wsi_path, x, y, level0_size)


def second_pass_data_hms(
    eval_model: torch.nn.Module,
    sources: Sequence[HMSSource],
    parents: torch.Tensor,
    act_mean: torch.Tensor,
    act_min: torch.Tensor,
    act_max: torch.Tensor,
    alive_eps: float,
    min_children_per_parent: int,
    device: str,
    tile_batch: int,
    gigapath_batch: int,
    max_tiles: int,
    gigapath_model: str,
    wsi_reader_mode: str,
    on_the_fly_uni: Optional[OnTheFlyUNIEncoder] = None,
    on_the_fly_reader: Optional["OnTheFlyWSIReader"] = None,
    progress_every_batches: int = 20,
) -> Dict[str, object]:
    gp_model, gp_transform, PILImage = load_gigapath_encoder(gigapath_model, device=device)
    wsi_reader = OnTheFlyWSIReader(mode=wsi_reader_mode)

    alive_mask = act_mean > float(alive_eps)
    alive_idx = torch.where(alive_mask)[0].long()
    n_alive = int(alive_idx.numel())
    if n_alive == 0:
        raise RuntimeError(f"No alive neurons found with alive_eps={alive_eps}")

    mins = act_min[alive_idx]
    maxs = act_max[alive_idx]
    rng = torch.clamp(maxs - mins, min=1e-8)

    proto_sum = torch.zeros((n_alive, 1536), dtype=torch.float32)
    weight_sum = torch.zeros((n_alive,), dtype=torch.float32)

    n_tiles = 0
    n_slides_used = 0
    total_slides = len(sources)
    global_batches = 0

    with torch.no_grad():
        for slide_i, src in enumerate(sources, start=1):
            slide_used = False
            slide_tiles = 0
            for feats_batch, ref_batch in iter_source_batches(
                src,
                batch_size=tile_batch,
                on_the_fly_uni=on_the_fly_uni,
                on_the_fly_reader=on_the_fly_reader,
            ):
                if max_tiles > 0 and n_tiles >= max_tiles:
                    break
                if max_tiles > 0:
                    remain = int(max_tiles - n_tiles)
                    if remain <= 0:
                        break
                    if int(feats_batch.shape[0]) > remain:
                        feats_batch = feats_batch[:remain]
                        ref_batch = ref_batch[:remain]
                global_batches += 1

                # SAE activations
                x = feats_batch.to(device, non_blocking=True)
                _, z, _ = eval_model(x)
                zc = z.detach().float().cpu()[:, alive_idx]
                weights = torch.clamp((zc - mins) / rng, min=0.0, max=1.0)  # [B, n_alive]

                # GigaPath embeddings
                emb_chunks: List[torch.Tensor] = []
                for start in range(0, len(ref_batch), max(1, int(gigapath_batch))):
                    refs = ref_batch[start:start + max(1, int(gigapath_batch))]
                    imgs = []
                    if src.tile_dir is not None:
                        for p in refs:
                            try:
                                img = PILImage.open(p).convert("RGB")
                                imgs.append(gp_transform(img))
                            except Exception:
                                imgs.append(torch.zeros(3, 224, 224))
                    else:
                        if src.wsi_path is None or src.locator_level0_tile_size is None:
                            raise RuntimeError(f"Missing wsi_path/level0_tile_size for locator source {src.slide_name}")
                        for c in refs:
                            x0, y0 = int(c[0]), int(c[1])
                            try:
                                arr = wsi_reader.read_patch(src.wsi_path, x=x0, y=y0, level0_size=int(src.locator_level0_tile_size))
                                img = PILImage.fromarray(arr)
                                imgs.append(gp_transform(img))
                            except Exception:
                                imgs.append(torch.zeros(3, 224, 224))
                    img_batch = torch.stack(imgs, dim=0).to(device, non_blocking=True)
                    emb_chunks.append(gp_model(img_batch).detach().float().cpu())
                emb = torch.cat(emb_chunks, dim=0) if emb_chunks else torch.zeros((0, 1536), dtype=torch.float32)

                proto_sum += weights.t() @ emb
                weight_sum += weights.sum(dim=0)

                slide_used = True
                n_tiles += int(weights.shape[0])
                slide_tiles += int(weights.shape[0])
                if progress_every_batches > 0 and (global_batches % progress_every_batches == 0):
                    print(
                        f"[data:second:progress] slide={slide_i}/{total_slides} "
                        f"slide_name={src.slide_name} batches={global_batches} tiles_total={n_tiles}",
                        flush=True,
                    )
                if max_tiles > 0 and n_tiles >= max_tiles:
                    break

            if slide_used:
                n_slides_used += 1
            print(
                f"[data:second:slide] {slide_i}/{total_slides} {src.slide_name} "
                f"slide_tiles={slide_tiles} tiles_total={n_tiles}",
                flush=True,
            )
            if max_tiles > 0 and n_tiles >= max_tiles:
                break

    protos = proto_sum / (weight_sum.unsqueeze(1) + 1e-8)
    protos = F.normalize(protos, p=2, dim=1)

    parent_to_children: Dict[int, List[int]] = defaultdict(list)
    parents_cpu = parents.long().cpu()
    alive_global = alive_idx.cpu().numpy()
    for local_i, global_j in enumerate(alive_global):
        parent_to_children[int(parents_cpu[int(global_j)].item())].append(local_i)

    per_parent = []
    scores = []
    for p, children in sorted(parent_to_children.items()):
        if len(children) < int(min_children_per_parent):
            continue
        score = mean_pairwise_cos(protos[children])
        if score is None:
            continue
        per_parent.append(
            {
                "parent_id": int(p),
                "cluster_size": int(len(children)),
                "hms_score": float(score),
                "child_ids": [int(alive_global[c]) for c in children],
            }
        )
        scores.append(float(score))

    return {
        "summary": summarize_scores(scores),
        "min_children_per_parent": int(min_children_per_parent),
        "alive_eps": float(alive_eps),
        "alive_count": int(n_alive),
        "total_latent": int(parents.shape[0]),
        "tiles_processed": int(n_tiles),
        "slides_processed": int(n_slides_used),
        "gigapath_model": gigapath_model,
        "wsi_reader_mode": wsi_reader_mode,
        "per_parent": per_parent,
    }


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    ckpt_path = (args.ckpt_path.resolve() if args.ckpt_path else (run_dir / "sdf2_final.pt").resolve())
    out_json = (args.out_json.resolve() if args.out_json else (run_dir / "hms_report.json").resolve())
    device = resolve_device(args.device)

    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    model, eval_model, ckpt_meta = load_sdf2_model(ckpt_path=ckpt_path, run_dir=run_dir, device=device)

    print(f"[model] ckpt={ckpt_path}")
    print(f"[model] d_in={ckpt_meta['d_in']} d_latent={ckpt_meta['d_latent']} d_level2={ckpt_meta['d_level2']}")
    print(f"[model] device={device} wrapper_ln={ckpt_meta['wrapper_layernorm']}")

    report: Dict[str, object] = {
        "config": {
            "run_dir": str(run_dir),
            "ckpt_path": str(ckpt_path),
            "device": device,
            "min_children_per_parent": int(args.min_children_per_parent),
            "alive_eps": float(args.alive_eps),
            "near_zero_eps": float(args.near_zero_eps),
            "run_data_hms": bool(args.run_data_hms),
            "tile_root": str(args.tile_root) if args.tile_root else None,
            "h5_root": str(args.h5_root) if args.h5_root else None,
            "encode_uni_on_the_fly": bool(args.encode_uni_on_the_fly),
            "locator_csv_dir": str(args.locator_csv_dir) if args.locator_csv_dir else None,
            "wsi_dir": str(args.wsi_dir) if args.wsi_dir else None,
            "gigapath_model": args.gigapath_model,
            "gigapath_batch": int(args.gigapath_batch),
            "tile_batch": int(args.tile_batch),
            "wsi_reader": args.wsi_reader,
            "wsi_patch_size_20x": int(args.wsi_patch_size_20x),
            "max_slides": int(args.max_slides),
            "max_tiles": int(args.max_tiles),
            "progress_every_batches": int(args.progress_every_batches),
        },
        "checkpoint": ckpt_meta,
    }

    parents = compute_parent_assignment(model=model)

    if args.run_data_hms:
        use_on_the_fly_uni = bool(args.encode_uni_on_the_fly)
        h5_root: Optional[Path] = None
        if args.h5_root is not None:
            h5_root = args.h5_root.resolve()
            if not h5_root.exists():
                raise SystemExit(f"Missing h5_root: {h5_root}")
        if (not use_on_the_fly_uni) and h5_root is None:
            raise SystemExit("--run_data_hms requires --h5_root unless --encode_uni_on_the_fly is set")

        if args.locator_csv_dir is not None:
            locator_csv_dir = args.locator_csv_dir.resolve()
            if not locator_csv_dir.exists():
                raise SystemExit(f"Missing locator_csv_dir: {locator_csv_dir}")
            wsi_dir = args.wsi_dir.resolve() if args.wsi_dir else None
            if wsi_dir is not None and not wsi_dir.exists():
                raise SystemExit(f"Missing wsi_dir: {wsi_dir}")
            sources = discover_locator_sources(
                locator_csv_dir=locator_csv_dir,
                h5_root=h5_root,
                wsi_dir=wsi_dir,
                max_slides=int(args.max_slides),
                wsi_patch_size_20x=int(args.wsi_patch_size_20x),
                require_h5=not use_on_the_fly_uni,
            )
            if not sources:
                raise SystemExit("No locator-based slide sources found (check locator CSVs/H5/WSI).")
            print(f"[data] discovered locator sources={len(sources)}")
        else:
            if use_on_the_fly_uni:
                raise SystemExit("--encode_uni_on_the_fly currently supports only --locator_csv_dir mode")
            if args.tile_root is None:
                raise SystemExit("--run_data_hms requires --tile_root unless --locator_csv_dir is used")
            tile_root = args.tile_root.resolve()
            if not tile_root.exists():
                raise SystemExit(f"Missing tile_root: {tile_root}")
            assert h5_root is not None
            sources = discover_tile_sources(tile_root=tile_root, h5_root=h5_root, max_slides=int(args.max_slides))
            if not sources:
                raise SystemExit("No tile-folder slide sources found between tile_root and h5_root.")
            print(f"[data] discovered tile sources={len(sources)}")

        on_the_fly_uni: Optional[OnTheFlyUNIEncoder] = None
        on_the_fly_reader: Optional[OnTheFlyWSIReader] = None
        if use_on_the_fly_uni:
            print("[data] loading UNI2-h for on-the-fly feature encoding...")
            on_the_fly_uni = OnTheFlyUNIEncoder(
                device=device,
                tile_size_20x=int(args.wsi_patch_size_20x),
            )
            on_the_fly_reader = OnTheFlyWSIReader(mode=args.wsi_reader)
            print(f"[data] on-the-fly UNI enabled, reader={args.wsi_reader}")

        first = first_pass_activation_stats(
            eval_model=eval_model,
            sources=sources,
            d_latent=int(ckpt_meta["d_latent"]),
            device=device,
            tile_batch=int(args.tile_batch),
            max_tiles=int(args.max_tiles),
            on_the_fly_uni=on_the_fly_uni,
            on_the_fly_reader=on_the_fly_reader,
            progress_every_batches=int(args.progress_every_batches),
        )
        print(f"[data] first pass tiles={first['n_tiles']} slides={first['n_slides_used']}")

        data_hms = second_pass_data_hms(
            eval_model=eval_model,
            sources=sources,
            parents=parents,
            act_mean=first["act_mean"],
            act_min=first["act_min"],
            act_max=first["act_max"],
            alive_eps=float(args.alive_eps),
            min_children_per_parent=int(args.min_children_per_parent),
            device=device,
            tile_batch=int(args.tile_batch),
            gigapath_batch=int(args.gigapath_batch),
            max_tiles=int(args.max_tiles),
            gigapath_model=args.gigapath_model,
            wsi_reader_mode=args.wsi_reader,
            on_the_fly_uni=on_the_fly_uni,
            on_the_fly_reader=on_the_fly_reader,
            progress_every_batches=int(args.progress_every_batches),
        )
        if on_the_fly_reader is not None:
            on_the_fly_reader.close()
        report["data_hms"] = data_hms
        print(f"[data] HMS mean={data_hms['summary']['mean']:.4f} n={data_hms['summary']['n']}")
    else:
        report["data_hms"] = None

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[ok] wrote {out_json}")


if __name__ == "__main__":
    main()
