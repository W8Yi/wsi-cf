#!/usr/bin/env python3
"""
Cache UNI2-h features for locator CSVs (WSI -> UNI -> H5).

Inputs:
  --wsi_dir          directory containing WSIs
  --locator_csv_dir  directory of locator CSVs (from sample_wsi_patches_20x_coverage.py)
  --out_dir          output directory for H5 files

Each locator CSV produces <out_dir>/<slide_name>.h5 containing:
  - features [1, N, 1536]
  - coords [1, N, 2]
  - coords_patching [N, 2]
  - annots [1, N, 1]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.uni import get_uni  # noqa: E402

try:
    from cucim import CuImage  # type: ignore
    HAS_CUCIM = True
except Exception:
    CuImage = None
    HAS_CUCIM = False

try:
    import openslide  # type: ignore
except Exception as exc:
    raise RuntimeError("openslide-python and system OpenSlide libraries are required.") from exc

UNI_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WSI_SUFFIXES = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wsi_dir", type=Path, required=True)
    ap.add_argument("--locator_csv_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--reader", type=str, default="auto", choices=["auto", "cucim", "openslide"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_slides", type=int, default=0)
    ap.add_argument("--tile_size_20x", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_wsi_path(wsi_dir: Path, slide_name: str, csv_wsi: str) -> Optional[Path]:
    if csv_wsi:
        p = Path(csv_wsi)
        if p.exists():
            return p
    for ext in WSI_SUFFIXES:
        p = wsi_dir / f"{slide_name}{ext}"
        if p.exists():
            return p
    for ext in WSI_SUFFIXES:
        cands = sorted(wsi_dir.glob(f"{slide_name}*{ext}"))
        if cands:
            return cands[0]
    return None


def infer_level0_scale_20x(wsi_path: Path) -> float:
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


def preprocess_batch_for_uni(batch_np: np.ndarray, tile_size_20x: int, device: str) -> torch.Tensor:
    x = torch.from_numpy(batch_np).to(device=device, dtype=torch.uint8, non_blocking=True)
    x = x.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)
    if x.shape[-2] != tile_size_20x or x.shape[-1] != tile_size_20x:
        x = F.interpolate(x, size=(tile_size_20x, tile_size_20x), mode="bilinear", align_corners=False)

    if x.shape[-2] >= UNI_INPUT_SIZE and x.shape[-1] >= UNI_INPUT_SIZE:
        top = max(0, (x.shape[-2] - UNI_INPUT_SIZE) // 2)
        left = max(0, (x.shape[-1] - UNI_INPUT_SIZE) // 2)
        x = x[:, :, top : top + UNI_INPUT_SIZE, left : left + UNI_INPUT_SIZE]
    else:
        x = F.interpolate(x, size=(UNI_INPUT_SIZE, UNI_INPUT_SIZE), mode="bilinear", align_corners=False)

    x = x / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class WSIReader:
    def __init__(self, mode: str):
        self.mode = mode
        self._openslide_key = None
        self._openslide_obj = None
        self._cucim_key = None
        self._cucim_obj = None
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

    def read_patch(self, wsi_path: Path, x: int, y: int, size: int) -> np.ndarray:
        if self.mode == "cucim":
            return self._read_cucim(wsi_path, x, y, size)
        if self.mode == "openslide":
            return self._read_openslide(wsi_path, x, y, size)
        if self._has_cucim:
            try:
                return self._read_cucim(wsi_path, x, y, size)
            except Exception:
                pass
        return self._read_openslide(wsi_path, x, y, size)


def read_locator_csv(path: Path) -> Tuple[List[Tuple[int, int]], List[int], str]:
    coords: List[Tuple[int, int]] = []
    level0_sizes: List[int] = []
    wsi_path = ""
    with path.open("r", newline="") as f:
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
            if not wsi_path:
                wsi_path = (row.get("wsi_path") or "").strip()
    return coords, level0_sizes, wsi_path


def write_h5(path: Path, features: np.ndarray, coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    annots = np.zeros((1, coords.shape[0], 1), dtype=np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features[None, :, :])
        f.create_dataset("coords", data=coords[None, :, :])
        f.create_dataset("coords_patching", data=coords)
        f.create_dataset("annots", data=annots)


def main() -> None:
    args = parse_args()
    if not args.wsi_dir.exists():
        raise SystemExit(f"Missing wsi_dir: {args.wsi_dir}")
    if not args.locator_csv_dir.exists():
        raise SystemExit(f"Missing locator_csv_dir: {args.locator_csv_dir}")

    device = resolve_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    uni_model, _ = get_uni(device=device)
    uni_model = uni_model.to(device).eval()
    reader = WSIReader(mode=args.reader)

    csvs = sorted(args.locator_csv_dir.glob("*.csv"))
    if args.max_slides > 0:
        csvs = csvs[: int(args.max_slides)]
    if not csvs:
        raise SystemExit("No locator CSVs found.")

    print(f"[start] csvs={len(csvs)} device={device} reader={args.reader} out_dir={out_dir}")

    for idx, csv_path in enumerate(csvs, start=1):
        slide_name = csv_path.stem
        out_h5 = out_dir / f"{slide_name}.h5"
        if out_h5.exists() and not args.overwrite:
            print(f"[skip] {slide_name} exists")
            continue

        coords, level0_sizes, csv_wsi = read_locator_csv(csv_path)
        if not coords:
            print(f"[warn] {slide_name} has no coords")
            continue

        wsi_path = resolve_wsi_path(args.wsi_dir, slide_name, csv_wsi)
        if wsi_path is None:
            print(f"[warn] missing WSI for {slide_name}")
            continue

        if level0_sizes:
            level0_tile_size = int(round(float(np.median(np.asarray(level0_sizes, dtype=np.float32)))))
        else:
            scale = infer_level0_scale_20x(wsi_path)
            level0_tile_size = max(1, int(round(float(args.tile_size_20x) * float(scale))))

        feats = []
        coords_np = np.asarray(coords, dtype=np.int64)
        for start in range(0, len(coords), int(args.batch_size)):
            chunk = coords[start : start + int(args.batch_size)]
            imgs = []
            kept = []
            for x, y in chunk:
                try:
                    arr = reader.read_patch(wsi_path, x=x, y=y, size=int(level0_tile_size))
                except Exception:
                    continue
                if arr.ndim != 3 or arr.shape[-1] != 3:
                    continue
                imgs.append(arr)
                kept.append((x, y))
            if not imgs:
                continue
            batch_np = np.stack(imgs, axis=0)
            x = preprocess_batch_for_uni(batch_np, tile_size_20x=int(args.tile_size_20x), device=device)
            with torch.inference_mode():
                emb = uni_model(x).detach().float().cpu().numpy()
            feats.append(emb)
            if len(kept) != len(imgs):
                coords_np = np.asarray(kept, dtype=np.int64)

        if not feats:
            print(f"[warn] no embeddings for {slide_name}")
            continue

        feats_np = np.concatenate(feats, axis=0)
        if feats_np.shape[0] != coords_np.shape[0]:
            coords_np = coords_np[: feats_np.shape[0]]
        write_h5(out_h5, feats_np, coords_np)
        print(f"[ok] {idx}/{len(csvs)} {slide_name} tiles={coords_np.shape[0]} -> {out_h5}")

    reader.close()
    print("[done]")


if __name__ == "__main__":
    main()
