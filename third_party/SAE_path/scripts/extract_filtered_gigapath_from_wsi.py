#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import errno
import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    import openslide
except Exception as exc:
    raise RuntimeError("openslide-python and system OpenSlide are required.") from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCH_ROOT_DEFAULT = Path("/common/users/wq50/bench")
MANIFEST_DEFAULT = Path("/common/users/wq50/assets/ckpts/manifest_tile_encoders.json")
CKPT_ROOT_DEFAULT = Path("/common/users/wq50/assets/ckpts")
WSI_EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")


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


def level0_size_from_20x(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(tile_size_20x * (objective_power / 20.0))))


def read_coords_csv(path: Path) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    with path.open("r", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            rows.append((int(float(row["coord_x"])), int(float(row["coord_y"]))))
    if not rows:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


class ClamH5Writer:
    def __init__(
        self,
        path: Path,
        embedding_dim: int,
        *,
        compression: str = "gzip",
        chunk_rows: int = 1024,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(path, "w")
        self._n = 0
        d = int(embedding_dim)
        c = max(1, int(chunk_rows))

        self._features = self._f.create_dataset(
            "features",
            shape=(1, 0, d),
            maxshape=(1, None, d),
            chunks=(1, c, d),
            dtype="float32",
            compression=compression,
            shuffle=True,
        )
        self._coords = self._f.create_dataset(
            "coords",
            shape=(1, 0, 2),
            maxshape=(1, None, 2),
            chunks=(1, c, 2),
            dtype="int32",
            compression=compression,
            shuffle=True,
        )
        self._coords_patching = self._f.create_dataset(
            "coords_patching",
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(c, 2),
            dtype="int32",
            compression=compression,
            shuffle=True,
        )
        self._annots = self._f.create_dataset(
            "annots",
            shape=(1, 0, 1),
            maxshape=(1, None, 1),
            chunks=(1, c, 1),
            dtype="int64",
            compression=compression,
            shuffle=True,
        )

    def append(self, features: np.ndarray, coords: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be 2D [N,D], got {features.shape}")
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be [N,2], got {coords.shape}")
        if features.shape[0] != coords.shape[0]:
            raise ValueError("features/coords row mismatch")

        n_new = int(features.shape[0])
        end = self._n + n_new

        self._features.resize((1, end, self._features.shape[2]))
        self._coords.resize((1, end, 2))
        self._coords_patching.resize((end, 2))
        self._annots.resize((1, end, 1))

        self._features[0, self._n:end, :] = features.astype(np.float32, copy=False)
        coords_i = coords.astype(np.int32, copy=False)
        self._coords[0, self._n:end, :] = coords_i
        self._coords_patching[self._n:end, :] = coords_i
        self._annots[0, self._n:end, :] = 0

        self._n = end

    def close(self) -> None:
        self._f.close()

    @property
    def n_rows(self) -> int:
        return self._n


def validate_feature_h5(path: Path, *, expected_rows: int | None = None) -> tuple[bool, str]:
    try:
        with h5py.File(path, "r") as f:
            required = {"features", "coords", "coords_patching", "annots"}
            keys = set(f.keys())
            missing = sorted(required - keys)
            if missing:
                return False, f"missing datasets: {missing}"

            feat = f["features"]
            coords = f["coords"]
            coords_patch = f["coords_patching"]
            annots = f["annots"]

            if feat.ndim != 3:
                return False, f"features ndim={feat.ndim}, expected=3"
            if coords.ndim != 3:
                return False, f"coords ndim={coords.ndim}, expected=3"
            if coords_patch.ndim != 2:
                return False, f"coords_patching ndim={coords_patch.ndim}, expected=2"
            if annots.ndim != 3:
                return False, f"annots ndim={annots.ndim}, expected=3"

            if feat.shape[0] != 1 or coords.shape[0] != 1 or annots.shape[0] != 1:
                return False, f"leading dims invalid feat={feat.shape} coords={coords.shape} annots={annots.shape}"
            if coords.shape[2] != 2 or coords_patch.shape[1] != 2 or annots.shape[2] != 1:
                return False, f"tail dims invalid coords={coords.shape} coords_patching={coords_patch.shape} annots={annots.shape}"

            n_feat = int(feat.shape[1])
            n_coords = int(coords.shape[1])
            n_patch = int(coords_patch.shape[0])
            n_ann = int(annots.shape[1])
            if not (n_feat == n_coords == n_patch == n_ann):
                return False, f"row mismatch feat={n_feat} coords={n_coords} coords_patching={n_patch} annots={n_ann}"
            if n_feat <= 0:
                return False, "empty feature rows"
            if expected_rows is not None and n_feat != int(expected_rows):
                return False, f"row mismatch vs coords_csv expected={int(expected_rows)} got={n_feat}"

        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def is_stale_file_handle_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ESTALE:
        return True
    return "stale file handle" in str(exc).lower()


def is_cuda_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "cuda error: out of memory" in msg
        or "outofmemoryerror" in msg
    )


def release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


@dataclass
class SlideResult:
    slide_key: str
    status: str
    h5_path: str
    kept_tiles: int
    embedding_dim: int
    elapsed_sec: float
    error: str = ""


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def find_slide_files(wsi_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(wsi_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in WSI_EXTS:
            out.append(p)
    return out


def run_filter_only(args: argparse.Namespace) -> None:
    cmd = [
        args.python_bin,
        str(ROOT_DIR / "scripts" / "extract_filtered_uni_from_wsi.py"),
        "--wsi_dir",
        str(args.wsi_dir),
        "--out_dir",
        str(args.out_dir),
        "--summary_dir",
        str(args.summary_dir),
        "--coords_dir",
        str(args.coords_dir),
        "--viz_dir",
        str(args.viz_dir),
        "--reader",
        str(args.reader),
        "--filter_workers",
        str(int(args.filter_workers)),
        "--tile_size_20x",
        str(int(args.tile_size_20x)),
        "--num_shards",
        str(int(args.num_shards)),
        "--shard_index",
        str(int(args.shard_index)),
        "--filter_only",
    ]
    if args.overwrite:
        cmd.append("--overwrite")

    print(f"[filter] {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"filter-only stage failed (rc={rc})")


def _load_gigapath_adapter(args: argparse.Namespace):
    bench_root = Path(args.bench_root)
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    from wsi_protocol.adapters import load_encoder
    from wsi_protocol.registry import Registry

    reg = Registry.from_manifest(manifest_path=Path(args.manifest), ckpt_root=Path(args.ckpt_root))
    return load_encoder("gigapath", registry=reg, device=args.device, precision="fp32")


def _iter_batches(coords: np.ndarray, batch_size: int):
    n = int(coords.shape[0])
    for i in range(0, n, batch_size):
        yield coords[i : i + batch_size]


def encode_one_slide(slide_path: Path, args: argparse.Namespace, adapter) -> SlideResult:
    slide_key = slide_path.stem
    coords_csv = args.coords_dir / f"{slide_key}.coords.csv"
    h5_path = args.out_dir / f"{slide_key}.h5"
    summary_path = args.summary_dir / f"{slide_key}.summary.json"

    t0 = time.time()

    if not coords_csv.exists():
        result = SlideResult(
            slide_key=slide_key,
            status="missing_coords_csv",
            h5_path=str(h5_path),
            kept_tiles=0,
            embedding_dim=0,
            elapsed_sec=0.0,
            error=f"missing coords: {coords_csv}",
        )
        write_json(summary_path, {"slide_key": slide_key, "encode": result.__dict__})
        return result

    coords = read_coords_csv(coords_csv)
    if coords.shape[0] == 0:
        result = SlideResult(
            slide_key=slide_key,
            status="no_tiles_kept",
            h5_path=str(h5_path),
            kept_tiles=0,
            embedding_dim=0,
            elapsed_sec=0.0,
        )
        write_json(summary_path, {"slide_key": slide_key, "encode": result.__dict__})
        return result

    expected_rows = int(coords.shape[0])

    if h5_path.exists() and not args.overwrite:
        ok, reason = validate_feature_h5(h5_path, expected_rows=expected_rows)
        if ok:
            result = SlideResult(
                slide_key=slide_key,
                status="reused_encode",
                h5_path=str(h5_path),
                kept_tiles=expected_rows,
                embedding_dim=0,
                elapsed_sec=0.0,
            )
            write_json(summary_path, {"slide_key": slide_key, "encode": result.__dict__})
            return result
        try:
            h5_path.unlink()
            print(f"[encode] removed invalid existing h5 {h5_path.name}: {reason}", flush=True)
        except Exception as exc:
            raise RuntimeError(f"invalid existing h5 and failed to delete ({h5_path}): {reason}; {exc}") from exc

    retries = max(0, int(getattr(args, "write_retries", 2)))
    min_batch_size = max(1, int(getattr(args, "min_batch_size", 8)))
    current_batch_size = max(min_batch_size, int(args.batch_size))
    stale_attempts = 0
    attempt = 0

    while True:
        attempt += 1
        tmp_h5_path = h5_path.parent / f".{h5_path.name}.tmp.{os.getpid()}.{attempt}"
        if tmp_h5_path.exists():
            tmp_h5_path.unlink(missing_ok=True)

        slide: openslide.OpenSlide | None = None
        writer: ClamH5Writer | None = None
        embedding_dim = 0
        try:
            slide = openslide.OpenSlide(str(slide_path))
            objective_power = infer_objective_power(slide)
            tile_size_level0 = level0_size_from_20x(args.tile_size_20x, objective_power)

            total = int(coords.shape[0])
            done = 0
            num_batches = max(1, math.ceil(total / max(1, current_batch_size)))

            for bidx, chunk in enumerate(_iter_batches(coords, current_batch_size), start=1):
                images = [
                    slide.read_region((int(x), int(y)), 0, (tile_size_level0, tile_size_level0)).convert("RGB")
                    for x, y in chunk
                ]
                batch = adapter.preprocess_images(images)
                feats = adapter.encode_batch(batch)

                if writer is None:
                    embedding_dim = int(feats.shape[1])
                    writer = ClamH5Writer(
                        tmp_h5_path,
                        embedding_dim=embedding_dim,
                        compression=args.compression,
                        chunk_rows=max(256, current_batch_size),
                    )

                writer.append(feats, chunk)
                done += int(chunk.shape[0])
                print(
                    f"[encode] {slide_key} batch {bidx}/{num_batches} tiles {done}/{total} "
                    f"batch_size={current_batch_size}",
                    flush=True,
                )

            if writer is None:
                raise RuntimeError("writer was not initialized")

            writer.close()
            writer = None

            ok, reason = validate_feature_h5(tmp_h5_path, expected_rows=expected_rows)
            if not ok:
                raise RuntimeError(f"tmp h5 validation failed: {tmp_h5_path} ({reason})")

            os.replace(tmp_h5_path, h5_path)

            elapsed = float(time.time() - t0)
            result = SlideResult(
                slide_key=slide_key,
                status="ok",
                h5_path=str(h5_path),
                kept_tiles=expected_rows,
                embedding_dim=embedding_dim,
                elapsed_sec=elapsed,
            )
            write_json(summary_path, {"slide_key": slide_key, "encode": result.__dict__})
            return result
        except Exception as exc:
            if is_stale_file_handle_error(exc) and stale_attempts < retries:
                stale_attempts += 1
                print(
                    f"[warn] stale file handle while writing {slide_key}; retry {stale_attempts}/{retries}",
                    flush=True,
                )
                release_torch_memory()
                time.sleep(min(5.0, 0.5 * attempt))
                continue

            if is_cuda_oom_error(exc) and current_batch_size > min_batch_size:
                next_batch_size = max(min_batch_size, current_batch_size // 2)
                if next_batch_size < current_batch_size:
                    print(
                        f"[warn] CUDA OOM for {slide_key}; retry with smaller batch size "
                        f"{current_batch_size} -> {next_batch_size}",
                        flush=True,
                    )
                    current_batch_size = next_batch_size
                    release_torch_memory()
                    time.sleep(min(5.0, 0.5 * attempt))
                    continue

            elapsed = float(time.time() - t0)
            result = SlideResult(
                slide_key=slide_key,
                status="failed",
                h5_path=str(h5_path),
                kept_tiles=0,
                embedding_dim=0,
                elapsed_sec=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
            write_json(summary_path, {"slide_key": slide_key, "encode": result.__dict__})
            return result
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as close_exc:
                    print(f"[warn] close failed for {tmp_h5_path.name}: {close_exc}", flush=True)
            tmp_h5_path.unlink(missing_ok=True)
            if slide is not None:
                slide.close()
            release_torch_memory()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Filter WSIs then encode GigaPath features into CLAM-style .h5")
    ap.add_argument("--wsi_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--summary_dir", type=Path, required=True)
    ap.add_argument("--coords_dir", type=Path, required=True)
    ap.add_argument("--viz_dir", type=Path, required=True)

    ap.add_argument("--bench_root", type=Path, default=BENCH_ROOT_DEFAULT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    ap.add_argument("--ckpt_root", type=Path, default=CKPT_ROOT_DEFAULT)
    ap.add_argument("--python_bin", default=sys.executable)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--gpu_list", type=str, default="")
    ap.add_argument("--dynamic_gpu", action="store_true")
    ap.add_argument("--reader", type=str, default="auto", choices=["auto", "openslide", "cucim"])

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--filter_workers", type=int, default=4)
    ap.add_argument("--loader_workers", type=int, default=0)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--tile_size_20x", type=int, default=256)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--max_slides", type=int, default=0)
    ap.add_argument("--slide_key", type=str, default="")

    ap.add_argument("--compression", type=str, default="gzip")
    ap.add_argument("--write_retries", type=int, default=int(os.environ.get("WRITE_RETRIES", "2")))
    ap.add_argument("--min_batch_size", type=int, default=int(os.environ.get("MIN_BATCH_SIZE", "8")))
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.wsi_dir.exists():
        raise SystemExit(f"Missing wsi_dir: {args.wsi_dir}")
    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard_index must satisfy 0 <= shard_index < num_shards")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.coords_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)

    args.device = resolve_device(args.device)
    print(f"[setup] device={args.device}", flush=True)
    print(f"[setup] ckpt_root={args.ckpt_root}", flush=True)

    run_filter_only(args)

    slides = find_slide_files(args.wsi_dir)
    if args.slide_key:
        slides = [p for p in slides if p.stem == args.slide_key]
    if args.max_slides > 0:
        slides = slides[: args.max_slides]
    if args.num_shards > 1:
        slides = [p for i, p in enumerate(slides) if (i % args.num_shards) == args.shard_index]

    if not slides:
        raise SystemExit("No matching slide files found after filters.")

    adapter = _load_gigapath_adapter(args)

    results: list[SlideResult] = []
    for idx, slide_path in enumerate(slides, start=1):
        print(f"[slide {idx}/{len(slides)}] {slide_path.stem}", flush=True)
        res = encode_one_slide(slide_path, args, adapter)
        results.append(res)
        print(
            f"[slide {idx}/{len(slides)}] {res.slide_key}: status={res.status} "
            f"tiles={res.kept_tiles} elapsed={res.elapsed_sec:.1f}s",
            flush=True,
        )

    summary_payload = {
        "encoder": "gigapath",
        "device": args.device,
        "wsi_dir": str(args.wsi_dir),
        "out_dir": str(args.out_dir),
        "coords_dir": str(args.coords_dir),
        "results": [r.__dict__ for r in results],
    }
    write_json(args.summary_dir / "run_summary.json", summary_payload)

    fails = sum(1 for r in results if r.status == "failed")
    print(f"[done] slides={len(results)} failed={fails}", flush=True)
    if fails > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
