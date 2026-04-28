#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

from utils.sae import load_sae_from_config, sae_encode_features


def project_balanced_h5_list_from_index(
    index_json: Path,
    *,
    slides_per_project: Optional[int],
    seed: int,
    require_h5_exists: bool = True,
) -> List[str]:
    idx = json.loads(index_json.read_text())

    by_proj = {}
    for _slide, e in idx.items():
        proj = e.get("dataset")
        h5 = e.get("h5_path")
        if not proj or not h5:
            continue
        p = Path(h5)
        if require_h5_exists and not p.exists():
            continue
        by_proj.setdefault(proj, []).append(p)

    rng = np.random.default_rng(seed)
    out: List[Path] = []
    for proj in sorted(by_proj.keys()):
        files = by_proj[proj]
        rng.shuffle(files)
        keep = files if slides_per_project is None else files[: min(slides_per_project, len(files))]
        out.extend(keep)

    rng.shuffle(out)
    return [str(p) for p in out]


def read_h5_subset(
    h5_path: Path,
    *,
    tiles_per_slide: int,
    rng: np.random.Generator,
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        dsX = f["features"]

        if dsX.ndim == 3 and dsX.shape[0] == 1:
            n = dsX.shape[1]
        elif dsX.ndim == 2:
            n = dsX.shape[0]
        else:
            raise ValueError(f"Unsupported features shape: {dsX.shape} in {h5_path}")

        if tiles_per_slide > 0 and n > tiles_per_slide:
            rows = rng.choice(n, size=tiles_per_slide, replace=False)
            rows = np.asarray(rows, dtype=np.int64)
            rows.sort()
            rows_list = rows.tolist()
        else:
            rows_list = None

        if rows_list is None:
            X = dsX[0, :, :] if dsX.ndim == 3 else dsX[:, :]
        else:
            X = dsX[0, rows_list, :] if dsX.ndim == 3 else dsX[rows_list, :]

        return np.asarray(X, dtype=np.float32)


def reservoir_update(
    reservoir: Optional[np.ndarray],
    z_np: np.ndarray,
    max_tiles: int,
    rng: np.random.Generator,
    total_seen: int,
) -> tuple[np.ndarray, int]:
    if max_tiles < 0:
        if reservoir is None:
            reservoir = z_np.copy()
        else:
            reservoir = np.concatenate([reservoir, z_np], axis=0)
        total_seen += z_np.shape[0]
        return reservoir, total_seen

    if reservoir is None:
        take = min(max_tiles, z_np.shape[0])
        reservoir = z_np[:take].copy()
        total_seen += z_np.shape[0]
        return reservoir, total_seen

    count = reservoir.shape[0]
    for i in range(z_np.shape[0]):
        total_seen += 1
        if count < max_tiles:
            reservoir = np.vstack([reservoir, z_np[i : i + 1]])
            count += 1
        else:
            j = int(rng.integers(0, total_seen))
            if j < max_tiles:
                reservoir[j] = z_np[i]
    return reservoir, total_seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute per-latent percentile values for SAE clamp.")

    ap.add_argument("--index-json", type=str, required=True, help="manifest_index.json path")
    ap.add_argument("--slides-per-project", type=int, default=-1, help="cap per project; -1 = all")
    ap.add_argument("--require-h5-exists", action="store_true")

    ap.add_argument("--sae-ckpt", type=str, required=True)
    ap.add_argument("--sae-cfg", type=str, required=True)

    ap.add_argument("--tiles-per-slide", type=int, default=-1, help="sample tiles per slide; -1 = all")
    ap.add_argument("--chunk-tiles", type=int, default=512)
    ap.add_argument("--max-tiles", type=int, default=50000, help="reservoir size; -1 = all (may be huge)")
    ap.add_argument("--percentile", type=float, default=95.0)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, required=True, help="output .npy path for clamp values")

    args = ap.parse_args()

    slides_per_project = None if args.slides_per_project < 0 else int(args.slides_per_project)
    tiles_per_slide = -1 if args.tiles_per_slide < 0 else int(args.tiles_per_slide)

    h5_files = project_balanced_h5_list_from_index(
        Path(args.index_json),
        slides_per_project=slides_per_project,
        seed=args.seed,
        require_h5_exists=args.require_h5_exists,
    )
    if not h5_files:
        raise SystemExit("No H5 files found from index.")

    device = args.device
    rng = np.random.default_rng(args.seed)

    sae, _d_in, _d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=device)

    reservoir = None
    total_seen = 0

    for h5p in h5_files:
        X = read_h5_subset(Path(h5p), tiles_per_slide=tiles_per_slide, rng=rng)
        if X.size == 0:
            continue

        for s in range(0, X.shape[0], args.chunk_tiles):
            xb = torch.from_numpy(X[s : s + args.chunk_tiles]).to(device, non_blocking=True)
            z = sae_encode_features(sae, xb)
            z_np = z.detach().float().cpu().numpy().astype(np.float32, copy=False)

            reservoir, total_seen = reservoir_update(
                reservoir,
                z_np,
                max_tiles=int(args.max_tiles),
                rng=rng,
                total_seen=total_seen,
            )

    if reservoir is None or reservoir.shape[0] == 0:
        raise SystemExit("No samples collected. Check inputs.")

    if not (0.0 < args.percentile < 100.0):
        raise SystemExit("--percentile must be in (0, 100).")

    values = np.percentile(reservoir, args.percentile, axis=0).astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, values)

    meta = {
        "index_json": str(args.index_json),
        "slides_per_project": slides_per_project,
        "tiles_per_slide": tiles_per_slide,
        "max_tiles": int(args.max_tiles),
        "percentile": float(args.percentile),
        "total_seen": int(total_seen),
        "num_samples": int(reservoir.shape[0]),
        "out_npy": str(out_path),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Saved:", out_path)
    print("Saved:", meta_path)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _WSI_SAE_SRC = Path(__file__).resolve().parents[2] / "wsi-sae" / "src"
    if _WSI_SAE_SRC.exists():
        sys.path.insert(0, str(_WSI_SAE_SRC))
        from wsi_sae.cli import main as _wsi_sae_main

        _wsi_sae_main(["compute-percentiles", *sys.argv[1:]])
    else:
        main()
