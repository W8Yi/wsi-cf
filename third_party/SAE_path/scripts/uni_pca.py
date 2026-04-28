from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SampleRecord:
    h5_path: str
    tile_index: int


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Fit PCA on UNI tile features from a manifest split and export PCA directions for uni_steer."
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/manifests/sae_manifest_hpv100_split0_h5.json"),
        help="Manifest JSON containing split lists (e.g. train/val/test) of H5 feature paths.",
    )
    ap.add_argument("--split", type=str, default="test", help="Split key inside manifest (default: test).")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for PCA artifacts.")
    ap.add_argument(
        "--n-components",
        type=int,
        default=16,
        help="Number of PCA components to fit/export (capped by sample count and feature dim).",
    )
    ap.add_argument(
        "--tiles-per-slide",
        type=int,
        default=512,
        help="Random tile samples per slide (without replacement when possible).",
    )
    ap.add_argument(
        "--max-total-samples",
        type=int,
        default=50000,
        help=(
            "Global cap on sampled feature rows kept for PCA (default 50k). "
            "Set 0 to disable, but memory usage can become very large."
        ),
    )
    ap.add_argument(
        "--max-slides",
        type=int,
        default=0,
        help="Optional cap on slides from the split (0 means use all slides).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-shuffle-slides",
        action="store_true",
        help="Process slides in manifest order instead of shuffling before sampling.",
    )
    ap.add_argument(
        "--topk-extremes",
        type=int,
        default=20,
        help="Save top/bottom sampled records by PC score for quick inspection.",
    )
    ap.add_argument(
        "--save-sample-records",
        action="store_true",
        help="Write full sampled provenance records (can be large for high sample counts).",
    )
    ap.add_argument(
        "--save-scores-sampled",
        action="store_true",
        help="Write PCA scores for all sampled rows (optional; can be large).",
    )
    return ap


def _load_manifest_split(manifest_path: Path, split: str) -> list[str]:
    payload = json.loads(manifest_path.read_text())
    if split not in payload:
        raise KeyError(f"Split '{split}' not found in {manifest_path}. Keys: {list(payload)[:20]}")
    items = payload[split]
    if not isinstance(items, list):
        raise TypeError(f"Manifest split '{split}' must be a list, got {type(items).__name__}")

    h5_paths: list[str] = []
    for item in items:
        if isinstance(item, str):
            h5_paths.append(item)
            continue
        if isinstance(item, dict):
            for key in ("h5_path", "path", "h5", "file"):
                if key in item and isinstance(item[key], str):
                    h5_paths.append(item[key])
                    break
            else:
                raise KeyError(f"Could not infer H5 path key from manifest item keys: {sorted(item.keys())}")
            continue
        raise TypeError(f"Unsupported manifest item type: {type(item).__name__}")
    return h5_paths


def _sample_rows_from_h5(
    h5_path: str,
    *,
    tiles_per_slide: int,
    rng: np.random.Generator,
    h5py_mod: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features[k, D], sampled_indices[k])."""
    with h5py_mod.File(h5_path, "r") as f:
        if "features" not in f:
            raise KeyError(f"{h5_path}: missing dataset 'features'")
        ds = f["features"]
        shape = ds.shape
        if len(shape) == 2:
            n, d = int(shape[0]), int(shape[1])
            if n <= 0:
                raise RuntimeError(f"{h5_path}: empty feature dataset")
            k = min(int(tiles_per_slide), n)
            idx = np.sort(rng.choice(n, size=k, replace=False).astype(np.int64))
            x = ds[idx]
        elif len(shape) == 3 and int(shape[0]) == 1:
            n, d = int(shape[1]), int(shape[2])
            if n <= 0:
                raise RuntimeError(f"{h5_path}: empty feature dataset")
            k = min(int(tiles_per_slide), n)
            idx = np.sort(rng.choice(n, size=k, replace=False).astype(np.int64))
            x = ds[0, idx]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {shape}")

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise RuntimeError(f"{h5_path}: expected sampled array [k,D], got {x.shape}")
    return x, idx


def _top_bottom_records(
    scores: np.ndarray,
    records: list[SampleRecord],
    *,
    topk: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n_pc = scores.shape[1]
    topk = max(1, min(int(topk), scores.shape[0]))
    for pc in range(n_pc):
        s = scores[:, pc]
        top_idx = np.argsort(s)[-topk:][::-1]
        bot_idx = np.argsort(s)[:topk]
        out[f"pc_{pc+1:03d}"] = {
            "top": [
                {**asdict(records[i]), "score": float(s[i])}
                for i in top_idx.tolist()
            ],
            "bottom": [
                {**asdict(records[i]), "score": float(s[i])}
                for i in bot_idx.tolist()
            ],
        }
    return out


def main() -> None:
    args = _build_argparser().parse_args()

    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(f"h5py is required to read UNI feature H5 files: {exc}")

    try:
        from sklearn.decomposition import PCA  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(f"scikit-learn is required for PCA: {exc}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    directions_dir = out_dir / "directions"
    directions_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = _load_manifest_split(args.manifest, args.split)
    if args.max_slides and args.max_slides > 0:
        h5_paths = h5_paths[: int(args.max_slides)]
    if not h5_paths:
        raise SystemExit("No H5 paths found for the requested split.")

    rng = np.random.default_rng(args.seed)
    if not args.no_shuffle_slides:
        h5_paths = [h5_paths[i] for i in rng.permutation(len(h5_paths)).tolist()]

    max_total_samples = int(args.max_total_samples)
    if max_total_samples < 0:
        raise SystemExit("--max-total-samples must be >= 0")

    x_chunks: list[np.ndarray] = []
    sample_records: list[SampleRecord] = []
    skipped: list[dict[str, str]] = []
    reached_sample_cap = False

    print(f"[1/4] Sampling UNI features from split '{args.split}' ({len(h5_paths)} slides)...")
    for i, h5_path in enumerate(h5_paths, start=1):
        if max_total_samples and len(sample_records) >= max_total_samples:
            reached_sample_cap = True
            break
        try:
            x, idx = _sample_rows_from_h5(
                h5_path,
                tiles_per_slide=args.tiles_per_slide,
                rng=rng,
                h5py_mod=h5py,
            )
            if max_total_samples:
                remaining = max_total_samples - len(sample_records)
                if remaining <= 0:
                    reached_sample_cap = True
                    break
                if x.shape[0] > remaining:
                    keep_local = np.sort(rng.choice(x.shape[0], size=remaining, replace=False).astype(np.int64))
                    x = x[keep_local]
                    idx = idx[keep_local]
                    reached_sample_cap = True
            x_chunks.append(x)
            sample_records.extend(
                SampleRecord(h5_path=str(h5_path), tile_index=int(j))
                for j in idx.tolist()
            )
        except Exception as exc:
            skipped.append({"h5_path": str(h5_path), "error": str(exc)})
        if i % 10 == 0 or i == len(h5_paths):
            kept = len(sample_records)
            print(f"  processed {i}/{len(h5_paths)} slides | sampled rows kept: {kept}")
        if reached_sample_cap:
            print(f"  reached --max-total-samples={max_total_samples}; stopping sampling early")
            break

    if not x_chunks:
        raise SystemExit("Failed to sample any features from the requested split.")

    X = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
    n_samples, d = int(X.shape[0]), int(X.shape[1])
    max_components = min(n_samples, d)
    n_components = max(1, min(int(args.n_components), max_components))

    print(f"[2/4] Fitting PCA on X={X.shape} (n_components={n_components})...")
    print(f"  approx X memory (float32): {X.nbytes / (1024**2):.1f} MiB")
    # 'randomized' is usually faster for this feature size and small/medium n_components.
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(args.seed))
    scores = pca.fit_transform(X)
    components = np.asarray(pca.components_, dtype=np.float32)  # [K, D], unit vectors

    print("[3/4] Writing PCA directions and metadata...")
    np.save(out_dir / "pca_components.npy", components)
    np.save(out_dir / "pca_mean.npy", np.asarray(pca.mean_, dtype=np.float32))
    np.save(out_dir / "pca_explained_variance.npy", np.asarray(pca.explained_variance_, dtype=np.float32))
    np.save(
        out_dir / "pca_explained_variance_ratio.npy",
        np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
    )
    if args.save_scores_sampled:
        np.save(out_dir / "pca_scores_sampled.npy", np.asarray(scores, dtype=np.float32))

    for i, comp in enumerate(components, start=1):
        # sklearn PCA components are already unit norm, but we normalize explicitly for robustness.
        v = comp.astype(np.float32, copy=True)
        v /= max(float(np.linalg.norm(v)), 1e-8)
        np.save(directions_dir / f"pc_{i:03d}.npy", v)

    if args.save_sample_records:
        provenance_payload = {
            "records": [asdict(r) for r in sample_records],
            "shape": [len(sample_records)],
        }
        (out_dir / "sample_records.json").write_text(json.dumps(provenance_payload, indent=2))

    extremes_payload = _top_bottom_records(scores, sample_records, topk=args.topk_extremes)
    (out_dir / "pc_score_extremes.json").write_text(json.dumps(extremes_payload, indent=2))

    meta = {
        "manifest": str(args.manifest),
        "split": args.split,
        "seed": int(args.seed),
        "tiles_per_slide": int(args.tiles_per_slide),
        "max_slides": int(args.max_slides),
        "slides_requested": len(h5_paths),
        "sampling_stopped_early_by_cap": bool(reached_sample_cap),
        "slides_skipped": len(skipped),
        "skipped": skipped,
        "n_samples": n_samples,
        "feature_dim": d,
        "n_components": n_components,
        "max_total_samples": int(args.max_total_samples),
        "shuffle_slides": not bool(args.no_shuffle_slides),
        "explained_variance_ratio_sum": float(np.asarray(pca.explained_variance_ratio_).sum()),
        "component_files": [f"directions/pc_{i:03d}.npy" for i in range(1, n_components + 1)],
        "saved_outputs": {
            "pca_scores_sampled.npy": bool(args.save_scores_sampled),
            "sample_records.json": bool(args.save_sample_records),
            "pc_score_extremes.json": True,
        },
        "notes": [
            "PCA component signs are arbitrary; use +/- vector_strength in uni_steer to inspect both directions.",
            "pca_components.npy matches sklearn orientation [n_components, feature_dim].",
            "For large manifests, increase --max-total-samples gradually to manage memory.",
        ],
    }
    (out_dir / "pca_meta.json").write_text(json.dumps(meta, indent=2))

    print("[4/4] Done.")
    print("Saved:", out_dir / "pca_meta.json")
    print("Saved:", directions_dir)
    print(
        "Next: run `python -m scripts.uni_steer --image-dir <tile_dir> --out-dir <out> "
        "--mode delta --vector-path "
        f"{(directions_dir / 'pc_001.npy')} --vector-strength 0.5 --blend 0.5`"
    )


if __name__ == "__main__":
    main()
