from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


IMGLESS_NOTE = (
    "This script mines raw UNI-dimension top tiles and computes metrics from H5 features only. "
    "Use export/cropping scripts separately if you want tile image contact sheets."
)


@dataclass(frozen=True)
class TileRec:
    h5_path: str
    tile_idx: int
    x: int
    y: int
    slide_id: str
    site: str


class ScoreReservoir:
    """Reservoir sampler for approximate score distribution metrics."""

    def __init__(self, max_size: int, rng: np.random.Generator):
        self.max_size = int(max_size)
        self.rng = rng
        self.buf = np.empty((max(0, self.max_size),), dtype=np.float32)
        self.n = 0

    def add_many(self, vals: np.ndarray) -> None:
        vals = np.asarray(vals, dtype=np.float32).reshape(-1)
        if vals.size == 0 or self.max_size <= 0:
            self.n += int(vals.size)
            return
        for v in vals:
            self.n += 1
            if self.n <= self.max_size:
                self.buf[self.n - 1] = float(v)
            else:
                j = int(self.rng.integers(0, self.n))
                if j < self.max_size:
                    self.buf[j] = float(v)

    def values(self) -> np.ndarray:
        m = min(self.n, self.max_size)
        return self.buf[:m].copy() if m > 0 else np.empty((0,), dtype=np.float32)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Mine raw UNI dimension top tiles and compute baseline metrics "
            "(top-tile purity/diversity/contrast/stability) from H5 features."
        )
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest JSON with split lists (same format used by scripts.uni_pca).",
    )
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out-json", type=Path, required=True)

    ap.add_argument(
        "--axes",
        type=str,
        default="",
        help="Comma-separated UNI dimension indices (e.g. 0,1,42). Required unless --axes-range is set.",
    )
    ap.add_argument(
        "--axes-range",
        type=str,
        default="",
        help="Inclusive range start:end (python-style end exclusive), e.g. 0:32.",
    )
    ap.add_argument(
        "--signs",
        type=str,
        default="pos,neg",
        help="Comma-separated signs to mine: pos,neg",
    )

    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument(
        "--top-pool-factor",
        type=int,
        default=5,
        help="Keep top_n * factor candidates before final per-slide/coord dedupe filtering.",
    )
    ap.add_argument(
        "--per-slide-cap",
        type=int,
        default=0,
        help="Optional cap of selected top tiles per slide for final top-N (0 disables).",
    )
    ap.add_argument(
        "--min-coord-dist-px",
        type=int,
        default=0,
        help="Optional suppression of nearby tiles on same slide in final top-N using Manhattan distance.",
    )

    ap.add_argument("--tiles-per-slide", type=int, default=512)
    ap.add_argument("--max-slides", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-tiles", type=int, default=4096)
    ap.add_argument(
        "--zscore",
        action="store_true",
        help="Rank by z-scored UNI dimension values (recommended for comparing axes fairly).",
    )

    ap.add_argument(
        "--score-reservoir-size",
        type=int,
        default=20000,
        help="Per-axis/sign reservoir size for approximate contrast quantiles.",
    )
    ap.add_argument(
        "--bootstrap-p",
        type=float,
        default=0.5,
        help="Bernoulli inclusion probability for approximate bootstrap Jaccard stability.",
    )
    ap.add_argument(
        "--bootstrap-seed",
        type=int,
        default=123,
        help="Seed for bootstrap inclusion sampling.",
    )
    ap.add_argument(
        "--skip-top-vectors",
        action="store_true",
        help="Skip pairwise cosine/diversity metric (faster, less H5 rereading).",
    )
    return ap


def _load_manifest_split(manifest_path: Path, split: str) -> list[str]:
    payload = json.loads(manifest_path.read_text())
    if split not in payload:
        raise KeyError(f"Split '{split}' not found in {manifest_path}. Keys: {list(payload)[:20]}")
    items = payload[split]
    if not isinstance(items, list):
        raise TypeError(f"Manifest split '{split}' must be a list.")
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for k in ("h5_path", "path", "h5", "file"):
                if k in item and isinstance(item[k], str):
                    out.append(item[k])
                    break
            else:
                raise KeyError(f"Could not infer h5 path from manifest item keys: {list(item.keys())}")
        else:
            raise TypeError(f"Unsupported manifest item type: {type(item).__name__}")
    return out


def _read_h5_subset(
    h5_path: str,
    *,
    tiles_per_slide: int,
    rng: np.random.Generator,
    h5py_mod: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X[k,D], coords[k,2], original tile indices[k]."""
    with h5py_mod.File(h5_path, "r") as f:
        if "features" not in f:
            raise KeyError(f"{h5_path}: missing dataset 'features'")
        dsX = f["features"]
        dsC = f.get("coords", None)

        if dsX.ndim == 3 and int(dsX.shape[0]) == 1:
            n = int(dsX.shape[1])
            read_feat = lambda rows: dsX[0, rows, :]
            if dsC is not None:
                if dsC.ndim == 3 and int(dsC.shape[0]) == 1:
                    read_coords = lambda rows: dsC[0, rows, :]
                elif dsC.ndim == 2:
                    read_coords = lambda rows: dsC[rows, :]
                else:
                    raise ValueError(f"{h5_path}: unsupported coords shape {dsC.shape}")
            else:
                read_coords = None
        elif dsX.ndim == 2:
            n = int(dsX.shape[0])
            read_feat = lambda rows: dsX[rows, :]
            if dsC is not None:
                if dsC.ndim == 2:
                    read_coords = lambda rows: dsC[rows, :]
                elif dsC.ndim == 3 and int(dsC.shape[0]) == 1:
                    read_coords = lambda rows: dsC[0, rows, :]
                else:
                    raise ValueError(f"{h5_path}: unsupported coords shape {dsC.shape}")
            else:
                read_coords = None
        else:
            raise ValueError(f"{h5_path}: unsupported features shape {dsX.shape}")

        if n <= 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 2), dtype=np.int32),
                np.empty((0,), dtype=np.int64),
            )
        k = min(int(tiles_per_slide), n) if int(tiles_per_slide) > 0 else n
        rows = np.sort(rng.choice(n, size=k, replace=False).astype(np.int64)) if k < n else np.arange(n, dtype=np.int64)
        X = np.asarray(read_feat(rows.tolist()), dtype=np.float32)
        if read_coords is None:
            C = np.zeros((rows.shape[0], 2), dtype=np.int32)
        else:
            C = np.asarray(read_coords(rows.tolist()), dtype=np.int32)
            if C.ndim != 2 or C.shape[1] < 2:
                C = np.zeros((rows.shape[0], 2), dtype=np.int32)
            else:
                C = C[:, :2]
    return X, C, rows


def _parse_axes(args: argparse.Namespace) -> list[int]:
    axes: list[int] = []
    if args.axes:
        axes.extend(int(x.strip()) for x in args.axes.split(",") if x.strip())
    if args.axes_range:
        s, e = args.axes_range.split(":")
        axes.extend(list(range(int(s), int(e))))
    axes = sorted(set(axes))
    if not axes:
        raise SystemExit("Provide --axes or --axes-range")
    return axes


def _parse_signs(text: str) -> list[str]:
    out = [x.strip().lower() for x in text.split(",") if x.strip()]
    bad = [x for x in out if x not in {"pos", "neg"}]
    if bad:
        raise SystemExit(f"Unsupported signs {bad}; use pos,neg")
    if not out:
        raise SystemExit("No signs selected")
    return out


def _tile_id_key(h5_path: str, tile_idx: int) -> str:
    return f"{h5_path}::tile::{int(tile_idx)}"


def _slide_id_from_h5(h5_path: str) -> str:
    return Path(h5_path).stem


def _site_from_h5(h5_path: str) -> str:
    p = Path(h5_path)
    if p.parent.name:
        return p.parent.name
    return "unknown"


def _entropy_from_counts(counts: dict[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = float(c) / total
        if p > 0:
            h -= p * math.log(p + 1e-12)
    return float(h)


def _cosine_mean_pairwise(X: np.ndarray) -> float | None:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.clip(nrm, 1e-8, None)
    Y = X / nrm
    S = Y @ Y.T
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    if iu[0].size == 0:
        return None
    return float(S[iu].mean())


def _load_tile_vectors(records: list[dict[str, Any]], h5py_mod: Any) -> np.ndarray:
    by_h5: dict[str, list[tuple[int, int]]] = {}
    for pos, rec in enumerate(records):
        by_h5.setdefault(str(rec["h5_path"]), []).append((int(rec["tile_idx"]), pos))

    out = None
    first_dim = None
    filled = 0
    for h5_path, pairs in by_h5.items():
        pairs_sorted = sorted(pairs, key=lambda t: t[0])
        idx = np.asarray([p[0] for p in pairs_sorted], dtype=np.int64)
        with h5py_mod.File(h5_path, "r") as f:
            ds = f["features"]
            if ds.ndim == 3 and int(ds.shape[0]) == 1:
                X = np.asarray(ds[0, idx.tolist(), :], dtype=np.float32)
            elif ds.ndim == 2:
                X = np.asarray(ds[idx.tolist(), :], dtype=np.float32)
            else:
                raise ValueError(f"{h5_path}: unsupported features shape {ds.shape}")
        if first_dim is None:
            first_dim = int(X.shape[1])
            out = np.zeros((len(records), first_dim), dtype=np.float32)
        assert out is not None
        for row_local, (_tile_idx, pos_out) in enumerate(pairs_sorted):
            out[pos_out] = X[row_local]
            filled += 1
    if out is None or filled != len(records):
        raise RuntimeError("Failed to load all top-tile feature vectors.")
    return out


def _percentiles_from_sample(sample: np.ndarray) -> dict[str, float]:
    sample = np.asarray(sample, dtype=np.float32).reshape(-1)
    if sample.size == 0:
        return {}
    p = np.percentile(sample, [50, 90, 95, 99]).astype(np.float32)
    top1_cut = float(p[-1])
    top1_vals = sample[sample >= top1_cut]
    return {
        "sample_count": int(sample.size),
        "p50": float(p[0]),
        "p90": float(p[1]),
        "p95": float(p[2]),
        "p99": float(p[3]),
        "top1pct_mean": float(top1_vals.mean()) if top1_vals.size else float(p[3]),
        "contrast_top1pct_over_median_eps": float((top1_vals.mean() if top1_vals.size else p[3]) / (float(p[0]) + 1e-6)),
    }


def _finalize_top_records(
    heap_items: list[tuple[float, tuple[str, int, int, int, float, float]]],
    *,
    top_n: int,
    per_slide_cap: int,
    min_coord_dist_px: int,
) -> list[dict[str, Any]]:
    # heap items hold rank_score, (h5, tile_idx, x, y, raw_value, zscore_value)
    sorted_items = sorted(heap_items, key=lambda t: t[0], reverse=True)
    out: list[dict[str, Any]] = []
    per_slide_used: dict[str, int] = {}
    selected_coords_by_slide: dict[str, list[tuple[int, int]]] = {}
    for rank_score, item in sorted_items:
        h5_path, tile_idx, x, y, raw_val, z_val = item
        slide_id = _slide_id_from_h5(h5_path)
        site = _site_from_h5(h5_path)
        if per_slide_cap > 0 and per_slide_used.get(slide_id, 0) >= per_slide_cap:
            continue
        if min_coord_dist_px > 0:
            prev = selected_coords_by_slide.get(slide_id, [])
            if any(abs(int(x) - px) + abs(int(y) - py) < int(min_coord_dist_px) for px, py in prev):
                continue
        out.append(
            {
                "score": float(rank_score),
                "raw_value": float(raw_val),
                "zscore_value": float(z_val),
                "h5_path": str(h5_path),
                "tile_idx": int(tile_idx),
                "x": int(x),
                "y": int(y),
                "slide_id": slide_id,
                "site": site,
            }
        )
        per_slide_used[slide_id] = per_slide_used.get(slide_id, 0) + 1
        selected_coords_by_slide.setdefault(slide_id, []).append((int(x), int(y)))
        if len(out) >= int(top_n):
            break
    return out


def main() -> None:
    args = _build_argparser().parse_args()
    axes = _parse_axes(args)
    signs = _parse_signs(args.signs)
    out_json = args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"h5py is required: {exc}")

    rng = np.random.default_rng(args.seed)
    h5_paths = _load_manifest_split(args.manifest, args.split)
    if args.max_slides and int(args.max_slides) > 0:
        h5_paths = h5_paths[: int(args.max_slides)]
    if not h5_paths:
        raise SystemExit("No H5 files found for split.")
    h5_paths = [h5_paths[i] for i in rng.permutation(len(h5_paths)).tolist()]

    # Pass 1: stats for z-scoring (selected dims only).
    k_axes = len(axes)
    sum_v = np.zeros((k_axes,), dtype=np.float64)
    sumsq_v = np.zeros((k_axes,), dtype=np.float64)
    count_v = 0

    print(f"[1/3] Pass1 stats on {len(h5_paths)} slides for {k_axes} raw UNI dims...")
    for i, h5p in enumerate(h5_paths, start=1):
        try:
            X, _C, _rows = _read_h5_subset(
                h5p,
                tiles_per_slide=int(args.tiles_per_slide),
                rng=rng,
                h5py_mod=h5py,
            )
        except Exception as exc:
            print(f"[warn] skip {h5p}: {exc}")
            continue
        if X.size == 0:
            continue
        if X.ndim != 2:
            print(f"[warn] skip {h5p}: bad feature shape {X.shape}")
            continue
        d = X.shape[1]
        if any(ax < 0 or ax >= d for ax in axes):
            raise SystemExit(f"Axis out of bounds for feature dim D={d}; requested axes include {[a for a in axes if a<0 or a>=d][:5]}")
        S = X[:, axes]
        sum_v += S.sum(axis=0, dtype=np.float64)
        sumsq_v += np.square(S, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count_v += int(S.shape[0])
        if i % 20 == 0 or i == len(h5_paths):
            print(f"  pass1 processed {i}/{len(h5_paths)} slides | tiles seen={count_v}")
    if count_v <= 0:
        raise SystemExit("No tiles were read in pass1.")
    mean_v = sum_v / max(count_v, 1)
    var_v = np.maximum((sumsq_v / max(count_v, 1)) - (mean_v * mean_v), 0.0)
    std_v = np.sqrt(var_v.astype(np.float64, copy=False))
    std_v = np.where(std_v > 1e-8, std_v, 1.0)
    axis_to_pos = {ax: i for i, ax in enumerate(axes)}

    # Pass 2: top heaps + metrics reservoirs + bootstrap heaps.
    pool_n = max(int(args.top_n), 1) * max(int(args.top_pool_factor), 1)
    sign_multiplier = {"pos": 1.0, "neg": -1.0}
    heaps: dict[tuple[int, str], list[tuple[float, tuple[str, int, int, int, float, float]]]] = {
        (ax, sg): [] for ax in axes for sg in signs
    }
    boot_heaps_a: dict[tuple[int, str], list[tuple[float, tuple[str, int]]]] = {(ax, sg): [] for ax in axes for sg in signs}
    boot_heaps_b: dict[tuple[int, str], list[tuple[float, tuple[str, int]]]] = {(ax, sg): [] for ax in axes for sg in signs}
    boot_rng = np.random.default_rng(int(args.bootstrap_seed))
    score_res: dict[tuple[int, str], ScoreReservoir] = {
        (ax, sg): ScoreReservoir(int(args.score_reservoir_size), np.random.default_rng(int(args.seed) + ax * 17 + (1 if sg == "neg" else 0)))
        for ax in axes for sg in signs
    }
    slide_total_counts: dict[tuple[int, str], dict[str, int]] = {(ax, sg): {} for ax in axes for sg in signs}
    site_total_counts: dict[tuple[int, str], dict[str, int]] = {(ax, sg): {} for ax in axes for sg in signs}

    print("[2/3] Pass2 top-tile mining + metric accumulators...")
    seen_tiles = 0
    for i, h5p in enumerate(h5_paths, start=1):
        try:
            X, C, rows = _read_h5_subset(
                h5p,
                tiles_per_slide=int(args.tiles_per_slide),
                rng=rng,
                h5py_mod=h5py,
            )
        except Exception as exc:
            print(f"[warn] skip {h5p}: {exc}")
            continue
        if X.size == 0:
            continue
        slide_id = _slide_id_from_h5(h5p)
        site = _site_from_h5(h5p)
        vals = X[:, axes]  # [n, k_axes]
        zvals = (vals - mean_v[None, :]) / std_v[None, :] if args.zscore else vals

        for s in signs:
            mult = float(sign_multiplier[s])
            rank_scores = mult * zvals  # [n, k_axes]
            for j, ax in enumerate(axes):
                rs = np.asarray(rank_scores[:, j], dtype=np.float32)
                score_res[(ax, s)].add_many(rs)
                slide_total_counts[(ax, s)][slide_id] = slide_total_counts[(ax, s)].get(slide_id, 0) + int(rs.shape[0])
                site_total_counts[(ax, s)][site] = site_total_counts[(ax, s)].get(site, 0) + int(rs.shape[0])

                # Approx bootstrap stability (same pool, independent Bernoulli includes).
                p = float(args.bootstrap_p)
                if p > 0:
                    inc_a = boot_rng.random(rs.shape[0]) < p
                    inc_b = boot_rng.random(rs.shape[0]) < p
                    ha = boot_heaps_a[(ax, s)]
                    hb = boot_heaps_b[(ax, s)]
                    for bi in np.flatnonzero(inc_a):
                        item_id = (str(h5p), int(rows[bi]))
                        sc = float(rs[bi])
                        if len(ha) < int(args.top_n):
                            heapq.heappush(ha, (sc, item_id))
                        elif sc > ha[0][0]:
                            heapq.heapreplace(ha, (sc, item_id))
                    for bi in np.flatnonzero(inc_b):
                        item_id = (str(h5p), int(rows[bi]))
                        sc = float(rs[bi])
                        if len(hb) < int(args.top_n):
                            heapq.heappush(hb, (sc, item_id))
                        elif sc > hb[0][0]:
                            heapq.heapreplace(hb, (sc, item_id))

                h = heaps[(ax, s)]
                raw_col = vals[:, j]
                z_col = zvals[:, j]
                for bi in range(rs.shape[0]):
                    sc = float(rs[bi])
                    item = (
                        str(h5p),
                        int(rows[bi]),
                        int(C[bi, 0]) if C.ndim == 2 and C.shape[1] >= 2 else 0,
                        int(C[bi, 1]) if C.ndim == 2 and C.shape[1] >= 2 else 0,
                        float(raw_col[bi]),
                        float(z_col[bi]),
                    )
                    if len(h) < pool_n:
                        heapq.heappush(h, (sc, item))
                    elif sc > h[0][0]:
                        heapq.heapreplace(h, (sc, item))

        seen_tiles += int(X.shape[0])
        if i % 10 == 0 or i == len(h5_paths):
            print(f"  pass2 processed {i}/{len(h5_paths)} slides | tiles seen={seen_tiles}")

    # Finalize per-axis/sign outputs.
    print("[3/3] Finalizing top tiles + metrics...")
    axis_results: dict[str, Any] = {}
    for ax in axes:
        axis_entry: dict[str, Any] = {
            "axis_index": int(ax),
            "global_axis_stats_on_sampled_tiles": {
                "mean": float(mean_v[axis_to_pos[ax]]),
                "std": float(std_v[axis_to_pos[ax]]),
                "var": float(var_v[axis_to_pos[ax]]),
                "count": int(count_v),
                "zscore_ranking": bool(args.zscore),
            },
            "signs": {},
        }

        for s in signs:
            final_records = _finalize_top_records(
                heaps[(ax, s)],
                top_n=int(args.top_n),
                per_slide_cap=int(args.per_slide_cap),
                min_coord_dist_px=int(args.min_coord_dist_px),
            )

            # Cosine purity/diversity in raw UNI space among top-N vectors.
            pairwise_mean_cos = None
            diversity = None
            if final_records and not args.skip_top_vectors:
                try:
                    X_top = _load_tile_vectors(final_records, h5py)
                    pairwise_mean_cos = _cosine_mean_pairwise(X_top)
                    diversity = None if pairwise_mean_cos is None else float(1.0 - pairwise_mean_cos)
                except Exception as exc:
                    print(f"[warn] axis {ax} sign {s}: failed top-vector load for cosine metric: {exc}")

            # Contrast from reservoir (approx).
            contrast_stats = _percentiles_from_sample(score_res[(ax, s)].values())

            # Bootstrap Jaccard@N (approx, same tile pool with independent Bernoulli inclusion).
            topA = {
                _tile_id_key(h5p, tile_idx)
                for _sc, (h5p, tile_idx) in sorted(boot_heaps_a[(ax, s)], key=lambda t: t[0], reverse=True)
            }
            topB = {
                _tile_id_key(h5p, tile_idx)
                for _sc, (h5p, tile_idx) in sorted(boot_heaps_b[(ax, s)], key=lambda t: t[0], reverse=True)
            }
            inter = len(topA & topB)
            union = len(topA | topB)
            jacc_boot = float(inter / union) if union > 0 else None

            # Top-N confound summaries.
            top_slide_counts: dict[str, int] = {}
            top_site_counts: dict[str, int] = {}
            for rec in final_records:
                top_slide_counts[rec["slide_id"]] = top_slide_counts.get(rec["slide_id"], 0) + 1
                top_site_counts[rec["site"]] = top_site_counts.get(rec["site"], 0) + 1

            sign_entry = {
                "top_tiles": final_records,
                "metrics": {
                    "top_n_actual": int(len(final_records)),
                    "intra_set_mean_pairwise_cosine_uni": pairwise_mean_cos,
                    "diversity_1_minus_mean_cosine": diversity,
                    "activation_contrast_reservoir_approx": contrast_stats,
                    "bootstrap_tile_jaccard_at_n_estimate": jacc_boot,
                    "bootstrap_inclusion_p": float(args.bootstrap_p),
                    "top_slide_entropy": _entropy_from_counts(top_slide_counts),
                    "top_site_entropy": _entropy_from_counts(top_site_counts),
                    "top_unique_slides": int(len(top_slide_counts)),
                    "top_unique_sites": int(len(top_site_counts)),
                    "top_slide_counts": top_slide_counts,
                    "top_site_counts": top_site_counts,
                },
                "notes": [
                    "Scores are sign-adjusted rank scores (pos uses +axis, neg uses -axis).",
                    "Contrast and quantiles are estimated from a reservoir sample unless score_reservoir_size<=0.",
                    "Bootstrap Jaccard is an approximate same-pool stability estimate using independent Bernoulli inclusion, not disjoint-half overlap.",
                ],
            }
            axis_entry["signs"][s] = sign_entry

        axis_results[str(ax)] = axis_entry

    payload = {
        "manifest": str(args.manifest),
        "split": args.split,
        "axes": axes,
        "signs": signs,
        "top_n": int(args.top_n),
        "tiles_per_slide": int(args.tiles_per_slide),
        "max_slides": int(args.max_slides),
        "seed": int(args.seed),
        "zscore": bool(args.zscore),
        "notes": [IMGLESS_NOTE],
        "config": {
            "top_pool_factor": int(args.top_pool_factor),
            "per_slide_cap": int(args.per_slide_cap),
            "min_coord_dist_px": int(args.min_coord_dist_px),
            "score_reservoir_size": int(args.score_reservoir_size),
            "bootstrap_p": float(args.bootstrap_p),
        },
        "axis_results": axis_results,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
