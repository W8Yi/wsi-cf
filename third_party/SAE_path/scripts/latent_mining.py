#!/usr/bin/env python3
"""
latent_mining.py

Single-entry CLI (no subparsers) for 2-pass SAE latent mining.

Folder layout:
  out_root/run_name/
    pass1_stats.json
    pass2_top_tiles_<tag>_<timestamp>.json   (every pass2 run writes a new file)

Usage:

# Pass1 (creates pass1_stats.json)
python latent_mining.py --run_name RUN --mode pass1 \
  --ckpt /path/to/ckpt.pt --stage topk \
  --index_json manifest_index_enriched_with_h5.json --slides_per_project 200 --require_h5_exists \
  --tiles_per_slide 1024 --chunk_tiles 512

# Pass2 (only run_name + selection; reads pass1_stats.json)
python latent_mining.py --run_name RUN --mode pass2 \
  --select_strategy top_variance --n_latents 64 --topn 50

# Both (pass1 then one pass2)
python latent_mining.py --run_name RUN --mode both \
  --ckpt /path/to/ckpt.pt --stage topk \
  --index_json manifest_index_enriched_with_h5.json --slides_per_project 200 --require_h5_exists \
  --tiles_per_slide 1024 --chunk_tiles 512 \
  --select_strategy top_activation --n_latents 64 --topn 50
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import h5py
import numpy as np
import torch

from data.dataloader import _build_pool2x2_groups_from_coords, Pool2x2GeometryError, _resolve_h5_path
from models.sae import ReLUSparseSAE, TopKSAE, BatchTopKSAE, SDFSAE2Level, InputNormWrapper


def _slide_seed(base_seed: int, h5_path: Path, phase: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(base_seed)).encode("utf-8"))
    h.update(b"|")
    h.update(str(phase).encode("utf-8"))
    h.update(b"|")
    h.update(str(h5_path).encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)


def _slide_rng(base_seed: int, h5_path: Path, phase: str) -> np.random.Generator:
    return np.random.default_rng(_slide_seed(base_seed=base_seed, h5_path=h5_path, phase=phase))


def _skip_reason_key(e: Exception) -> str:
    msg = str(e).strip()
    kind = type(e).__name__
    if kind == "Pool2x2GeometryError":
        if "inconsistent grid spacing" in msg:
            return "Pool2x2GeometryError: inconsistent grid spacing"
        if "duplicate quantized" in msg:
            return "Pool2x2GeometryError: duplicate quantized positions"
        if "coords required" in msg:
            return "Pool2x2GeometryError: coords required"
    if kind == "KeyError" and "coords required for 10x_pool2x2" in msg:
        return "KeyError: coords required for 10x_pool2x2"
    if kind == "ValueError":
        if "Unsupported features shape" in msg:
            return "ValueError: unsupported features shape"
        if "Unsupported coords shape" in msg:
            return "ValueError: unsupported coords shape"
        if "Unsupported magnification" in msg:
            return "ValueError: unsupported magnification"
    compact = msg.split(";")[0]
    compact = compact.split(" in ")[0]
    return f"{kind}: {compact}"


def _print_skip_summary(prefix: str, skip_counts: Dict[str, int]) -> None:
    total = int(sum(int(v) for v in skip_counts.values()))
    if total <= 0:
        print(f"[{prefix}] skip summary: none")
        return
    parts = [f"{k}={int(v)}" for k, v in sorted(skip_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))]
    print(f"[{prefix}] skip summary (total={total}): " + "; ".join(parts))


def _pool_cache_key_10x(h5_path: Path, require_complete: bool) -> str:
    st = h5_path.stat()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(h5_path).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(st.st_size)).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(st.st_mtime_ns)).encode("utf-8"))
    h.update(b"|")
    h.update(b"complete" if require_complete else b"partial")
    h.update(b"|pool2x2_groups_v1")
    return h.hexdigest()


def _load_or_build_pool2x2_groups_cached(
    *,
    h5_path: Path,
    dsC: "h5py.Dataset",
    cache_dir: Optional[Path],
    require_complete: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (groups[K,4], anchors[K,2]) where anchors are top-left coords for each pooled tile.
    Uses a temporary on-disk cache when cache_dir is provided.
    """
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{_pool_cache_key_10x(h5_path, require_complete)}.npz"
        if cache_path.exists():
            try:
                obj = np.load(cache_path, allow_pickle=False)
                groups = np.asarray(obj["groups"], dtype=np.int64)
                anchors = np.asarray(obj["anchors"], dtype=np.int32)
                if groups.ndim == 2 and groups.shape[1] == 4 and anchors.ndim == 2 and anchors.shape[1] == 2 and groups.shape[0] == anchors.shape[0]:
                    return groups, anchors
            except Exception:
                # Corrupt cache entry: rebuild and overwrite.
                pass

    if dsC.ndim == 3 and dsC.shape[0] == 1:
        coords_all = dsC[0, :, :]
    elif dsC.ndim == 2:
        coords_all = dsC[:, :]
    else:
        raise ValueError(f"Unsupported coords shape: {dsC.shape} in {h5_path}")
    coords_all = np.asarray(coords_all)

    groups, _info = _build_pool2x2_groups_from_coords(coords_all, require_complete=require_complete)
    groups = np.asarray(groups, dtype=np.int64)
    anchors = np.asarray(coords_all[groups[:, 0]], dtype=np.int32) if groups.size else np.empty((0, 2), dtype=np.int32)

    if cache_path is not None:
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            with tmp.open("wb") as ftmp:
                np.savez_compressed(ftmp, groups=groups.astype(np.int32, copy=False), anchors=anchors.astype(np.int32, copy=False))
            tmp.replace(cache_path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    return groups, anchors


def infer_topk_k(ckpt: dict, cfg: dict) -> int:
    if cfg.get("topk_k", None) is not None:
        return int(cfg["topk_k"])
    stage_name = str(ckpt.get("stage_name", ""))
    m = re.search(r"k=(\d+)", stage_name)
    if m:
        return int(m.group(1))
    raise KeyError("topk_k is required for topk/batch_topk mining (pass --topk_k).")


def unwrap_base_model(model: torch.nn.Module) -> torch.nn.Module:
    base = model
    # Handles InputNormWrapper and any similar wrappers exposing `.base`.
    while hasattr(base, "base"):
        base = getattr(base, "base")
    return base


@torch.no_grad()
def build_sdf_hierarchy_payload(
    model: torch.nn.Module,
    selected_latents: List[int],
) -> Optional[dict]:
    base = unwrap_base_model(model)
    if not isinstance(base, SDFSAE2Level):
        return None

    parent = base.parent_assignment().detach().cpu().numpy().astype(np.int64, copy=False)  # (d_level1,)
    d_level2 = int(base.U.shape[1])

    counts = np.bincount(parent, minlength=d_level2).astype(np.int64, copy=False)
    selected_by_parent: Dict[int, List[int]] = {k: [] for k in range(d_level2)}
    for lj in selected_latents:
        k = int(parent[int(lj)])
        selected_by_parent[k].append(int(lj))

    return {
        "level1_dim": int(parent.shape[0]),
        "level2_dim": d_level2,
        "level1_to_level2_parent_selected": {int(lj): int(parent[int(lj)]) for lj in selected_latents},
        "level2_child_count_all_level1": {int(k): int(counts[k]) for k in range(d_level2)},
        "level2_selected_level1_children": {int(k): [int(x) for x in v] for k, v in selected_by_parent.items() if len(v) > 0},
    }


# -----------------------------
# Run helpers
# -----------------------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def run_dir(out_root: Path, run_name: str) -> Path:
    p = out_root.expanduser().resolve() / run_name
    p.mkdir(parents=True, exist_ok=True)
    return p

def pass1_path(run: Path) -> Path:
    return run / "pass1_stats.json"

def write_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2))


# -----------------------------
# Project-balanced H5 list
# -----------------------------
def project_balanced_h5_list_from_index(
    index_json: Path,
    *,
    slides_per_project: Optional[int],
    seed: int,
    require_h5_exists: bool = True,
) -> List[str]:
    idx = json.loads(index_json.read_text())

    by_proj: Dict[str, List[Path]] = defaultdict(list)
    for _slide, e in idx.items():
        proj = e.get("dataset")
        h5 = e.get("h5_path")
        if not proj or not h5:
            continue
        p = Path(h5)
        resolved = Path(_resolve_h5_path(str(p)))
        if require_h5_exists and not resolved.exists():
            continue
        by_proj[proj].append(p)

    rng = np.random.default_rng(seed)
    out: List[Path] = []
    for proj in sorted(by_proj.keys()):
        files = by_proj[proj]
        rng.shuffle(files)
        keep = files if slides_per_project is None else files[: min(slides_per_project, len(files))]
        out.extend(keep)

    rng.shuffle(out)
    return [str(p) for p in out]


def _validate_pass1_inputs(h5_files: List[Path]) -> None:
    if len(h5_files) <= 0:
        raise ValueError(
            "No H5 files were selected for pass1. "
            "Check the index/manifest paths and TCGA feature root/subdirectory mapping."
        )


def _validate_global_stats_for_selection(global_stats: Dict[str, np.ndarray]) -> None:
    mx = np.asarray(global_stats["max"], dtype=np.float32)
    finite = np.isfinite(mx)
    if int(finite.sum()) <= 0:
        raise ValueError(
            "Pass1 produced no finite latent activation maxima. "
            "Refusing to run pass2 on an empty/invalid pass1_stats.json."
        )


# -----------------------------
# SAE load + forward (TopK robust)
# -----------------------------
def load_sae_from_config(cfg: dict, device: str, ckpt_override: Optional[str] = None) -> torch.nn.Module:
    ckpt_path = ckpt_override or cfg["ckpt"]
    stage: Literal["relu", "topk", "batch_topk", "sdf2"] = cfg["stage"]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    wrapper_ln = bool(ckpt.get("wrapper_layernorm", False))
    model_sd = ckpt["model"]

    d_in = int(cfg["d_in"])
    d_latent = int(cfg["latent_dim"])
    tied = bool(cfg.get("tied", False))

    if stage == "relu":
        base = ReLUSparseSAE(d_in=d_in, d_latent=d_latent, tied=tied, use_pre_bias=True)
    elif stage == "topk":
        k = infer_topk_k(ckpt, cfg)
        base = TopKSAE(
            d_in=d_in,
            d_latent=d_latent,
            k=k,
            tied=tied,
            use_pre_bias=True,
            topk_mode=cfg.get("topk_mode", "value"),
            nonneg=bool(cfg.get("topk_nonneg", False)),
        )
    elif stage == "batch_topk":
        k = infer_topk_k(ckpt, cfg)
        base = BatchTopKSAE(
            d_in=d_in,
            d_latent=d_latent,
            k=k,
            tied=tied,
            use_pre_bias=True,
            topk_mode=cfg.get("topk_mode", "value"),
            nonneg=bool(cfg.get("topk_nonneg", False)),
        )
    elif stage == "sdf2":
        # Prefer explicit config; otherwise infer Level-2 width from checkpoint tensor shape.
        d_level2 = cfg.get("sdf_n_level2", None)
        if d_level2 is None:
            U = model_sd.get("U")
            if U is None:
                raise KeyError("sdf2 checkpoint is missing key 'U' needed to infer --sdf_n_level2.")
            d_level2 = int(U.shape[1])
        base = SDFSAE2Level(
            d_in=d_in,
            d_latent=d_latent,
            d_level2=int(d_level2),
            tied=tied,
            use_pre_bias=True,
            coeff_nonneg=bool(cfg.get("sdf_coeff_nonneg", False)),
            coeff_simplex=bool(cfg.get("sdf_coeff_simplex", False)),
        )
    else:
        raise ValueError(f"Unsupported stage '{stage}'")

    base.load_state_dict(model_sd, strict=True)
    model = InputNormWrapper(d_in, base) if wrapper_ln else base
    model.eval().to(device)
    return model

def forward_topk(model: torch.nn.Module, xb: torch.Tensor):
    out = model(xb, return_topk=True)
    if isinstance(out, tuple) and len(out) == 5:
        return out
    if isinstance(out, tuple) and len(out) == 4:
        x_hat, z, a, aux = out
        if isinstance(aux, tuple) and len(aux) == 2:
            topk_val, topk_idx = aux
            return x_hat, z, a, topk_val, topk_idx
    raise RuntimeError("Unsupported TopK return signature.")


def flatten_topk_hits(
    topk_idx: torch.Tensor,
    topk_val: torch.Tensor,
    *,
    batch_size: int,
    d_latent: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize TopK/batch_topk selected activations into flat hit tuples.

    Returns:
      rows: (M,) sample indices within the current batch/chunk
      cols: (M,) latent indices
      vals: (M,) activation values

    Supports:
    - TopKSAE return_topk -> topk_idx shape (B, K)
    - BatchTopKSAE return_topk -> topk_idx shape (B*K,) flat indices into (B, d_latent)
    """
    idx_cpu = topk_idx.detach().cpu().numpy()
    val_cpu = topk_val.detach().cpu().numpy().astype(np.float32, copy=False)

    if idx_cpu.ndim == 2:
        rows = np.repeat(np.arange(int(batch_size), dtype=np.int64), idx_cpu.shape[1])
        cols = idx_cpu.reshape(-1).astype(np.int64, copy=False)
        vals = val_cpu.reshape(-1).astype(np.float32, copy=False)
        return rows, cols, vals

    if idx_cpu.ndim == 1:
        flat = idx_cpu.astype(np.int64, copy=False)
        rows = (flat // int(d_latent)).astype(np.int64, copy=False)
        cols = (flat % int(d_latent)).astype(np.int64, copy=False)
        vals = val_cpu.reshape(-1).astype(np.float32, copy=False)
        return rows, cols, vals

    raise RuntimeError(f"Unsupported topk_idx shape {idx_cpu.shape}; expected 1D (batch_topk) or 2D (topk).")


# -----------------------------
# H5 reading (fix h5py indexing)
# -----------------------------
def read_h5_subset(
    h5_path: Path,
    *,
    tiles_per_slide: int,
    rng: np.random.Generator,
    magnification: str = "20x",
    pool2x2_require_complete: bool = True,
    pool2x2_temp_cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    resolved_h5_path = Path(_resolve_h5_path(str(h5_path)))
    with h5py.File(resolved_h5_path, "r") as f:
        dsX = f["features"]
        dsC = f.get("coords", None)

        if dsX.ndim == 3 and dsX.shape[0] == 1:
            N = dsX.shape[1]
        elif dsX.ndim == 2:
            N = dsX.shape[0]
        else:
            raise ValueError(f"Unsupported features shape: {dsX.shape} in {resolved_h5_path}")

        mag = str(magnification).lower()
        if mag == "20x":
            rows = rng.choice(N, size=tiles_per_slide, replace=False) if (tiles_per_slide > 0 and N > tiles_per_slide) else np.arange(N)
            rows = np.asarray(rows, dtype=np.int64).reshape(-1)
            rows.sort()
            rows_list = rows.tolist()

            X = dsX[0, rows_list, :] if dsX.ndim == 3 else dsX[rows_list, :]

            if dsC is None:
                C = np.zeros((len(rows_list), 2), dtype=np.int32)
            else:
                if dsC.ndim == 3 and dsC.shape[0] == 1:
                    C = dsC[0, rows_list, :]
                elif dsC.ndim == 2:
                    C = dsC[rows_list, :]
                else:
                    raise ValueError(f"Unsupported coords shape: {dsC.shape} in {resolved_h5_path}")
            return np.asarray(X, dtype=np.float32), np.asarray(C, dtype=np.int32)

        if mag != "10x_pool2x2":
            raise ValueError(f"Unsupported magnification={magnification!r} (expected '20x' or '10x_pool2x2')")
        if dsC is None:
            raise KeyError(f"coords required for 10x_pool2x2 in {resolved_h5_path}")

        groups, anchors_all = _load_or_build_pool2x2_groups_cached(
            h5_path=resolved_h5_path,
            dsC=dsC,
            cache_dir=pool2x2_temp_cache_dir,
            require_complete=pool2x2_require_complete,
        )
        M = int(groups.shape[0])
        if M <= 0:
            return np.empty((0, dsX.shape[-1]), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

        g_rows = rng.choice(M, size=tiles_per_slide, replace=False) if (tiles_per_slide > 0 and M > tiles_per_slide) else np.arange(M)
        g_rows = np.asarray(g_rows, dtype=np.int64).reshape(-1)
        g_rows.sort()

        chosen = np.asarray(groups[g_rows], dtype=np.int64)  # (K,4)
        flat = chosen.reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        uniq_list = uniq.tolist()
        srcX = dsX[0, uniq_list, :] if dsX.ndim == 3 else dsX[uniq_list, :]
        srcX = np.asarray(srcX, dtype=np.float32)
        X = srcX[inv].reshape(chosen.shape[0], 4, srcX.shape[1]).mean(axis=1, dtype=np.float32)

        # Use top-left pooled member coord for visualization/export anchoring.
        C = np.asarray(anchors_all[g_rows], dtype=np.int32)
        return X, C


# -----------------------------
# Pass 1 stats
# -----------------------------
def init_stats(d_latent: int):
    return {
        "max": np.full((d_latent,), -np.inf, dtype=np.float32),
        "sum": np.zeros((d_latent,), dtype=np.float64),
        "sum_sq": np.zeros((d_latent,), dtype=np.float64),
        "count": 0,
        "nnz": np.zeros((d_latent,), dtype=np.int64),
    }

def finalize_stats(stats: dict) -> Dict[str, np.ndarray]:
    count = max(int(stats["count"]), 1)
    mean = stats["sum"] / count
    var = (stats["sum_sq"] / count) - (mean * mean)
    var = np.maximum(var, 0.0).astype(np.float32)

    firing_rate = (stats["nnz"] / count).astype(np.float32)
    sparsity = (1.0 - firing_rate).astype(np.float32)
    return {"max": stats["max"].astype(np.float32), "var": var, "sparsity": sparsity}

@torch.no_grad()
def pass1_collect_stats(
    h5_files: List[Path],
    model: torch.nn.Module,
    stage: Literal["relu", "topk", "batch_topk", "sdf2"],
    d_latent: int,
    tiles_per_slide: int,
    chunk_tiles: int,
    device: str,
    seed: int,
    magnification: str = "20x",
    pool2x2_require_complete: bool = True,
    pool2x2_temp_cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    stats = init_stats(d_latent)
    t0 = time.time()
    skip_counts: Dict[str, int] = defaultdict(int)
    empty_slides = 0
    used_slides = 0
    total_tiles = 0
    total_chunks = 0
    n_files = len(h5_files)

    for i, h5p in enumerate(h5_files, start=1):
        slide_rng = _slide_rng(base_seed=seed, h5_path=h5p, phase="pass1")
        try:
            X, _C = read_h5_subset(
                h5p,
                tiles_per_slide=tiles_per_slide,
                rng=slide_rng,
                magnification=magnification,
                pool2x2_require_complete=pool2x2_require_complete,
                pool2x2_temp_cache_dir=pool2x2_temp_cache_dir,
            )
        except (Pool2x2GeometryError, KeyError, ValueError, FileNotFoundError, OSError) as e:
            print(f"[skip] pass1 {h5p.name}: {type(e).__name__}: {e}")
            skip_counts[_skip_reason_key(e)] += 1
            continue
        if X.shape[0] == 0:
            empty_slides += 1
            continue
        used_slides += 1
        total_tiles += int(X.shape[0])

        for s in range(0, X.shape[0], chunk_tiles):
            total_chunks += 1
            xb = torch.from_numpy(X[s:s+chunk_tiles]).to(device, non_blocking=True)

            if stage in {"topk", "batch_topk"}:
                _xhat, _z, _a, topk_val, topk_idx = forward_topk(model, xb)
                B = int(xb.shape[0])
                stats["count"] += B
                rows, cols, vals = flatten_topk_hits(topk_idx, topk_val, batch_size=B, d_latent=d_latent)
                for lj, v in zip(cols, vals):
                    lj = int(lj)
                    v = float(v)
                    if v > stats["max"][lj]:
                        stats["max"][lj] = v
                    stats["sum"][lj] += v
                    stats["sum_sq"][lj] += v * v
                    stats["nnz"][lj] += 1
            else:
                _xhat, z, _a = model(xb)
                zf = z.float()
                stats["count"] += int(zf.shape[0])

                stats["max"] = np.maximum(stats["max"], zf.max(dim=0).values.detach().cpu().numpy().astype(np.float32))
                stats["sum"] += zf.sum(dim=0).detach().cpu().numpy()
                stats["sum_sq"] += (zf * zf).sum(dim=0).detach().cpu().numpy()
                stats["nnz"] += (zf > 0).sum(dim=0).detach().cpu().numpy().astype(np.int64)

        if (i % 50 == 0) or (i == n_files):
            dt = max(1e-6, time.time() - t0)
            print(
                f"[pass1] progress {i}/{n_files} used={used_slides} empty={empty_slides} "
                f"skipped={int(sum(skip_counts.values()))} tiles={total_tiles} chunks={total_chunks} "
                f"elapsed={dt:.1f}s"
            )

    _print_skip_summary("pass1", skip_counts)
    print(
        f"[pass1] done slides={n_files} used={used_slides} empty={empty_slides} "
        f"skipped={int(sum(skip_counts.values()))} tiles={total_tiles} chunks={total_chunks} "
        f"elapsed={max(1e-6, time.time()-t0):.1f}s"
    )
    return finalize_stats(stats)


# -----------------------------
# Latent selection
# -----------------------------
def select_latents(
    global_stats: Dict[str, np.ndarray],
    strategy: Literal["top_activation", "top_variance", "top_sparsity", "manual", "sdf_parent_balanced"],
    n_latents: int,
    manual: Optional[List[int]],
    parent_assignment_all_level1: Optional[np.ndarray] = None,
    parent_max_children_per_selected_parent: int = 6,
    parent_preferred_children_per_selected_parent: int = 4,
    parent_target_count: Optional[int] = None,
) -> Tuple[List[int], Optional[dict]]:
    if n_latents <= 0:
        raise ValueError("n_latents must be > 0")

    if strategy == "manual":
        if not manual:
            raise ValueError("manual strategy requires --latent_indices")
        return manual, None

    if strategy == "sdf_parent_balanced":
        if parent_assignment_all_level1 is None:
            raise ValueError("sdf_parent_balanced requires parent_assignment_all_level1.")
        parent = np.asarray(parent_assignment_all_level1, dtype=np.int64).reshape(-1)
        if parent.shape[0] != global_stats["max"].shape[0]:
            raise ValueError(
                "parent_assignment_all_level1 length must match latent dimension: "
                f"{parent.shape[0]} vs {global_stats['max'].shape[0]}"
            )
        max_children = max(1, int(parent_max_children_per_selected_parent))
        preferred_children = max(1, int(parent_preferred_children_per_selected_parent))
        preferred_children = min(preferred_children, max_children)
        if parent_target_count is None or int(parent_target_count) <= 0:
            target_parent_count = int(math.ceil(float(n_latents) / float(preferred_children)))
        else:
            target_parent_count = int(parent_target_count)
        target_parent_count = max(1, target_parent_count)

        scores = np.asarray(global_stats["max"], dtype=np.float32).reshape(-1)
        parent_to_latents: Dict[int, List[int]] = defaultdict(list)
        for lj in range(scores.shape[0]):
            parent_to_latents[int(parent[lj])].append(int(lj))
        for pid in parent_to_latents:
            parent_to_latents[pid].sort(key=lambda lj: (float(scores[lj]), -int(lj)), reverse=True)

        ranked_parents = sorted(
            parent_to_latents.keys(),
            key=lambda pid: (float(scores[parent_to_latents[pid][0]]), -int(pid)),
            reverse=True,
        )
        selected_parents = ranked_parents[: min(target_parent_count, len(ranked_parents))]

        selected_parent_set = set(selected_parents)
        parent_capacity: Dict[int, int] = {
            int(pid): min(int(max_children), len(parent_to_latents.get(int(pid), [])))
            for pid in ranked_parents
        }
        # Use the average of the top few children as a parent-strength proxy so stronger
        # parents can earn more children without forcing a uniform 4-per-parent layout.
        parent_strength: Dict[int, float] = {}
        for pid in ranked_parents:
            latents = parent_to_latents.get(int(pid), [])
            if not latents:
                parent_strength[int(pid)] = 0.0
                continue
            top_latents = latents[: max(1, min(preferred_children, len(latents)))]
            parent_strength[int(pid)] = float(np.mean(scores[np.asarray(top_latents, dtype=np.int64)]))
        selected_parent_top_scores = [
            float(scores[parent_to_latents[int(pid)][0]])
            for pid in selected_parents
            if parent_to_latents.get(int(pid))
        ]
        if selected_parent_top_scores:
            balance_penalty = float(np.median(np.asarray(selected_parent_top_scores, dtype=np.float32)))
        else:
            balance_penalty = float(np.max(scores)) if scores.size > 0 else 1.0
        balance_penalty = balance_penalty / float(max(2, preferred_children * 2))

        quota: Dict[int, int] = defaultdict(int)

        def _seed_parents(parent_ids: List[int]) -> None:
            for pid in parent_ids:
                pid = int(pid)
                if sum(quota.values()) >= n_latents:
                    return
                if quota[pid] > 0:
                    continue
                if parent_capacity.get(pid, 0) <= 0:
                    continue
                quota[pid] = 1

        def _allocate_with_diminishing_returns(parent_ids: List[int], cap: int) -> None:
            capped_parent_ids = [int(pid) for pid in parent_ids]
            while sum(quota.values()) < n_latents:
                best_pid = None
                best_key = None
                for pid in capped_parent_ids:
                    current = int(quota.get(pid, 0))
                    cap_here = min(int(cap), int(parent_capacity.get(pid, 0)))
                    if current >= cap_here:
                        continue
                    latents = parent_to_latents.get(pid, [])
                    if current >= len(latents):
                        continue
                    next_latent = int(latents[current])
                    next_score = float(scores[next_latent])
                    utility = next_score - (balance_penalty * float(current))
                    key = (utility, next_score, -current, -pid)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_pid = pid
                if best_pid is None:
                    break
                quota[int(best_pid)] += 1

        _seed_parents(selected_parents)
        _allocate_with_diminishing_returns(selected_parents, max_children)

        if sum(quota.values()) < n_latents:
            remaining_parents = [int(pid) for pid in ranked_parents if int(pid) not in selected_parent_set]
            _seed_parents(remaining_parents)
            _allocate_with_diminishing_returns(remaining_parents, max_children)

        selected: List[int] = []
        for pid in ranked_parents:
            take = int(quota.get(int(pid), 0))
            if take <= 0:
                continue
            selected.extend(parent_to_latents[int(pid)][:take])
            if len(selected) >= n_latents:
                break
        selected = selected[:n_latents]
        selected_parents_final: Dict[int, int] = defaultdict(int)
        for lj in selected:
            selected_parents_final[int(parent[lj])] += 1

        summary = {
            "strategy": "sdf_parent_balanced",
            "parent_max_children_per_selected_parent": int(max_children),
            "parent_preferred_children_per_selected_parent": int(preferred_children),
            "parent_target_count": int(target_parent_count),
            "selected_parent_count": int(len(selected_parents_final)),
            "selected_parent_child_counts": {int(k): int(v) for k, v in sorted(selected_parents_final.items())},
        }
        return selected, summary

    scores = global_stats["max"] if strategy == "top_activation" else global_stats["var"] if strategy == "top_variance" else global_stats["sparsity"]
    idx = np.argsort(scores)[-n_latents:][::-1]
    return idx.tolist(), None


# -----------------------------
# Pass 2: top tiles
# -----------------------------
@torch.no_grad()
def pass2_top_tiles(
    h5_files: List[Path],
    model: torch.nn.Module,
    stage: Literal["relu", "topk", "batch_topk", "sdf2"],
    selected_latents: List[int],
    topn: int,
    heap_topn: Optional[int],
    tiles_per_slide: int,
    chunk_tiles: int,
    device: str,
    seed: int,
    magnification: str = "20x",
    pool2x2_require_complete: bool = True,
    pool2x2_temp_cache_dir: Optional[Path] = None,
) -> Dict[int, List[Dict]]:
    sel_set = set(selected_latents)
    heap_limit = max(int(topn), int(heap_topn) if heap_topn is not None else int(topn))
    heaps: Dict[int, List[Tuple[float, Tuple[str, int, int, int]]]] = {lj: [] for lj in selected_latents}
    t0 = time.time()
    skip_counts: Dict[str, int] = defaultdict(int)
    empty_slides = 0
    used_slides = 0
    total_tiles = 0
    total_chunks = 0
    n_files = len(h5_files)

    for i, h5p in enumerate(h5_files, start=1):
        slide_rng = _slide_rng(base_seed=seed, h5_path=h5p, phase="pass2")
        try:
            X, C = read_h5_subset(
                h5p,
                tiles_per_slide=tiles_per_slide,
                rng=slide_rng,
                magnification=magnification,
                pool2x2_require_complete=pool2x2_require_complete,
                pool2x2_temp_cache_dir=pool2x2_temp_cache_dir,
            )
        except (Pool2x2GeometryError, KeyError, ValueError, FileNotFoundError, OSError) as e:
            print(f"[skip] pass2 {h5p.name}: {type(e).__name__}: {e}")
            skip_counts[_skip_reason_key(e)] += 1
            continue
        if X.shape[0] == 0:
            empty_slides += 1
            continue
        used_slides += 1
        total_tiles += int(X.shape[0])

        for s in range(0, X.shape[0], chunk_tiles):
            total_chunks += 1
            xb = torch.from_numpy(X[s:s+chunk_tiles]).to(device, non_blocking=True)

            if stage in {"topk", "batch_topk"}:
                _xhat, _z, _a, topk_val, topk_idx = forward_topk(model, xb)
                B = int(xb.shape[0])
                rows, cols, vals = flatten_topk_hits(
                    topk_idx,
                    topk_val,
                    batch_size=B,
                    d_latent=int(_z.shape[1]),
                )

                for bi, lj_raw, score_raw in zip(rows, cols, vals):
                    lj = int(lj_raw)
                    if lj not in sel_set:
                        continue
                    bi = int(bi)
                    tile_global = s + bi
                    x0, y0 = int(C[tile_global, 0]), int(C[tile_global, 1])
                    score = float(score_raw)
                    item = (str(h5p), int(tile_global), x0, y0)
                    h = heaps[lj]
                    if len(h) < heap_limit:
                        heapq.heappush(h, (score, item))
                    elif score > h[0][0]:
                        heapq.heapreplace(h, (score, item))
            else:
                _xhat, z, _a = model(xb)
                zf = z.float().detach().cpu().numpy()
                for bi in range(zf.shape[0]):
                    tile_global = s + bi
                    x0, y0 = int(C[tile_global, 0]), int(C[tile_global, 1])
                    row = zf[bi]
                    for lj in selected_latents:
                        score = float(row[lj])
                        if score <= 0:
                            continue
                        item = (str(h5p), int(tile_global), x0, y0)
                        h = heaps[lj]
                        if len(h) < heap_limit:
                            heapq.heappush(h, (score, item))
                        elif score > h[0][0]:
                            heapq.heapreplace(h, (score, item))

        if (i % 50 == 0) or (i == n_files):
            dt = max(1e-6, time.time() - t0)
            print(
                f"[pass2] progress {i}/{n_files} used={used_slides} empty={empty_slides} "
                f"skipped={int(sum(skip_counts.values()))} tiles={total_tiles} chunks={total_chunks} "
                f"elapsed={dt:.1f}s"
            )

    out: Dict[int, List[Dict]] = {}
    for lj in selected_latents:
        items = sorted(heaps[lj], key=lambda t: t[0], reverse=True)
        out[lj] = [{"score": float(s), "h5_path": p, "tile_idx": ti, "x": int(x), "y": int(y)} for (s, (p, ti, x, y)) in items]
    _print_skip_summary("pass2", skip_counts)
    print(
        f"[pass2] done slides={n_files} used={used_slides} empty={empty_slides} "
        f"skipped={int(sum(skip_counts.values()))} tiles={total_tiles} chunks={total_chunks} "
        f"elapsed={max(1e-6, time.time()-t0):.1f}s"
    )
    return out


# -----------------------------
# Pass1/Pass2 file IO
# -----------------------------
def save_pass1(run: Path, cfg: dict, h5_files: List[str], global_stats: Dict[str, np.ndarray]) -> Path:
    p = pass1_path(run)
    payload = {
        "config": cfg,
        "h5_files": h5_files,
        "global_stats": {
            "max": global_stats["max"].tolist(),
            "var": global_stats["var"].tolist(),
            "sparsity": global_stats["sparsity"].tolist(),
        },
    }
    write_json(p, payload)
    return p

def load_pass1(p: Path) -> Tuple[dict, List[Path], Dict[str, np.ndarray]]:
    obj = json.loads(p.read_text())
    cfg = obj["config"]
    h5_files = [Path(x) for x in obj["h5_files"]]
    gs = obj["global_stats"]
    global_stats = {
        "max": np.asarray(gs["max"], dtype=np.float32),
        "var": np.asarray(gs["var"], dtype=np.float32),
        "sparsity": np.asarray(gs["sparsity"], dtype=np.float32),
    }
    return cfg, h5_files, global_stats

def save_pass2(run: Path, payload: dict, tag: str) -> Path:
    safe = re.sub(r"[^\w\-\.]+", "_", tag).strip("_")
    p = run / f"pass2_top_tiles_{safe}_{ts()}.json"
    write_json(p, payload)
    return p


def apply_per_slide_cap_to_top_tiles(
    top_tiles: Dict[int, List[Dict]],
    *,
    topn: Optional[int] = None,
    max_tiles_per_slide_per_latent: int,
    min_distance_px_same_slide_per_latent: int = -1,
) -> Dict[int, List[Dict]]:
    """
    Post-filter pass2 results so one slide/H5 does not dominate a latent exemplar list,
    and optionally enforce a same-slide minimum coordinate distance.
    Keeps the original score ordering and can stop at topn after filtering.
    """
    cap = int(max_tiles_per_slide_per_latent)
    min_dist_px = int(min_distance_px_same_slide_per_latent)
    final_topn = None if topn is None or int(topn) <= 0 else int(topn)
    if cap <= 0 and min_dist_px <= 0 and final_topn is None:
        return top_tiles
    min_dist_sq = float(min_dist_px) * float(min_dist_px)

    out: Dict[int, List[Dict]] = {}
    for lj, items in top_tiles.items():
        per_h5: Dict[str, int] = {}
        kept_xy_by_h5: Dict[str, List[Tuple[int, int]]] = {}
        kept: List[Dict] = []
        for t in items:
            h5 = str(t.get("h5_path", ""))
            if cap > 0:
                n = per_h5.get(h5, 0)
                if n >= cap:
                    continue
            if min_dist_px > 0:
                x = int(t.get("x", 0))
                y = int(t.get("y", 0))
                pts = kept_xy_by_h5.get(h5, [])
                too_close = False
                for px, py in pts:
                    dx = float(x - px)
                    dy = float(y - py)
                    if (dx * dx + dy * dy) < min_dist_sq:
                        too_close = True
                        break
                if too_close:
                    continue
                pts.append((x, y))
                kept_xy_by_h5[h5] = pts
            if cap > 0:
                per_h5[h5] = per_h5.get(h5, 0) + 1
            kept.append(t)
            if final_topn is not None and len(kept) >= final_topn:
                break
        out[int(lj)] = kept
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_root", type=str, default="./sae_mining")
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--mode", choices=["pass1", "pass2", "both"], default="both")

    # Pass1-only inputs (ignored in pass2; pass2 reads pass1_stats.json)
    ap.add_argument("--index_json", type=str, default="")
    ap.add_argument("--slides_per_project", type=int, default=10, help="cap per project; -1 = all")
    ap.add_argument("--require_h5_exists", action="store_true")
    ap.add_argument("--h5_list", type=str, default="")
    ap.add_argument("--feat_dir", type=str, default="")

    # Model (required for pass1; pass2 inferred from pass1, can override ckpt/device)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--stage", choices=["relu", "topk", "batch_topk", "sdf2"], default="")
    ap.add_argument("--d_in", type=int, default=1536)
    ap.add_argument("--latent_dim", type=int, default=12288)
    ap.add_argument("--tied", action="store_true")
    ap.add_argument("--topk_k", type=int, default=None)
    ap.add_argument("--topk_mode", choices=["value", "magnitude"], default="value")
    ap.add_argument("--topk_nonneg", action="store_true")
    ap.add_argument("--sdf_n_level2", type=int, default=None, help="Level-2 width for sdf2; inferred from ckpt if omitted")
    ap.add_argument("--sdf_coeff_nonneg", action="store_true", help="Match sdf2 coeff_nonneg training setting if needed")
    ap.add_argument("--sdf_coeff_simplex", action="store_true", help="Match sdf2 coeff_simplex training setting if needed")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--chunk_tiles", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tiles_per_slide", type=int, default=2048)
    ap.add_argument("--magnification", choices=["20x", "10x_pool2x2"], default="20x")
    ap.add_argument("--pool_allow_partial", action="store_true", help="Allow partial 2x2 blocks in 10x_pool2x2 mode (pads by repeating).")

    # Pass2 selection (used for both/both and pass2)
    ap.add_argument(
        "--select_strategy",
        choices=["top_activation", "top_variance", "top_sparsity", "manual", "sdf_parent_balanced"],
        default="top_activation",
    )
    ap.add_argument("--n_latents", type=int, default=32)
    ap.add_argument("--latent_indices", type=str, default="")
    ap.add_argument(
        "--parent_max_children_per_selected_parent",
        type=int,
        default=6,
        help="For sdf_parent_balanced selection: cap children selected per parent.",
    )
    ap.add_argument(
        "--parent_preferred_children_per_selected_parent",
        type=int,
        default=4,
        help="For sdf_parent_balanced selection: preferred children per parent before filling up to the cap.",
    )
    ap.add_argument(
        "--parent_target_count",
        type=int,
        default=-1,
        help="For sdf_parent_balanced selection: target number of parents to seed. -1 uses ceil(n_latents/preferred).",
    )
    ap.add_argument("--topn", type=int, default=64)
    ap.add_argument(
        "--topn_buffer_factor",
        type=float,
        default=4.0,
        help="When diversity filters are enabled, keep topn*factor candidates first, then filter to final topn. <=1 disables overfetch.",
    )
    ap.add_argument(
        "--max_tiles_per_slide_per_latent",
        type=int,
        default=3,
        help="Post-filter pass2 examples so each latent keeps at most this many tiles from one slide/H5 (-1 disables).",
    )
    ap.add_argument(
        "--min_distance_px_same_slide_per_latent",
        type=int,
        default=512,
        help="Minimum Euclidean distance in level-0 pixels between examples from the same slide for one latent (-1 disables).",
    )

    # Optional overrides for pass2 (still “run_name-based” if left empty)
    ap.add_argument("--ckpt_override", type=str, default="")
    ap.add_argument("--device_override", type=str, default="")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    run = run_dir(out_root, args.run_name)
    print("starting latent mining")
    print(f"Run dir: {run}")
    pool2x2_temp_cache_dir: Optional[Path] = None

    def ensure_pool2x2_temp_cache_dir() -> Path:
        nonlocal pool2x2_temp_cache_dir
        if pool2x2_temp_cache_dir is None:
            tmp = tempfile.mkdtemp(prefix="pool2x2_mining_", dir=str(run))
            pool2x2_temp_cache_dir = Path(tmp)
            print(f"[10x-cache] enabled temporary pool cache: {pool2x2_temp_cache_dir}")
        return pool2x2_temp_cache_dir

    try:
        # ---------------- pass1 ----------------
        if args.mode in ("pass1", "both"):
            print(f"=== PASS 1 ===")
            if not args.ckpt or not args.stage:
                raise SystemExit("--ckpt and --stage are required for pass1/both")

            if not args.index_json and not args.h5_list and not args.feat_dir:
                raise SystemExit("pass1 requires one of: --index_json or --h5_list or --feat_dir")

            if args.index_json:
                spp = None if args.slides_per_project < 0 else args.slides_per_project
                h5_strs = project_balanced_h5_list_from_index(
                    Path(args.index_json),
                    slides_per_project=spp,
                    seed=args.seed,
                    require_h5_exists=args.require_h5_exists,
                )
                h5_files = [Path(p) for p in h5_strs]
            elif args.h5_list:
                h5_files = [Path(x) for x in Path(args.h5_list).read_text().splitlines() if x.strip()]
            else:
                h5_files = sorted(Path(args.feat_dir).glob("*.h5"))
            _validate_pass1_inputs(h5_files)

            device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

            cfg = {
                "ckpt": args.ckpt,
                "stage": args.stage,
                "d_in": args.d_in,
                "latent_dim": args.latent_dim,
                "tied": bool(args.tied),
                "topk_k": args.topk_k,
                "topk_mode": args.topk_mode,
                "topk_nonneg": bool(args.topk_nonneg),
                "sdf_n_level2": args.sdf_n_level2,
                "sdf_coeff_nonneg": bool(args.sdf_coeff_nonneg),
                "sdf_coeff_simplex": bool(args.sdf_coeff_simplex),
                "tiles_per_slide": args.tiles_per_slide,
                "chunk_tiles": args.chunk_tiles,
                "magnification": args.magnification,
                "pool2x2_require_complete": (not args.pool_allow_partial),
                "seed": args.seed,
                "device": device,
                "index_json": args.index_json,
                "slides_per_project": args.slides_per_project,
                "require_h5_exists": bool(args.require_h5_exists),
            }

            p2_cache_dir_pass1 = ensure_pool2x2_temp_cache_dir() if str(args.magnification) == "10x_pool2x2" else None

            model = load_sae_from_config(cfg, device=device)
            global_stats = pass1_collect_stats(
                h5_files=h5_files,
                model=model,
                stage=args.stage,
                d_latent=args.latent_dim,
                tiles_per_slide=args.tiles_per_slide,
                chunk_tiles=args.chunk_tiles,
                device=device,
                seed=args.seed,
                magnification=args.magnification,
                pool2x2_require_complete=(not args.pool_allow_partial),
                pool2x2_temp_cache_dir=p2_cache_dir_pass1,
            )
            p1 = save_pass1(run, cfg=cfg, h5_files=[str(p) for p in h5_files], global_stats=global_stats)
            print(f"Wrote: {p1}")

        # ---------------- pass2 ----------------
        print(f"=== PASS 2 ===")
        if args.mode in ("pass2", "both"):
            p1 = pass1_path(run)
            if not p1.exists():
                raise SystemExit(f"Missing {p1}. Run with --mode pass1 first.")

            cfg1, h5_files, global_stats = load_pass1(p1)
            _validate_pass1_inputs(h5_files)
            _validate_global_stats_for_selection(global_stats)

            device = args.device_override or cfg1.get("device", "cuda")
            device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
            ckpt_override = args.ckpt_override or None

            model = load_sae_from_config(cfg1, device=device, ckpt_override=ckpt_override)
            stage: Literal["relu", "topk", "batch_topk", "sdf2"] = cfg1["stage"]

            manual = None
            if args.select_strategy == "manual":
                manual = [int(x.strip()) for x in args.latent_indices.split(",") if x.strip()]
            parent_assignment_all_level1 = None
            if args.select_strategy == "sdf_parent_balanced":
                base_model = unwrap_base_model(model)
                if not isinstance(base_model, SDFSAE2Level):
                    raise ValueError("select_strategy=sdf_parent_balanced requires an sdf2 model/checkpoint.")
                parent_assignment_all_level1 = (
                    base_model.parent_assignment().detach().cpu().numpy().astype(np.int64, copy=False)
                )

            selected_latents, selection_summary = select_latents(
                global_stats=global_stats,
                strategy=args.select_strategy,
                n_latents=args.n_latents,
                manual=manual,
                parent_assignment_all_level1=parent_assignment_all_level1,
                parent_max_children_per_selected_parent=int(args.parent_max_children_per_selected_parent),
                parent_preferred_children_per_selected_parent=int(args.parent_preferred_children_per_selected_parent),
                parent_target_count=(None if int(args.parent_target_count) <= 0 else int(args.parent_target_count)),
            )

            tiles_per_slide = int(cfg1["tiles_per_slide"])
            chunk_tiles = int(cfg1["chunk_tiles"])
            seed2 = int(cfg1["seed"]) + 1
            magnification = str(cfg1.get("magnification", "20x"))
            pool2x2_require_complete = bool(cfg1.get("pool2x2_require_complete", True))

            p2_cache_dir_pass2 = ensure_pool2x2_temp_cache_dir() if magnification == "10x_pool2x2" else None

            diversity_enabled = (
                int(args.max_tiles_per_slide_per_latent) > 0
                or int(args.min_distance_px_same_slide_per_latent) > 0
            )
            buffer_factor = float(args.topn_buffer_factor)
            candidate_topn = int(args.topn)
            if diversity_enabled and buffer_factor > 1.0:
                candidate_topn = max(int(args.topn), int(math.ceil(float(args.topn) * buffer_factor)))

            top_tiles = pass2_top_tiles(
                h5_files=h5_files,
                model=model,
                stage=stage,
                selected_latents=selected_latents,
                topn=args.topn,
                heap_topn=candidate_topn,
                tiles_per_slide=tiles_per_slide,
                chunk_tiles=chunk_tiles,
                device=device,
                seed=seed2,
                magnification=magnification,
                pool2x2_require_complete=pool2x2_require_complete,
                pool2x2_temp_cache_dir=p2_cache_dir_pass2,
            )
            top_tiles = apply_per_slide_cap_to_top_tiles(
                top_tiles,
                topn=int(args.topn),
                max_tiles_per_slide_per_latent=int(args.max_tiles_per_slide_per_latent),
                min_distance_px_same_slide_per_latent=int(args.min_distance_px_same_slide_per_latent),
            )

            payload = {
                "pass1_source": str(p1),
                "config_pass1": cfg1,
                "config_pass2": {
                    "select_strategy": args.select_strategy,
                    "n_latents": len(selected_latents),
                    "topn_per_latent": args.topn,
                    "latent_indices": args.latent_indices,
                    "device": device,
                    "ckpt_override": ckpt_override or "",
                    "magnification_from_pass1": magnification,
                    "parent_max_children_per_selected_parent": int(args.parent_max_children_per_selected_parent),
                    "parent_preferred_children_per_selected_parent": int(args.parent_preferred_children_per_selected_parent),
                    "parent_target_count": int(args.parent_target_count),
                    "topn_buffer_factor": float(args.topn_buffer_factor),
                    "max_tiles_per_slide_per_latent": int(args.max_tiles_per_slide_per_latent),
                    "min_distance_px_same_slide_per_latent": int(args.min_distance_px_same_slide_per_latent),
                    "candidate_topn_per_latent_before_diversity": int(candidate_topn),
                    "used_temp_pool2x2_cache": bool(p2_cache_dir_pass2 is not None),
                },
                "selected_latents": selected_latents,
                "global_stats_selected": {
                    "max_activation": {int(i): float(global_stats["max"][i]) for i in selected_latents},
                    "variance": {int(i): float(global_stats["var"][i]) for i in selected_latents},
                    "sparsity_score": {int(i): float(global_stats["sparsity"][i]) for i in selected_latents},
                },
                "top_tiles": {int(lj): top_tiles[lj] for lj in selected_latents},
            }
            if selection_summary is not None:
                payload["selection_summary"] = selection_summary
            if stage == "sdf2":
                sdf_hierarchy = build_sdf_hierarchy_payload(model=model, selected_latents=selected_latents)
                if sdf_hierarchy is not None:
                    payload["sdf_hierarchy"] = sdf_hierarchy

            tag = f"{args.select_strategy}_n{len(selected_latents)}_top{args.topn}"
            out = save_pass2(run, payload=payload, tag=tag)
            print(f"Wrote: {out}")
    finally:
        if pool2x2_temp_cache_dir is not None:
            try:
                shutil.rmtree(pool2x2_temp_cache_dir, ignore_errors=True)
                print(f"[10x-cache] removed temporary pool cache: {pool2x2_temp_cache_dir}")
            except Exception as e:
                print(f"[warn] failed to remove temporary 10x pool cache ({pool2x2_temp_cache_dir}): {e}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _WSI_SAE_SRC = Path(__file__).resolve().parents[2] / "wsi-sae" / "src"
    if _WSI_SAE_SRC.exists():
        sys.path.insert(0, str(_WSI_SAE_SRC))
        from wsi_sae.cli import main as _wsi_sae_main

        _wsi_sae_main(["mine", *sys.argv[1:]])
    else:
        main()
