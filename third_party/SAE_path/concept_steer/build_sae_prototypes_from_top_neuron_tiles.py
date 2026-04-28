#!/usr/bin/env python3
"""
Build full SAE latent prototype vectors from top_neuron_tiles.csv.

This bridges the HNSC HPV neuron-mining pipeline with prototype-based steering:
- input: top_neuron_tiles.csv from run_hnsc_hpv_sae_neuron_pipeline.py
- output: NPZ/JSON compatible with scripts/sae_steer_sweep.py prototype_* modes
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sae import load_sae_from_config, sae_encode_features


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles-csv", type=Path, required=True, help="top_neuron_tiles.csv")
    ap.add_argument("--sae-ckpt", type=Path, required=True, help="SAE checkpoint used for encoding.")
    ap.add_argument("--sae-cfg", type=Path, required=True, help="SAE config JSON.")
    ap.add_argument(
        "--manifest-index",
        type=Path,
        default=REPO_ROOT / "metadata" / "indexes" / "manifest_index.json",
        help="Slide-key -> absolute H5 path index used to resolve dataset-specific H5 locations.",
    )
    ap.add_argument("--out-npz", type=Path, required=True, help="Output NPZ with prototype vectors.")
    ap.add_argument("--out-json", type=Path, required=True, help="Output JSON metadata/stats.")
    ap.add_argument("--top-n", type=int, default=0, help="Use first N prototype-ranked tiles per latent (0 = all).")
    ap.add_argument("--latent-limit", type=int, default=0, help="Optional cap on latent count.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512, help="SAE encoding batch size.")
    ap.add_argument(
        "--save-top-codes",
        action="store_true",
        help="Also save per-tile SAE codes used for each latent. This can be large.",
    )
    ap.add_argument(
        "--topk-dims",
        type=int,
        default=16,
        help="Store top abs dims of each prototype in metadata for quick inspection.",
    )
    return ap


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_h5_feature_rows(h5_path: str, idx: np.ndarray, h5py_mod: Any) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    order = np.argsort(idx)
    idx_sorted = idx[order]
    with h5py_mod.File(h5_path, "r") as f:
        if "features" not in f:
            raise KeyError(f"{h5_path}: missing dataset 'features'")
        ds = f["features"]
        shape = ds.shape
        if len(shape) == 2:
            n = int(shape[0])
            if idx_sorted[-1] >= n:
                raise IndexError(f"{h5_path}: tile_idx out of bounds (max {idx_sorted[-1]} >= N={n})")
            x_sorted = ds[idx_sorted]
        elif len(shape) == 3 and int(shape[0]) == 1:
            n = int(shape[1])
            if idx_sorted[-1] >= n:
                raise IndexError(f"{h5_path}: tile_idx out of bounds (max {idx_sorted[-1]} >= N={n})")
            x_sorted = ds[0, idx_sorted]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {shape}")

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return np.asarray(x_sorted, dtype=np.float32)[inv]


def _encode_sae_in_batches(
    sae_model: torch.nn.Module,
    x_np: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if x_np.ndim != 2:
        raise ValueError(f"Expected x_np [N,D], got {x_np.shape}")
    if x_np.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)

    outs: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, x_np.shape[0], int(batch_size)):
            xb = torch.from_numpy(x_np[i : i + int(batch_size)]).to(device=device, dtype=torch.float32)
            zb = sae_encode_features(sae_model, xb)
            outs.append(zb.detach().float().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outs, axis=0) if outs else np.empty((0, 0), dtype=np.float32)


def _top_abs_dims(vec: np.ndarray, k: int) -> list[dict[str, float]]:
    if k <= 0:
        return []
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return []
    k = min(int(k), int(v.size))
    idx = np.argsort(np.abs(v))[-k:][::-1]
    return [{"dim": int(i), "value": float(v[i])} for i in idx.tolist()]


def _latent_order(rows_by_latent: dict[int, list[dict[str, Any]]]) -> list[int]:
    ordered = sorted(
        rows_by_latent.keys(),
        key=lambda lid: (
            rows_by_latent[lid][0].get("selected_direction", ""),
            rows_by_latent[lid][0].get("prototype_rank", 10**9),
            lid,
        ),
    )
    return [int(x) for x in ordered]


def _resolve_h5_path(
    raw_h5_path: str,
    *,
    slide_key: str,
    manifest_index: dict[str, Any],
) -> str:
    p = Path(raw_h5_path)
    if p.is_absolute() and p.exists():
        return str(p)

    repo_rel = (REPO_ROOT / p).resolve()
    if repo_rel.exists():
        return str(repo_rel)

    rec = manifest_index.get(slide_key)
    if isinstance(rec, dict):
        h5_abs = rec.get("h5_path")
        if isinstance(h5_abs, str) and Path(h5_abs).exists():
            return h5_abs

    return raw_h5_path


def main() -> None:
    args = _build_argparser().parse_args()

    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"h5py is required to read UNI feature H5 files: {exc}")

    rows_raw = _read_csv(args.tiles_csv)
    if not rows_raw:
        raise SystemExit(f"No rows found in {args.tiles_csv}")
    manifest_index = _load_json(args.manifest_index)
    if not isinstance(manifest_index, dict):
        raise SystemExit(f"Expected dict root in {args.manifest_index}")

    rows_by_latent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_raw:
        latent_idx = _safe_int(row.get("latent_idx"), -1)
        if latent_idx < 0:
            continue
        rec = {
            "latent_idx": latent_idx,
            "selected_direction": str(row.get("selected_direction", "")),
            "prototype_rank": _safe_int(row.get("prototype_rank"), 10**9),
            "label": _safe_int(row.get("label"), -1),
            "pred": _safe_int(row.get("pred"), -1),
            "prob_pos": _safe_float(row.get("prob_pos"), 0.0),
            "attention": _safe_float(row.get("attention"), 0.0),
            "sae_activation": _safe_float(row.get("sae_activation"), 0.0),
            "attention_weighted_activation": _safe_float(row.get("attention_weighted_activation"), 0.0),
            "case_id": str(row.get("case_id", "")),
            "slide_key": str(row.get("slide_key", "")),
            "tile_index": _safe_int(row.get("tile_index"), -1),
            "h5_path": _resolve_h5_path(
                str(row.get("h5_path", "")),
                slide_key=str(row.get("slide_key", "")),
                manifest_index=manifest_index,
            ),
        }
        if rec["tile_index"] < 0 or not rec["h5_path"]:
            continue
        rows_by_latent[latent_idx].append(rec)

    latent_order = _latent_order(rows_by_latent)
    if args.latent_limit and args.latent_limit > 0:
        latent_order = latent_order[: int(args.latent_limit)]
    if not latent_order:
        raise SystemExit("No valid latents found in tiles CSV.")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    print("[setup] Loading SAE...")
    sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=args.device)

    latent_ids_out: list[int] = []
    selected_direction_out: list[str] = []
    prototype_mean_rows: list[np.ndarray] = []
    prototype_median_rows: list[np.ndarray] = []
    self_score_stats: dict[str, dict[str, float]] = {}
    per_latent_meta: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    all_top_codes: list[np.ndarray] = []
    top_code_row_counts: list[int] = []

    for i, latent_idx in enumerate(latent_order, start=1):
        recs = sorted(rows_by_latent[int(latent_idx)], key=lambda r: (r["prototype_rank"], -r["attention_weighted_activation"]))
        if args.top_n and args.top_n > 0:
            recs = recs[: int(args.top_n)]
        if not recs:
            skipped.append({"latent_idx": int(latent_idx), "reason": "no_records_after_top_n"})
            continue

        by_h5: dict[str, list[int]] = defaultdict(list)
        raw_scores: list[float] = []
        for rec in recs:
            by_h5[rec["h5_path"]].append(int(rec["tile_index"]))
            raw_scores.append(float(rec["sae_activation"]))

        x_chunks: list[np.ndarray] = []
        load_errors: list[str] = []
        for h5_path, idxs in by_h5.items():
            try:
                x = _load_h5_feature_rows(h5_path, np.asarray(idxs, dtype=np.int64), h5py_mod=h5py)
                x_chunks.append(x)
            except Exception as exc:
                load_errors.append(f"{h5_path}: {exc}")

        if not x_chunks:
            skipped.append({"latent_idx": int(latent_idx), "reason": "all_h5_loads_failed", "errors": load_errors[:5]})
            continue

        X = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
        if X.ndim != 2 or X.shape[1] != d_in:
            skipped.append(
                {
                    "latent_idx": int(latent_idx),
                    "reason": "bad_feature_shape",
                    "shape": list(X.shape),
                    "expected_d_in": int(d_in),
                }
            )
            continue

        Z = _encode_sae_in_batches(sae_model, X, device=args.device, batch_size=int(args.batch_size))
        if Z.ndim != 2 or Z.shape[1] != d_latent:
            skipped.append(
                {
                    "latent_idx": int(latent_idx),
                    "reason": "bad_sae_code_shape",
                    "shape": list(Z.shape),
                    "expected_d_latent": int(d_latent),
                }
            )
            continue

        z_mean = Z.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
        z_median = np.median(Z, axis=0).astype(np.float32, copy=False)

        latent_ids_out.append(int(latent_idx))
        selected_direction_out.append(str(recs[0]["selected_direction"]))
        prototype_mean_rows.append(z_mean)
        prototype_median_rows.append(z_median)
        if args.save_top_codes:
            all_top_codes.append(Z.astype(np.float32, copy=False))
            top_code_row_counts.append(int(Z.shape[0]))

        self_vals = Z[:, int(latent_idx)].astype(np.float32, copy=False)
        self_score_stats[str(latent_idx)] = {
            "count": float(self_vals.shape[0]),
            "min": float(self_vals.min()),
            "p50": float(np.percentile(self_vals, 50)),
            "p75": float(np.percentile(self_vals, 75)),
            "p90": float(np.percentile(self_vals, 90)),
            "max": float(self_vals.max()),
            "mean": float(self_vals.mean()),
            "std": float(self_vals.std()),
        }

        per_latent_meta[str(latent_idx)] = {
            "latent_idx": int(latent_idx),
            "selected_direction": str(recs[0]["selected_direction"]),
            "tiles_csv_rows_used": int(len(recs)),
            "unique_h5_count": int(len(by_h5)),
            "raw_score_count": int(len(raw_scores)),
            "raw_score_mean": float(np.mean(np.asarray(raw_scores, dtype=np.float32))),
            "prototype_mean_l2": float(np.linalg.norm(z_mean)),
            "prototype_median_l2": float(np.linalg.norm(z_median)),
            "top_abs_dims_mean": _top_abs_dims(z_mean, int(args.topk_dims)),
            "top_abs_dims_median": _top_abs_dims(z_median, int(args.topk_dims)),
            "self_activation_stats": self_score_stats[str(latent_idx)],
            "source_examples": [
                {
                    "prototype_rank": int(rec["prototype_rank"]),
                    "case_id": rec["case_id"],
                    "slide_key": rec["slide_key"],
                    "tile_index": int(rec["tile_index"]),
                    "label": int(rec["label"]),
                    "pred": int(rec["pred"]),
                    "prob_pos": float(rec["prob_pos"]),
                    "attention": float(rec["attention"]),
                    "sae_activation": float(rec["sae_activation"]),
                    "attention_weighted_activation": float(rec["attention_weighted_activation"]),
                    "h5_path": rec["h5_path"],
                }
                for rec in recs[: min(len(recs), 5)]
            ],
        }

        if i % 10 == 0 or i == len(latent_order):
            print(f"[progress] processed {i}/{len(latent_order)} latents", flush=True)

    if not latent_ids_out:
        raise SystemExit("No prototype vectors were built successfully.")

    out_arrays: dict[str, np.ndarray] = {
        "latent_ids": np.asarray(latent_ids_out, dtype=np.int64),
        "selected_direction": np.asarray(selected_direction_out, dtype="<U16"),
        "prototype_mean": np.stack(prototype_mean_rows, axis=0).astype(np.float32, copy=False),
        "prototype_median": np.stack(prototype_median_rows, axis=0).astype(np.float32, copy=False),
    }
    if args.save_top_codes:
        out_arrays["top_code_row_counts"] = np.asarray(top_code_row_counts, dtype=np.int32)
        out_arrays["top_codes_concat"] = np.concatenate(all_top_codes, axis=0).astype(np.float32, copy=False)
    np.savez_compressed(args.out_npz, **out_arrays)

    out_meta = {
        "source_tiles_csv": str(args.tiles_csv),
        "sae_ckpt": str(args.sae_ckpt),
        "sae_cfg": str(args.sae_cfg),
        "device": args.device,
        "d_in": int(d_in),
        "d_latent": int(d_latent),
        "top_n_requested": int(args.top_n),
        "latents_requested": int(len(latent_order)),
        "latents_written": int(len(latent_ids_out)),
        "latent_ids": [int(x) for x in latent_ids_out],
        "selected_direction_by_latent": {str(lid): direction for lid, direction in zip(latent_ids_out, selected_direction_out)},
        "prototype_npz": str(args.out_npz),
        "notes": [
            "Use prototype_median or prototype_mean with scripts/sae_steer_sweep.py.",
            "prototype_target interpolates an input tile toward the selected latent's aggregate SAE code.",
            "prototype_delta adds prototype - baseline as a full-code steering direction.",
        ],
        "per_latent": per_latent_meta,
        "skipped": skipped,
    }
    args.out_json.write_text(json.dumps(out_meta, indent=2))

    print(f"[ok] wrote {args.out_npz}")
    print(f"[ok] wrote {args.out_json}")


if __name__ == "__main__":
    main()
