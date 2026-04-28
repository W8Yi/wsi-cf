from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sae import load_sae_from_config, sae_encode_features


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build full SAE latent prototype vectors from pass2 top-tile mining metadata. "
            "For each selected latent, load the top tiles (h5_path + tile_idx), encode full SAE codes, "
            "and aggregate them (mean/median) into prototype vectors."
        )
    )
    ap.add_argument("--pass2-json", type=Path, required=True, help="Pass2 JSON with `top_tiles` records.")
    ap.add_argument("--sae-ckpt", type=Path, required=True, help="SAE checkpoint used for encoding.")
    ap.add_argument("--sae-cfg", type=Path, required=True, help="SAE config JSON.")
    ap.add_argument("--out-npz", type=Path, required=True, help="Output NPZ containing prototype arrays.")
    ap.add_argument("--out-json", type=Path, required=True, help="Output JSON metadata/stats.")
    ap.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Use first N top tiles per latent (0 = use all listed in pass2).",
    )
    ap.add_argument("--latent-limit", type=int, default=0, help="Optional cap on number of selected latents.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512, help="SAE encoding batch size for loaded UNI features.")
    ap.add_argument(
        "--save-top-codes",
        action="store_true",
        help="Also save all top-tile SAE codes (can be large). Default stores prototypes only.",
    )
    ap.add_argument(
        "--topk-dims",
        type=int,
        default=16,
        help="Store top abs dims of each prototype in metadata for quick inspection.",
    )
    return ap


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _sorted_latent_order(payload: dict[str, Any]) -> list[int]:
    top_tiles = payload.get("top_tiles", {})
    selected_latents = payload.get("selected_latents", [])
    if isinstance(selected_latents, list) and selected_latents:
        ordered = [int(x) for x in selected_latents if str(int(x)) in top_tiles]
        seen = set(ordered)
        ordered += sorted(int(k) for k in top_tiles.keys() if int(k) not in seen)
        return ordered
    return sorted(int(k) for k in top_tiles.keys())


def _load_h5_feature_rows(h5_path: str, idx: np.ndarray, h5py_mod: Any) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    idx_sorted = np.sort(idx)
    with h5py_mod.File(h5_path, "r") as f:
        if "features" not in f:
            raise KeyError(f"{h5_path}: missing dataset 'features'")
        ds = f["features"]
        shape = ds.shape
        if len(shape) == 2:
            n = int(shape[0])
            if idx_sorted[-1] >= n:
                raise IndexError(f"{h5_path}: tile_idx out of bounds (max {idx_sorted[-1]} >= N={n})")
            x = ds[idx_sorted]
        elif len(shape) == 3 and int(shape[0]) == 1:
            n = int(shape[1])
            if idx_sorted[-1] >= n:
                raise IndexError(f"{h5_path}: tile_idx out of bounds (max {idx_sorted[-1]} >= N={n})")
            x = ds[0, idx_sorted]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {shape}")
    return np.asarray(x, dtype=np.float32)


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


def main() -> None:
    args = _build_argparser().parse_args()

    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"h5py is required to read UNI feature H5 files: {exc}")

    payload = _load_json(args.pass2_json)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected dict root in {args.pass2_json}, got {type(payload).__name__}")
    if "top_tiles" not in payload or not isinstance(payload["top_tiles"], dict):
        raise SystemExit(f"{args.pass2_json} is missing dict `top_tiles`.")
    top_tiles = payload["top_tiles"]

    latent_order = _sorted_latent_order(payload)
    if args.latent_limit and args.latent_limit > 0:
        latent_order = latent_order[: int(args.latent_limit)]
    if not latent_order:
        raise SystemExit("No latents found in pass2 JSON.")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    print("[setup] Loading SAE...")
    sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=args.device)

    latent_ids_out: list[int] = []
    proto_mean_rows: list[np.ndarray] = []
    proto_median_rows: list[np.ndarray] = []
    self_score_stats: dict[str, dict[str, float]] = {}
    per_latent_meta: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    all_top_codes: list[np.ndarray] = []
    top_code_row_counts: list[int] = []

    for i, latent_idx in enumerate(latent_order, start=1):
        recs = top_tiles.get(str(latent_idx), [])
        if not isinstance(recs, list) or not recs:
            skipped.append({"latent_idx": int(latent_idx), "reason": "missing_or_empty_top_tiles"})
            continue
        if args.top_n and args.top_n > 0:
            recs = recs[: int(args.top_n)]

        by_h5: dict[str, list[int]] = {}
        raw_scores: list[float] = []
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            h5_path = rec.get("h5_path")
            tile_idx = rec.get("tile_idx")
            if not isinstance(h5_path, str):
                continue
            try:
                tile_idx_i = int(tile_idx)
            except Exception:
                continue
            by_h5.setdefault(h5_path, []).append(tile_idx_i)
            if "score" in rec:
                try:
                    raw_scores.append(float(rec["score"]))
                except Exception:
                    pass

        if not by_h5:
            skipped.append({"latent_idx": int(latent_idx), "reason": "no_valid_records"})
            continue

        x_chunks: list[np.ndarray] = []
        tile_count = 0
        load_errors: list[str] = []
        for h5_path, idxs in by_h5.items():
            idx_arr = np.asarray(idxs, dtype=np.int64)
            try:
                x = _load_h5_feature_rows(h5_path, idx_arr, h5py_mod=h5py)
                x_chunks.append(x)
                tile_count += int(x.shape[0])
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
        proto_mean_rows.append(z_mean)
        proto_median_rows.append(z_median)
        if args.save_top_codes:
            all_top_codes.append(Z.astype(np.float32, copy=False))
            top_code_row_counts.append(int(Z.shape[0]))

        # Sanity-check against the selected latent's encoded activation on these tiles.
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
            "top_tiles_used": int(Z.shape[0]),
            "unique_h5_count": int(len(by_h5)),
            "raw_score_count": int(len(raw_scores)),
            "raw_score_mean": None if not raw_scores else float(np.mean(np.asarray(raw_scores, dtype=np.float32))),
            "prototype_mean_l2": float(np.linalg.norm(z_mean)),
            "prototype_median_l2": float(np.linalg.norm(z_median)),
            "self_activation_stats_encoded": self_score_stats[str(latent_idx)],
            "prototype_mean_top_abs_dims": _top_abs_dims(z_mean, int(args.topk_dims)),
            "prototype_median_top_abs_dims": _top_abs_dims(z_median, int(args.topk_dims)),
            "load_error_count": int(len(load_errors)),
            "load_error_examples": load_errors[:3],
        }

        if i % 10 == 0 or i == len(latent_order):
            print(f"  processed {i}/{len(latent_order)} latents | kept {len(latent_ids_out)} prototypes")

    if not proto_mean_rows:
        raise SystemExit("No prototypes were built (all latents failed).")

    latent_ids_arr = np.asarray(latent_ids_out, dtype=np.int64)
    prototype_mean = np.stack(proto_mean_rows, axis=0).astype(np.float32, copy=False)
    prototype_median = np.stack(proto_median_rows, axis=0).astype(np.float32, copy=False)

    npz_payload: dict[str, Any] = {
        "latent_ids": latent_ids_arr,
        "prototype_mean": prototype_mean,
        "prototype_median": prototype_median,
    }
    if args.save_top_codes:
        # Flatten per-latent arrays to a single matrix with row counts for reconstruction.
        z_all = np.concatenate(all_top_codes, axis=0).astype(np.float32, copy=False) if all_top_codes else np.empty((0, d_latent), dtype=np.float32)
        npz_payload["top_codes_all"] = z_all
        npz_payload["top_codes_row_counts"] = np.asarray(top_code_row_counts, dtype=np.int32)

    np.savez_compressed(args.out_npz, **npz_payload)

    out_meta = {
        "source_pass2_json": str(args.pass2_json),
        "source_sae_ckpt": str(args.sae_ckpt),
        "source_sae_cfg": str(args.sae_cfg),
        "top_n_requested": int(args.top_n),
        "latent_limit_requested": int(args.latent_limit),
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "d_in": int(d_in),
        "d_latent": int(d_latent),
        "latents_requested": len(latent_order),
        "latents_built": int(len(latent_ids_out)),
        "skipped": skipped,
        "npz_fields": sorted(npz_payload.keys()),
        "usage_notes": [
            "prototype_mean/prototype_median are full SAE latent vectors aggregated from top tiles.",
            "Use them with sae_steer_sweep prototype_target mode to steer toward contact-sheet-like SAE codes.",
            "prototype_median is often more robust than prototype_mean to outlier tiles.",
        ],
        "per_latent": per_latent_meta,
    }
    args.out_json.write_text(json.dumps(out_meta, indent=2))

    print(f"Wrote prototypes NPZ:  {args.out_npz}")
    print(f"Wrote prototypes meta: {args.out_json}")
    print(f"Built prototypes for {len(latent_ids_out)} latents (skipped {len(skipped)})")


if __name__ == "__main__":
    main()
