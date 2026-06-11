#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from export_concept_package import export_concept_package
from wsi_cf.common.io import write_json
from wsi_cf.common.paths import (
    DEFAULT_SAE_CFG,
    DEFAULT_SAE_CKPT,
    SAE_VARIANTS,
    resolve_sae_paths,
    resource_path,
)
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.concepts.task_config import (
    build_task_cohort,
    label_slug,
    resolve_task_config,
    select_split,
)
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JSON-driven SAE concept discovery from TCGA label/cohort definitions."
    )
    parser.add_argument("--task-json", type=Path, required=True, help="Task definition JSON.")
    parser.add_argument("--out-root", type=Path, default=WSI_CF_ROOT / "artifacts/concept_discovery_json")
    parser.add_argument("--device", type=str, default=None, help="Override task JSON device. Defaults to cuda:0.")
    parser.add_argument("--seed", type=int, default=None, help="Override task JSON seed. Defaults to 7.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override task JSON batch_size.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-export-concept-package",
        action="store_true",
        help="Disable concept_export package writing under each label directory.",
    )
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(str(key))
                    seen.add(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def minmax(values: list[float]) -> dict[int, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {idx: 1.0 for idx in range(len(values))}
    return {idx: float((float(value) - lo) / (hi - lo)) for idx, value in enumerate(values)}


def read_h5_features_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        features = np.asarray(handle["features"][:], dtype=np.float32)
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if features.ndim != 2:
        raise ValueError(f"{path}: expected features [N,D] or [1,N,D], got {features.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path}: expected coords [N,2] or [1,N,2], got {coords.shape}")
    if features.shape[0] != coords.shape[0]:
        raise ValueError(f"{path}: features N={features.shape[0]} but coords N={coords.shape[0]}")
    return features, coords


def maybe_subsample_tiles(features: np.ndarray, *, max_tiles: int, seed: int, slide_key: str) -> np.ndarray:
    if int(max_tiles) <= 0 or features.shape[0] <= int(max_tiles):
        return features
    digest = hashlib.md5(f"{int(seed)}::{slide_key}".encode("utf-8")).hexdigest()
    local_seed = int(digest[:8], 16)
    rng = np.random.default_rng(local_seed)
    idx = rng.choice(features.shape[0], size=int(max_tiles), replace=False)
    idx.sort()
    return features[idx]


@torch.no_grad()
def summarize_slide(
    *,
    features: np.ndarray,
    sae_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    d_latent: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tiles = int(features.shape[0])
    sum_z = np.zeros((d_latent,), dtype=np.float64)
    sum_active = np.zeros((d_latent,), dtype=np.float64)
    max_z = np.full((d_latent,), -np.inf, dtype=np.float32)
    for start in range(0, n_tiles, int(batch_size)):
        end = min(n_tiles, start + int(batch_size))
        x = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
        z = sae_encode_features(sae_model, x).detach().cpu().numpy().astype(np.float32, copy=False)
        sum_z += z.sum(axis=0, dtype=np.float64)
        sum_active += (z > 0).sum(axis=0, dtype=np.float64)
        max_z = np.maximum(max_z, z.max(axis=0))
    mean_activation = (sum_z / max(n_tiles, 1)).astype(np.float32)
    fraction_active = (sum_active / max(n_tiles, 1)).astype(np.float32)
    max_z[~np.isfinite(max_z)] = 0.0
    return mean_activation, fraction_active, max_z.astype(np.float32)


def cohen_d(class_values: np.ndarray, rest_values: np.ndarray) -> np.ndarray:
    n1 = int(class_values.shape[0])
    n2 = int(rest_values.shape[0])
    mean1 = class_values.mean(axis=0)
    mean2 = rest_values.mean(axis=0)
    var1 = class_values.var(axis=0, ddof=1) if n1 > 1 else np.zeros_like(mean1)
    var2 = rest_values.var(axis=0, ddof=1) if n2 > 1 else np.zeros_like(mean2)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    return ((mean1 - mean2) / np.maximum(pooled, 1e-8)).astype(np.float32)


def build_association_rows(
    *,
    metric_name: str,
    values: np.ndarray,
    labels: list[str],
    latent_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels_arr = np.asarray(labels)
    for class_label in sorted(set(labels)):
        mask = labels_arr == class_label
        rest = ~mask
        if int(mask.sum()) == 0 or int(rest.sum()) == 0:
            continue
        class_values = values[mask]
        rest_values = values[rest]
        mean_class = class_values.mean(axis=0)
        mean_rest = rest_values.mean(axis=0)
        diff = mean_class - mean_rest
        d = cohen_d(class_values, rest_values)
        for latent_idx, mc, mr, delta, dz in zip(latent_ids.tolist(), mean_class.tolist(), mean_rest.tolist(), diff.tolist(), d.tolist()):
            rows.append(
                {
                    "latent_idx": int(latent_idx),
                    "metric": metric_name,
                    "class_label": class_label,
                    "n_class": int(mask.sum()),
                    "n_rest": int(rest.sum()),
                    "mean_class": float(mc),
                    "mean_rest": float(mr),
                    "diff_class_minus_rest": float(delta),
                    "abs_diff": float(abs(delta)),
                    "cohen_d": float(dz),
                }
            )
    rows.sort(
        key=lambda r: (
            str(r["metric"]),
            str(r["class_label"]),
            -float(r["cohen_d"]),
            -float(r["abs_diff"]),
            int(r["latent_idx"]),
        )
    )
    return rows


def association_candidates(
    association_rows: list[dict[str, Any]],
    *,
    task: str,
    class_label: str,
    metric: str,
    min_cohen_d: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in association_rows:
        if str(row.get("metric", "")) != str(metric):
            continue
        if str(row.get("class_label", "")) != str(class_label):
            continue
        diff = float(row.get("diff_class_minus_rest", 0.0))
        cohen = float(row.get("cohen_d", 0.0))
        if diff <= 0.0 or cohen < float(min_cohen_d):
            continue
        rows.append(
            {
                "task": task,
                "class_label": class_label,
                "latent_idx": int(row["latent_idx"]),
                "metric": str(row["metric"]),
                "n_class": int(row["n_class"]),
                "n_rest": int(row["n_rest"]),
                "mean_class": float(row["mean_class"]),
                "mean_rest": float(row["mean_rest"]),
                "diff_class_minus_rest": diff,
                "abs_diff": float(row["abs_diff"]),
                "cohen_d": cohen,
            }
        )
    if not rows:
        raise ValueError(f"No positive associated latents for task={task}, class_label={class_label}, metric={metric}")
    norm_cohen = minmax([float(row["cohen_d"]) for row in rows])
    norm_abs = minmax([float(row["abs_diff"]) for row in rows])
    norm_diff = minmax([float(row["diff_class_minus_rest"]) for row in rows])
    for idx, row in enumerate(rows):
        row["association_score"] = float((norm_cohen[idx] + norm_abs[idx] + norm_diff[idx]) / 3.0)
    rows.sort(key=lambda r: (-float(r["association_score"]), -float(r["cohen_d"]), -float(r["abs_diff"]), int(r["latent_idx"])))
    for rank, row in enumerate(rows, start=1):
        row["association_rank"] = int(rank)
    return rows


def heap_push(heap: list[tuple[float, int, dict[str, Any]]], row: dict[str, Any], *, score_key: str, limit: int, counter: int) -> int:
    score = float(row[score_key])
    item = (score, int(counter), row)
    if len(heap) < int(limit):
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)
    return int(counter) + 1


def heap_to_ranked_rows(
    heap: list[tuple[float, int, dict[str, Any]]],
    *,
    rank_key: str,
    sort_key: str,
) -> list[dict[str, Any]]:
    rows = [item[2] for item in heap]
    rows.sort(key=lambda row: (-float(row[sort_key]), str(row["slide_key"]), int(row["tile_index"])))
    return [{**row, rank_key: int(rank)} for rank, row in enumerate(rows, start=1)]


def prevalence_quality_score(prevalence: float, *, target: float, sigma: float) -> float:
    p = max(float(prevalence), 1e-12)
    target = max(float(target), 1e-12)
    sigma = max(float(sigma), 1e-6)
    z = (np.log10(p) - np.log10(target)) / sigma
    return float(np.exp(-0.5 * z * z))


def _read_one_feature(path: Path, tile_index: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        feats = handle["features"]
        if feats.ndim == 3 and feats.shape[0] == 1:
            arr = np.asarray(feats[0, int(tile_index)], dtype=np.float32)
        elif feats.ndim == 2:
            arr = np.asarray(feats[int(tile_index)], dtype=np.float32)
        else:
            raise ValueError(f"{path}: unsupported features shape {tuple(feats.shape)}")
    return arr.astype(np.float32, copy=False)


def representative_feature_coherence(rows: list[dict[str, Any]], *, top_k: int) -> float:
    rows = sorted(rows, key=lambda row: int(row.get("tile_rank", 10**9)))[: max(int(top_k), 0)]
    if len(rows) < 2:
        return 0.0
    feats: list[np.ndarray] = []
    for row in rows:
        try:
            feats.append(_read_one_feature(Path(str(row["h5_path"])), int(row["tile_index"])))
        except Exception:
            continue
    if len(feats) < 2:
        return 0.0
    x = np.stack(feats, axis=0).astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    sim = x @ x.T
    tri = sim[np.triu_indices(sim.shape[0], k=1)]
    return float(np.clip(np.mean(tri), -1.0, 1.0)) if tri.size else 0.0


def load_attention_model(cfg: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module | None, str, str]:
    classifier_run_dir = Path(str(cfg.get("classifier_run_dir", ""))) if cfg.get("classifier_run_dir") else None
    classifier_ckpt = Path(str(cfg.get("classifier_ckpt", ""))) if cfg.get("classifier_ckpt") else None
    if classifier_run_dir is None and classifier_ckpt is None:
        return None, "none", "no classifier_run_dir configured"
    ckpt_path = classifier_ckpt or (classifier_run_dir / "best_model.pt" if classifier_run_dir is not None else None)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Configured attention-aware ranking but classifier checkpoint is missing: {ckpt_path}")
    return build_mil_from_checkpoint(ckpt_path, device=device), "mil", ""


def compute_attention(
    *,
    model: torch.nn.Module | None,
    backend: str,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray | None, int | None, float | None]:
    if model is None or backend == "none":
        return None, None, None
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.inference_mode():
        _, y_prob, y_hat, a_raw, _ = model(x)
        attention = F.softmax(a_raw, dim=1).detach().cpu().numpy().reshape(-1)
        pred = int(y_hat.detach().cpu().reshape(-1)[0].item())
        probs = y_prob.detach().cpu().numpy().reshape(-1)
        prob_pred = float(probs[pred]) if 0 <= pred < len(probs) else float(np.max(probs))
    return attention.astype(np.float32, copy=False), int(pred), float(prob_pred)


def representative_scan(
    *,
    slides: list[dict[str, Any]],
    candidate_latents: list[dict[str, Any]],
    sae_model: torch.nn.Module,
    d_in: int,
    device: torch.device,
    batch_size: int,
    top_tiles_per_concept: int,
    attention_model: torch.nn.Module | None,
    attention_backend: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]], dict[str, Any]]:
    latent_ids = [int(row["latent_idx"]) for row in candidate_latents]
    latent_positions = {latent: idx for idx, latent in enumerate(latent_ids)}
    activation_heaps: dict[int, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    weighted_heaps: dict[int, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    attention_support_values: dict[int, list[float]] = defaultdict(list)
    attention_peak_values: dict[int, list[float]] = defaultdict(list)
    activation_sum: dict[int, float] = defaultdict(float)
    activation_sq_sum: dict[int, float] = defaultdict(float)
    activation_positive_count: dict[int, int] = defaultdict(int)
    activation_total_count: dict[int, int] = defaultdict(int)
    heap_counter = 0
    scan_summary: dict[str, Any] = {
        "slides_seen": int(len(slides)),
        "slides_processed": 0,
        "slides_skipped": 0,
        "tiles_processed": 0,
        "attention_predictions": [],
        "skipped_slides": [],
    }
    for slide in slides:
        h5_path = Path(str(slide.get("h5_path", "")))
        try:
            features, coords = read_h5_features_coords(h5_path)
            if int(features.shape[1]) != int(d_in):
                raise ValueError(f"feature_dim_{features.shape[1]}_expected_{d_in}")
        except Exception as exc:
            scan_summary["slides_skipped"] += 1
            scan_summary["skipped_slides"].append({"slide_key": slide.get("slide_key", ""), "h5_path": str(h5_path), "reason": str(exc)})
            continue

        attention, pred, prob_pred = compute_attention(model=attention_model, backend=attention_backend, features=features, device=device)
        if attention is None:
            attention = np.ones((features.shape[0],), dtype=np.float32)
        if attention.shape[0] != features.shape[0]:
            scan_summary["slides_skipped"] += 1
            scan_summary["skipped_slides"].append(
                {"slide_key": slide.get("slide_key", ""), "h5_path": str(h5_path), "reason": f"attention_len_{attention.shape[0]}_feature_len_{features.shape[0]}"}
            )
            continue

        attn_norm = attention / max(float(np.max(attention)) if attention.size else 0.0, 1e-8)
        scan_summary["slides_processed"] += 1
        scan_summary["tiles_processed"] += int(features.shape[0])
        if pred is not None:
            scan_summary["attention_predictions"].append(
                {"case_id": slide.get("case_id", ""), "slide_key": slide.get("slide_key", ""), "label": slide.get("label", ""), "pred": int(pred), "prob_pred": float(prob_pred)}
            )

        for start in range(0, int(features.shape[0]), int(batch_size)):
            end = min(int(features.shape[0]), start + int(batch_size))
            x = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
            with torch.inference_mode():
                z = sae_encode_features(sae_model, x)
                z_sel = z[:, latent_ids].detach().cpu()
            attn_batch = attention[start:end]
            attn_norm_batch = attn_norm[start:end]
            for latent_idx in latent_ids:
                pos = latent_positions[int(latent_idx)]
                scores = z_sel[:, pos]
                scores_np = scores.numpy()
                activation_sum[int(latent_idx)] += float(np.sum(scores_np))
                activation_sq_sum[int(latent_idx)] += float(np.sum(scores_np * scores_np))
                activation_positive_count[int(latent_idx)] += int(np.count_nonzero(scores_np > 1e-6))
                activation_total_count[int(latent_idx)] += int(scores_np.size)
                k = min(int(top_tiles_per_concept), int(scores.numel()))
                if k <= 0:
                    continue
                vals, inds = torch.topk(scores, k=k, largest=True)
                for val, local_idx_t in zip(vals.tolist(), inds.tolist()):
                    tile_index = int(start) + int(local_idx_t)
                    activation = float(val)
                    attn_value = float(attn_batch[int(local_idx_t)])
                    attn_norm_value = float(attn_norm_batch[int(local_idx_t)])
                    weighted = float(activation * attn_norm_value)
                    base_row = {
                        "task": str(slide.get("task", "")),
                        "class_label": str(slide.get("label", "")),
                        "latent_idx": int(latent_idx),
                        "activation": activation,
                        "attention": attn_value if attention_model is not None else "",
                        "attention_norm": attn_norm_value if attention_model is not None else "",
                        "attention_weighted_activation": weighted if attention_model is not None else "",
                        "case_id": str(slide.get("case_id", "")),
                        "slide_key": str(slide.get("slide_key", "")),
                        "project_dir": str(slide.get("project_dir", "")),
                        "label": str(slide.get("label", "")),
                        "h5_path": str(h5_path),
                        "tile_index": int(tile_index),
                        "coord_x": int(coords[tile_index, 0]),
                        "coord_y": int(coords[tile_index, 1]),
                    }
                    heap_counter = heap_push(activation_heaps[int(latent_idx)], {**base_row, "ranking_method": "activation"}, score_key="activation", limit=int(top_tiles_per_concept), counter=heap_counter)
                    if attention_model is not None:
                        heap_counter = heap_push(weighted_heaps[int(latent_idx)], {**base_row, "ranking_method": "attention_weighted"}, score_key="attention_weighted_activation", limit=int(top_tiles_per_concept), counter=heap_counter)

    rep_rows: list[dict[str, Any]] = []
    for latent_idx in latent_ids:
        rep_rows.extend(heap_to_ranked_rows(activation_heaps[int(latent_idx)], rank_key="tile_rank", sort_key="activation"))
        if attention_model is not None:
            weighted_rows = heap_to_ranked_rows(weighted_heaps[int(latent_idx)], rank_key="tile_rank", sort_key="attention_weighted_activation")
            rep_rows.extend(weighted_rows)
            positive_weighted = [float(row["attention_weighted_activation"]) for row in weighted_rows if float(row["activation"]) > 0.0]
            peak_attn = [float(row["attention_norm"]) for row in weighted_rows if float(row["activation"]) > 0.0]
            if positive_weighted:
                attention_support_values[int(latent_idx)].append(float(np.mean(positive_weighted)))
                attention_peak_values[int(latent_idx)].append(float(np.mean(peak_attn)))

    support: dict[int, dict[str, float]] = {}
    for latent_idx in latent_ids:
        mean = activation_sum[int(latent_idx)] / max(int(activation_total_count[int(latent_idx)]), 1)
        var = activation_sq_sum[int(latent_idx)] / max(int(activation_total_count[int(latent_idx)]), 1) - mean * mean
        support[int(latent_idx)] = {
            "attention_support_raw": float(np.mean(attention_support_values.get(int(latent_idx), []))) if attention_support_values.get(int(latent_idx), []) else 0.0,
            "attention_peak_mean": float(np.mean(attention_peak_values.get(int(latent_idx), []))) if attention_peak_values.get(int(latent_idx), []) else 0.0,
            "activation_mean": float(mean),
            "activation_std": float(max(var, 0.0) ** 0.5),
            "activation_prevalence": float(activation_positive_count[int(latent_idx)] / max(int(activation_total_count[int(latent_idx)]), 1)),
        }
    return rep_rows, support, scan_summary


def prepare_associations(
    *,
    cfg: dict[str, Any],
    association_slides: list[dict[str, Any]],
    sae_model: torch.nn.Module,
    d_in: int,
    d_latent: int,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    mean_rows: list[np.ndarray] = []
    fraction_rows: list[np.ndarray] = []
    max_rows: list[np.ndarray] = []
    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, slide in enumerate(association_slides, start=1):
        h5_path = Path(str(slide["h5_path"]))
        try:
            features, _ = read_h5_features_coords(h5_path)
            if int(features.shape[1]) != int(d_in):
                raise ValueError(f"feature_dim_{features.shape[1]}_expected_{d_in}")
            features = maybe_subsample_tiles(
                features,
                max_tiles=int(cfg.get("max_tiles_per_slide", 0)),
                seed=int(seed),
                slide_key=str(slide["slide_key"]),
            )
            mean_z, frac_z, max_z = summarize_slide(features=features, sae_model=sae_model, device=device, batch_size=batch_size, d_latent=d_latent)
        except Exception as exc:
            skipped.append({**slide, "reason": str(exc)})
            continue
        mean_rows.append(mean_z)
        fraction_rows.append(frac_z)
        max_rows.append(max_z)
        processed.append({**slide, "n_tiles_used": int(features.shape[0])})
        if idx % 100 == 0:
            print(f"[progress] {idx}/{len(association_slides)} association slides scanned", file=sys.stderr)

    if len(processed) < 2 or len({row["label"] for row in processed}) < 2:
        raise RuntimeError(f"Need at least two processed labels for association; processed={Counter(row['label'] for row in processed)}, skipped={len(skipped)}")

    labels = [str(row["label"]) for row in processed]
    latent_ids = np.arange(int(d_latent), dtype=np.int64)
    arrays = {
        "mean_activation": np.stack(mean_rows, axis=0).astype(np.float32),
        "fraction": np.stack(fraction_rows, axis=0).astype(np.float32),
        "max_activation": np.stack(max_rows, axis=0).astype(np.float32),
        "latent_ids": latent_ids,
        "slide_keys": np.asarray([row["slide_key"] for row in processed]),
        "labels": np.asarray(labels),
    }
    rows: list[dict[str, Any]] = []
    rows.extend(build_association_rows(metric_name="mean_activation", values=arrays["mean_activation"], labels=labels, latent_ids=latent_ids))
    rows.extend(build_association_rows(metric_name="fraction", values=arrays["fraction"], labels=labels, latent_ids=latent_ids))
    rows.extend(build_association_rows(metric_name="prevalence", values=arrays["fraction"], labels=labels, latent_ids=latent_ids))
    rows.extend(build_association_rows(metric_name="max_activation", values=arrays["max_activation"], labels=labels, latent_ids=latent_ids))
    return rows, {"processed": processed, "skipped": skipped, "label_counts": dict(Counter(labels))}, arrays


def build_concepts_for_label(
    *,
    cfg: dict[str, Any],
    class_label: str,
    association_rows: list[dict[str, Any]],
    representative_slides: list[dict[str, Any]],
    sae_model: torch.nn.Module,
    d_in: int,
    d_latent: int,
    device: torch.device,
    task_dir: Path,
    batch_size: int,
    attention_model: torch.nn.Module | None,
    attention_backend: str,
    attention_fallback_reason: str,
    command: str,
    args_payload: dict[str, Any],
    no_export: bool,
) -> dict[str, Any]:
    task_name = str(cfg["task_name"])
    label_dir = task_dir / "labels" / label_slug(class_label)
    label_dir.mkdir(parents=True, exist_ok=True)
    candidates_all = association_candidates(
        association_rows,
        task=task_name,
        class_label=class_label,
        metric=str(cfg["metric"]),
        min_cohen_d=float(cfg["min_cohen_d"]),
    )
    candidate_pool = candidates_all[: max(int(cfg["top_concepts"]), int(cfg["candidate_latents"]))]
    for row in candidate_pool:
        row["task"] = task_name
    rep_rows, attention_support, scan_summary = representative_scan(
        slides=representative_slides,
        candidate_latents=candidate_pool,
        sae_model=sae_model,
        d_in=d_in,
        device=device,
        batch_size=batch_size,
        top_tiles_per_concept=int(cfg["top_tiles_per_concept"]),
        attention_model=attention_model,
        attention_backend=attention_backend,
    )

    effective_mode = "attention_aware" if attention_model is not None else "labels_only"
    support_raw = [attention_support[int(row["latent_idx"])]["attention_support_raw"] for row in candidate_pool]
    support_norm_map = minmax(support_raw)
    rep_by_latent_method: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rep_rows:
        rep_by_latent_method[(int(row["latent_idx"]), str(row["ranking_method"]))].append(row)
    coherence_raw_by_latent: dict[int, float] = {}
    for row in candidate_pool:
        latent_idx = int(row["latent_idx"])
        coherence_raw_by_latent[latent_idx] = representative_feature_coherence(
            rep_by_latent_method.get((latent_idx, "activation"), []),
            top_k=int(cfg["morphology_coherence_top_k"]),
        )
    prevalence_scores = [
        prevalence_quality_score(
            attention_support[int(row["latent_idx"])]["activation_prevalence"],
            target=float(cfg["morphology_target_prevalence"]),
            sigma=float(cfg["morphology_prevalence_sigma"]),
        )
        for row in candidate_pool
    ]
    coherence_norm_map = minmax([coherence_raw_by_latent[int(row["latent_idx"])] for row in candidate_pool])
    activation_mean_norm_map = minmax([attention_support[int(row["latent_idx"])]["activation_mean"] for row in candidate_pool])

    concept_cards: list[dict[str, Any]] = []
    for idx, row in enumerate(candidate_pool):
        latent_idx = int(row["latent_idx"])
        attention_support_score = float(support_norm_map[idx]) if effective_mode == "attention_aware" else 0.0
        activation_reps = rep_by_latent_method.get((latent_idx, "activation"), [])
        attention_reps = rep_by_latent_method.get((latent_idx, "attention_weighted"), [])
        top_activation = activation_reps[0] if activation_reps else {}
        top_attention = attention_reps[0] if attention_reps else {}
        activation_prevalence = float(attention_support[latent_idx]["activation_prevalence"])
        activation_mean = float(attention_support[latent_idx]["activation_mean"])
        activation_std = float(attention_support[latent_idx]["activation_std"])
        prevalence_score = float(prevalence_scores[idx])
        coherence_raw = float(coherence_raw_by_latent[latent_idx])
        coherence_score = float(coherence_norm_map[idx])
        activation_mean_score = float(activation_mean_norm_map[idx])
        if effective_mode == "attention_aware":
            final_score = (
                0.45 * float(row["association_score"])
                + 0.20 * attention_support_score
                + 0.15 * prevalence_score
                + 0.15 * coherence_score
                + 0.05 * activation_mean_score
            )
        else:
            final_score = (
                0.55 * float(row["association_score"])
                + 0.20 * prevalence_score
                + 0.20 * coherence_score
                + 0.05 * activation_mean_score
            )
        concept_cards.append(
            {
                **row,
                "concept_quality_mode": str(cfg["concept_quality_mode"]),
                "activation_mean": activation_mean,
                "activation_std": activation_std,
                "activation_prevalence": activation_prevalence,
                "prevalence_quality_score": prevalence_score,
                "prototype_coherence_raw": coherence_raw,
                "prototype_coherence_score": coherence_score,
                "activation_mean_score": activation_mean_score,
                "attention_support_raw": float(attention_support[latent_idx]["attention_support_raw"]),
                "attention_peak_mean": float(attention_support[latent_idx]["attention_peak_mean"]),
                "attention_support_score": attention_support_score,
                "final_score": float(final_score),
                "steering_direction": f"toward_{class_label}",
                "top_activation_slide_key": top_activation.get("slide_key", ""),
                "top_activation_tile_index": top_activation.get("tile_index", ""),
                "top_activation_coord_x": top_activation.get("coord_x", ""),
                "top_activation_coord_y": top_activation.get("coord_y", ""),
                "top_activation": top_activation.get("activation", ""),
                "top_attention_slide_key": top_attention.get("slide_key", ""),
                "top_attention_tile_index": top_attention.get("tile_index", ""),
                "top_attention_coord_x": top_attention.get("coord_x", ""),
                "top_attention_coord_y": top_attention.get("coord_y", ""),
                "top_attention_weighted_activation": top_attention.get("attention_weighted_activation", ""),
            }
        )
    concept_cards.sort(key=lambda r: (-float(r["final_score"]), -float(r["association_score"]), int(r["latent_idx"])))
    selected = concept_cards[: int(cfg["top_concepts"])]
    for rank, row in enumerate(selected, start=1):
        row["concept_rank"] = int(rank)

    selected_latents = {int(row["latent_idx"]) for row in selected}
    rep_rows = [row for row in rep_rows if int(row["latent_idx"]) in selected_latents]
    rep_rows.sort(key=lambda r: (int(r["latent_idx"]), str(r["ranking_method"]), int(r["tile_rank"])))

    concept_fields = [
        "concept_rank", "task", "class_label", "latent_idx", "metric", "association_rank", "association_score",
        "cohen_d", "abs_diff", "diff_class_minus_rest", "mean_class", "mean_rest", "n_class", "n_rest",
        "concept_quality_mode", "activation_mean", "activation_std", "activation_prevalence", "prevalence_quality_score",
        "prototype_coherence_raw", "prototype_coherence_score", "activation_mean_score", "attention_support_raw",
        "attention_peak_mean", "attention_support_score", "final_score", "steering_direction", "top_activation_slide_key",
        "top_activation_tile_index", "top_activation_coord_x", "top_activation_coord_y", "top_activation",
        "top_attention_slide_key", "top_attention_tile_index", "top_attention_coord_x", "top_attention_coord_y",
        "top_attention_weighted_activation",
    ]
    rep_fields = [
        "task", "class_label", "latent_idx", "ranking_method", "tile_rank", "activation", "attention", "attention_norm",
        "attention_weighted_activation", "case_id", "slide_key", "project_dir", "label", "h5_path", "tile_index", "coord_x", "coord_y",
    ]
    write_csv(label_dir / "concept_cards.csv", selected, concept_fields)
    write_csv(label_dir / "representative_tiles.csv", rep_rows, rep_fields)
    write_json(label_dir / "selected_concepts.json", {"task": task_name, "class_label": class_label, "mode": effective_mode, "concept_quality_mode": str(cfg["concept_quality_mode"]), "concepts": selected})
    summary = {
        "args": {**args_payload, "task": task_name, "class_label": class_label},
        "command": command,
        "requested_mode": str(cfg["ranking_mode"]),
        "effective_mode": effective_mode,
        "concept_quality_mode": str(cfg["concept_quality_mode"]),
        "morphology_target_prevalence": float(cfg["morphology_target_prevalence"]),
        "morphology_prevalence_sigma": float(cfg["morphology_prevalence_sigma"]),
        "morphology_coherence_top_k": int(cfg["morphology_coherence_top_k"]),
        "attention_backend": attention_backend,
        "attention_fallback_reason": attention_fallback_reason,
        "association_weight": float(cfg["association_weight"]),
        "attention_weight": float(cfg["attention_weight"]),
        "sae_d_in": int(d_in),
        "sae_d_latent": int(d_latent),
        "association_task": task_name,
        "task_dir": str(task_dir),
        "slides_source": str(task_dir / "cohort_slides.csv"),
        "slides_for_class": int(len(representative_slides)),
        "candidate_latents_total": int(len(candidates_all)),
        "candidate_latents_scanned": int(len(candidate_pool)),
        "concept_cards": int(len(selected)),
        "representative_tiles": int(len(rep_rows)),
        "scan": scan_summary,
        "outputs": {
            "concept_cards_csv": str(label_dir / "concept_cards.csv"),
            "representative_tiles_csv": str(label_dir / "representative_tiles.csv"),
            "selected_concepts_json": str(label_dir / "selected_concepts.json"),
            "summary_json": str(label_dir / "summary.json"),
        },
    }
    write_json(label_dir / "summary.json", summary)
    if not no_export:
        export_args = argparse.Namespace(
            concept_dir=label_dir,
            out_dir=label_dir / "concept_export",
            target_magnification=float(cfg["concept_export_target_magnification"]),
            tile_size_px=int(cfg["concept_export_tile_size_px"]),
            coord_space="level0_h5_coords",
            feature_name="UNI2",
        )
        export_outputs = export_concept_package(export_args)
        summary["outputs"]["concept_export_dir"] = str(label_dir / "concept_export")
        summary["outputs"]["concept_export"] = export_outputs
        write_json(label_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = resolve_task_config(args.task_json)
    if args.device is not None:
        cfg["device"] = str(args.device)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    cfg.setdefault("device", "cuda:0")
    cfg.setdefault("seed", 7)

    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))
    task_dir = resource_path(args.out_root) / str(cfg["task_name"])
    expected = [task_dir / "task_summary.json"] + [
        task_dir / "labels" / label_slug(label) / "selected_concepts.json" for label in cfg["concept_labels"]
    ]
    if bool(args.skip_existing) and all(path.exists() for path in expected):
        print(json.dumps({"skipped": True, "task_dir": str(task_dir), "outputs": [str(path) for path in expected]}, indent=2))
        return

    task_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:])))
    args_payload = {k: str(v) if isinstance(v, Path) else v for k, v in {**vars(args), **cfg}.items()}

    cohort, skipped, cohort_summary = build_task_cohort(cfg)
    association_slides = select_split(cohort, str(cfg["association_split"]))
    if len({row["label"] for row in association_slides}) < 2:
        raise RuntimeError(f"Association split {cfg['association_split']!r} must contain at least two labels; counts={Counter(row['label'] for row in association_slides)}")
    write_csv(task_dir / "cohort_slides.csv", cohort, ["task", "case_id", "slide_key", "sample_id", "sample_code", "project_dir", "label", "label_name", "raw_label", "split", "h5_path"])
    write_csv(task_dir / "skipped_slides.csv", skipped)

    sae_ckpt, sae_cfg = resolve_sae_paths(str(cfg["sae_variant"]), cfg.get("sae_ckpt"), cfg.get("sae_cfg"))
    cfg["sae_ckpt"] = str(sae_ckpt)
    cfg["sae_cfg"] = str(sae_cfg)
    write_json(task_dir / "task_config.resolved.json", cfg)
    args_payload = {k: str(v) if isinstance(v, Path) else v for k, v in {**vars(args), **cfg}.items()}
    sae_model, d_in, d_latent = load_sae_from_config(sae_ckpt, sae_cfg, device=str(device))
    sae_model.eval()

    attention_model: torch.nn.Module | None = None
    attention_backend = "none"
    attention_fallback_reason = "no classifier_run_dir configured"
    if str(cfg.get("ranking_mode", "")) == "attention_aware_optional" and str(cfg.get("classifier_run_dir", "")):
        attention_model, attention_backend, attention_fallback_reason = load_attention_model(cfg, device)

    association_rows, association_summary, arrays = prepare_associations(
        cfg=cfg,
        association_slides=association_slides,
        sae_model=sae_model,
        d_in=int(d_in),
        d_latent=int(d_latent),
        device=device,
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg["seed"]),
    )
    write_csv(
        task_dir / "latent_label_associations.csv",
        association_rows,
        ["latent_idx", "metric", "class_label", "n_class", "n_rest", "mean_class", "mean_rest", "diff_class_minus_rest", "abs_diff", "cohen_d"],
    )
    write_csv(task_dir / "association_processed_slides.csv", association_summary["processed"])
    write_csv(task_dir / "association_skipped_slides.csv", association_summary["skipped"])
    np.savez_compressed(
        task_dir / "slide_sae_summary.npz",
        latent_ids=arrays["latent_ids"],
        slide_keys=arrays["slide_keys"],
        labels=arrays["labels"],
        mean_activation=arrays["mean_activation"],
        fraction=arrays["fraction"],
        max_activation=arrays["max_activation"],
    )

    label_summaries: dict[str, Any] = {}
    for class_label in cfg["concept_labels"]:
        representative_slides = [
            row for row in select_split(cohort, str(cfg["representative_split"])) if str(row["label"]) == str(class_label)
        ]
        if not representative_slides:
            raise RuntimeError(f"No representative slides for label={class_label!r} split={cfg['representative_split']!r}")
        label_summaries[str(class_label)] = build_concepts_for_label(
            cfg=cfg,
            class_label=str(class_label),
            association_rows=association_rows,
            representative_slides=representative_slides,
            sae_model=sae_model,
            d_in=int(d_in),
            d_latent=int(d_latent),
            device=device,
            task_dir=task_dir,
            batch_size=int(cfg["batch_size"]),
            attention_model=attention_model,
            attention_backend=attention_backend,
            attention_fallback_reason=attention_fallback_reason,
            command=command,
            args_payload=args_payload,
            no_export=bool(args.no_export_concept_package),
        )

    task_summary = {
        "task_name": str(cfg["task_name"]),
        "command": command,
        "args": args_payload,
        "cohort": cohort_summary,
        "association": {
            "split": str(cfg["association_split"]),
            "processed_label_counts": association_summary["label_counts"],
            "n_processed": int(len(association_summary["processed"])),
            "n_skipped": int(len(association_summary["skipped"])),
        },
        "representative_split": str(cfg["representative_split"]),
        "sae_d_in": int(d_in),
        "sae_d_latent": int(d_latent),
        "attention_backend": attention_backend,
        "attention_fallback_reason": attention_fallback_reason,
        "labels": {
            label: {
                "concept_cards": int(summary["concept_cards"]),
                "representative_tiles": int(summary["representative_tiles"]),
                "effective_mode": str(summary["effective_mode"]),
                "output_dir": str(task_dir / "labels" / label_slug(label)),
            }
            for label, summary in label_summaries.items()
        },
        "outputs": {
            "task_config_resolved": str(task_dir / "task_config.resolved.json"),
            "cohort_slides": str(task_dir / "cohort_slides.csv"),
            "skipped_slides": str(task_dir / "skipped_slides.csv"),
            "latent_label_associations": str(task_dir / "latent_label_associations.csv"),
            "slide_sae_summary_npz": str(task_dir / "slide_sae_summary.npz"),
            "task_summary": str(task_dir / "task_summary.json"),
        },
    }
    write_json(task_dir / "task_summary.json", task_summary)
    print(json.dumps(task_summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
