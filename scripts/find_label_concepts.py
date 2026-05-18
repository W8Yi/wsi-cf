#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import shlex
import sys
from collections import defaultdict
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
    DEFAULT_HNSCC_CLAM_CKPT,
    DEFAULT_HNSCC_MIL_CKPT,
    DEFAULT_SAE_CFG,
    DEFAULT_SAE_CKPT,
    DEFAULT_SAE_VARIANT,
    SAE_VARIANTS,
    resolve_sae_paths,
)
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, run_mil_attention
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features


SUPPORTED_ATTENTION_TASKS = {"hnsc_hpv", "hnscc_hpv"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find label-relevant SAE concepts and representative tiles. "
            "Supports association-only concept cards and optional classifier-attention evidence."
        )
    )
    parser.add_argument("--task", type=str, default="hnsc_hpv")
    parser.add_argument("--class-label", type=str, default="HPV+")
    parser.add_argument("--association-root", type=Path, default=WSI_CF_ROOT / "artifacts/concept_label_associations_all")
    parser.add_argument(
        "--association-task",
        type=str,
        default="",
        help="Association artifact task name. Defaults to --task; useful when classifier task names differ from association artifact names.",
    )
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/label_concepts")
    parser.add_argument("--mode", type=str, default="labels_only", choices=["labels_only", "attention_aware"])
    parser.add_argument(
        "--concept-quality-mode",
        type=str,
        default="association",
        choices=["association", "morphology"],
        help="morphology adds SAE prevalence/coherence quality signals to the final concept ranking.",
    )
    parser.add_argument("--backend", type=str, default="mil", choices=["mil", "clam"])
    parser.add_argument("--metric", type=str, default="fraction")
    parser.add_argument("--top-concepts", type=int, default=20)
    parser.add_argument("--candidate-latents", type=int, default=100)
    parser.add_argument("--top-tiles-per-concept", type=int, default=25)
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--association-weight", type=float, default=0.7)
    parser.add_argument("--attention-weight", type=float, default=0.3)
    parser.add_argument("--morphology-target-prevalence", type=float, default=0.03)
    parser.add_argument("--morphology-prevalence-sigma", type=float, default=0.75)
    parser.add_argument("--morphology-coherence-top-k", type=int, default=10)
    parser.add_argument("--min-cohen-d", type=float, default=0.0)
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--sae-ckpt", type=Path, default=None, help=f"Explicit SAE checkpoint override. Defaults to --sae-variant ({DEFAULT_SAE_CKPT}).")
    parser.add_argument("--sae-cfg", type=Path, default=None, help=f"Explicit SAE config override. Defaults to --sae-variant ({DEFAULT_SAE_CFG}).")
    parser.add_argument("--mil-ckpt", type=Path, default=DEFAULT_HNSCC_MIL_CKPT)
    parser.add_argument("--clam-ckpt", type=Path, default=DEFAULT_HNSCC_CLAM_CKPT)
    parser.add_argument(
        "--classifier-run-dir",
        type=Path,
        default=None,
        help="Optional trained classifier bundle. When set, uses task_manifest.csv and best_model.pt from this run.",
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint override used with --classifier-run-dir or generic MIL attention.",
    )
    parser.add_argument(
        "--slides-csv",
        type=Path,
        default=None,
        help="Optional slide manifest override. Defaults to classifier task_manifest.csv when --classifier-run-dir is set.",
    )
    parser.add_argument("--slide-label-column", type=str, default="label_name")
    parser.add_argument("--attn-class", type=str, default="pred", choices=["pred", "pos", "neg"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-export-concept-package",
        action="store_true",
        help="Disable the portable concept_export package. By default every concept discovery run writes one.",
    )
    parser.add_argument(
        "--concept-export-dir",
        type=Path,
        default=None,
        help="Optional output directory for the portable package. Defaults to <concept output dir>/concept_export.",
    )
    parser.add_argument("--concept-export-target-magnification", type=float, default=20.0)
    parser.add_argument("--concept-export-tile-size-px", type=int, default=256)
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_export_concept_package(args: argparse.Namespace, concept_dir: Path) -> dict[str, str]:
    if bool(args.no_export_concept_package):
        return {}
    export_args = argparse.Namespace(
        concept_dir=concept_dir,
        out_dir=args.concept_export_dir,
        target_magnification=float(args.concept_export_target_magnification),
        tile_size_px=int(args.concept_export_tile_size_px),
        coord_space="level0_h5_coords",
        feature_name="UNI2",
    )
    return export_concept_package(export_args)


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


def load_association_candidates(
    *,
    task_dir: Path,
    task: str,
    class_label: str,
    metric: str,
    min_cohen_d: float,
) -> list[dict[str, Any]]:
    assoc_path = task_dir / "latent_label_associations.csv"
    if not assoc_path.exists():
        raise FileNotFoundError(f"Missing association file: {assoc_path}")
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(assoc_path):
        if str(row.get("metric", "")) != str(metric):
            continue
        if str(row.get("class_label", "")) != str(class_label):
            continue
        diff = float(row.get("diff_class_minus_rest", 0.0))
        cohen_d = float(row.get("cohen_d", 0.0))
        if diff <= 0.0 or cohen_d < float(min_cohen_d):
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
                "cohen_d": cohen_d,
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


def load_task_slides(task_dir: Path, class_label: str, max_slides: int) -> list[dict[str, str]]:
    cohort_path = task_dir / "cohort_slides.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Missing cohort slides file: {cohort_path}")
    rows = [row for row in read_csv_rows(cohort_path) if str(row.get("label", "")) == str(class_label)]
    rows.sort(key=lambda r: (str(r.get("project_dir", "")), str(r.get("case_id", "")), str(r.get("slide_key", ""))))
    if int(max_slides) > 0:
        rows = rows[: int(max_slides)]
    return rows


def load_manifest_slides(manifest_path: Path, class_label: str, max_slides: int, *, label_column: str) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing slide manifest: {manifest_path}")
    rows: list[dict[str, str]] = []
    for row in read_csv_rows(manifest_path):
        label = str(row.get(label_column, row.get("label_name", row.get("label", ""))))
        if label != str(class_label):
            continue
        rows.append(
            {
                "case_id": str(row.get("case_id", "")),
                "slide_key": str(row.get("slide_key", "")),
                "project_dir": str(row.get("project_dir", "")),
                "label": label,
                "label_id": str(row.get("label_id", "")),
                "h5_path": str(row.get("h5_path", "")),
            }
        )
    rows.sort(key=lambda r: (str(r.get("project_dir", "")), str(r.get("case_id", "")), str(r.get("slide_key", ""))))
    if int(max_slides) > 0:
        rows = rows[: int(max_slides)]
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
    """Bell-shaped score that favors latents active in a meaningful but not ubiquitous tile fraction."""
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
    """Mean pairwise cosine similarity among representative UNI features."""
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
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-8)
    sim = x @ x.T
    tri = sim[np.triu_indices(sim.shape[0], k=1)]
    if tri.size == 0:
        return 0.0
    return float(np.clip(np.mean(tri), -1.0, 1.0))


def load_clam_mb(ckpt_path: Path, device: torch.device):
    from wsi_cf.models.clam import CLAM_MB

    model = CLAM_MB(gate=True, size_arg="small", n_classes=2, embed_dim=1536)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_clam_attention(model: torch.nn.Module, features: np.ndarray, *, device: torch.device, attn_class: str) -> tuple[np.ndarray, int, float]:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    logits, y_prob, y_hat, a_raw, _ = model(x)
    attn_all = F.softmax(a_raw, dim=1)
    pred = int(y_hat.item())
    prob_pos = float(y_prob[0, 1].item())
    if attn_class == "pred":
        row = pred
    elif attn_class == "pos":
        row = 1
    else:
        row = 0
    return attn_all[row].detach().cpu().numpy().astype(np.float32).reshape(-1), pred, prob_pos


def load_attention_model(
    *,
    task: str,
    backend: str,
    device: torch.device,
    mil_ckpt: Path,
    clam_ckpt: Path,
    classifier_run_dir: Path | None = None,
    classifier_ckpt: Path | None = None,
) -> tuple[torch.nn.Module | None, str, str]:
    if classifier_run_dir is not None:
        ckpt_path = classifier_ckpt or classifier_run_dir / "best_model.pt"
        if not ckpt_path.exists():
            return None, "none", f"missing classifier checkpoint: {ckpt_path}"
        return build_mil_from_checkpoint(ckpt_path, device=device), "mil", ""
    if classifier_ckpt is not None:
        if not classifier_ckpt.exists():
            return None, "none", f"missing classifier checkpoint: {classifier_ckpt}"
        return build_mil_from_checkpoint(classifier_ckpt, device=device), "mil", ""
    if task not in SUPPORTED_ATTENTION_TASKS:
        return None, "none", f"attention-aware mode currently supports {sorted(SUPPORTED_ATTENTION_TASKS)}, got {task}"
    if backend == "mil":
        if not mil_ckpt.exists():
            return None, "none", f"missing MIL checkpoint: {mil_ckpt}"
        return build_mil_from_checkpoint(mil_ckpt, device=device), "mil", ""
    if backend == "clam":
        if not clam_ckpt.exists():
            return None, "none", f"missing CLAM checkpoint: {clam_ckpt}"
        return load_clam_mb(clam_ckpt, device=device), "clam", ""
    return None, "none", f"unsupported backend: {backend}"


def compute_attention(
    *,
    model: torch.nn.Module | None,
    backend: str,
    features: np.ndarray,
    device: torch.device,
    attn_class: str,
) -> tuple[np.ndarray | None, int | None, float | None]:
    if model is None or backend == "none":
        return None, None, None
    if backend == "mil":
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        with torch.inference_mode():
            _, y_prob, y_hat, a_raw, _ = model(x)
            attention = F.softmax(a_raw, dim=1).detach().cpu().numpy().reshape(-1)
            pred = int(y_hat.detach().cpu().reshape(-1)[0].item())
            probs = y_prob.detach().cpu().numpy().reshape(-1)
            prob_pred = float(probs[pred]) if 0 <= pred < len(probs) else float(np.max(probs))
        return attention.astype(np.float32, copy=False), int(pred), float(prob_pred)
    attention, pred, prob_pos = run_clam_attention(model, features, device=device, attn_class=attn_class)
    return attention.astype(np.float32, copy=False), int(pred), float(prob_pos)


def representative_scan(
    *,
    slides: list[dict[str, str]],
    candidate_latents: list[dict[str, Any]],
    sae_model: torch.nn.Module,
    d_in: int,
    device: torch.device,
    batch_size: int,
    top_tiles_per_concept: int,
    attention_model: torch.nn.Module | None,
    attention_backend: str,
    attn_class: str,
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
        if not h5_path.exists():
            scan_summary["slides_skipped"] += 1
            scan_summary["skipped_slides"].append({"slide_key": slide.get("slide_key", ""), "h5_path": str(h5_path), "reason": "missing_h5"})
            continue
        try:
            features, coords = read_h5_features_coords(h5_path)
        except Exception as exc:
            scan_summary["slides_skipped"] += 1
            scan_summary["skipped_slides"].append({"slide_key": slide.get("slide_key", ""), "h5_path": str(h5_path), "reason": str(exc)})
            continue
        if int(features.shape[1]) != int(d_in):
            scan_summary["slides_skipped"] += 1
            scan_summary["skipped_slides"].append(
                {
                    "slide_key": slide.get("slide_key", ""),
                    "h5_path": str(h5_path),
                    "reason": f"feature_dim_{features.shape[1]}_expected_{d_in}",
                }
            )
            continue

        attention, pred, prob_pos = compute_attention(
            model=attention_model,
            backend=attention_backend,
            features=features,
            device=device,
            attn_class=attn_class,
        )
        if attention is None:
            attention = np.ones((features.shape[0],), dtype=np.float32)
            pred = None
            prob_pos = None
        if attention.shape[0] != features.shape[0]:
            scan_summary["slides_skipped"] += 1
            scan_summary["skipped_slides"].append(
                {
                    "slide_key": slide.get("slide_key", ""),
                    "h5_path": str(h5_path),
                    "reason": f"attention_len_{attention.shape[0]}_feature_len_{features.shape[0]}",
                }
            )
            continue

        attn_max = float(np.max(attention)) if attention.size else 0.0
        attn_norm = attention / max(attn_max, 1e-8)
        scan_summary["slides_processed"] += 1
        scan_summary["tiles_processed"] += int(features.shape[0])
        if pred is not None:
            scan_summary["attention_predictions"].append(
                {
                    "case_id": slide.get("case_id", ""),
                    "slide_key": slide.get("slide_key", ""),
                    "label": slide.get("label", ""),
                    "pred": int(pred),
                    "prob_pred": float(prob_pos),
                }
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
                    heap_counter = heap_push(
                        activation_heaps[int(latent_idx)],
                        {**base_row, "ranking_method": "activation"},
                        score_key="activation",
                        limit=int(top_tiles_per_concept),
                        counter=heap_counter,
                    )
                    if attention_model is not None:
                        heap_counter = heap_push(
                            weighted_heaps[int(latent_idx)],
                            {**base_row, "ranking_method": "attention_weighted"},
                            score_key="attention_weighted_activation",
                            limit=int(top_tiles_per_concept),
                            counter=heap_counter,
                        )

    rep_rows: list[dict[str, Any]] = []
    for latent_idx in latent_ids:
        rep_rows.extend(heap_to_ranked_rows(activation_heaps[int(latent_idx)], rank_key="tile_rank", sort_key="activation"))
        if attention_model is not None:
            weighted_rows = heap_to_ranked_rows(
                weighted_heaps[int(latent_idx)],
                rank_key="tile_rank",
                sort_key="attention_weighted_activation",
            )
            rep_rows.extend(weighted_rows)
            positive_weighted = [float(row["attention_weighted_activation"]) for row in weighted_rows if float(row["activation"]) > 0.0]
            peak_attn = [float(row["attention_norm"]) for row in weighted_rows if float(row["activation"]) > 0.0]
            if positive_weighted:
                attention_support_values[int(latent_idx)].append(float(np.mean(positive_weighted)))
                attention_peak_values[int(latent_idx)].append(float(np.mean(peak_attn)))

    support: dict[int, dict[str, float]] = {}
    for latent_idx in latent_ids:
        values = attention_support_values.get(int(latent_idx), [])
        peaks = attention_peak_values.get(int(latent_idx), [])
        support[int(latent_idx)] = {
            "attention_support_raw": float(np.mean(values)) if values else 0.0,
            "attention_peak_mean": float(np.mean(peaks)) if peaks else 0.0,
            "activation_mean": float(activation_sum[int(latent_idx)] / max(int(activation_total_count[int(latent_idx)]), 1)),
            "activation_std": float(
                max(
                    activation_sq_sum[int(latent_idx)] / max(int(activation_total_count[int(latent_idx)]), 1)
                    - (activation_sum[int(latent_idx)] / max(int(activation_total_count[int(latent_idx)]), 1)) ** 2,
                    0.0,
                )
                ** 0.5
            ),
            "activation_prevalence": float(
                activation_positive_count[int(latent_idx)] / max(int(activation_total_count[int(latent_idx)]), 1)
            ),
        }
    return rep_rows, support, scan_summary


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.sae_ckpt, args.sae_cfg = resolve_sae_paths(args.sae_variant, args.sae_ckpt, args.sae_cfg)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = args.out_dir / str(args.task) / str(args.class_label).replace("/", "_").replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = [out_dir / "concept_cards.csv", out_dir / "representative_tiles.csv", out_dir / "selected_concepts.json", out_dir / "summary.json"]
    if bool(args.skip_existing) and all(path.exists() for path in expected):
        export_outputs = maybe_export_concept_package(args, out_dir)
        print(f"[skip] outputs already exist in {out_dir}")
        if export_outputs:
            print(json.dumps({"concept_export": export_outputs}, indent=2))
        return

    association_task = str(args.association_task or args.task)
    task_dir = args.association_root / association_task
    candidates_all = load_association_candidates(
        task_dir=task_dir,
        task=association_task,
        class_label=str(args.class_label),
        metric=str(args.metric),
        min_cohen_d=float(args.min_cohen_d),
    )
    candidate_pool = candidates_all[: max(int(args.top_concepts), int(args.candidate_latents))]
    for row in candidate_pool:
        row["task"] = str(args.task)

    sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    sae_model.eval()

    requested_mode = str(args.mode)
    attention_model: torch.nn.Module | None = None
    attention_backend = "none"
    attention_fallback_reason = ""
    if requested_mode == "attention_aware":
        attention_model, attention_backend, attention_fallback_reason = load_attention_model(
            task=str(args.task),
            backend=str(args.backend),
            device=device,
            mil_ckpt=args.mil_ckpt,
            clam_ckpt=args.clam_ckpt,
            classifier_run_dir=args.classifier_run_dir,
            classifier_ckpt=args.classifier_ckpt,
        )
    effective_mode = "attention_aware" if attention_model is not None else "labels_only"

    if args.slides_csv is not None:
        slides = load_manifest_slides(args.slides_csv, str(args.class_label), int(args.max_slides), label_column=str(args.slide_label_column))
        slides_source = str(args.slides_csv)
    elif args.classifier_run_dir is not None:
        slides_source_path = args.classifier_run_dir / "task_manifest.csv"
        slides = load_manifest_slides(slides_source_path, str(args.class_label), int(args.max_slides), label_column=str(args.slide_label_column))
        slides_source = str(slides_source_path)
    else:
        slides = load_task_slides(task_dir, str(args.class_label), int(args.max_slides))
        slides_source = str(task_dir / "cohort_slides.csv")
    for slide in slides:
        slide["task"] = str(args.task)
    rep_rows, attention_support, scan_summary = representative_scan(
        slides=slides,
        candidate_latents=candidate_pool,
        sae_model=sae_model,
        d_in=int(d_in),
        device=device,
        batch_size=int(args.batch_size),
        top_tiles_per_concept=int(args.top_tiles_per_concept),
        attention_model=attention_model,
        attention_backend=attention_backend,
        attn_class=str(args.attn_class),
    )

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
            top_k=int(args.morphology_coherence_top_k),
        )
    prevalence_scores = [
        prevalence_quality_score(
            attention_support[int(row["latent_idx"])]["activation_prevalence"],
            target=float(args.morphology_target_prevalence),
            sigma=float(args.morphology_prevalence_sigma),
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
        if str(args.concept_quality_mode) == "morphology":
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
        else:
            final_score = (
                float(args.association_weight) * float(row["association_score"])
                + float(args.attention_weight) * attention_support_score
                if effective_mode == "attention_aware"
                else float(row["association_score"])
            )
        concept_cards.append(
            {
                **row,
                "concept_quality_mode": str(args.concept_quality_mode),
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
                "steering_direction": f"toward_{args.class_label}",
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
    selected = concept_cards[: int(args.top_concepts)]
    for rank, row in enumerate(selected, start=1):
        row["concept_rank"] = int(rank)

    selected_latents = {int(row["latent_idx"]) for row in selected}
    rep_rows = [row for row in rep_rows if int(row["latent_idx"]) in selected_latents]
    rep_rows.sort(key=lambda r: (int(r["latent_idx"]), str(r["ranking_method"]), int(r["tile_rank"])))

    concept_fields = [
        "concept_rank",
        "task",
        "class_label",
        "latent_idx",
        "metric",
        "association_rank",
        "association_score",
        "cohen_d",
        "abs_diff",
        "diff_class_minus_rest",
        "mean_class",
        "mean_rest",
        "n_class",
        "n_rest",
        "concept_quality_mode",
        "activation_mean",
        "activation_std",
        "activation_prevalence",
        "prevalence_quality_score",
        "prototype_coherence_raw",
        "prototype_coherence_score",
        "activation_mean_score",
        "attention_support_raw",
        "attention_peak_mean",
        "attention_support_score",
        "final_score",
        "steering_direction",
        "top_activation_slide_key",
        "top_activation_tile_index",
        "top_activation_coord_x",
        "top_activation_coord_y",
        "top_activation",
        "top_attention_slide_key",
        "top_attention_tile_index",
        "top_attention_coord_x",
        "top_attention_coord_y",
        "top_attention_weighted_activation",
    ]
    rep_fields = [
        "task",
        "class_label",
        "latent_idx",
        "ranking_method",
        "tile_rank",
        "activation",
        "attention",
        "attention_norm",
        "attention_weighted_activation",
        "case_id",
        "slide_key",
        "project_dir",
        "label",
        "h5_path",
        "tile_index",
        "coord_x",
        "coord_y",
    ]

    write_csv(out_dir / "concept_cards.csv", selected, concept_fields)
    write_csv(out_dir / "representative_tiles.csv", rep_rows, rep_fields)
    write_json(
        out_dir / "selected_concepts.json",
        {
            "task": str(args.task),
            "class_label": str(args.class_label),
            "mode": effective_mode,
            "concept_quality_mode": str(args.concept_quality_mode),
            "concepts": selected,
        },
    )
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:]))),
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "concept_quality_mode": str(args.concept_quality_mode),
        "morphology_target_prevalence": float(args.morphology_target_prevalence),
        "morphology_prevalence_sigma": float(args.morphology_prevalence_sigma),
        "morphology_coherence_top_k": int(args.morphology_coherence_top_k),
        "attention_backend": attention_backend,
        "attention_fallback_reason": attention_fallback_reason,
        "association_weight": float(args.association_weight),
        "attention_weight": float(args.attention_weight),
        "sae_d_in": int(d_in),
        "sae_d_latent": int(d_latent),
        "association_task": association_task,
        "task_dir": str(task_dir),
        "slides_source": slides_source,
        "slides_for_class": int(len(slides)),
        "candidate_latents_total": int(len(candidates_all)),
        "candidate_latents_scanned": int(len(candidate_pool)),
        "concept_cards": int(len(selected)),
        "representative_tiles": int(len(rep_rows)),
        "scan": scan_summary,
        "outputs": {
            "concept_cards_csv": str(out_dir / "concept_cards.csv"),
            "representative_tiles_csv": str(out_dir / "representative_tiles.csv"),
            "selected_concepts_json": str(out_dir / "selected_concepts.json"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    export_outputs = maybe_export_concept_package(args, out_dir)
    if export_outputs:
        summary["outputs"]["concept_export_dir"] = str((args.concept_export_dir or (out_dir / "concept_export")))
        summary["outputs"]["concept_export"] = export_outputs
        write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
