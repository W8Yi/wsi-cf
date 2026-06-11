from __future__ import annotations

import math
from pathlib import Path
import random
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from wsi_cf.models.concept_risk import LinearProportionalOddsRisk
from wsi_cf.steering.sae_runtime import sae_encode_features


AGGREGATE_NAMES = ("mean_activation", "fraction_active", "top_fraction_mean")


def aggregate_activation_array(
    activations: np.ndarray,
    *,
    active_threshold: float = 1e-6,
    top_fraction: float = 0.05,
) -> dict[str, np.ndarray]:
    z = np.asarray(activations, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError(f"Expected non-empty tile-by-latent array, got {z.shape}")
    n_top = max(1, int(math.ceil(float(top_fraction) * int(z.shape[0]))))
    top_values = np.partition(z, kth=int(z.shape[0]) - n_top, axis=0)[-n_top:]
    return {
        "mean_activation": z.mean(axis=0, dtype=np.float64).astype(np.float32),
        "fraction_active": (z > float(active_threshold)).mean(axis=0).astype(np.float32),
        "top_fraction_mean": top_values.mean(axis=0, dtype=np.float64).astype(np.float32),
    }


@torch.no_grad()
def summarize_feature_bag_with_sae(
    features: np.ndarray,
    *,
    sae_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    d_latent: int,
    active_threshold: float,
    top_fraction: float,
    selected_latents: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"Expected a non-empty feature bag, got {x.shape}")
    n_tiles = int(x.shape[0])
    latent_ids = None
    out_dim = int(d_latent)
    if selected_latents is not None:
        latent_ids = torch.as_tensor(np.asarray(selected_latents, dtype=np.int64), device=device)
        out_dim = int(latent_ids.numel())
    n_top = max(1, int(math.ceil(float(top_fraction) * n_tiles)))
    activation_sum = np.zeros((out_dim,), dtype=np.float64)
    active_sum = np.zeros((out_dim,), dtype=np.float64)
    top_values: torch.Tensor | None = None
    for start in range(0, n_tiles, int(batch_size)):
        end = min(start + int(batch_size), n_tiles)
        batch = torch.as_tensor(x[start:end], dtype=torch.float32, device=device)
        z = sae_encode_features(sae_model, batch).to(dtype=torch.float32)
        if latent_ids is not None:
            z = torch.index_select(z, dim=1, index=latent_ids)
        activation_sum += z.sum(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
        active_sum += (z > float(active_threshold)).sum(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
        candidates = z if top_values is None else torch.cat((top_values, z), dim=0)
        keep = min(n_top, int(candidates.shape[0]))
        top_values = torch.topk(candidates, k=keep, dim=0, largest=True, sorted=False).values
    if top_values is None:
        raise RuntimeError("No SAE activations computed")
    return {
        "mean_activation": (activation_sum / n_tiles).astype(np.float32),
        "fraction_active": (active_sum / n_tiles).astype(np.float32),
        "top_fraction_mean": top_values.mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return ranks


def spearman_columns(values: np.ndarray, target: Iterable[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    y_rank = _average_ranks(np.asarray(list(target), dtype=np.float64))
    y_centered = y_rank - y_rank.mean()
    y_norm = float(np.sqrt(np.sum(y_centered * y_centered)))
    correlations = np.zeros((x.shape[1],), dtype=np.float32)
    if y_norm == 0.0:
        return correlations
    for column in range(x.shape[1]):
        x_rank = _average_ranks(x[:, column])
        centered = x_rank - x_rank.mean()
        norm = float(np.sqrt(np.sum(centered * centered)))
        if norm > 0.0:
            correlations[column] = float(np.sum(centered * y_centered) / (norm * y_norm))
    return correlations


def select_top_latents(
    aggregates: dict[str, np.ndarray],
    targets: Iterable[float],
    *,
    latent_ids: np.ndarray,
    top_n: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    correlations = {name: spearman_columns(aggregates[name], targets) for name in AGGREGATE_NAMES}
    rows: list[dict[str, Any]] = []
    for index, latent_id in enumerate(np.asarray(latent_ids).tolist()):
        statistic = max(AGGREGATE_NAMES, key=lambda name: (abs(float(correlations[name][index])), -AGGREGATE_NAMES.index(name)))
        rows.append(
            {
                "latent_idx": int(latent_id),
                "rank_score": abs(float(correlations[statistic][index])),
                "winning_statistic": statistic,
                **{f"spearman_{name}": float(correlations[name][index]) for name in AGGREGATE_NAMES},
            }
        )
    rows.sort(key=lambda row: (-float(row["rank_score"]), int(row["latent_idx"])))
    selected_rows = rows[: min(int(top_n), len(rows))]
    for rank, row in enumerate(selected_rows, start=1):
        row["selection_rank"] = rank
    selected = np.asarray([int(row["latent_idx"]) for row in selected_rows], dtype=np.int64)
    return selected, selected_rows


def select_stable_top_latents(
    aggregates: dict[str, np.ndarray],
    targets: Iterable[float],
    *,
    strata: Sequence[str],
    latent_ids: np.ndarray,
    top_n: int,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select latents repeatedly supported by stratified training-only folds."""
    y = np.asarray(list(targets), dtype=np.float32)
    if int(n_folds) <= 1:
        return select_top_latents(aggregates, y.tolist(), latent_ids=latent_ids, top_n=top_n)
    if y.shape[0] != len(strata):
        raise ValueError("Selection targets and strata must contain the same number of examples")
    fold_ids = np.zeros((y.shape[0],), dtype=np.int64)
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        by_stratum[str(stratum)].append(index)
    for stratum, indices in sorted(by_stratum.items()):
        shuffled = list(indices)
        random.Random(int(seed) + sum(ord(char) for char in stratum)).shuffle(shuffled)
        for offset, index in enumerate(shuffled):
            fold_ids[index] = offset % int(n_folds)

    _, full_rows = select_top_latents(aggregates, y.tolist(), latent_ids=latent_ids, top_n=len(latent_ids))
    full_by_latent = {int(row["latent_idx"]): dict(row) for row in full_rows}
    fold_scores: dict[int, list[float]] = defaultdict(list)
    selected_counts: dict[int, int] = defaultdict(int)
    completed_folds = 0
    for fold in range(int(n_folds)):
        keep = fold_ids != fold
        if not np.any(keep):
            continue
        fold_selected, fold_rows = select_top_latents(
            {name: values[keep] for name, values in aggregates.items()},
            y[keep].tolist(),
            latent_ids=latent_ids,
            top_n=top_n,
        )
        completed_folds += 1
        fold_score_by_latent = {int(row["latent_idx"]): float(row["rank_score"]) for row in fold_rows}
        for latent in fold_selected.tolist():
            selected_counts[int(latent)] += 1
            fold_scores[int(latent)].append(fold_score_by_latent[int(latent)])
    if completed_folds == 0:
        raise ValueError("Stable concept selection produced no folds")

    rows: list[dict[str, Any]] = []
    for latent in np.asarray(latent_ids).tolist():
        latent_int = int(latent)
        row = full_by_latent[latent_int]
        values = fold_scores.get(latent_int, [])
        row.update(
            {
                "selection_fold_count": int(selected_counts.get(latent_int, 0)),
                "selection_fold_fraction": float(selected_counts.get(latent_int, 0) / completed_folds),
                "selection_fold_mean_score": float(np.mean(values)) if values else 0.0,
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -int(row["selection_fold_count"]),
            -float(row["selection_fold_mean_score"]),
            -float(row["rank_score"]),
            int(row["latent_idx"]),
        )
    )
    selected_rows = rows[: min(int(top_n), len(rows))]
    for rank, row in enumerate(selected_rows, start=1):
        row["selection_rank"] = rank
    selected = np.asarray([int(row["latent_idx"]) for row in selected_rows], dtype=np.int64)
    return selected, selected_rows


def aggregate_summaries_by_case(
    rows: list[dict[str, Any]],
    aggregates: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Average slide-level SAE summaries so each patient contributes once."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["case_id"])].append(index)
    case_rows: list[dict[str, Any]] = []
    case_aggregates: dict[str, list[np.ndarray]] = {name: [] for name in AGGREGATE_NAMES}
    for case_id in sorted(grouped):
        indices = grouped[case_id]
        first = dict(rows[indices[0]])
        grades = {str(rows[index]["raw_grade"]) for index in indices}
        splits = {str(rows[index]["split"]) for index in indices}
        if len(grades) != 1 or len(splits) != 1:
            raise ValueError(f"Case {case_id} spans multiple grades or model splits")
        first.update(
            {
                "slide_key": case_id,
                "sample_id": case_id,
                "h5_path": "",
                "n_slides": len(indices),
                "slide_keys": ";".join(str(rows[index]["slide_key"]) for index in indices),
            }
        )
        case_rows.append(first)
        for name in AGGREGATE_NAMES:
            case_aggregates[name].append(np.asarray(aggregates[name])[indices].mean(axis=0))
    combined = {name: np.stack(values).astype(np.float32) for name, values in case_aggregates.items()}
    combined["latent_ids"] = np.asarray(aggregates["latent_ids"], dtype=np.int64)
    return case_rows, combined


def build_feature_matrix(
    aggregates: dict[str, np.ndarray],
    *,
    selected_latents: np.ndarray,
    latent_ids: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    index_by_latent = {int(latent): index for index, latent in enumerate(np.asarray(latent_ids).tolist())}
    indices = np.asarray([index_by_latent[int(latent)] for latent in selected_latents], dtype=np.int64)
    matrix = np.concatenate([np.asarray(aggregates[name])[:, indices] for name in AGGREGATE_NAMES], axis=1).astype(np.float32)
    columns = [
        {"feature_index": offset * len(indices) + index, "statistic": name, "latent_idx": int(latent)}
        for offset, name in enumerate(AGGREGATE_NAMES)
        for index, latent in enumerate(selected_latents.tolist())
    ]
    return matrix, columns


def fit_feature_scaler(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(train_features, dtype=np.float64).mean(axis=0).astype(np.float32)
    scale = np.asarray(train_features, dtype=np.float64).std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def apply_feature_scaler(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(features, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / np.asarray(scale, dtype=np.float32)).astype(np.float32)


def load_sae_grade_risk_model(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
) -> tuple[LinearProportionalOddsRisk, dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = LinearProportionalOddsRisk(
        n_features=int(config["n_features"]),
        n_thresholds=int(config["n_thresholds"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
