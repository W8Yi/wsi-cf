from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

from wsi_cf.models.mil import (
    AttentionMILOrdinalRegressor,
    AttentionMILRegressor,
    GatedAttentionMILOrdinalRegressor,
    GatedAttentionMILRegressor,
)


def build_grade_risk_model(args: dict[str, Any]) -> torch.nn.Module:
    common = {
        "embed_dim": int(args.get("embed_dim", 1536)),
        "hidden_dim": int(args.get("hidden_dim", 512)),
        "attn_dim": int(args.get("attn_dim", 256)),
        "dropout": float(args.get("dropout", 0.25)),
    }
    is_ordinal = str(args.get("objective", "regression")) == "ordinal"
    if is_ordinal:
        common["n_thresholds"] = int(args.get("n_thresholds", 3))
    if str(args.get("model", "gated")) == "gated":
        model_cls = GatedAttentionMILOrdinalRegressor if is_ordinal else GatedAttentionMILRegressor
        return model_cls(
            **common,
            learnable_temperature=not bool(args.get("fixed_temperature", False)),
            init_temperature=float(args.get("init_temperature", 1.0)),
        )
    model_cls = AttentionMILOrdinalRegressor if is_ordinal else AttentionMILRegressor
    return model_cls(**common)


def load_grade_risk_model(checkpoint_path, *, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = dict(checkpoint.get("args", {}))
    model = build_grade_risk_model(args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def score_feature_bag(model: torch.nn.Module, features: np.ndarray | torch.Tensor, *, device: torch.device) -> float:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    score, _, _ = model(x)
    return float(score.detach().cpu().reshape(-1)[0].item())


def _average_ranks(values: np.ndarray) -> np.ndarray:
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


def grade_risk_metrics(targets: Iterable[float], scores: Iterable[float]) -> dict[str, float | None]:
    y = np.asarray(list(targets), dtype=np.float64)
    pred = np.asarray(list(scores), dtype=np.float64)
    if y.size == 0:
        raise ValueError("Cannot compute grade-risk metrics on no examples")
    err = pred - y
    pearson = None
    spearman = None
    if y.size > 1 and float(np.std(y)) > 0 and float(np.std(pred)) > 0:
        pearson = float(np.corrcoef(y, pred)[0, 1])
        spearman = float(np.corrcoef(_average_ranks(y), _average_ranks(pred))[0, 1])
    return {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "pearson": pearson,
        "spearman": spearman,
        "within_one_grade_step": float(np.mean(np.abs(err) <= (1.0 / 3.0 + 1e-8))),
    }


def infer_coord_tile_size(coords: np.ndarray, fallback: int = 256) -> int:
    arr = np.asarray(coords, dtype=np.int64)
    candidates: list[int] = []
    for axis in (0, 1):
        values = np.unique(arr[:, axis])
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    return int(min(candidates)) if candidates else int(fallback)


def map_region_cells_to_bag(
    coords: np.ndarray,
    *,
    region_gx0: int,
    region_gy0: int,
    grid_shape: tuple[int, int],
    tile_size_px: int | None = None,
) -> dict[tuple[int, int], int]:
    step = int(tile_size_px) if tile_size_px is not None else infer_coord_tile_size(coords)
    coord_to_index = {
        (int(round(float(x) / step)), int(round(float(y) / step))): int(index)
        for index, (x, y) in enumerate(np.asarray(coords))
    }
    out: dict[tuple[int, int], int] = {}
    grid_h, grid_w = grid_shape
    for gy in range(int(grid_h)):
        for gx in range(int(grid_w)):
            index = coord_to_index.get((int(region_gx0) + gx, int(region_gy0) + gy))
            if index is not None:
                out[(gx, gy)] = index
    return out


def replace_region_features(
    original_features: np.ndarray,
    generated_grid: np.ndarray,
    cell_to_bag_index: dict[tuple[int, int], int],
    *,
    cells: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, int]:
    edited = np.asarray(original_features, dtype=np.float32).copy()
    replaced = 0
    for (gx, gy), bag_index in cell_to_bag_index.items():
        if cells is not None and (gx, gy) not in cells:
            continue
        edited[bag_index] = generated_grid[gy, gx]
        replaced += 1
    return edited, replaced
