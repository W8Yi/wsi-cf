from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.classifier import AttentionMIL, GatedAttentionMIL
from utils.sae import load_sae_from_config, sae_decode_latents, sae_encode_features
from utils.uni import get_uni

try:
    import openslide
except Exception:  # pragma: no cover
    openslide = None


def load_json(path: Path | str) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_manifest_index(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else (REPO_ROOT / "metadata" / "indexes" / "manifest_index.json")
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict root in {path}")
    return data


def resolve_h5_path_with_source(
    raw_h5_path: str,
    *,
    slide_key: str,
    manifest_index: dict[str, Any],
    preferred_h5_roots: Iterable[Path | str] | None = None,
) -> tuple[Path, str]:
    p = Path(raw_h5_path) if raw_h5_path else Path()
    if raw_h5_path and p.is_absolute() and p.exists():
        return p, "raw_absolute"

    if raw_h5_path:
        roots = [Path(root) for root in (preferred_h5_roots or [])]
        for root in roots:
            cand = (root / p).resolve()
            if cand.exists():
                return cand, f"preferred_root:{root}"
            cand_name = (root / p.name).resolve()
            if cand_name.exists():
                return cand_name, f"preferred_root_name:{root}"

        repo_rel = (REPO_ROOT / p).resolve()
        if repo_rel.exists():
            return repo_rel, "repo_relative"

    rec = manifest_index.get(slide_key)
    if isinstance(rec, dict):
        h5_abs = rec.get("h5_path")
        if isinstance(h5_abs, str):
            hp = Path(h5_abs)
            if hp.exists():
                return hp, "manifest_index"

    raise FileNotFoundError(f"Could not resolve H5 path for slide_key={slide_key}: {raw_h5_path}")


def resolve_h5_path(raw_h5_path: str, *, slide_key: str, manifest_index: dict[str, Any]) -> Path:
    path, _source = resolve_h5_path_with_source(raw_h5_path, slide_key=slide_key, manifest_index=manifest_index)
    return path


def resolve_wsi_path(slide_key: str, wsi_dir: Path | str) -> Path | None:
    wsi_dir = Path(wsi_dir)
    direct = wsi_dir / f"{slide_key}.svs"
    if direct.exists():
        return direct
    matches = sorted(wsi_dir.glob(f"{slide_key}*.svs"))
    if matches:
        return matches[0]
    return None


def load_predictions_df(
    pred_csv: Path | str,
    *,
    manifest_index: dict[str, Any] | None = None,
    preferred_h5_roots: Iterable[Path | str] | None = None,
) -> pd.DataFrame:
    manifest_index = manifest_index if manifest_index is not None else load_manifest_index()
    rows = read_csv_rows(pred_csv)
    out: list[dict[str, Any]] = []
    for row in rows:
        slide_key = str(row["slide_key"])
        resolved_h5_path, resolved_h5_source = resolve_h5_path_with_source(
            row["h5_path"],
            slide_key=slide_key,
            manifest_index=manifest_index,
            preferred_h5_roots=preferred_h5_roots,
        )
        out.append(
            {
                "case_id": row["case_id"],
                "slide_key": slide_key,
                "label": int(row["label"]),
                "pred": int(row["pred"]),
                "prob_pos": float(row["prob_pos"]),
                "raw_h5_path": row["h5_path"],
                "resolved_h5_path": str(resolved_h5_path),
                "resolved_h5_source": resolved_h5_source,
            }
        )
    return pd.DataFrame(out)


def load_top_neuron_tiles_df(
    tiles_csv: Path | str,
    *,
    manifest_index: dict[str, Any] | None = None,
    preferred_h5_roots: Iterable[Path | str] | None = None,
) -> pd.DataFrame:
    manifest_index = manifest_index if manifest_index is not None else load_manifest_index()
    rows = read_csv_rows(tiles_csv)
    out: list[dict[str, Any]] = []
    for row in rows:
        slide_key = str(row["slide_key"])
        resolved_h5_path, resolved_h5_source = resolve_h5_path_with_source(
            row["h5_path"],
            slide_key=slide_key,
            manifest_index=manifest_index,
            preferred_h5_roots=preferred_h5_roots,
        )
        out.append(
            {
                "latent_idx": int(row["latent_idx"]),
                "selected_direction": row["selected_direction"],
                "prototype_rank": int(row["prototype_rank"]),
                "label": int(row["label"]),
                "pred": int(row["pred"]),
                "prob_pos": float(row["prob_pos"]),
                "case_id": row["case_id"],
                "slide_key": slide_key,
                "tile_index": int(row["tile_index"]),
                "attention": float(row["attention"]),
                "sae_activation": float(row["sae_activation"]),
                "attention_weighted_activation": float(row["attention_weighted_activation"]),
                "coord_x": int(row["coord_x"]),
                "coord_y": int(row["coord_y"]),
                "raw_h5_path": row["h5_path"],
                "resolved_h5_path": str(resolved_h5_path),
                "resolved_h5_source": resolved_h5_source,
            }
        )
    return pd.DataFrame(out)


def load_exported_latent_tile_dir_df(
    latent_dir: Path | str,
    *,
    manifest_index: dict[str, Any] | None = None,
    top_neuron_df: pd.DataFrame | None = None,
    latent_idx: int | None = None,
    preferred_h5_roots: Iterable[Path | str] | None = None,
) -> pd.DataFrame:
    manifest_index = manifest_index if manifest_index is not None else load_manifest_index()
    latent_dir = Path(latent_dir)
    tiles_dir = latent_dir / "tiles" if (latent_dir / "tiles").exists() else latent_dir
    pat = re.compile(r"^rank_(\d+)__(.+)__tile_(\d+)\.png$")

    rows: list[dict[str, Any]] = []
    for png_path in sorted(tiles_dir.glob("*.png")):
        m = pat.match(png_path.name)
        if m is None:
            continue
        prototype_rank = int(m.group(1))
        slide_key = str(m.group(2))
        tile_index = int(m.group(3))
        rec: dict[str, Any] = {
            "prototype_rank": prototype_rank,
            "slide_key": slide_key,
            "tile_index": tile_index,
            "tile_png_path": str(png_path),
        }
        try:
            resolved_h5_path, resolved_h5_source = resolve_h5_path_with_source(
                "",
                slide_key=slide_key,
                manifest_index=manifest_index,
                preferred_h5_roots=preferred_h5_roots,
            )
            rec["resolved_h5_path"] = str(resolved_h5_path)
            rec["resolved_h5_source"] = resolved_h5_source
        except Exception:
            continue
        rows.append(rec)

    if not rows:
        raise ValueError(f"No exported tile PNGs found in {tiles_dir}")

    df = pd.DataFrame(rows).sort_values(["prototype_rank", "slide_key", "tile_index"]).reset_index(drop=True)

    if top_neuron_df is not None:
        meta_df = top_neuron_df.copy()
        if latent_idx is not None and "latent_idx" in meta_df.columns:
            meta_df = meta_df.loc[meta_df["latent_idx"] == int(latent_idx)].copy()
        keep_cols = [
            c
            for c in [
                "latent_idx",
                "selected_direction",
                "label",
                "pred",
                "prob_pos",
                "case_id",
                "attention",
                "sae_activation",
                "attention_weighted_activation",
                "coord_x",
                "coord_y",
                "raw_h5_path",
                "resolved_h5_path",
                "resolved_h5_source",
            ]
            if c in meta_df.columns
        ]
        meta_df = meta_df[["slide_key", "tile_index", "prototype_rank", *keep_cols]].drop_duplicates(
            subset=["slide_key", "tile_index", "prototype_rank"]
        )
        df = df.merge(meta_df, on=["slide_key", "tile_index", "prototype_rank"], how="left", suffixes=("", "_meta"))
        if "resolved_h5_path_meta" in df.columns:
            df["resolved_h5_path"] = df["resolved_h5_path_meta"].fillna(df["resolved_h5_path"])
            df = df.drop(columns=["resolved_h5_path_meta"])
        if "resolved_h5_source_meta" in df.columns:
            df["resolved_h5_source"] = df["resolved_h5_source_meta"].fillna(df["resolved_h5_source"])
            df = df.drop(columns=["resolved_h5_source_meta"])

    return df


def load_prototype_table(
    npz_path: Path | str,
    *,
    key: str = "prototype_median",
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=False) as data:
        latent_ids = np.asarray(data["latent_ids"], dtype=np.int64).reshape(-1)
        protos = np.asarray(data[key], dtype=np.float32)
    if protos.ndim != 2:
        raise ValueError(f"{npz_path}:{key} expected [M,L], got {protos.shape}")
    table = {int(lid): protos[i] for i, lid in enumerate(latent_ids.tolist())}
    return table, latent_ids, protos


def build_mil_model_from_checkpoint(ckpt_path: Path | str, device: str | torch.device) -> torch.nn.Module:
    device = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    saved_args = ckpt.get("args", {})

    common = {
        "embed_dim": int(saved_args.get("embed_dim", 1536)),
        "hidden_dim": int(saved_args.get("hidden_dim", 512)),
        "attn_dim": int(saved_args.get("attn_dim", 256)),
        "n_classes": 2,
        "dropout": float(saved_args.get("dropout", 0.25)),
    }
    model_type = str(saved_args.get("model", "attention"))
    if model_type == "gated":
        model = GatedAttentionMIL(
            **common,
            learnable_temperature=(not bool(saved_args.get("fixed_temperature", False))),
            init_temperature=float(saved_args.get("init_temperature", 1.0)),
        )
    else:
        model = AttentionMIL(**common)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_bag_h5(h5_path: Path | str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(str(h5_path), "r") as f:
        feats = f["features"]
        if feats.ndim == 3 and feats.shape[0] == 1:
            x = feats[0]
        elif feats.ndim == 2:
            x = feats[:]
        else:
            raise ValueError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")

        coords = None
        if "coords" in f:
            c = f["coords"]
            if c.ndim == 3 and c.shape[0] == 1:
                coords = c[0]
            elif c.ndim == 2 and c.shape[1] == 2:
                coords = c[:]

    return np.asarray(x, dtype=np.float32), (None if coords is None else np.asarray(coords, dtype=np.int64))


def load_h5_feature_rows(h5_path: Path | str, tile_indices: Iterable[int]) -> np.ndarray:
    idx = np.asarray([int(x) for x in tile_indices], dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    order = np.argsort(idx)
    idx_sorted = idx[order]
    with h5py.File(str(h5_path), "r") as f:
        ds = f["features"]
        if ds.ndim == 2:
            x_sorted = ds[idx_sorted]
        elif ds.ndim == 3 and ds.shape[0] == 1:
            x_sorted = ds[0, idx_sorted]
        else:
            raise ValueError(f"{h5_path}: unsupported features shape {tuple(ds.shape)}")
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return np.asarray(x_sorted, dtype=np.float32)[inv]


@torch.no_grad()
def predict_bag(model: torch.nn.Module, bag_np: np.ndarray, *, device: str | torch.device) -> dict[str, Any]:
    device = torch.device(device)
    h = torch.from_numpy(np.asarray(bag_np, dtype=np.float32)).to(device=device, dtype=torch.float32)
    logits, y_prob, y_hat, a_raw, results = model(h, return_features=True)
    attn = F.softmax(a_raw, dim=1).detach().cpu().numpy().reshape(-1)
    probs = y_prob.detach().cpu().numpy().reshape(-1)
    logits_np = logits.detach().cpu().numpy().reshape(-1)
    return {
        "pred": int(y_hat.detach().cpu().item()),
        "prob_neg": float(probs[0]),
        "prob_pos": float(probs[1]),
        "logit_neg": float(logits_np[0]),
        "logit_pos": float(logits_np[1]),
        "attention": attn,
        "bag_feature": results.get("features", torch.empty(0)).detach().cpu().numpy() if results else None,
    }


def make_attention_df(
    attention: np.ndarray,
    *,
    coords: np.ndarray | None = None,
) -> pd.DataFrame:
    attn = np.asarray(attention, dtype=np.float32).reshape(-1)
    df = pd.DataFrame(
        {
            "tile_index": np.arange(attn.shape[0], dtype=np.int64),
            "attention": attn,
        }
    )
    df["attention_rank"] = df["attention"].rank(ascending=False, method="first").astype(int)
    if coords is not None:
        coords = np.asarray(coords, dtype=np.int64)
        if coords.shape[0] != attn.shape[0] or coords.shape[1] != 2:
            raise ValueError(f"coords must be [N,2], got {coords.shape}")
        df["coord_x"] = coords[:, 0]
        df["coord_y"] = coords[:, 1]
    return df.sort_values(["attention", "tile_index"], ascending=[False, True]).reset_index(drop=True)


@torch.no_grad()
def steer_bag_features(
    bag_np: np.ndarray,
    *,
    sae_model: torch.nn.Module,
    selected_indices: Iterable[int],
    device: str | torch.device,
    mode: str = "prototype_target",
    prototype_vec: np.ndarray | None = None,
    baseline_vec: np.ndarray | None = None,
    strength: float = 0.5,
    latent_idx: int | None = None,
    latent_delta: float | None = None,
    latent_target: float | None = None,
    blend: float = 1.0,
    max_feature_delta_norm: float | None = None,
    preserve_reconstruction_residual: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not (0.0 <= blend <= 1.0):
        raise ValueError("blend must be in [0,1]")
    device = torch.device(device)

    bag = np.asarray(bag_np, dtype=np.float32)
    x = torch.from_numpy(bag).to(device=device, dtype=torch.float32)
    z = sae_encode_features(sae_model, x)
    x_recon_orig = sae_decode_latents(sae_model, z)
    recon_residual = x - x_recon_orig
    z_edit = z.clone()

    sel = np.asarray(sorted({int(i) for i in selected_indices}), dtype=np.int64)
    if sel.size == 0:
        raise ValueError("selected_indices is empty")
    if sel.min() < 0 or sel.max() >= bag.shape[0]:
        raise IndexError(f"selected_indices must be within [0, {bag.shape[0] - 1}]")
    sel_t = torch.as_tensor(sel, device=device, dtype=torch.long)

    if mode == "prototype_target":
        if prototype_vec is None:
            raise ValueError("prototype_vec is required for prototype_target")
        proto = torch.from_numpy(np.asarray(prototype_vec, dtype=np.float32)).to(device=device, dtype=z.dtype).reshape(-1)
        if proto.numel() != z.shape[1]:
            raise ValueError(f"prototype_vec dim {proto.numel()} != latent dim {z.shape[1]}")
        z_edit[sel_t] = (1.0 - float(strength)) * z_edit[sel_t] + float(strength) * proto.view(1, -1)
    elif mode == "prototype_delta":
        if prototype_vec is None:
            raise ValueError("prototype_vec is required for prototype_delta")
        proto = np.asarray(prototype_vec, dtype=np.float32).reshape(-1)
        if baseline_vec is None:
            baseline_vec = np.zeros_like(proto)
        direction = torch.from_numpy(proto - np.asarray(baseline_vec, dtype=np.float32).reshape(-1))
        direction = direction.to(device=device, dtype=z.dtype)
        if direction.numel() != z.shape[1]:
            raise ValueError(f"prototype direction dim {direction.numel()} != latent dim {z.shape[1]}")
        z_edit[sel_t] = z_edit[sel_t] + float(strength) * direction.view(1, -1)
    elif mode == "latent_delta":
        if latent_idx is None or latent_delta is None:
            raise ValueError("latent_idx and latent_delta are required for latent_delta")
        z_edit[sel_t, int(latent_idx)] = z_edit[sel_t, int(latent_idx)] + float(latent_delta)
    elif mode == "latent_target":
        if latent_idx is None or latent_target is None:
            raise ValueError("latent_idx and latent_target are required for latent_target")
        z_edit[sel_t, int(latent_idx)] = float(latent_target)
    else:
        raise ValueError(f"Unsupported mode {mode}")

    x_rec = sae_decode_latents(sae_model, z_edit)
    if preserve_reconstruction_residual:
        x_rec = x_rec + recon_residual
    if max_feature_delta_norm is not None:
        d = x_rec - x
        dn = d.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scl = torch.clamp(float(max_feature_delta_norm) / dn, max=1.0)
        x_rec = x + d * scl

    x_new = x.clone()
    x_new[sel_t] = (1.0 - float(blend)) * x[sel_t] + float(blend) * x_rec[sel_t]

    out = x_new.detach().cpu().numpy().astype(np.float32, copy=False)
    debug = {
        "mode": mode,
        "selected_count": int(sel.size),
        "selected_indices": sel.tolist(),
        "blend": float(blend),
        "strength": float(strength),
        "latent_idx": None if latent_idx is None else int(latent_idx),
        "latent_delta": None if latent_delta is None else float(latent_delta),
        "latent_target": None if latent_target is None else float(latent_target),
        "preserve_reconstruction_residual": bool(preserve_reconstruction_residual),
        "mean_abs_reconstruction_residual_selected": float(torch.mean(torch.abs(recon_residual[sel_t])).item()),
        "max_abs_reconstruction_residual_selected": float(torch.max(torch.abs(recon_residual[sel_t])).item()),
        "mean_abs_feature_shift_selected": float(np.mean(np.abs(out[sel] - bag[sel]))),
        "max_abs_feature_shift_selected": float(np.max(np.abs(out[sel] - bag[sel]))),
    }
    return out, debug


def compare_predictions(before: dict[str, Any], after: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state": "before", "pred": before["pred"], "prob_pos": before["prob_pos"], "logit_pos": before["logit_pos"]},
            {"state": "after", "pred": after["pred"], "prob_pos": after["prob_pos"], "logit_pos": after["logit_pos"]},
            {
                "state": "delta",
                "pred": int(after["pred"]) - int(before["pred"]),
                "prob_pos": float(after["prob_pos"]) - float(before["prob_pos"]),
                "logit_pos": float(after["logit_pos"]) - float(before["logit_pos"]),
            },
        ]
    )


def predict_single_tiles(
    model: torch.nn.Module,
    bag_np: np.ndarray,
    *,
    tile_indices: Iterable[int],
    device: str | torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bag = np.asarray(bag_np, dtype=np.float32)
    for tile_idx in tile_indices:
        tile_idx = int(tile_idx)
        pred = predict_bag(model, bag[[tile_idx]], device=device)
        rows.append(
            {
                "tile_index": tile_idx,
                "pred": int(pred["pred"]),
                "prob_neg": float(pred["prob_neg"]),
                "prob_pos": float(pred["prob_pos"]),
                "logit_neg": float(pred["logit_neg"]),
                "logit_pos": float(pred["logit_pos"]),
            }
        )
    return pd.DataFrame(rows)


def build_prototype_from_top_neuron_df(
    top_neuron_df: pd.DataFrame,
    *,
    latent_idx: int,
    sae_model: torch.nn.Module,
    device: str | torch.device,
    top_n: int = 0,
    agg: str = "median",
    batch_size: int = 512,
) -> tuple[np.ndarray, pd.DataFrame]:
    rows = top_neuron_df.loc[top_neuron_df["latent_idx"] == int(latent_idx)].copy()
    if rows.empty:
        raise ValueError(f"No rows for latent {latent_idx}")
    rows = rows.sort_values(["prototype_rank", "attention_weighted_activation"], ascending=[True, False]).reset_index(drop=True)
    if top_n and top_n > 0:
        rows = rows.head(int(top_n)).copy()

    x_chunks: list[np.ndarray] = []
    used_groups: list[pd.DataFrame] = []
    for h5_path, group in rows.groupby("resolved_h5_path", sort=False):
        try:
            feats = load_h5_feature_rows(h5_path, group["tile_index"].astype(int).tolist())
        except Exception:
            continue
        x_chunks.append(feats)
        used_groups.append(group.copy())
    if not x_chunks:
        raise RuntimeError(f"No valid H5 groups could be loaded for latent {latent_idx}")
    X = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
    rows = pd.concat(used_groups, axis=0, ignore_index=True)

    device = torch.device(device)
    outs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, X.shape[0], int(batch_size)):
            xb = torch.from_numpy(X[start : start + int(batch_size)]).to(device=device, dtype=torch.float32)
            zb = sae_encode_features(sae_model, xb)
            outs.append(zb.detach().cpu().numpy().astype(np.float32, copy=False))
    Z = np.concatenate(outs, axis=0)

    if agg == "mean":
        proto = Z.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
    elif agg == "median":
        proto = np.median(Z, axis=0).astype(np.float32, copy=False)
    else:
        raise ValueError("agg must be 'mean' or 'median'")
    return proto, rows


def build_prototype_from_exported_latent_tile_dir(
    latent_dir: Path | str,
    *,
    sae_model: torch.nn.Module,
    device: str | torch.device,
    manifest_index: dict[str, Any] | None = None,
    top_neuron_df: pd.DataFrame | None = None,
    latent_idx: int | None = None,
    preferred_h5_roots: Iterable[Path | str] | None = None,
    top_n: int = 0,
    agg: str = "median",
    batch_size: int = 512,
) -> tuple[np.ndarray, pd.DataFrame]:
    rows = load_exported_latent_tile_dir_df(
        latent_dir,
        manifest_index=manifest_index,
        top_neuron_df=top_neuron_df,
        latent_idx=latent_idx,
        preferred_h5_roots=preferred_h5_roots,
    )
    if top_n and top_n > 0:
        rows = rows.head(int(top_n)).copy()

    x_chunks: list[np.ndarray] = []
    used_groups: list[pd.DataFrame] = []
    for h5_path, group in rows.groupby("resolved_h5_path", sort=False):
        try:
            feats = load_h5_feature_rows(h5_path, group["tile_index"].astype(int).tolist())
        except Exception:
            continue
        x_chunks.append(feats)
        used_groups.append(group.copy())
    if not x_chunks:
        raise RuntimeError(f"No valid H5 groups could be loaded from {latent_dir}")
    X = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
    rows = pd.concat(used_groups, axis=0, ignore_index=True)
    rows = rows.sort_values(["prototype_rank", "slide_key", "tile_index"], ascending=[True, True, True]).reset_index(drop=True)

    device = torch.device(device)
    outs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, X.shape[0], int(batch_size)):
            xb = torch.from_numpy(X[start : start + int(batch_size)]).to(device=device, dtype=torch.float32)
            zb = sae_encode_features(sae_model, xb)
            outs.append(zb.detach().cpu().numpy().astype(np.float32, copy=False))
    Z = np.concatenate(outs, axis=0)

    if agg == "mean":
        proto = Z.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
    elif agg == "median":
        proto = np.median(Z, axis=0).astype(np.float32, copy=False)
    else:
        raise ValueError("agg must be 'mean' or 'median'")
    return proto, rows


@torch.no_grad()
def decode_latent_vector_to_feature(
    latent_vec: np.ndarray,
    *,
    sae_model: torch.nn.Module,
    device: str | torch.device,
) -> np.ndarray:
    device = torch.device(device)
    z = torch.from_numpy(np.asarray(latent_vec, dtype=np.float32).reshape(1, -1)).to(device=device, dtype=torch.float32)
    x = sae_decode_latents(sae_model, z)
    return x.detach().cpu().numpy().astype(np.float32, copy=False)


def infer_objective_power(slide: "openslide.OpenSlide") -> float:
    props = slide.properties
    for key in ("openslide.objective-power", "aperio.AppMag"):
        if key in props:
            try:
                v = float(props[key])
            except Exception:
                v = -1.0
            if v > 0:
                return v
    try:
        mpp_x = float(props.get("openslide.mpp-x", -1.0))
    except Exception:
        mpp_x = -1.0
    if 0 < mpp_x <= 0.30:
        return 40.0
    if 0 < mpp_x <= 0.60:
        return 20.0
    return 20.0


def level0_tile_size(tile_size_20x: int, objective_power: float) -> int:
    return max(1, int(round(tile_size_20x * (objective_power / 20.0))))


def crop_slide_tile(
    slide_path: Path | str,
    *,
    coord_x: int,
    coord_y: int,
    tile_size_20x: int = 256,
    out_tile_size: int = 256,
) -> Image.Image:
    if openslide is None:  # pragma: no cover
        raise RuntimeError("openslide-python is required for tile cropping")
    slide = openslide.OpenSlide(str(slide_path))
    try:
        objective = infer_objective_power(slide)
        crop_px = level0_tile_size(tile_size_20x=tile_size_20x, objective_power=objective)
        rgba = slide.read_region((int(coord_x), int(coord_y)), 0, (crop_px, crop_px)).convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(bg, rgba).convert("RGB")
        if crop_px != out_tile_size:
            rgb = rgb.resize((out_tile_size, out_tile_size), resample=Image.BILINEAR)
        return rgb
    finally:
        slide.close()


@torch.no_grad()
def encode_uni_from_pil(
    img: Image.Image,
    *,
    uni_model: torch.nn.Module,
    uni_transform,
    device: str | torch.device,
) -> np.ndarray:
    device = torch.device(device)
    x = uni_transform(img).unsqueeze(0).to(device=device)
    feat = uni_model(x)
    return feat.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def load_uni_bundle(device: str | torch.device):
    return get_uni(str(device))


def crop_tiles_for_indices(
    *,
    slide_key: str,
    attn_df: pd.DataFrame,
    tile_indices: Iterable[int],
    wsi_dir: Path | str,
    tile_size_20x: int = 256,
    out_tile_size: int = 256,
) -> list[Image.Image]:
    slide_path = resolve_wsi_path(slide_key, wsi_dir)
    if slide_path is None:
        raise FileNotFoundError(f"Could not resolve WSI for {slide_key} in {wsi_dir}")
    lookup = attn_df.set_index("tile_index")
    images: list[Image.Image] = []
    for tile_idx in tile_indices:
        row = lookup.loc[int(tile_idx)]
        images.append(
            crop_slide_tile(
                slide_path,
                coord_x=int(row["coord_x"]),
                coord_y=int(row["coord_y"]),
                tile_size_20x=tile_size_20x,
                out_tile_size=out_tile_size,
            )
        )
    return images


def load_sae_bundle(
    sae_ckpt: Path | str,
    sae_cfg: Path | str,
    *,
    device: str | torch.device,
) -> tuple[torch.nn.Module, int, int]:
    return load_sae_from_config(sae_ckpt, sae_cfg, device=str(device))
