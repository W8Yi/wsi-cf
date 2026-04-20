from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from wsi_cf.common.paths import ensure_legacy_repo_root_on_path

ensure_legacy_repo_root_on_path()

from models.classifier import AttentionMIL, GatedAttentionMIL  # type: ignore
from utils.sae import load_sae_from_config  # type: ignore
from utils.sae_edit import edit_uni_z_grid_with_sae  # type: ignore


def build_mil_from_checkpoint(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})
    model_type = saved_args.get("model", "attention")
    common = {
        "embed_dim": int(saved_args.get("embed_dim", 1536)),
        "hidden_dim": int(saved_args.get("hidden_dim", 512)),
        "attn_dim": int(saved_args.get("attn_dim", 256)),
        "n_classes": 2,
        "dropout": float(saved_args.get("dropout", 0.25)),
    }
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


def run_mil_attention(mil_model: torch.nn.Module, x: np.ndarray, *, device: torch.device) -> tuple[np.ndarray, int, float]:
    xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        _, y_prob, y_hat, a_raw, _ = mil_model(xt)
        attn = F.softmax(a_raw, dim=1).detach().cpu().numpy().reshape(-1)
        pred = int(y_hat.detach().cpu()[0, 0].item())
        prob_pos = float(y_prob.detach().cpu()[0, 1].item())
    return attn, pred, prob_pos


def read_h5_features_coords(h5_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path, "r") as handle:
        feats = handle["features"]
        if feats.ndim == 2:
            x = feats[:]
        elif feats.ndim == 3 and feats.shape[0] == 1:
            x = feats[0]
        else:
            raise ValueError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")
        coords = None
        if "coords" in handle:
            c = handle["coords"]
            if c.ndim == 2 and c.shape[1] == 2:
                coords = c[:]
            elif c.ndim == 3 and c.shape[0] == 1:
                coords = c[0]
    return np.asarray(x, dtype=np.float32), (np.asarray(coords) if coords is not None else None)


def resolve_test_rows(*, split_json: Path, split_tsv: Path, features_root: Path) -> list[dict[str, Any]]:
    payload = json.loads(split_json.read_text())
    test_list = payload.get("test", [])
    test_map = {Path(str(p)).name.split(".")[0]: str(p) for p in test_list}
    rows: list[dict[str, Any]] = []
    with split_tsv.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("split", "")) != "test":
                continue
            slide_key = str(row.get("slide_key", ""))
            if slide_key not in test_map:
                continue
            label = int(row.get("label", -1))
            if label not in (0, 1):
                continue
            h5_path = features_root / f"{slide_key}.h5"
            if not h5_path.exists():
                candidate = Path(str(row.get("h5_path", "")))
                if candidate.exists():
                    h5_path = candidate
                else:
                    json_candidate = Path(test_map[slide_key])
                    if json_candidate.exists():
                        h5_path = json_candidate
                    else:
                        continue
            rows.append(
                {
                    "slide_key": slide_key,
                    "case_id": str(row.get("case_id", slide_key)),
                    "label": label,
                    "h5_path": str(h5_path),
                }
            )
    rows.sort(key=lambda item: (int(item["label"]), str(item["slide_key"])))
    return rows


def load_prototypes(npz_path: Path, key: str) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    with np.load(npz_path, allow_pickle=False) as data:
        latent_ids = np.asarray(data["latent_ids"], dtype=np.int64).reshape(-1)
        selected_direction = np.asarray(data["selected_direction"]).reshape(-1)
        vectors = np.asarray(data[key], dtype=np.float32)
    proto_by_latent: dict[int, np.ndarray] = {}
    direction_by_latent: dict[int, str] = {}
    for idx, latent in enumerate(latent_ids.tolist()):
        proto_by_latent[int(latent)] = np.asarray(vectors[idx], dtype=np.float32)
        direction_by_latent[int(latent)] = str(selected_direction[idx])
    return proto_by_latent, direction_by_latent


def pick_prototype_latent(
    proto_by_latent: dict[int, np.ndarray],
    direction_by_latent: dict[int, str],
    *,
    preferred: int | None,
    direction: str,
) -> int:
    if preferred is not None and int(preferred) in proto_by_latent:
        return int(preferred)
    candidates = sorted(
        latent for latent, selected_direction in direction_by_latent.items() if selected_direction == direction
    )
    if not candidates:
        raise RuntimeError(f"No prototype latent found for direction={direction}")
    return int(candidates[0])


def steer_feature(
    *,
    sae_model: torch.nn.Module,
    feature: np.ndarray,
    proto_vec: np.ndarray,
    strength: float,
    blend: float,
) -> np.ndarray:
    x = np.asarray(feature, dtype=np.float32).reshape(1, 1, -1)
    x_t = torch.from_numpy(x)
    edited, _ = edit_uni_z_grid_with_sae(
        sae_model,
        x_t,
        target_latent_vector=torch.from_numpy(np.asarray(proto_vec, dtype=np.float32)),
        target_latent_vector_strength=float(strength),
        blend=float(blend),
        return_debug=False,
    )
    return edited.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def run_counterfactual_eval(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    rows = resolve_test_rows(split_json=args.split_json, split_tsv=args.split_tsv, features_root=args.features_root)
    mil_model = build_mil_from_checkpoint(args.mil_ckpt, device=device)
    sae_model, d_in, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    proto_by_latent, direction_by_latent = load_prototypes(args.prototype_npz, args.prototype_key)

    pos_latent = pick_prototype_latent(
        proto_by_latent,
        direction_by_latent,
        preferred=int(args.pos_latent),
        direction="hpv_pos",
    )
    neg_latent = pick_prototype_latent(
        proto_by_latent,
        direction_by_latent,
        preferred=int(args.neg_latent),
        direction="hpv_neg",
    )
    proto_pos = proto_by_latent[pos_latent]
    proto_neg = proto_by_latent[neg_latent]

    results: list[dict[str, Any]] = []
    for row in rows:
        x, coords = read_h5_features_coords(str(row["h5_path"]))
        if x.shape[1] != d_in:
            continue
        attn, pred_orig, prob_orig = run_mil_attention(mil_model, x, device=device)
        top_tiles = np.argsort(-attn)[: int(args.top_tiles_per_slide)]
        for rank, tile_idx in enumerate(top_tiles.tolist(), start=1):
            x_orig = np.asarray(x[tile_idx], dtype=np.float32)
            for steer_direction, proto_vec in (("to_hpv_pos", proto_pos), ("to_hpv_neg", proto_neg)):
                x_cf_tile = steer_feature(
                    sae_model=sae_model,
                    feature=x_orig,
                    proto_vec=proto_vec,
                    strength=float(args.prototype_strength),
                    blend=float(args.blend),
                )
                x_cf = np.asarray(x, dtype=np.float32).copy()
                x_cf[tile_idx] = x_cf_tile
                _, pred_cf, prob_cf = run_mil_attention(mil_model, x_cf, device=device)
                results.append(
                    {
                        "slide_key": str(row["slide_key"]),
                        "case_id": str(row["case_id"]),
                        "label": int(row["label"]),
                        "tile_rank_by_attention": int(rank),
                        "tile_index": int(tile_idx),
                        "attention": float(attn[tile_idx]),
                        "coord_x": (int(coords[tile_idx, 0]) if coords is not None else ""),
                        "coord_y": (int(coords[tile_idx, 1]) if coords is not None else ""),
                        "pred_orig": int(pred_orig),
                        "prob_pos_orig": float(prob_orig),
                        "pred_cf": int(pred_cf),
                        "prob_pos_cf": float(prob_cf),
                        "delta_prob_pos": float(prob_cf - prob_orig),
                        "steer_direction": steer_direction,
                    }
                )

    summary = {
        "n_results": len(results),
        "prototype_key": str(args.prototype_key),
        "prototype_strength": float(args.prototype_strength),
        "blend": float(args.blend),
        "pos_latent": int(pos_latent),
        "neg_latent": int(neg_latent),
    }
    return results, summary
