#!/usr/bin/env python3
"""
extract_clam_test_attention_and_gradients.py

Goal
----
For each TEST slide (given by a JSON manifest with keys: meta/train/val/test),
compute per-tile:
  - attention weight a_i (from CLAM)
  - gradient norm || d logit_c / d h_i ||  (feature-space sensitivity)
  - score = a_i * grad_norm  (Attention × Gradient)
and save everything needed to choose tiles to edit.

Assumptions
-----------
- Each .h5 contains:
    features: (N, D) float
    coords:   (N, 2) int (top-left at level 0)
- You have a trained CLAM checkpoint and can import the CLAM model class.

This script is written to be robust to common CLAM forward signatures:
  forward(x) returns one of:
    (logits, Y_prob, Y_hat, A_raw, h)  or
    (logits, Y_prob, Y_hat, A_raw)     or
    dict with keys containing logits / attention / embeddings

If 'h' (instance embeddings used by attention) is not returned, we will treat
the input features as the instance embedding (common when you pre-extract UNI
and CLAM uses them directly). Gradients will then be w.r.t. the input features.

Outputs
-------
- out_dir/tiles/<slide_id>.parquet  (per tile rows)
- out_dir/slides.parquet            (per slide summary)
- out_dir/config.json               (run config)

Usage
-----
python extract_clam_test_attention_and_gradients.py \
  --manifest /path/to/manifest.json \
  --ckpt /path/to/clam_checkpoint.pt \
  --out_dir /path/to/out_grad_att \
  --clam_repo /common/users/wq50/CLAM \
  --model_name clam_sb \
  --n_classes 2 \
  --pos_class 1 \
  --device cuda:0 \
  --topk 200

Notes
-----
- "pos_class" is the class index you want gradients for (often 1 for HPV+).
- For binary classification, using the logit for pos_class is standard.
- We DO NOT update model weights; we only backprop to instance embeddings.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch


# -------------------------
# Utilities
# -------------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _slide_id_from_h5_path(h5_path: str) -> str:
    return Path(h5_path).stem


def _load_h5_features_coords(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        if "features" not in f:
            raise KeyError(f"{h5_path}: missing dataset 'features'")
        feats = f["features"][:]
        coords = f["coords"][:] if "coords" in f else None
    if coords is None:
        coords = np.zeros((feats.shape[0], 2), dtype=np.int64)
    return feats, coords


def _normalize_attention(A: torch.Tensor) -> torch.Tensor:
    """
    CLAM may return attention as:
      - shape (1, N) or (N,)
      - or multi-head (K, N) / (1, K, N)
    We return a single attention per instance by:
      - squeezing batch dims
      - if multi-head: average over heads
      - apply softmax over N if not already normalized (best-effort)
    """
    A = A.detach()
    # squeeze batch dims
    while A.dim() > 2 and A.size(0) == 1:
        A = A.squeeze(0)
    if A.dim() == 2:
        # could be (K, N) or (1, N)
        if A.size(0) > 1:
            A = A.mean(dim=0)  # average heads
        else:
            A = A.squeeze(0)
    if A.dim() != 1:
        A = A.view(-1)
    # best-effort normalization:
    # if sums close to 1 and nonnegative -> assume already softmaxed
    s = float(A.sum().cpu().item())
    if not (0.95 <= s <= 1.05) or (A.min().item() < -1e-6):
        A = torch.softmax(A, dim=0)
    return A


def _infer_split_labels_from_csv(split_csv: Optional[str], test_h5_list: list[str]) -> Dict[str, Optional[int]]:
    """
    Best-effort mapping from slide_id -> label using split_csv.

    We do NOT assume exact CLAM csv format (varies across repos).
    We try common columns:
      - slide_id / slides / filename / case_id
      - label / y / class
    If not found, return label=None for all slides.

    The split_csv path is provided in manifest["meta"]["split_csv"].
    """
    labels: Dict[str, Optional[int]] = { _slide_id_from_h5_path(p): None for p in test_h5_list }
    if not split_csv:
        return labels
    split_csv = str(split_csv)
    if not os.path.exists(split_csv):
        return labels

    df = pd.read_csv(split_csv)
    cols = {c.lower(): c for c in df.columns}

    # find id column
    id_candidates = ["slide_id", "slide", "slides", "filename", "file", "case_id", "patient_id"]
    id_col = None
    for c in id_candidates:
        if c in cols:
            id_col = cols[c]
            break
    if id_col is None:
        # fallback: first column
        id_col = df.columns[0]

    # find label column
    lab_candidates = ["label", "y", "class", "target"]
    lab_col = None
    for c in lab_candidates:
        if c in cols:
            lab_col = cols[c]
            break

    if lab_col is None:
        return labels

    # make mapping
    tmp = {}
    for _, row in df.iterrows():
        sid_raw = str(row[id_col])
        sid = Path(sid_raw).stem  # strip any extension
        try:
            y = int(row[lab_col])
        except Exception:
            y = None
        tmp[sid] = y

    for sid in labels.keys():
        if sid in tmp:
            labels[sid] = tmp[sid]
    return labels


@dataclass
class ForwardPack:
    logits: torch.Tensor          # (1, C) or (C,)
    probs: torch.Tensor           # (1, C) or (C,)
    yhat: torch.Tensor            # (1,) or scalar
    attention: torch.Tensor       # (N,) after normalize
    inst_emb: torch.Tensor        # (N, D) tensor with requires_grad=True (for grad)


def _unpack_clam_forward(out: Any, x_in: torch.Tensor, pos_class: int) -> ForwardPack:
    """
    Unpack model output robustly, and decide which tensor to differentiate w.r.t.
    Prefer 'h' if returned; otherwise use x_in (input features).
    """
    logits = probs = yhat = A_raw = h = None

    if isinstance(out, dict):
        # common keys
        for k in ["logits", "logit"]:
            if k in out:
                logits = out[k]
                break
        for k in ["Y_prob", "probs", "prob", "y_prob"]:
            if k in out:
                probs = out[k]
                break
        for k in ["Y_hat", "yhat", "pred", "prediction"]:
            if k in out:
                yhat = out[k]
                break
        for k in ["A", "attn", "attention", "att"]:
            if k in out:
                A_raw = out[k]
                break
        for k in ["h", "H", "instance_emb", "embeddings"]:
            if k in out:
                h = out[k]
                break
    elif isinstance(out, (tuple, list)):
        # common CLAM signature: (logits, Y_prob, Y_hat, A, h) or (logits, Y_prob, Y_hat, A)
        if len(out) >= 4:
            logits, probs, yhat, A_raw = out[:4]
            if len(out) >= 5:
                h = out[4]
        else:
            raise RuntimeError(f"Unsupported forward tuple length: {len(out)}")
    else:
        raise RuntimeError(f"Unsupported forward output type: {type(out)}")

    if logits is None:
        raise RuntimeError("Could not find logits in model output.")
    if probs is None:
        # fallback: softmax logits
        probs = torch.softmax(logits, dim=-1)
    if yhat is None:
        yhat = torch.argmax(probs, dim=-1)

    if A_raw is None:
        raise RuntimeError("Could not find attention (A) in model output.")

    attention = _normalize_attention(A_raw)

    # choose instance embedding tensor for gradients:
    # If h exists and looks like (1, N, D) or (N, D), use it. Else use x_in.
    inst = None
    if h is not None and torch.is_tensor(h):
        ht = h
        # squeeze batch dims
        while ht.dim() > 2 and ht.size(0) == 1:
            ht = ht.squeeze(0)
        if ht.dim() == 2:
            inst = ht
    if inst is None:
        # use input features; remove batch if present
        xt = x_in
        while xt.dim() > 2 and xt.size(0) == 1:
            xt = xt.squeeze(0)
        if xt.dim() != 2:
            raise RuntimeError(f"Input features unexpected shape for inst_emb: {tuple(xt.shape)}")
        inst = xt

    # ensure attention length matches N
    N = inst.size(0)
    if attention.numel() != N:
        raise RuntimeError(f"Attention length {attention.numel()} != N {N}. Raw A shape may be different than expected.")

    # standardize logits/probs shapes
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    if yhat.dim() == 0:
        yhat = yhat.unsqueeze(0)

    # sanity for pos_class
    C = logits.size(-1)
    if not (0 <= pos_class < C):
        raise ValueError(f"--pos_class {pos_class} out of range for C={C}")

    return ForwardPack(
        logits=logits,
        probs=probs,
        yhat=yhat,
        attention=attention,
        inst_emb=inst,
    )


def _freeze_model_params(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


# -------------------------
# Model loading
# -------------------------

def _load_clam_model(
    ckpt_path: str,
    clam_repo: Optional[str],
    model_name: str,
    n_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    You must adapt this function to your exact CLAM repo structure if needed.

    The defaults assume the CLAM repo has model definitions importable.
    Common patterns:
      - from models.model_clam import CLAM_SB, CLAM_MB
    """
    if clam_repo:
        sys.path.insert(0, str(Path(clam_repo).resolve()))

    model_name = model_name.lower()
    if model_name in ["clam_sb", "sb", "clam-sb"]:
        try:
            from models.model_clam import CLAM_SB  # type: ignore
            model = CLAM_SB(n_classes=n_classes)
        except Exception as e:
            raise RuntimeError(f"Failed importing CLAM_SB from your repo. Edit _load_clam_model(). Error: {e}")
    elif model_name in ["clam_mb", "mb", "clam-mb"]:
        try:
            
            # adjust import to your repo
            from models.model_clam import CLAM_MB

            model = CLAM_MB(gate=True, size_arg="small", n_classes=2, embed_dim=1536)
     
    
        except Exception as e:
            raise RuntimeError(f"Failed importing CLAM_MB from your repo. Edit _load_clam_model(). Error: {e}")
    else:
        raise ValueError(f"Unknown --model_name {model_name}. Supported: clam_sb, clam_mb")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # support common checkpoint formats
    state = ckpt.get("state_dict", None) if isinstance(ckpt, dict) else None
    if state is None and isinstance(ckpt, dict):
        # sometimes stored as 'model'
        state = ckpt.get("model", None)
    if state is None:
        # maybe the dict is already a state_dict
        state = ckpt if isinstance(ckpt, dict) else None
    if state is None:
        raise RuntimeError("Could not locate state_dict in checkpoint.")

    # strip 'module.' if saved from DDP
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys in state_dict (strict=False): {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys in state_dict (strict=False): {len(unexpected)}")

    model.to(device)
    model.eval()
    _freeze_model_params(model)
    return model


# -------------------------
# Main extraction
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON with keys meta/train/val/test")
    ap.add_argument("--ckpt", required=True, help="CLAM checkpoint (.pt)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clam_repo", default=None, help="Path to CLAM repo to import models/")
    ap.add_argument("--model_name", default="clam_sb", help="clam_sb or clam_mb")
    ap.add_argument("--n_classes", type=int, default=2)
    ap.add_argument("--pos_class", type=int, default=1, help="class index to take gradients for (e.g., HPV+ = 1)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=200, help="how many top tiles (by att*grad) to include in slide summary")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"], help="input tensor dtype")
    ap.add_argument("--save_full_grad", action="store_true", help="store full gradient vectors (large). Default stores grad_norm only.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tiles_dir = out_dir / "tiles"
    _safe_mkdir(out_dir)
    _safe_mkdir(tiles_dir)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    test_list = manifest.get("test", [])
    if not test_list:
        raise RuntimeError("Manifest has empty 'test' list.")

    split_csv = manifest.get("meta", {}).get("split_csv", None)
    labels_map = _infer_split_labels_from_csv(split_csv, test_list)

    # dtype for features tensor
    if args.dtype == "fp16":
        tdtype = torch.float16
    elif args.dtype == "bf16":
        tdtype = torch.bfloat16
    else:
        tdtype = torch.float32

    model = _load_clam_model(
        ckpt_path=args.ckpt,
        clam_repo=args.clam_repo,
        model_name=args.model_name,
        n_classes=args.n_classes,
        device=device,
    )

    slide_rows = []

    for idx, h5_path in enumerate(test_list):
        sid = _slide_id_from_h5_path(h5_path)
        y_true = labels_map.get(sid, None)

        feats_np, coords_np = _load_h5_features_coords(h5_path)
        if feats_np.ndim != 2:
            raise RuntimeError(f"{h5_path}: expected features (N,D), got {feats_np.shape}")
        N, D = feats_np.shape

        # tensor
        x = torch.from_numpy(feats_np).to(device=device, dtype=tdtype)
        # CLAM usually expects (1, N, D)
        x = x.unsqueeze(0)
        x.requires_grad_(True)

        # Forward + backward for gradients
        # We only need grad wrt x (or returned h). Parameters are frozen.
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()

        out = model(x)
        pack = _unpack_clam_forward(out, x, pos_class=args.pos_class)

        # choose target logit
        # Using logit for pos_class is standard; more stable than prob.
        target_logit = pack.logits[0, args.pos_class]
        target_logit.backward(retain_graph=False)

        # gradients:
        # If pack.inst_emb is x-derived, its grad will be in x.grad; if pack.inst_emb is an internal tensor,
        # it may have grad attached directly. We handle both.
        inst = pack.inst_emb
        grad = None
        if inst.grad is not None:
            grad = inst.grad.detach()
        else:
            # fallback: if inst is x squeezed view, use x.grad
            if x.grad is None:
                raise RuntimeError(f"{sid}: could not obtain gradients. Ensure instance embeddings require_grad.")
            # x: (1,N,D) -> squeeze to (N,D)
            grad = x.grad.detach().squeeze(0)

        if grad.dim() != 2 or grad.size(0) != N:
            raise RuntimeError(f"{sid}: grad shape {tuple(grad.shape)} mismatch N={N}")

        att = pack.attention.detach().float().cpu().numpy()  # (N,)
        grad_norm = torch.linalg.vector_norm(grad.float(), dim=1).cpu().numpy()  # (N,)
        score = att * grad_norm

        probs = pack.probs.detach().float().cpu().numpy().squeeze(0)  # (C,)
        y_hat = int(pack.yhat.detach().cpu().numpy().squeeze())

        # per-tile dataframe
        df_tiles = pd.DataFrame({
            "slide_id": sid,
            "tile_idx": np.arange(N, dtype=np.int64),
            "x": coords_np[:, 0].astype(np.int64),
            "y": coords_np[:, 1].astype(np.int64),
            "attention": att.astype(np.float32),
            "grad_norm": grad_norm.astype(np.float32),
            "att_x_grad": score.astype(np.float32),
        })

        if args.save_full_grad:
            # store gradient vector as list (large); parquet will be heavy.
            # Alternative is saving npy per slide; but keep it simple here.
            df_tiles["grad_vec"] = [g.astype(np.float32) for g in grad.float().cpu().numpy()]

        # rank tiles for editing
        order = np.argsort(-score)
        topk = min(args.topk, N)
        top_idx = order[:topk]
        df_tiles["rank_att_x_grad"] = np.empty(N, dtype=np.int64)
        df_tiles.loc[top_idx, "rank_att_x_grad"] = np.arange(topk, dtype=np.int64)
        # fill others with -1
        mask = np.ones(N, dtype=bool)
        mask[top_idx] = False
        df_tiles.loc[mask, "rank_att_x_grad"] = -1

        # save per-slide tiles
        out_tiles_path = tiles_dir / f"{sid}.parquet"
        df_tiles.to_parquet(out_tiles_path, index=False)

        # slide summary
        slide_rows.append({
            "slide_id": sid,
            "h5_path": h5_path,
            "y_true": y_true,
            "y_hat": y_hat,
            "prob_0": float(probs[0]) if probs.size > 0 else None,
            "prob_pos": float(probs[args.pos_class]) if probs.size > args.pos_class else None,
            "N_tiles": int(N),
            "topk": int(topk),
            "top1_tile_idx": int(top_idx[0]) if topk > 0 else None,
            "top1_x": int(coords_np[top_idx[0], 0]) if topk > 0 else None,
            "top1_y": int(coords_np[top_idx[0], 1]) if topk > 0 else None,
            "top1_attention": float(att[top_idx[0]]) if topk > 0 else None,
            "top1_grad_norm": float(grad_norm[top_idx[0]]) if topk > 0 else None,
            "top1_att_x_grad": float(score[top_idx[0]]) if topk > 0 else None,
            "tiles_parquet": str(out_tiles_path),
        })

        print(f"[{idx+1}/{len(test_list)}] {sid}: y_hat={y_hat} prob_pos={slide_rows[-1]['prob_pos']:.4f} N={N} saved={out_tiles_path}")

    # save slide summary
    df_slides = pd.DataFrame(slide_rows)
    slides_path = out_dir / "slides.parquet"
    df_slides.to_parquet(slides_path, index=False)

    # save config
    cfg = {
        "manifest": args.manifest,
        "ckpt": args.ckpt,
        "clam_repo": args.clam_repo,
        "model_name": args.model_name,
        "n_classes": args.n_classes,
        "pos_class": args.pos_class,
        "split_csv": split_csv,
        "device": str(device),
        "dtype": args.dtype,
        "topk": args.topk,
        "save_full_grad": args.save_full_grad,
        "outputs": {
            "slides": str(slides_path),
            "tiles_dir": str(tiles_dir),
        },
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Done. Slides summary: {slides_path}")


if __name__ == "__main__":
    main()
