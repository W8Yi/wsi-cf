#!/usr/bin/env python3
"""
linear_probe_hpv_from_sae_latents_splits_fast_match.py

Fast SAE linear probe for HPV with robust ID matching.

Key features
------------
1) FAST H5 loading: reads only sampled rows from H5 (no full features[:] load)
2) FAST aggregation: accumulators on GPU; only final summaries moved to CPU
3) ROBUST label matching: handles mixed naming in your manifest and CSV:
   - H5 stems that are case-only:            TCGA-CV-6936.h5
   - H5 stems with slide barcode + UUID:     TCGA-CV-7097-01Z-00-DX1.<uuid>_001.h5
   - CSV slide_id entries that are case-only, slide+UUID, or weird strings like "TCGA-MZ-A6I9-ext 2"

Matching priority (per slide)
-----------------------------
(1) raw stem exact match (uppercased)            -> handles TCGA-CV-6936 / TCGA-MZ-A6I9-ext 2
(2) slide barcode match (drop UUID and _suffix)  -> TCGA-..-..-01Z-00-DX1
(3) case_id fallback                             -> TCGA-..-..

Outputs
-------
- slide_features_{train,val,test}.csv
- preds_{train,val,test}.csv
- probe_coef.csv, top_latents.txt, probe.joblib
- metrics_and_config.json
- match_report_{train,val,test}.csv (shows which key matched or why dropped)

Usage
-----
python linear_probe_hpv_from_sae_latents_splits_fast_match.py \
  --manifest /common/users/wq50/CLAM/splits/HPV_100/manifest_splits_0.json \
  --hnscc_csv /common/users/wq50/CLAM/dataset_csv/HNSCC.csv \
  --sae_ckpt /common/users/wq50/SAE_path/runs/tcga_topk64_single/last.pt \
  --sae_cfg  /common/users/wq50/SAE_path/runs/tcga_topk64_single/config.json \
  --stage topk \
  --out_dir /common/users/wq50/SAE_path/counterfactual/sae_probe_hpv_s0 \
  --device cuda:0 \
  --tiles_per_slide 512 \
  --batch_size 1024 \
  --seed 66
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib


# -------------------------
# ID parsing + label maps
# -------------------------

TCGA_CASE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
TCGA_SLIDE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]-\d{2}-DX\d)", re.IGNORECASE)


def parse_ids_from_name(name: str) -> Tuple[Optional[str], Optional[str], str, str]:
    """
    Returns:
      case_id:        'TCGA-XX-YYYY' (if found)
      slide_barcode:  'TCGA-XX-YYYY-01Z-00-DX1' (if found)
      raw_stem_upper: filename stem uppercased (keeps weird forms)
      core_upper:     stem stripped of '_...' and '.UUID' then uppercased
    """
    stem = Path(str(name)).stem
    raw = stem.upper()

    core = stem.split("_")[0]   # drop _001, _001_001
    core = core.split(".")[0]   # drop .UUID
    core_up = core.upper()

    mcase = TCGA_CASE_RE.search(core_up)
    case_id = mcase.group(1).upper() if mcase else None

    mslide = TCGA_SLIDE_RE.search(core_up)
    slide_id = mslide.group(1).upper() if mslide else None

    return case_id, slide_id, raw, core_up


def load_hpv_label_maps(csv_path: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Build 3 maps:
      by_raw_stem_upper: exact stem from CSV slide_id (uppercased)
      by_slide_barcode:  normalized slide barcode without UUID/suffix
      by_case_id:        TCGA-XX-YYYY

    This matches your CSV which mixes slide-level and case-level IDs.
    """
    df = pd.read_csv(csv_path)
    need = {"case_id", "slide_id", "hpv_status"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(need - set(df.columns))}")

    by_raw: Dict[str, int] = {}
    by_slide: Dict[str, int] = {}
    by_case: Dict[str, int] = {}

    for _, r in df.iterrows():
        hpv = str(r["hpv_status"]).strip().upper()
        if hpv == "HPV+":
            y = 1
        elif hpv == "HPV-":
            y = 0
        else:
            continue

        # case_id is clean
        cid = str(r["case_id"]).strip().upper()
        if cid:
            by_case[cid] = y

        # slide_id can be anything; store multiple keys
        sid_raw = str(r["slide_id"]).strip()
        if sid_raw:
            cid2, slide2, raw2, core2 = parse_ids_from_name(sid_raw)

            # exact raw stem match (handles TCGA-CV-6936 and TCGA-MZ-A6I9-ext 2)
            by_raw[raw2] = y

            # core (without uuid/_suffix) is also useful for case-only forms
            by_raw[core2] = y

            # slide barcode match
            if slide2 is not None:
                by_slide[slide2] = y

            # case fallback from parsed slide_id
            if cid2 is not None:
                by_case[cid2] = y

    return by_raw, by_slide, by_case


def lookup_label_and_reason(
    h5_path: str,
    by_raw: Dict[str, int],
    by_slide: Dict[str, int],
    by_case: Dict[str, int],
) -> Tuple[Optional[int], str, Optional[str], Optional[str], str, str]:
    """
    Returns:
      y, reason, case_id, slide_barcode, raw_stem_upper, core_upper
    """
    cid, slide, raw, core = parse_ids_from_name(h5_path)

    # 1) exact raw stem or core match
    if raw in by_raw:
        return by_raw[raw], "match_raw", cid, slide, raw, core
    if core in by_raw:
        return by_raw[core], "match_core", cid, slide, raw, core

    # 2) slide barcode match
    if slide is not None and slide in by_slide:
        return by_slide[slide], "match_slide_barcode", cid, slide, raw, core

    # 3) case fallback
    if cid is not None and cid in by_case:
        return by_case[cid], "match_case_fallback", cid, slide, raw, core

    return None, "no_match", cid, slide, raw, core


# -------------------------
# Misc utilities
# -------------------------

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_manifest_splits(manifest_path: str) -> Dict[str, List[str]]:
    m = load_json(manifest_path)
    splits = {k: m.get(k, []) for k in ("train", "val", "test")}
    if not splits["train"]:
        raise RuntimeError("manifest['train'] is empty")
    return splits


def stable_int_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def sample_tile_indices(n: int, cap: Optional[int], seed: int) -> np.ndarray:
    if cap is None or cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=cap, replace=False).astype(np.int64)


def load_h5_feature_rows(h5_path: str, idx: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Read only selected rows from H5 'features'.

    Returns:
      x: (len(idx), D)
      shape: (N, D) of the full dataset
    """
    idx = np.asarray(idx, dtype=np.int64)
    with h5py.File(h5_path, "r") as f:
        if "features" not in f:
            raise KeyError(f"{h5_path}: missing dataset 'features'")
        dset = f["features"]
        if dset.ndim != 2:
            raise RuntimeError(f"{h5_path}: expected (N,D) features, got {dset.shape}")
        N, D = dset.shape

        if idx.size == 0:
            raise RuntimeError(f"{h5_path}: empty idx")
        if idx.min() < 0 or idx.max() >= N:
            raise RuntimeError(f"{h5_path}: idx out of bounds for N={N}")

        order = np.argsort(idx)
        idx_sorted = idx[order]
        x_sorted = dset[idx_sorted, :]
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        x = x_sorted[inv]

    return x, (int(N), int(D))


# -------------------------
# SAE loading
# -------------------------

def load_sae(
    sae_ckpt: str,
    sae_cfg: str,
    stage: str,
    device: torch.device,
):
    """
    Expects:
      ckpt["model"] contains the model state dict
      ckpt may have "wrapper_layernorm": bool
      cfg JSON contains d_in, latent_dim, tied, and for TopK: topk_k, topk_mode, topk_nonneg
    """
    from models.sae import ReLUSparseSAE, TopKSAE, InputNormWrapper

    ckpt = torch.load(sae_ckpt, map_location="cpu")
    cfg = load_json(sae_cfg)

    wrapper_ln = bool(ckpt.get("wrapper_layernorm", False))
    d_in = int(cfg["d_in"])
    d_latent = int(cfg["latent_dim"])
    tied = bool(cfg.get("tied", False))

    if stage == "relu":
        base = ReLUSparseSAE(d_in=d_in, d_latent=d_latent, tied=tied, use_pre_bias=True)
    elif stage == "topk":
        base = TopKSAE(
            d_in=d_in,
            d_latent=d_latent,
            k=int(cfg["topk_k"]),
            tied=tied,
            use_pre_bias=True,
            topk_mode=str(cfg.get("topk_mode", "value")),
            nonneg=bool(cfg.get("topk_nonneg", False)),
        )
    else:
        raise ValueError("--stage must be 'relu' or 'topk'")

    base.load_state_dict(ckpt["model"], strict=True)
    model = InputNormWrapper(d_in, base) if wrapper_ln else base
    model.eval().to(device)
    return model, d_in, d_latent


@torch.no_grad()
def sae_encode_z_fast(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Fast path to get z.
    If you add model.encode(x)->z later, it will be used automatically.
    """
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        z = model.encode(x)
        if not torch.is_tensor(z):
            raise RuntimeError("model.encode(x) did not return a tensor")
        return z

    out = model(x)  # (x_hat, z, a) or similar
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError(f"Unexpected SAE output type={type(out)}")
    z = out[1]
    if not torch.is_tensor(z):
        raise RuntimeError("SAE output[1] is not a tensor")
    return z


# -------------------------
# Slide aggregation (GPU accumulators)
# -------------------------

@torch.no_grad()
def slide_latent_summary_fast(
    sae: torch.nn.Module,
    h5_path: str,
    device: torch.device,
    tiles_cap: Optional[int],
    batch_size: int,
    seed: int,
    d_in_expected: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads only sampled rows from H5, runs SAE, accumulates on GPU:
      mean(z), max(z), nzfrac(z)
    """
    # get (N,D) without loading all
    with h5py.File(h5_path, "r") as f:
        dset = f["features"]
        N, D = dset.shape

    if int(D) != int(d_in_expected):
        raise RuntimeError(f"{Path(h5_path).stem}: features dim {D} != SAE d_in {d_in_expected}")

    idx = sample_tile_indices(int(N), tiles_cap, seed)
    x_np, _shape = load_h5_feature_rows(h5_path, idx)

    total = x_np.shape[0]
    if total == 0:
        raise RuntimeError("No tiles selected")

    z_sum = None
    z_max = None
    nz_sum = None

    for s in range(0, total, batch_size):
        xb = torch.from_numpy(x_np[s : s + batch_size]).to(device=device, dtype=torch.float32, non_blocking=False)
        z = sae_encode_z_fast(sae, xb).float()  # GPU

        if z_sum is None:
            L = z.size(1)
            z_sum = torch.zeros(L, device=device, dtype=torch.float32)
            z_max = torch.full((L,), -1e18, device=device, dtype=torch.float32)
            nz_sum = torch.zeros(L, device=device, dtype=torch.float32)

        z_sum += z.sum(dim=0)
        z_max = torch.maximum(z_max, z.max(dim=0).values)
        nz_sum += (z != 0).float().sum(dim=0)

    z_mean = (z_sum / float(total)).detach().cpu().numpy()
    z_max_np = z_max.detach().cpu().numpy()
    z_nzfrac = (nz_sum / float(total)).detach().cpu().numpy()
    return z_mean, z_max_np, z_nzfrac


def build_phi(z_mean: np.ndarray, z_max: np.ndarray, z_nzfrac: np.ndarray) -> np.ndarray:
    return np.concatenate([z_mean, z_max, z_nzfrac], axis=0)


def make_feature_columns(d_latent: int) -> List[str]:
    return (
        [f"mean_{j}" for j in range(d_latent)] +
        [f"max_{j}" for j in range(d_latent)] +
        [f"nzfrac_{j}" for j in range(d_latent)]
    )


# -------------------------
# Split dataset building
# -------------------------

def build_split_dataset_fast(
    split_name: str,
    h5_list: List[str],
    by_raw: Dict[str, int],
    by_slide: Dict[str, int],
    by_case: Dict[str, int],
    sae: torch.nn.Module,
    d_in: int,
    d_latent: int,
    device: torch.device,
    tiles_cap: Optional[int],
    batch_size: int,
    base_seed: int,
    out_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_rows: List[dict] = []
    report_rows: List[dict] = []

    for k, h5_path in enumerate(h5_list):
        y, reason, cid, slide, raw, core = lookup_label_and_reason(h5_path, by_raw, by_slide, by_case)
        report_rows.append({
            "split": split_name,
            "h5_path": h5_path,
            "raw_stem_upper": raw,
            "core_upper": core,
            "case_id": cid,
            "slide_barcode": slide,
            "match_reason": reason,
            "y": (int(y) if y is not None else None),
        })
        if y is None:
            continue

        # stable per-sample seed based on core string (more stable than raw)
        seed_key = core if core else raw
        sseed = int(base_seed + stable_int_hash(seed_key))

        z_mean, z_max, z_nzfrac = slide_latent_summary_fast(
            sae=sae,
            h5_path=h5_path,
            device=device,
            tiles_cap=tiles_cap,
            batch_size=batch_size,
            seed=sseed,
            d_in_expected=d_in,
        )
        phi = build_phi(z_mean, z_max, z_nzfrac)

        X_list.append(phi.astype(np.float32))
        y_list.append(int(y))
        meta_rows.append({"split": split_name, "slide_id": core, "h5_path": h5_path, "y": int(y)})

        if (k + 1) % 25 == 0:
            print(f"[{split_name}] processed {k+1}/{len(h5_list)} slides (kept: {len(X_list)})")

    # Write match report always
    pd.DataFrame(report_rows).to_csv(out_dir / f"match_report_{split_name}.csv", index=False)

    if len(X_list) == 0:
        raise RuntimeError(
            f"[{split_name}] no labeled slides after matching. "
            f"Inspect {out_dir / f'match_report_{split_name}.csv'}"
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    df_meta = pd.DataFrame(meta_rows)
    df_feat = pd.DataFrame(X, columns=make_feature_columns(d_latent))

    # basic split stats
    pos = int(y.sum())
    neg = int(len(y) - pos)
    print(f"[{split_name}] kept={len(y)} pos={pos} neg={neg}")

    return X, y, df_meta, df_feat


# -------------------------
# Eval
# -------------------------

def eval_probe(probe: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    prob = probe.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)

    auc = float("nan")
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, prob))

    acc = float(accuracy_score(y, pred))
    bacc = float(balanced_accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, zero_division=0))
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {
        "n": int(len(y)),
        "auc": auc,
        "acc": acc,
        "balanced_acc": bacc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hnscc_csv", required=True)
    ap.add_argument("--sae_ckpt", required=True)
    ap.add_argument("--sae_cfg", required=True)
    ap.add_argument("--stage", choices=["relu", "topk"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tiles_per_slide", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=66)

    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--penalty", choices=["l2", "l1"], default="l2")
    ap.add_argument("--solver", choices=["lbfgs", "liblinear", "saga"], default="lbfgs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (("cuda" in args.device and torch.cuda.is_available()) or "cpu" in args.device) else "cpu"
    )

    splits = load_manifest_splits(args.manifest)
    by_raw, by_slide, by_case = load_hpv_label_maps(args.hnscc_csv)

    print(f"Label maps: by_raw={len(by_raw)} by_slide={len(by_slide)} by_case={len(by_case)}")

    sae, d_in, d_latent = load_sae(args.sae_ckpt, args.sae_cfg, args.stage, device)
    tiles_cap = args.tiles_per_slide if args.tiles_per_slide and args.tiles_per_slide > 0 else None

    X_train, y_train, meta_train, feat_train = build_split_dataset_fast(
        "train", splits["train"], by_raw, by_slide, by_case, sae, d_in, d_latent, device,
        tiles_cap, args.batch_size, args.seed, out_dir
    )

    X_val = y_val = meta_val = feat_val = None
    X_test = y_test = meta_test = feat_test = None

    if splits["val"]:
        X_val, y_val, meta_val, feat_val = build_split_dataset_fast(
            "val", splits["val"], by_raw, by_slide, by_case, sae, d_in, d_latent, device,
            tiles_cap, args.batch_size, args.seed, out_dir
        )

    if splits["test"]:
        X_test, y_test, meta_test, feat_test = build_split_dataset_fast(
            "test", splits["test"], by_raw, by_slide, by_case, sae, d_in, d_latent, device,
            tiles_cap, args.batch_size, args.seed, out_dir
        )

    if len(y_train) < 10:
        raise RuntimeError(f"Too few labeled TRAIN slides: {len(y_train)}. Inspect match_report_train.csv")

    if args.penalty == "l1" and args.solver not in ("liblinear", "saga"):
        raise ValueError("penalty=l1 requires solver=liblinear or saga")

    probe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            C=args.C,
            penalty=args.penalty,
            solver=args.solver,
            max_iter=5000,
            class_weight="balanced",
        )),
    ])

    probe.fit(X_train, y_train)

    metrics = {"train": eval_probe(probe, X_train, y_train)}
    print(f"[TRAIN] {metrics['train']}")

    if X_val is not None:
        metrics["val"] = eval_probe(probe, X_val, y_val)
        print(f"[VAL]   {metrics['val']}")

    if X_test is not None:
        metrics["test"] = eval_probe(probe, X_test, y_test)
        print(f"[TEST]  {metrics['test']}")

    # Save features per split
    pd.concat([meta_train, feat_train], axis=1).to_csv(out_dir / "slide_features_train.csv", index=False)
    if X_val is not None:
        pd.concat([meta_val, feat_val], axis=1).to_csv(out_dir / "slide_features_val.csv", index=False)
    if X_test is not None:
        pd.concat([meta_test, feat_test], axis=1).to_csv(out_dir / "slide_features_test.csv", index=False)

    # Save predictions per split
    def save_preds(name: str, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> None:
        prob = probe.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(np.int64)
        dfp = meta.copy()
        dfp["prob_hpv_pos"] = prob
        dfp["pred"] = pred
        dfp.to_csv(out_dir / f"preds_{name}.csv", index=False)

    save_preds("train", X_train, y_train, meta_train)
    if X_val is not None:
        save_preds("val", X_val, y_val, meta_val)
    if X_test is not None:
        save_preds("test", X_test, y_test, meta_test)

    # Coefficients
    clf: LogisticRegression = probe.named_steps["clf"]
    coef = clf.coef_.reshape(-1)  # (3*d_latent,)
    blocks = [("mean", 0), ("max", d_latent), ("nzfrac", 2 * d_latent)]

    recs = []
    for bname, off in blocks:
        for j in range(d_latent):
            w = float(coef[off + j])
            recs.append({
                "block": bname,
                "latent": j,
                "coef": w,
                "abscoef": abs(w),
                "direction": "HPV+" if w > 0 else ("HPV-" if w < 0 else "0"),
            })
    pd.DataFrame(recs).sort_values(["block", "abscoef"], ascending=[True, False]).to_csv(
        out_dir / "probe_coef.csv", index=False
    )

    # Top latents summary
    best = []
    for j in range(d_latent):
        w_mean = float(coef[0 + j])
        w_max = float(coef[d_latent + j])
        w_nz = float(coef[2 * d_latent + j])
        wabs = max(abs(w_mean), abs(w_max), abs(w_nz))
        best.append((j, wabs, w_mean, w_max, w_nz))
    best.sort(key=lambda t: -t[1])

    with open(out_dir / "top_latents.txt", "w") as f:
        f.write("Ranked by max abs(coef) across {mean,max,nzfrac}\n")
        f.write("latent  max_abs  coef_mean  coef_max  coef_nzfrac\n")
        for (j, wabs, wm, wx, wn) in best[:200]:
            f.write(f"{j:6d}  {wabs:8.4f}  {wm:9.4f}  {wx:8.4f}  {wn:11.4f}\n")

    joblib.dump(probe, out_dir / "probe.joblib")

    cfg_out = {
        "manifest": args.manifest,
        "hnscc_csv": args.hnscc_csv,
        "sae_ckpt": args.sae_ckpt,
        "sae_cfg": args.sae_cfg,
        "stage": args.stage,
        "device": str(device),
        "tiles_per_slide": args.tiles_per_slide,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "d_in": d_in,
        "d_latent": d_latent,
        "probe": {"C": args.C, "penalty": args.penalty, "solver": args.solver, "class_weight": "balanced"},
        "metrics": metrics,
    }
    with open(out_dir / "metrics_and_config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _WSI_SAE_SRC = Path(__file__).resolve().parents[2] / "wsi-sae" / "src"
    if _WSI_SAE_SRC.exists():
        sys.path.insert(0, str(_WSI_SAE_SRC))
        from wsi_sae.cli import main as _wsi_sae_main

        _wsi_sae_main(["probe", *sys.argv[1:]])
    else:
        main()
