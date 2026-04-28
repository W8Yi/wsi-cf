#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_steer.run_hnsc_hpv_sae_neuron_pipeline import build_mil_from_checkpoint, run_mil_attention
from utils.sae import load_sae_from_config
from utils.sae_edit import edit_uni_z_grid_with_sae


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Single-tile counterfactual sweep: select one test slide, choose one tile, "
            "predict on that tile, then steer with multiple strengths toward HPV+ / HPV- prototypes."
        )
    )
    ap.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.json",
    )
    ap.add_argument(
        "--split-tsv",
        type=Path,
        default=REPO_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.tsv",
    )
    ap.add_argument(
        "--features-root",
        type=Path,
        default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features"),
        help="Canonical features root where files are <slide_key>.h5",
    )
    ap.add_argument(
        "--mil-ckpt",
        type=Path,
        default=REPO_ROOT / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt",
    )
    ap.add_argument(
        "--sae-ckpt",
        type=Path,
        default=REPO_ROOT / "runs/tcga_sae_batch_topk_20x_interp/batch_topk_final.pt",
    )
    ap.add_argument(
        "--sae-cfg",
        type=Path,
        default=REPO_ROOT / "runs/tcga_sae_batch_topk_20x_interp/run_config.json",
    )
    ap.add_argument(
        "--prototype-npz",
        type=Path,
        default=REPO_ROOT
        / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/sae_neuron_pipeline_batch_topk/split_0/prototype_vectors_from_top_neuron_tiles.npz",
    )
    ap.add_argument(
        "--prototype-key",
        type=str,
        default="prototype_median",
        choices=["prototype_mean", "prototype_median"],
    )

    ap.add_argument(
        "--slide-key",
        type=str,
        default="",
        help="If set, use this exact slide_key from split_0 test set.",
    )
    ap.add_argument(
        "--target-label",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="When --slide-key is not set: filter candidate slides by label (0/1), -1 means any.",
    )
    ap.add_argument(
        "--selection-mode",
        type=str,
        default="borderline",
        choices=["first", "borderline"],
        help="How to choose one slide when --slide-key is not set.",
    )

    ap.add_argument(
        "--tile-index",
        type=int,
        default=-1,
        help="If >=0, use this tile index in the chosen slide.",
    )
    ap.add_argument(
        "--attention-rank",
        type=int,
        default=1,
        help="If --tile-index not set: pick tile by attention rank (1=top attention).",
    )

    ap.add_argument(
        "--strengths",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated steering strengths.",
    )
    ap.add_argument(
        "--directions",
        type=str,
        default="to_hpv_pos,to_hpv_neg",
        help="Comma-separated: to_hpv_pos,to_hpv_neg",
    )
    ap.add_argument("--pos-latent", type=int, default=2645)
    ap.add_argument("--neg-latent", type=int, default=7036)
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument(
        "--identity-at-zero",
        action="store_true",
        help="If set, strength==0 uses exact original feature (no SAE reconstruction drift).",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/single_tile_counterfactual_sweep",
    )
    return ap.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _parse_slide_key_from_h5_path(h5_path: str) -> str:
    return Path(h5_path).name.split(".")[0]


def _parse_float_csv(v: str) -> list[float]:
    out = [float(x.strip()) for x in str(v).split(",") if x.strip()]
    if not out:
        raise ValueError("No values parsed from --strengths")
    return out


def _parse_str_csv(v: str) -> list[str]:
    out = [str(x.strip()) for x in str(v).split(",") if x.strip()]
    if not out:
        raise ValueError("No values parsed from --directions")
    return out


def _load_split_json_test_map(split_json: Path) -> dict[str, str]:
    payload = json.loads(split_json.read_text())
    out: dict[str, str] = {}
    for p in payload.get("test", []):
        k = _parse_slide_key_from_h5_path(str(p))
        out.setdefault(k, str(p))
    return out


def _resolve_h5_for_slide(slide_key: str, tsv_h5: str, json_h5: str | None, features_root: Path) -> Path | None:
    cands = [
        features_root / f"{slide_key}.h5",
        Path(tsv_h5) if tsv_h5 else None,
        Path(json_h5) if json_h5 else None,
    ]
    for c in cands:
        if c is not None and c.exists():
            return c
    return None


def _load_test_rows(split_json: Path, split_tsv: Path, features_root: Path) -> list[dict[str, Any]]:
    test_map = _load_split_json_test_map(split_json)
    rows: list[dict[str, Any]] = []
    with split_tsv.open("r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if str(row.get("split", "")) != "test":
                continue
            slide_key = str(row.get("slide_key", ""))
            if slide_key not in test_map:
                continue
            label = int(row.get("label", -1))
            if label not in (0, 1):
                continue
            h5_path = _resolve_h5_for_slide(
                slide_key=slide_key,
                tsv_h5=str(row.get("h5_path", "")),
                json_h5=test_map.get(slide_key),
                features_root=features_root,
            )
            if h5_path is None:
                continue
            rows.append(
                {
                    "slide_key": slide_key,
                    "case_id": str(row.get("case_id", slide_key)),
                    "label": label,
                    "h5_path": str(h5_path),
                }
            )
    rows.sort(key=lambda x: (int(x["label"]), str(x["slide_key"])))
    return rows


def _read_h5_features_coords(h5_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]
        if feats.ndim == 2:
            x = feats[:]
        elif feats.ndim == 3 and int(feats.shape[0]) == 1:
            x = feats[0]
        else:
            raise RuntimeError(f"{h5_path}: unsupported features shape {tuple(feats.shape)}")

        coords = None
        if "coords" in f:
            c = f["coords"]
            if c.ndim == 2 and c.shape[1] == 2:
                coords = c[:]
            elif c.ndim == 3 and int(c.shape[0]) == 1:
                coords = c[0]
    return np.asarray(x, dtype=np.float32), (np.asarray(coords) if coords is not None else None)


def _load_prototypes(npz_path: Path, key: str) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    with np.load(npz_path, allow_pickle=False) as d:
        latent_ids = np.asarray(d["latent_ids"], dtype=np.int64).reshape(-1)
        vecs = np.asarray(d[key], dtype=np.float32)
        dirs = np.asarray(d["selected_direction"]).astype(str)
    table = {int(latent_ids[i]): vecs[i] for i in range(latent_ids.shape[0])}
    dir_table = {int(latent_ids[i]): str(dirs[i]) for i in range(latent_ids.shape[0])}
    return table, dir_table


def _pick_latent(dir_table: dict[int, str], table: dict[int, np.ndarray], preferred: int, direction: str) -> int:
    if preferred in table:
        return int(preferred)
    cands = sorted([lid for lid, d in dir_table.items() if d == direction])
    if not cands:
        raise RuntimeError(f"No prototype latent for direction={direction}")
    return int(cands[0])


def _steer_feature(
    x_feat: np.ndarray,
    *,
    sae_model: torch.nn.Module,
    proto_vec: np.ndarray,
    strength: float,
    blend: float,
    device: torch.device,
) -> np.ndarray:
    z_grid = torch.from_numpy(np.asarray(x_feat, dtype=np.float32)).to(device=device).view(1, 1, -1)
    z_edit, _ = edit_uni_z_grid_with_sae(
        sae_model=sae_model,
        z_grid=z_grid,
        latent_idx=None,
        target_latent_vector=proto_vec,
        target_latent_vector_strength=float(strength),
        blend=float(blend),
        latent_strength=1.0,
        keep_non_selected=True,
        return_debug=True,
    )
    return z_edit.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def _choose_one_slide(
    rows: list[dict[str, Any]],
    *,
    slide_key: str,
    target_label: int,
    selection_mode: str,
    mil_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    if slide_key:
        for r in rows:
            if str(r["slide_key"]) == str(slide_key):
                return r
        raise RuntimeError(f"--slide-key {slide_key} not found in test rows.")

    cands = rows
    if target_label in (0, 1):
        cands = [r for r in cands if int(r["label"]) == int(target_label)]
    if not cands:
        raise RuntimeError("No candidate slides after label filter.")

    if selection_mode == "first":
        return cands[0]

    # borderline
    scored = []
    for r in cands:
        x, _ = _read_h5_features_coords(str(r["h5_path"]))
        _, pred, prob = run_mil_attention(mil_model, x, device=device)
        rr = dict(r)
        rr["pred_init"] = int(pred)
        rr["prob_init"] = float(prob)
        rr["margin_to_0p5"] = float(abs(prob - 0.5))
        scored.append(rr)
    scored.sort(key=lambda z: z["margin_to_0p5"])
    return scored[0]


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(p) for p in [args.split_json, args.split_tsv, args.features_root, args.mil_ckpt, args.sae_ckpt, args.sae_cfg, args.prototype_npz] if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))

    strengths = _parse_float_csv(args.strengths)
    directions = _parse_str_csv(args.directions)
    valid_dirs = {"to_hpv_pos", "to_hpv_neg"}
    for d in directions:
        if d not in valid_dirs:
            raise ValueError(f"Unsupported direction {d}. Allowed: {sorted(valid_dirs)}")

    print(f"[setup] device={device}")
    rows = _load_test_rows(args.split_json, args.split_tsv, args.features_root)
    n0 = sum(1 for r in rows if int(r["label"]) == 0)
    n1 = sum(1 for r in rows if int(r["label"]) == 1)
    print(f"[setup] test rows={len(rows)} (HPV-={n0}, HPV+={n1})")
    if not rows:
        raise SystemExit("No test rows resolved.")

    print(f"[setup] loading MIL: {args.mil_ckpt}")
    mil_model = build_mil_from_checkpoint(args.mil_ckpt, device)
    print(f"[setup] loading SAE: {args.sae_ckpt}")
    sae_model, d_in, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))

    proto_table, dir_table = _load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = _pick_latent(dir_table, proto_table, args.pos_latent, "hpv_pos")
    neg_latent = _pick_latent(dir_table, proto_table, args.neg_latent, "hpv_neg")
    proto_pos = np.asarray(proto_table[pos_latent], dtype=np.float32)
    proto_neg = np.asarray(proto_table[neg_latent], dtype=np.float32)
    print(f"[setup] prototype latents: hpv_pos={pos_latent}, hpv_neg={neg_latent}")

    row = _choose_one_slide(
        rows,
        slide_key=args.slide_key,
        target_label=args.target_label,
        selection_mode=args.selection_mode,
        mil_model=mil_model,
        device=device,
    )
    slide_key = str(row["slide_key"])
    label = int(row["label"])
    print(f"[slide] chosen: {slide_key} label={label} h5={row['h5_path']}")

    x, coords = _read_h5_features_coords(str(row["h5_path"]))
    if x.ndim != 2 or x.shape[1] != d_in:
        raise RuntimeError(f"Feature shape {x.shape} incompatible with SAE d_in={d_in}")

    attn, pred_slide_orig, prob_slide_orig = run_mil_attention(mil_model, x, device=device)
    print(f"[slide] baseline pred={pred_slide_orig} prob_pos={prob_slide_orig:.6f}")

    if args.tile_index >= 0:
        tile_idx = int(args.tile_index)
        if tile_idx < 0 or tile_idx >= x.shape[0]:
            raise RuntimeError(f"--tile-index {tile_idx} out of range [0, {x.shape[0]-1}]")
        tile_rank = None
    else:
        rank = max(1, int(args.attention_rank))
        order = np.argsort(attn)[::-1]
        if rank > len(order):
            raise RuntimeError(f"--attention-rank {rank} > number of tiles {len(order)}")
        tile_idx = int(order[rank - 1])
        tile_rank = rank

    x_tile = x[tile_idx]
    _, pred_tile_orig, prob_tile_orig = run_mil_attention(mil_model, x_tile.reshape(1, -1), device=device)
    coord_x = int(coords[tile_idx, 0]) if coords is not None else None
    coord_y = int(coords[tile_idx, 1]) if coords is not None else None

    print(
        f"[tile] idx={tile_idx} rank={tile_rank if tile_rank is not None else 'manual'} "
        f"attn={attn[tile_idx]:.6f} coord=({coord_x},{coord_y})"
    )
    print(f"[tile] baseline pred={pred_tile_orig} prob_pos={prob_tile_orig:.6f}")

    rows_out: list[dict[str, Any]] = []
    for direction in directions:
        proto_vec = proto_pos if direction == "to_hpv_pos" else proto_neg
        for s in strengths:
            x_cf_tile = _steer_feature(
                x_tile,
                sae_model=sae_model,
                proto_vec=proto_vec,
                strength=float(s),
                blend=float(args.blend),
                device=device,
            ) if (not args.identity_at_zero or abs(float(s)) > 1e-12) else x_tile.copy()
            _, pred_tile_cf, prob_tile_cf = run_mil_attention(mil_model, x_cf_tile.reshape(1, -1), device=device)

            x_cf = x.copy()
            x_cf[tile_idx] = x_cf_tile
            _, pred_slide_cf, prob_slide_cf = run_mil_attention(mil_model, x_cf, device=device)

            rec = {
                "slide_key": slide_key,
                "case_id": str(row["case_id"]),
                "label": int(label),
                "tile_index": int(tile_idx),
                "tile_attention": float(attn[tile_idx]),
                "coord_x": coord_x if coord_x is not None else "",
                "coord_y": coord_y if coord_y is not None else "",
                "direction": direction,
                "strength": float(s),
                "tile_pred_orig": int(pred_tile_orig),
                "tile_prob_orig": float(prob_tile_orig),
                "tile_pred_cf": int(pred_tile_cf),
                "tile_prob_cf": float(prob_tile_cf),
                "tile_delta_prob_cf_minus_orig": float(prob_tile_cf - prob_tile_orig),
                "slide_pred_orig": int(pred_slide_orig),
                "slide_prob_orig": float(prob_slide_orig),
                "slide_pred_cf": int(pred_slide_cf),
                "slide_prob_cf": float(prob_slide_cf),
                "slide_delta_prob_cf_minus_orig": float(prob_slide_cf - prob_slide_orig),
                "prototype_latent_pos": int(pos_latent),
                "prototype_latent_neg": int(neg_latent),
            }
            rows_out.append(rec)

            print(
                f"[{direction} s={s:.3f}] "
                f"tile: {pred_tile_orig}->{pred_tile_cf}, p {prob_tile_orig:.6f}->{prob_tile_cf:.6f} | "
                f"slide: {pred_slide_orig}->{pred_slide_cf}, p {prob_slide_orig:.6f}->{prob_slide_cf:.6f}"
            )

    csv_path = args.out_dir / "single_tile_strength_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        if rows_out:
            w.writeheader()
            w.writerows(rows_out)

    summary = {
        "slide_key": slide_key,
        "label": int(label),
        "tile_index": int(tile_idx),
        "tile_attention": float(attn[tile_idx]),
        "tile_coord": [coord_x, coord_y] if coord_x is not None and coord_y is not None else None,
        "n_rows": int(len(rows_out)),
        "strengths": [float(s) for s in strengths],
        "directions": directions,
        "prototype_latent_pos": int(pos_latent),
        "prototype_latent_neg": int(neg_latent),
        "device": str(device),
    }
    summary_path = args.out_dir / "single_tile_strength_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[done] wrote:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
