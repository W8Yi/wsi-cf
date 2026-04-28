#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
            "Counterfactual tile steering on HNSC HPV test split: "
            "select both HPV- and HPV+ slides, steer top-attention tiles toward HPV+ and HPV- prototypes, "
            "and measure MIL prediction changes."
        )
    )
    ap.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.json",
        help="Split JSON containing `test` H5 list.",
    )
    ap.add_argument(
        "--split-tsv",
        type=Path,
        default=REPO_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.tsv",
        help="Split TSV with labels and slide keys.",
    )
    ap.add_argument(
        "--features-root",
        type=Path,
        default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features"),
        help="Canonical features root where files are <slide_key>.h5.",
    )
    ap.add_argument(
        "--mil-ckpt",
        type=Path,
        default=REPO_ROOT / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt",
        help="MIL checkpoint to evaluate.",
    )
    ap.add_argument(
        "--sae-ckpt",
        type=Path,
        default=REPO_ROOT / "runs/tcga_sae_batch_topk_20x_interp/batch_topk_final.pt",
        help="SAE checkpoint used for steering.",
    )
    ap.add_argument(
        "--sae-cfg",
        type=Path,
        default=REPO_ROOT / "runs/tcga_sae_batch_topk_20x_interp/run_config.json",
        help="SAE config JSON.",
    )
    ap.add_argument(
        "--prototype-npz",
        type=Path,
        default=REPO_ROOT
        / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/sae_neuron_pipeline_batch_topk/split_0/prototype_vectors_from_top_neuron_tiles.npz",
        help="Prototype NPZ containing latent_ids + selected_direction + prototype vectors.",
    )
    ap.add_argument(
        "--prototype-key",
        type=str,
        default="prototype_median",
        choices=["prototype_mean", "prototype_median"],
        help="Which prototype vectors to use from NPZ.",
    )
    ap.add_argument(
        "--pos-latent",
        type=int,
        default=2645,
        help="Preferred hpv_pos prototype latent id (fallback auto if missing).",
    )
    ap.add_argument(
        "--neg-latent",
        type=int,
        default=7036,
        help="Preferred hpv_neg prototype latent id (fallback auto if missing).",
    )
    ap.add_argument("--n-neg-slides", type=int, default=3, help="Number of HPV- test slides to evaluate.")
    ap.add_argument("--n-pos-slides", type=int, default=3, help="Number of HPV+ test slides to evaluate.")
    ap.add_argument("--top-tiles-per-slide", type=int, default=2, help="Top attention tiles per selected slide.")
    ap.add_argument(
        "--prototype-strength",
        type=float,
        default=0.8,
        help="Interpolation strength toward prototype vector in [0,1].",
    )
    ap.add_argument("--blend", type=float, default=1.0, help="Feature blend used by edit_uni_z_grid_with_sae.")
    ap.add_argument(
        "--selection-mode",
        type=str,
        default="first",
        choices=["first", "borderline"],
        help=(
            "`first`: pick first N per class by slide_key; "
            "`borderline`: run baseline on full test set and pick slides closest to p=0.5 per class."
        ),
    )
    ap.add_argument("--device", type=str, default="auto", help="cuda:0|cpu|auto")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/counterfactual_tile_steering",
        help="Directory for CSV/JSON outputs.",
    )
    return ap.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _parse_slide_key_from_h5_path(h5_path: str) -> str:
    return Path(h5_path).name.split(".")[0]


def _load_split_json_test_map(split_json: Path) -> dict[str, str]:
    payload = json.loads(split_json.read_text())
    test_list = payload.get("test", [])
    out: dict[str, str] = {}
    for p in test_list:
        key = _parse_slide_key_from_h5_path(str(p))
        out.setdefault(key, str(p))
    return out


def _resolve_h5_path_for_slide(
    *,
    slide_key: str,
    tsv_h5_path: str,
    json_h5_path: str | None,
    features_root: Path,
) -> Path | None:
    candidates = [
        features_root / f"{slide_key}.h5",
        Path(tsv_h5_path) if tsv_h5_path else None,
        Path(json_h5_path) if json_h5_path else None,
    ]
    for c in candidates:
        if c is not None and c.exists():
            return c
    return None


def _load_test_rows(split_json: Path, split_tsv: Path, features_root: Path) -> list[dict[str, Any]]:
    test_map = _load_split_json_test_map(split_json)
    rows: list[dict[str, Any]] = []

    with split_tsv.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if str(r.get("split", "")) != "test":
                continue
            slide_key = str(r.get("slide_key", ""))
            if slide_key not in test_map:
                continue
            label = int(r.get("label", -1))
            if label not in (0, 1):
                continue
            h5_path = _resolve_h5_path_for_slide(
                slide_key=slide_key,
                tsv_h5_path=str(r.get("h5_path", "")),
                json_h5_path=test_map.get(slide_key),
                features_root=features_root,
            )
            if h5_path is None:
                continue
            rows.append(
                {
                    "slide_key": slide_key,
                    "case_id": str(r.get("case_id", slide_key)),
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
        raise RuntimeError(f"No prototype latent found for direction={direction}")
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


def _choose_rows(
    rows: list[dict[str, Any]],
    *,
    n_neg: int,
    n_pos: int,
    selection_mode: str,
    mil_model: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    neg = [r for r in rows if int(r["label"]) == 0]
    pos = [r for r in rows if int(r["label"]) == 1]

    if selection_mode == "first":
        return neg[: int(n_neg)] + pos[: int(n_pos)]

    # borderline mode
    scored: list[dict[str, Any]] = []
    for r in rows:
        x, _ = _read_h5_features_coords(str(r["h5_path"]))
        _, pred, prob = run_mil_attention(mil_model, x, device=device)
        rr = dict(r)
        rr["pred_init"] = int(pred)
        rr["prob_init"] = float(prob)
        rr["margin_to_0p5"] = float(abs(prob - 0.5))
        scored.append(rr)

    neg_sc = sorted([r for r in scored if int(r["label"]) == 0], key=lambda r: r["margin_to_0p5"])
    pos_sc = sorted([r for r in scored if int(r["label"]) == 1], key=lambda r: r["margin_to_0p5"])
    return neg_sc[: int(n_neg)] + pos_sc[: int(n_pos)]


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[(int(r["label"]), str(r["steer_direction"]))].append(r)

    summary: dict[str, Any] = {}
    for (label, direction), vals in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        deltas = np.asarray([float(v["delta_prob_cf_minus_orig"]) for v in vals], dtype=np.float32)
        flips = sum(1 for v in vals if int(v["pred_orig"]) != int(v["pred_cf"]))
        summary[f"label_{label}__{direction}"] = {
            "n": int(len(vals)),
            "mean_delta_prob": float(deltas.mean()) if deltas.size else 0.0,
            "min_delta_prob": float(deltas.min()) if deltas.size else 0.0,
            "max_delta_prob": float(deltas.max()) if deltas.size else 0.0,
            "slide_pred_flips": int(flips),
            "flip_rate": float(flips / len(vals)) if vals else 0.0,
        }
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        args.split_json,
        args.split_tsv,
        args.features_root,
        args.mil_ckpt,
        args.sae_ckpt,
        args.sae_cfg,
        args.prototype_npz,
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))

    print(f"[setup] device={device}")
    print(f"[setup] loading test rows from {args.split_json} + {args.split_tsv}")
    rows = _load_test_rows(args.split_json, args.split_tsv, args.features_root)
    if not rows:
        raise SystemExit("No valid test rows resolved.")
    n0 = sum(1 for r in rows if int(r["label"]) == 0)
    n1 = sum(1 for r in rows if int(r["label"]) == 1)
    print(f"[setup] resolved test rows={len(rows)} (HPV-={n0}, HPV+={n1})")

    print(f"[setup] loading MIL checkpoint {args.mil_ckpt}")
    mil_model = build_mil_from_checkpoint(args.mil_ckpt, device)

    selected_rows = _choose_rows(
        rows,
        n_neg=args.n_neg_slides,
        n_pos=args.n_pos_slides,
        selection_mode=args.selection_mode,
        mil_model=mil_model,
        device=device,
    )
    if not selected_rows:
        raise SystemExit("No slides selected after class-balanced sampling.")
    print(f"[setup] selected slides={len(selected_rows)} mode={args.selection_mode}")
    for r in selected_rows:
        print(f"  slide={r['slide_key']} label={r['label']} h5={r['h5_path']}")

    print(f"[setup] loading SAE {args.sae_ckpt}")
    sae_model, d_in, _ = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))

    print(f"[setup] loading prototypes {args.prototype_npz} ({args.prototype_key})")
    proto_table, dir_table = _load_prototypes(args.prototype_npz, args.prototype_key)
    pos_latent = _pick_latent(dir_table, proto_table, args.pos_latent, "hpv_pos")
    neg_latent = _pick_latent(dir_table, proto_table, args.neg_latent, "hpv_neg")
    proto_pos = np.asarray(proto_table[pos_latent], dtype=np.float32)
    proto_neg = np.asarray(proto_table[neg_latent], dtype=np.float32)
    print(f"[setup] hpv_pos latent={pos_latent} | hpv_neg latent={neg_latent}")

    baseline_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for i, row in enumerate(selected_rows, start=1):
        slide_key = str(row["slide_key"])
        case_id = str(row["case_id"])
        label = int(row["label"])
        x, coords = _read_h5_features_coords(str(row["h5_path"]))
        if x.ndim != 2 or x.shape[1] != d_in:
            print(f"[skip] {slide_key}: feature shape {x.shape} incompatible with SAE d_in={d_in}")
            continue

        attn, pred_orig, prob_orig = run_mil_attention(mil_model, x, device=device)
        baseline_rows.append(
            {
                "slide_key": slide_key,
                "case_id": case_id,
                "label": int(label),
                "pred_orig": int(pred_orig),
                "prob_orig": float(prob_orig),
                "n_tiles": int(x.shape[0]),
            }
        )
        print(f"[slide {i}/{len(selected_rows)}] {slide_key} label={label} pred_orig={pred_orig} prob_orig={prob_orig:.6f}")

        k = min(int(args.top_tiles_per_slide), int(x.shape[0]))
        top_idx = np.argsort(attn)[-k:][::-1].astype(np.int64)

        for rank, tile_idx in enumerate(top_idx.tolist(), start=1):
            x_orig = x[tile_idx]
            _, tile_pred_orig, tile_prob_orig = run_mil_attention(mil_model, x_orig.reshape(1, -1), device=device)

            for steer_direction, proto_vec in (("to_hpv_pos", proto_pos), ("to_hpv_neg", proto_neg)):
                x_cf_tile = _steer_feature(
                    x_orig,
                    sae_model=sae_model,
                    proto_vec=proto_vec,
                    strength=float(args.prototype_strength),
                    blend=float(args.blend),
                    device=device,
                )
                _, tile_pred_cf, tile_prob_cf = run_mil_attention(mil_model, x_cf_tile.reshape(1, -1), device=device)

                x_cf = x.copy()
                x_cf[tile_idx] = x_cf_tile
                _, pred_cf, prob_cf = run_mil_attention(mil_model, x_cf, device=device)

                rec = {
                    "slide_key": slide_key,
                    "case_id": case_id,
                    "label": int(label),
                    "tile_rank_by_attention": int(rank),
                    "tile_index": int(tile_idx),
                    "attention": float(attn[tile_idx]),
                    "coord_x": (int(coords[tile_idx, 0]) if coords is not None else ""),
                    "coord_y": (int(coords[tile_idx, 1]) if coords is not None else ""),
                    "steer_direction": steer_direction,
                    "prototype_strength": float(args.prototype_strength),
                    "prototype_latent_pos": int(pos_latent),
                    "prototype_latent_neg": int(neg_latent),
                    "pred_orig": int(pred_orig),
                    "prob_orig": float(prob_orig),
                    "pred_cf": int(pred_cf),
                    "prob_cf": float(prob_cf),
                    "delta_prob_cf_minus_orig": float(prob_cf - prob_orig),
                    "tile_pred_orig": int(tile_pred_orig),
                    "tile_prob_orig": float(tile_prob_orig),
                    "tile_pred_cf": int(tile_pred_cf),
                    "tile_prob_cf": float(tile_prob_cf),
                }
                results.append(rec)

                print(
                    "  "
                    f"tile_rank={rank} tile_idx={tile_idx} {steer_direction} | "
                    f"orig(y={label}, pred={pred_orig}, p={prob_orig:.6f}) -> "
                    f"new(pred={pred_cf}, p={prob_cf:.6f})"
                )

    summary = _summarize(results)
    print("[summary]")
    print(json.dumps(summary, indent=2))

    baseline_csv = args.out_dir / "baseline_selected_slides.csv"
    results_csv = args.out_dir / "counterfactual_tile_results.csv"
    summary_json = args.out_dir / "counterfactual_summary.json"
    config_json = args.out_dir / "counterfactual_config.json"

    _write_csv(baseline_csv, baseline_rows)
    _write_csv(results_csv, results)
    summary_json.write_text(json.dumps(summary, indent=2))
    config_json.write_text(
        json.dumps(
            {
                "split_json": str(args.split_json),
                "split_tsv": str(args.split_tsv),
                "features_root": str(args.features_root),
                "mil_ckpt": str(args.mil_ckpt),
                "sae_ckpt": str(args.sae_ckpt),
                "sae_cfg": str(args.sae_cfg),
                "prototype_npz": str(args.prototype_npz),
                "prototype_key": str(args.prototype_key),
                "pos_latent": int(pos_latent),
                "neg_latent": int(neg_latent),
                "n_neg_slides": int(args.n_neg_slides),
                "n_pos_slides": int(args.n_pos_slides),
                "top_tiles_per_slide": int(args.top_tiles_per_slide),
                "prototype_strength": float(args.prototype_strength),
                "blend": float(args.blend),
                "selection_mode": str(args.selection_mode),
                "device": str(device),
            },
            indent=2,
        )
    )

    print("[done] wrote:")
    print(f"  {baseline_csv}")
    print(f"  {results_csv}")
    print(f"  {summary_json}")
    print(f"  {config_json}")


if __name__ == "__main__":
    main()

