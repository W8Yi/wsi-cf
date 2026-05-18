#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.common.paths import DEFAULT_SAE_CFG, DEFAULT_SAE_CKPT, DEFAULT_SAE_VARIANT, SAE_VARIANTS, resolve_sae_paths
from wsi_cf.data.region_bank import parse_region_bank_csv
from wsi_cf.steering.progressive import load_progressive_edit_manifest
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_decode_latents, sae_encode_features


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct task concept steering before diffusion and compare SAE latent edits "
            "against decoded UNI feature grids. This is a diagnostic for checking whether "
            "different concepts produce different PixCell conditioning features."
        )
    )
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--edit-manifest", type=Path, required=True)
    parser.add_argument("--concepts-json", type=Path, required=True)
    parser.add_argument("--representative-tiles-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--concept-class-label", type=str, default="")
    parser.add_argument("--concept-ranking-method", type=str, default="attention_weighted")
    parser.add_argument("--concept-target-stat", type=str, default="median", choices=["median", "mean", "q75", "max"])
    parser.add_argument("--concept-target-top-k", type=int, default=5, help="Use only the top K representative tiles per concept to estimate target activation; 0 uses all rows.")
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--max-concepts", type=int, default=0)
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--sae-ckpt", type=Path, default=None, help=f"Explicit SAE checkpoint override. Defaults to --sae-variant ({DEFAULT_SAE_CKPT}).")
    parser.add_argument("--sae-cfg", type=Path, default=None, help=f"Explicit SAE config override. Defaults to --sae-variant ({DEFAULT_SAE_CFG}).")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def load_concepts(
    *,
    concepts_json: Path,
    representative_tiles_csv: Path,
    class_label: str,
    ranking_method: str,
    target_stat: str,
    target_top_k: int,
    max_concepts: int,
) -> tuple[list[dict[str, object]], dict[int, float]]:
    payload = json.loads(concepts_json.read_text())
    concepts = list(payload.get("concepts", []))
    if class_label:
        concepts = [row for row in concepts if str(row.get("class_label", "")) == str(class_label)]
    concepts.sort(key=lambda row: (int(row.get("concept_rank", 10**9)), -float(row.get("final_score", 0.0)), int(row["latent_idx"])))
    if int(max_concepts) > 0:
        concepts = concepts[: int(max_concepts)]
    if not concepts:
        raise ValueError(f"No concepts found in {concepts_json} for class_label={class_label!r}")

    latent_ids = {int(row["latent_idx"]) for row in concepts}
    values_by_latent: dict[int, list[float]] = {latent: [] for latent in latent_ids}
    rows_by_latent: dict[int, list[dict[str, str]]] = {latent: [] for latent in latent_ids}
    for row in read_csv_rows(representative_tiles_csv):
        latent = int(row.get("latent_idx", -1))
        if latent not in rows_by_latent:
            continue
        if str(row.get("ranking_method", "")) != str(ranking_method):
            continue
        rows_by_latent[latent].append(row)
    for latent, rows in rows_by_latent.items():
        rows.sort(key=lambda row: int(row.get("tile_rank", 10**9)))
        if int(target_top_k) > 0:
            rows = rows[: int(target_top_k)]
        for row in rows:
            activation = str(row.get("activation", "")).strip()
            if activation:
                values_by_latent[latent].append(float(activation))

    targets: dict[int, float] = {}
    for concept in concepts:
        latent = int(concept["latent_idx"])
        vals = np.asarray(values_by_latent.get(latent, []), dtype=np.float32)
        if vals.size:
            if target_stat == "median":
                target = float(np.median(vals))
            elif target_stat == "mean":
                target = float(np.mean(vals))
            elif target_stat == "q75":
                target = float(np.percentile(vals, 75.0))
            else:
                target = float(np.max(vals))
        else:
            target = float(concept.get("mean_class", concept.get("top_activation", 1.0)))
        targets[latent] = target
    return concepts, targets


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    denom = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    x = x / denom
    return x @ x.T


def save_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["concept"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [float(v) for v in row])


def save_heatmap(path: Path, matrix: np.ndarray, labels: list[str], title: str, vmin=None, vmax=None, cmap="viridis") -> None:
    fig_w = max(6.0, 0.55 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w), dpi=180)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    args.sae_ckpt, args.sae_cfg = resolve_sae_paths(args.sae_variant, args.sae_ckpt, args.sae_cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requests = load_progressive_edit_manifest(args.edit_manifest)
    if args.run_id:
        request = next((item for item in requests if item.run_id == args.run_id), None)
        if request is None:
            raise ValueError(f"run_id={args.run_id!r} not found in {args.edit_manifest}")
    else:
        request = requests[int(args.request_index)]

    rows = parse_region_bank_csv(args.region_bank_csv)
    row_by_id = {row.region_id: row for row in rows}
    region = row_by_id.get(request.region_id)
    if region is None:
        raise ValueError(f"region_id={request.region_id!r} not found in {args.region_bank_csv}")

    z_grid_np = np.asarray(np.load(region.feature_grid_path), dtype=np.float32)
    gh, gw, d = z_grid_np.shape
    selected_cells = [(int(gx), int(gy)) for gx, gy in request.target_cells]
    selected_flat = np.asarray([gy * gw + gx for gx, gy in selected_cells], dtype=np.int64)
    mask_np = np.zeros((gh, gw), dtype=np.float32)
    for gx, gy in selected_cells:
        mask_np[gy, gx] = 1.0

    concepts, target_values = load_concepts(
        concepts_json=args.concepts_json,
        representative_tiles_csv=args.representative_tiles_csv,
        class_label=args.concept_class_label,
        ranking_method=args.concept_ranking_method,
        target_stat=args.concept_target_stat,
        target_top_k=int(args.concept_target_top_k),
        max_concepts=int(args.max_concepts),
    )

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    sae_model, _, latent_dim = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))

    x = torch.from_numpy(z_grid_np.reshape(gh * gw, d)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        z_lat = sae_encode_features(sae_model, x)

    summary_rows: list[dict[str, object]] = []
    mean_delta_vectors: list[np.ndarray] = []
    mean_edited_vectors: list[np.ndarray] = []
    labels: list[str] = []
    delta_grids: list[np.ndarray] = []

    for concept in concepts:
        rank = int(concept.get("concept_rank", len(labels) + 1))
        latent = int(concept["latent_idx"])
        target = float(target_values[latent])
        label = f"r{rank:02d}_z{latent}"
        labels.append(label)

        z_edit = z_lat.clone()
        sel_t = torch.as_tensor(selected_flat, device=device, dtype=torch.long)
        cur = z_edit[sel_t, latent]
        tgt = torch.full_like(cur, target)
        z_edit[sel_t, latent] = (1.0 - float(args.prototype_strength)) * cur + float(args.prototype_strength) * tgt
        with torch.no_grad():
            x_rec = sae_decode_latents(sae_model, z_edit)

        w = torch.from_numpy(mask_np.reshape(gh * gw, 1)).to(device=device, dtype=torch.float32)
        x_new = x * (1.0 - float(args.steer_blend) * w) + x_rec * (float(args.steer_blend) * w)
        delta = (x_new - x).detach().cpu().numpy().reshape(gh, gw, d).astype(np.float32)
        edited = x_new.detach().cpu().numpy().reshape(gh, gw, d).astype(np.float32)
        z_before_sel = z_lat[sel_t].detach().cpu().numpy().astype(np.float32)
        z_after_sel = z_edit[sel_t].detach().cpu().numpy().astype(np.float32)
        selected_delta = delta.reshape(gh * gw, d)[selected_flat]
        selected_edited = edited.reshape(gh * gw, d)[selected_flat]

        concept_dir = args.out_dir / label
        concept_dir.mkdir(parents=True, exist_ok=True)
        np.save(concept_dir / "concept_uni_feature_grid.npy", edited)
        np.save(concept_dir / "concept_uni_delta_grid.npy", delta)
        np.save(concept_dir / "concept_uni_delta_l2_grid.npy", np.linalg.norm(delta, axis=-1).astype(np.float32))
        np.save(concept_dir / "selected_concept_uni_features.npy", selected_edited.astype(np.float32))
        np.save(concept_dir / "selected_concept_uni_delta.npy", selected_delta.astype(np.float32))
        np.savez_compressed(
            concept_dir / "selected_sae_latents_before_after.npz",
            before=z_before_sel,
            after=z_after_sel,
            selected_cells=np.asarray(selected_cells, dtype=np.int64),
            latent_idx=np.asarray([latent], dtype=np.int64),
            target_value=np.asarray([target], dtype=np.float32),
        )

        delta_l2_grid = np.linalg.norm(delta, axis=-1)
        delta_grids.append(delta_l2_grid)
        mean_delta_vectors.append(selected_delta.mean(axis=0))
        mean_edited_vectors.append(selected_edited.mean(axis=0))

        fig, ax = plt.subplots(figsize=(4, 4), dpi=180)
        im = ax.imshow(delta_l2_grid, cmap="magma")
        ax.set_title(f"UNI delta L2: {label}")
        ax.set_xticks(range(gw))
        ax.set_yticks(range(gh))
        for gx, gy in selected_cells:
            ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=2))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(concept_dir / "uni_delta_l2_heatmap.png")
        plt.close(fig)

        summary_rows.append(
            {
                "concept_label": label,
                "concept_rank": rank,
                "latent_idx": latent,
                "target_value": target,
                "selected_cells": json.dumps([{"gx": gx, "gy": gy} for gx, gy in selected_cells]),
                "selected_sae_before_mean": float(z_before_sel[:, latent].mean()),
                "selected_sae_before_max": float(z_before_sel[:, latent].max()),
                "selected_sae_after_mean": float(z_after_sel[:, latent].mean()),
                "selected_sae_after_max": float(z_after_sel[:, latent].max()),
                "selected_sae_delta_mean": float((z_after_sel[:, latent] - z_before_sel[:, latent]).mean()),
                "selected_uni_delta_l2_mean": float(np.linalg.norm(selected_delta, axis=1).mean()),
                "selected_uni_delta_l2_max": float(np.linalg.norm(selected_delta, axis=1).max()),
                "full_grid_uni_delta_l2_mean": float(delta_l2_grid.mean()),
                "full_grid_uni_delta_l2_max": float(delta_l2_grid.max()),
                "concept_uni_feature_grid": str(concept_dir / "concept_uni_feature_grid.npy"),
                "concept_uni_delta_grid": str(concept_dir / "concept_uni_delta_grid.npy"),
                "selected_sae_latents_before_after": str(concept_dir / "selected_sae_latents_before_after.npz"),
            }
        )

    summary_csv = args.out_dir / "concept_uni_feature_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    delta_vectors = np.stack(mean_delta_vectors, axis=0)
    edited_vectors = np.stack(mean_edited_vectors, axis=0)
    cos = cosine_matrix(delta_vectors)
    l2 = np.linalg.norm(delta_vectors[:, None, :] - delta_vectors[None, :, :], axis=-1)
    save_matrix_csv(args.out_dir / "pairwise_selected_uni_delta_cosine.csv", labels, cos)
    save_matrix_csv(args.out_dir / "pairwise_selected_uni_delta_l2.csv", labels, l2)
    save_heatmap(args.out_dir / "pairwise_selected_uni_delta_cosine.png", cos, labels, "Pairwise cosine of mean selected UNI deltas", vmin=-1, vmax=1, cmap="coolwarm")
    save_heatmap(args.out_dir / "pairwise_selected_uni_delta_l2.png", l2, labels, "Pairwise L2 of mean selected UNI deltas", cmap="viridis")

    coords = pca_2d(np.concatenate([edited_vectors, delta_vectors], axis=0))
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ax.scatter(coords[: len(labels), 0], coords[: len(labels), 1], label="edited selected UNI mean", s=50)
    ax.scatter(coords[len(labels) :, 0], coords[len(labels) :, 1], label="UNI delta mean", s=50, marker="x")
    for idx, label in enumerate(labels):
        ax.text(coords[idx, 0], coords[idx, 1], label, fontsize=7)
    ax.set_title("Per-concept decoded UNI features are separable")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "concept_uni_feature_pca.png")
    plt.close(fig)

    n = len(labels)
    cols = min(5, n)
    rows_n = int(np.ceil(n / cols))
    vmax = max(float(np.max(g)) for g in delta_grids)
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.0 * cols, 3.0 * rows_n), dpi=180)
    axes_arr = np.asarray(axes).reshape(-1)
    for idx, (label, grid) in enumerate(zip(labels, delta_grids)):
        ax = axes_arr[idx]
        im = ax.imshow(grid, cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for gx, gy in selected_cells:
            ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=1.5))
    for ax in axes_arr[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes_arr[:n].tolist(), fraction=0.02, pad=0.02)
    fig.savefig(args.out_dir / "all_concepts_uni_delta_l2_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    write_json(
        args.out_dir / "analysis_manifest.json",
        {
            "region_id": request.region_id,
            "run_id": request.run_id,
            "region_feature_grid_path": region.feature_grid_path,
            "grid_shape": [gh, gw, d],
            "selected_cells": [{"gx": gx, "gy": gy} for gx, gy in selected_cells],
            "concepts_json": str(args.concepts_json),
            "representative_tiles_csv": str(args.representative_tiles_csv),
            "concept_class_label": args.concept_class_label,
            "concept_target_stat": args.concept_target_stat,
            "concept_target_top_k": int(args.concept_target_top_k),
            "concept_ranking_method": args.concept_ranking_method,
            "prototype_strength": float(args.prototype_strength),
            "steer_blend": float(args.steer_blend),
            "latent_dim": int(latent_dim),
            "summary_csv": str(summary_csv),
            "outputs": {
                "pairwise_selected_uni_delta_cosine": str(args.out_dir / "pairwise_selected_uni_delta_cosine.csv"),
                "pairwise_selected_uni_delta_l2": str(args.out_dir / "pairwise_selected_uni_delta_l2.csv"),
                "concept_uni_feature_pca": str(args.out_dir / "concept_uni_feature_pca.png"),
                "all_concepts_uni_delta_l2_heatmaps": str(args.out_dir / "all_concepts_uni_delta_l2_heatmaps.png"),
            },
        },
    )
    print(f"[ok] wrote concept UNI feature diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
