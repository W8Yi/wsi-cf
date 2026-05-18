#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.common.paths import DEFAULT_SAE_CFG, DEFAULT_SAE_CKPT, DEFAULT_SAE_VARIANT, SAE_VARIANTS, WSI_CF_ROOT, resolve_sae_paths
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features


DEFAULT_PROJECTS = "TCGA-LUAD,TCGA-LUSC"
DEFAULT_MASTER_LABELS = WSI_CF_ROOT / "resources/labels/master/slide_labels_master.tsv"
DEFAULT_FEATURE_ROOT = Path("/research/projects/mllab/WSI/TCGA_features")
LABEL_COLUMNS = [
    "stage",
    "immune_subtype",
    "os_status",
    "pfs_status",
    "tp53_mutated",
    "kras_mutated",
    "tumor_purity",
    "hpv_status",
    "msi_status",
    "tumor_grade",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank representative tiles for whole TCGA cohorts by raw SAE latent activation. "
            "This mode does not require task-label associations; it reports top tiles per latent and cohort."
        )
    )
    parser.add_argument("--projects", type=str, default=DEFAULT_PROJECTS)
    parser.add_argument("--master-labels", type=Path, default=DEFAULT_MASTER_LABELS)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WSI_CF_ROOT / "artifacts/representative_tiles_luad_lusc_tcga_uni2_sae_relu_v1",
    )
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--sae-ckpt", type=Path, default=None, help=f"Explicit SAE checkpoint override. Defaults to --sae-variant ({DEFAULT_SAE_CKPT}).")
    parser.add_argument("--sae-cfg", type=Path, default=None, help=f"Explicit SAE config override. Defaults to --sae-variant ({DEFAULT_SAE_CFG}).")
    parser.add_argument("--top-tiles-per-latent", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--latent-ids", type=str, default="")
    parser.add_argument("--max-latents", type=int, default=0, help="Debug limiter. 0 means all SAE latents.")
    parser.add_argument("--max-slides-per-cohort", type=int, default=0, help="Debug limiter. 0 means all slides.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def parse_csv_list(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


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


def read_master_rows(path: Path, projects: set[str], features_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            project = str(row.get("project_dir", ""))
            if project not in projects:
                continue
            slide_key = str(row.get("slide_key", ""))
            if not slide_key:
                continue
            h5_path = features_root / project / "features_uni2" / f"{slide_key}.h5"
            item = dict(row)
            item["resolved_h5_path"] = str(h5_path)
            item["resolved_h5_exists"] = str(h5_path.exists())
            rows.append(item)
    rows.sort(key=lambda r: (str(r.get("project_dir", "")), str(r.get("slide_key", ""))))
    return rows


def read_h5_features_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        feats = np.asarray(handle["features"][:], dtype=np.float32)
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if feats.ndim != 2:
        raise ValueError(f"{path}: expected features [N,D] or [1,N,D], got {feats.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path}: expected coords [N,2] or [1,N,2], got {coords.shape}")
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(f"{path}: features N={feats.shape[0]} but coords N={coords.shape[0]}")
    return feats, coords


def parse_latent_ids(value: str, d_latent: int, max_latents: int) -> list[int]:
    if value.strip():
        ids = [int(token) for token in parse_csv_list(value)]
    else:
        ids = list(range(int(d_latent)))
    if int(max_latents) > 0:
        ids = ids[: int(max_latents)]
    bad = [idx for idx in ids if idx < 0 or idx >= int(d_latent)]
    if bad:
        raise ValueError(f"Latent ids out of range for d_latent={d_latent}: {bad[:10]}")
    if not ids:
        raise ValueError("No latent ids selected.")
    return ids


class PerLatentTopK:
    def __init__(self, *, latent_ids: list[int], top_k: int, device: torch.device):
        self.latent_ids = torch.tensor(latent_ids, dtype=torch.long, device=device)
        self.latent_ids_cpu = [int(x) for x in latent_ids]
        self.top_k = int(top_k)
        self.device = device
        n_latents = len(latent_ids)
        self.values = torch.full((self.top_k, n_latents), -torch.inf, dtype=torch.float32, device=device)
        self.slide_row_idx = torch.full((self.top_k, n_latents), -1, dtype=torch.long, device=device)
        self.tile_idx = torch.full((self.top_k, n_latents), -1, dtype=torch.long, device=device)
        self.coord_x = torch.full((self.top_k, n_latents), -1, dtype=torch.long, device=device)
        self.coord_y = torch.full((self.top_k, n_latents), -1, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(
        self,
        z: torch.Tensor,
        *,
        slide_row_idx: int,
        tile_indices: np.ndarray,
        coords: np.ndarray,
    ) -> None:
        z_track = z.index_select(dim=1, index=self.latent_ids)
        k_batch = min(self.top_k, int(z_track.shape[0]))
        if k_batch <= 0:
            return
        batch_vals, batch_pos = torch.topk(z_track.float(), k=k_batch, dim=0, largest=True, sorted=True)
        tile_index_t = torch.as_tensor(tile_indices, dtype=torch.long, device=self.device)
        coord_x_t = torch.as_tensor(coords[:, 0], dtype=torch.long, device=self.device)
        coord_y_t = torch.as_tensor(coords[:, 1], dtype=torch.long, device=self.device)
        batch_tile = tile_index_t.index_select(dim=0, index=batch_pos.reshape(-1)).reshape_as(batch_pos)
        batch_x = coord_x_t.index_select(dim=0, index=batch_pos.reshape(-1)).reshape_as(batch_pos)
        batch_y = coord_y_t.index_select(dim=0, index=batch_pos.reshape(-1)).reshape_as(batch_pos)
        batch_slide = torch.full_like(batch_tile, int(slide_row_idx), dtype=torch.long, device=self.device)

        cand_values = torch.cat([self.values, batch_vals], dim=0)
        cand_slide = torch.cat([self.slide_row_idx, batch_slide], dim=0)
        cand_tile = torch.cat([self.tile_idx, batch_tile], dim=0)
        cand_x = torch.cat([self.coord_x, batch_x], dim=0)
        cand_y = torch.cat([self.coord_y, batch_y], dim=0)

        new_values, new_pos = torch.topk(cand_values, k=self.top_k, dim=0, largest=True, sorted=True)
        self.values = new_values
        self.slide_row_idx = cand_slide.gather(dim=0, index=new_pos)
        self.tile_idx = cand_tile.gather(dim=0, index=new_pos)
        self.coord_x = cand_x.gather(dim=0, index=new_pos)
        self.coord_y = cand_y.gather(dim=0, index=new_pos)

    def to_rows(self, slide_rows: list[dict[str, str]], project: str) -> list[dict[str, Any]]:
        values = self.values.detach().cpu().numpy()
        slide_idx = self.slide_row_idx.detach().cpu().numpy()
        tile_idx = self.tile_idx.detach().cpu().numpy()
        coord_x = self.coord_x.detach().cpu().numpy()
        coord_y = self.coord_y.detach().cpu().numpy()
        rows: list[dict[str, Any]] = []
        for latent_col, latent_idx in enumerate(self.latent_ids_cpu):
            for rank in range(self.top_k):
                activation = float(values[rank, latent_col])
                if not np.isfinite(activation):
                    continue
                source_row_idx = int(slide_idx[rank, latent_col])
                if source_row_idx < 0:
                    continue
                source = slide_rows[source_row_idx]
                row: dict[str, Any] = {
                    "project_dir": project,
                    "latent_idx": int(latent_idx),
                    "tile_rank": int(rank + 1),
                    "activation": activation,
                    "case_id": source.get("case_id", ""),
                    "slide_key": source.get("slide_key", ""),
                    "h5_path": source.get("resolved_h5_path", ""),
                    "tile_index": int(tile_idx[rank, latent_col]),
                    "coord_x": int(coord_x[rank, latent_col]),
                    "coord_y": int(coord_y[rank, latent_col]),
                }
                for key in LABEL_COLUMNS:
                    row[key] = source.get(key, "")
                rows.append(row)
        return rows


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.sae_ckpt, args.sae_cfg = resolve_sae_paths(args.sae_variant, args.sae_ckpt, args.sae_cfg)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.out_dir / "cohort_top_tiles.csv"
    if bool(args.skip_existing) and output_csv.exists() and (args.out_dir / "summary.json").exists():
        print(f"[skip] outputs already exist in {args.out_dir}")
        return

    projects = parse_csv_list(args.projects)
    if not projects:
        raise ValueError("--projects must contain at least one project id")

    sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))
    sae_model.eval()
    latent_ids = parse_latent_ids(args.latent_ids, d_latent=int(d_latent), max_latents=int(args.max_latents))

    master_rows = read_master_rows(args.master_labels, set(projects), args.features_root)
    slide_label_rows: list[dict[str, Any]] = []
    all_top_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:]))),
        "sae_d_in": int(d_in),
        "sae_d_latent": int(d_latent),
        "tracked_latent_count": int(len(latent_ids)),
        "tracked_latent_min": int(min(latent_ids)),
        "tracked_latent_max": int(max(latent_ids)),
        "features_root_used": str(args.features_root),
        "projects": {},
        "skipped_slides": [],
    }

    for project in projects:
        project_rows = [row for row in master_rows if row.get("project_dir") == project]
        project_summary = {
            "slides_in_labels": int(len(project_rows)),
            "slides_with_h5": 0,
            "slides_processed": 0,
            "slides_skipped": 0,
            "tiles_processed": 0,
            "output_csv": str(args.out_dir / project / "cohort_top_tiles.csv"),
        }
        tracker = PerLatentTopK(latent_ids=latent_ids, top_k=int(args.top_tiles_per_latent), device=device)
        processed = 0
        for row_idx, row in enumerate(project_rows):
            h5_path = Path(row["resolved_h5_path"])
            if h5_path.exists():
                project_summary["slides_with_h5"] += 1
            else:
                project_summary["slides_skipped"] += 1
                summary["skipped_slides"].append(
                    {"project_dir": project, "slide_key": row.get("slide_key", ""), "reason": "missing_h5", "h5_path": str(h5_path)}
                )
                continue
            if int(args.max_slides_per_cohort) > 0 and processed >= int(args.max_slides_per_cohort):
                break
            try:
                feats, coords = read_h5_features_coords(h5_path)
            except Exception as exc:
                project_summary["slides_skipped"] += 1
                summary["skipped_slides"].append(
                    {"project_dir": project, "slide_key": row.get("slide_key", ""), "reason": f"read_error: {exc}", "h5_path": str(h5_path)}
                )
                continue
            if feats.shape[1] != int(d_in):
                project_summary["slides_skipped"] += 1
                summary["skipped_slides"].append(
                    {
                        "project_dir": project,
                        "slide_key": row.get("slide_key", ""),
                        "reason": f"feature_dim_{feats.shape[1]}_expected_{d_in}",
                        "h5_path": str(h5_path),
                    }
                )
                continue

            n_tiles = int(feats.shape[0])
            for start in range(0, n_tiles, int(args.batch_size)):
                end = min(start + int(args.batch_size), n_tiles)
                batch = torch.as_tensor(feats[start:end], dtype=torch.float32, device=device)
                z = sae_encode_features(sae_model, batch)
                tracker.update(
                    z,
                    slide_row_idx=row_idx,
                    tile_indices=np.arange(start, end, dtype=np.int64),
                    coords=coords[start:end],
                )
            processed += 1
            project_summary["slides_processed"] += 1
            project_summary["tiles_processed"] += n_tiles
            label_item = {
                "project_dir": project,
                "case_id": row.get("case_id", ""),
                "slide_key": row.get("slide_key", ""),
                "h5_path": str(h5_path),
                "tile_count": n_tiles,
            }
            for key in LABEL_COLUMNS:
                label_item[key] = row.get(key, "")
            slide_label_rows.append(label_item)
            if processed % 25 == 0:
                print(f"[progress] {project}: processed {processed} slides, {project_summary['tiles_processed']} tiles")

        rows = tracker.to_rows(project_rows, project)
        rows.sort(key=lambda r: (str(r["project_dir"]), int(r["latent_idx"]), int(r["tile_rank"])))
        write_csv(args.out_dir / project / "cohort_top_tiles.csv", rows)
        all_top_rows.extend(rows)
        summary["projects"][project] = project_summary
        print(
            f"[ok] {project}: wrote {len(rows)} rows from "
            f"{project_summary['slides_processed']} slides / {project_summary['tiles_processed']} tiles"
        )

    all_top_rows.sort(key=lambda r: (str(r["project_dir"]), int(r["latent_idx"]), int(r["tile_rank"])))
    write_csv(output_csv, all_top_rows)
    positive_rows = [row for row in all_top_rows if float(row["activation"]) > 0.0]
    write_csv(args.out_dir / "cohort_top_tiles_positive.csv", positive_rows)

    latent_activity: dict[tuple[str, int], dict[str, Any]] = {}
    for row in all_top_rows:
        key = (str(row["project_dir"]), int(row["latent_idx"]))
        activation = float(row["activation"])
        item = latent_activity.setdefault(
            key,
            {
                "project_dir": key[0],
                "latent_idx": key[1],
                "max_activation": activation,
                "has_positive_activation": 0,
                "positive_top_tile_count": 0,
                "top_tile_count": 0,
            },
        )
        item["top_tile_count"] += 1
        item["max_activation"] = max(float(item["max_activation"]), activation)
        if activation > 0.0:
            item["has_positive_activation"] = 1
            item["positive_top_tile_count"] += 1
    activity_rows = sorted(latent_activity.values(), key=lambda r: (str(r["project_dir"]), int(r["latent_idx"])))
    write_csv(
        args.out_dir / "latent_activity_summary.csv",
        activity_rows,
        fieldnames=[
            "project_dir",
            "latent_idx",
            "max_activation",
            "has_positive_activation",
            "positive_top_tile_count",
            "top_tile_count",
        ],
    )
    summary["positive_only_outputs"] = {
        "cohort_top_tiles_positive_csv": str(args.out_dir / "cohort_top_tiles_positive.csv"),
        "latent_activity_summary_csv": str(args.out_dir / "latent_activity_summary.csv"),
        "positive_rows": int(len(positive_rows)),
        "active_latents_by_project": {
            project: int(sum(1 for row in activity_rows if row["project_dir"] == project and int(row["has_positive_activation"]) == 1))
            for project in projects
        },
        "dead_latents_by_project": {
            project: int(sum(1 for row in activity_rows if row["project_dir"] == project and int(row["has_positive_activation"]) == 0))
            for project in projects
        },
    }
    write_csv(args.out_dir / "cohort_slide_labels.csv", slide_label_rows)
    write_json(args.out_dir / "summary.json", summary)
    print(f"[ok] wrote combined cohort top tiles to {output_csv}")


if __name__ == "__main__":
    main()
