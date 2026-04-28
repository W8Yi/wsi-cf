#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
VENDORED_SAE_ROOT = WSI_CF_ROOT / "third_party" / "SAE_path"
EXTERNAL_SAE_ROOT = Path("/common/users/wq50/SAE_path")
SAE_ROOT = VENDORED_SAE_ROOT if (VENDORED_SAE_ROOT / "utils").exists() else EXTERNAL_SAE_ROOT
if str(SAE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAE_ROOT))

from utils.sae import load_sae_from_config, sae_encode_features  # type: ignore


UNKNOWN_VALUES = {"", "Unknown", "unknown", "NA", "N/A", "nan", "None", "missing"}


TASK_PRESETS: dict[str, dict[str, object]] = {
    "hnsc_hpv": {
        "target_file": "hpv_status_case.tsv",
        "label_column": "hpv_status",
        "kind": "categorical",
        "projects": ["TCGA-HNSC"],
        "classes": ["HPV-", "HPV+"],
    },
    "cesc_hpv": {
        "target_file": "hpv_status_case.tsv",
        "label_column": "hpv_status_from_viral_threshold6",
        "kind": "categorical",
        "projects": ["TCGA-CESC"],
        "classes": ["HPV-", "HPV+"],
    },
    "msi_coad_stad": {
        "target_file": "msi_status_case.tsv",
        "label_column": "msi_status",
        "kind": "categorical",
        "projects": ["TCGA-COAD", "TCGA-STAD"],
    },
    "kirc_grade": {
        "target_file": "tumor_grade_case.tsv",
        "label_column": "tumor_grade",
        "kind": "categorical",
        "projects": ["TCGA-KIRC"],
    },
    "immune_subtype": {
        "target_file": "immune_subtype_case.tsv",
        "label_column": "immune_subtype",
        "kind": "categorical",
        "projects": [],
    },
    "brca_pam50": {
        "target_file": "pam50_case.tsv",
        "label_column": "pam50_subtype",
        "kind": "categorical",
        "projects": ["TCGA-BRCA", "TCGA-BRCA_IDC"],
    },
    "tumor_purity": {
        "target_file": "tumor_purity_case.tsv",
        "label_column": "tumor_purity",
        "kind": "continuous",
        "projects": [],
    },
    "tp53_mutation": {
        "target_file": "mutation_tp53_kras_case.tsv",
        "label_column": "tp53_mutated",
        "kind": "categorical",
        "projects": [],
        "classes": ["0", "1"],
    },
    "kras_mutation": {
        "target_file": "mutation_tp53_kras_case.tsv",
        "label_column": "kras_mutated",
        "kind": "categorical",
        "projects": [],
        "classes": ["0", "1"],
    },
}

TASK_GROUPS = {
    "paper": ["hnsc_hpv", "cesc_hpv", "msi_coad_stad", "kirc_grade"],
    "all": list(TASK_PRESETS),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def parse_projects(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple)):
        return {str(x) for x in value if str(x)}
    return {x.strip() for x in str(value).split(",") if x.strip()}


def normalize_class_value(value: str) -> str | None:
    value = str(value).strip()
    if value in UNKNOWN_VALUES:
        return None
    return value


def parse_float(value: str) -> float | None:
    value = str(value).strip()
    if value in UNKNOWN_VALUES:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def read_h5_features(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{path}: missing dataset 'features'")
        data = handle["features"]
        if data.ndim == 3 and data.shape[0] == 1:
            arr = data[0]
        elif data.ndim == 2:
            arr = data[:]
        else:
            raise ValueError(f"{path}: unsupported features shape {tuple(data.shape)}")
    return np.asarray(arr, dtype=np.float32)


def resolve_h5_path(
    *,
    slide_row: dict[str, str],
    features_root: Path,
    feature_subdir: str,
    use_master_h5: bool,
) -> Path | None:
    if use_master_h5:
        raw = slide_row.get("h5_path", "")
        if raw:
            p = Path(raw)
            if p.exists():
                return p
    project = slide_row.get("project_dir") or slide_row.get("gdc_project_id")
    slide_key = slide_row.get("slide_key", "")
    if not project or not slide_key:
        return None
    p = features_root / project / feature_subdir / f"{slide_key}.h5"
    if p.exists():
        return p
    return None


def encode_slide_summary(
    *,
    h5_path: Path,
    sae_model: torch.nn.Module,
    d_in: int,
    d_latent: int,
    device: torch.device,
    batch_size: int,
    max_tiles: int,
    seed: int,
) -> dict[str, np.ndarray | int | float]:
    x = read_h5_features(h5_path)
    if x.ndim != 2 or x.shape[1] != d_in:
        raise RuntimeError(f"{h5_path}: feature shape {tuple(x.shape)} incompatible with SAE d_in={d_in}")
    n_tiles_total = int(x.shape[0])
    if max_tiles > 0 and n_tiles_total > max_tiles:
        rng = np.random.default_rng(seed + int(stable_hash(str(h5_path), 8), 16))
        idx = np.sort(rng.choice(n_tiles_total, size=int(max_tiles), replace=False))
        x = x[idx]

    sum_pos = np.zeros((d_latent,), dtype=np.float64)
    sum_bin = np.zeros((d_latent,), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device=device, dtype=torch.float32)
            z = sae_encode_features(sae_model, xb).float().detach().cpu().numpy()
            z_pos = np.maximum(z, 0.0)
            sum_pos += z_pos.sum(axis=0, dtype=np.float64)
            sum_bin += (z_pos > 0).sum(axis=0, dtype=np.float64)

    total_mass = float(sum_pos.sum())
    fraction = (sum_pos / total_mass) if total_mass > 0 else np.zeros((d_latent,), dtype=np.float64)
    mean_activation = sum_pos / max(1, int(x.shape[0]))
    prevalence = sum_bin / max(1, int(x.shape[0]))
    return {
        "n_tiles_total": n_tiles_total,
        "n_tiles_used": int(x.shape[0]),
        "total_mass": total_mass,
        "fraction": fraction.astype(np.float32),
        "mean_activation": mean_activation.astype(np.float32),
        "prevalence": prevalence.astype(np.float32),
    }


def load_or_compute_slide_summary(
    *,
    cache_dir: Path,
    slide_key: str,
    h5_path: Path,
    sae_model: torch.nn.Module,
    d_in: int,
    d_latent: int,
    device: torch.device,
    batch_size: int,
    max_tiles: int,
    seed: int,
    reuse_cache: bool,
) -> dict[str, np.ndarray | int | float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = stable_hash(f"{slide_key}|{h5_path}|max_tiles={max_tiles}|d={d_latent}")
    cache_path = cache_dir / f"{slide_key}__{cache_key}.npz"
    if reuse_cache and cache_path.exists():
        data = np.load(cache_path)
        return {
            "n_tiles_total": int(data["n_tiles_total"]),
            "n_tiles_used": int(data["n_tiles_used"]),
            "total_mass": float(data["total_mass"]),
            "fraction": data["fraction"].astype(np.float32),
            "mean_activation": data["mean_activation"].astype(np.float32),
            "prevalence": data["prevalence"].astype(np.float32),
        }
    summary = encode_slide_summary(
        h5_path=h5_path,
        sae_model=sae_model,
        d_in=d_in,
        d_latent=d_latent,
        device=device,
        batch_size=batch_size,
        max_tiles=max_tiles,
        seed=seed,
    )
    np.savez_compressed(
        cache_path,
        n_tiles_total=np.asarray(summary["n_tiles_total"], dtype=np.int64),
        n_tiles_used=np.asarray(summary["n_tiles_used"], dtype=np.int64),
        total_mass=np.asarray(summary["total_mass"], dtype=np.float64),
        fraction=np.asarray(summary["fraction"], dtype=np.float32),
        mean_activation=np.asarray(summary["mean_activation"], dtype=np.float32),
        prevalence=np.asarray(summary["prevalence"], dtype=np.float32),
    )
    return summary


def load_task_rows(
    *,
    target_path: Path,
    label_column: str,
    kind: str,
    projects: set[str],
    slide_master: list[dict[str, str]],
    features_root: Path,
    feature_subdir: str,
    use_master_h5: bool,
    classes: list[str] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    label_rows = read_tsv(target_path)
    labels_by_case: dict[str, str | float] = {}
    raw_counts: Counter[str] = Counter()
    for row in label_rows:
        case_id = row.get("case_id", "")
        project = row.get("project_dir", "")
        if projects and project not in projects:
            continue
        if kind == "continuous":
            parsed = parse_float(row.get(label_column, ""))
        else:
            parsed = normalize_class_value(row.get(label_column, ""))
            if parsed is not None and classes and parsed not in classes:
                parsed = None
        if parsed is None:
            continue
        labels_by_case[case_id] = parsed
        raw_counts[str(parsed)] += 1

    slide_rows: list[dict[str, object]] = []
    missing_h5 = 0
    for row in slide_master:
        case_id = row.get("case_id", "")
        project = row.get("project_dir", "")
        if case_id not in labels_by_case:
            continue
        if projects and project not in projects:
            continue
        h5_path = resolve_h5_path(
            slide_row=row,
            features_root=features_root,
            feature_subdir=feature_subdir,
            use_master_h5=use_master_h5,
        )
        if h5_path is None:
            missing_h5 += 1
            continue
        slide_rows.append(
            {
                "case_id": case_id,
                "slide_key": row.get("slide_key", ""),
                "project_dir": project,
                "label": labels_by_case[case_id],
                "h5_path": str(h5_path),
            }
        )

    summary = {
        "target_path": str(target_path),
        "label_column": label_column,
        "kind": kind,
        "projects": sorted(projects),
        "case_label_count": len(labels_by_case),
        "case_label_counts": dict(sorted(raw_counts.items())),
        "slide_count_with_features": len(slide_rows),
        "missing_feature_slides": int(missing_h5),
    }
    return slide_rows, summary


def aggregate_case_summaries(
    *,
    slide_rows: list[dict[str, object]],
    slide_vectors: dict[str, dict[str, np.ndarray | int | float]],
    d_latent: int,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in slide_rows:
        by_case[str(row["case_id"])].append(row)

    case_rows: list[dict[str, object]] = []
    fractions: list[np.ndarray] = []
    mean_acts: list[np.ndarray] = []
    prevalences: list[np.ndarray] = []
    for case_id, rows in sorted(by_case.items()):
        frac = np.zeros((d_latent,), dtype=np.float64)
        mean_act = np.zeros((d_latent,), dtype=np.float64)
        prev = np.zeros((d_latent,), dtype=np.float64)
        n = 0
        n_tiles = 0
        slide_keys: list[str] = []
        h5_paths: list[str] = []
        for row in rows:
            slide_key = str(row["slide_key"])
            if slide_key not in slide_vectors:
                continue
            summary = slide_vectors[slide_key]
            frac += np.asarray(summary["fraction"], dtype=np.float64)
            mean_act += np.asarray(summary["mean_activation"], dtype=np.float64)
            prev += np.asarray(summary["prevalence"], dtype=np.float64)
            n += 1
            n_tiles += int(summary["n_tiles_used"])
            slide_keys.append(slide_key)
            h5_paths.append(str(row["h5_path"]))
        if n == 0:
            continue
        fractions.append((frac / n).astype(np.float32))
        mean_acts.append((mean_act / n).astype(np.float32))
        prevalences.append((prev / n).astype(np.float32))
        first = rows[0]
        case_rows.append(
            {
                "case_id": case_id,
                "project_dir": first["project_dir"],
                "label": first["label"],
                "n_slides": n,
                "n_tiles_used": n_tiles,
                "slide_keys": ";".join(slide_keys),
                "h5_paths": ";".join(h5_paths),
            }
        )
    if not fractions:
        raise RuntimeError("No case summaries could be aggregated.")
    return case_rows, np.stack(fractions), np.stack(mean_acts), np.stack(prevalences)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    pooled = ((x.size - 1) * vx + (y.size - 1) * vy) / max(1, x.size + y.size - 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled))


def categorical_associations(
    *,
    case_rows: list[dict[str, object]],
    matrix: np.ndarray,
    metric_name: str,
    min_cases_per_class: int,
) -> list[dict[str, object]]:
    labels = np.asarray([str(r["label"]) for r in case_rows], dtype=object)
    classes = sorted(set(labels.tolist()))
    rows: list[dict[str, object]] = []
    d_latent = int(matrix.shape[1])
    for cls in classes:
        mask = labels == cls
        rest = ~mask
        if int(mask.sum()) < min_cases_per_class or int(rest.sum()) < min_cases_per_class:
            continue
        x = matrix[mask]
        y = matrix[rest]
        mean_cls = x.mean(axis=0)
        mean_rest = y.mean(axis=0)
        diff = mean_cls - mean_rest
        for j in range(d_latent):
            rows.append(
                {
                    "latent_idx": j,
                    "metric": metric_name,
                    "class_label": cls,
                    "n_class": int(mask.sum()),
                    "n_rest": int(rest.sum()),
                    "mean_class": float(mean_cls[j]),
                    "mean_rest": float(mean_rest[j]),
                    "diff_class_minus_rest": float(diff[j]),
                    "abs_diff": float(abs(diff[j])),
                    "cohen_d": cohen_d(x[:, j], y[:, j]),
                }
            )
    rows.sort(key=lambda r: (str(r["class_label"]), -abs(float(r["cohen_d"])), -float(r["abs_diff"]), int(r["latent_idx"])))
    return rows


def continuous_associations(
    *,
    case_rows: list[dict[str, object]],
    matrix: np.ndarray,
    metric_name: str,
) -> list[dict[str, object]]:
    y = np.asarray([float(r["label"]) for r in case_rows], dtype=np.float64)
    y_center = y - y.mean()
    y_norm = float(np.linalg.norm(y_center))
    rows: list[dict[str, object]] = []
    for j in range(int(matrix.shape[1])):
        x = matrix[:, j].astype(np.float64)
        x_center = x - x.mean()
        denom = float(np.linalg.norm(x_center) * y_norm)
        corr = float(np.dot(x_center, y_center) / denom) if denom > 0 else 0.0
        rows.append(
            {
                "latent_idx": j,
                "metric": metric_name,
                "n_cases": int(len(case_rows)),
                "pearson_r": corr,
                "abs_pearson_r": abs(corr),
                "mean_metric": float(x.mean()),
                "std_metric": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            }
        )
    rows.sort(key=lambda r: (-float(r["abs_pearson_r"]), int(r["latent_idx"])))
    return rows


def top_case_rows(
    *,
    case_rows: list[dict[str, object]],
    matrix: np.ndarray,
    association_rows: list[dict[str, object]],
    max_latents: int,
    top_cases: int,
    metric_name: str,
) -> list[dict[str, object]]:
    selected_latents: list[int] = []
    seen: set[int] = set()
    for row in association_rows:
        latent = int(row["latent_idx"])
        if latent in seen:
            continue
        selected_latents.append(latent)
        seen.add(latent)
        if len(selected_latents) >= max_latents:
            break
    out: list[dict[str, object]] = []
    for latent in selected_latents:
        order = np.argsort(-matrix[:, latent])[:top_cases]
        for rank, idx in enumerate(order.tolist(), start=1):
            row = case_rows[idx]
            out.append(
                {
                    "latent_idx": latent,
                    "metric": metric_name,
                    "case_rank": rank,
                    "case_id": row["case_id"],
                    "project_dir": row["project_dir"],
                    "label": row["label"],
                    "value": float(matrix[idx, latent]),
                    "n_slides": row["n_slides"],
                    "n_tiles_used": row["n_tiles_used"],
                    "slide_keys": row["slide_keys"],
                }
            )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare task-agnostic SAE concept-label association tables.")
    parser.add_argument("--tasks", type=str, default="paper", help="Comma list of presets or groups: paper, all, hnsc_hpv, msi_coad_stad, ...")
    parser.add_argument("--labels-root", type=Path, default=SAE_ROOT / "metadata/labels")
    parser.add_argument("--slide-master", type=Path, default=SAE_ROOT / "metadata/labels/master/slide_labels_master.tsv")
    parser.add_argument("--features-root", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features"))
    parser.add_argument("--feature-subdir", type=str, default="features_uni2")
    parser.add_argument("--use-master-h5", action="store_true", help="Prefer h5_path from slide_labels_master when it exists")
    parser.add_argument("--sae-ckpt", type=Path, default=SAE_ROOT / "runs/relu_sae_base/relu_final.pt")
    parser.add_argument("--sae-cfg", type=Path, default=SAE_ROOT / "runs/relu_sae_base/run_config.json")
    parser.add_argument("--out-dir", type=Path, default=Path("/common/users/wq50/wsi_cf/artifacts/concept_label_associations"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-tiles-per-slide", type=int, default=0)
    parser.add_argument("--max-cases-per-task", type=int, default=0)
    parser.add_argument("--min-cases-per-class", type=int, default=10)
    parser.add_argument("--top-latents", type=int, default=50)
    parser.add_argument("--top-cases-per-latent", type=int, default=10)
    parser.add_argument("--reuse-slide-cache", action="store_true", default=True)
    parser.add_argument("--no-reuse-slide-cache", dest="reuse_slide_cache", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Only join labels to available feature files and write cohort manifests")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def expand_tasks(task_arg: str) -> list[str]:
    out: list[str] = []
    for part in [x.strip() for x in task_arg.split(",") if x.strip()]:
        if part in TASK_GROUPS:
            out.extend(TASK_GROUPS[part])
        elif part in TASK_PRESETS:
            out.append(part)
        else:
            raise SystemExit(f"Unknown task preset: {part}. Available: {sorted(TASK_PRESETS)} plus {sorted(TASK_GROUPS)}")
    seen: set[str] = set()
    unique: list[str] = []
    for task in out:
        if task not in seen:
            unique.append(task)
            seen.add(task)
    return unique


def main() -> None:
    args = parse_args()
    tasks = expand_tasks(args.tasks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out_dir / "_slide_cache")
    slide_master = read_tsv(args.slide_master)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    sae_model = None
    d_in = d_latent = None
    if not args.dry_run:
        sae_model, d_in, d_latent = load_sae_from_config(args.sae_ckpt, args.sae_cfg, device=str(device))

    run_summary: dict[str, object] = {
        "tasks": tasks,
        "dry_run": bool(args.dry_run),
        "labels_root": str(args.labels_root),
        "slide_master": str(args.slide_master),
        "features_root": str(args.features_root),
        "feature_subdir": str(args.feature_subdir),
        "sae_ckpt": str(args.sae_ckpt),
        "sae_cfg": str(args.sae_cfg),
        "device": str(device),
        "max_tiles_per_slide": int(args.max_tiles_per_slide),
        "task_summaries": {},
    }

    for task_name in tasks:
        preset = TASK_PRESETS[task_name]
        target_path = args.labels_root / "targets" / str(preset["target_file"])
        label_column = str(preset["label_column"])
        kind = str(preset["kind"])
        projects = parse_projects(preset.get("projects", []))
        classes = list(preset.get("classes", [])) if preset.get("classes") else None
        task_dir = args.out_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        slide_rows, coverage = load_task_rows(
            target_path=target_path,
            label_column=label_column,
            kind=kind,
            projects=projects,
            slide_master=slide_master,
            features_root=args.features_root,
            feature_subdir=args.feature_subdir,
            use_master_h5=bool(args.use_master_h5),
            classes=classes,
        )
        if args.max_cases_per_task > 0:
            keep_cases = sorted({str(r["case_id"]) for r in slide_rows})[: int(args.max_cases_per_task)]
            keep = set(keep_cases)
            slide_rows = [r for r in slide_rows if str(r["case_id"]) in keep]
        write_csv(
            task_dir / "cohort_slides.csv",
            ["case_id", "slide_key", "project_dir", "label", "h5_path"],
            slide_rows,
        )
        coverage["slide_count_after_cap"] = len(slide_rows)
        coverage["case_count_after_cap"] = len({str(r["case_id"]) for r in slide_rows})
        write_json(task_dir / "coverage_summary.json", coverage)
        if args.dry_run:
            run_summary["task_summaries"][task_name] = coverage
            continue

        assert sae_model is not None and d_in is not None and d_latent is not None
        slide_vectors: dict[str, dict[str, np.ndarray | int | float]] = {}
        slide_summary_rows: list[dict[str, object]] = []
        for i, row in enumerate(slide_rows, start=1):
            slide_key = str(row["slide_key"])
            h5_path = Path(str(row["h5_path"]))
            summary = load_or_compute_slide_summary(
                cache_dir=cache_dir / stable_hash(f"{args.sae_ckpt}|{args.sae_cfg}|{args.feature_subdir}|{args.max_tiles_per_slide}", 12),
                slide_key=slide_key,
                h5_path=h5_path,
                sae_model=sae_model,
                d_in=int(d_in),
                d_latent=int(d_latent),
                device=device,
                batch_size=int(args.batch_size),
                max_tiles=int(args.max_tiles_per_slide),
                seed=int(args.seed),
                reuse_cache=bool(args.reuse_slide_cache),
            )
            slide_vectors[slide_key] = summary
            slide_summary_rows.append(
                {
                    **row,
                    "n_tiles_total": int(summary["n_tiles_total"]),
                    "n_tiles_used": int(summary["n_tiles_used"]),
                    "total_mass": float(summary["total_mass"]),
                    "top_fraction_latent": int(np.argmax(np.asarray(summary["fraction"]))),
                    "top_fraction": float(np.max(np.asarray(summary["fraction"]))),
                }
            )
            if i % 25 == 0 or i == len(slide_rows):
                print(f"[{task_name}] encoded {i}/{len(slide_rows)} slides", flush=True)

        case_rows, fractions, mean_acts, prevalences = aggregate_case_summaries(
            slide_rows=slide_rows,
            slide_vectors=slide_vectors,
            d_latent=int(d_latent),
        )
        write_csv(
            task_dir / "slide_sae_summary.csv",
            [
                "case_id",
                "slide_key",
                "project_dir",
                "label",
                "h5_path",
                "n_tiles_total",
                "n_tiles_used",
                "total_mass",
                "top_fraction_latent",
                "top_fraction",
            ],
            slide_summary_rows,
        )
        write_csv(
            task_dir / "case_summary.csv",
            ["case_id", "project_dir", "label", "n_slides", "n_tiles_used", "slide_keys", "h5_paths"],
            case_rows,
        )
        np.savez_compressed(
            task_dir / "case_latent_summaries.npz",
            case_id=np.asarray([str(r["case_id"]) for r in case_rows]),
            label=np.asarray([str(r["label"]) for r in case_rows]),
            fraction=fractions.astype(np.float32),
            mean_activation=mean_acts.astype(np.float32),
            prevalence=prevalences.astype(np.float32),
        )

        association_rows: list[dict[str, object]] = []
        if kind == "continuous":
            for metric_name, mat in [("fraction", fractions), ("mean_activation", mean_acts), ("prevalence", prevalences)]:
                association_rows.extend(continuous_associations(case_rows=case_rows, matrix=mat, metric_name=metric_name))
        else:
            for metric_name, mat in [("fraction", fractions), ("mean_activation", mean_acts), ("prevalence", prevalences)]:
                association_rows.extend(
                    categorical_associations(
                        case_rows=case_rows,
                        matrix=mat,
                        metric_name=metric_name,
                        min_cases_per_class=int(args.min_cases_per_class),
                    )
                )

        if kind == "continuous":
            assoc_fields = ["latent_idx", "metric", "n_cases", "pearson_r", "abs_pearson_r", "mean_metric", "std_metric"]
        else:
            assoc_fields = [
                "latent_idx",
                "metric",
                "class_label",
                "n_class",
                "n_rest",
                "mean_class",
                "mean_rest",
                "diff_class_minus_rest",
                "abs_diff",
                "cohen_d",
            ]
        write_csv(task_dir / "latent_label_associations.csv", assoc_fields, association_rows)
        top_rows = top_case_rows(
            case_rows=case_rows,
            matrix=fractions,
            association_rows=[r for r in association_rows if r.get("metric") == "fraction"],
            max_latents=int(args.top_latents),
            top_cases=int(args.top_cases_per_latent),
            metric_name="fraction",
        )
        write_csv(
            task_dir / "top_cases_for_top_latents.csv",
            ["latent_idx", "metric", "case_rank", "case_id", "project_dir", "label", "value", "n_slides", "n_tiles_used", "slide_keys"],
            top_rows,
        )
        label_counts = Counter(str(r["label"]) for r in case_rows)
        task_summary = {
            **coverage,
            "case_count_with_sae": len(case_rows),
            "label_counts_with_sae": dict(sorted(label_counts.items())),
            "d_in": int(d_in),
            "d_latent": int(d_latent),
            "outputs": {
                "cohort_slides": str(task_dir / "cohort_slides.csv"),
                "case_summary": str(task_dir / "case_summary.csv"),
                "case_latent_summaries": str(task_dir / "case_latent_summaries.npz"),
                "latent_label_associations": str(task_dir / "latent_label_associations.csv"),
                "top_cases_for_top_latents": str(task_dir / "top_cases_for_top_latents.csv"),
            },
        }
        write_json(task_dir / "task_summary.json", task_summary)
        run_summary["task_summaries"][task_name] = task_summary

    write_json(args.out_dir / "run_summary.json", run_summary)
    print(f"[ok] wrote {args.out_dir / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
