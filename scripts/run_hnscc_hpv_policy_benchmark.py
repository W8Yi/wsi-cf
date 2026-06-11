#!/usr/bin/env python3
"""Paper-style benchmark for HNSCC HPV counterfactual edit policies."""

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.paths import DEFAULT_HNSCC_MIL_CKPT, DEFAULT_HNSCC_PROTOTYPE_NPZ


DEFAULT_OUT_DIR = WSI_CF_ROOT / "artifacts/hnscc_hpv_paper_benchmark"
DEFAULT_REGION_BANK_CSV = WSI_CF_ROOT / "artifacts/hnscc_hpv_paper_benchmark/regions/region_bank.csv"
DEFAULT_EDIT_MANIFEST = WSI_CF_ROOT / "artifacts/hnscc_hpv_paper_benchmark/regions/progressive_edit_manifest.json"
DEFAULT_SPLIT_TSV = WSI_CF_ROOT / "resources/manifests/hnsc_hpv_5fold/split_0.tsv"
DEFAULT_TCGA_FEATURES_ROOT = Path("/research/projects/mllab/WSI/TCGA_features")
DEFAULT_POLICIES = [
    "ours=configs/edit_policies/showcase_best.json",
    "naive_no_preserve=configs/edit_policies/naive_no_preserve.json",
    "naive_full_duration=configs/edit_policies/baseline_no_preserve_full_duration.json",
]


LABEL_TO_ID = {"HPV-": 0, "hpv_neg": 0, "0": 0, "HPV+": 1, "hpv_pos": 1, "1": 1}
ID_TO_LABEL = {0: "HPV-", 1: "HPV+"}
DIRECTION_TO_TARGET_ID = {"hpv_neg": 0, "hpv_pos": 1}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    policy_path: Path


@dataclass(frozen=True)
class RunSpec:
    method: str
    direction: str
    run_id: str
    run_dir: Path
    run_manifest: Path
    source_image: Path
    generated_image: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--region-bank-csv", type=Path, default=DEFAULT_REGION_BANK_CSV)
    parser.add_argument("--edit-manifest", type=Path, default=DEFAULT_EDIT_MANIFEST)
    parser.add_argument("--split-tsv", type=Path, default=DEFAULT_SPLIT_TSV)
    parser.add_argument("--policy", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--direction", choices=["hpv_pos", "hpv_neg"], default="hpv_neg")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--generation-device", default=None)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--sae-ckpt", type=Path, default=None)
    parser.add_argument("--sae-cfg", type=Path, default=None)
    parser.add_argument("--mil-ckpt", type=Path, default=DEFAULT_HNSCC_MIL_CKPT)
    parser.add_argument("--prototype-npz", type=Path, default=DEFAULT_HNSCC_PROTOTYPE_NPZ)
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--association-csv", type=Path, default=None)
    parser.add_argument("--prepare-associations", action="store_true")
    parser.add_argument("--association-top-k", type=int, default=20)
    parser.add_argument("--association-max-slides-per-class", type=int, default=0)
    parser.add_argument("--association-max-tiles-per-slide", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--grid-step-px", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-edits", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-mode", choices=["minimal", "debug"], default="debug")
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--require-paper-deps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WSI_CF_ROOT / path


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(str(key))
                    fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hash_or_missing(path: Path) -> str:
    return sha256_file(path) if path.exists() else ""


def parse_method_specs(policy_args: list[str] | None) -> list[MethodSpec]:
    specs: list[MethodSpec] = []
    for raw in policy_args or DEFAULT_POLICIES:
        if "=" not in raw:
            raise ValueError(f"--policy must be NAME=PATH, got {raw!r}")
        name, path_text = raw.split("=", 1)
        path = resolve_path(Path(path_text.strip()))
        if not path.exists():
            raise FileNotFoundError(f"Policy file does not exist for method {name!r}: {path}")
        specs.append(MethodSpec(name=name.strip(), policy_path=path))
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate method names: {names}")
    return specs


def label_id(value: Any) -> int:
    text = str(value)
    if text in LABEL_TO_ID:
        return int(LABEL_TO_ID[text])
    raise ValueError(f"Cannot parse HPV label: {value!r}")


def resolve_split_h5_path(row: dict[str, Any]) -> Path | None:
    raw_text = str(row.get("h5_path", "")).strip()
    if raw_text:
        raw_path = Path(raw_text)
        candidates = [raw_path if raw_path.is_absolute() else resolve_path(raw_path)]
    else:
        candidates = []

    slide_key = str(row.get("slide_key", "")).strip()
    project_dir = str(row.get("project_dir", "TCGA-HNSC")).strip() or "TCGA-HNSC"
    if slide_key:
        candidates.append(DEFAULT_TCGA_FEATURES_ROOT / project_dir / "features_uni2" / f"{slide_key}.h5")
        candidates.append(DEFAULT_TCGA_FEATURES_ROOT / "TCGA-HNSC" / "features_uni2" / f"{slide_key}.h5")

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def direction_for_source_label(source_label_id: int) -> str:
    return "hpv_neg" if int(source_label_id) == 1 else "hpv_pos"


def target_label(direction: str) -> str:
    return ID_TO_LABEL[DIRECTION_TO_TARGET_ID[str(direction)]]


def target_probability(row: dict[str, Any], direction: str) -> float:
    return float(row["prob_hpv_neg"] if direction == "hpv_neg" else row["prob_hpv_pos"])


def policy_run_root(out_dir: Path, method: str, direction: str | None = None) -> Path:
    base = out_dir / "runs" / method
    return base / str(direction) if direction else base


def split_manifest_by_direction(
    *,
    manifest_rows: list[dict[str, Any]],
    region_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    region_label_by_id: dict[str, int] = {}
    for row in region_rows:
        rid = str(row.get("region_id", ""))
        if not rid:
            continue
        if row.get("label", "") != "":
            region_label_by_id[rid] = label_id(row["label"])
        elif row.get("hpv_status", ""):
            region_label_by_id[rid] = label_id(row["hpv_status"])

    out = {"hpv_neg": [], "hpv_pos": []}
    for item in manifest_rows:
        row = dict(item)
        rid = str(row.get("region_id", ""))
        source_label = row.get("source_label_id", row.get("label", None))
        if source_label is None:
            source_label = region_label_by_id.get(rid)
        if source_label is None:
            raise ValueError(f"Cannot determine source label for manifest run_id={row.get('run_id')!r}")
        source_id = label_id(source_label)
        direction = str(row.get("direction_recommendation") or direction_for_source_label(source_id))
        target_id = DIRECTION_TO_TARGET_ID[direction]
        row["source_label_id"] = int(source_id)
        row["source_label"] = ID_TO_LABEL[int(source_id)]
        row["target_label_id"] = int(target_id)
        row["target_label"] = ID_TO_LABEL[int(target_id)]
        row["direction"] = direction
        out[direction].append(row)
    return out


def write_direction_manifests(out_dir: Path, edit_manifest: Path, region_bank_csv: Path, bidirectional: bool, direction: str) -> dict[str, Path]:
    manifest_rows = list(read_json(edit_manifest))
    if bidirectional:
        region_rows = read_csv_rows(region_bank_csv)
        grouped = split_manifest_by_direction(manifest_rows=manifest_rows, region_rows=region_rows)
    else:
        grouped = {direction: manifest_rows}
    manifest_dir = out_dir / "manifests"
    paths: dict[str, Path] = {}
    for dir_name, rows in grouped.items():
        if not rows:
            continue
        path = manifest_dir / f"to_{dir_name}_manifest.json"
        write_json(path, rows)
        paths[dir_name] = path
    return paths


def generation_command(args: argparse.Namespace, spec: MethodSpec, direction: str, manifest_path: Path) -> list[str]:
    cmd = [
        str(args.python),
        str(WSI_CF_ROOT / "scripts/run_progressive_region_edit.py"),
        "--edit-policy",
        str(spec.policy_path),
        "--region-bank-csv",
        str(resolve_path(args.region_bank_csv)),
        "--edit-manifest",
        str(manifest_path),
        "--direction",
        str(direction),
        "--out-dir",
        str(policy_run_root(args.out_dir.resolve(), spec.name, direction if args.bidirectional else None)),
        "--device",
        str(args.generation_device or args.device),
        "--dtype",
        str(args.dtype),
        "--output-mode",
        str(args.output_mode),
    ]
    if int(args.max_runs) > 0:
        cmd += ["--max-runs", str(int(args.max_runs))]
    if args.sae_ckpt is not None:
        cmd += ["--sae-ckpt", str(resolve_path(args.sae_ckpt))]
    if args.sae_cfg is not None:
        cmd += ["--sae-cfg", str(resolve_path(args.sae_cfg))]
    if bool(args.skip_existing):
        cmd.append("--skip-existing")
    return cmd


def run_generation(args: argparse.Namespace, specs: list[MethodSpec], direction_manifests: dict[str, Path]) -> list[dict[str, Any]]:
    command_rows: list[dict[str, Any]] = []
    for spec in specs:
        for direction, manifest_path in direction_manifests.items():
            cmd = generation_command(args, spec, direction, manifest_path)
            command_rows.append({"method": spec.name, "direction": direction, "command": " ".join(shlex.quote(part) for part in cmd)})
            subprocess.run(cmd, check=True, cwd=str(WSI_CF_ROOT))
    return command_rows


def discover_runs(out_dir: Path, specs: list[MethodSpec], direction_manifests: dict[str, Path], bidirectional: bool) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for spec in specs:
        for direction in direction_manifests:
            root = policy_run_root(out_dir, spec.name, direction if bidirectional else None)
            summary_path = root / "run_summary.json"
            if not root.exists():
                continue
            if summary_path.exists():
                summary = read_json(summary_path)
                run_ids = [str(row["run_id"]) for row in summary.get("runs", []) if row.get("run_id")]
            else:
                run_ids = sorted(path.name for path in root.iterdir() if (path / "run_manifest.json").exists())
            for run_id in run_ids:
                run_dir = root / run_id
                paths = [run_dir / "run_manifest.json", run_dir / "source_region_actual.png", run_dir / "generated.png"]
                missing = [path for path in paths if not path.exists()]
                if missing:
                    raise FileNotFoundError(f"Missing artifacts for {spec.name}/{direction}/{run_id}: {missing}")
                runs.append(
                    RunSpec(
                        method=spec.name,
                        direction=direction,
                        run_id=run_id,
                        run_dir=run_dir,
                        run_manifest=paths[0],
                        source_image=paths[1],
                        generated_image=paths[2],
                    )
                )
    if not runs:
        raise FileNotFoundError(f"No benchmark runs found under {out_dir / 'runs'}")
    return runs


def resolve_device(device_name: str):
    from wsi_cf.common.runtime import resolve_device as resolve_runtime_device

    return resolve_runtime_device(device_name)


def preflight_dependencies(require_paper_deps: bool) -> dict[str, str]:
    mods = ["sklearn"]
    if require_paper_deps:
        mods.extend(["skimage", "lpips"])
    versions: dict[str, str] = {}
    missing: list[str] = []
    for mod_name in mods:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = str(getattr(mod, "__version__", "installed"))
        except Exception as exc:
            versions[mod_name] = f"missing: {exc}"
            missing.append(mod_name)
    if missing:
        raise RuntimeError("Missing required benchmark dependencies: " + ", ".join(missing))
    return versions


def load_models(device_name: str, mil_ckpt: Path):
    from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint
    from wsi_cf.generation.pixcell import load_uni2

    device = resolve_device(device_name)
    mil_model = build_mil_from_checkpoint(mil_ckpt, device=device)
    uni_model, uni_transform = load_uni2(device)
    return device, mil_model, uni_model, uni_transform


def encode_image_grid(image_path: Path, grid_path: Path, grid_step_px: int, device: Any, uni_model: Any, uni_transform: Any, force: bool) -> np.ndarray:
    if grid_path.exists() and not force:
        return np.asarray(np.load(grid_path), dtype=np.float32)

    import torch
    from wsi_cf.generation.pixcell import build_uni_grid_from_image

    image = Image.open(image_path).convert("RGB")
    z_grid = build_uni_grid_from_image(
        image,
        uni_model=uni_model,
        uni_transform=uni_transform,
        grid_step_px=int(grid_step_px),
        device=device,
        out_dtype=torch.float32,
    )
    arr = z_grid.detach().cpu().numpy().astype(np.float32)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(grid_path, arr)
    return arr


def eval_grid(mil_model: Any, device: Any, z_grid: np.ndarray) -> dict[str, Any]:
    from wsi_cf.eval.hnsc_hpv import run_mil_attention

    bag = np.asarray(z_grid, dtype=np.float32).reshape(-1, z_grid.shape[-1])
    _, pred, prob_pos = run_mil_attention(mil_model, bag, device=device)
    return {
        "pred_label_id": int(pred),
        "pred_label": ID_TO_LABEL[int(pred)],
        "prob_hpv_pos": float(prob_pos),
        "prob_hpv_neg": float(1.0 - prob_pos),
    }


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def target_mask(shape: tuple[int, int], target_cells: list[dict[str, Any]], grid_step_px: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for cell in target_cells:
        gx = int(cell["gx"])
        gy = int(cell["gy"])
        x0 = max(0, gx * int(grid_step_px))
        y0 = max(0, gy * int(grid_step_px))
        x1 = min(width, x0 + int(grid_step_px))
        y1 = min(height, y0 + int(grid_step_px))
        mask[y0:y1, x0:x1] = True
    return mask


def finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()) if arr.size else math.nan


def maybe_ssim(source: np.ndarray, generated: np.ndarray, require: bool) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        if require:
            raise
        return math.nan
    return float(structural_similarity(source.astype(np.uint8), generated.astype(np.uint8), channel_axis=2, data_range=255))


def load_lpips_model(device: Any, require: bool):
    try:
        import lpips
    except Exception:
        if require:
            raise
        return None
    return lpips.LPIPS(net="alex").to(device).eval()


def lpips_distance(source_path: Path, generated_path: Path, model: Any, device: Any) -> float:
    if model is None:
        return math.nan
    import torch

    def load_tensor(path: Path):
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        return float(model(load_tensor(source_path), load_tensor(generated_path)).detach().cpu().reshape(-1)[0].item())


def rgb_diff_metrics(source_image: Path, generated_image: Path, manifest: dict[str, Any], lpips_model: Any, device: Any, require_deps: bool) -> dict[str, float]:
    source = load_rgb(source_image)
    generated = load_rgb(generated_image)
    if source.shape != generated.shape:
        raise ValueError(f"Image shape mismatch: {source.shape} vs {generated.shape}")
    diff = np.abs(generated - source).mean(axis=2)
    mask = target_mask(diff.shape, list(manifest.get("target_cells", [])), int(manifest.get("grid_step_px", 256)))
    context = ~mask
    target_mean = finite_mean(diff[mask])
    context_mean = finite_mean(diff[context])
    return {
        "rgb_abs_mean": float(diff.mean()),
        "rgb_abs_median": float(np.median(diff)),
        "rgb_abs_p95": float(np.percentile(diff, 95.0)),
        "rgb_abs_target_mean": float(target_mean),
        "rgb_abs_context_mean": float(context_mean),
        "outside_region_pixel_change": float(context_mean),
        "masked_edit_ratio": float(target_mean / context_mean) if context_mean > 0 else math.nan,
        "ssim": maybe_ssim(source, generated, require=require_deps),
        "lpips": lpips_distance(source_image, generated_image, lpips_model, device),
    }


def rgb_cell_metrics(source_image: Path, generated_image: Path, grid_step_px: int, target_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = load_rgb(source_image)
    generated = load_rgb(generated_image)
    if source.shape != generated.shape:
        raise ValueError(f"Image shape mismatch: {source.shape} vs {generated.shape}")
    diff = np.abs(generated - source).mean(axis=2)
    grid_h = int(math.ceil(diff.shape[0] / int(grid_step_px)))
    grid_w = int(math.ceil(diff.shape[1] / int(grid_step_px)))
    mask = cell_mask((grid_h, grid_w), target_cells)
    rows: list[dict[str, Any]] = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            y0, y1 = gy * int(grid_step_px), min((gy + 1) * int(grid_step_px), diff.shape[0])
            x0, x1 = gx * int(grid_step_px), min((gx + 1) * int(grid_step_px), diff.shape[1])
            rows.append(
                {
                    "cell_gx": int(gx),
                    "cell_gy": int(gy),
                    "is_target_cell": bool(mask[gy, gx]),
                    "rgb_abs_mean": float(diff[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else math.nan,
                }
            )
    return rows


def cell_mask(grid_shape: tuple[int, int], target_cells: list[dict[str, Any]]) -> np.ndarray:
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for cell in target_cells:
        gx = int(cell["gx"])
        gy = int(cell["gy"])
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            mask[gy, gx] = True
    return mask


def uni_cell_metrics(source_grid: np.ndarray, generated_grid: np.ndarray, target_cells: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if source_grid.shape != generated_grid.shape:
        raise ValueError(f"UNI2 grid shape mismatch: {source_grid.shape} vs {generated_grid.shape}")
    delta = generated_grid.astype(np.float32) - source_grid.astype(np.float32)
    l2 = np.linalg.norm(delta, axis=2)
    dot = np.sum(source_grid * generated_grid, axis=2)
    src_norm = np.linalg.norm(source_grid, axis=2)
    gen_norm = np.linalg.norm(generated_grid, axis=2)
    cosine_distance = 1.0 - (dot / np.maximum(src_norm * gen_norm, 1e-8))
    mask = cell_mask(source_grid.shape[:2], target_cells)
    context = ~mask
    summary = {
        "uni_l2_mean": float(l2.mean()),
        "uni_l2_target_mean": finite_mean(l2[mask]),
        "uni_l2_context_mean": finite_mean(l2[context]),
        "outside_region_uni_l2_mean": finite_mean(l2[context]),
        "uni_cosine_distance_mean": float(cosine_distance.mean()),
        "uni_cosine_distance_target_mean": finite_mean(cosine_distance[mask]),
        "uni_cosine_distance_context_mean": finite_mean(cosine_distance[context]),
        "outside_region_uni_cosine_distance_mean": finite_mean(cosine_distance[context]),
    }
    rows: list[dict[str, Any]] = []
    for gy in range(int(source_grid.shape[0])):
        for gx in range(int(source_grid.shape[1])):
            rows.append(
                {
                    "cell_gx": int(gx),
                    "cell_gy": int(gy),
                    "is_target_cell": bool(mask[gy, gx]),
                    "uni_l2": float(l2[gy, gx]),
                    "uni_cosine_distance": float(cosine_distance[gy, gx]),
                }
            )
    return summary, rows


def seam_score(image: np.ndarray, bounds: list[dict[str, int]]) -> float:
    vals: list[float] = []
    h, w = image.shape[:2]
    for b in bounds:
        x0, y0, x1, y1 = int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"])
        if 0 < x0 < w:
            vals.append(float(np.abs(image[y0:y1, x0] - image[y0:y1, x0 - 1]).mean()))
        if 0 < x1 < w:
            vals.append(float(np.abs(image[y0:y1, x1] - image[y0:y1, x1 - 1]).mean()))
        if 0 < y0 < h:
            vals.append(float(np.abs(image[y0, x0:x1] - image[y0 - 1, x0:x1]).mean()))
        if 0 < y1 < h:
            vals.append(float(np.abs(image[y1, x0:x1] - image[y1 - 1, x0:x1]).mean()))
    return float(np.mean(vals)) if vals else math.nan


def window_consistency_metrics(source_image: Path, generated_image: Path, manifest: dict[str, Any], run_dir: Path) -> dict[str, float]:
    source = load_rgb(source_image)
    canvas = source.copy()
    committed = np.zeros(source.shape[:2], dtype=bool)
    overlap_values: list[float] = []
    bounds: list[dict[str, int]] = []
    for idx, step in enumerate(manifest.get("window_history", []), start=1):
        b = step.get("commit_bounds_global", {})
        if not b:
            continue
        x0, y0, x1, y1 = int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"])
        steered_path_text = str(step.get("steered_window_path", ""))
        steered_path = Path(steered_path_text) if steered_path_text else Path()
        if not steered_path_text or not steered_path.exists() or steered_path.is_dir():
            steered_path = run_dir / "steps" / f"step_{idx:02d}" / "steered_window.png"
        if not steered_path.exists():
            continue
        steered = load_rgb(steered_path)[: y1 - y0, : x1 - x0]
        overlap = committed[y0:y1, x0:x1]
        if np.any(overlap):
            prior = canvas[y0:y1, x0:x1]
            overlap_values.append(float(np.abs(steered[overlap] - prior[overlap]).mean()))
        canvas[y0:y1, x0:x1] = steered
        committed[y0:y1, x0:x1] = True
        bounds.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
    generated = load_rgb(generated_image)
    seam_raw = seam_score(generated, bounds)
    seam_src = seam_score(source, bounds)
    return {
        "overlap_consistency_rgb_abs_mean": float(np.mean(overlap_values)) if overlap_values else math.nan,
        "overlap_consistency_count": int(len(overlap_values)),
        "seam_score_rgb_abs": seam_raw,
        "source_seam_score_rgb_abs": seam_src,
        "seam_score_excess_rgb_abs": float(seam_raw - seam_src) if np.isfinite(seam_raw) and np.isfinite(seam_src) else math.nan,
    }


def classification_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    out: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows if str(row["method"]) != "source"}):
        method_rows = [row for row in rows if str(row["method"]) == method]
        y_true = np.asarray([int(row["target_label_id"]) for row in method_rows], dtype=np.int64)
        y_pred = np.asarray([int(row["pred_label_id"]) for row in method_rows], dtype=np.int64)
        y_score = np.asarray([float(row["prob_hpv_pos"]) for row in method_rows], dtype=np.float32)
        try:
            auroc = float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else math.nan
        except ValueError:
            auroc = math.nan
        out.append(
            {
                "method": method,
                "n_runs": int(len(method_rows)),
                "target_accuracy": float(accuracy_score(y_true, y_pred)) if len(method_rows) else math.nan,
                "target_f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(method_rows) else math.nan,
                "target_auroc": auroc,
            }
        )
    source_rows = [row for row in rows if str(row["method"]) == "source"]
    if source_rows:
        y_true = np.asarray([int(row["source_label_id"]) for row in source_rows], dtype=np.int64)
        y_pred = np.asarray([int(row["pred_label_id"]) for row in source_rows], dtype=np.int64)
        y_score = np.asarray([float(row["prob_hpv_pos"]) for row in source_rows], dtype=np.float32)
        try:
            auroc = float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else math.nan
        except ValueError:
            auroc = math.nan
        out.append(
            {
                "method": "source_original_label",
                "n_runs": int(len(source_rows)),
                "target_accuracy": float(accuracy_score(y_true, y_pred)),
                "target_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "target_auroc": auroc,
                "source_accuracy": float(accuracy_score(y_true, y_pred)),
                "source_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "source_auroc": auroc,
            }
        )
    return out


def load_prototypes(npz_path: Path, key: str) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    with np.load(npz_path, allow_pickle=False) as data:
        latent_ids = np.asarray(data["latent_ids"], dtype=np.int64)
        directions = np.asarray(data["selected_direction"])
        vectors = np.asarray(data[key], dtype=np.float32)
    return {int(latent): vectors[i] for i, latent in enumerate(latent_ids)}, {int(latent): str(directions[i]) for i, latent in enumerate(latent_ids)}


def cosine_to_vector(z: np.ndarray, vec: np.ndarray) -> np.ndarray:
    dot = np.sum(z * vec.reshape(1, -1), axis=1)
    zn = np.linalg.norm(z, axis=1)
    vn = float(np.linalg.norm(vec))
    return dot / np.maximum(zn * vn, 1e-8)


def concept_fidelity_from_latents(
    source_latents: np.ndarray,
    generated_latents: np.ndarray,
    target_mask_flat: np.ndarray,
    target_proto: np.ndarray,
    source_proto: np.ndarray,
    target_latent: int,
    source_latent: int,
    target_assoc_latents: list[int],
    source_assoc_latents: list[int],
) -> dict[str, float]:
    mask = np.asarray(target_mask_flat, dtype=bool)
    src = np.asarray(source_latents, dtype=np.float32)[mask]
    gen = np.asarray(generated_latents, dtype=np.float32)[mask]
    if src.size == 0:
        return {}
    target_cos_before = finite_mean(cosine_to_vector(src, target_proto))
    target_cos_after = finite_mean(cosine_to_vector(gen, target_proto))
    source_cos_before = finite_mean(cosine_to_vector(src, source_proto))
    source_cos_after = finite_mean(cosine_to_vector(gen, source_proto))
    target_assoc_latents = [idx for idx in target_assoc_latents if 0 <= int(idx) < gen.shape[1]]
    source_assoc_latents = [idx for idx in source_assoc_latents if 0 <= int(idx) < gen.shape[1]]
    return {
        "target_proto_cos_before": target_cos_before,
        "target_proto_cos_after": target_cos_after,
        "delta_target_proto_cos": float(target_cos_after - target_cos_before),
        "source_proto_cos_before": source_cos_before,
        "source_proto_cos_after": source_cos_after,
        "delta_source_proto_cos": float(source_cos_after - source_cos_before),
        "target_latent_activation_before": finite_mean(src[:, int(target_latent)]),
        "target_latent_activation_after": finite_mean(gen[:, int(target_latent)]),
        "delta_target_latent_activation": finite_mean(gen[:, int(target_latent)] - src[:, int(target_latent)]),
        "source_latent_activation_before": finite_mean(src[:, int(source_latent)]),
        "source_latent_activation_after": finite_mean(gen[:, int(source_latent)]),
        "delta_source_latent_activation": finite_mean(gen[:, int(source_latent)] - src[:, int(source_latent)]),
        "target_associated_activation_delta": finite_mean(gen[:, target_assoc_latents] - src[:, target_assoc_latents]) if target_assoc_latents else math.nan,
        "source_associated_activation_delta": finite_mean(gen[:, source_assoc_latents] - src[:, source_assoc_latents]) if source_assoc_latents else math.nan,
    }


def association_top_latents(path: Path | None, top_k: int) -> dict[int, list[int]]:
    if path is None or not path.exists():
        return {0: [], 1: []}
    rows = read_csv_rows(path)
    out: dict[int, list[int]] = {0: [], 1: []}
    for class_label, label in [("HPV-", 0), ("HPV+", 1), ("0", 0), ("1", 1)]:
        candidates = [
            row
            for row in rows
            if str(row.get("class_label", "")) == class_label
            and str(row.get("metric", "fraction")) in {"fraction", "prevalence"}
            and float(row.get("diff_class_minus_rest", 0.0)) > 0
        ]
        candidates.sort(key=lambda row: (-float(row.get("cohen_d", 0.0)), -float(row.get("abs_diff", 0.0)), int(row["latent_idx"])))
        if candidates and not out[label]:
            out[label] = [int(row["latent_idx"]) for row in candidates[: int(top_k)]]
    return out


def prepare_hnscc_associations(args: argparse.Namespace, out_dir: Path) -> Path:
    import h5py
    import torch
    from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features

    split_rows = read_csv_rows(resolve_path(args.split_tsv), delimiter="\t")
    by_label: dict[int, list[dict[str, str]]] = {0: [], 1: []}
    missing_h5: list[dict[str, Any]] = []
    for row in split_rows:
        if str(row.get("split", "")) != "train":
            continue
        try:
            lab = label_id(row.get("label", row.get("hpv_status", "")))
        except ValueError:
            continue
        h5_path = resolve_split_h5_path(row)
        if h5_path is None:
            missing_h5.append(
                {
                    "slide_key": row.get("slide_key", ""),
                    "label": lab,
                    "raw_h5_path": row.get("h5_path", ""),
                    "project_dir": row.get("project_dir", ""),
                }
            )
            continue
        resolved = dict(row)
        resolved["h5_path_resolved"] = str(h5_path)
        by_label[lab].append(resolved)
    n = min(len(by_label[0]), len(by_label[1]))
    if int(args.association_max_slides_per_class) > 0:
        n = min(n, int(args.association_max_slides_per_class))
    assoc_dir = out_dir / "associations"
    if missing_h5:
        write_csv(assoc_dir / "missing_train_h5.csv", missing_h5)
    if n <= 0:
        raise RuntimeError(
            "Could not prepare HNSCC concept associations because no balanced train slides with UNI2 H5 files were found. "
            f"Resolved train H5 counts: HPV-={len(by_label[0])}, HPV+={len(by_label[1])}. "
            f"Missing train H5 rows: {len(missing_h5)}. "
            "Expected either valid split TSV h5_path entries or files under "
            f"{DEFAULT_TCGA_FEATURES_ROOT}/<project_dir>/features_uni2/<slide_key>.h5."
        )
    chosen = by_label[0][:n] + by_label[1][:n]
    device = resolve_device(str(args.device))
    sae_model, d_in, d_latent = load_sae_from_config(resolve_path(args.sae_ckpt), resolve_path(args.sae_cfg), device=str(device))
    rng = np.random.default_rng(int(args.seed))
    frac_rows: list[np.ndarray] = []
    labels: list[int] = []
    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in chosen:
        h5_path = Path(str(row["h5_path_resolved"]))
        with h5py.File(h5_path, "r") as handle:
            feats = handle["features"]
            arr = feats[:] if feats.ndim == 2 else feats[0]
        x_np = np.asarray(arr, dtype=np.float32)
        if x_np.shape[1] != int(d_in):
            skipped.append(
                {
                    "slide_key": row.get("slide_key", ""),
                    "label": row.get("label", ""),
                    "h5_path": str(h5_path),
                    "reason": f"feature_dim_{x_np.shape[1]}_!=_sae_d_in_{int(d_in)}",
                }
            )
            continue
        if int(args.association_max_tiles_per_slide) > 0 and x_np.shape[0] > int(args.association_max_tiles_per_slide):
            idx = rng.choice(x_np.shape[0], size=int(args.association_max_tiles_per_slide), replace=False)
            idx.sort()
            x_np = x_np[idx]
        active_sum = np.zeros((int(d_latent),), dtype=np.float64)
        for start in range(0, x_np.shape[0], int(args.batch_size)):
            batch = torch.as_tensor(x_np[start : start + int(args.batch_size)], dtype=torch.float32, device=device)
            with torch.inference_mode():
                z = sae_encode_features(sae_model, batch).detach().cpu().numpy()
            active_sum += (z > 0).sum(axis=0)
        frac_rows.append((active_sum / max(x_np.shape[0], 1)).astype(np.float32))
        lab = label_id(row["label"])
        labels.append(lab)
        processed.append({"slide_key": row.get("slide_key", ""), "label": lab, "h5_path": str(h5_path), "n_tiles_used": int(x_np.shape[0])})
    if skipped:
        write_csv(assoc_dir / "skipped_train_h5.csv", skipped)
    if not frac_rows:
        raise RuntimeError(
            "Could not prepare HNSCC concept associations because every resolved train H5 was skipped. "
            f"Balanced slides requested per class: {n}. Skipped rows: {len(skipped)}. "
            f"See {assoc_dir / 'skipped_train_h5.csv'} if it was written."
        )
    frac = np.stack(frac_rows, axis=0)
    labels_np = np.asarray(labels)
    rows: list[dict[str, Any]] = []
    for lab in (0, 1):
        mask = labels_np == lab
        rest = ~mask
        mean_class = frac[mask].mean(axis=0)
        mean_rest = frac[rest].mean(axis=0)
        diff = mean_class - mean_rest
        for latent_idx in range(frac.shape[1]):
            rows.append(
                {
                    "latent_idx": latent_idx,
                    "metric": "fraction",
                    "class_label": ID_TO_LABEL[lab],
                    "n_class": int(mask.sum()),
                    "n_rest": int(rest.sum()),
                    "mean_class": float(mean_class[latent_idx]),
                    "mean_rest": float(mean_rest[latent_idx]),
                    "diff_class_minus_rest": float(diff[latent_idx]),
                    "abs_diff": float(abs(diff[latent_idx])),
                    "cohen_d": float(diff[latent_idx]),
                }
            )
    path = assoc_dir / "latent_label_associations.csv"
    write_csv(path, rows)
    write_csv(assoc_dir / "cohort_slides.csv", processed)
    write_json(
        assoc_dir / "summary.json",
        {
            "n_per_class": int(n),
            "split": "train",
            "association_csv": str(path),
            "resolved_train_h5_counts": {"HPV-": len(by_label[0]), "HPV+": len(by_label[1])},
            "missing_train_h5_rows": len(missing_h5),
            "skipped_train_h5_rows": len(skipped),
        },
    )
    return path


def encode_sae_latents(sae_model: Any, grid: np.ndarray, device: Any) -> np.ndarray:
    import torch
    from wsi_cf.steering.sae_runtime import sae_encode_features

    x = torch.as_tensor(grid.reshape(-1, grid.shape[-1]), dtype=torch.float32, device=device)
    with torch.inference_mode():
        return sae_encode_features(sae_model, x).detach().cpu().numpy().astype(np.float32)


def source_grid_key(run_id: str) -> str:
    return f"source__{run_id}"


def generated_grid_key(method: str, direction: str, run_id: str) -> str:
    return f"{method}__{direction}__{run_id}"


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod_name in ["numpy", "PIL", "sklearn", "skimage", "lpips", "torch"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = str(getattr(mod, "__version__", "installed"))
        except Exception as exc:
            versions[mod_name] = f"unavailable: {exc}"
    return versions


def git_state() -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=str(WSI_CF_ROOT), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    status = run(["git", "status", "--short"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "status_short": status.splitlines()}


def run_config(args: argparse.Namespace, specs: list[MethodSpec], direction_manifests: dict[str, Path]) -> dict[str, Any]:
    region_bank = resolve_path(args.region_bank_csv)
    edit_manifest = resolve_path(args.edit_manifest)
    return {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python_executable": sys.executable,
        "package_versions": package_versions(),
        "git": git_state(),
        "policy_hashes": {spec.name: {"path": str(spec.policy_path), "sha256": sha256_file(spec.policy_path)} for spec in specs},
        "region_bank": {"path": str(region_bank), "sha256": file_hash_or_missing(region_bank)},
        "edit_manifest": {"path": str(edit_manifest), "sha256": file_hash_or_missing(edit_manifest)},
        "direction_manifests": {k: {"path": str(v), "sha256": file_hash_or_missing(v)} for k, v in direction_manifests.items()},
    }


def compute_metrics(args: argparse.Namespace, specs: list[MethodSpec], direction_manifests: dict[str, Path]) -> dict[str, Any]:
    from wsi_cf.steering.sae_runtime import load_sae_from_config

    out_dir = args.out_dir.resolve()
    metrics_dir = out_dir / "metrics"
    grid_dir = metrics_dir / "encoded_uni2_grids"
    runs = discover_runs(out_dir, specs, direction_manifests, bool(args.bidirectional))

    dep_versions = preflight_dependencies(bool(args.require_paper_deps))
    device, mil_model, uni_model, uni_transform = load_models(str(args.device), resolve_path(args.mil_ckpt))
    lpips_model = load_lpips_model(device, require=bool(args.require_paper_deps))
    sae_model, _, _ = load_sae_from_config(resolve_path(args.sae_ckpt), resolve_path(args.sae_cfg), device=str(device))
    proto_by_latent, _ = load_prototypes(resolve_path(args.prototype_npz), str(args.prototype_key))
    assoc_csv = resolve_path(args.association_csv) if args.association_csv is not None else None
    assoc_top = association_top_latents(assoc_csv, int(args.association_top_k))

    source_cache: dict[str, tuple[Path, np.ndarray, dict[str, Any], np.ndarray]] = {}
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    concept_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []

    for run in runs:
        manifest = read_json(run.run_manifest)
        meta = dict(manifest.get("target_metadata", {}))
        direction = str(manifest.get("prototype_direction", run.direction))
        target_id = int(meta.get("target_label_id", DIRECTION_TO_TARGET_ID[direction]))
        source_id = int(meta.get("source_label_id", 1 - target_id))
        grid_step_px = int(manifest.get("grid_step_px", args.grid_step_px))
        src_key = source_grid_key(run.run_id)
        src_grid_path = grid_dir / f"{src_key}.npy"
        if src_key not in source_cache:
            source_grid = encode_image_grid(run.source_image, src_grid_path, grid_step_px, device, uni_model, uni_transform, bool(args.force_reencode))
            source_eval = eval_grid(mil_model, device, source_grid)
            source_latents = encode_sae_latents(sae_model, source_grid, device)
            source_cache[src_key] = (src_grid_path, source_grid, source_eval, source_latents)
            prediction_rows.append(
                {
                    "method": "source",
                    "direction": direction,
                    "run_id": run.run_id,
                    "source_label_id": source_id,
                    "target_label_id": target_id,
                    "image_path": str(run.source_image),
                    "uni2_grid_path": str(src_grid_path),
                    **source_eval,
                    "target_prob": target_probability(source_eval, direction),
                }
            )
        source_grid_path, source_grid, source_eval, source_latents = source_cache[src_key]

        gen_grid_path = grid_dir / f"{generated_grid_key(run.method, direction, run.run_id)}.npy"
        generated_grid = encode_image_grid(run.generated_image, gen_grid_path, grid_step_px, device, uni_model, uni_transform, bool(args.force_reencode))
        generated_eval = eval_grid(mil_model, device, generated_grid)
        generated_latents = encode_sae_latents(sae_model, generated_grid, device)
        prediction_rows.append(
            {
                "method": run.method,
                "direction": direction,
                "run_id": run.run_id,
                "source_label_id": source_id,
                "target_label_id": target_id,
                "image_path": str(run.generated_image),
                "uni2_grid_path": str(gen_grid_path),
                **generated_eval,
                "target_prob": target_probability(generated_eval, direction),
            }
        )

        rgb_metrics = rgb_diff_metrics(run.source_image, run.generated_image, manifest, lpips_model, device, bool(args.require_paper_deps))
        uni_metrics, cell_rows = uni_cell_metrics(source_grid, generated_grid, list(manifest.get("target_cells", [])))
        rgb_cell_rows = rgb_cell_metrics(run.source_image, run.generated_image, grid_step_px, list(manifest.get("target_cells", [])))
        rgb_cell_by_key = {(int(row["cell_gx"]), int(row["cell_gy"])): row for row in rgb_cell_rows}
        consistency = window_consistency_metrics(run.source_image, run.generated_image, manifest, run.run_dir)
        consistency_rows.append({"method": run.method, "direction": direction, "run_id": run.run_id, **consistency})
        target_prob_before = target_probability(source_eval, direction)
        target_prob_after = target_probability(generated_eval, direction)
        target_latent = int(args.neg_latent if target_id == 0 else args.pos_latent)
        source_latent = int(args.pos_latent if target_id == 0 else args.neg_latent)
        cmask = cell_mask(source_grid.shape[:2], list(manifest.get("target_cells", []))).reshape(-1)
        concept = concept_fidelity_from_latents(
            source_latents,
            generated_latents,
            cmask,
            proto_by_latent[target_latent],
            proto_by_latent[source_latent],
            target_latent,
            source_latent,
            assoc_top.get(target_id, []),
            assoc_top.get(source_id, []),
        )
        concept_row = {"method": run.method, "direction": direction, "run_id": run.run_id, **concept}
        concept_rows.append(concept_row)
        row = {
            "method": run.method,
            "direction": direction,
            "run_id": run.run_id,
            "source_label_id": source_id,
            "target_label_id": target_id,
            "target_label": ID_TO_LABEL[target_id],
            "source_pred_label": source_eval["pred_label"],
            "generated_pred_label": generated_eval["pred_label"],
            "target_prob_before": float(target_prob_before),
            "target_prob_after": float(target_prob_after),
            "delta_target_prob": float(target_prob_after - target_prob_before),
            "flip_success": bool(int(generated_eval["pred_label_id"]) == int(target_id)),
            "num_target_cells": int(len(manifest.get("target_cells", []))),
            "num_windows": int(len(manifest.get("window_history", []))),
            "run_dir": str(run.run_dir),
            "source_image_path": str(run.source_image),
            "generated_image_path": str(run.generated_image),
            "source_uni2_grid_path": str(source_grid_path),
            "generated_uni2_grid_path": str(gen_grid_path),
            **rgb_metrics,
            **uni_metrics,
            **consistency,
            **concept,
        }
        metric_rows.append(row)
        for cell_row in cell_rows:
            rgb_cell = rgb_cell_by_key.get((int(cell_row["cell_gx"]), int(cell_row["cell_gy"])), {})
            per_cell_rows.append({"method": run.method, "direction": direction, "run_id": run.run_id, **rgb_cell, **cell_row})

    class_rows = classification_summary(prediction_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for method in [spec.name for spec in specs]:
        rows = [row for row in metric_rows if row["method"] == method]
        if not rows:
            continue
        cls = next((row for row in class_rows if row["method"] == method), {})
        aggregate_rows.append(
            {
                "method": method,
                "n_runs": int(len(rows)),
                "flip_success_rate": float(np.mean([float(bool(row["flip_success"])) for row in rows])),
                "mean_delta_target_prob": float(np.mean([float(row["delta_target_prob"]) for row in rows])),
                "target_accuracy": cls.get("target_accuracy", math.nan),
                "target_f1": cls.get("target_f1", math.nan),
                "target_auroc": cls.get("target_auroc", math.nan),
                "mean_rgb_abs": float(np.mean([float(row["rgb_abs_mean"]) for row in rows])),
                "mean_rgb_abs_target": float(np.mean([float(row["rgb_abs_target_mean"]) for row in rows])),
                "mean_rgb_abs_context": float(np.mean([float(row["rgb_abs_context_mean"]) for row in rows])),
                "mean_lpips": float(np.nanmean([float(row["lpips"]) for row in rows])),
                "mean_ssim": float(np.nanmean([float(row["ssim"]) for row in rows])),
                "mean_uni_l2_target": float(np.mean([float(row["uni_l2_target_mean"]) for row in rows])),
                "mean_uni_l2_context": float(np.mean([float(row["uni_l2_context_mean"]) for row in rows])),
                "mean_delta_target_proto_cos": float(np.nanmean([float(row.get("delta_target_proto_cos", math.nan)) for row in rows])),
                "mean_seam_score_excess_rgb_abs": float(np.nanmean([float(row["seam_score_excess_rgb_abs"]) for row in rows])),
            }
        )

    write_csv(metrics_dir / "benchmark_predictions.csv", prediction_rows)
    write_csv(metrics_dir / "benchmark_classification_summary.csv", class_rows)
    write_csv(metrics_dir / "benchmark_metrics_by_run.csv", metric_rows)
    write_csv(metrics_dir / "benchmark_summary_by_method.csv", aggregate_rows)
    write_csv(metrics_dir / "benchmark_per_cell_metrics.csv", per_cell_rows)
    write_csv(metrics_dir / "benchmark_concept_fidelity.csv", concept_rows)
    write_csv(metrics_dir / "benchmark_window_consistency.csv", consistency_rows)

    summary = {
        "benchmark": "hnscc_hpv_paper_benchmark",
        "out_dir": str(out_dir),
        "metrics_dir": str(metrics_dir),
        "dependency_versions": dep_versions,
        "prediction_method": "image_reencoded_uni2_region_mil",
        "association_csv": "" if assoc_csv is None else str(assoc_csv),
        "metric_files": {
            "by_run": str(metrics_dir / "benchmark_metrics_by_run.csv"),
            "by_method": str(metrics_dir / "benchmark_summary_by_method.csv"),
            "predictions": str(metrics_dir / "benchmark_predictions.csv"),
            "classification": str(metrics_dir / "benchmark_classification_summary.csv"),
            "concept_fidelity": str(metrics_dir / "benchmark_concept_fidelity.csv"),
            "window_consistency": str(metrics_dir / "benchmark_window_consistency.csv"),
            "per_cell": str(metrics_dir / "benchmark_per_cell_metrics.csv"),
        },
        "aggregate_rows": aggregate_rows,
    }
    write_json(metrics_dir / "benchmark_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.region_bank_csv = resolve_path(args.region_bank_csv)
    args.edit_manifest = resolve_path(args.edit_manifest)
    if args.sae_ckpt is None or args.sae_cfg is None:
        proto_meta = read_json(resolve_path(DEFAULT_HNSCC_PROTOTYPE_NPZ).with_suffix(".json"))
        args.sae_ckpt = Path(str(proto_meta.get("sae_ckpt", "")))
        args.sae_cfg = Path(str(proto_meta.get("sae_cfg", "")))
    specs = parse_method_specs(args.policy)
    direction_manifests = write_direction_manifests(args.out_dir, args.edit_manifest, args.region_bank_csv, bool(args.bidirectional), str(args.direction))
    command_rows = [
        {"method": spec.name, "direction": direction, "command": " ".join(shlex.quote(part) for part in generation_command(args, spec, direction, manifest_path))}
        for spec in specs
        for direction, manifest_path in direction_manifests.items()
    ]
    write_json(args.out_dir / "run_commands.json", {"commands": command_rows})
    write_json(args.out_dir / "run_config.json", run_config(args, specs, direction_manifests))

    if args.dry_run:
        print(json.dumps({"commands": command_rows}, indent=2))
        return

    if bool(args.run_edits):
        run_generation(args, specs, direction_manifests)

    if bool(args.prepare_associations) and args.association_csv is None:
        args.association_csv = prepare_hnscc_associations(args, args.out_dir)

    summary = compute_metrics(args, specs, direction_manifests)
    print(json.dumps({"metrics_dir": summary["metrics_dir"], "aggregate_rows": summary["aggregate_rows"]}, indent=2))


if __name__ == "__main__":
    main()
