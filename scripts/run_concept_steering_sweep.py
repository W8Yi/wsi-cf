#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import save_png, write_json
from wsi_cf.common.paths import DEFAULT_SAE_VARIANT, SAE_VARIANTS
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, run_mil_attention
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2
from wsi_cf.steering.progressive import draw_cells_overlay


SETTING_PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "prototype_strength": 0.9,
        "preserve_edit_strength": 0.0,
        "preserve_visited_strength": 0.84,
        "preserve_fresh_context_strength": 0.22,
        "mid_steer_start_ratio": 0.55,
        "mid_steer_end_ratio": 1.0,
        "mid_steer_alpha_start": 0.4,
        "mid_steer_alpha_end": 1.0,
        "edit_support": "center_2x2",
    },
    "low_strength": {"prototype_strength": 0.4},
    "strength_1": {"prototype_strength": 1.0},
    "high_strength": {"prototype_strength": 1.2},
    "very_high_strength": {"prototype_strength": 1.5},
    "stronger_edit_preserve": {
        "prototype_strength": 1.2,
        "preserve_edit_strength": 0.30,
        "preserve_fresh_context_strength": 0.50,
    },
    "early_steer": {
        "prototype_strength": 1.2,
        "mid_steer_start_ratio": 0.25,
        "mid_steer_alpha_start": 0.7,
    },
    "border_relaxed": {
        "prototype_strength": 1.2,
        "edit_support": "border_relaxed",
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a LUAD/LUSC concept-by-setting progressive steering sweep.")
    parser.add_argument("--source-region-bank-csv", type=Path, required=True)
    parser.add_argument("--source-edit-manifest", type=Path, required=True)
    parser.add_argument("--source-label", type=str, required=True)
    parser.add_argument("--target-label", type=str, required=True)
    parser.add_argument("--concept-root", type=Path, required=True)
    parser.add_argument("--concept-modes", type=str, default="labels_only,attention_aware")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="luad_lusc")
    parser.add_argument("--classifier-run-dir", type=Path, default=WSI_CF_ROOT / "artifacts/classifier_training/luad_lusc")
    parser.add_argument("--settings", type=str, default="default,low_strength,high_strength,very_high_strength,stronger_edit_preserve,early_steer,border_relaxed")
    parser.add_argument("--top-concepts", type=int, default=5)
    parser.add_argument("--max-regions", type=int, default=10)
    parser.add_argument("--concept-target-top-k", type=int, default=5)
    parser.add_argument("--concept-target-stat", type=str, default="median", choices=["median", "mean", "q75", "max"])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument("--sae-variant", type=str, default=DEFAULT_SAE_VARIANT, choices=sorted(SAE_VARIANTS))
    parser.add_argument("--output-mode", type=str, default="minimal", choices=["minimal", "debug"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_csv_list(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


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


def safe_token(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p").replace("-", "_")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def concept_dir_for(root: Path, mode: str, task: str, class_label: str) -> Path:
    candidates = [
        root / mode / class_label,
        root / mode / task / class_label,
        root / task / class_label,
    ]
    for cand in candidates:
        if (cand / "selected_concepts.json").exists() and (cand / "representative_tiles.csv").exists():
            return cand
    raise FileNotFoundError(f"Could not find concept bundle for mode={mode}, task={task}, class={class_label} under {root}")


def selected_concepts(concepts_json: Path, *, class_label: str, top_concepts: int) -> list[dict[str, Any]]:
    payload = load_json(concepts_json)
    concepts = [row for row in payload.get("concepts", []) if str(row.get("class_label", class_label)) == str(class_label)]
    concepts.sort(key=lambda row: (int(row.get("concept_rank", 10**9)), -float(row.get("final_score", 0.0)), int(row["latent_idx"])))
    return concepts[: int(top_concepts)]


def write_single_concept_json(src_json: Path, concept: dict[str, Any], out_path: Path) -> None:
    payload = load_json(src_json)
    payload["concepts"] = [concept]
    payload["single_concept_rank"] = int(concept.get("concept_rank", 0))
    payload["single_concept_latent_idx"] = int(concept["latent_idx"])
    write_json(out_path, payload)


def region_manifest_by_id(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    return {str(row["region_id"]): row for row in data}


def parse_target_cells(row: dict[str, Any]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for item in row.get("target_cells", []):
        cells.append((int(item["gx"]), int(item["gy"])))
    return cells


def cells_for_edit_support(cells: list[tuple[int, int]], edit_support: str) -> list[tuple[int, int]]:
    if str(edit_support) == "border_relaxed":
        return list(cells)
    center = {(1, 1), (2, 1), (1, 2), (2, 2)}
    kept = [cell for cell in cells if cell in center]
    return kept if kept else sorted(center)


def eval_classifier(model: torch.nn.Module, z_grid: np.ndarray, *, target_label_id: int, device: torch.device) -> dict[str, Any]:
    x = np.asarray(z_grid, dtype=np.float32).reshape(-1, z_grid.shape[-1])
    _, pred, prob_pred = run_mil_attention(model, x, device=device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    with torch.inference_mode():
        _, y_prob, _, _, _ = model(xt)
        probs = y_prob.detach().cpu().numpy().reshape(-1)
    return {
        "pred": int(pred),
        "prob_pred": float(prob_pred),
        "target_prob": float(probs[int(target_label_id)]),
    }


def encode_image_grid(image_path: Path, *, uni_model: torch.nn.Module, uni_transform: Any, device: torch.device) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    grid = build_uni_grid_from_image(
        img,
        uni_model=uni_model,
        uni_transform=uni_transform,
        grid_step_px=256,
        device=device,
        out_dtype=torch.float32,
    )
    return grid.detach().cpu().numpy().astype(np.float32)


def make_direct_replacement_grid(source_grid: np.ndarray, generated_grid: np.ndarray, cells: list[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(source_grid, dtype=np.float32).copy()
    for gx, gy in cells:
        if 0 <= int(gy) < out.shape[0] and 0 <= int(gx) < out.shape[1]:
            out[int(gy), int(gx), :] = generated_grid[int(gy), int(gx), :]
    return out


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "experiment_args.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "command": " ".join(shlex.quote(part) for part in ([sys.executable, __file__] + (list(argv) if argv is not None else sys.argv[1:]))),
        },
    )

    region_rows = read_csv_rows(args.source_region_bank_csv)[: int(args.max_regions)]
    manifest_by_region = region_manifest_by_id(args.source_edit_manifest)
    modes = parse_csv_list(args.concept_modes)
    settings = parse_csv_list(args.settings)
    for setting in settings:
        if setting not in SETTING_PRESETS:
            raise ValueError(f"Unknown setting preset {setting!r}. Available: {sorted(SETTING_PRESETS)}")
    label_mapping = load_json(args.classifier_run_dir / "label_mapping.json")
    label_to_id = {str(k): int(v) for k, v in label_mapping.get("label_to_id", label_mapping).items()}
    if str(args.target_label) not in label_to_id:
        raise ValueError(f"target_label={args.target_label!r} not in label mapping {label_to_id}")
    target_label_id = int(label_to_id[str(args.target_label)])
    classifier = build_mil_from_checkpoint(args.classifier_run_dir / "best_model.pt", device=device)
    uni_model = None
    uni_transform = None

    manifest_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for mode in modes:
        cdir = concept_dir_for(args.concept_root, mode, str(args.task), str(args.target_label))
        concepts_json = cdir / "selected_concepts.json"
        reps_csv = cdir / "representative_tiles.csv"
        concepts = selected_concepts(concepts_json, class_label=str(args.target_label), top_concepts=int(args.top_concepts))
        for concept in concepts:
            rank = int(concept.get("concept_rank", 0))
            latent = int(concept["latent_idx"])
            single_json = args.out_dir / "_single_concepts" / mode / str(args.target_label) / f"rank_{rank:02d}_latent_{latent}.json"
            write_single_concept_json(concepts_json, concept, single_json)
            for setting_name in settings:
                preset = dict(SETTING_PRESETS["default"])
                preset.update(SETTING_PRESETS[setting_name])
                for region_index, region_row in enumerate(region_rows, start=1):
                    region_id = str(region_row["region_id"])
                    source_request = manifest_by_region.get(region_id)
                    if source_request is None:
                        raise KeyError(f"Missing edit manifest request for region_id={region_id}")
                    source_cells = parse_target_cells(source_request)
                    run_id = (
                        f"{safe_token(args.source_label)}_to_{safe_token(args.target_label)}"
                        f"__{mode}__rank_{rank:02d}_latent_{latent}"
                        f"__{setting_name}__region_{region_index:02d}"
                    )
                    run_dir = args.out_dir / run_id
                    input_dir = run_dir / "_inputs"
                    input_dir.mkdir(parents=True, exist_ok=True)
                    bank_path = input_dir / "region_bank.csv"
                    manifest_path = input_dir / "progressive_edit_manifest.json"
                    single_region = dict(region_row)
                    single_region["region_id"] = region_id
                    write_csv(bank_path, [single_region], fieldnames=list(region_row.keys()))
                    cells = cells_for_edit_support(source_cells, str(preset["edit_support"]))
                    single_request = dict(source_request)
                    single_request["run_id"] = run_id
                    single_request["target_cells"] = [{"gx": int(gx), "gy": int(gy)} for gx, gy in cells]
                    single_request["original_target_cells"] = [{"gx": int(gx), "gy": int(gy)} for gx, gy in source_cells]
                    single_request["source_label"] = str(args.source_label)
                    single_request["target_label"] = str(args.target_label)
                    single_request["concept_mode"] = mode
                    single_request["concept_rank"] = rank
                    single_request["concept_latent_idx"] = latent
                    single_request["setting"] = setting_name
                    write_json(manifest_path, [single_request])

                    overlay_path = run_dir / "edit_overlay.png"
                    source_image = Image.open(region_row["image_path"]).convert("RGB")
                    save_png(draw_cells_overlay(source_image, cells=cells, grid_step_px=int(region_row.get("grid_step_px", 256))), overlay_path)
                    cmd = [
                        sys.executable,
                        str(SCRIPT_DIR / "run_progressive_region_edit.py"),
                        "--task",
                        str(args.task),
                        "--region-bank-csv",
                        str(bank_path),
                        "--edit-manifest",
                        str(manifest_path),
                        "--out-dir",
                        str(run_dir),
                        "--concepts-json",
                        str(single_json),
                        "--representative-tiles-csv",
                        str(reps_csv),
                        "--concept-class-label",
                        str(args.target_label),
                        "--concept-ranking-method",
                        "attention_weighted" if mode == "attention_aware" else "activation",
                        "--concept-target-stat",
                        str(args.concept_target_stat),
                        "--concept-target-top-k",
                        str(args.concept_target_top_k),
                        "--max-concepts",
                        "0",
                        "--target-magnification",
                        str(args.target_magnification),
                        "--sae-variant",
                        str(args.sae_variant),
                        "--prototype-strength",
                        str(preset["prototype_strength"]),
                        "--steer-blend",
                        str(args.steer_blend),
                        "--preserve-edit-strength",
                        str(preset["preserve_edit_strength"]),
                        "--preserve-visited-strength",
                        str(preset["preserve_visited_strength"]),
                        "--preserve-fresh-context-strength",
                        str(preset["preserve_fresh_context_strength"]),
                        "--mid-steer-start-ratio",
                        str(preset["mid_steer_start_ratio"]),
                        "--mid-steer-end-ratio",
                        str(preset["mid_steer_end_ratio"]),
                        "--mid-steer-alpha-start",
                        str(preset["mid_steer_alpha_start"]),
                        "--mid-steer-alpha-end",
                        str(preset["mid_steer_alpha_end"]),
                        "--mid-steer-alpha-schedule",
                        "linear",
                        "--edit-support",
                        str(preset["edit_support"]),
                        "--steps",
                        str(args.steps),
                        "--guidance",
                        str(args.guidance),
                        "--patch-batch",
                        str(args.patch_batch),
                        "--output-mode",
                        str(args.output_mode),
                        "--device",
                        str(args.device),
                    ]
                    generated_path = run_dir / run_id / "generated.png"
                    row_common = {
                        "direction": f"{args.source_label}_to_{args.target_label}",
                        "source_label": str(args.source_label),
                        "target_label": str(args.target_label),
                        "concept_mode": mode,
                        "concept_rank": rank,
                        "latent_idx": latent,
                        "setting": setting_name,
                        "region_index": region_index,
                        "region_id": region_id,
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "source_image_path": str(region_row["image_path"]),
                        "generated_path": str(generated_path),
                        "edit_overlay_path": str(overlay_path),
                        "command": " ".join(shlex.quote(part) for part in cmd),
                    }
                    manifest_rows.append({**row_common, **preset})
                    if bool(args.dry_run):
                        continue
                    if not (bool(args.skip_existing) and generated_path.exists()):
                        subprocess.run(cmd, check=True, cwd=str(WSI_CF_ROOT))
                    shutil.copy2(region_row["image_path"], run_dir / "source_region_actual.png")
                    shutil.copy2(generated_path, run_dir / "generated.png")
                    if uni_model is None or uni_transform is None:
                        uni_model, uni_transform = load_uni2(device)
                    source_grid = np.asarray(np.load(region_row["feature_grid_path"]), dtype=np.float32)
                    generated_grid = encode_image_grid(generated_path, uni_model=uni_model, uni_transform=uni_transform, device=device)
                    direct_grid = make_direct_replacement_grid(source_grid, generated_grid, cells)
                    source_eval = eval_classifier(classifier, source_grid, target_label_id=target_label_id, device=device)
                    generated_eval = eval_classifier(classifier, generated_grid, target_label_id=target_label_id, device=device)
                    direct_eval = eval_classifier(classifier, direct_grid, target_label_id=target_label_id, device=device)
                    result_rows.append(
                        {
                            **row_common,
                            **preset,
                            "source_pred": source_eval["pred"],
                            "source_target_prob": source_eval["target_prob"],
                            "generated_pred": generated_eval["pred"],
                            "generated_target_prob": generated_eval["target_prob"],
                            "generated_target_delta": generated_eval["target_prob"] - source_eval["target_prob"],
                            "generated_flip_to_target": int(generated_eval["pred"] == target_label_id),
                            "direct_replacement_pred": direct_eval["pred"],
                            "direct_replacement_target_prob": direct_eval["target_prob"],
                            "direct_replacement_target_delta": direct_eval["target_prob"] - source_eval["target_prob"],
                            "direct_replacement_flip_to_target": int(direct_eval["pred"] == target_label_id),
                        }
                    )
                    write_csv(args.out_dir / "sweep_results.csv", result_rows)
                    write_csv(args.out_dir / "sweep_manifest.csv", manifest_rows)

    write_csv(args.out_dir / "sweep_manifest.csv", manifest_rows)
    write_csv(args.out_dir / "sweep_results.csv", result_rows)
    summary: dict[str, Any] = {"n_runs_planned": len(manifest_rows), "n_runs_completed": len(result_rows), "groups": {}}
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[(str(row["direction"]), str(row["concept_mode"]), int(row["concept_rank"]), str(row["setting"]))].append(row)
    for key, rows in grouped.items():
        vals = np.asarray([float(row["generated_target_delta"]) for row in rows], dtype=np.float32)
        flips = np.asarray([int(row["generated_flip_to_target"]) for row in rows], dtype=np.float32)
        group_key = "|".join([str(x) for x in key])
        summary["groups"][group_key] = {
            "n": int(len(rows)),
            "mean_generated_target_delta": float(vals.mean()) if vals.size else 0.0,
            "median_generated_target_delta": float(np.median(vals)) if vals.size else 0.0,
            "flip_rate": float(flips.mean()) if flips.size else 0.0,
        }
    write_json(args.out_dir / "summary_by_direction_mode_concept_setting.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
