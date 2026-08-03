#!/usr/bin/env python3
"""Run a controlled 2048x2048 steering-strength sweep and build its paper figure."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wsi_cf.common.paths import SAE_VARIANTS, resolve_sae_paths
from wsi_cf.paper.strength_sweep import (
    DEFAULT_STRENGTHS,
    expand_requests_to_all_region_cells,
    expand_requests_to_random_valid_blocks,
    expand_requests_to_valid_feature_cells,
    json_safe,
    manifest_digest,
    parse_strengths,
    prepare_selected_requests,
    read_json_requests,
    read_region_bank,
    select_random_requests,
    strength_slug,
    summarize_monotonicity,
    write_csv,
)
from wsi_cf.steering.edit_policy import DEFAULT_EDIT_POLICY


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--direction", required=True, help="Paper-facing direction name, such as hpv_pos_to_hpv_neg.")
    parser.add_argument("--runner-direction", required=True, choices=["hpv_pos", "hpv_neg"])
    parser.add_argument("--source-label", default="")
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--label-order", required=True, help="Comma-separated classifier label order.")
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--base-edit-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--request-selector", default="", help="Optional manifest selector filter, for example attention.")
    parser.add_argument("--request-budget", type=int, default=0, help="Optional exact manifest cell-budget filter.")
    parser.add_argument("--region-id", action="append", default=[], help="Use an explicit region instead of random seeded selection.")
    parser.add_argument("--n-regions", type=int, default=1)
    parser.add_argument("--region-size", type=int, default=2048)
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--all-region-cells",
        action="store_true",
        help="Steer every grid cell in each selected region (64 cells for a 2048x2048 region at 256-pixel spacing).",
    )
    mask_group.add_argument(
        "--all-valid-feature-cells",
        action="store_true",
        help="Steer every cell marked valid by each region's valid_feature_mask.npy, excluding missing/blank grid cells.",
    )
    mask_group.add_argument(
        "--center-valid-feature-cells",
        action="store_true",
        help="Steer valid cells in a contiguous center block inset by --center-margin-cells.",
    )
    mask_group.add_argument(
        "--random-valid-block-side-cells",
        type=int,
        default=0,
        help=(
            "Steer one seeded random square block whose cells are all valid; "
            "use 4 for a 1024x1024 block at the standard 256-pixel grid step."
        ),
    )
    parser.add_argument(
        "--center-margin-cells",
        type=int,
        default=1,
        help="Grid-cell margin for --center-valid-feature-cells (1 gives the central 6x6 block of an 8x8 region).",
    )
    parser.add_argument("--strengths", default=",".join(str(item) for item in DEFAULT_STRENGTHS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--classifier-run-dir", type=Path, default=None)
    parser.add_argument("--classifier-ckpt", type=Path, default=None)
    parser.add_argument("--score-scope", choices=["local_region", "slide_bag"], default="local_region")
    parser.add_argument("--local-source", choices=["source_image", "feature_grid"], default="source_image")
    parser.add_argument(
        "--probability-source",
        choices=["selected_cells", "full_region"],
        default="selected_cells",
        help="Plot the targeted cell-replacement probability or the probability from the entire generated region.",
    )
    parser.add_argument("--edit-policy", type=Path, default=DEFAULT_EDIT_POLICY)
    parser.add_argument("--sae-variant", choices=sorted(SAE_VARIANTS), default="relu_sae_base")
    parser.add_argument("--sae-ckpt", type=Path, default=None)
    parser.add_argument("--sae-cfg", type=Path, default=None)
    parser.add_argument("--prototype-npz", type=Path, default=None, help="Optional full-code prototype bundle override.")
    parser.add_argument("--prototype-key", choices=["prototype_mean", "prototype_median"], default="prototype_median")
    parser.add_argument("--concept-latent", type=int, action="append", default=[])
    parser.add_argument("--concepts-json", type=Path, default=None)
    parser.add_argument("--representative-tiles-csv", type=Path, default=None)
    parser.add_argument("--concept-class-label", default="")
    parser.add_argument("--concept-ranking-method", choices=["attention_weighted", "activation"], default="attention_weighted")
    parser.add_argument("--concept-target-stat", choices=["median", "mean", "q75", "max"], default="median")
    parser.add_argument("--concept-target-top-k", type=int, default=5)
    parser.add_argument("--concept-steering-mode", choices=["prototype_vector", "latent_target"], default="prototype_vector")
    parser.add_argument("--max-concepts", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--edit-support", choices=["center_2x2", "padded_center_2x2", "border_relaxed"], default="padded_center_2x2")
    parser.add_argument("--window-stride-cells", type=int, default=1)
    parser.add_argument("--window-selection-mode", choices=["coverage", "overlap"], default="overlap")
    parser.add_argument("--commit-mode", choices=["full_window", "support_cells", "edit_cells"], default="full_window")
    parser.add_argument("--output-mode", choices=["minimal", "debug"], default="minimal")
    parser.add_argument("--runner-extra", default="", help="Extra progressive-runner arguments, parsed with shell-like quoting.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--title", default="")
    return parser


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")


def _git_provenance() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        result = subprocess.run(cmd, cwd=ROOT, check=False, text=True, capture_output=True)
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "python": sys.version,
    }


def _select_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    requests = read_json_requests(_resolve(args.base_edit_manifest))
    if args.request_selector:
        requests = [row for row in requests if str(row.get("selector", "")) == str(args.request_selector)]
    if int(args.request_budget) > 0:
        requests = [
            row
            for row in requests
            if int(row.get("budget") or len(row.get("target_cells", []))) == int(args.request_budget)
        ]
    by_region: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        by_region.setdefault(str(request.get("region_id", "")), []).append(request)
    duplicates = {region_id: len(values) for region_id, values in by_region.items() if len(values) > 1}
    if duplicates:
        preview = ", ".join(f"{region_id} ({count})" for region_id, count in list(sorted(duplicates.items()))[:5])
        raise ValueError(
            "The filtered manifest still has multiple requests per region. "
            f"Use --request-selector and/or --request-budget so the target mask is unambiguous: {preview}"
        )
    region_by_id = read_region_bank(_resolve(args.region_bank_csv))
    if args.region_id:
        wanted = set(args.region_id)
        selected = [row for row in requests if str(row.get("region_id", "")) in wanted]
        missing = wanted - {str(row.get("region_id", "")) for row in selected}
        if missing:
            raise ValueError(f"Explicit region_id values not found in edit manifest: {sorted(missing)}")
    else:
        selected = select_random_requests(
            requests,
            region_by_id,
            seed=int(args.seed),
            n_regions=int(args.n_regions),
            region_size=int(args.region_size),
            source_label=str(args.source_label),
        )
    if args.all_region_cells:
        selected = expand_requests_to_all_region_cells(
            selected,
            region_by_id,
            region_size=int(args.region_size),
        )
    elif args.all_valid_feature_cells or args.center_valid_feature_cells:
        selected = expand_requests_to_valid_feature_cells(
            selected,
            region_by_id,
            root=ROOT,
            center_margin_cells=int(args.center_margin_cells) if args.center_valid_feature_cells else 0,
        )
    elif int(args.random_valid_block_side_cells) > 0:
        selected = expand_requests_to_random_valid_blocks(
            selected,
            region_by_id,
            root=ROOT,
            block_side_cells=int(args.random_valid_block_side_cells),
            seed=int(args.seed),
        )
    return prepare_selected_requests(
        selected,
        task_name=str(args.task_name),
        direction=str(args.direction),
        source_label=str(args.source_label),
        target_label=str(args.target_label),
        seed=int(args.seed),
    )


def _generation_command(
    args: argparse.Namespace,
    *,
    strength: float,
    selected_manifest: Path,
    generated_root: Path,
) -> list[str]:
    cmd = [
        str(args.python),
        "scripts/run_progressive_region_edit.py",
        "--task",
        str(args.task_name),
        "--region-bank-csv",
        str(_resolve(args.region_bank_csv)),
        "--edit-manifest",
        str(selected_manifest),
        "--out-dir",
        str(generated_root),
        "--edit-policy",
        str(_resolve(args.edit_policy)),
        "--direction",
        str(args.runner_direction),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--prototype-strength",
        f"{float(strength):.8g}",
        "--sae-variant",
        str(args.sae_variant),
        "--steps",
        str(int(args.steps)),
        "--patch-batch",
        str(int(args.patch_batch)),
        "--edit-support",
        str(args.edit_support),
        "--window-stride-cells",
        str(int(args.window_stride_cells)),
        "--window-selection-mode",
        str(args.window_selection_mode),
        "--commit-mode",
        str(args.commit_mode),
        "--output-mode",
        str(args.output_mode),
    ]
    if args.sae_ckpt is not None:
        cmd.extend(["--sae-ckpt", str(_resolve(args.sae_ckpt)), "--sae-cfg", str(_resolve(args.sae_cfg))])
    if args.prototype_npz is not None:
        cmd.extend(["--prototype-npz", str(_resolve(args.prototype_npz))])
    cmd.extend(["--prototype-key", str(args.prototype_key)])
    if args.concepts_json is not None:
        if args.representative_tiles_csv is None:
            raise ValueError("--representative-tiles-csv is required with --concepts-json")
        cmd.extend(
            [
                "--concepts-json",
                str(_resolve(args.concepts_json)),
                "--representative-tiles-csv",
                str(_resolve(args.representative_tiles_csv)),
                "--concept-class-label",
                str(args.concept_class_label),
                "--concept-ranking-method",
                str(args.concept_ranking_method),
                "--concept-target-stat",
                str(args.concept_target_stat),
                "--concept-target-top-k",
                str(int(args.concept_target_top_k)),
                "--concept-steering-mode",
                str(args.concept_steering_mode),
                "--max-concepts",
                str(int(args.max_concepts)),
            ]
        )
    if bool(args.skip_existing):
        cmd.append("--skip-existing")
    cmd.extend(shlex.split(str(args.runner_extra)))
    return cmd


def _concept_latents(args: argparse.Namespace) -> list[int]:
    if args.concept_latent:
        return sorted(set(int(item) for item in args.concept_latent))
    if args.concepts_json is not None:
        payload = json.loads(_resolve(args.concepts_json).read_text())
        concepts = list(payload.get("concepts", []))
        if args.concept_class_label:
            concepts = [row for row in concepts if str(row.get("class_label", "")) == str(args.concept_class_label)]
        concepts.sort(
            key=lambda row: (
                int(row.get("concept_rank", 10**9)),
                -float(row.get("final_score", 0.0)),
                int(row["latent_idx"]),
            )
        )
        if int(args.max_concepts) > 0:
            concepts = concepts[: int(args.max_concepts)]
        return [int(row["latent_idx"]) for row in concepts]
    if str(args.task_name) in {"hnscc_hpv", "hnsc_hpv"}:
        task = json.loads((ROOT / "resources/tasks/hnscc_hpv.json").read_text())
        key = "pos_latent" if str(args.runner_direction) == "hpv_pos" else "neg_latent"
        return [int(task["sae"][key])]
    raise ValueError("Provide --concept-latent or --concepts-json so concept activation can be measured")


def _evaluate_strength(
    args: argparse.Namespace,
    *,
    selected_manifest: Path,
    generated_root: Path,
    metrics_dir: Path,
) -> list[dict[str, Any]]:
    import evaluate_prediction_transition_edits as evaluator

    namespace = argparse.Namespace(
        region_bank_csv=_resolve(args.region_bank_csv),
        edit_manifest=selected_manifest,
        generated_root=generated_root,
        classifier_run_dir=_resolve(args.classifier_run_dir),
        classifier_ckpt=_resolve(args.classifier_ckpt),
        out_dir=metrics_dir,
        task_name=str(args.task_name),
        direction=str(args.direction),
        target_label=str(args.target_label),
        label_order=str(args.label_order),
        grade_label_order="GG1,GG2,GG3,GG4,GG5",
        score_scope=str(args.score_scope),
        local_source=str(args.local_source),
        device=str(args.device),
        force_reencode=bool(args.force_reencode),
        allow_missing=False,
        skip_source_grid_check=False,
    )
    evaluator.evaluate(namespace)
    with (metrics_dir / "prediction_transition_by_run.csv").open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _add_concept_activations(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    selected_requests: list[dict[str, Any]],
    latent_ids: list[int],
) -> None:
    import torch

    from wsi_cf.common.runtime import resolve_device
    from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features

    sae_ckpt, sae_cfg = resolve_sae_paths(args.sae_variant, _resolve(args.sae_ckpt), _resolve(args.sae_cfg))
    device = resolve_device(str(args.device))
    sae_model, _, latent_dim = load_sae_from_config(sae_ckpt, sae_cfg, device=str(device))
    if any(latent < 0 or latent >= int(latent_dim) for latent in latent_ids):
        raise ValueError(f"Concept latent(s) {latent_ids} outside SAE latent dimension {latent_dim}")
    cells_by_run = {
        str(request["run_id"]): [
            (int(cell["gx"]), int(cell["gy"])) if isinstance(cell, dict) else (int(cell[0]), int(cell[1]))
            for cell in request["target_cells"]
        ]
        for request in selected_requests
    }
    target_prototype: np.ndarray | None = None
    if args.concepts_json is None and str(args.task_name) in {"hnscc_hpv", "hnsc_hpv"}:
        prototype_path = _resolve(args.prototype_npz) or ROOT / "resources/prototypes/hnscc_hpv/prototype_vectors_for_selected_sae.npz"
        with np.load(prototype_path, allow_pickle=False) as data:
            prototype_ids = np.asarray(data["latent_ids"], dtype=np.int64)
            matches = np.where(prototype_ids == int(latent_ids[0]))[0]
            if matches.size != 1:
                raise ValueError(f"Expected one prototype row for latent {latent_ids[0]} in {prototype_path}")
            target_prototype = np.asarray(data[str(args.prototype_key)][int(matches[0])], dtype=np.float32)
    with torch.inference_mode():
        for row in rows:
            grid = np.asarray(np.load(str(row["encoded_grid_path"])), dtype=np.float32)
            height, width, dim = grid.shape
            flat_indices = [
                int(gy) * int(width) + int(gx)
                for gx, gy in cells_by_run[str(row["run_id"])]
                if 0 <= int(gx) < int(width) and 0 <= int(gy) < int(height)
            ]
            if not flat_indices:
                raise ValueError(f"No target cells are inside encoded grid for run_id={row['run_id']}")
            features = torch.from_numpy(grid.reshape(height * width, dim)).to(device=device, dtype=torch.float32)
            latents = sae_encode_features(sae_model, features)
            selected = latents[torch.as_tensor(flat_indices, device=device, dtype=torch.long)]
            per_latent = selected[:, torch.as_tensor(latent_ids, device=device, dtype=torch.long)].mean(dim=0)
            values = per_latent.detach().cpu().numpy().astype(np.float64)
            row["concept_activation"] = float(values.mean())
            row["concept_latent_ids"] = ";".join(str(item) for item in latent_ids)
            for latent, value in zip(latent_ids, values):
                row[f"concept_activation_z{latent}"] = float(value)
            if target_prototype is not None:
                codes = selected.detach().cpu().numpy().astype(np.float32)
                numerator = np.sum(codes * target_prototype.reshape(1, -1), axis=1)
                denominator = np.maximum(
                    np.linalg.norm(codes, axis=1) * float(np.linalg.norm(target_prototype)),
                    1e-8,
                )
                alignment = float(np.mean(numerator / denominator))
                row["prototype_alignment"] = alignment
                row["concept_score"] = alignment
                row["concept_score_name"] = f"Target prototype cosine (selector z{latent_ids[0]})"
            else:
                row["prototype_alignment"] = ""
                row["concept_score"] = float(row["concept_activation"])
                row["concept_score_name"] = f"Target SAE activation (z{row['concept_latent_ids']})"


def main() -> None:
    args = build_arg_parser().parse_args()
    if (args.sae_ckpt is None) != (args.sae_cfg is None):
        raise ValueError("--sae-ckpt and --sae-cfg must be supplied together")
    if args.classifier_run_dir is None and args.classifier_ckpt is None:
        raise ValueError("Provide --classifier-run-dir or --classifier-ckpt")
    strengths = parse_strengths(args.strengths)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_requests = _select_requests(args)
    selected_manifest = out_dir / "selected_edit_manifest.json"
    _write_json(selected_manifest, selected_requests)
    latent_ids = _concept_latents(args)

    commands: list[dict[str, Any]] = []
    for strength in strengths:
        slug = strength_slug(strength)
        generated_root = out_dir / "generated" / slug
        cmd = _generation_command(
            args,
            strength=strength,
            selected_manifest=selected_manifest,
            generated_root=generated_root,
        )
        commands.append({"strength": strength, "command": cmd})
    provenance = {
        "task_name": str(args.task_name),
        "direction": str(args.direction),
        "strengths": strengths,
        "seed": int(args.seed),
        "region_size": int(args.region_size),
        "selected_region_ids": [str(row["region_id"]) for row in selected_requests],
        "selected_manifest": str(selected_manifest),
        "selected_manifest_sha256": manifest_digest(selected_requests),
        "target_concept_latents": latent_ids,
        "commands": commands,
        "git": _git_provenance(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    _write_json(out_dir / "sweep_provenance.json", provenance)

    if args.dry_run:
        for item in commands:
            print(f"[{strength_slug(float(item['strength']))}] {shlex.join(item['command'])}")
        print(selected_manifest)
        return

    if not args.skip_generation:
        for item in commands:
            print(f"[generate] {strength_slug(float(item['strength']))}", flush=True)
            subprocess.run(item["command"], cwd=ROOT, check=True)

    combined: list[dict[str, Any]] = []
    if not args.skip_evaluation:
        for strength in strengths:
            slug = strength_slug(strength)
            print(f"[evaluate] {slug}", flush=True)
            rows = _evaluate_strength(
                args,
                selected_manifest=selected_manifest,
                generated_root=out_dir / "generated" / slug,
                metrics_dir=out_dir / "metrics" / slug,
            )
            for row in rows:
                probability_key = (
                    "edited_target_probability"
                    if str(args.probability_source) == "selected_cells"
                    else "full_region_target_probability"
                )
                row.update(
                    {
                        "steering_strength": float(strength),
                        "seed": int(args.seed),
                        "target_probability": float(row[probability_key]),
                        "target_probability_source": str(args.probability_source),
                        "image_path": str(out_dir / "generated" / slug / str(row["run_id"]) / "generated.png"),
                    }
                )
            combined.extend(rows)
        _add_concept_activations(args, combined, selected_requests=selected_requests, latent_ids=latent_ids)
        combined.sort(key=lambda row: (str(row["region_id"]), float(row["steering_strength"])))
        metrics_csv = out_dir / "steering_strength_metrics.csv"
        write_csv(metrics_csv, combined)
        monotonicity = summarize_monotonicity(
            combined,
            value_columns=("target_probability", "concept_score"),
        )
        write_csv(out_dir / "steering_strength_monotonicity.csv", monotonicity)
        _write_json(
            out_dir / "steering_strength_summary.json",
            {
                "n_regions": len(selected_requests),
                "n_strengths": len(strengths),
                "target_concept_latents": latent_ids,
                "monotonicity": monotonicity,
            },
        )
    else:
        metrics_csv = out_dir / "steering_strength_metrics.csv"

    if not args.skip_plot:
        if not metrics_csv.is_file():
            raise FileNotFoundError(f"Cannot plot without metrics CSV: {metrics_csv}")
        plot_cmd = [
            str(args.python),
            "scripts/plot_steering_strength_sweep.py",
            "--metrics-csv",
            str(metrics_csv),
            "--out-dir",
            str(out_dir / "figures"),
            "--title",
            str(args.title or f"{args.task_name}: {args.direction}"),
        ]
        subprocess.run(plot_cmd, cwd=ROOT, check=True)
    print(out_dir)


if __name__ == "__main__":
    main()
