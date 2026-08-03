#!/usr/bin/env python3
"""Sweep the fraction of valid tissue cells at one fixed steering strength."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import shutil
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

import run_steering_strength_sweep as strength_runner
from wsi_cf.common.paths import SAE_VARIANTS, resolve_sae_paths
from wsi_cf.paper.strength_sweep import (
    expand_requests_to_valid_feature_cells,
    json_safe,
    manifest_digest,
    monotonicity_record,
    prepare_selected_requests,
    read_json_requests,
    read_region_bank,
    select_random_requests,
    write_csv,
)
from wsi_cf.steering.edit_policy import DEFAULT_EDIT_POLICY


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--runner-direction", required=True, choices=["hpv_pos", "hpv_neg"])
    parser.add_argument("--source-label", default="")
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--label-order", required=True)
    parser.add_argument("--region-bank-csv", type=Path, required=True)
    parser.add_argument("--base-edit-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--region-id", action="append", default=[])
    parser.add_argument("--n-regions", type=int, default=1)
    parser.add_argument("--region-size", type=int, default=2048)
    parser.add_argument("--fractions", default="0,0.2,0.4,0.6,0.8,1")
    parser.add_argument("--fixed-strength", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--classifier-run-dir", type=Path, default=None)
    parser.add_argument("--classifier-ckpt", type=Path, default=None)
    parser.add_argument("--score-scope", choices=["local_region", "slide_bag"], default="local_region")
    parser.add_argument("--local-source", choices=["source_image", "feature_grid"], default="source_image")
    parser.add_argument("--edit-policy", type=Path, default=DEFAULT_EDIT_POLICY)
    parser.add_argument("--sae-variant", choices=sorted(SAE_VARIANTS), default="relu_sae_base")
    parser.add_argument("--sae-ckpt", type=Path, default=None)
    parser.add_argument("--sae-cfg", type=Path, default=None)
    parser.add_argument("--prototype-npz", type=Path, default=None)
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
    parser.add_argument("--patch-batch", type=int, default=128)
    parser.add_argument("--edit-support", choices=["center_2x2", "padded_center_2x2", "border_relaxed"], default="padded_center_2x2")
    parser.add_argument("--window-stride-cells", type=int, default=1)
    parser.add_argument("--window-selection-mode", choices=["coverage", "overlap"], default="overlap")
    parser.add_argument("--commit-mode", choices=["full_window", "support_cells", "edit_cells"], default="full_window")
    parser.add_argument("--output-mode", choices=["minimal", "debug"], default="minimal")
    parser.add_argument("--runner-extra", default="")
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


def parse_fractions(text: str) -> tuple[float, ...]:
    values = tuple(sorted({float(token.strip()) for token in str(text).split(",") if token.strip()}))
    if not values or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("--fractions must contain finite values in [0, 1]")
    return values


def fraction_slug(value: float) -> str:
    return f"fraction_{int(round(100.0 * float(value))):03d}"


def _select_full_valid_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    requests = read_json_requests(_resolve(args.base_edit_manifest))
    by_region: dict[str, dict[str, Any]] = {}
    for request in requests:
        by_region.setdefault(str(request.get("region_id", "")), request)
    unique_requests = list(by_region.values())
    region_by_id = read_region_bank(_resolve(args.region_bank_csv))
    if args.region_id:
        wanted = set(args.region_id)
        selected = [request for request in unique_requests if str(request.get("region_id", "")) in wanted]
        missing = wanted - {str(request.get("region_id", "")) for request in selected}
        if missing:
            raise ValueError(f"Explicit region_id values not found: {sorted(missing)}")
    else:
        selected = select_random_requests(
            unique_requests,
            region_by_id,
            seed=int(args.seed),
            n_regions=int(args.n_regions),
            region_size=int(args.region_size),
            source_label=str(args.source_label),
        )
    selected = expand_requests_to_valid_feature_cells(selected, region_by_id, root=ROOT)
    return prepare_selected_requests(
        selected,
        task_name=str(args.task_name),
        direction=str(args.direction),
        source_label=str(args.source_label),
        target_label=str(args.target_label),
        seed=int(args.seed),
    )


def _stable_cell_order(request: dict[str, Any], seed: int) -> list[dict[str, int]]:
    region_id = str(request["region_id"])

    def key(cell: dict[str, Any]) -> bytes:
        payload = f"{int(seed)}:{region_id}:{int(cell['gx'])}:{int(cell['gy'])}".encode()
        return hashlib.sha256(payload).digest()

    return [
        {"gx": int(cell["gx"]), "gy": int(cell["gy"])}
        for cell in sorted(request["target_cells"], key=key)
    ]


def _fraction_requests(
    full_requests: list[dict[str, Any]],
    fractions: tuple[float, ...],
    *,
    seed: int,
) -> dict[float, list[dict[str, Any]]]:
    output: dict[float, list[dict[str, Any]]] = {fraction: [] for fraction in fractions}
    for request in full_requests:
        ordered = _stable_cell_order(request, seed)
        for fraction in fractions:
            count = len(ordered) if fraction >= 1.0 else int(round(fraction * len(ordered)))
            count = max(0, min(count, len(ordered)))
            row = dict(request)
            row.update(
                {
                    "selector": "seeded_nested_valid_fraction",
                    "target_cell_mode": "seeded_nested_valid_fraction",
                    "target_cells": ordered[:count],
                    "budget": count,
                    "budget_fraction": float(fraction),
                    "budget_is_full": int(count == len(ordered)),
                    "tile_fraction": float(fraction),
                    "valid_cell_count": len(ordered),
                    "fixed_steering_strength": float(0.0),
                    "cell_order_seed": int(seed),
                }
            )
            output[fraction].append(row)
    return output


def _prepare_zero_baseline(
    args: argparse.Namespace,
    requests: list[dict[str, Any]],
    full_requests: list[dict[str, Any]],
    generated_root: Path,
) -> None:
    region_by_id = read_region_bank(_resolve(args.region_bank_csv))
    full_by_region = {str(request["region_id"]): request for request in full_requests}
    for request in requests:
        region = region_by_id[str(request["region_id"])]
        source = _resolve(Path(str(region["image_path"])))
        run_dir = generated_root / str(request["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, run_dir / "generated.png")
        shutil.copy2(source, run_dir / "source_region_actual.png")
        full_request = full_by_region[str(request["region_id"])]
        _write_json(
            run_dir / "run_manifest.json",
            {
                "run_id": str(request["run_id"]),
                "region_id": str(request["region_id"]),
                "source_image_path": str(source),
                "grid_step_px": int(float(region["grid_step_px"])),
                "target_cells": [],
                "eligible_cells": list(full_request["target_cells"]),
                "baseline": "untouched_source",
            },
        )


def _add_fixed_support_concept_scores(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    request_by_fraction: dict[float, list[dict[str, Any]]],
    full_requests: list[dict[str, Any]],
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
    full_by_region = {str(request["region_id"]): request for request in full_requests}
    cells_by_fraction_run = {
        (float(fraction), str(request["run_id"])): [
            (int(cell["gx"]), int(cell["gy"])) for cell in request["target_cells"]
        ]
        for fraction, requests in request_by_fraction.items()
        for request in requests
    }
    with torch.inference_mode():
        for row in rows:
            fraction = float(row["tile_fraction"])
            generated = np.asarray(np.load(str(row["encoded_grid_path"])), dtype=np.float32)
            source_path = (
                Path(str(row["encoded_grid_path"])).parents[1]
                / "encoded_source_grids"
                / f"{row['region_id']}.npy"
            )
            source = np.asarray(np.load(source_path), dtype=np.float32)
            hybrid = source.copy()
            for gx, gy in cells_by_fraction_run[(fraction, str(row["run_id"]))]:
                hybrid[gy, gx] = generated[gy, gx]
            valid_cells = [
                (int(cell["gx"]), int(cell["gy"]))
                for cell in full_by_region[str(row["region_id"])]["target_cells"]
            ]
            features = torch.from_numpy(hybrid.reshape(-1, hybrid.shape[-1])).to(device=device, dtype=torch.float32)
            latents = sae_encode_features(sae_model, features)
            width = int(hybrid.shape[1])
            indices = torch.as_tensor([gy * width + gx for gx, gy in valid_cells], device=device, dtype=torch.long)
            selected = latents[indices][:, torch.as_tensor(latent_ids, device=device, dtype=torch.long)]
            values = selected.mean(dim=0).detach().cpu().numpy().astype(np.float64)
            row["concept_activation"] = float(values.mean())
            row["concept_latent_ids"] = ";".join(str(item) for item in latent_ids)
            for latent, value in zip(latent_ids, values):
                row[f"concept_activation_z{latent}"] = float(value)
            row["concept_score"] = float(row["concept_activation"])
            row["concept_score_name"] = "Target SAE concept activation"


def main() -> None:
    args = build_arg_parser().parse_args()
    if not 0.0 <= float(args.fixed_strength) <= 1.0:
        raise ValueError("--fixed-strength must be in [0, 1]")
    if (args.sae_ckpt is None) != (args.sae_cfg is None):
        raise ValueError("--sae-ckpt and --sae-cfg must be supplied together")
    if args.classifier_run_dir is None and args.classifier_ckpt is None:
        raise ValueError("Provide --classifier-run-dir or --classifier-ckpt")
    fractions = parse_fractions(args.fractions)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_requests = _select_full_valid_requests(args)
    requests_by_fraction = _fraction_requests(full_requests, fractions, seed=int(args.seed))
    for requests in requests_by_fraction.values():
        for request in requests:
            request["fixed_steering_strength"] = float(args.fixed_strength)
    latent_ids = strength_runner._concept_latents(args)

    commands: list[dict[str, Any]] = []
    all_requests: list[dict[str, Any]] = []
    for fraction in fractions:
        slug = fraction_slug(fraction)
        manifest = out_dir / "manifests" / f"{slug}.json"
        requests = requests_by_fraction[fraction]
        _write_json(manifest, requests)
        all_requests.extend(requests)
        if fraction > 0.0:
            command = strength_runner._generation_command(
                args,
                strength=float(args.fixed_strength),
                selected_manifest=manifest,
                generated_root=out_dir / "generated" / slug,
            )
            commands.append({"tile_fraction": fraction, "command": command})
    _write_json(out_dir / "all_fraction_requests.json", all_requests)
    _write_json(
        out_dir / "sweep_provenance.json",
        {
            "task_name": str(args.task_name),
            "direction": str(args.direction),
            "fractions": fractions,
            "fixed_strength": float(args.fixed_strength),
            "seed": int(args.seed),
            "selection": "seeded nested random ordering of valid_feature_mask cells",
            "selected_region_ids": [str(request["region_id"]) for request in full_requests],
            "valid_cell_counts": {str(request["region_id"]): len(request["target_cells"]) for request in full_requests},
            "all_requests_sha256": manifest_digest(all_requests),
            "target_concept_latents": latent_ids,
            "commands": commands,
            "git": strength_runner._git_provenance(),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )
    if args.dry_run:
        for item in commands:
            print(f"[{fraction_slug(float(item['tile_fraction']))}] {shlex.join(item['command'])}")
        return

    zero_fractions = [fraction for fraction in fractions if fraction == 0.0]
    if zero_fractions:
        _prepare_zero_baseline(
            args,
            requests_by_fraction[0.0],
            full_requests,
            out_dir / "generated" / fraction_slug(0.0),
        )
    if not args.skip_generation:
        for item in commands:
            print(f"[generate] {fraction_slug(float(item['tile_fraction']))}", flush=True)
            subprocess.run(item["command"], cwd=ROOT, check=True)

    combined: list[dict[str, Any]] = []
    metrics_csv = out_dir / "cell_fraction_metrics.csv"
    if not args.skip_evaluation:
        for fraction in fractions:
            slug = fraction_slug(fraction)
            print(f"[evaluate] {slug}", flush=True)
            rows = strength_runner._evaluate_strength(
                args,
                selected_manifest=out_dir / "manifests" / f"{slug}.json",
                generated_root=out_dir / "generated" / slug,
                metrics_dir=out_dir / "metrics" / slug,
            )
            for row in rows:
                row.update(
                    {
                        "tile_fraction": float(fraction),
                        "steering_strength": float(args.fixed_strength),
                        "seed": int(args.seed),
                        "target_probability": float(row["edited_target_probability"]),
                        "target_probability_source": "full_local_region_with_selected_cell_replacement",
                        "image_path": str(out_dir / "generated" / slug / str(row["run_id"]) / "generated.png"),
                    }
                )
            combined.extend(rows)
        _add_fixed_support_concept_scores(
            args,
            combined,
            request_by_fraction=requests_by_fraction,
            full_requests=full_requests,
            latent_ids=latent_ids,
        )
        combined.sort(key=lambda row: (str(row["region_id"]), float(row["tile_fraction"])))
        write_csv(metrics_csv, combined)
        monotonicity: list[dict[str, Any]] = []
        for region_id in sorted({str(row["region_id"]) for row in combined}):
            region_rows = [row for row in combined if str(row["region_id"]) == region_id]
            for metric in ("target_probability", "concept_score"):
                record = monotonicity_record(
                    [float(row["tile_fraction"]) for row in region_rows],
                    [float(row[metric]) for row in region_rows],
                )
                monotonicity.append({"region_id": region_id, "metric": metric, **record})
        write_csv(out_dir / "cell_fraction_monotonicity.csv", monotonicity)
        _write_json(
            out_dir / "cell_fraction_summary.json",
            {
                "n_regions": len(full_requests),
                "fractions": fractions,
                "fixed_strength": float(args.fixed_strength),
                "target_concept_latents": latent_ids,
                "monotonicity": monotonicity,
            },
        )

    if not args.skip_plot:
        plot_command = [
            str(args.python),
            "scripts/plot_steering_strength_sweep.py",
            "--metrics-csv",
            str(metrics_csv),
            "--out-dir",
            str(out_dir / "figures"),
            "--prefix",
            "cell_fraction",
            "--sweep-column",
            "tile_fraction",
            "--sweep-label",
            "Edited valid tissue tiles (%)",
            "--show-source",
            "--hide-title",
            "--hide-footer",
            "--dpi",
            "300",
        ]
        subprocess.run(plot_command, cwd=ROOT, check=True)
    print(out_dir)


if __name__ == "__main__":
    main()
