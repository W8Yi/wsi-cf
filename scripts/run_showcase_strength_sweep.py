#!/usr/bin/env python
"""Run and evaluate original-28 showcase steering strength sweep.

This is a focused diagnostic runner for the figure4 showcase case. It keeps the
original 28-cell manifest fixed and sweeps SAE prototype strength, mid-steering
timing, and preservation strengths using the canonical showcase SAE/prototype.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACK = ROOT / "artifacts/showcase_regions/figure4_element_pack"
SELECTOR = ROOT / "artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_clean_sae_expand_attn_seed_p90"
CANONICAL_RUN = ROOT / (
    "artifacts/showcase_regions/"
    "TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_progressive_clean_sae_p90_to_hpvneg_border_relaxed/"
    "case_clean_sae_expand_attn_seed_p90__image_first_tile_selection"
)
RUNNER = ROOT / "scripts/run_progressive_region_edit.py"
SAE_CKPT = Path("/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt")
SAE_CFG = Path("/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json")
PROTO_NPZ = ROOT / "artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"
RUN_ID = "case_clean_sae_expand_attn_seed_p90__image_first_tile_selection"


@dataclass(frozen=True)
class Candidate:
    name: str
    prototype_strength: float
    preserve_edit: float
    preserve_visited: float
    preserve_fresh: float
    mid_start: float
    mid_end: float = 1.0
    alpha_start: float = 0.60
    alpha_end: float = 1.0

    @property
    def out_dir(self) -> Path:
        return PACK / "candidate_runs" / f"original28_sweep_{self.name}"

    @property
    def image_path(self) -> Path:
        return self.out_dir / RUN_ID / "generated.png"


BASELINE_IMAGES = {
    "source": CANONICAL_RUN / "source_region_actual.png",
    "canonical_0p8": CANONICAL_RUN / "generated.png",
    "moderate_0p9_existing": PACK / "candidate_runs/original28_strength_moderate_canonical_sae" / RUN_ID / "generated.png",
    "strong_preserved_1p0_existing": PACK / "candidate_runs/original28_strength_strong_preserved_canonical_sae" / RUN_ID / "generated.png",
    "aggressive_1p0_existing": PACK / "candidate_runs/original28_stronger_hpvneg_canonical_sae" / RUN_ID / "generated.png",
}


CANDIDATES: list[Candidate] = [
    Candidate("p095_pe003_pv084_pf022_mid048", 0.95, 0.03, 0.84, 0.22, 0.48),
    Candidate("p100_pe002_pv084_pf022_mid048", 1.00, 0.02, 0.84, 0.22, 0.48),
    Candidate("p100_pe001_pv084_pf022_mid048", 1.00, 0.01, 0.84, 0.22, 0.48),
    Candidate("p100_pe000_pv084_pf022_mid048", 1.00, 0.00, 0.84, 0.22, 0.48),
    Candidate("p100_pe002_pv080_pf020_mid048", 1.00, 0.02, 0.80, 0.20, 0.48),
    Candidate("p100_pe001_pv080_pf020_mid048", 1.00, 0.01, 0.80, 0.20, 0.48),
    Candidate("p100_pe000_pv080_pf020_mid048", 1.00, 0.00, 0.80, 0.20, 0.48),
    Candidate("p095_pe000_pv080_pf020_mid048", 0.95, 0.00, 0.80, 0.20, 0.48),
    Candidate("p090_pe000_pv080_pf020_mid048", 0.90, 0.00, 0.80, 0.20, 0.48),
    Candidate("p100_pe002_pv078_pf018_mid048", 1.00, 0.02, 0.78, 0.18, 0.48),
    Candidate("p100_pe001_pv078_pf018_mid048", 1.00, 0.01, 0.78, 0.18, 0.48),
    Candidate("p095_pe001_pv078_pf018_mid048", 0.95, 0.01, 0.78, 0.18, 0.48),
    Candidate("p090_pe000_pv075_pf015_mid045", 0.90, 0.00, 0.75, 0.15, 0.45),
    Candidate("p095_pe000_pv075_pf015_mid045", 0.95, 0.00, 0.75, 0.15, 0.45),
    Candidate("p100_pe003_pv088_pf025_mid040_alpha070", 1.00, 0.03, 0.88, 0.25, 0.40, alpha_start=0.70),
    Candidate("p100_pe000_pv075_pf015_mid055_alpha050", 1.00, 0.00, 0.75, 0.15, 0.55, alpha_start=0.50),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true", help="Run missing generation candidates.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate generated candidates.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing candidate outputs.")
    return parser.parse_args()


def run_candidate(candidate: Candidate, *, args: argparse.Namespace) -> None:
    if candidate.image_path.exists() and not args.force:
        print(f"[skip] {candidate.name}: generated.png exists")
        return
    cmd = [
        args.python,
        str(RUNNER),
        "--region-bank-csv",
        str(SELECTOR / "region_bank.csv"),
        "--edit-manifest",
        str(SELECTOR / "progressive_edit_manifest.json"),
        "--out-dir",
        str(candidate.out_dir),
        "--direction",
        "hpv_neg",
        "--edit-support",
        "border_relaxed",
        "--seed",
        "7",
        "--device",
        args.device,
        "--dtype",
        "fp16",
        "--steps",
        "30",
        "--guidance",
        "2.0",
        "--patch-batch",
        "128",
        "--prototype-strength",
        str(candidate.prototype_strength),
        "--steer-blend",
        "1.0",
        "--preserve-edit-strength",
        str(candidate.preserve_edit),
        "--preserve-visited-strength",
        str(candidate.preserve_visited),
        "--preserve-fresh-context-strength",
        str(candidate.preserve_fresh),
        "--mid-steer-start-ratio",
        str(candidate.mid_start),
        "--mid-steer-end-ratio",
        str(candidate.mid_end),
        "--mid-steer-alpha-start",
        str(candidate.alpha_start),
        "--mid-steer-alpha-end",
        str(candidate.alpha_end),
        "--mid-steer-alpha-schedule",
        "linear",
        "--sae-ckpt",
        str(SAE_CKPT),
        "--sae-cfg",
        str(SAE_CFG),
        "--prototype-npz",
        str(PROTO_NPZ),
        "--output-mode",
        "debug",
    ]
    log_dir = PACK / "logs/strength_sweep_original28"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{candidate.name}.log"
    print(f"[run] {candidate.name} -> {log_path}")
    with log_path.open("w") as log_f:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log_f, stderr=subprocess.STDOUT)


def steer_score(meta: dict[str, Any]) -> float:
    if meta.get("kind") == "source":
        return 0.0
    proto = float(meta["prototype_strength"])
    preserve_edit = float(meta["preserve_edit"])
    preserve_visited = float(meta["preserve_visited"])
    preserve_fresh = float(meta["preserve_fresh"])
    mid_start = float(meta["mid_start"])
    alpha_start = float(meta["alpha_start"])
    return (
        proto
        + 2.0 * (0.05 - preserve_edit)
        + 1.0 * (0.95 - preserve_visited)
        + 0.75 * (0.35 - preserve_fresh)
        + 0.35 * (0.50 - mid_start)
        + 0.25 * (alpha_start - 0.50)
    )


def evaluate(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(SRC))
    import torch
    import matplotlib.pyplot as plt

    from wsi_cf.common.paths import DEFAULT_HNSCC_MIL_CKPT
    from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, run_mil_attention
    from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2

    rows: list[dict[str, Any]] = []
    image_items: list[tuple[str, Path, dict[str, Any]]] = []
    for name, path in BASELINE_IMAGES.items():
        if not path.exists():
            continue
        if name == "source":
            meta = {"kind": "source"}
        elif name == "canonical_0p8":
            meta = {
                "kind": "baseline",
                "prototype_strength": 0.8,
                "preserve_edit": 0.05,
                "preserve_visited": 0.95,
                "preserve_fresh": 0.35,
                "mid_start": 0.50,
                "alpha_start": 0.50,
            }
        elif name == "moderate_0p9_existing":
            meta = {
                "kind": "existing",
                "prototype_strength": 0.9,
                "preserve_edit": 0.03,
                "preserve_visited": 0.90,
                "preserve_fresh": 0.25,
                "mid_start": 0.50,
                "alpha_start": 0.55,
            }
        elif name == "strong_preserved_1p0_existing":
            meta = {
                "kind": "existing",
                "prototype_strength": 1.0,
                "preserve_edit": 0.03,
                "preserve_visited": 0.88,
                "preserve_fresh": 0.25,
                "mid_start": 0.48,
                "alpha_start": 0.60,
            }
        else:
            meta = {
                "kind": "existing",
                "prototype_strength": 1.0,
                "preserve_edit": 0.0,
                "preserve_visited": 0.75,
                "preserve_fresh": 0.15,
                "mid_start": 0.45,
                "alpha_start": 0.60,
            }
        image_items.append((name, path, meta))
    for candidate in CANDIDATES:
        if candidate.image_path.exists():
            meta = asdict(candidate)
            meta["kind"] = "sweep"
            image_items.append((candidate.name, candidate.image_path, meta))

    regions = {
        "whole": None,
        "top_left": (0, 0, 1024, 1024),
        "top_right": (1024, 0, 2048, 1024),
        "bottom_left": (0, 1024, 1024, 2048),
        "bottom_right": (1024, 1024, 2048, 2048),
    }
    out_data = PACK / "data"
    out_charts = PACK / "charts"
    out_quads = PACK / "quadrants/strength_sweep_original28"
    out_grids = out_data / "encoded_uni2_grids/strength_sweep_original28"
    out_cmp = PACK / "comparison/strength_sweep_original28"
    for path in (out_data, out_charts, out_quads, out_grids, out_cmp):
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    uni_model, uni_transform = load_uni2(device)
    mil_model = build_mil_from_checkpoint(DEFAULT_HNSCC_MIL_CKPT, device=device)
    source = np.asarray(Image.open(BASELINE_IMAGES["source"]).convert("RGB"), dtype=np.float32)
    for sample, image_path, meta in image_items:
        image = Image.open(image_path).convert("RGB")
        arr = np.asarray(image, dtype=np.float32)
        diff = np.abs(arr - source)
        pix = diff.mean(axis=2)
        if sample != "source":
            cmp_dir = out_cmp / sample
            cmp_dir.mkdir(parents=True, exist_ok=True)
            image.save(cmp_dir / "generated.png")
        for region_name, box in regions.items():
            crop = image if box is None else image.crop(box)
            crop_path = image_path if box is None else out_quads / f"{sample}_{region_name}.png"
            if box is not None:
                crop.save(crop_path)
            z_t = build_uni_grid_from_image(
                crop,
                uni_model=uni_model,
                uni_transform=uni_transform,
                grid_step_px=256,
                device=device,
                out_dtype=torch.float32,
            )
            z = z_t.detach().cpu().numpy().astype(np.float32)
            grid_path = out_grids / f"{sample}_{region_name}_uni2_grid.npy"
            np.save(grid_path, z)
            _, pred, prob_pos = run_mil_attention(mil_model, z.reshape(-1, z.shape[-1]), device=device)
            row = {
                "sample": sample,
                "region": region_name,
                "pred_label": "HPV+" if int(pred) == 1 else "HPV-",
                "prob_hpv_pos": float(prob_pos),
                "prob_hpv_neg": float(1.0 - prob_pos),
                "image_path": str(crop_path),
                "uni2_grid_path": str(grid_path),
                "steer_score": float(steer_score(meta)),
                **meta,
            }
            if region_name == "whole":
                row.update(
                    {
                        "mean_abs_rgb_diff_vs_source_0_255": float(diff.mean()),
                        "median_abs_rgb_diff_vs_source_0_255": float(np.median(pix)),
                        "pct_pixels_abs_rgb_diff_gt_25": float((pix > 25).mean()),
                    }
                )
            rows.append(row)

    csv_path = out_data / "hpv_original28_comprehensive_strength_sweep.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_data / "hpv_original28_comprehensive_strength_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")

    whole = [row for row in rows if row["region"] == "whole"]
    flips = [row for row in whole if row["pred_label"] == "HPV-"]
    flips_sorted = sorted(flips, key=lambda row: (float(row["steer_score"]), float(row["mean_abs_rgb_diff_vs_source_0_255"])))
    rank_csv = out_data / "hpv_original28_strength_sweep_flip_ranking.csv"
    with rank_csv.open("w", newline="") as f:
        fieldnames = [
            "rank",
            "sample",
            "prob_hpv_neg",
            "steer_score",
            "mean_abs_rgb_diff_vs_source_0_255",
            "prototype_strength",
            "preserve_edit",
            "preserve_visited",
            "preserve_fresh",
            "mid_start",
            "alpha_start",
            "image_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(flips_sorted, start=1):
            writer.writerow({"rank": rank, **{key: row.get(key, "") for key in fieldnames if key != "rank"}})

    plot_rows = sorted([row for row in whole if row["sample"] != "source"], key=lambda row: float(row["steer_score"]))
    x = np.arange(len(plot_rows))
    fig, ax1 = plt.subplots(figsize=(max(8.0, 0.45 * len(plot_rows)), 3.8), dpi=220)
    neg = [float(row["prob_hpv_neg"]) for row in plot_rows]
    colors = ["#4c8f8a" if row["pred_label"] == "HPV-" else "#7aa6c2" for row in plot_rows]
    ax1.bar(x, neg, color=colors)
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax1.set_ylim(0, 1.04)
    ax1.set_ylabel("HPV- probability")
    ax1.set_xticks(x)
    ax1.set_xticklabels([row["sample"].replace("_", "\n") for row in plot_rows], fontsize=5.5)
    ax2 = ax1.twinx()
    ax2.plot(x, [float(row["mean_abs_rgb_diff_vs_source_0_255"]) for row in plot_rows], color="#333333", marker="o", linewidth=1.2, markersize=2.5)
    ax2.set_ylabel("Mean abs RGB diff")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    chart_path = out_charts / "hpv_original28_comprehensive_strength_sweep.png"
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    print(csv_path)
    print(rank_csv)
    print(chart_path)
    if flips_sorted:
        best = flips_sorted[0]
        print("[best_flip]", best["sample"], "neg=", f"{float(best['prob_hpv_neg']):.6f}", "score=", f"{float(best['steer_score']):.4f}", "diff=", f"{float(best['mean_abs_rgb_diff_vs_source_0_255']):.2f}")
    else:
        print("[best_flip] none")


def main() -> None:
    args = parse_args()
    if not args.run and not args.evaluate:
        args.run = True
        args.evaluate = True
    if args.run:
        for candidate in CANDIDATES:
            run_candidate(candidate, args=args)
    if args.evaluate:
        evaluate(args)


if __name__ == "__main__":
    main()
