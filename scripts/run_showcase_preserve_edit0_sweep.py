#!/usr/bin/env python
"""Sweep edit strength/duration with preserve_edit_strength fixed at zero."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACK = ROOT / "artifacts/showcase_regions/figure4_element_pack"
SELECTOR = ROOT / "artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_clean_sae_expand_attn_seed_p90"
RUNNER = ROOT / "scripts/run_progressive_region_edit.py"
SAE_CKPT = Path("/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt")
SAE_CFG = Path("/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json")
PROTO_NPZ = ROOT / "artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"
RUN_ID = "case_clean_sae_expand_attn_seed_p90__image_first_tile_selection"
CANONICAL_RUN = ROOT / (
    "artifacts/showcase_regions/"
    "TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_progressive_clean_sae_p90_to_hpvneg_border_relaxed/"
    f"{RUN_ID}"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    prototype_strength: float
    preserve_visited: float
    preserve_fresh: float
    mid_start: float
    alpha_start: float
    mid_end: float = 1.0
    alpha_end: float = 1.0
    preserve_edit: float = 0.0

    @property
    def out_dir(self) -> Path:
        return PACK / "candidate_runs" / f"original28_pe0_duration_{self.name}"

    @property
    def image_path(self) -> Path:
        return self.out_dir / RUN_ID / "generated.png"


CANDIDATES: list[Candidate] = [
    Candidate("p070_pv080_pf020_mid055_a040", 0.70, 0.80, 0.20, 0.55, 0.40),
    Candidate("p075_pv080_pf020_mid055_a040", 0.75, 0.80, 0.20, 0.55, 0.40),
    Candidate("p080_pv080_pf020_mid055_a040", 0.80, 0.80, 0.20, 0.55, 0.40),
    Candidate("p085_pv080_pf020_mid055_a040", 0.85, 0.80, 0.20, 0.55, 0.40),
    Candidate("p090_pv080_pf020_mid055_a040", 0.90, 0.80, 0.20, 0.55, 0.40),
    Candidate("p080_pv080_pf020_mid060_a035", 0.80, 0.80, 0.20, 0.60, 0.35),
    Candidate("p085_pv080_pf020_mid060_a035", 0.85, 0.80, 0.20, 0.60, 0.35),
    Candidate("p090_pv080_pf020_mid060_a035", 0.90, 0.80, 0.20, 0.60, 0.35),
    Candidate("p090_pv080_pf020_mid065_a030", 0.90, 0.80, 0.20, 0.65, 0.30),
    Candidate("p095_pv080_pf020_mid065_a030", 0.95, 0.80, 0.20, 0.65, 0.30),
    Candidate("p100_pv080_pf020_mid065_a030", 1.00, 0.80, 0.20, 0.65, 0.30),
    Candidate("p090_pv084_pf022_mid055_a040", 0.90, 0.84, 0.22, 0.55, 0.40),
    Candidate("p095_pv084_pf022_mid055_a040", 0.95, 0.84, 0.22, 0.55, 0.40),
    Candidate("p100_pv084_pf022_mid055_a040", 1.00, 0.84, 0.22, 0.55, 0.40),
    Candidate("p100_pv084_pf022_mid060_a035", 1.00, 0.84, 0.22, 0.60, 0.35),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_candidate(candidate: Candidate, args: argparse.Namespace) -> None:
    if candidate.image_path.exists() and not args.force:
        print(f"[skip] {candidate.name}")
        return
    log_dir = PACK / "logs/preserve_edit0_duration_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{candidate.name}.log"
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
        "0.0",
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
    print(f"[run] {candidate.name} -> {log_path}")
    with log_path.open("w") as log_f:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log_f, stderr=subprocess.STDOUT)


def minor_score(row: dict[str, Any]) -> float:
    return (
        1.0 * float(row["prototype_strength"])
        + 0.7 * (1.0 - float(row["mid_start"]))
        + 0.4 * float(row["alpha_start"])
        + 0.5 * (0.95 - float(row["preserve_visited"]))
        + 0.35 * (0.35 - float(row["preserve_fresh"]))
    )


def evaluate(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(SRC))
    import torch
    import matplotlib.pyplot as plt

    from wsi_cf.common.paths import DEFAULT_HNSCC_MIL_CKPT
    from wsi_cf.eval.hnsc_hpv import build_mil_from_checkpoint, run_mil_attention
    from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2

    out_data = PACK / "data"
    out_charts = PACK / "charts"
    out_cmp = PACK / "comparison/preserve_edit0_duration_sweep"
    out_grids = out_data / "encoded_uni2_grids/preserve_edit0_duration_sweep"
    for path in (out_data, out_charts, out_cmp, out_grids):
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    uni_model, uni_transform = load_uni2(device)
    mil_model = build_mil_from_checkpoint(DEFAULT_HNSCC_MIL_CKPT, device=device)
    source_path = CANONICAL_RUN / "source_region_actual.png"
    source_arr = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32)

    items: list[tuple[str, Path, dict[str, Any]]] = [
        (
            "canonical_0p8",
            CANONICAL_RUN / "generated.png",
            {
                "prototype_strength": 0.8,
                "preserve_edit": 0.05,
                "preserve_visited": 0.95,
                "preserve_fresh": 0.35,
                "mid_start": 0.50,
                "alpha_start": 0.50,
                "kind": "baseline",
            },
        )
    ]
    for candidate in CANDIDATES:
        if candidate.image_path.exists():
            meta = asdict(candidate)
            meta["kind"] = "sweep"
            items.append((candidate.name, candidate.image_path, meta))

    rows: list[dict[str, Any]] = []
    for sample, image_path, meta in items:
        image = Image.open(image_path).convert("RGB")
        arr = np.asarray(image, dtype=np.float32)
        diff = np.abs(arr - source_arr)
        pix = diff.mean(axis=2)
        z_t = build_uni_grid_from_image(
            image,
            uni_model=uni_model,
            uni_transform=uni_transform,
            grid_step_px=256,
            device=device,
            out_dtype=torch.float32,
        )
        z = z_t.detach().cpu().numpy().astype(np.float32)
        grid_path = out_grids / f"{sample}_whole_uni2_grid.npy"
        np.save(grid_path, z)
        _, pred, prob_pos = run_mil_attention(mil_model, z.reshape(-1, z.shape[-1]), device=device)
        cmp_dir = out_cmp / sample
        cmp_dir.mkdir(parents=True, exist_ok=True)
        image.save(cmp_dir / "generated.png")
        row = {
            "sample": sample,
            "pred_label": "HPV+" if int(pred) == 1 else "HPV-",
            "prob_hpv_pos": float(prob_pos),
            "prob_hpv_neg": float(1.0 - prob_pos),
            "mean_abs_rgb_diff_vs_source_0_255": float(diff.mean()),
            "median_abs_rgb_diff_vs_source_0_255": float(np.median(pix)),
            "pct_pixels_abs_rgb_diff_gt_25": float((pix > 25).mean()),
            "image_path": str(image_path),
            "uni2_grid_path": str(grid_path),
            **meta,
        }
        row["minor_score"] = minor_score(row)
        rows.append(row)

    csv_path = out_data / "hpv_original28_preserve_edit0_duration_sweep.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_data / "hpv_original28_preserve_edit0_duration_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")

    flips = sorted(
        [row for row in rows if row["pred_label"] == "HPV-"],
        key=lambda row: (float(row["mean_abs_rgb_diff_vs_source_0_255"]), float(row["minor_score"])),
    )
    rank_path = out_data / "hpv_original28_preserve_edit0_duration_sweep_flip_ranking.csv"
    with rank_path.open("w", newline="") as f:
        fieldnames = [
            "rank",
            "sample",
            "prob_hpv_neg",
            "mean_abs_rgb_diff_vs_source_0_255",
            "minor_score",
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
        for rank, row in enumerate(flips, 1):
            writer.writerow({"rank": rank, **{key: row.get(key, "") for key in fieldnames if key != "rank"}})

    plot_rows = sorted(rows, key=lambda row: float(row["minor_score"]))
    x = np.arange(len(plot_rows))
    fig, ax1 = plt.subplots(figsize=(max(8.0, 0.55 * len(plot_rows)), 3.6), dpi=220)
    ax1.bar(x, [float(row["prob_hpv_neg"]) for row in plot_rows], color=["#4c8f8a" if row["pred_label"] == "HPV-" else "#7aa6c2" for row in plot_rows])
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax1.set_ylim(0, 1.04)
    ax1.set_ylabel("HPV- probability")
    ax1.set_xticks(x)
    ax1.set_xticklabels([row["sample"].replace("_", "\n") for row in plot_rows], fontsize=6)
    ax2 = ax1.twinx()
    ax2.plot(x, [float(row["mean_abs_rgb_diff_vs_source_0_255"]) for row in plot_rows], color="#333333", marker="o", linewidth=1.2, markersize=3)
    ax2.set_ylabel("Mean abs RGB diff")
    fig.tight_layout()
    chart_path = out_charts / "hpv_original28_preserve_edit0_duration_sweep.png"
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    print(csv_path)
    print(rank_path)
    print(chart_path)
    if flips:
        best = flips[0]
        print("[lowest_diff_flip]", best["sample"], f"neg={float(best['prob_hpv_neg']):.6f}", f"diff={float(best['mean_abs_rgb_diff_vs_source_0_255']):.2f}", f"score={float(best['minor_score']):.4f}")
    else:
        print("[lowest_diff_flip] none")


def main() -> None:
    args = parse_args()
    if not args.run and not args.evaluate:
        args.run = True
        args.evaluate = True
    if args.run:
        for candidate in CANDIDATES:
            run_candidate(candidate, args)
    if args.evaluate:
        evaluate(args)


if __name__ == "__main__":
    main()
