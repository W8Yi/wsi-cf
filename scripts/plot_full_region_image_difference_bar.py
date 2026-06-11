#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "artifacts/showcase_regions/figure4_element_pack/data/per_cell_image_difference_p090_pv084_pf022_mid055_a040.csv"
DEFAULT_OUT_STEM = ROOT / "artifacts/showcase_regions/figure4_element_pack/charts/image_difference_mean_abs_rgb_full_region_bar_p090_pv084_pf022_mid055_a040"
DEFAULT_STATS = ROOT / "artifacts/showcase_regions/figure4_element_pack/data/image_difference_mean_abs_rgb_full_region_bar_p090_pv084_pf022_mid055_a040_stats.csv"
SOURCE_METHODS = ("Naive", "Counterfactual")
DISPLAY_NAMES = ("Naive", "Progressive")
COLORS = ("#a7a7a7", "#0877ad")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot full-region mean absolute RGB difference with paired-cell statistics.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-stem", type=Path, default=DEFAULT_OUT_STEM)
    parser.add_argument("--stats-csv", type=Path, default=DEFAULT_STATS)
    return parser


def load_paired_cell_values(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, dict[tuple[int, int], float]] = {method: {} for method in SOURCE_METHODS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            method = str(row["method"])
            if method not in values:
                continue
            coord = (int(row["gx"]), int(row["gy"]))
            values[method][coord] = float(row["mean_abs_rgb"])
    coords = sorted(set(values[SOURCE_METHODS[0]]) & set(values[SOURCE_METHODS[1]]))
    if not coords or any(len(values[method]) != len(coords) for method in SOURCE_METHODS):
        raise ValueError(f"Expected matching full-region cells for {SOURCE_METHODS} in {path}")
    return {
        method: np.asarray([values[method][coord] for coord in coords], dtype=np.float64)
        for method in SOURCE_METHODS
    }


def save_stats(path: Path, values: dict[str, np.ndarray], p_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "n_paired_cells", "mean_abs_rgb", "sd", "sem", "paired_wilcoxon_p"])
        writer.writeheader()
        for display_name, method in zip(DISPLAY_NAMES, SOURCE_METHODS):
            arr = values[method]
            writer.writerow(
                {
                    "method": display_name,
                    "n_paired_cells": int(arr.size),
                    "mean_abs_rgb": float(np.mean(arr)),
                    "sd": float(np.std(arr, ddof=1)),
                    "sem": float(np.std(arr, ddof=1) / np.sqrt(arr.size)),
                    "paired_wilcoxon_p": float(p_value),
                }
            )


def draw_chart(out_stem: Path, values: dict[str, np.ndarray]) -> None:
    means = np.asarray([np.mean(values[method]) for method in SOURCE_METHODS], dtype=float)
    sems = np.asarray([np.std(values[method], ddof=1) / np.sqrt(values[method].size) for method in SOURCE_METHODS], dtype=float)
    x = np.arange(len(SOURCE_METHODS))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.7, 3.1))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    bars = ax.bar(
        x,
        means,
        yerr=sems,
        capsize=4,
        width=0.42,
        color=COLORS,
        edgecolor="#202020",
        linewidth=0.7,
        error_kw={"elinewidth": 1.1, "capthick": 1.1, "ecolor": "#202020"},
    )
    ax.set_xticks(x, DISPLAY_NAMES, fontsize=14)
    ax.set_ylabel(r"Mean $|\Delta \mathrm{RGB}|$", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, 54)
    ax.yaxis.grid(True, color="#d9d9d9", linewidth=0.7)
    ax.set_axisbelow(True)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 2.2, f"{mean:.2f}", ha="center", va="bottom", fontsize=13)

    fig.tight_layout(pad=0.65)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=400, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    values = load_paired_cell_values(args.input_csv)
    _, p_value = wilcoxon(values[SOURCE_METHODS[0]], values[SOURCE_METHODS[1]], alternative="two-sided")
    save_stats(args.stats_csv, values, float(p_value))
    draw_chart(args.out_stem, values)
    print(f"Wrote {args.out_stem.with_suffix('.pdf')}")
    print(f"Wrote {args.stats_csv}")


if __name__ == "__main__":
    main()
