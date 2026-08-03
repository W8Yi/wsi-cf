#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wsi_cf.paper.strength_sweep import json_safe, read_csv_rows, summarize_monotonicity, write_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paper-ready steering-strength image ladders and monotonic response plots."
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--prefix", default="steering_strength")
    parser.add_argument("--sweep-column", default="steering_strength")
    parser.add_argument("--sweep-label", default="Steering strength (%)")
    parser.add_argument("--region-id", default="", help="Region to show in the image ladder; defaults to the first region.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--show-edit-box",
        action="store_true",
        help="Prepend the untouched source image with a box around the requested edit cells.",
    )
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Prepend the untouched source image without an edit outline.",
    )
    parser.add_argument("--hide-title", action="store_true", help="Omit the overall figure title.")
    parser.add_argument("--hide-footer", action="store_true", help="Omit the matched-seed footer note.")
    return parser


def _numeric_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for key in ("steering_strength", "tile_fraction", "target_probability", "concept_activation", "concept_score", "seed"):
            if row.get(key, "") not in ("", None):
                item[key] = float(row[key])
        output.append(item)
    return output


def _aggregate(
    rows: list[dict[str, Any]],
    column: str,
    *,
    sweep_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strengths = sorted({float(row[sweep_column]) for row in rows})
    means: list[float] = []
    errors: list[float] = []
    for strength in strengths:
        values = np.asarray(
            [
                float(row[column])
                for row in rows
                if float(row[sweep_column]) == strength and row.get(column, "") not in ("", None)
            ],
            dtype=np.float64,
        )
        means.append(float(values.mean()) if values.size else float("nan"))
        errors.append(float(1.96 * values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0)
    return np.asarray(strengths), np.asarray(means), np.asarray(errors)


def _plot_metric(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    column: str,
    ylabel: str,
    color: str,
    sweep_column: str,
    sweep_label: str,
) -> None:
    by_region: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get(column, "") in ("", None):
            continue
        by_region.setdefault(str(row.get("region_id", "")), []).append(row)
    for region_rows in by_region.values():
        region_rows.sort(key=lambda row: float(row[sweep_column]))
        ax.plot(
            [100.0 * float(row[sweep_column]) for row in region_rows],
            [float(row[column]) for row in region_rows],
            color=color,
            alpha=0.18 if len(by_region) > 1 else 0.95,
            linewidth=0.9 if len(by_region) > 1 else 2.0,
            marker="o" if len(by_region) == 1 else None,
            markersize=4,
        )
    x, mean, error = _aggregate(rows, column, sweep_column=sweep_column)
    if len(by_region) > 1:
        ax.fill_between(100.0 * x, mean - error, mean + error, color=color, alpha=0.16, linewidth=0)
        ax.plot(100.0 * x, mean, color=color, linewidth=2.2, marker="o", markersize=4)
    elif len(by_region) == 1:
        for strength, value in zip(x, mean):
            if column == "target_probability":
                if value >= 0.9995:
                    label = f"{value:.5f}"
                elif value >= 0.995:
                    label = f"{value:.4f}"
                else:
                    label = f"{value:.3f}"
            else:
                label = f"{value:.2f}"
            ax.annotate(
                label,
                (100.0 * strength, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=color,
                bbox=(
                    {"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.92}
                    if column == "target_probability"
                    else None
                ),
                zorder=5,
            )
    ax.set_xlabel(sweep_label)
    ax.set_ylabel(ylabel)
    ax.set_xticks(100.0 * x)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _source_edit_box(ladder: list[dict[str, Any]]) -> tuple[Path, tuple[int, int, int, int]]:
    generated_path = Path(str(ladder[0].get("image_path", "")))
    manifest_path = generated_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot show edit box without run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    source_path = Path(str(manifest.get("source_image_path", "")))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image for edit-box panel does not exist: {source_path}")
    raw_cells = manifest.get("eligible_cells") or manifest.get("target_cells", [])
    cells = [(int(cell["gx"]), int(cell["gy"])) for cell in raw_cells]
    if not cells:
        raise ValueError(f"Run manifest has no target_cells: {manifest_path}")
    step = int(manifest.get("grid_step_px", 256))
    gx0 = min(gx for gx, _ in cells)
    gy0 = min(gy for _, gy in cells)
    gx1 = max(gx for gx, _ in cells) + 1
    gy1 = max(gy for _, gy in cells) + 1
    return source_path, (gx0 * step, gy0 * step, (gx1 - gx0) * step, (gy1 - gy0) * step)


def make_figure(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[Path]:
    sweep_column = str(getattr(args, "sweep_column", "steering_strength"))
    sweep_label = str(getattr(args, "sweep_label", "Steering strength (%)"))
    region_ids = sorted({str(row.get("region_id", "")) for row in rows})
    region_id = str(args.region_id) if args.region_id else region_ids[0]
    ladder = sorted(
        [row for row in rows if str(row.get("region_id", "")) == region_id],
        key=lambda row: float(row[sweep_column]),
    )
    if not ladder:
        raise ValueError(f"No rows found for region_id={region_id!r}")

    show_edit_box = bool(getattr(args, "show_edit_box", False))
    show_source = bool(getattr(args, "show_source", False))
    show_source_panel = bool(show_edit_box or show_source)
    n_images = len(ladder) + int(show_source_panel)
    n_columns = max(2, n_images)
    figure = plt.figure(figsize=(max(10.5, 1.9 * n_columns), 5.05), dpi=int(args.dpi))
    grid = figure.add_gridspec(2, n_columns, height_ratios=(1.0, 0.82), hspace=0.29, wspace=0.045)
    image_offset = 0
    if show_source_panel:
        source_path, (box_x, box_y, box_w, box_h) = _source_edit_box(ladder)
        ax = figure.add_subplot(grid[0, 0])
        ax.imshow(Image.open(source_path).convert("RGB"))
        if show_edit_box:
            # Inset the presentation outline slightly when the target touches
            # the image boundary. This keeps the complete stroke inside.
            box_inset = 24.0
            ax.add_patch(
                Rectangle(
                    (box_x + box_inset, box_y + box_inset),
                    max(1.0, box_w - 2.0 * box_inset),
                    max(1.0, box_h - 2.0 * box_inset),
                    fill=False,
                    edgecolor="#d62728",
                    linewidth=2.4,
                    zorder=10,
                )
            )
        ax.set_title("Source + edit box" if show_edit_box else "Source", fontsize=10, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_anchor("S")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#777777")
        image_offset = 1

    for index, row in enumerate(ladder, start=image_offset):
        ax = figure.add_subplot(grid[0, index])
        image_path = Path(str(row.get("image_path", "")))
        if image_path.is_file():
            ax.imshow(Image.open(image_path).convert("RGB"))
        else:
            ax.set_facecolor("#f2f2f2")
            ax.text(0.5, 0.5, "image\nmissing", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(f"{100.0 * float(row[sweep_column]):.0f}%", fontsize=11, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_anchor("S")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#777777")

    for index in range(n_images, n_columns):
        figure.add_subplot(grid[0, index]).axis("off")
    # Reserve room for the first plot's tick labels and vertical label within
    # the image-row footprint.  The final plot still ends at the last image.
    bottom_grid = grid[1, :].subgridspec(
        1,
        4,
        width_ratios=(0.16, 1.0, 0.20, 1.0),
        wspace=0.0,
    )
    ax_probability = figure.add_subplot(bottom_grid[0, 1])
    ax_concept = figure.add_subplot(bottom_grid[0, 3])
    _plot_metric(
        ax_probability,
        rows,
        column="target_probability",
        ylabel="Classifier\nP(target class)",
        color="#1f77b4",
        sweep_column=sweep_column,
        sweep_label=sweep_label,
    )
    score_name = str(rows[0].get("concept_score_name") or "Target concept score")
    if score_name.startswith("Target prototype cosine"):
        score_ylabel = "Target-prototype\ncosine alignment"
    else:
        score_ylabel = "Target SAE\nconcept activation"
    _plot_metric(
        ax_concept,
        rows,
        column="concept_score",
        ylabel=score_ylabel,
        color="#d55e00",
        sweep_column=sweep_column,
        sweep_label=sweep_label,
    )
    if not bool(getattr(args, "hide_title", False)):
        title = str(args.title).strip() or f"{ladder[0].get('task_name', '')}: {ladder[0].get('direction', '')}"
        figure.suptitle(title, fontsize=13, y=0.985)
    if not bool(getattr(args, "hide_footer", False)):
        raw_seed = ladder[0].get("seed", "")
        try:
            seed_text = str(int(float(raw_seed)))
        except (TypeError, ValueError):
            seed_text = str(raw_seed)
        figure.text(
            0.5,
            0.006,
            f"Fixed region and target cells; generated with matched seed {seed_text}.",
            ha="center",
            fontsize=8,
            color="#555555",
        )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.10 if bool(getattr(args, "hide_footer", False)) else 0.075,
        top=0.96 if bool(getattr(args, "hide_title", False)) else 0.91,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.out_dir / f"{args.prefix}.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=int(args.dpi))
    plt.close(figure)
    return paths


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = _numeric_rows(read_csv_rows(args.metrics_csv))
    required = {
        "task_name",
        "direction",
        "region_id",
        "seed",
        str(args.sweep_column),
        "target_probability",
        "concept_score",
        "image_path",
    }
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"{args.metrics_csv} is missing required columns: {sorted(missing)}")
    rows.sort(key=lambda row: (str(row.get("region_id", "")), float(row[args.sweep_column])))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}_plot_data.csv", rows)
    monotonicity = summarize_monotonicity(
        rows,
        value_columns=("target_probability", "concept_score"),
        sweep_column=str(args.sweep_column),
        tolerance=float(args.tolerance),
    )
    write_csv(args.out_dir / f"{args.prefix}_monotonicity.csv", monotonicity)
    summary = {
        "metrics_csv": str(args.metrics_csv),
        "n_rows": len(rows),
        "n_regions": len({str(row["region_id"]) for row in rows}),
        "figure_options": {
            "show_edit_box": bool(args.show_edit_box),
            "show_source": bool(args.show_source),
            "hide_title": bool(args.hide_title),
            "hide_footer": bool(args.hide_footer),
            "dpi": int(args.dpi),
            "sweep_column": str(args.sweep_column),
            "sweep_label": str(args.sweep_label),
        },
        "monotonicity": monotonicity,
    }
    (args.out_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    for path in make_figure(rows, args):
        print(path)


if __name__ == "__main__":
    main()
