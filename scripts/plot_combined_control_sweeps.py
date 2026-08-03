#!/usr/bin/env python3
"""Combine steering-intensity and valid-tile-fraction sweeps into one paper figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DISPLAY_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength-metrics", type=Path, required=True)
    parser.add_argument("--fraction-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--panels-dir",
        type=Path,
        default=None,
        help="Optional folder for exporting each image-ladder and response panel separately.",
    )
    parser.add_argument("--prefix", default="combined_control_sweeps")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _read_rows(path: Path, sweep_column: str) -> list[dict[str, Any]]:
    with path.open("r", newline="") as handle:
        rows: list[dict[str, Any]] = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        for key in (sweep_column, "target_probability", "concept_score"):
            row[key] = float(row[key])
    rows.sort(key=lambda row: (str(row.get("region_id", "")), float(row[sweep_column])))
    return rows


def _single_region(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    region_ids = sorted({str(row.get("region_id", "")) for row in rows})
    if not region_ids:
        raise ValueError("Metrics CSV has no region rows")
    return [row for row in rows if str(row.get("region_id", "")) == region_ids[0]]


def _display_rows(rows: list[dict[str, Any]], sweep_column: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for level in DISPLAY_LEVELS:
        matches = [row for row in rows if np.isclose(float(row[sweep_column]), level)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {sweep_column}={level} row, found {len(matches)}")
        selected.append(matches[0])
    return selected


def _source_context(rows: list[dict[str, Any]]) -> tuple[Path, tuple[int, int, int, int]]:
    generated_path = Path(str(rows[0]["image_path"]))
    manifest_path = generated_path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = Path(str(manifest["source_image_path"]))
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    raw_cells = manifest.get("eligible_cells") or manifest.get("target_cells", [])
    cells = [(int(cell["gx"]), int(cell["gy"])) for cell in raw_cells]
    if not cells:
        raise ValueError(f"No target or eligible cells in {manifest_path}")
    step = int(manifest.get("grid_step_px", 256))
    gx0 = min(gx for gx, _ in cells)
    gy0 = min(gy for _, gy in cells)
    gx1 = max(gx for gx, _ in cells) + 1
    gy1 = max(gy for _, gy in cells) + 1
    return source_path, (gx0 * step, gy0 * step, (gx1 - gx0) * step, (gy1 - gy0) * step)


def _style_image_axis(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_anchor("S")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#777777")


def _plot_image_row(
    figure: Any,
    slot: Any,
    rows: list[dict[str, Any]],
    *,
    sweep_column: str,
    source_title: str,
    show_box: bool = True,
) -> None:
    grid = slot.subgridspec(1, 1 + len(DISPLAY_LEVELS), wspace=0.055)
    source_path, (box_x, box_y, box_w, box_h) = _source_context(rows)
    source_ax = figure.add_subplot(grid[0, 0])
    source_ax.imshow(Image.open(source_path).convert("RGB"))
    if show_box:
        box_inset = 24.0
        source_ax.add_patch(
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
    source_ax.set_title(source_title, fontsize=10, pad=5, loc="left")
    _style_image_axis(source_ax)

    for column, row in enumerate(_display_rows(rows, sweep_column), start=1):
        ax = figure.add_subplot(grid[0, column])
        ax.imshow(Image.open(Path(str(row["image_path"]))).convert("RGB"))
        ax.set_title(f"{100.0 * float(row[sweep_column]):.0f}%", fontsize=10, pad=5)
        _style_image_axis(ax)


def _probability_label(value: float) -> str:
    if value >= 0.9995:
        return f"{value:.5f}"
    if value >= 0.995:
        return f"{value:.4f}"
    return f"{value:.3f}"


def _plot_response(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    sweep_column: str,
    value_column: str,
    xlabel: str,
    ylabel: str,
    color: str,
    title: str,
) -> None:
    x = np.asarray([100.0 * float(row[sweep_column]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[value_column]) for row in rows], dtype=np.float64)
    ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=4)
    for x_value, y_value in zip(x, y):
        label = _probability_label(float(y_value)) if value_column == "target_probability" else f"{y_value:.2f}"
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=color,
            bbox=(
                {"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.92}
                if value_column == "target_probability"
                else None
            ),
            zorder=5,
        )
    if title:
        ax.set_title(title, fontsize=10, loc="left", pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_response_row(
    figure: Any,
    slot: Any,
    rows: list[dict[str, Any]],
    *,
    sweep_column: str,
    xlabel: str,
    probability_title: str,
    concept_title: str,
    concept_ylabel: str,
) -> None:
    response_grid = slot.subgridspec(
        1,
        4,
        width_ratios=(0.13, 1.0, 0.20, 1.0),
        wspace=0.0,
    )
    probability_ax = figure.add_subplot(response_grid[0, 1])
    concept_ax = figure.add_subplot(response_grid[0, 3])
    _plot_response(
        probability_ax,
        rows,
        sweep_column=sweep_column,
        value_column="target_probability",
        xlabel=xlabel,
        ylabel="Classifier\nP(target class)",
        color="#1f77b4",
        title=probability_title,
    )
    _plot_response(
        concept_ax,
        rows,
        sweep_column=sweep_column,
        value_column="concept_score",
        xlabel=xlabel,
        ylabel=concept_ylabel,
        color="#d55e00",
        title=concept_title,
    )


def make_figure(
    strength_rows: list[dict[str, Any]],
    fraction_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[Path]:
    figure = plt.figure(figsize=(12.8, 10.0), dpi=int(args.dpi))
    outer = figure.add_gridspec(
        4,
        1,
        height_ratios=(0.90, 1.05, 0.90, 1.05),
        hspace=0.48,
    )
    _plot_image_row(
        figure,
        outer[0, 0],
        strength_rows,
        sweep_column="steering_strength",
        source_title="(a) Source + edit area",
    )
    _plot_response_row(
        figure,
        outer[1, 0],
        strength_rows,
        sweep_column="steering_strength",
        xlabel="Steering strength (%)",
        probability_title="(b)",
        concept_title="(c)",
        concept_ylabel="Mean target SAE\nactivation (16 tiles)",
    )
    _plot_image_row(
        figure,
        outer[2, 0],
        fraction_rows,
        sweep_column="tile_fraction",
        source_title="(d) Source",
        show_box=False,
    )
    _plot_response_row(
        figure,
        outer[3, 0],
        fraction_rows,
        sweep_column="tile_fraction",
        xlabel="Edited valid tissue tiles (%)",
        probability_title="(e)",
        concept_title="(f)",
        concept_ylabel="Mean target SAE\nactivation (55 valid tiles)",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.065)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.out_dir / f"{args.prefix}.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=int(args.dpi))
    plt.close(figure)
    return paths


def _save_panel_formats(figure: Any, out_dir: Path, title: str, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{title}.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=int(dpi))
    plt.close(figure)
    return paths


def _export_image_ladder(
    rows: list[dict[str, Any]],
    *,
    sweep_column: str,
    title: str,
    show_box: bool,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    figure = plt.figure(figsize=(12.8, 2.45), dpi=int(dpi))
    grid = figure.add_gridspec(1, 1)
    _plot_image_row(
        figure,
        grid[0, 0],
        rows,
        sweep_column=sweep_column,
        source_title=title,
        show_box=show_box,
    )
    figure.subplots_adjust(left=0.02, right=0.995, top=0.92, bottom=0.02)
    return _save_panel_formats(figure, out_dir, title, dpi)


def _export_response_panel(
    rows: list[dict[str, Any]],
    *,
    sweep_column: str,
    value_column: str,
    xlabel: str,
    ylabel: str,
    color: str,
    title: str,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    figure, ax = plt.subplots(figsize=(6.15, 4.0), dpi=int(dpi))
    _plot_response(
        ax,
        rows,
        sweep_column=sweep_column,
        value_column=value_column,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        title="",
    )
    figure.subplots_adjust(left=0.17, right=0.985, top=0.90, bottom=0.16)
    return _save_panel_formats(figure, out_dir, title, dpi)


def export_separate_panels(
    strength_rows: list[dict[str, Any]],
    fraction_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    outputs: list[Path] = []
    outputs.extend(
        _export_image_ladder(
            strength_rows,
            sweep_column="steering_strength",
            title="Source + edit area",
            show_box=True,
            out_dir=out_dir,
            dpi=dpi,
        )
    )
    outputs.extend(
        _export_response_panel(
            strength_rows,
            sweep_column="steering_strength",
            value_column="target_probability",
            xlabel="Steering strength (%)",
            ylabel="Classifier\nP(target class)",
            color="#1f77b4",
            title="Classifier response - fixed 16-tile area",
            out_dir=out_dir,
            dpi=dpi,
        )
    )
    outputs.extend(
        _export_response_panel(
            strength_rows,
            sweep_column="steering_strength",
            value_column="concept_score",
            xlabel="Steering strength (%)",
            ylabel="Mean target SAE\nactivation (16 tiles)",
            color="#d55e00",
            title="Concept response - fixed 16-tile area",
            out_dir=out_dir,
            dpi=dpi,
        )
    )
    outputs.extend(
        _export_image_ladder(
            fraction_rows,
            sweep_column="tile_fraction",
            title="Source",
            show_box=False,
            out_dir=out_dir,
            dpi=dpi,
        )
    )
    outputs.extend(
        _export_response_panel(
            fraction_rows,
            sweep_column="tile_fraction",
            value_column="target_probability",
            xlabel="Edited valid tissue tiles (%)",
            ylabel="Classifier\nP(target class)",
            color="#1f77b4",
            title="Classifier response - strength fixed at 90%",
            out_dir=out_dir,
            dpi=dpi,
        )
    )
    outputs.extend(
        _export_response_panel(
            fraction_rows,
            sweep_column="tile_fraction",
            value_column="concept_score",
            xlabel="Edited valid tissue tiles (%)",
            ylabel="Mean target SAE\nactivation (55 valid tiles)",
            color="#d55e00",
            title="Concept response - strength fixed at 90%",
            out_dir=out_dir,
            dpi=dpi,
        )
    )
    return outputs


def main() -> None:
    args = build_arg_parser().parse_args()
    strength_rows = _single_region(_read_rows(args.strength_metrics, "steering_strength"))
    fraction_rows = _single_region(_read_rows(args.fraction_metrics, "tile_fraction"))
    if str(strength_rows[0]["region_id"]) != str(fraction_rows[0]["region_id"]):
        raise ValueError("Combined figure requires both sweeps to use the same region")
    for path in make_figure(strength_rows, fraction_rows, args):
        print(path)
    if args.panels_dir is not None:
        for path in export_separate_panels(
            strength_rows,
            fraction_rows,
            out_dir=args.panels_dir,
            dpi=int(args.dpi),
        ):
            print(path)


if __name__ == "__main__":
    main()
