#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsi_cf_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS_DIR = Path(
    "paper_outputs/prediction_transition_benchmark_test_only_unbalanced/metrics/hnscc_hpv/hpv_pos_to_hpv_neg"
)
DEFAULT_OUT_DIR = Path("paper_outputs/prediction_transition_benchmark_test_only_unbalanced/plots/hnscc_hpv/hpv_pos_to_hpv_neg")

TASK_LABELS = {
    "hnscc_hpv": "HNSCC HPV",
    "luad_normal_tumor": "LUAD normal->tumor",
    "coad_normal_tumor": "COAD normal->tumor",
    "kirc_normal_tumor": "KIRC normal->tumor",
    "brca_normal_tumor": "BRCA normal->tumor",
    "prad_morphology_group": "PRAD morphology",
}

DIRECTION_LABELS = {
    "hpv_pos_to_hpv_neg": "HPV+ -> HPV-",
    "hpv_neg_to_hpv_pos": "HPV- -> HPV+",
    "normal_to_tumor": "normal -> tumor",
    "tumor_to_normal": "tumor -> normal",
    "well_to_p4": "well-formed -> pattern 4",
    "p4_to_p5": "pattern 4 -> pattern 5",
    "p4_to_well": "pattern 4 -> well-formed",
}

SELECTOR_COLORS = {
    "attention": "#0877ad",
    "random": "#b7b7b7",
}
SELECTOR_LABELS = {
    "attention": "Attention-ranked",
    "random": "Random",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reusable paper plots from prediction-transition benchmark metrics. "
            "Each metrics directory should contain prediction_transition_by_run.csv."
        )
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        action="append",
        default=None,
        help="Metric directory containing prediction_transition_by_run.csv. Can be repeated.",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=None,
        help=(
            "Discover all task/direction metrics under this root. Expected layout is "
            "<metrics-root>/<task>/<direction>/prediction_transition_by_run.csv."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", type=str, default="prediction_transition")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Only include metric directories whose benchmark_summary has n_scored == n_requests.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude task/direction keys matching this string, e.g. prad_morphology_group/p4_to_p5.",
    )
    parser.add_argument(
        "--max-shared-budget",
        type=int,
        default=48,
        help="Largest common budget to use for attention-vs-random comparison plots.",
    )
    parser.add_argument(
        "--report-budget",
        type=int,
        default=48,
        help="Budget used for the compact report bar summary.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf,svg",
        help="Comma-separated output formats.",
    )
    return parser


def display_task_direction(task_name: str, direction: str) -> str:
    task_label = TASK_LABELS.get(str(task_name), str(task_name))
    direction_label = DIRECTION_LABELS.get(str(direction), str(direction))
    if str(direction) == "normal_to_tumor" and str(task_name).endswith("_normal_tumor"):
        return task_label
    return f"{task_label}: {direction_label}"


def discover_metrics_dirs(metrics_root: Path) -> list[Path]:
    candidates = sorted(metrics_root.glob("*/*/prediction_transition_by_run.csv"))
    return [path.parent for path in candidates if "eval_shards" not in path.parts and "stream_" not in path.parts]


def read_status(metrics_dir: Path) -> dict[str, object]:
    summary_path = metrics_dir / "benchmark_summary.json"
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return {"n_requests": None, "n_scored": None, "complete": None}
    try:
        payload = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {"n_requests": None, "n_scored": None, "complete": None}
    n_requests = payload.get("n_requests")
    n_scored = payload.get("n_scored")
    complete = bool(n_requests == n_scored) if n_requests is not None and n_scored is not None else None
    return {"n_requests": n_requests, "n_scored": n_scored, "complete": complete}


def filter_metrics_dirs(metrics_dirs: list[Path], *, complete_only: bool, exclude: list[str]) -> list[Path]:
    out: list[Path] = []
    for metrics_dir in metrics_dirs:
        key = "/".join(metrics_dir.parts[-2:])
        if any(pattern in key or pattern in str(metrics_dir) for pattern in exclude):
            continue
        status = read_status(metrics_dir)
        if complete_only and status.get("complete") is not True:
            continue
        out.append(metrics_dir)
    return out


def read_by_run(metrics_dir: Path) -> pd.DataFrame:
    path = metrics_dir / "prediction_transition_by_run.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty by-run metric CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows in {path}")
    df["metrics_dir"] = str(metrics_dir)
    status = read_status(metrics_dir)
    df["benchmark_n_requests"] = status.get("n_requests")
    df["benchmark_n_scored"] = status.get("n_scored")
    df["benchmark_complete"] = status.get("complete")
    return df


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "task_name",
        "direction",
        "selector",
        "budget",
        "source_target_probability",
        "edited_target_probability",
        "target_probability_delta",
        "flip_to_target",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required metric column(s): {missing}")
    out = df.copy()
    out["budget"] = pd.to_numeric(out["budget"], errors="raise").astype(int)
    for col in [
        "source_target_probability",
        "edited_target_probability",
        "target_probability_delta",
        "flip_to_target",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "edited_pred_is_target" in out.columns:
        out["edited_pred_is_target"] = pd.to_numeric(out["edited_pred_is_target"], errors="coerce")
    else:
        out["edited_pred_is_target"] = out["flip_to_target"]
    if "expected_grade_delta" in out.columns:
        out["expected_grade_delta"] = pd.to_numeric(out["expected_grade_delta"], errors="coerce")
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["task_name", "direction", "selector", "budget"]
    agg = {
        "run_id": "count",
        "source_target_probability": ["mean", "median"],
        "edited_target_probability": ["mean", "median"],
        "target_probability_delta": ["mean", "median"],
        "flip_to_target": "mean",
        "edited_pred_is_target": "mean",
    }
    if "expected_grade_delta" in df.columns and df["expected_grade_delta"].notna().any():
        agg["expected_grade_delta"] = ["mean", "median"]
    summary = df.groupby(group_cols, dropna=False).agg(agg)
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    summary = summary.rename(
        columns={
            "run_id_count": "n",
            "source_target_probability_mean": "mean_source_target_probability",
            "source_target_probability_median": "median_source_target_probability",
            "edited_target_probability_mean": "mean_edited_target_probability",
            "edited_target_probability_median": "median_edited_target_probability",
            "target_probability_delta_mean": "mean_target_probability_delta",
            "target_probability_delta_median": "median_target_probability_delta",
            "flip_to_target_mean": "flip_to_target_rate",
            "edited_pred_is_target_mean": "target_prediction_rate_after_edit",
        }
    )
    return summary.sort_values(["task_name", "direction", "selector", "budget"]).reset_index(drop=True)


def attention_random_advantage(summary: pd.DataFrame, max_budget: int) -> pd.DataFrame:
    needed = summary[summary["selector"].isin(["attention", "random"])].copy()
    needed = needed[needed["budget"] <= int(max_budget)]
    wide = needed.pivot_table(
        index=["task_name", "direction", "budget"],
        columns="selector",
        values="mean_target_probability_delta",
        aggfunc="first",
    ).reset_index()
    if "attention" not in wide.columns or "random" not in wide.columns:
        return pd.DataFrame()
    wide = wide.dropna(subset=["attention", "random"]).copy()
    wide["attention_minus_random_delta"] = wide["attention"] - wide["random"]
    return wide.rename(
        columns={
            "attention": "mean_attention_target_probability_delta",
            "random": "mean_random_target_probability_delta",
        }
    )


def task_direction_summary(by_run: pd.DataFrame, summary: pd.DataFrame, metrics_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (task_name, direction), df in by_run.groupby(["task_name", "direction"], dropna=False):
        status_rows = df[["benchmark_n_requests", "benchmark_n_scored", "benchmark_complete"]].drop_duplicates()
        status = status_rows.iloc[0].to_dict() if not status_rows.empty else {}
        row: dict[str, object] = {
            "task_name": task_name,
            "direction": direction,
            "display_name": display_task_direction(str(task_name), str(direction)),
            "n_rows": int(len(df)),
            "n_requests": status.get("benchmark_n_requests", ""),
            "complete": status.get("benchmark_complete", ""),
            "n_slides": int(df["slide_key"].nunique()) if "slide_key" in df.columns else "",
            "n_regions": int(df["region_id"].nunique()) if "region_id" in df.columns else "",
            "mean_source_target_probability": float(df["source_target_probability"].mean()),
        }
        sub_summary = summary[(summary["task_name"] == task_name) & (summary["direction"] == direction)]
        for budget in [1, 8, 16, 32, 48]:
            for selector in ["attention", "random"]:
                sub = sub_summary[(sub_summary["selector"] == selector) & (sub_summary["budget"] == budget)]
                if not sub.empty:
                    row[f"{selector}_delta_b{budget}"] = float(sub["mean_target_probability_delta"].iloc[0])
                    row[f"{selector}_flip_b{budget}"] = float(sub["flip_to_target_rate"].iloc[0])
            if f"attention_delta_b{budget}" in row and f"random_delta_b{budget}" in row:
                row[f"attention_minus_random_delta_b{budget}"] = (
                    float(row[f"attention_delta_b{budget}"]) - float(row[f"random_delta_b{budget}"])
                )
        for selector in ["attention", "random"]:
            full = df[(df["selector"] == selector) & (df["budget_is_full"].astype(int) == 1)]
            if not full.empty:
                row[f"{selector}_full_n"] = int(len(full))
                row[f"{selector}_full_budget_mean"] = float(full["budget"].mean())
                row[f"{selector}_full_delta"] = float(full["target_probability_delta"].mean())
                row[f"{selector}_full_flip_rate"] = float(full["flip_to_target"].mean())
                row[f"{selector}_full_target_pred_rate"] = float(full["edited_pred_is_target"].mean())
                if "expected_grade_delta" in full.columns and full["expected_grade_delta"].notna().any():
                    row[f"{selector}_full_expected_grade_delta"] = float(full["expected_grade_delta"].mean())
        if "attention_full_delta" in row and "random_full_delta" in row:
            row["attention_minus_random_full_delta"] = float(row["attention_full_delta"]) - float(row["random_full_delta"])
        auc_path = None
        for metrics_dir in metrics_dirs:
            if not (metrics_dir / "prediction_transition_by_run.csv").exists():
                continue
            try:
                first = pd.read_csv(metrics_dir / "prediction_transition_by_run.csv", nrows=1)
            except Exception:
                continue
            if not first.empty and str(first["task_name"].iloc[0]) == str(task_name) and str(first["direction"].iloc[0]) == str(direction):
                auc_path = metrics_dir / "prediction_transition_auc_by_region.csv"
                break
        if auc_path is not None and auc_path.exists() and auc_path.stat().st_size > 0:
            auc = pd.read_csv(auc_path)
            for selector in ["attention", "random"]:
                sub = auc[auc["selector"] == selector]
                if not sub.empty:
                    row[f"{selector}_transition_auc"] = float(sub["transition_auc"].mean())
            if "attention_transition_auc" in row and "random_transition_auc" in row:
                row["attention_minus_random_transition_auc"] = (
                    float(row["attention_transition_auc"]) - float(row["random_transition_auc"])
                )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["task_name", "direction"]).reset_index(drop=True)


def save_figure(fig: plt.Figure, out_stem: Path, formats: Iterable[str]) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        clean = fmt.strip().lstrip(".")
        if not clean:
            continue
        fig.savefig(out_stem.with_suffix(f".{clean}"), dpi=400, bbox_inches="tight", transparent=True)
    plt.close(fig)


def write_markdown_table(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    use = df[columns].copy()
    for col in use.columns:
        if pd.api.types.is_float_dtype(use[col]):
            if "rate" in col or "complete" in col:
                use[col] = use[col].map(lambda value: "" if pd.isna(value) else f"{100 * float(value):.1f}%")
            else:
                use[col] = use[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    use = use.fillna("").astype(str)
    widths = {
        col: max(len(str(col)), *(len(value) for value in use[col].tolist()))
        for col in use.columns
    }

    def fmt_row(values: Iterable[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[col]) for col, value in zip(use.columns, values)) + " |"

    lines = [
        fmt_row(use.columns),
        "| " + " | ".join("-" * widths[col] for col in use.columns) + " |",
    ]
    lines.extend(fmt_row(row) for row in use.itertuples(index=False, name=None))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_line_metric(
    summary: pd.DataFrame,
    *,
    y_col: str,
    ylabel: str,
    title: str,
    out_stem: Path,
    formats: Iterable[str],
    max_budget: int | None = None,
) -> None:
    data = summary.copy()
    if max_budget is not None:
        data = data[data["budget"] <= int(max_budget)]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    for selector in ["attention", "random"]:
        sub = data[data["selector"] == selector].sort_values("budget")
        if sub.empty:
            continue
        ax.plot(
            sub["budget"],
            sub[y_col],
            marker="o",
            linewidth=2.0,
            markersize=4.8,
            color=SELECTOR_COLORS.get(selector, None),
            label=SELECTOR_LABELS.get(selector, selector),
        )
    ax.set_xlabel("Steered cells")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=12.5)
    ax.set_ylim(-0.04, 1.04) if y_col != "attention_minus_random_delta" else None
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.8)
    save_figure(fig, out_stem, formats)


def plot_advantage(adv: pd.DataFrame, *, title: str, out_stem: Path, formats: Iterable[str]) -> None:
    if adv.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    data = adv.sort_values("budget")
    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.plot(
        data["budget"],
        data["attention_minus_random_delta"],
        marker="o",
        linewidth=2.0,
        markersize=4.8,
        color="#0877ad",
    )
    ax.set_xlabel("Steered cells")
    ax.set_ylabel("Attention - random delta")
    if title:
        ax.set_title(title, fontsize=12.5)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    fig.tight_layout(pad=0.8)
    save_figure(fig, out_stem, formats)


def plot_report_budget_bar(
    summary: pd.DataFrame,
    *,
    report_budget: int,
    title: str,
    out_stem: Path,
    formats: Iterable[str],
) -> pd.DataFrame:
    sub = summary[(summary["budget"] == int(report_budget)) & summary["selector"].isin(["attention", "random"])].copy()
    if sub.empty:
        budgets = sorted(int(v) for v in summary["budget"].unique())
        if not budgets:
            return sub
        nearest = min(budgets, key=lambda value: abs(value - int(report_budget)))
        sub = summary[(summary["budget"] == nearest) & summary["selector"].isin(["attention", "random"])].copy()
    if sub.empty:
        return sub
    sub["selector_label"] = sub["selector"].map(lambda value: SELECTOR_LABELS.get(str(value), str(value)))
    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    colors = [SELECTOR_COLORS.get(str(value), "#777777") for value in sub["selector"]]
    bars = ax.bar(
        np.arange(len(sub)),
        sub["mean_target_probability_delta"],
        color=colors,
        edgecolor="#222222",
        linewidth=0.7,
        width=0.52,
    )
    ax.set_xticks(np.arange(len(sub)), sub["selector_label"], rotation=0)
    budget_label = int(sub["budget"].iloc[0])
    ax.set_ylabel("Mean target-probability delta")
    ax.set_title(title or f"{budget_label} steered cells", fontsize=12.5)
    ax.set_ylim(0, 1.04)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, sub["mean_target_probability_delta"]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.025, f"{float(value):.2f}", ha="center", va="bottom")
    fig.tight_layout(pad=0.8)
    save_figure(fig, out_stem, formats)
    return sub


def plot_small_multiples(summary: pd.DataFrame, *, title: str, out_stem: Path, formats: Iterable[str], max_budget: int) -> None:
    keys = list(summary[["task_name", "direction"]].drop_duplicates().itertuples(index=False, name=None))
    if len(keys) <= 1:
        return
    ncols = min(3, len(keys))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)
    fig.patch.set_alpha(0.0)
    for ax in axes.flat:
        ax.patch.set_alpha(0.0)
        ax.set_visible(False)
    for ax, (task, direction) in zip(axes.flat, keys):
        ax.set_visible(True)
        sub_all = summary[(summary["task_name"] == task) & (summary["direction"] == direction) & (summary["budget"] <= int(max_budget))]
        for selector in ["attention", "random"]:
            sub = sub_all[sub_all["selector"] == selector].sort_values("budget")
            if sub.empty:
                continue
            ax.plot(
                sub["budget"],
                sub["mean_target_probability_delta"],
                marker="o",
                linewidth=1.8,
                markersize=4.2,
                color=SELECTOR_COLORS.get(selector, None),
                label=SELECTOR_LABELS.get(selector, selector),
            )
        ax.set_title(f"{task}\\n{direction}", fontsize=10.5)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, color="#dddddd", linewidth=0.6)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Steered cells")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Mean target-probability delta")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
        fig.subplots_adjust(top=0.86)
    if title:
        fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(pad=0.9)
    save_figure(fig, out_stem, formats)


def plot_cross_task_bar(
    table: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    out_stem: Path,
    formats: Iterable[str],
    color: str = "#0877ad",
    xzero: bool = False,
) -> None:
    if value_col not in table.columns:
        return
    data = table.dropna(subset=[value_col]).sort_values(value_col, ascending=True)
    if data.empty:
        return
    fig_height = max(3.0, 0.42 * len(data) + 1.0)
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    y = np.arange(len(data))
    ax.barh(y, data[value_col], color=color, edgecolor="#222222", linewidth=0.5)
    ax.set_yticks(y, data["display_name"])
    ax.set_xlabel(ylabel)
    if title:
        ax.set_title(title, fontsize=12.5)
    if xzero:
        ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    save_figure(fig, out_stem, formats)


def plot_budget_heatmap(
    adv: pd.DataFrame,
    *,
    title: str,
    out_stem: Path,
    formats: Iterable[str],
) -> None:
    if adv.empty:
        return
    data = adv.copy()
    data["display_name"] = [
        display_task_direction(task, direction) for task, direction in zip(data["task_name"], data["direction"])
    ]
    pivot = data.pivot_table(
        index="display_name",
        columns="budget",
        values="attention_minus_random_delta",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(8.2, max(3.0, 0.38 * len(pivot) + 1.0)))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    vmax = max(float(vmax), 1e-6)
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)), [str(int(col)) for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Steered cells")
    ax.set_title(title or "Attention advantage over random", fontsize=12.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Attention - random target-probability delta")
    for yi in range(values.shape[0]):
        for xi in range(values.shape[1]):
            value = values[yi, xi]
            if np.isfinite(value):
                ax.text(xi, yi, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout(pad=0.8)
    save_figure(fig, out_stem, formats)


def write_readme(
    out_dir: Path,
    *,
    metrics_dirs: list[Path],
    prefix: str,
    n_rows: int,
    n_missing_note: str,
) -> None:
    lines = [
        "# Prediction-Transition Plot Pack",
        "",
        f"Rows plotted: `{n_rows}`",
        "",
        "Metric sources:",
    ]
    lines.extend(f"- `{path}`" for path in metrics_dirs)
    if n_missing_note:
        lines.extend(["", n_missing_note])
    lines.extend(
        [
            "",
            "Main files:",
            f"- `{prefix}_paper_task_summary.csv` and `.md`",
            f"- `{prefix}_paper_auc_advantage_bar.*`",
            f"- `{prefix}_paper_budget32_attention_delta_bar.*`",
            f"- `{prefix}_paper_full_flip_rate_bar.*`",
            f"- `{prefix}_paper_attention_advantage_heatmap.*`",
            f"- `{prefix}_target_probability_delta_by_tiles.*`",
            f"- `{prefix}_flip_rate_by_tiles.*`",
            f"- `{prefix}_target_probability_by_tiles.*`",
            f"- `{prefix}_attention_vs_random_advantage.*`",
            f"- `{prefix}_summary_budget_bar.*`",
            f"- `{prefix}_prediction_transition_plot_data.csv`",
            f"- `{prefix}_attention_minus_random_advantage.csv`",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.metrics_root is not None:
        metrics_dirs = discover_metrics_dirs(Path(args.metrics_root))
        if args.metrics_dir:
            metrics_dirs.extend(args.metrics_dir)
    else:
        metrics_dirs = args.metrics_dir or [DEFAULT_METRICS_DIR]
    metrics_dirs = [Path(path) for path in metrics_dirs]
    metrics_dirs = filter_metrics_dirs(metrics_dirs, complete_only=bool(args.complete_only), exclude=list(args.exclude or []))
    if not metrics_dirs:
        raise ValueError("No metric directories selected.")
    formats = [item.strip() for item in str(args.formats).split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = [read_by_run(path) for path in metrics_dirs]
    by_run = normalize_frame(pd.concat(frames, ignore_index=True))
    summary = summarize(by_run)
    adv = attention_random_advantage(summary, max_budget=int(args.max_shared_budget))
    paper_table = task_direction_summary(by_run, summary, metrics_dirs)

    by_run.to_csv(args.out_dir / f"{args.prefix}_prediction_transition_by_run.csv", index=False)
    summary.to_csv(args.out_dir / f"{args.prefix}_prediction_transition_plot_data.csv", index=False)
    paper_table.to_csv(args.out_dir / f"{args.prefix}_paper_task_summary.csv", index=False)
    markdown_cols = [
        col
        for col in [
            "display_name",
            "n_requests",
            "n_rows",
            "complete",
            "n_slides",
            "n_regions",
            "attention_delta_b32",
            "random_delta_b32",
            "attention_minus_random_delta_b32",
            "attention_full_delta",
            "attention_full_flip_rate",
            "attention_minus_random_transition_auc",
        ]
        if col in paper_table.columns
    ]
    write_markdown_table(args.out_dir / f"{args.prefix}_paper_task_summary.md", paper_table, markdown_cols)
    if not adv.empty:
        adv.to_csv(args.out_dir / f"{args.prefix}_attention_minus_random_advantage.csv", index=False)

    setup_matplotlib()
    title = args.title
    plot_line_metric(
        summary,
        y_col="mean_target_probability_delta",
        ylabel="Mean target-probability delta",
        title=title,
        out_stem=args.out_dir / f"{args.prefix}_target_probability_delta_by_tiles",
        formats=formats,
        max_budget=int(args.max_shared_budget),
    )
    plot_line_metric(
        summary,
        y_col="flip_to_target_rate",
        ylabel="Flip-to-target rate",
        title=title,
        out_stem=args.out_dir / f"{args.prefix}_flip_rate_by_tiles",
        formats=formats,
        max_budget=int(args.max_shared_budget),
    )
    plot_line_metric(
        summary,
        y_col="mean_edited_target_probability",
        ylabel="Mean target probability after edit",
        title=title,
        out_stem=args.out_dir / f"{args.prefix}_target_probability_by_tiles",
        formats=formats,
        max_budget=int(args.max_shared_budget),
    )
    plot_advantage(
        adv,
        title=title,
        out_stem=args.out_dir / f"{args.prefix}_attention_vs_random_advantage",
        formats=formats,
    )
    report = plot_report_budget_bar(
        summary,
        report_budget=int(args.report_budget),
        title=title or f"Budget {args.report_budget}",
        out_stem=args.out_dir / f"{args.prefix}_summary_budget_bar",
        formats=formats,
    )
    if not report.empty:
        report.to_csv(args.out_dir / f"{args.prefix}_report_bar_summary_budget.csv", index=False)
    plot_small_multiples(
        summary,
        title=title,
        out_stem=args.out_dir / f"{args.prefix}_delta_small_multiples",
        formats=formats,
        max_budget=int(args.max_shared_budget),
    )
    plot_cross_task_bar(
        paper_table,
        value_col="attention_minus_random_transition_auc",
        ylabel="Attention - random transition AUC",
        title=title or "Attention-ranked steering beats random selection",
        out_stem=args.out_dir / f"{args.prefix}_paper_auc_advantage_bar",
        formats=formats,
        color="#0877ad",
        xzero=True,
    )
    plot_cross_task_bar(
        paper_table,
        value_col="attention_delta_b32",
        ylabel="Mean target-probability delta at 32 cells",
        title=title or "Prediction shift at 32 steered cells",
        out_stem=args.out_dir / f"{args.prefix}_paper_budget32_attention_delta_bar",
        formats=formats,
        color="#3b8f4f",
    )
    plot_cross_task_bar(
        paper_table,
        value_col="attention_full_flip_rate",
        ylabel="Flip-to-target rate at full local budget",
        title=title or "Full-budget flip-to-target rate",
        out_stem=args.out_dir / f"{args.prefix}_paper_full_flip_rate_bar",
        formats=formats,
        color="#7a4cb0",
    )
    plot_budget_heatmap(
        adv,
        title=title or "Attention advantage over random by cell budget",
        out_stem=args.out_dir / f"{args.prefix}_paper_attention_advantage_heatmap",
        formats=formats,
    )

    missing_note = ""
    summaries = []
    for metrics_dir in metrics_dirs:
        summary_path = metrics_dir / "benchmark_summary.json"
        if summary_path.exists() and summary_path.stat().st_size > 0:
            try:
                payload = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                continue
            summaries.append(payload)
    if summaries:
        missing = sum(int(item.get("n_missing_generated", 0)) for item in summaries)
        requested = sum(int(item.get("n_requests", 0)) for item in summaries)
        scored = sum(int(item.get("n_scored", 0)) for item in summaries)
        missing_note = f"Scored `{scored}` of `{requested}` requested edits; missing generated outputs: `{missing}`."
        (args.out_dir / f"{args.prefix}_plot_status.json").write_text(
            json.dumps({"n_requested": requested, "n_scored": scored, "n_missing_generated": missing, "sources": summaries}, indent=2)
            + "\n"
        )
    write_readme(
        args.out_dir,
        metrics_dirs=metrics_dirs,
        prefix=args.prefix,
        n_rows=int(len(by_run)),
        n_missing_note=missing_note,
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": int(len(by_run)), "summary_rows": int(len(summary))}, indent=2))


if __name__ == "__main__":
    main()
