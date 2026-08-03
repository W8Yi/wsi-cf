#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("paper_outputs/full_test_streaming_benchmark_v1")

TASK_LABELS = {
    "hnscc_hpv": "HNSCC HPV",
    "luad_normal_tumor": "LUAD",
    "coad_normal_tumor": "COAD",
    "kirc_normal_tumor": "KIRC",
    "brca_normal_tumor": "BRCA",
    "prad_morphology_group": "PRAD",
}

DIRECTION_LABELS = {
    "hpv_pos_to_hpv_neg": "HPV+ -> HPV-",
    "hpv_neg_to_hpv_pos": "HPV- -> HPV+",
    "normal_to_tumor": "normal -> tumor",
    "well_to_p4": "well -> P4",
    "p4_to_p5": "P4 -> P5",
}

SELECTOR_COLORS = {
    "attention": "#1769aa",
    "random": "#9e9e9e",
}

METHOD_COLORS = {
    "ours": "#1769aa",
    "bad_naive": "#c7533a",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready plots from the unified full-test streaming benchmark. "
            "Inputs are the saved CSV metrics only; generated images are not required."
        )
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--formats", type=str, default="png,pdf,svg")
    parser.add_argument("--report-budget", type=int, default=64)
    parser.add_argument(
        "--visual-area",
        type=str,
        default="full_region",
        help="Area from visual_perturbation_by_checkpoint.csv to use for perturbation plots.",
    )
    parser.add_argument(
        "--border-area",
        type=str,
        default="selected_vs_unselected_boundary",
        help="Preferred seam_area for border plots. Falls back to the first available seam area.",
    )
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(sum(clean) / len(clean)) if clean else math.nan


def sem(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) <= 1:
        return 0.0 if clean else math.nan
    mu = mean(clean)
    var = sum((value - mu) ** 2 for value in clean) / (len(clean) - 1)
    return float(math.sqrt(var) / math.sqrt(len(clean)))


def task_label(task_name: str, direction: str) -> str:
    task = TASK_LABELS.get(task_name, task_name)
    direction_label = DIRECTION_LABELS.get(direction, direction)
    if direction == "normal_to_tumor":
        return f"{task} {direction_label}"
    return f"{task}: {direction_label}"


def discover(root: Path, relative: str) -> list[Path]:
    return sorted(path for path in root.glob(relative) if path.exists() and path.stat().st_size > 0)


def read_prediction_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover(root, "metrics/*/*/prediction_transition_by_checkpoint.csv"):
        for row in read_csv(path):
            row["metrics_path"] = str(path)
            row["task_name"] = row.get("task_name") or path.parts[-3]
            row["direction"] = row.get("direction") or path.parts[-2]
            row["method"] = row.get("method") or "ours"
            row["budget"] = to_int(row.get("budget"))
            for key in [
                "target_probability_delta",
                "flip_to_target",
                "edited_pred_is_target",
                "expected_grade_delta",
                "source_target_probability",
                "edited_target_probability",
            ]:
                row[key] = to_float(row.get(key))
            rows.append(row)
    return rows


def read_visual_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover(root, "metrics_visual/*/*/visual_perturbation_by_checkpoint.csv"):
        for row in read_csv(path):
            row["metrics_path"] = str(path)
            row["task_name"] = row.get("task_name") or path.parts[-3]
            row["direction"] = row.get("direction") or path.parts[-2]
            row["budget"] = to_int(row.get("budget"))
            row["mean_abs_rgb"] = to_float(row.get("mean_abs_rgb"))
            rows.append(row)
    return rows


def read_border_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover(root, "metrics_visual/*/*/border_discontinuity_by_checkpoint.csv"):
        for row in read_csv(path):
            row["metrics_path"] = str(path)
            row["task_name"] = row.get("task_name") or path.parts[-3]
            row["direction"] = row.get("direction") or path.parts[-2]
            row["budget"] = to_int(row.get("budget"))
            row["seam_excess_mean_abs_rgb"] = to_float(row.get("seam_excess_mean_abs_rgb"))
            rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    return grouped


def prediction_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = ["task_name", "direction", "method", "selector", "budget"]
    for key, group in sorted(group_rows(rows, keys).items()):
        deltas = [to_float(row.get("target_probability_delta")) for row in group]
        flips = [to_float(row.get("flip_to_target")) for row in group]
        pred_target = [to_float(row.get("edited_pred_is_target")) for row in group]
        expected_grade_delta = [to_float(row.get("expected_grade_delta")) for row in group]
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "n": len(group),
                "mean_target_probability_delta": mean(deltas),
                "sem_target_probability_delta": sem(deltas),
                "flip_to_target_rate": mean(flips),
                "target_prediction_rate_after_edit": mean(pred_target),
            }
        )
        if any(math.isfinite(value) for value in expected_grade_delta):
            row["mean_expected_grade_delta"] = mean(expected_grade_delta)
        out.append(row)
    return out


def visual_summary(rows: list[dict[str, Any]], *, area: str) -> list[dict[str, Any]]:
    wanted = [row for row in rows if row.get("area") == area]
    out: list[dict[str, Any]] = []
    for key, group in sorted(group_rows(wanted, ["task_name", "direction", "method", "budget"]).items()):
        values = [to_float(row.get("mean_abs_rgb")) for row in group]
        row = {name: value for name, value in zip(["task_name", "direction", "method", "budget"], key)}
        row.update({"area": area, "n": len(group), "mean_abs_rgb": mean(values), "sem_abs_rgb": sem(values)})
        out.append(row)
    return out


def choose_border_area(rows: list[dict[str, Any]], preferred: str) -> str:
    areas = sorted({str(row.get("seam_area", "")) for row in rows if str(row.get("seam_area", "")).strip()})
    if preferred in areas:
        return preferred
    for candidate in ["selected_cells_boundary", "edited_cells_boundary", "visited_cells_boundary"]:
        if candidate in areas:
            return candidate
    return areas[0] if areas else preferred


def border_summary(rows: list[dict[str, Any]], *, seam_area: str) -> list[dict[str, Any]]:
    wanted = [row for row in rows if row.get("seam_area") == seam_area]
    out: list[dict[str, Any]] = []
    for key, group in sorted(group_rows(wanted, ["task_name", "direction", "method", "budget"]).items()):
        values = [to_float(row.get("seam_excess_mean_abs_rgb")) for row in group]
        row = {name: value for name, value in zip(["task_name", "direction", "method", "budget"], key)}
        row.update({"seam_area": seam_area, "n": len(group), "mean_seam_excess_abs_rgb": mean(values), "sem_seam_excess_abs_rgb": sem(values)})
        out.append(row)
    return out


def paired_advantage(
    summary: list[dict[str, Any]],
    *,
    value_key: str,
    left_method: str,
    right_method: str,
    output_key: str,
) -> list[dict[str, Any]]:
    grouped = group_rows(summary, ["task_name", "direction", "budget"])
    out: list[dict[str, Any]] = []
    for (task_name, direction, budget), group in sorted(grouped.items()):
        by_method = {str(row.get("method")): row for row in group}
        if left_method not in by_method or right_method not in by_method:
            continue
        left = to_float(by_method[left_method].get(value_key))
        right = to_float(by_method[right_method].get(value_key))
        out.append(
            {
                "task_name": task_name,
                "direction": direction,
                "budget": budget,
                output_key: right - left if math.isfinite(left) and math.isfinite(right) else math.nan,
                f"{left_method}_{value_key}": left,
                f"{right_method}_{value_key}": right,
            }
        )
    return out


def attention_random_gain(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_rows(summary, ["task_name", "direction", "method", "budget"])
    out: list[dict[str, Any]] = []
    for (task_name, direction, method, budget), group in sorted(grouped.items()):
        by_selector = {str(row.get("selector")): row for row in group}
        if "attention" not in by_selector or "random" not in by_selector:
            continue
        attn = to_float(by_selector["attention"].get("mean_target_probability_delta"))
        rand = to_float(by_selector["random"].get("mean_target_probability_delta"))
        out.append(
            {
                "task_name": task_name,
                "direction": direction,
                "method": method,
                "budget": budget,
                "mean_attention_target_probability_delta": attn,
                "mean_random_target_probability_delta": rand,
                "attention_minus_random_delta": attn - rand if math.isfinite(attn) and math.isfinite(rand) else math.nan,
            }
        )
    return out


def load_plot_libs():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsi_cf_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def savefig(fig: Any, path: Path, formats: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")


def set_clean_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.7)


def save_prediction_lines(summary: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[dict[str, str]]:
    if not summary:
        return []
    plt = load_plot_libs()
    written: list[dict[str, str]] = []
    for (task_name, direction), group in group_rows(summary, ["task_name", "direction"]).items():
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharex=True)
        for selector in ["attention", "random"]:
            rows = sorted([row for row in group if row.get("selector") == selector and row.get("method") == "ours"], key=lambda row: int(row["budget"]))
            if not rows:
                continue
            x = [int(row["budget"]) for row in rows]
            y_delta = [to_float(row["mean_target_probability_delta"]) for row in rows]
            y_flip = [to_float(row["flip_to_target_rate"]) for row in rows]
            label = "attention" if selector == "attention" else "random"
            color = SELECTOR_COLORS[selector]
            axes[0].plot(x, y_delta, marker="o", color=color, label=label)
            axes[1].plot(x, y_flip, marker="o", color=color, label=label)
        axes[0].axhline(0.0, color="#444444", linewidth=0.8)
        axes[0].set_ylabel("target probability delta")
        axes[1].set_ylabel("flip-to-target rate")
        for ax in axes:
            ax.set_xlabel("steered cells")
            ax.legend(frameon=False)
            set_clean_axes(ax)
        fig.suptitle(task_label(str(task_name), str(direction)))
        fig.tight_layout()
        stem = out_dir / "prediction_transition" / f"{task_name}__{direction}"
        savefig(fig, stem, formats)
        plt.close(fig)
        written.append({"kind": "prediction_transition", "task_name": str(task_name), "direction": str(direction), "path_stem": str(stem)})
    return written


def save_gain_bar(rows: list[dict[str, Any]], out_dir: Path, formats: list[str], *, report_budget: int) -> list[dict[str, str]]:
    rows = [row for row in rows if int(row.get("budget", -1)) == int(report_budget) and row.get("method") == "ours"]
    if not rows:
        return []
    plt = load_plot_libs()
    rows = sorted(rows, key=lambda row: task_label(str(row["task_name"]), str(row["direction"])))
    labels = [task_label(str(row["task_name"]), str(row["direction"])) for row in rows]
    values = [to_float(row.get("attention_minus_random_delta")) for row in rows]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 3.8))
    colors = ["#1769aa" if value >= 0 else "#c7533a" for value in values]
    ax.bar(range(len(labels)), values, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("attention - random target delta")
    ax.set_title(f"Attention advantage at {report_budget} steered cells")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    set_clean_axes(ax)
    fig.tight_layout()
    stem = out_dir / "attention_vs_random_gain"
    savefig(fig, stem, formats)
    plt.close(fig)
    return [{"kind": "attention_vs_random_gain", "path_stem": str(stem)}]


def save_method_bar(
    rows: list[dict[str, Any]],
    out_dir: Path,
    formats: list[str],
    *,
    value_key: str,
    ylabel: str,
    title: str,
    stem_name: str,
    report_budget: int,
) -> list[dict[str, str]]:
    rows = [row for row in rows if int(row.get("budget", -1)) == int(report_budget)]
    if not rows:
        return []
    plt = load_plot_libs()
    groups = sorted(group_rows(rows, ["task_name", "direction"]).items(), key=lambda item: task_label(str(item[0][0]), str(item[0][1])))
    labels = [task_label(str(key[0]), str(key[1])) for key, _group in groups]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.0, 0.58 * len(labels)), 3.8))
    for offset, method in [(-width / 2, "ours"), (width / 2, "bad_naive")]:
        values = []
        for _key, group in groups:
            row = next((candidate for candidate in group if candidate.get("method") == method), None)
            values.append(to_float(row.get(value_key)) if row else math.nan)
        ax.bar([idx + offset for idx in x], values, width=width, label=method, color=METHOD_COLORS[method])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} at {report_budget} steered cells")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.legend(frameon=False)
    set_clean_axes(ax)
    fig.tight_layout()
    stem = out_dir / stem_name
    savefig(fig, stem, formats)
    plt.close(fig)
    return [{"kind": stem_name, "path_stem": str(stem)}]


def save_denominator_plot(root: Path, out_dir: Path, formats: list[str]) -> list[dict[str, str]]:
    rows = read_csv(root / "audit" / "benchmark_denominator_summary.csv")
    if not rows:
        return []
    plt = load_plot_libs()
    labels = [task_label(row["task_name"], row["direction"]) for row in rows]
    eligible = [to_int(row.get("eligible_test_slides")) for row in rows]
    included = [to_int(row.get("included_slides_with_regions")) for row in rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.58 * len(labels)), 3.8))
    ax.bar(x, eligible, color="#d4d4d4", label="eligible test slides")
    ax.bar(x, included, color="#1769aa", label="included with regions")
    ax.set_ylabel("slides")
    ax.set_title("Full-test denominator")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.legend(frameon=False)
    set_clean_axes(ax)
    fig.tight_layout()
    stem = out_dir / "full_test_denominator"
    savefig(fig, stem, formats)
    plt.close(fig)
    return [{"kind": "full_test_denominator", "path_stem": str(stem)}]


def main() -> None:
    args = build_arg_parser().parse_args()
    root = args.out_root
    out_dir = args.plot_dir or (root / "plots")
    formats = [token.strip() for token in str(args.formats).split(",") if token.strip()]

    prediction_rows = read_prediction_rows(root)
    visual_rows = read_visual_rows(root)
    raw_border_rows = read_border_rows(root)
    seam_area = choose_border_area(raw_border_rows, str(args.border_area))

    pred_summary = prediction_summary(prediction_rows)
    gain_rows = attention_random_gain(pred_summary)
    vis_summary = visual_summary(visual_rows, area=str(args.visual_area))
    seam_summary = border_summary(raw_border_rows, seam_area=seam_area)
    perturb_advantage = paired_advantage(
        vis_summary,
        value_key="mean_abs_rgb",
        left_method="ours",
        right_method="bad_naive",
        output_key="naive_minus_ours_mean_abs_rgb",
    )
    border_advantage = paired_advantage(
        seam_summary,
        value_key="mean_seam_excess_abs_rgb",
        left_method="ours",
        right_method="bad_naive",
        output_key="naive_minus_ours_seam_excess_abs_rgb",
    )

    tables_dir = out_dir / "tables"
    write_csv(tables_dir / "prediction_summary_by_budget.csv", pred_summary)
    write_csv(tables_dir / "attention_vs_random_gain_by_budget.csv", gain_rows)
    write_csv(tables_dir / "visual_perturbation_summary_by_budget.csv", vis_summary)
    write_csv(tables_dir / "border_discontinuity_summary_by_budget.csv", seam_summary)
    write_csv(tables_dir / "ours_vs_naive_visual_advantage_by_budget.csv", perturb_advantage)
    write_csv(tables_dir / "ours_vs_naive_border_advantage_by_budget.csv", border_advantage)

    written: list[dict[str, str]] = []
    written.extend(save_prediction_lines(pred_summary, out_dir, formats))
    written.extend(save_gain_bar(gain_rows, out_dir, formats, report_budget=int(args.report_budget)))
    written.extend(
        save_method_bar(
            vis_summary,
            out_dir,
            formats,
            value_key="mean_abs_rgb",
            ylabel="mean absolute RGB difference",
            title=f"Visual perturbation ({args.visual_area})",
            stem_name="ours_vs_naive_rgb_perturbation",
            report_budget=int(args.report_budget),
        )
    )
    written.extend(
        save_method_bar(
            seam_summary,
            out_dir,
            formats,
            value_key="mean_seam_excess_abs_rgb",
            ylabel="excess border inconsistency",
            title=f"Border discontinuity ({seam_area})",
            stem_name="ours_vs_naive_border_discontinuity",
            report_budget=int(args.report_budget),
        )
    )
    written.extend(save_denominator_plot(root, out_dir, formats))

    payload = {
        "out_root": str(root),
        "plot_dir": str(out_dir),
        "n_prediction_rows": len(prediction_rows),
        "n_visual_rows": len(visual_rows),
        "n_border_rows": len(raw_border_rows),
        "visual_area": str(args.visual_area),
        "border_area": seam_area,
        "report_budget": int(args.report_budget),
        "formats": formats,
        "plots": written,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plot_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
