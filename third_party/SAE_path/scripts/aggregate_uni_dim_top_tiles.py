from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Aggregate multiple scripts.uni_dim_top_tiles JSON outputs into one merged file + flat ranking CSV."
    )
    ap.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Input JSON paths or glob patterns (e.g. outputs/uni_axis_top_tiles/test_axes_*.json).",
    )
    ap.add_argument("--out-json", type=Path, required=True, help="Merged output JSON.")
    ap.add_argument("--out-csv", type=Path, required=True, help="Flat ranking CSV (one row per axis/sign).")
    ap.add_argument(
        "--sort-by",
        type=str,
        default="diversity_1_minus_mean_cosine",
        choices=[
            "diversity_1_minus_mean_cosine",
            "intra_set_mean_pairwise_cosine_uni",
            "bootstrap_tile_jaccard_at_n_estimate",
            "contrast_top1pct_over_median_eps",
            "top_site_entropy",
            "top_slide_entropy",
        ],
        help="Metric column to sort CSV by.",
    )
    ap.add_argument(
        "--sort-desc",
        action="store_true",
        help="Sort descending instead of ascending. Defaults to descending for similarity/stability/contrast, ascending for diversity/entropy.",
    )
    return ap


def _expand_inputs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in patterns:
        p = Path(raw)
        if any(ch in raw for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(raw)))
        elif p.exists():
            out.append(p)
    # de-dup preserve order
    seen = set()
    uniq = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def _nested_metric(metrics: dict[str, Any], key: str) -> float | None:
    return metrics.get(key)


def _default_sort_desc(metric_key: str) -> bool:
    # Higher similarity, stability, contrast are "better" (or more extreme); lower diversity/entropy are "purer".
    return metric_key not in {
        "diversity_1_minus_mean_cosine",
        "top_site_entropy",
        "top_slide_entropy",
    }


def main() -> None:
    args = _build_argparser().parse_args()
    in_files = _expand_inputs(args.inputs)
    if not in_files:
        raise SystemExit("No input files matched.")

    merged_axis_results: dict[str, Any] = {}
    source_files: list[str] = []
    configs: list[dict[str, Any]] = []
    axes_seen: set[int] = set()

    for p in in_files:
        obj = json.loads(p.read_text())
        if not isinstance(obj, dict):
            print(f"[warn] skip {p}: root is not dict")
            continue
        axis_results = obj.get("axis_results")
        if not isinstance(axis_results, dict):
            print(f"[warn] skip {p}: missing axis_results")
            continue
        source_files.append(str(p))
        configs.append(
            {
                "manifest": obj.get("manifest"),
                "split": obj.get("split"),
                "top_n": obj.get("top_n"),
                "zscore": obj.get("zscore"),
                "config": obj.get("config", {}),
            }
        )
        for ax_str, entry in axis_results.items():
            try:
                ax = int(ax_str)
            except Exception:
                continue
            if ax in axes_seen:
                print(f"[warn] duplicate axis {ax} (keeping first seen, skipping from {p.name})")
                continue
            merged_axis_results[str(ax)] = entry
            axes_seen.add(ax)

    if not merged_axis_results:
        raise SystemExit("No axis_results were loaded from inputs.")

    # Flat rows for CSV ranking.
    rows: list[dict[str, Any]] = []
    for ax_str, entry in merged_axis_results.items():
        signs = entry.get("signs", {})
        if not isinstance(signs, dict):
            continue
        for sign, sign_entry in signs.items():
            metrics = sign_entry.get("metrics", {}) if isinstance(sign_entry, dict) else {}
            contrast = metrics.get("activation_contrast_reservoir_approx", {}) if isinstance(metrics, dict) else {}
            row = {
                "axis_index": int(ax_str),
                "sign": sign,
                "top_n_actual": metrics.get("top_n_actual"),
                "intra_set_mean_pairwise_cosine_uni": metrics.get("intra_set_mean_pairwise_cosine_uni"),
                "diversity_1_minus_mean_cosine": metrics.get("diversity_1_minus_mean_cosine"),
                "bootstrap_tile_jaccard_at_n_estimate": metrics.get("bootstrap_tile_jaccard_at_n_estimate"),
                "top_slide_entropy": metrics.get("top_slide_entropy"),
                "top_site_entropy": metrics.get("top_site_entropy"),
                "top_unique_slides": metrics.get("top_unique_slides"),
                "top_unique_sites": metrics.get("top_unique_sites"),
                "contrast_top1pct_over_median_eps": contrast.get("contrast_top1pct_over_median_eps") if isinstance(contrast, dict) else None,
                "p99_score_reservoir": contrast.get("p99") if isinstance(contrast, dict) else None,
                "p50_score_reservoir": contrast.get("p50") if isinstance(contrast, dict) else None,
            }
            rows.append(row)

    sort_key = args.sort_by
    sort_desc = bool(args.sort_desc) or _default_sort_desc(sort_key)
    rows_sorted = sorted(
        rows,
        key=lambda r: (float("-inf") if _nested_metric(r, sort_key) is None else float(_nested_metric(r, sort_key))),
        reverse=sort_desc,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    merged = {
        "source_files": source_files,
        "n_source_files": len(source_files),
        "n_axes_merged": len(merged_axis_results),
        "sort_by": sort_key,
        "sort_desc": sort_desc,
        "configs_seen": configs,
        "axis_results": {k: merged_axis_results[k] for k in sorted(merged_axis_results.keys(), key=lambda x: int(x))},
    }
    args.out_json.write_text(json.dumps(merged, indent=2))

    fieldnames = [
        "axis_index",
        "sign",
        "top_n_actual",
        "intra_set_mean_pairwise_cosine_uni",
        "diversity_1_minus_mean_cosine",
        "bootstrap_tile_jaccard_at_n_estimate",
        "contrast_top1pct_over_median_eps",
        "p99_score_reservoir",
        "p50_score_reservoir",
        "top_slide_entropy",
        "top_site_entropy",
        "top_unique_slides",
        "top_unique_sites",
    ]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

    print("Saved merged JSON:", args.out_json)
    print("Saved ranking CSV:", args.out_csv)


if __name__ == "__main__":
    main()
