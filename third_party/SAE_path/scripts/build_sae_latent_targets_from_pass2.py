from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build per-latent SAE target/clamp values from pass2 top-tile mining metadata "
            "(e.g. pass2_top_tiles_top_activation_*.json)."
        )
    )
    ap.add_argument(
        "--pass2-json",
        type=str,
        required=True,
        help="Path to pass2 mining JSON containing `top_tiles` and `selected_latents`.",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        required=True,
        help="Output JSON with per-latent stats and target/clamp presets.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Use only the first N top tiles per latent (0 = use all listed tiles).",
    )
    ap.add_argument(
        "--score-key",
        type=str,
        default="score",
        help="Key used for activation score in each top-tile record (default: score).",
    )
    return ap


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("Cannot compute percentile of empty list.")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 100:
        return float(sorted_vals[-1])
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Expected non-empty score list.")
    xs = sorted(float(v) for v in values)
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {
        "count": float(n),
        "min": float(xs[0]),
        "p10": _percentile(xs, 10),
        "p25": _percentile(xs, 25),
        "p50": _percentile(xs, 50),
        "p75": _percentile(xs, 75),
        "p90": _percentile(xs, 90),
        "p95": _percentile(xs, 95),
        "p99": _percentile(xs, 99),
        "max": float(xs[-1]),
        "mean": float(mean),
        "std": float(math.sqrt(max(var, 0.0))),
    }


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _to_int_list(vals: list[Any]) -> list[int]:
    out: list[int] = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def main() -> None:
    args = _build_argparser().parse_args()
    in_path = Path(args.pass2_json)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_json(in_path)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected dict at root in {in_path}, got {type(data).__name__}")
    if "top_tiles" not in data or not isinstance(data["top_tiles"], dict):
        raise SystemExit(f"{in_path} does not contain a dict `top_tiles` section.")

    top_tiles: dict[str, Any] = data["top_tiles"]
    selected_latents = _to_int_list(data.get("selected_latents", []))
    selected_latent_set = {int(x) for x in selected_latents}

    per_latent: dict[str, Any] = {}
    skipped_empty: list[int] = []

    # Preserve the pass2-selected order if available; otherwise sort keys numerically.
    if selected_latents:
        latent_order = [int(k) for k in selected_latents if str(int(k)) in top_tiles]
        latent_order += sorted(
            [int(k) for k in top_tiles.keys() if int(k) not in set(latent_order)]
        )
    else:
        latent_order = sorted(int(k) for k in top_tiles.keys())

    for latent_idx in latent_order:
        records = top_tiles.get(str(latent_idx), [])
        if not isinstance(records, list):
            continue
        if args.top_n and args.top_n > 0:
            records = records[: int(args.top_n)]

        scores: list[float] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if args.score_key not in rec:
                continue
            try:
                scores.append(float(rec[args.score_key]))
            except Exception:
                continue

        if not scores:
            skipped_empty.append(int(latent_idx))
            continue

        st = _stats(scores)
        # Presets for SAE editing. These are pragmatic defaults, not "ground truth".
        per_latent[str(latent_idx)] = {
            "latent_idx": int(latent_idx),
            "source_top_tiles_count_used": int(len(scores)),
            "score_key": args.score_key,
            "stats_top_scores": st,
            "target_presets": {
                "soft": float(st["p50"]),
                "medium": float(st["p75"]),
                "strong": float(st["p90"]),
                "very_strong": float(st["p95"]),
            },
            "clamp_presets": {
                "soft": float(st["p75"]),
                "medium": float(st["p50"]),
                "strong": float(st["p25"]),
            },
            "notes": [
                "Targets/clamps are derived from pass2 top-tile activation scores for this latent.",
                "For latent_target mode, start with target_presets.soft or medium.",
                "For latent_clamp mode (suppression), start with clamp_presets.soft or medium.",
            ],
        }

    # Flat maps are convenient for quick lookups and future CLI integration.
    flat_target_p50 = {k: v["stats_top_scores"]["p50"] for k, v in per_latent.items()}
    flat_target_p75 = {k: v["stats_top_scores"]["p75"] for k, v in per_latent.items()}
    flat_target_p90 = {k: v["stats_top_scores"]["p90"] for k, v in per_latent.items()}

    out = {
        "source_pass2_json": str(in_path),
        "source_top_n_requested": int(args.top_n),
        "score_key": args.score_key,
        "selected_latents_count": int(len(selected_latents)),
        "latents_with_targets_count": int(len(per_latent)),
        "selected_latents_in_output_order": [int(x) for x in latent_order if str(int(x)) in per_latent],
        "skipped_latents_no_scores": skipped_empty,
        "usage_notes": [
            "Use per_latent[<idx>].target_presets.* for latent_target mode.",
            "Use per_latent[<idx>].clamp_presets.* for latent_clamp mode.",
            "These values are based on top activations and may be too strong on a different tile domain; start with soft/medium.",
        ],
        "flat_maps": {
            "target_p50": flat_target_p50,
            "target_p75": flat_target_p75,
            "target_p90": flat_target_p90,
        },
        "per_latent": per_latent,
    }

    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote latent targets for {len(per_latent)} latents -> {out_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _WSI_SAE_SRC = Path(__file__).resolve().parents[2] / "wsi-sae" / "src"
    if _WSI_SAE_SRC.exists():
        sys.path.insert(0, str(_WSI_SAE_SRC))
        from wsi_sae.cli import main as _wsi_sae_main

        _wsi_sae_main(["build-targets", *sys.argv[1:]])
    else:
        main()
