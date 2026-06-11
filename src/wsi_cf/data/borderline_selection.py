"""Pure helpers for borderline HNSCC HPV region ranking."""

from __future__ import annotations

from typing import Any, Sequence


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def band_score(value: float, *, center: float, half_width: float) -> float:
    width = max(float(half_width), 1e-8)
    return clamp01(1.0 - abs(float(value) - float(center)) / width)


def source_target_probabilities(*, label: int, prob_pos: float) -> tuple[float, float]:
    prob_pos = clamp01(float(prob_pos))
    if int(label) == 1:
        return float(prob_pos), float(1.0 - prob_pos)
    return float(1.0 - prob_pos), float(prob_pos)


def borderline_region_score(
    *,
    attention_seed_fraction: float,
    region_target_prob: float,
    tissue_score: float,
    valid_feature_fraction: float,
    attention_center: float = 0.359375,
    attention_width: float = 0.30,
    target_prob_peak: float = 0.55,
    target_prob_width: float = 0.45,
) -> dict[str, float]:
    attention_score = band_score(
        float(attention_seed_fraction),
        center=float(attention_center),
        half_width=float(attention_width),
    )
    target_prob_score = band_score(
        float(region_target_prob),
        center=float(target_prob_peak),
        half_width=float(target_prob_width),
    )
    tissue = clamp01(float(tissue_score))
    valid = clamp01(float(valid_feature_fraction))
    total = 0.45 * attention_score + 0.35 * target_prob_score + 0.15 * tissue + 0.05 * valid
    return {
        "borderline_attention_band_score": float(attention_score),
        "borderline_target_prob_score": float(target_prob_score),
        "borderline_region_score": float(total),
    }


def region_overlap_fraction(a: dict[str, Any], b: dict[str, Any], *, side: int) -> float:
    ax0 = int(a["region_gx0"])
    ay0 = int(a["region_gy0"])
    bx0 = int(b["region_gx0"])
    by0 = int(b["region_gy0"])
    ax1 = ax0 + int(side)
    ay1 = ay0 + int(side)
    bx1 = bx0 + int(side)
    by1 = by0 + int(side)
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    return float(ix * iy) / float(max(int(side) * int(side), 1))


def select_nonoverlapping_regions(
    rows: Sequence[dict[str, Any]],
    *,
    side: int,
    max_regions: int,
    max_overlap_fraction: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if len(selected) >= int(max_regions):
            break
        if any(region_overlap_fraction(row, kept, side=int(side)) > float(max_overlap_fraction) for kept in selected):
            continue
        selected.append(dict(row))
    return selected
