"""Deterministic final region selection helpers."""

from __future__ import annotations

from typing import Any


def select_final_region_candidates(
    eligible_by_label: dict[int, list[dict[str, Any]]],
    *,
    final_regions_per_label: int,
    final_slides_per_label: int = 0,
    final_regions_per_slide: int = 0,
) -> list[dict[str, Any]]:
    """Select final attention-mined regions while preserving deterministic ranking."""
    if int(final_slides_per_label) <= 0 and int(final_regions_per_slide) <= 0:
        final_rows: list[dict[str, Any]] = []
        for label in (0, 1):
            for rank, row in enumerate(eligible_by_label.get(label, [])[: int(final_regions_per_label)], start=1):
                picked = dict(row)
                picked["selection_rank_in_label"] = int(rank)
                final_rows.append(picked)
        return final_rows

    if int(final_regions_per_slide) <= 0:
        raise ValueError("--final-regions-per-slide must be positive when --final-slides-per-label is set")

    final_rows = []
    for label in (0, 1):
        rows_by_slide: dict[str, list[dict[str, Any]]] = {}
        for row in eligible_by_label.get(label, []):
            rows_by_slide.setdefault(str(row["slide_key"]), []).append(row)
        for slide_rows in rows_by_slide.values():
            slide_rows.sort(key=lambda row: int(row.get("candidate_rank_in_slide", row.get("candidate_rank", 0))))
        eligible_slides = [slide_key for slide_key in sorted(rows_by_slide) if len(rows_by_slide[slide_key]) >= int(final_regions_per_slide)]
        requested_slides = int(final_slides_per_label)
        if requested_slides <= 0:
            requested_slides = len(eligible_slides)
        if len(eligible_slides) < int(requested_slides):
            raise RuntimeError(
                f"Label {label} has only {len(eligible_slides)} slides with at least "
                f"{int(final_regions_per_slide)} eligible regions; requested {int(requested_slides)}"
            )
        rank_in_label = 0
        for slide_rank, slide_key in enumerate(eligible_slides[: int(requested_slides)], start=1):
            for region_rank, row in enumerate(rows_by_slide[slide_key][: int(final_regions_per_slide)], start=1):
                rank_in_label += 1
                picked = dict(row)
                picked["selection_rank_in_label"] = int(rank_in_label)
                picked["selection_slide_rank_in_label"] = int(slide_rank)
                picked["selection_region_rank_in_slide"] = int(region_rank)
                final_rows.append(picked)
    return final_rows
