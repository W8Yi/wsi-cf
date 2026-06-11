from __future__ import annotations

from wsi_cf.data.borderline_selection import (
    borderline_region_score,
    select_nonoverlapping_regions,
)


def test_borderline_score_prefers_moderate_attention_band() -> None:
    centered = borderline_region_score(
        attention_seed_fraction=0.359375,
        region_target_prob=0.55,
        tissue_score=1.0,
        valid_feature_fraction=1.0,
    )
    sparse = borderline_region_score(
        attention_seed_fraction=0.02,
        region_target_prob=0.55,
        tissue_score=1.0,
        valid_feature_fraction=1.0,
    )
    saturated = borderline_region_score(
        attention_seed_fraction=0.95,
        region_target_prob=0.55,
        tissue_score=1.0,
        valid_feature_fraction=1.0,
    )

    assert centered["borderline_region_score"] > sparse["borderline_region_score"]
    assert centered["borderline_region_score"] > saturated["borderline_region_score"]


def test_borderline_score_prefers_borderline_target_probability() -> None:
    borderline = borderline_region_score(
        attention_seed_fraction=0.359375,
        region_target_prob=0.55,
        tissue_score=1.0,
        valid_feature_fraction=1.0,
    )
    already_target_like = borderline_region_score(
        attention_seed_fraction=0.359375,
        region_target_prob=0.98,
        tissue_score=1.0,
        valid_feature_fraction=1.0,
    )
    source_like = borderline_region_score(
        attention_seed_fraction=0.359375,
        region_target_prob=0.05,
        tissue_score=1.0,
        valid_feature_fraction=1.0,
    )

    assert borderline["borderline_region_score"] > already_target_like["borderline_region_score"]
    assert borderline["borderline_region_score"] > source_like["borderline_region_score"]


def test_select_nonoverlapping_regions_suppresses_near_duplicates() -> None:
    rows = [
        {"region_gx0": 0, "region_gy0": 0, "borderline_region_score": 1.0},
        {"region_gx0": 2, "region_gy0": 0, "borderline_region_score": 0.9},
        {"region_gx0": 8, "region_gy0": 0, "borderline_region_score": 0.8},
    ]

    selected = select_nonoverlapping_regions(
        rows,
        side=8,
        max_regions=5,
        max_overlap_fraction=0.50,
    )

    assert [(row["region_gx0"], row["region_gy0"]) for row in selected] == [(0, 0), (8, 0)]


def test_select_final_region_candidates_all_slides_per_label() -> None:
    from wsi_cf.data.region_selection import select_final_region_candidates

    eligible_by_label: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for label in (0, 1):
        for slide_idx in range(2):
            for rank in range(3):
                eligible_by_label[label].append(
                    {
                        "label": label,
                        "slide_key": f"label{label}_slide{slide_idx}",
                        "candidate_rank_in_slide": rank + 1,
                    }
                )

    rows = select_final_region_candidates(
        eligible_by_label,
        final_regions_per_label=99,
        final_slides_per_label=0,
        final_regions_per_slide=2,
    )

    assert len(rows) == 8
    assert {row["selection_region_rank_in_slide"] for row in rows} == {1, 2}
    assert {row["selection_slide_rank_in_label"] for row in rows} == {1, 2}
