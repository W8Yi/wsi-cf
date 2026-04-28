# HPV Label Source Review (CESC + HNSC)

Date: 2026-02-27

## Goal
Compare available HPV label sources for TCGA CESC/HNSC and identify the best source for benchmarking in this repo.

## External Reference Sources

1. CESC (TCGA Nature 2017):
   - Paper: https://www.nature.com/articles/nature21386
   - Supplementary ZIP: https://static-content.springer.com/esm/art%3A10.1038%2Fnature21386/MediaObjects/41586_2017_BFnature21386_MOESM213_ESM.zip
   - HPV table used: `Supplemental Table 3-HPV_Consensus_Table.xlsx`
2. HNSC (TCGA Nature 2015):
   - Paper: https://www.nature.com/articles/nature14129
   - Supplementary ZIP: https://static-content.springer.com/esm/art%3A10.1038%2Fnature14129/MediaObjects/41586_2015_BFnature14129_MOESM116_ESM.zip
   - HPV table used: `nature14129-s2/1.2.xlsx` (contains `Final_HPV_Status`)
3. GDC publication landing page for CESC:
   - https://gdc.cancer.gov/about-data/publications/cesc_2017

## Extracted Official Tables (saved in repo)

- `metadata/labels/sources/curated/hpv/nature2017_cesc_hpv_consensus.tsv` (178 cases)
- `metadata/labels/sources/curated/hpv/nature2015_hnsc_hpv_status.tsv` (279 cases)

## Cohort Comparison Summary

See full table: `metadata/labels/hpv_label_source_comparison.tsv`

Key points:

1. CESC
   - Official Nature 2017 overlap with this cohort: 146/269 cases (54.3%), HPV+=138, HPV-=8.
2. HNSC
   - Official Nature 2015 overlap with this cohort: 222/450 cases (49.3%).
   - cBio subtype-derived HPV labels (`HNSC_HPV+/-`) cover 412/450 cases (91.6%) and agree 216/218 (99.1%) on overlap with official labels.

## Recommended Best Sources

1. CESC HPV status/type:
   - Best high-confidence source: Nature 2017 Supplementary Table 3.
   - Operationally in this repo: use `sources/curated/hpv/nature2017_cesc_hpv_consensus.tsv`.
2. HNSC HPV status:
   - Best practical source for coverage + reliability: cBio subtype (`HNSC_HPV+/-`) from `tcga_slide_labels_merged.tsv`.
   - Use Nature 2015 table as audit/anchor where overlap exists.

## Known mismatches vs official HNSC table

- `TCGA-BA-6869`: official HPV-, cBio HPV+
- `TCGA-CV-5970`: official HPV-, cBio HPV+
