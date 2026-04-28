# Labels Guide (Plain Language)

Date: 2026-02-27

## What this file is

This explains our label dataset in simple terms for people without a medical background.
Think of this as a "data dictionary + pipeline map" for the files under `metadata/labels/`.

## What is a "label" here?

A label is just a tag attached to a case (patient/sample) that we can use for analysis.

Examples:
- `HPV+` or `HPV-`
- cancer stage
- whether certain genes are mutated
- survival status (`Alive`/`Dead`)

## Current data size

- Cases: `9,547`
- Slides: `11,594`
- Tracked label conflicts: `2` (both in HPV for HNSC)

Source: `metadata/labels/qc/build_summary.json`

## Labels we currently have

Coverage means "how much of the dataset has a non-empty value for that label."

1. `hpv_status`
- Plain meaning: whether HPV virus signal is present.
- Used for: HNSC/CESC projects.
- Coverage: `562 / 9,547` overall (`5.89%`), `562 / 719` where applicable (`78.16%`).

2. `immune_subtype`
- Plain meaning: broad immune-pattern category.
- Values: `C1` to `C6` (six immune landscape groups).
- Coverage: `8,237 / 9,547` (`86.28%`).

Simple intuition for `C1`-`C6`:
- `C1`: wound-healing-like immune environment
- `C2`: interferon-gamma dominant (inflamed)
- `C3`: inflammatory but generally less aggressive than C2
- `C4`: lymphocyte-depleted (fewer adaptive immune cells)
- `C5`: immunologically quiet
- `C6`: TGF-beta dominant (immune suppression pattern)

3. `msi_status`
- Plain meaning: whether tumor shows mismatch-repair instability pattern.
- Used for: COAD/STAD projects.
- Coverage: `650 / 9,547` overall (`6.81%`), `650 / 809` where applicable (`80.35%`).

4. `pam50_subtype`
- Plain meaning: breast-cancer molecular subtype group.
- Used for: BRCA projects.
- Coverage: `942 / 9,547` overall (`9.87%`), `942 / 1,055` where applicable (`89.29%`).

5. `tumor_grade`
- Plain meaning: how abnormal/aggressive tissue appears (pathology grading system).
- Coverage: `3,790 / 9,547` (`39.70%`).

6. `stage`
- Plain meaning: disease spread level (clinical/pathologic stage).
- Coverage: `6,338 / 9,547` (`66.39%`).

7. `os_status`
- Plain meaning: overall survival status (`Alive` or `Dead`).
- Coverage: `9,526 / 9,547` (`99.78%`).

8. `tp53_mutated`
- Plain meaning: whether TP53 gene has a mutation.
- Coverage: `8,710 / 9,547` (`91.23%`).

9. `kras_mutated`
- Plain meaning: whether KRAS gene has a mutation.
- Coverage: `8,710 / 9,547` (`91.23%`).

10. `tumor_purity`
- Plain meaning: estimated fraction of tumor cells in the sample.
- Coverage: `8,902 / 9,547` (`93.24%`).

Source: `metadata/labels/qc/coverage_by_target.tsv`

## Simple interpretation examples

- `Unknown` means no confident value was available from our current sources.
- A label can be "low overall coverage" but still "high applicable coverage":
  - Example: `pam50_subtype` is only for BRCA, so overall looks low, but BRCA-only coverage is high.

## Our labeling scheme (how data flows)

1. Raw sources (`metadata/labels/sources/raw`)
- Downloaded/source snapshots (cBio, subtype, viral, purity, mutation tables).

2. Intermediate slide tables (`metadata/labels/sources/intermediate`)
- Slide-level joins from cBio and GDC APIs.

3. Curated reference sources (`metadata/labels/sources/curated/hpv`)
- Manually curated HPV references for CESC/HNSC.

4. Final outputs
- Per-target files: `metadata/labels/targets/*.tsv`
- Master case table: `metadata/labels/master/case_labels_master.tsv`
- Master slide table: `metadata/labels/master/slide_labels_master.tsv`

5. Quality control and traceability (`metadata/labels/qc`)
- `coverage_by_target.tsv`: coverage stats
- `conflicts.tsv`: source disagreements
- `build_summary.json`: source hashes + output metadata
- `build_pipeline_run.json`: exact commands and output hashes for reproducibility

## Why this is error-proof and debuggable

- Every build records file hashes and timestamps.
- Conflicts are explicitly logged.
- Source files are versioned as raw snapshots.
- Rebuild is one command:

```bash
python metadata/labels/scripts/rebuild_labels_pipeline.py
```

## Important caveats

- These are research labels, not clinical diagnoses.
- Some labels are project-specific and not meaningful outside those projects.
- `Unknown` can mean missing data, not "negative."
- `immune_subtype` comes from a pan-cancer RNA-based reference and may not exist for every case.
