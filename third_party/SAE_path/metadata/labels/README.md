# Label Data Layout

This directory is organized to keep labels reproducible, auditable, and easy to debug.

## Structure

- `metadata/labels/sources/raw`
  - Raw downloaded/source snapshots.
- `metadata/labels/sources/intermediate`
  - Slide-level intermediate tables from cBio + GDC.
- `metadata/labels/sources/curated/hpv`
  - Curated HPV reference tables.
- `metadata/labels/targets`
  - One file per target label task (case level).
- `metadata/labels/master`
  - Unified master tables (case + slide).
- `metadata/labels/qc`
  - Coverage/conflict/build metadata for traceability.
- `metadata/labels/docs`
  - Background docs and source reviews.
  - Plain-language overview: `metadata/labels/docs/LABELS_PLAIN_LANGUAGE.md`
- `metadata/labels/scripts`
  - Label build/fetch/rebuild scripts.

## Main Outputs

- `metadata/labels/targets/*.tsv`
  - `hpv_status_case.tsv`
  - `immune_subtype_case.tsv`
  - `msi_status_case.tsv`
  - `pam50_case.tsv`
  - `tumor_grade_case.tsv`
  - `stage_case.tsv`
  - `survival_case.tsv`
  - `mutation_tp53_kras_case.tsv`
  - `tumor_purity_case.tsv`
- `metadata/labels/master/case_labels_master.tsv`
- `metadata/labels/master/slide_labels_master.tsv`
- `metadata/labels/qc/coverage_by_target.tsv`
- `metadata/labels/qc/conflicts.tsv`
- `metadata/labels/qc/build_summary.json`
- `metadata/labels/qc/build_pipeline_run.json`

## Rebuild (One Command)

From repo root:

```bash
python metadata/labels/scripts/rebuild_labels_pipeline.py
```

Optional refresh:

```bash
python metadata/labels/scripts/rebuild_labels_pipeline.py --refresh_sources --refresh_cbio_download
```

## Step-by-step Scripts

- `metadata/labels/scripts/fetch_label_sources.py`
- `metadata/labels/scripts/build_tcga_label_catalog.py`
- `metadata/labels/scripts/fetch_gdc_slide_labels.py`
- `metadata/labels/scripts/fetch_cbio_mutation_flags.py`
- `metadata/labels/scripts/build_master_labels.py`

## Debug / Audit

- Verify source hashes:
  - `metadata/labels/sources/raw/source_download_manifest.json`
- Verify build provenance + input hashes:
  - `metadata/labels/qc/build_summary.json`
- Verify executed command history + output hashes:
  - `metadata/labels/qc/build_pipeline_run.json`
- Check label coverage by task:
  - `metadata/labels/qc/coverage_by_target.tsv`
- Check conflicts:
  - `metadata/labels/qc/conflicts.tsv`
