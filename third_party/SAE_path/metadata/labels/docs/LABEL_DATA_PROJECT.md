# TCGA Label Data Project

Date: 2026-02-27

## Scope
Build a reproducible label data product for the SAE project.

Primary outputs:
- `metadata/labels/targets/*.tsv`
- `metadata/labels/master/case_labels_master.tsv`
- `metadata/labels/master/slide_labels_master.tsv`

## External Inputs Used For Final Merged Labels

1. cBio clinical table (downloaded snapshot):
   - URL: `https://raw.githubusercontent.com/GerkeLab/TCGAclinical/master/data/cBioportal_data.tsv`
   - Local cached file: `metadata/labels/sources/raw/cBioportal_data.tsv`
   - Used by: `metadata/labels/scripts/build_tcga_label_catalog.py`

2. GDC Files API (live query):
   - Endpoint: `https://api.gdc.cancer.gov/files`
   - Used by: `metadata/labels/scripts/fetch_gdc_slide_labels.py`

3. Immune subtype reference table (C1-C6):
   - URL: `https://raw.githubusercontent.com/CRI-iAtlas/ImmuneSubtypeClassifier/master/inst/extdata/five_signature_mclust_ensemble_results.tsv.gz`
   - Local cached file: `metadata/labels/sources/raw/immune_subtype_mclust.tsv.gz`
   - Used by: `metadata/labels/scripts/build_master_labels.py`

## Local Input

1. Manifest index (maps slides/files/patients):
   - `metadata/indexes/manifest_index.json`
   - Used by both scripts above.

## Build Pipeline

1. Fetch external raw sources:
```bash
python metadata/labels/scripts/fetch_label_sources.py \
  --out_dir metadata/labels/sources/raw
```

2. Build cBio slide catalog:
```bash
python metadata/labels/scripts/build_tcga_label_catalog.py \
  --manifest_index metadata/indexes/manifest_index.json \
  --out_dir metadata/labels/sources/intermediate \
  --raw_dir metadata/labels/sources/raw
```

3. Build GDC slide labels:
```bash
python metadata/labels/scripts/fetch_gdc_slide_labels.py \
  --manifest_index metadata/indexes/manifest_index.json \
  --out_dir metadata/labels/sources/intermediate
```

4. Build target + master labels:
```bash
python metadata/labels/scripts/build_master_labels.py \
  --manifest_index metadata/indexes/manifest_index.json \
  --cbio_slide_tsv metadata/labels/sources/intermediate/tcga_slide_label_catalog.tsv \
  --gdc_slide_tsv metadata/labels/sources/intermediate/tcga_gdc_slide_labels.tsv \
  --out_dir metadata/labels
```

5. Or run all steps with validation:
```bash
python metadata/labels/scripts/rebuild_labels_pipeline.py
```

## Notes On HPV External Tables

These are used in master-target build for CESC/HNSC HPV labels:
- `metadata/labels/sources/curated/hpv/nature2017_cesc_hpv_consensus.tsv`
- `metadata/labels/sources/curated/hpv/nature2015_hnsc_hpv_status.tsv`

Use them as reference/audit labels for HPV-focused experiments.
