# TCGA UNI2H Project Pipeline

This folder provides a simple pipeline to process TCGA cohorts one-by-one:

1. Download WSIs
2. Extract UNI2H features
3. Upload to Hugging Face
4. Delete local slides for that project

## Local output layout

All project outputs go to:

`/research/projects/mllab/WSI/TCGA-XXXX/`

with this structure:

```text
TCGA-XXXX/
  slides/
  features/           # .h5 only
  coords/             # internal .coords.csv files
  vis/
  shard_0.log
  shard_1.log
  ...
  slide_keys.txt
```

## Hugging Face layout

For each project:

```text
TCGA-XXXX/features
TCGA-XXXX/vis
```

No summary JSON files are kept in project outputs or uploaded.
`coords/` is kept locally only and is not uploaded.
Only `.h5` files are uploaded under `features`.

## Download from Hugging Face

Authenticate once:

```bash
/common/users/wq50/envs/pace/bin/hf auth login
```

Set repo id:

```bash
DATASET_ID="W8Yi/tcga-wsi-uni2h-features"
```

Download the full dataset (all uploaded projects/files):

```bash
/common/users/wq50/envs/pace/bin/hf download "$DATASET_ID" \
  --repo-type dataset \
  --local-dir /research/projects/mllab/WSI/TCGA_features_hf
```

Download one project only (example: `TCGA-GBM`, both features and overlays):

```bash
/common/users/wq50/envs/pace/bin/hf download "$DATASET_ID" \
  --repo-type dataset \
  --include "TCGA-GBM/features/*" \
  --include "TCGA-GBM/vis/*" \
  --local-dir /research/projects/mllab/WSI/TCGA_features_hf
```

Download only `.h5` features for one project:

```bash
/common/users/wq50/envs/pace/bin/hf download "$DATASET_ID" \
  --repo-type dataset \
  --include "TCGA-GBM/features/*.h5" \
  --local-dir /research/projects/mllab/WSI/TCGA_features_hf
```

Dry-run before downloading (preview matched files):

```bash
/common/users/wq50/envs/pace/bin/hf download "$DATASET_ID" \
  --repo-type dataset \
  --include "TCGA-GBM/features/*" \
  --include "TCGA-GBM/vis/*" \
  --dry-run
```

Notes:
- Re-running the same command is safe and resumes/uses cache for already downloaded files.
- Increase parallel download workers if needed: `--max-workers 16`.

## Run one project

```bash
bash scripts/tcga_project_pipeline/run_tcga_project_uni2h.sh --project TCGA-HNSC
```

The script prints progress like:
- `[download 23/472] ...`
- extractor prints `[filter i/N]` and `[encode i/N]`

## Run all projects (predefined list)

```bash
bash scripts/tcga_project_pipeline/run_all_tcga_projects_uni2h.sh
```

This prints project-level progress:
- `[project 1/33] TCGA-ACC`

## Skip control

```bash
# Run only selected projects
ONLY_PROJECTS=TCGA-HNSC,TCGA-CESC \
bash scripts/tcga_project_pipeline/run_all_tcga_projects_uni2h.sh

# Skip selected projects
SKIP_PROJECTS=TCGA-GBM,TCGA-LGG \
bash scripts/tcga_project_pipeline/run_all_tcga_projects_uni2h.sh
```

## Useful environment variables

```bash
DATASET_ID=w8yi/tcga-wsi-uni2h-features
UPLOAD_MODE=standard    # or large
GPU_LIST=0,1,2,3        # or cpu
BATCH_SIZE=1024
FILTER_WORKERS=4
GDC_TOKEN=/path/to/gdc-user-token.txt
DELETE_SLIDES_AFTER_PROJECT=1
DOWNLOAD_JOBS=4
DOWNLOAD_N_PROCESSES=4
```

You can also skip stages:

```bash
bash scripts/tcga_project_pipeline/run_tcga_project_uni2h.sh \
  --project TCGA-HNSC --skip-download --skip-extract
```

## Migrate existing legacy HNSC outputs

```bash
# default MODE=copy (safe, keeps old folders)
bash scripts/tcga_project_pipeline/migrate_hnsc_legacy_to_project_layout.sh

# move mode (removes old legacy folders after sync)
MODE=move bash scripts/tcga_project_pipeline/migrate_hnsc_legacy_to_project_layout.sh
```
