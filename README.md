# WSI Counterfactual Steering

`wsi_cf` is now narrowed to the workflow we are actively using:

- export balanced `10x`, `1024x1024` HNSCC region banks with aligned `4x4` UNI feature grids
- run SAE prototype steering over selected cells inside those saved `10x` regions with PixCell-1024

This repo is intentionally no longer a grab-bag of every experimental runner. The current focus is the sampled-bank `10x` SAE case workflow built around [`export_hnscc_region_bank_10x.py`](scripts/export_hnscc_region_bank_10x.py) and [`run_region_bank_10x_sae_cases.py`](scripts/run_region_bank_10x_sae_cases.py).

## What Belongs Here

- balanced `10x` region-bank export
- SAE selected-cell steering on saved `10x` region banks
- the small shared modules those two workflows depend on
- focused tests, docs, and configs for that pipeline

## What Stays In The Legacy Repo

- generic SAE training
- generic MIL training
- broad mining and visualization utilities
- notebooks, legacy outputs, `wandb/`, and unrelated runs

## Directory Overview

- `docs/`: migration notes and design docs
- `docs/METHODS.md`: current steering method, formulas, and implementation caveats
- `docs/PAPER.md`: living paper outline, claims, figure plan, and writing checklist
- `docs/PROGRESS.md`: running log of experiments, results, failures, and next ideas
- `configs/`: example path/config templates
- `scripts/`: thin CLI entrypoints for the focused CF workflows
- `src/wsi_cf/common/`: shared runtime, path, and I/O helpers
- `src/wsi_cf/data/`: H5 loading, slide utilities, donor-pool helpers
- `src/wsi_cf/steering/`: manifest parsing, real-grid validation, feature replacement
- `src/wsi_cf/generation/`: PixCell windowing and MultiDiffusion helpers
- `src/wsi_cf/eval/`: HNSCC HPV evaluation helpers
- `tests/`: focused unit and smoke tests
- `artifacts/`: local manifests, previews, smoke-test outputs, and generated examples

## Data Assumptions

The current workflow assumes:

- local HNSCC slides under `/common/users/wq50/HNSCC/HNSCC_slides`
- TCGA HNSC UNI2 H5 features under `/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2`
- split metadata from `metadata/manifests/hnsc_hpv_5fold`
- PixCell backbones `StonyBrook-CVLab/PixCell-256` and `StonyBrook-CVLab/PixCell-1024`
- repo-local SAE prototype bundle under `artifacts/sae_prototypes/hnscc_hpv_split0_selected`

Raw slides and large feature stores stay outside `wsi_cf`.

## Supported Workflow

- export a balanced bank of real `10x` `1024x1024` regions with aligned `4x4x1536` UNI grids
- run SAE-selected-cell steering cases like `baseline`, `random_two`, `neighbor_three`, and `block_2x2`
- compare source actual, source regenerated baseline, and steered outputs by source region

## Quickstarts

These examples assume you are running from `/common/users/wq50/SAE_path` with the `pace` environment active.

### 1. Export A Random `10x` Region Bank With Aligned Features

```bash
python wsi_cf/scripts/export_hnscc_region_bank_10x.py \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --features-dir /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2 \
  --slides-dir /common/users/wq50/HNSCC/HNSCC_slides \
  --out-dir /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_1024 \
  --target-magnification 10 \
  --region-size 1024 \
  --grid-step-px 256 \
  --regions-total 20 \
  --seed 7 \
  --device cuda:0
```

### 2. Run The 10x SAE Case Sweep

The current repo-local prototype bundle lives here:

- `artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz`
- `artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.json`

The default concept latents are:

- HPV+ prototype latent: `2645`
- HPV- prototype latent: `7036`

```bash
python wsi_cf/scripts/run_region_bank_10x_sae_cases.py \
  --region-bank-csv /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_1024_sample4/region_bank.csv \
  --out-dir /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_1024_sample4_sae_cases_preserve3_s0.2_mid0.5_rerun \
  --cases baseline,random_two,neighbor_three,block_2x2 \
  --direction hpv_pos \
  --max-sources 4 \
  --grid-step-px 256 \
  --dtype fp16 \
  --pix_model_id StonyBrook-CVLab/PixCell-1024 \
  --pix_pipeline_id StonyBrook-CVLab/PixCell-pipeline \
  --vae_model_id stabilityai/stable-diffusion-3-medium-diffusers \
  --vae_subfolder vae \
  --steps 30 \
  --guidance 2.0 \
  --patch-batch 256 \
  --prototype-strength 0.8 \
  --steer-blend 1.0 \
  --preserve-outside-latents \
  --preserve-outside-strength 0.2 \
  --mid-steer-start-ratio 0.5 \
  --mid-steer-end-ratio 1.0 \
  --mid-steer-alpha-start 0.5 \
  --mid-steer-alpha-end 1.0 \
  --mid-steer-alpha-schedule linear \
  --seed 7 \
  --sae-ckpt /common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt \
  --sae-cfg /common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json \
  --prototype-npz /common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz \
  --prototype-key prototype_median \
  --pos-latent 2645 \
  --neg-latent 7036 \
  --device cuda:0
```

## Current Layout

- `scripts/export_hnscc_region_bank_10x.py`: build the sampled `10x` bank
- `scripts/run_region_bank_10x_sae_cases.py`: run the selected-cell SAE steering sweep
- `src/wsi_cf/`: shared helpers for slide reading, region-bank I/O, PixCell generation, and prototype loading
- `tests/`: focused unit and smoke tests for the kept workflow
