# WSI Counterfactual Steering

`wsi_cf` is the focused home for pathology-grounded whole-slide-image counterfactual steering in this repo.

The initial scope is HNSCC HPV experiments built around:

- PixCell generation with `PixCell-256` and `PixCell-1024`
- UNI2-H tile features and steering manifests
- donor pool construction from real slides
- one-tile, naive `4x4`, and real contiguous `4x4` donor-grid steering
- MIL-based evaluation on TCGA HNSC HPV

This is intentionally not the new home for all legacy SAE or MIL training code. Stage 1 keeps the counterfactual core here and leaves broad training/mining utilities in the legacy repo layout.

## What Belongs Here

- Counterfactual generation and steering utilities
- donor feature export and donor-pool tooling
- PixCell MultiDiffusion sampling utilities
- HNSCC HPV evaluation entrypoints
- focused tests, docs, and configs for the counterfactual pipeline

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

## Supported Experiment Types

- one-tile replacement with `--steer_tile gx,gy,feature.npy`
- naive `4x4` donor-grid replacement from 16 donor features
- real contiguous `4x4` donor-grid replacement from one donor region
- SAE prototype steering of the original source tile or full source `4x4` grid
- SAE prototype steering of any custom set of selected cells in the source `4x4` grid
- one-off 10x local-region SAE tests from fresh slide crops
- MIL-based HNSCC HPV counterfactual evaluation

## Quickstarts

These examples assume you are running from `/common/users/wq50/SAE_path` with the `pace` environment active.

### 1. Export One Donor Feature

```bash
python wsi_cf/scripts/export_h5_tile_feature.py \
  --h5 /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2/TCGA-BB-4225-01Z-00-DX1.h5 \
  --tile-index 472 \
  --out /common/users/wq50/wsi_cf/artifacts/example_tile_472.npy
```

### 2. Build A Donor Pool Or A Real `4x4` Donor Grid

Balanced donor pool:

```bash
python wsi_cf/scripts/export_hnscc_tile_feature_pool.py \
  --slides-dir /common/users/wq50/HNSCC/HNSCC_slides \
  --features-dir /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2 \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --total-tiles 50 \
  --tiles-per-slide 1 \
  --out-dir /common/users/wq50/wsi_cf/artifacts/hnscc_tile_feature_pool
```

Real contiguous donor grid:

```bash
python wsi_cf/scripts/export_real_grid_steer_manifest.py \
  --slide /common/users/wq50/HNSCC/HNSCC_slides/TCGA-HD-A633-01Z-00-DX1.0E062B5C-BCD1-42FF-A48F-4EC231D705BF.svs \
  --h5 /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2/TCGA-HD-A633-01Z-00-DX1.h5 \
  --tile-index 472 \
  --grid-side 4 \
  --out-dir /common/users/wq50/wsi_cf/artifacts/realgrid_hd_a633_tile472
```

Random `10x` region bank with aligned features:

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

Concept-centered `10x` bank from current `20x` representative tiles:

```bash
python wsi_cf/scripts/export_10x_concept_regions_from_20x_representatives.py \
  --prototype-json /common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.json \
  --tiles-csv /common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/sae_neuron_pipeline_batch_topk/split_0/top_neuron_tiles.csv \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --slides-dir /common/users/wq50/HNSCC/HNSCC_slides \
  --features-dir /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2 \
  --out-dir /common/users/wq50/wsi_cf/artifacts/hnscc_10x_concept_regions_from_20x_representatives \
  --selected-direction all \
  --examples-per-latent 1 \
  --target-magnification 10 \
  --region-size 1024 \
  --grid-step-px 256 \
  --device cuda:0
```

### 3. Run PixCell Steering And Evaluation

One-tile PixCell-1024 steering:

```bash
python wsi_cf/scripts/multidiff_img2img_wsi.py \
  --input_svs /common/users/wq50/HNSCC/HNSCC_slides/TCGA-BB-4225-01Z-00-DX1.cfce62af-e565-4673-9970-afd08767b062_001.svs \
  --x 11264 --y 0 \
  --region_w 1024 --region_h 1024 \
  --grid_step_px 256 \
  --pix_model_id StonyBrook-CVLab/PixCell-1024 \
  --steer_tile 1,1,/common/users/wq50/wsi_cf/artifacts/example_tile_472.npy \
  --steer_blend 1.0 \
  --save_real \
  --out_dir /common/users/wq50/wsi_cf/artifacts/one_tile_example \
  --device cuda:0
```

HNSCC HPV evaluation:

```bash
python wsi_cf/scripts/counterfactual_tile_steering_eval.py \
  --split-json /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.json \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --out-dir /common/users/wq50/wsi_cf/artifacts/counterfactual_eval
```

SAE prototype region-bank steering:

```bash
python wsi_cf/scripts/run_region_bank_sae_experiments.py \
  --region-roles-csv /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_1024/region_roles.csv \
  --out-dir /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_1024_sae_runs \
  --prototype-npz /common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz \
  --pos-latent 2645 \
  --neg-latent 7036 \
  --prototype-key prototype_median \
  --prototype-strength 0.8 \
  --pix_model_id StonyBrook-CVLab/PixCell-1024 \
  --device cuda:0
```

The current repo-local prototype bundle was copied from the validated old run and organized here:

- `artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz`
- `artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.json`

The default concept latents are:

- HPV+ prototype latent: `2645`
- HPV- prototype latent: `7036`

One-off 10x selected-cells SAE test:

```bash
python wsi_cf/scripts/run_sae_10x_selected_cells_test.py \
  --input-svs /common/users/wq50/HNSCC/HNSCC_slides/TCGA-BB-4225-01Z-00-DX1.cfce62af-e565-4673-9970-afd08767b062_001.svs \
  --out-dir /common/users/wq50/wsi_cf/artifacts/tenx_selected_cells_example \
  --target-magnification 10 \
  --selection-mode block \
  --anchor-gx 1 --anchor-gy 1 \
  --block-w 2 --block-h 2 \
  --direction hpv_pos \
  --pix_model_id StonyBrook-CVLab/PixCell-1024 \
  --device cuda:0
```

10x sampled-bank SAE case sweep:

```bash
python wsi_cf/scripts/run_region_bank_10x_sae_cases.py \
  --region-bank-csv /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_1024/region_bank.csv \
  --out-dir /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_1024_sae_cases \
  --cases random_two,neighbor_three,block_2x2,block_2x3 \
  --direction hpv_pos \
  --preserve-outside-latents \
  --preserve-outside-strength 0.95 \
  --seed 7 \
  --device cuda:0
```

Attention-guided local `1024` steering:

```bash
python wsi_cf/scripts/run_attention_guided_local_1024.py \
  --split-json /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.json \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --features-root /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2 \
  --slides-dir /common/users/wq50/HNSCC/HNSCC_slides \
  --direction hpv_pos \
  --target-magnification 10 \
  --attention-percentile 90 \
  --max-high-attention-cells 8 \
  --out-dir /common/users/wq50/wsi_cf/artifacts/attention_guided_local_1024 \
  --device cuda:0
```

Attention-guided suitable `10x` region bank:

```bash
python wsi_cf/scripts/export_attention_guided_region_bank_10x.py \
  --split-json /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.json \
  --split-tsv /common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv \
  --features-root /research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2 \
  --slides-dir /common/users/wq50/HNSCC/HNSCC_slides \
  --target-magnification 10 \
  --attention-percentile 90 \
  --min-high-attention-cells 1 \
  --max-high-attention-cells 10 \
  --out-dir /common/users/wq50/wsi_cf/artifacts/attention_guided_region_bank_10x \
  --device cuda:0
```

## Migration Status

Stage 1 is implemented here:

- package layout
- docs and config template
- focused data, steering, generation, and evaluation helpers
- thin CLI entrypoints
- focused tests

Stage 2 will continue moving vetted logic here and leave compatibility wrappers in the legacy locations until imports are fully updated.
