# Region Bank Test Protocol

## Goal

Prepare the first fixed `1024x1024` HNSCC region bank for PixCell-1024 counterfactual steering, with each sample containing:

- a real tissue image crop
- an aligned `4x4` UNI2 feature grid
- a visual cell preview
- enough metadata to reproduce the sample exactly

This bank is the first-step artifact for the counterfactual experiments.

## Bank Composition

The initial bank must contain:

- `20` HPV-negative regions
- `20` HPV-positive regions
- `40` total regions
- `1` region per slide in v1

Sampling source:

- all labeled local slides that exist on disk
- balanced random tissue-rich sampling
- not attention-guided in this first step

## Saved Artifact Format

Root:

- `wsi_cf/artifacts/hnscc_region_bank_1024/`

Per-region bundle layout:

- `label_0_hpv_neg/<region_id>/`
- `label_1_hpv_pos/<region_id>/`

Each bundle must contain:

- `region.png`
- `region_zgrid.npy`
- `region_cells.png`
- `region_meta.json`

Root-level files:

- `region_bank.csv`
- `region_roles.csv`
- `region_bank_summary.json`
- `label_0_contact_sheet.png`
- `label_1_contact_sheet.png`

## Role Split

Use deterministic assignment after sorting by:

1. `label`
2. `slide_key`
3. `region_id`

Role partition:

- first `10` per label = `source`
- next `10` per label = `donor`

This yields:

- `10` HPV-negative sources
- `10` HPV-positive sources
- `10` HPV-negative donors
- `10` HPV-positive donors

## First Experiment Conditions

For each source region, run:

- `baseline`: no steering
- `same_label_one_cell`: replace cell `(1,1)` with one donor cell from a same-label donor region
- `cross_label_one_cell`: replace cell `(1,1)` with one donor cell from an opposite-label donor region
- `same_label_full_grid`: replace the full `4x4` grid with a same-label donor region
- `cross_label_full_grid`: replace the full `4x4` grid with an opposite-label donor region

Pairing rule:

- assign donors deterministically by role rank within the donor subsets
- keep source/donor pairing fixed across runs

## Output Naming Conventions

For exported bank samples:

- `region_id = <slide_key>__x_<region_x>__y_<region_y>`

For downstream generation results:

- `baseline/<region_id>/`
- `same_label_one_cell/<region_id>/`
- `cross_label_one_cell/<region_id>/`
- `same_label_full_grid/<region_id>/`
- `cross_label_full_grid/<region_id>/`

Suggested generation metadata should record:

- source `region_id`
- donor `region_id`
- condition
- replaced cell index if applicable
- PixCell model id
- seed

## SAE Prototype Run

The current focused runner is the sampled-bank `10x` SAE case sweep. It keeps the original source region and steers selected source cells toward an HPV+ or HPV- prototype instead of copying donor cells.

Selected-cells local multi-tile test:

- use repeated `--steer-cell gx,gy` to define any custom set of edited cells
- examples:
  - neighboring two: `--steer-cell 1,1 --steer-cell 2,1`
  - neighboring three: `--steer-cell 1,1 --steer-cell 2,1 --steer-cell 2,2`
  - `2x2`: `--steer-cell 1,1 --steer-cell 2,1 --steer-cell 1,2 --steer-cell 2,2`
  - `2x3`: `--steer-cell 1,0 --steer-cell 2,0 --steer-cell 1,1 --steer-cell 2,1 --steer-cell 1,2 --steer-cell 2,2`
- use conditions:
  - `to_hpv_pos_selected_cells`
  - `to_hpv_neg_selected_cells`

First-step 10x random region sampling:

- use `scripts/export_hnscc_region_bank_10x.py`
- sample balanced random `10x` `1024x1024` regions first, before steering
- save for each sampled region:
  - `region.png`
  - `region_zgrid.npy`
  - `region_cells.png`
  - `region_meta.json`
- this step prepares the input pool for later local SAE steering tests

Bank-based 10x SAE case sweep:

- use `scripts/run_region_bank_10x_sae_cases.py`
- consume the saved `region_bank.csv` from the 10x sampling step
- run named selected-cell cases directly on the saved aligned `4x4` grids
- supported cases:
  - `random_two`
  - `neighbor_two`
  - `neighbor_three`
  - `block_2x2`
  - `block_2x3`
  - `manual`
- save outputs by case and by source region for side-by-side review
- optional preservation mode:
  - `--preserve-outside-latents`
  - keeps non-edited regions anchored to the original source image latents
  - use this when you want the selected edited cells to move while the rest of the region changes as little as possible

Mid-diffusion locality test:

- keep `baseline` as the unchanged reference
- run `to_hpv_pos_one_cell` or `to_hpv_neg_one_cell` with `--mid-steer-start-ratio 0.5 --mid-steer-end-ratio 1.0`
- compare against the same condition with `--mid-steer-start-ratio 0.0 --mid-steer-end-ratio 1.0`
- then sweep later starts such as `0.75` and `0.85`
- goal: preserve more global structure while still moving the target cell toward the prototype

Scheduled-strength test:

- keep `--mid-steer-start-ratio 0.0 --mid-steer-end-ratio 1.0`
- set `--mid-steer-alpha-start 0.2 --mid-steer-alpha-end 1.0`
- compare `linear` vs `cosine` with `--mid-steer-alpha-schedule`
- goal: start with weak conditioning and gradually increase the edited conditioning influence over diffusion steps

Runner command:

```bash
python /common/users/wq50/wsi_cf/scripts/run_region_bank_10x_sae_cases.py \
  --region-bank-csv /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_1024_sample4/region_bank.csv \
  --out-dir /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_1024_sample4_sae_cases_preserve3_s0.2_mid0.5_rerun \
  --cases baseline,random_two,neighbor_three,block_2x2 \
  --direction hpv_pos \
  --max-sources 4 \
  --grid-step-px 256 \
  --dtype fp16 \
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
  --device cuda:0
```

Each `by_source/<source_region_id>/` folder from the SAE run should contain:

- `source_region.png`
- one generated image per SAE case
- `comparison_contact_sheet.png`

Use the repo-local prototype bundle by default:

- `/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz`
- `/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.json`

Current pinned concept latents:

- HPV+ prototype latent: `2645`
- HPV- prototype latent: `7036`

## Current Steering Mechanism

The current SAE steering pipeline for local region experiments is:

1. Start from a real sampled source region.
   - `source_region_actual.png` is the real tissue crop from the slide.

2. Encode that region into a UNI feature grid.
   - for `1024x1024` with `grid_step_px=256`, this is a `4x4x1536` grid
   - file: `region_zgrid.npy`

3. Choose the cells to edit.
   - one-cell, selected-cells, or full-grid
   - selected cells define a binary `tile_mask` over the `4x4` grid

4. Edit only the selected UNI cells in SAE latent space.
   - implementation: `/common/users/wq50/SAE_path/utils/sae_edit.py`
   - function: `edit_uni_z_grid_with_sae(...)`

Exact prototype edit mode used by the current `wsi_cf` runners:

- flatten the source UNI grid from `[Gh,Gw,D]` to `[N,D]`
- encode each UNI feature vector into SAE latent space:
  - `z_lat = sae_encode_features(sae_model, x)`
- for selected tiles only, interpolate the full SAE latent code toward the chosen prototype vector:
  - `z_edit[sel] = (1 - s) * z_lat[sel] + s * prototype`
  - where:
    - `sel` = selected tile indices from `tile_mask`
    - `s` = `target_latent_vector_strength` = `--prototype-strength`
    - `prototype` = full SAE latent vector loaded from `prototype_vectors_for_selected_sae.npz`
- decode the edited SAE latents back to UNI feature space:
  - `x_rec = sae_decode_latents(sae_model, z_edit)`
- blend decoded edited UNI features back into the original UNI grid:
  - if `keep_non_selected=True`, non-selected cells stay unchanged
  - selected cells use:
    - `x_new = (1 - b) * x + b * x_rec`
    - where `b = --steer-blend`

So the current runners do **full-code prototype interpolation** in SAE latent space, not single-neuron clamping.

Important consequences:

- `--prototype-strength` controls how strongly the selected SAE code moves toward the prototype
- `--steer-blend` controls how strongly the decoded edited UNI feature replaces the original UNI feature
- with `--prototype-strength 1.0 --steer-blend 1.0`, selected cells are fully moved to the prototype code and then fully replaced in UNI feature space

## Current Preservation Mechanism

Optional preservation mode:

- `--preserve-outside-latents`
- `--preserve-outside-strength`

This mechanism does **not** preserve via UNI features. It preserves in VAE image latent space during diffusion:

1. Encode the real source image into VAE latents.
2. Build a spatial rectangular mask from the selected grid cells.
3. During diffusion, outside the selected mask, pull the image latents back toward the noised trajectory of the original source image latents.

So:

- selected cells change because their UNI conditioning features were edited
- non-selected regions are anchored toward the original image latents

Current limitation:

- the preserve mask is currently a hard grid-aligned rectangle
- this can create visible rectangular boundaries when preservation is strong

## Actual vs Generated Source Images

The comparison folders now save both:

- `source_region_actual.png`
  - the real sampled tissue region
- `source_region_generated.png`
  - the baseline diffusion regeneration of that same region

Compatibility alias:

- `source_region.png`
  - currently points to the generated baseline version for browsing compatibility

The contact sheet should include:

- `source_actual`
- `source_generated`
- edited cases

## Magnification Caveat

Current default prototype bundle:

- `/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz`

This bundle was built from the default SAE:

- SAE config: `/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json`
- magnification: `20x`

Current `10x` local-region runners:

- `export_hnscc_region_bank_10x.py`
- `run_region_bank_10x_sae_cases.py`

These encode actual `10x` image crops into UNI features.

Therefore, the current default `10x` SAE steering setup has a representation mismatch:

- prototype / SAE: `20x`
- source steering input: actual `10x`

This is a real caveat and should be stated explicitly in experiments and writing.

Available closer alternative:

- `/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_10x_pool2x2/batch_topk_ckpt_best.pt`
- `/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_10x_pool2x2/run_config.json`

That run uses:

- magnification: `10x_pool2x2`

This is closer to the current `10x` workflow than the default `20x` SAE, but it is still not identical to actual optical `10x`.

## 10x Concept Bank Anchored On Current 20x Representative Tiles

For the next `10x` concept test, we keep the currently validated `20x` representative tiles as anchors and recrop around them at `10x`.

Workflow:

1. Read the repo-local prototype bundle JSON:
   - `/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.json`
2. Recover the original representative tile coordinates from:
   - `/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/sae_neuron_pipeline_batch_topk/split_0/top_neuron_tiles.csv`
3. Use each representative `20x` tile as the center anchor for a real `10x` `1024x1024` crop.
4. Re-encode that real `10x` region into an aligned `4x4` UNI feature grid.
5. Save the result as a reusable concept-centered `10x` bank that can feed the existing `10x` SAE runners.

Each exported bundle contains:

- `region.png`
- `region_zgrid.npy`
- `region_cells.png`
- `center_tile_20x.png`
- `region_with_center_tile.png`
- `region_meta.json`

Important interpretation:

- the original `20x` representative tile supplies the anchor location only
- the saved `10x` `region_zgrid.npy` is newly computed from the real `10x` crop
- this is the bridge experiment between current `20x` concept discovery and new `10x` steering

## QC Checklist

Before treating the bank as ready:

- no blank or background-dominant regions
- no severe artifact-only regions
- visible tissue present in every `region.png`
- every `region_zgrid.npy` has shape `[4,4,1536]`
- label balance is exactly `20` / `20`
- role balance is exactly `10` sources and `10` donors per label
- every path in `region_bank.csv` exists
- every `region_meta.json` contains label, slide key, paths, coords, grid step, feature dim, tissue score, and seed

## Acceptance Criteria

The bank is ready when:

- all `40` region bundles exist
- both label contact sheets exist
- `region_bank.csv`, `region_roles.csv`, and `region_bank_summary.json` exist and load successfully
- QC passes without missing files or invalid tensor shapes
- the bank can be used directly for one-cell and full-`4x4` PixCell-1024 steering
