# Migration Plan For `wsi_cf`

This document tracks what should move into the focused counterfactual package and what should stay in the legacy repo for now.

## Stage 1

- establish the package layout under `wsi_cf/`
- add docs, config template, tests, and thin CLI entrypoints
- migrate reusable counterfactual helpers into `src/wsi_cf`
- keep legacy scripts usable during transition

## Stage 2

- move vetted counterfactual logic fully onto local `wsi_cf` modules
- leave thin wrappers in legacy locations where needed
- shrink remaining direct dependencies on legacy helpers

## Transfer Table

| Source Path | Target | Action | Reason | Dependencies / Blockers |
| --- | --- | --- | --- | --- |
| `scripts/multidiff_img2img_wsi.py` | `src/wsi_cf/generation/pixcell.py` + `scripts/multidiff_img2img_wsi.py` | Refactor | Core PixCell + MultiDiffusion generation belongs in the CF package | Heavy runtime deps: `diffusers`, `timm`, OpenSlide |
| `utils/diffusion.py` | `src/wsi_cf/generation/pixcell.py` | Refactor | Shared PixCell window config, token lookup, and latent blending helpers | Keep APIs aligned with existing experiments |
| `scripts/export_h5_tile_feature.py` | `src/wsi_cf/data/h5.py` + `scripts/export_h5_tile_feature.py` | Move / Refactor | Single-feature export is a core donor-prep workflow | None |
| `scripts/export_hnscc_tile_feature_pool.py` | `src/wsi_cf/data/donor_pool.py` + `scripts/export_hnscc_tile_feature_pool.py` | Move / Refactor | Donor-pool creation is a first-class CF workflow | OpenSlide required for real tile export |
| `scripts/export_real_grid_steer_manifest.py` | `src/wsi_cf/steering/manifest.py` + `scripts/export_real_grid_steer_manifest.py` | Move / Refactor | Real contiguous `4x4` donor grids are central to PixCell-1024 steering | OpenSlide required for preview tile export |
| `scripts/build_pixcell_1024_naive_manifest.py` | `src/wsi_cf/steering/manifest.py` + `scripts/build_pixcell_1024_naive_manifest.py` | Move / Refactor | Naive donor-grid manifests are part of the baseline experiment ladder | None |
| `concept_steer/counterfactual_tile_steering_eval.py` | `src/wsi_cf/eval/hnsc_hpv.py` + `scripts/counterfactual_tile_steering_eval.py` | Refactor | HNSCC HPV evaluation belongs with the CF package | Transitional dependency on legacy SAE helpers |
| `concept_steer/run_hnsc_hpv_sae_neuron_pipeline.py` | `src/wsi_cf/eval/hnsc_hpv.py` | Wrap / Reuse selectively | Needed only for minimal MIL attention helper behavior | Keep only the minimal helper surface in v1 |
| `models/classifier.py` | `src/wsi_cf/eval/hnsc_hpv.py` | Reuse selectively | Minimal attention MIL helpers are needed for evaluation | Full training code stays in legacy repo |
| `scripts/train_sae.py` | legacy repo | Leave | Generic SAE training is out of v1 scope | Not a CF-core workflow |
| `scripts/train_hnsc_hpv_mil_5fold.py` | legacy repo | Leave | Generic MIL training is out of v1 scope | Not a CF-core workflow |
| `notebook/` | legacy repo | Leave | Notebooks are not part of the focused package | Cleanup can happen later |
| `outputs/`, `runs/`, `wandb/` | legacy repo | Leave | Historical artifacts should not move into the package | Large and unrelated to package code |
| broad mining / visualization scripts | legacy repo | Leave For Later | Useful, but not part of the first focused CF core | Revisit after Stage 2 stabilizes |

## Keep In `wsi_cf` Now

- counterfactual generation
- manifest tooling
- donor-pool tooling
- HNSCC HPV counterfactual evaluation
- focused tests, docs, and config templates

## Leave For Later

- SAE training and generic evaluation
- generic feature extraction pipelines
- unrelated visualization and mining utilities
- old notebooks and large historical artifacts

## Compatibility Strategy

- preserve `steer_manifest.json` as a list of `{ "gx": int, "gy": int, "path": str }`
- preserve donor-pool CSV fields for label, slide, tile index, coords, feature path, and image path
- preserve the conceptual single-tile interface `gx,gy,feature_path`
- keep PixCell-1024 behavior as `patch_px=1024`, `cond_grid_side=4`
- allow Stage 1 code to depend on selected legacy helpers where a full migration would be excessive
