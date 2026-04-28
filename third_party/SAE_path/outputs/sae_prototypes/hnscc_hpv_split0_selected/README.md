# SAE Prototype Bundle

This folder contains the repo-local SAE concept prototype bundle used by `wsi_cf`.

Files:

- `prototype_vectors_for_selected_sae.npz`
- `prototype_vectors_for_selected_sae.json`

Source:

- copied from `/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/notebook_counterfactual_single_tile_and_area/`

Bundle summary:

- disease/task: HNSCC HPV
- SAE checkpoint: `/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt`
- SAE config: `/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json`
- latent count: `40`
- directions: `20` HPV+, `20` HPV-
- prototype shape: `(40, 12288)` for both `prototype_mean` and `prototype_median`

Pinned concept latents used by the current `wsi_cf` defaults:

- HPV+ prototype latent: `2645`
- HPV- prototype latent: `7036`

These files are the default prototype source for:

- `scripts/run_region_bank_sae_experiments.py`
- `scripts/counterfactual_tile_steering_eval.py`
