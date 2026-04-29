# WSI Counterfactual Steering

`wsi_cf` is organized around one smooth counterfactual workflow:

1. visualize attention on whole-slide feature bags,
2. find useful `20x`-equivalent regions,
3. run progressive region editing with SAE concept steering,
4. optionally evaluate edited regions/features downstream.

The default task is HNSCC HPV. The default progressive-edit demo starts from:

`artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048/region_top_right_2048.png`

## Canonical Scripts

- `scripts/visualize_attention.py`: whole-slide attention heatmaps, CLAM-first.
- `scripts/find_regions.py`: region discovery by attention/pathology-aware scoring.
- `scripts/run_progressive_region_edit.py`: image-first or region-bank progressive editing.

Core HPV examples live in `examples/hnscc_hpv/`:

```bash
bash examples/hnscc_hpv/01_visualize_attention.sh
bash examples/hnscc_hpv/02_find_regions.sh
bash examples/hnscc_hpv/03_run_showcase_edit.sh
bash examples/hnscc_hpv/04_run_region_eval.sh
```

## Project Layout

- `src/wsi_cf/models/`: SAE, MIL, and CLAM model definitions maintained inside this repo.
- `src/wsi_cf/steering/`: SAE runtime/editing plus progressive edit planning.
- `src/wsi_cf/generation/`: PixCell/UNI helpers and diffusion windowing.
- `src/wsi_cf/data/`: slide, H5, region-bank, and proposal utilities.
- `src/wsi_cf/eval/`: task-specific classifier/evaluation helpers.
- `resources/`: versioned labels, manifests, model weights, prototypes, and task configs.
- `examples/hnscc_hpv/`: minimal runnable HPV workflow scripts.
- `docs/`: paper, method, testing, and migration notes.

Large raw slides and precomputed feature stores stay external. The repo stores only compact resources needed to reproduce the workflow configuration.

## Default Resources

- SAE checkpoint: `resources/models/sae/relu_sae_base/relu_final.pt`
- SAE config: `resources/models/sae/relu_sae_base/run_config.json`
- HNSCC HPV MIL checkpoint: `resources/models/classifiers/hnscc_hpv/mil_split0.pt`
- HNSCC HPV CLAM checkpoint: `resources/models/classifiers/hnscc_hpv/clam_split0.pt`
- HNSCC HPV prototypes: `resources/prototypes/hnscc_hpv/prototype_vectors_for_selected_sae.npz`
- Task registry: `resources/tasks/hnscc_hpv.json`

## Quick Demo

```bash
python scripts/run_progressive_region_edit.py \
  --task hnscc_hpv \
  --direction hpv_neg \
  --output-mode debug \
  --out-dir artifacts/hnscc_hpv_showcase_progressive_edit
```

This uses the default showcase image and the repo-provided default edit manifest, so no checkpoint/prototype paths are needed on the command line.
