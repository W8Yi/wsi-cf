# Script Registry

This file is the central map of active repository entrypoints. Default feature inputs are 20x-equivalent UNI2 H5 bags under `/research/projects/mllab/WSI/TCGA_features/{project}/features_uni2`.

## Core Counterfactual Workflow

### `scripts/visualize_attention.py`
Visualizes MIL or CLAM attention on whole slides. Outputs attention overlays, heatmaps, top-attention tile CSVs, and per-slide summaries. No contact sheets are produced.

Example:
```bash
bash examples/hnscc_hpv/01_visualize_attention.sh
```

### `scripts/find_regions.py`
Finds manual or attention-guided regions for progressive editing. Outputs region images, feature grids, overlays, region metadata, edit manifests, and `region_bank.csv`.

Example:
```bash
bash examples/hnscc_hpv/02_find_regions.sh
```

### `scripts/run_progressive_region_edit.py`
Runs the canonical progressive region editor from a region image, region bank, or edit manifest. Outputs source/final images and run metadata by default, with debug artifacts when requested.
Supports reusable edit policies through `--edit-policy`, for example
`configs/edit_policies/showcase_best.json`; explicit CLI flags override policy values.

Example:
```bash
bash examples/hnscc_hpv/03_run_showcase_edit.sh
```

### `scripts/analyze_concept_uni_features.py`
Reconstructs task concept steering before diffusion and compares the decoded UNI conditioning features across concepts. Use this when generated images look similar and we want to verify whether SAE latent edits and decoded UNI grids are actually different.

By default, concept target activations are estimated from the top 5 representative tiles per concept (`--concept-target-top-k 5`). The representative CSV can still contain 50 tiles per concept for inspection.

Typical outputs:
- `concept_uni_feature_summary.csv`
- one `concept_uni_feature_grid.npy` per concept
- one `selected_sae_latents_before_after.npz` per concept
- `pairwise_selected_uni_delta_l2.csv/png`
- `pairwise_selected_uni_delta_cosine.csv/png`
- `concept_uni_feature_pca.png`
- `all_concepts_uni_delta_l2_heatmaps.png`

Example:
```bash
/common/users/wq50/envs/pace/bin/python scripts/analyze_concept_uni_features.py \
  --region-bank-csv artifacts/luad_to_lusc_20x_2048_regions10/region_bank.csv \
  --edit-manifest artifacts/luad_to_lusc_20x_2048_regions10/progressive_edit_manifest.json \
  --concepts-json artifacts/classifier_label_concepts_all_top10_top50/luad_lusc/LUSC/selected_concepts.json \
  --representative-tiles-csv artifacts/classifier_label_concepts_all_top10_top50/luad_lusc/LUSC/representative_tiles.csv \
  --concept-class-label LUSC \
  --out-dir artifacts/luad_to_lusc_concept_uni_feature_diagnostics \
  --device cuda:0
```

## Concept Discovery

### `scripts/prepare_classifier_concept_associations.py`
Builds `latent_label_associations.csv` from a trained classifier bundle. This is the bridge from a newly trained task classifier to concept discovery: it uses that classifier's `task_manifest.csv`, scans only the slides for that task, computes slide-level SAE summaries, and writes association artifacts in the same format consumed by `find_label_concepts.py`.

Typical outputs:
- `latent_label_associations.csv`
- `cohort_slides.csv`
- `slide_sae_summary.npz`
- `skipped_slides.csv`
- `task_summary.json`

Example:
```bash
/common/users/wq50/envs/pace/bin/python scripts/prepare_classifier_concept_associations.py \
  --task-name luad_lusc \
  --classifier-run-dir artifacts/classifier_training/luad_lusc \
  --out-root artifacts/concept_label_associations_classifier \
  --device cuda:0
```

### `scripts/find_label_concepts.py`
Selects label-relevant SAE concepts and representative tiles. It supports `labels_only` ranking from `latent_label_associations.csv` and `attention_aware` ranking when a classifier is available.

Typical outputs:
- `concept_cards.csv`
- `representative_tiles.csv`
- `selected_concepts.json`
- `summary.json`
- `concept_export/` portable local-PC package with `manifest.json`, compact CSVs,
  `slide_path_map.template.csv`, and `FORMAT.md`

Example:
```bash
/common/users/wq50/envs/pace/bin/python scripts/find_label_concepts.py \
  --task luad_lusc \
  --association-root artifacts/concept_label_associations_classifier \
  --class-label LUAD \
  --mode attention_aware \
  --backend mil \
  --classifier-run-dir artifacts/classifier_training/luad_lusc \
  --top-concepts 10 \
  --top-tiles-per-concept 50 \
  --out-dir artifacts/classifier_label_concepts \
  --device cuda:0
```

End-to-end classifier concept discovery:
```bash
bash examples/concept_discovery/01_run_all_classifier_concepts.sh
```

This example runs association preparation and attention-aware concept-card generation for every trained classifier task by default:
- `kirc_grade`
- `kirc_low_vs_high_grade`
- `msi_coad_stad`
- `luad_lusc`
- `cancer_type_all_tcga`

Useful overrides:
```bash
TASKS=luad_lusc,msi_coad_stad \
MAX_SLIDES_PER_CLASS=100 \
MAX_TILES_PER_SLIDE=4096 \
bash examples/concept_discovery/01_run_all_classifier_concepts.sh
```

Tumor purity low/high concepts:
```bash
bash examples/tumor_purity/00_find_low_high_purity_concepts_labels_only.sh
```

This builds a slide-level `purity_group` label source from
`resources/labels/targets/tumor_purity_case.tsv`, prepares SAE label associations
directly from low/high labels, and writes labels-only concept cards for `low`
and `high` purity. This path does not train or use a classifier. By default,
`low` and `high` are the bottom and top quartiles of case-level `tumor_purity`;
override with explicit thresholds:

```bash
LOW_THRESHOLD=0.40 HIGH_THRESHOLD=0.80 \
bash examples/tumor_purity/00_find_low_high_purity_concepts_labels_only.sh
```

The classifier-attention version is also available when attention-aware concept
ranking is needed:

```bash
bash examples/tumor_purity/01_find_low_high_purity_concepts.sh
```

After labels-only concepts exist, generate ten diverse paired examples in both
directions:

```bash
bash examples/tumor_purity/02_generate_10_low_high_pairs_labels_only.sh
```

### `scripts/generate_representative_tiles.py`
Legacy-compatible CSV generator for task-associated representative tiles. Use `find_label_concepts.py` for new concept-card workflows.

### `scripts/generate_cohort_sae_top_tiles.py`
Ranks top tiles per SAE latent for whole cohorts without requiring label associations. Useful for broad concept inspection, such as LUAD/LUSC cohort representative tiles.

## Classifier Training

### `scripts/train_attention_classifier.py`
Trains repo-native attention MIL classifiers from TCGA UNI2 feature bags. It reads labels from `resources/labels/master/slide_labels_master.tsv`, uses the patient-level SAE 90/10 train/test split, and writes checkpoints compatible with downstream MIL attention utilities.

Typical outputs:
- `final_model.pt`
- `best_model.pt`
- `args.json`
- `task_manifest.csv`
- `train_metrics.csv`
- `test_predictions.csv`
- `label_mapping.json`
- `summary.json`

Task examples:
```bash
bash examples/classifier_training/01_train_kirc_grade.sh
bash examples/classifier_training/02_train_msi_coad_stad.sh
bash examples/classifier_training/03_train_luad_lusc.sh
bash examples/classifier_training/04_train_cancer_type_all_tcga.sh
bash examples/classifier_training/05_train_kirc_low_vs_high_grade.sh
```

Implemented classifier task presets:

| Task preset | Example bash | Label definition | Notes |
| --- | --- | --- | --- |
| KIRC grade | `01_train_kirc_grade.sh` | `TCGA-KIRC`, `tumor_grade`, `G1/G2/G3/G4` | Four-way grade task; useful but hard and class-imbalanced. |
| MSI COAD/STAD | `02_train_msi_coad_stad.sh` | `TCGA-COAD,TCGA-STAD`, `msi_status`, `MSI/NonMSI` | Binary molecular-status task. |
| LUAD vs LUSC | `03_train_luad_lusc.sh` | `project_dir`, `LUAD/LUSC` | Binary lung cancer-type task. |
| All TCGA cancer type | `04_train_cancer_type_all_tcga.sh` | `project_dir`, all projects with at least 30 slides | Multiclass pan-cancer classifier. |
| KIRC low vs high grade | `05_train_kirc_low_vs_high_grade.sh` | `G1,G2 -> low`; `G3,G4 -> high` | Recommended KIRC grade classifier for more stable evaluation. |

All classifier presets are thin wrappers around `scripts/train_attention_classifier.py`, so they can accept extra overrides. For example:

```bash
bash examples/classifier_training/05_train_kirc_low_vs_high_grade.sh \
  --out-dir artifacts/classifier_training_smoke \
  --epochs 1 \
  --max-train-slides 8 \
  --max-test-slides 4 \
  --max-tiles-per-slide 512
```

Dry-run manifest check:
```bash
/common/users/wq50/envs/pace/bin/python scripts/train_attention_classifier.py \
  --task-name luad_lusc \
  --projects TCGA-LUAD,TCGA-LUSC \
  --label-column project_dir \
  --label-map TCGA-LUAD:LUAD,TCGA-LUSC:LUSC \
  --include-labels LUAD,LUSC \
  --out-dir artifacts/classifier_training_dryrun \
  --dry-run
```

Tumor-vs-normal is intentionally not included yet because the current master label table and TCGA feature store do not contain true normal-slide feature bags. For now, use `04_train_cancer_type_all_tcga.sh` as the broad multi-cohort classifier.

## HPV Examples

The core HNSCC HPV examples live in `examples/hnscc_hpv/`:
- `01_visualize_attention.sh`
- `02_find_regions.sh`
- `03_run_showcase_edit.sh`
- `04_run_region_eval.sh`
