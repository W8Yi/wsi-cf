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
For paper-style balanced sampling, use `--final-slides-per-label N` with
`--final-regions-per-slide K`; set `--final-slides-per-label 0` with
`--final-regions-per-slide K` to export K regions for every eligible slide. When
both are unset, the older `--final-regions-per-label` selection is preserved.

Current paper region-selection default for HNSCC HPV is available through
`--region-selection-mode borderline`. It ranks tissue-passing 2048 regions by a
score combining moderate whole-slide attention, local region classifier
target-probability around 0.55, tissue score, and valid feature density, then
uses per-slide non-maximum suppression before exporting the top regions.

Current paper cell-selection default:
`configs/edit_cell_selection/attention_percentile_smooth.json`. This selects
target cells by the local attention percentile calibrated from the successful
showcase top-23 seed selection, then applies one 4-neighbor smoothing pass. In
the 8x8 showcase region, top-23 corresponds to percentile `64.0625`. The number
of edited cells in future regions is determined by the attention distribution,
not by a fixed top-N or target count.
`find_regions.py` defaults to this through
`--edit-cell-selection-mode attention_percentile_smooth`; use
`--edit-cell-selection-mode showcase_smoothed28` only for exact reproduction of
the historical fixed top-23/prune-28 showcase artifact, or
`--edit-cell-selection-mode importance_mass_sae_neighbors` to recover the older
attention+SAE mass selector. Pair the default selector with
`configs/edit_policies/showcase_best.json`.

Example:
```bash
bash examples/hnscc_hpv/02_find_regions.sh
```

### `scripts/run_progressive_region_edit.py`
Runs the canonical progressive region editor from a region image, region bank, or edit manifest. Outputs source/final images and run metadata by default, with debug artifacts when requested.
Supports reusable edit policies through `--edit-policy`, for example
`configs/edit_policies/showcase_best.json`; explicit CLI flags override policy values.

Default HNSCC steering example:
```bash
PYTHON=/common/users/wq50/envs/pace/bin/python \
DEVICE=cuda:3 \
examples/hnscc_hpv_showcase_smoothed28/01_run_progressive_edit.sh
```

This default example uses the fixed smoothed-28 showcase manifest,
`configs/edit_policies/showcase_best.json`, `--direction hpv_neg`, and the
legacy `relu_sae_base` paths matching the current HNSCC prototype bundle.
Because the policy enforces `center_2x2` edit support, the historical 28-cell
request executes on 14 coverable cells over 5 windows and records the 14
unsupported border cells in `run_manifest.json`.
`examples/hnscc_hpv/03_run_showcase_edit.sh` delegates to the same command.

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

### `scripts/run_hnscc_hpv_policy_benchmark.py`
Runs or reuses matched HNSCC HPV progressive-edit outputs for multiple edit
policies and writes comparison metrics. The default policy set compares the
current `showcase_best` policy against `naive_no_preserve` and
`baseline_no_preserve_full_duration`. With `--bidirectional`, it splits the
input manifest into HPV+->HPV- and HPV-->HPV+ direction manifests and runs each
policy in both directions.

Typical outputs:
- `benchmark_predictions.csv`
- `benchmark_metrics_by_run.csv`
- `benchmark_summary_by_method.csv`
- `benchmark_per_cell_metrics.csv`
- `benchmark_concept_fidelity.csv`
- `benchmark_window_consistency.csv`
- `benchmark_summary.json`

Example:
```bash
/common/users/wq50/envs/pace/bin/python scripts/run_hnscc_hpv_policy_benchmark.py \
  --run-edits \
  --skip-existing \
  --device cuda:3 \
  --generation-device cuda:3
```

Paper showcase wrapper:
```bash
PYTHON=/common/users/wq50/envs/pace/bin/python \
DEVICE=cuda:3 \
examples/hnscc_hpv_showcase_smoothed28/04_run_paper_policy_benchmark.sh
```

Full bidirectional HNSCC HPV paper benchmark wrapper:
```bash
PYTHON=/common/users/wq50/envs/pace/bin/python \
DEVICE=cuda:3 \
examples/hnscc_hpv/06_run_paper_benchmark.sh
```

Dry-run the full wrapper without launching generation:
```bash
DRY_RUN=1 examples/hnscc_hpv/06_run_paper_benchmark.sh
```

## Concept Discovery

### `scripts/prepare_classifier_concept_associations.py`
Legacy helper for building `latent_label_associations.csv` from a trained classifier bundle. New concept discovery runs should prefer the JSON-driven `find_label_concepts.py` workflow below, which builds the cohort, associations, and concept cards in one command.

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
Builds label-relevant SAE concepts from a task JSON. The JSON defines the cohort, label column/map, requested concept labels, SAE variant, and optional classifier attention. The script builds the cohort from `resources/labels/master/slide_labels_master.tsv`, computes slide-level SAE associations, then writes per-label concept cards and representative tiles.

Typical outputs:
- `cohort_slides.csv`
- `latent_label_associations.csv`
- `slide_sae_summary.npz`
- `labels/<label>/concept_cards.csv`
- `labels/<label>/representative_tiles.csv`
- `labels/<label>/selected_concepts.json`
- `labels/<label>/summary.json`
- `labels/<label>/concept_export/` portable local-PC package with `manifest.json`, compact CSVs,
  `slide_path_map.template.csv`, and `FORMAT.md`

Example:
```bash
/common/users/wq50/envs/pace/bin/python scripts/find_label_concepts.py \
  --task-json configs/concept_tasks/luad_lusc.json \
  --out-root artifacts/concept_discovery_json \
  --device cuda:0
```

Curated JSON task examples live in `configs/concept_tasks/`:
- `luad_lusc.json`
- `kirc_low_vs_high_grade.json`
- `tumor_purity_low_high.json`
- `hnsc_hpv.json`
- `prad_gleason_score.json`
- `prad_low_vs_high_grade.json`

End-to-end classifier concept discovery:
```bash
bash examples/concept_discovery/01_run_all_classifier_concepts.sh
```

This example runs the curated JSON tasks by default:
- `luad_lusc`
- `kirc_low_vs_high_grade`
- `tumor_purity_low_high`
- `hnsc_hpv`

Useful overrides:
```bash
TASK_JSONS=configs/concept_tasks/luad_lusc.json,configs/concept_tasks/hnsc_hpv.json \
CONCEPT_OUT=artifacts/concept_discovery_json_batch_topk \
bash examples/concept_discovery/01_run_all_classifier_concepts.sh
```

Prepare TCGA-PRAD Gleason labels from GDC and mine both score-specific and
low/high-grade concepts from existing UNI2 bags:
```bash
DEVICE=cuda:0 bash examples/concept_discovery/05_run_prad_gleason_concepts.sh
```

The score task uses `GS6`, `GS7`, `GS8`, `GS9`, and `GS10`. The steering
task maps ISUP Grade Groups `GG1-GG2` to `low` and `GG3-GG5` to `high`.

Train the matching PRAD low/high attention MIL classifier:
```bash
DEVICE=cuda:0 bash examples/classifier_training/12_train_prad_low_vs_high_grade.sh
```

After this classifier exists, `prad_low_vs_high_grade.json` uses it for
attention-aware concept ranking.

Prepare normal-versus-tumor TCGA tasks from the GDC slide inventory:
```bash
bash examples/concept_discovery/02_prepare_normal_tumor_tasks.sh
```

This writes normal SVS download manifests and concept task JSON files for
LUAD, COAD, BRCA, and KIRC. The steering labels are `normal` and `tumor`;
for normal-to-tumor steering, `tumor` is the target concept label. Normal
UNI2 feature bags must be generated from the downloaded SVS files before
`find_label_concepts.py` can mine those concepts.

Download a starter set of 20 GDC normal SVS files into this repository:
```bash
TASK=kirc_normal_tumor \
bash examples/concept_discovery/03_download_normal_tumor_normal_slides.sh
```

Set `TASK` to `luad_normal_tumor`, `coad_normal_tumor`,
`brca_normal_tumor`, or `kirc_normal_tumor`. Set `ALL=1` to download the
complete normal inventory for the task; this can require tens to over one
hundred GB depending on cohort.

Extract UNI2 normal-slide feature bags after the SVS downloads are complete:
```bash
DEVICE=cuda:0 \
bash examples/concept_discovery/04_extract_normal_tumor_uni2_features.sh
```

The extractor is resumable: it processes only downloaded normal slides that
do not already have an H5 bag under `artifacts/normal_tumor_features/`.
Set `TASKS="kirc_normal_tumor"` to encode a single cohort.

### `scripts/overlay_sae_concepts_on_wsi.py`
Recomputes SAE latent activations on supplied or randomly sampled TCGA WSIs and
overlays only feature-backed tiles on WSI thumbnails. By default, every latent
in the selected SAE is eligible. The combined panel colors each tile by its
strongest SAE concept, while
`tile_winner_concepts.csv` records the winning concept, latent, raw activation,
normalized activation, and thumbnail bounds for every feature tile. By default
it samples 10 matched slide/H5 pairs from `/research/projects/mllab/WSI/TCGA/store`
and `/research/projects/mllab/WSI/TCGA_features`.

Example:
```bash
PYTHON=/common/users/wq50/envs/pace/bin/python
$PYTHON scripts/overlay_sae_concepts_on_wsi.py \
  --sample-random-slides 10 \
  --out-dir paper_example/sae_concept_wsi_overlay
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
| KIRC continuous grade risk | `06_train_kirc_continuous_grade_risk.sh` | `G1=0.00`, `G2=0.33`, `G3=0.66`, `G4=1.00` | Continuous endpoint for measuring risk-score movement after steering. |
| KIRC ordinal grade risk | `07_train_kirc_ordinal_grade_risk.sh` | Cumulative `>=G2`, `>=G3`, `>=G4` targets | Ordered endpoint with grade-balanced loss and pairwise ranking loss. |
| KIRC SAE ordinal grade risk | `08_train_kirc_sae_ordinal_grade_risk.sh` | Batch-TopK SAE concept aggregates -> cumulative grade risk | Concept-only linear ordinal endpoint with interpretable coefficients. |
| KIRC SAE case-level ordinal risk | `09_train_kirc_sae_case_continuous_grade_risk.sh` | Patient-averaged Batch-TopK SAE concept aggregates -> continuous ordinal risk | Uses stable concept selection and an auxiliary low/high boundary loss for sparse G1 robustness. |
| LUAD/COAD/BRCA/KIRC normal vs tumor | `10_train_all_normal_tumor.sh` | `normal` vs `tumor` per organ | Requires downloaded normal slides and extracted UNI2 bags from the normal/tumor preparation workflow. |

Classifier presets `01` through `05` are thin wrappers around `scripts/train_attention_classifier.py`, so they can accept extra overrides. For example:

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

After normal SVS extraction, train all four organ-specific normal-versus-tumor classifiers:
```bash
DEVICE=cuda:0 \
bash examples/classifier_training/10_train_all_normal_tumor.sh
```

The wrapper checks that all normal H5 bags in each task are available before
training, so a partially extracted normal cohort is not used accidentally.

Run preparation, downloaded-slide validation, resumable UNI2 extraction, and
all four classifier trainings in one command:
```bash
DEVICE=cuda:0 \
bash examples/classifier_training/11_encode_and_train_all_normal_tumor.sh
```

Set `SLIDE_ROOT=/path/to/downloads` if the normal SVS files were downloaded
outside `artifacts/normal_tumor_slides/`.
Use `CHECK_ONLY=1` to validate that all expected downloaded slides are visible
before starting UNI2 extraction.

LUAD-to-LUSC top-1 steering cell-budget classifier test:
```bash
DEVICE=cuda:0 \
bash examples/morphology_label_concept_review/03_luad_to_lusc_cell_budget_classifier_test.sh
```

This reuses the ten LUAD source regions from
`artifacts/morphology_label_concept_review_top1_showcase_best_10slides`,
regenerates LUSC-directed edits for `1`, `4`, `16`, `32`, and `64` cells, and
reports whole-slide `P(LUSC)` shifts using `artifacts/classifier_training/luad_lusc`.
Cells are ranked by classifier attention within those compatible with the
`showcase_best` center-support policy for the first four budgets. The `64`-cell
(`100%`) condition edits the entire 8x8 region with `border_relaxed` support;
classifier scoring reports how many of those cells are represented in the
source UNI2 feature bag.

### `scripts/train_grade_risk_regressor.py`
Trains a bounded slide-level MIL regression score for ordinal KIRC grade. It
holds a validation subset out of the original training cases for checkpoint
selection and evaluates the original patient-level test split only after
selecting the best checkpoint.

```bash
DEVICE=cuda:0 bash examples/classifier_training/06_train_kirc_continuous_grade_risk.sh
```

Ordinal model for sharper grade ordering:
```bash
DEVICE=cuda:0 bash examples/classifier_training/07_train_kirc_ordinal_grade_risk.sh
```

The ordinal preset predicts three cumulative grade thresholds and defines its
risk score as their mean, yielding the same `0.00`, `0.33`, `0.66`, `1.00`
scale while directly supervising the grade ordering. It enables grade-balanced
loss and a pairwise ranking loss by default.

### `scripts/train_sae_concept_grade_risk.py`
Trains a concept-only linear ordinal predictor using Batch-TopK SAE summaries
rather than raw UNI2 tile embeddings. For each slide it caches mean activation,
active-tile fraction, and top-5%-tile mean activation, selects the top 100
latents on training slides only, and writes interpretable model coefficients.

```bash
DEVICE=cuda:0 bash examples/classifier_training/08_train_kirc_sae_ordinal_grade_risk.sh
```

When the SAE feature cache has already been computed with the same settings:
```bash
DEVICE=cuda:0 REUSE_FEATURE_CACHE=1 \
bash examples/classifier_training/08_train_kirc_sae_ordinal_grade_risk.sh
```

Case-level continuous endpoint for the sparse-G1 KIRC setting:
```bash
DEVICE=cuda:0 bash examples/classifier_training/09_train_kirc_sae_case_continuous_grade_risk.sh
```

This preset reuses the `08` Batch-TopK feature cache, averages slide-level
SAE summaries within patients, retains 30 concepts using five-fold
training-only stability selection, and adds an auxiliary `G1/G2` versus
`G3/G4` boundary loss while preserving the continuous ordinal risk score.

### `scripts/evaluate_kirc_grade_risk_edits.py`
Re-encodes generated KIRC regions with UNI2, replaces their corresponding cells
in each original whole-slide feature bag, and measures continuous grade-risk
movement. The primary endpoint is `full_region_risk_delta`; positive movement
supports low-to-high edits, while negative movement is the reverse-direction
control. A `target_cells_risk_delta` is also written as a focused diagnostic.

```bash
DEVICE=cuda:0 bash examples/kirc_grade/06_evaluate_top1_grade_risk_shift.sh
```

### `scripts/evaluate_kirc_sae_grade_risk_edits.py`
Scores generated KIRC edits with the concept-only ordinal predictor by
recomputing SAE slide summaries after feature replacement. Existing generated
edits are marked as binary-classifier-selected retrospective evaluations.

```bash
DEVICE=cuda:0 bash examples/kirc_grade/07_evaluate_sae_ordinal_grade_risk_shift.sh
```

## HPV Examples

The core HNSCC HPV examples live in `examples/hnscc_hpv/`:
- `01_visualize_attention.sh`
- `02_find_regions.sh`
- `03_run_showcase_edit.sh`
- `04_run_region_eval.sh`
