# Experiment Task Plan

This document prepares the test pipeline for the paper experiments. It is organized around the current framing in `PAPER.md`: a multi-scale, concept-grounded counterfactual framework that supports tile-level, region-level, and slide-level analysis.

The goal is to turn the current HNSCC pipeline into a repeatable evaluation suite, then reuse the same structure for additional pathology tasks.

---

## Core Questions

The experiments should answer four questions:

1. Does concept-guided steering move generated tissue toward the intended target class or concept?
2. Does region-level or slide-level steering produce more coherent edits than isolated tile editing?
3. Are changes localized to selected tissue while preserving the rest of the region or slide?
4. Can the same SAE concept bank and generation pipeline be reused across downstream tasks without retraining the concept bank or generator?

---

## Shared Pipeline

All tasks should follow the same high-level pipeline.

1. Train or load a downstream classifier.
   - HNSCC: HPV MIL or CLAM classifier
   - other tasks: task-specific MIL/CLAM classifier

2. Score the source slide or region.
   - prediction probability
   - attention or importance map
   - optional SAE concept relevance map

3. Select edit targets.
   - tile-level: one or more 256 tiles
   - region-level: selected cells inside a 1024 or 2048 region
   - slide-level: multiple attention-selected regions across the WSI

4. Apply counterfactual steering.
   - baseline: donor feature replacement
   - main method: SAE prototype steering
   - optional ablations: random concepts, wrong-direction concepts, no preservation

5. Generate edited image output.
   - PixCell-1024 for region-level generation
   - progressive editor for larger or slide-level composed edits

6. Re-encode generated tissue.
   - UNI feature grid for edited region
   - replace edited features into the original bag for classifier evaluation

7. Evaluate.
   - visual quality
   - locality
   - concept movement
   - classifier shift
   - pathologist review when available

---

## HNSCC HPV Primary Experiment

This is the main paper task and should be the most complete.

### Inputs

- HNSCC slides from the curated CLAM or local HNSCC slide set
- HPV labels
- CLAM or MIL classifier checkpoint
- UNI/UNI2 feature bags and coordinates
- SAE checkpoint and HPV prototype bundle
- PixCell-1024 generation stack

### Region Selection

Use three selection modes:

- `manual_showcase`: manually chosen strong tissue region for the main figure
- `attention_guided`: top attention or attention-mass regions from the classifier
- `pathology_aware`: attention plus tissue/cell density plus SAE current-label similarity

Recommended inclusion criteria:

- source classifier prediction matches true label
- source confidence is high enough for the task
- selected region contains sufficient tissue and cells
- selected edit cells are spatially coherent
- high-attention region is not entirely background, stroma-only, or artifact-heavy

### Main Direction

Use both directions, but choose one hero showcase:

- primary showcase: HPV+ to HPV-
- reverse validation: HPV- to HPV+

Rationale:

- HPV+ to HPV- often starts from stronger visible phenotype and can make a clearer figure
- bidirectional testing guards against a one-direction artifact

### Required Runs

Run the following conditions on the same selected regions:

- `baseline_regeneration`: no feature edit
- `donor_replacement`: opposite-label donor feature replacement
- `sae_steer`: SAE prototype steering toward target label
- `sae_steer_no_preservation`: ablation
- `sae_steer_random_targets`: attention-selection ablation
- `sae_steer_wrong_direction`: concept-direction control

Run at these scales:

- tile-level: one or a few selected 256 cells
- region-level: 1024 and 2048 regions
- slide-level: multiple selected regions, re-encoded and written back into WSI feature bag

### HNSCC Metrics

Classifier metrics:

- source probability
- edited probability
- target-class probability shift
- class flip rate
- dose-response across steering strength
- attention redistribution after edited-feature replacement

Feature metrics:

- cosine movement toward target prototype
- cosine movement away from source prototype
- selected-cell feature delta
- non-selected-cell feature delta
- edited vs unedited feature-change ratio

Image/locality metrics:

- pixel or perceptual change inside selected cells
- pixel or perceptual change outside selected cells
- inside/outside change ratio
- boundary artifact score near edited-cell borders
- stain/color shift outside edited regions
- preservation of non-edited tissue structure

Human/pathology metrics:

- plausibility score
- label-consistent morphology score
- artifact score
- whether edited area remains diagnostically recognizable tissue

---

## Slide-Level HNSCC Experiment

This experiment tests the claim that the method can produce a true slide-level counterfactual when needed.

### Pipeline

1. Run classifier and attention over the full WSI bag.
2. Select high-attention regions using attention mass and tissue quality.
3. Edit selected regions using the canonical progressive editor.
4. Re-encode only edited generated regions into UNI features.
5. Replace the corresponding original WSI bag entries.
6. Rerun the slide classifier on the edited bag.
7. Save local before/after region visualizations and a slide-level edited-region map.

### Required Outputs

- selected high-attention tile list
- region plan CSV
- original local regions
- edited local regions
- edited-region overlay on slide thumbnail
- replacement manifest
- before/after WSI feature bag prediction
- summary JSON and CSV

### Slide-Level Metrics

- slide-level target probability shift
- number of edited tiles and regions
- edited attention mass fraction
- probability shift per edited tile
- probability shift per edited attention mass
- feature replacement coverage
- fraction of slide left unchanged
- visual plausibility of edited local regions

---

## Cross-Task Validation

Use the same framework but swap the classifier, labels, concept-class association, and target concepts.

### Concept-Label Association Preparation

Before running generation for a new task, prepare task-specific SAE concept associations. This step reuses:

- labels from `resources/labels/targets`
- slide-to-feature mappings from `resources/labels/master/slide_labels_master.tsv`
- trained SAE checkpoints from `resources/models/sae`
- UNI2 feature bags from `/research/projects/mllab/WSI/TCGA_features/<PROJECT>/features_uni2`

The reusable preparer is:

```bash
/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/prepare_concept_label_associations.py
```

Fast coverage check for the main paper tasks:

```bash
/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/prepare_concept_label_associations.py \
  --tasks paper \
  --dry-run \
  --out-dir /common/users/wq50/wsi_cf/artifacts/concept_label_associations_dryrun_paper
```

Full association run for the main paper tasks:

```bash
/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/prepare_concept_label_associations.py \
  --tasks paper \
  --out-dir /common/users/wq50/wsi_cf/artifacts/concept_label_associations_paper \
  --sae-ckpt /common/users/wq50/wsi_cf/resources/models/sae/relu_sae_base/relu_final.pt \
  --sae-cfg /common/users/wq50/wsi_cf/resources/models/sae/relu_sae_base/run_config.json \
  --device cuda:0
```

Run a specific task:

```bash
/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/prepare_concept_label_associations.py \
  --tasks msi_coad_stad \
  --out-dir /common/users/wq50/wsi_cf/artifacts/concept_label_associations_msi \
  --device cuda:0
```

Available built-in task presets:

- `hnsc_hpv`: HNSCC HPV status
- `cesc_hpv`: CESC HPV status using viral-threshold labels
- `msi_coad_stad`: COAD/STAD MSI status
- `kirc_grade`: KIRC tumor grade
- `immune_subtype`: pan-cancer immune subtype
- `brca_pam50`: BRCA PAM50 subtype
- `tumor_purity`: continuous pan-cancer tumor purity
- `tp53_mutation`: pan-cancer TP53 mutation status
- `kras_mutation`: pan-cancer KRAS mutation status
- `paper`: `hnsc_hpv`, `cesc_hpv`, `msi_coad_stad`, and `kirc_grade`
- `all`: all presets above

Outputs per task:

- `cohort_slides.csv`: slides with labels and available feature files
- `coverage_summary.json`: label counts and feature coverage
- `slide_sae_summary.csv`: per-slide SAE summary metadata
- `case_summary.csv`: per-case label and slide aggregation
- `case_latent_summaries.npz`: case-level latent summary arrays
- `latent_label_associations.csv`: ranked latent-label associations
- `top_cases_for_top_latents.csv`: cases most activating top associated latents
- `task_summary.json`: task-level metadata

Association metrics:

- categorical labels use one-vs-rest latent enrichment
- continuous labels use Pearson correlation
- each task computes three latent summaries:
  - `fraction`: latent activation mass normalized within each slide/case
  - `mean_activation`: average positive activation
  - `prevalence`: fraction of tiles where the latent is active

Use the top associated latents from `latent_label_associations.csv` as candidate concept directions for steering and for prototype-tile review.

### LUAD Growth Pattern

Goal:

- test whether local tissue structure can move between histologic growth patterns, such as lepidic and solid.

Recommended setup:

- classifier: LUAD subtype or growth-pattern classifier
- region selection: attention-guided tumor-rich regions
- steering: concepts associated with source and target growth patterns

Key metrics:

- target subtype probability shift
- preservation of tumor identity
- structural coherence
- pathologist rating for growth-pattern plausibility

Risk:

- this task likely needs thoracic pathologist review.

### COAD/STAD MSI

Goal:

- test whether subtle immune-associated morphology can be introduced or reduced.

Recommended setup:

- classifier: MSI vs MSS
- region selection: attention-guided tumor-immune interface or lymphocyte-rich regions
- steering: MSI-associated concepts

Key metrics:

- MSI probability shift
- immune-infiltration visual score
- lymphocyte/tumor-region preservation
- artifact score

Risk:

- visual signal may be subtle and classifier-dependent.

### KIRC Grade

Goal:

- test gradual changes in nuclear morphology and architecture associated with tumor grade.

Recommended setup:

- classifier: grade low vs high, or ordinal grade model
- region selection: tumor-rich, cellular regions
- steering: grade-associated concepts

Key metrics:

- grade probability or ordinal score shift
- monotonic dose-response with steering strength
- nuclear morphology plausibility
- tissue preservation outside selected regions

Risk:

- grading criteria should be reviewed by a renal pathologist.

### CESC Cross-Cancer HPV

Goal:

- test whether HPV-associated concepts learned in HNSCC transfer to cervical cancer.

Recommended setup:

- classifier: CESC HPV or related phenotype if available
- steering: HNSCC-derived HPV concepts applied to CESC regions
- comparison: recomputed CESC concept-class association if labels are available

Key metrics:

- target probability shift
- concept transfer consistency
- morphology plausibility in CESC tissue
- failure modes where HNSCC concepts do not transfer

### Tumor vs Normal Baseline

Goal:

- provide an easier baseline where large structural changes should be detectable.

Recommended setup:

- classifier: tumor vs normal across one or more TCGA cohorts
- region selection: high-confidence tumor or normal regions
- steering: tumor-associated or normal-associated concepts

Key metrics:

- prediction shift
- class flip rate
- visual coherence
- preservation of non-target tissue context

---

## Metric Summary Table

Use this table to decide which metrics belong in the main paper versus supplement.

| Metric family | Metric | Main use |
| --- | --- | --- |
| Prediction | target probability shift | primary quantitative outcome |
| Prediction | class flip rate | easy-to-read summary |
| Prediction | dose-response slope | shows steering strength matters |
| Feature | target prototype cosine increase | confirms concept movement |
| Feature | source prototype cosine decrease | confirms directionality |
| Feature | selected/non-selected delta ratio | confirms locality |
| Image | inside/outside image-change ratio | confirms spatial constraint |
| Image | boundary artifact score | tests region integration |
| Image | stain/color drift outside target | tests preservation |
| Slide | edited attention mass fraction | explains intervention size |
| Slide | probability shift per edited tile | normalizes slide-level effect |
| Human | pathology plausibility score | validates morphology |
| Human | artifact score | screens visual failures |

---

## Minimum Experiment Matrix

### Main HNSCC Matrix

Run at least:

- 20 HPV+ source regions steered to HPV-
- 20 HPV- source regions steered to HPV+
- 3 steering strengths: `0.4`, `0.8`, `1.2`
- 3 scales: tile, region, slide
- 3 method conditions: donor replacement, SAE steering, SAE steering with preservation
- controls: random targets and wrong-direction steering

### Cross-Task Pilot Matrix

For each additional task:

- 10 source examples per label or class direction
- 2 steering strengths: `0.8`, `1.2`
- region-level first
- slide-level only after region-level sanity check passes
- one qualitative figure plus quantitative prediction shift summary

---

## Acceptance Criteria

Before claiming a task works, require:

- selected source examples are correctly classified before editing
- edited images remain plausible tissue
- target probability shifts in the intended direction
- edits are stronger inside selected regions than outside
- SAE steering outperforms or is more interpretable than donor replacement
- random-target and wrong-direction controls show weaker or opposite effects
- metadata is sufficient to reproduce every run

For slide-level claims, additionally require:

- edited feature bag can be rerun through the classifier
- replacement manifest maps edited regions back to valid slide coordinates
- slide-level probability shift is reported separately from local-region score shift
- visualization clearly marks which slide regions were edited

---

## Immediate Preparation Tasks

1. Freeze the HNSCC showcase.
   - choose one HPV+ to HPV- hero region
   - save source, attention overlay, selected cells, generated output, and slide box overview

2. Build the HNSCC quantitative cohort.
   - select balanced HPV+ and HPV- regions
   - run donor replacement and SAE steering
   - sweep strength and preservation settings

3. Finalize slide-level HNSCC run.
   - choose one or more slides
   - edit high-attention regions
   - re-encode edited tiles only
   - rerun classifier with edited bag

4. Prepare cross-task pilots.
   - identify available classifier/checkpoint/feature roots for LUAD, MSI, KIRC, CESC, and tumor-normal
   - verify label manifests
   - run region-level sanity checks before slide-level tests

5. Create reporting artifacts.
   - `candidate_scan.csv`
   - `selected_regions.csv`
   - `run_results.csv`
   - `summary_by_task.json`
   - `summary_by_strength.json`
   - paper-ready before/after panels

---

## Reporting Rules

Keep three claims separate:

- region-level image counterfactual: generated image changes in a selected region
- slide-level feature counterfactual: edited features are inserted into the WSI bag and the classifier is rerun
- slide-level visualization: edited local regions are composed back into a slide view for interpretation

This separation keeps the paper strong without overstating one-shot whole-slide image synthesis.
