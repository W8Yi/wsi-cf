# Steering Task Selection Thread Notes

Date: 2026-06-11

## Scope

This note records the project orientation and task-ranking result from this thread.
It is intentionally practical: what the current steering system can actually do,
which tasks fit it, and what should be inspected before committing to a new
paper-style steering task.

Visual review artifact:

- `visual_review/source_generated_contact_sheet.png`

## Project Understanding

The current repo implements local WSI counterfactual steering, not arbitrary
pixel-level editing. The canonical pipeline:

1. selects a real tissue region,
2. loads an aligned UNI feature grid,
3. edits selected grid cells in SAE latent space,
4. decodes back to UNI conditioning,
5. runs PixCell-1024 over progressive overlapping windows,
6. re-encodes the generated region and evaluates classifier movement.

Important constraints:

- A `1024x1024` PixCell window is a `4x4` UNI grid.
- The canonical progressive editor only edits local center `2x2` cells.
- A `2048x2048` region is typically an `8x8` grid, planned through overlapping
  `4x4` windows.
- The method is strongest when the target is a local morphology/content change
  and the surrounding tissue architecture can remain mostly stable.
- Large global transformations, exact gland boundary rewriting, and molecular
  labels without a visible histologic correlate are weaker fits.

## Task Ranking

### 1. LUAD <-> LUSC

Recommended as the top new steering task.

Why:

- Clinically meaningful and morphologic: glandular/lepidic/acinar adenocarcinoma
  versus squamous morphology.
- Same organ comparison, so the overall slide context is less confounded than
  pan-cancer labels.
- Classifier is very strong: LUAD/LUSC best-test balanced accuracy `0.989`,
  AUROC `0.9996`.
- Existing concept review ranks both LUAD and LUSC as `best`.

Current evidence:

- Concept signals are strong: LUAD top latent `701`, LUSC top latent `3103`.
- Cell-budget classifier test shows directionally increasing LUSC probability:
  4 cells gives mean delta `0.140`, 16 cells `0.393`, 32 cells `0.488`.
- Flip rate rises to `0.4` at 16 cells and `0.5` at 32/64 cells.

Visual read from existing generated examples:

- At small target budgets, edits are visible but often subtle: local density,
  texture, and stromal/tumor appearance shift more than large architecture.
- Some examples look nearly unchanged, so region selection matters.

Best next use:

- Use LUAD -> LUSC first with 16 and 32 cell budgets.
- Keep the 1 and 4 cell settings as locality controls, not primary success
  settings.
- Choose regions where the source has editable tumor texture but not a large
  structure that would need wholesale architectural rewriting.

### 2. HNSCC HPV Status

Keep as the calibration/default task, but do not rely on it as the only steering
claim.

Why:

- It is already implemented end to end.
- The showcase has a clear classifier trajectory after re-encoding:
  source HPV+ `0.998`, final progressive HPV- `0.955` in the quick metric run.
- Stage trajectory shows the flip happening progressively after enough windows:
  HPV- probability rises from `0.0018` to `0.9547`.

Caveat:

- HPV status is molecular and may not be as locally interpretable as LUAD/LUSC,
  grade, or tumor purity.
- The benchmark summary shows `ours` is more local than naive methods but has
  lower target flip rate than the strongest naive baselines: `0.46` for ours,
  `0.50` naive-no-preserve, `0.54` naive-full-duration.

Best next use:

- Keep HPV as a regression benchmark for progressive planning, preservation,
  and classifier-consistent re-encoding.
- Use it to show the current system works, but add a more visibly morphologic
  task for the main task-selection story.

### 3. Tumor Purity Low <-> High

Strong candidate for local content steering, but needs a classifier/evaluator
path before it can be a primary result.

Why:

- Very local and clinically meaningful: tumor-rich versus stromal/normal/immune
  admixture is exactly the kind of content change the current method can express.
- Existing concept review ranks both high and low purity as `best`.
- Concepts are mined labels-only over large cohorts, with strong selected-label
  summary scores.

Caveat:

- Current concept discovery is labels-only; there is no equally established
  downstream classifier score in the reviewed artifacts.
- Because it spans all TCGA projects, some concepts may reflect tissue/site
  differences unless region selection is controlled.

Best next use:

- Run within one cancer type first, or train/evaluate a purity classifier or
  regressor before making it a top result.
- Visually inspect high/low purity representative tiles after generating tile
  contact sheets from SVS paths.

### 4. KIRC Low <-> High Grade

Good clinical task, but currently weaker than LUAD/LUSC.

Why:

- Grade is clinically meaningful and morphologic.
- Same cancer type, so the task is cleaner than pan-cancer steering.
- Concept review marks high/low KIRC as `good`.

Current evidence:

- Binary classifier is moderate: best balanced accuracy `0.690`, AUROC `0.722`.
- Grade-risk evaluation on existing generated edits is weak:
  high->low succeeds directionally but with very small full-region mean delta
  `-0.00064`; low->high fails at full-region level and only target cells move
  weakly positive.

Visual read from existing generated examples:

- Changes are generally subtle and mostly texture/color/cellularity shifts.
- Existing examples do not yet show a robust high-grade transformation.

Best next use:

- Keep as second-wave after LUAD/LUSC.
- Improve region selection and concept choice first.
- Prefer target-cell risk delta as a diagnostic, but require full-region or
  slide-level movement before treating it as a success.

### 5. PRAD Low <-> High Grade

Promising but less proven in the current generated-artifact set.

Why:

- Clinically meaningful and morphologic.
- Classifier is stronger than KIRC low/high: best balanced accuracy `0.830`,
  AUROC `0.867`.

Caveat:

- I did not find an existing comparable generated-steering visual review set in
  the artifacts inspected here.
- Prostate grading can depend on gland architecture, which may be harder for
  coarse `4x4` cell steering if the required change is structural.

Best next use:

- Mine and visually inspect PRAD low/high concepts.
- Run a small 10-region pilot before promoting it above KIRC or tumor purity.

### 6. MSI COAD/STAD

Secondary candidate.

Why:

- Classifier is strong: best balanced accuracy `0.940`, AUROC `0.974`.
- MSI can have local histologic correlates such as lymphocyte-rich regions.

Caveat:

- MSI is molecular, so a classifier shift may not correspond to a clean local
  visual concept.
- Existing concept review marks MSI as `secondary`, not primary.

Best next use:

- Only use if representative concept tiles clearly show an interpretable local
  correlate such as lymphocyte-rich tumor microenvironment.

## Tasks To Avoid As Primary Steering Targets

- Pan-cancer labels: useful for concept validation but confound organ/site and
  tumor type.
- Normal versus tumor as the first task: clinically meaningful, but often
  requires changing large tissue architecture and may exceed the current local
  steering assumption unless constrained to local tumor-content insertion or
  removal.
- Arbitrary random SAE concepts: retained as diagnostics only; prior random
  concept probing did not produce reliable showcase-quality results.

## Visual Inspection Notes

I inspected the generated contact sheet at:

- `visual_review/source_generated_contact_sheet.png`

Observed:

- LUAD/LUSC examples show plausible local texture and density shifts, but many
  examples are subtle at low cell counts.
- KIRC grade examples are even subtler; not yet enough for a primary claim.
- The HPV showcase has a clearer local visual alteration and a strong classifier
  trajectory, but the clinical label is less intrinsically local.

Representative concept packages under `artifacts/morphology_label_concept_review`
are CSV/JSON exports, not image contact sheets. The export format expects local
SVS paths in `slide_path_map.template.csv`. In this environment, OpenSlide is
not installed and the searched TCGA store path did not expose the needed SVS
files, so full representative-tile visual inspection should be done after those
paths are available.

## Recommended Immediate Plan

1. Make LUAD -> LUSC the next main steering task.
2. Use 16-cell and 32-cell budgets as primary classifier-consistent settings.
3. Keep HPV as the method regression benchmark and paper-style existing
   showcase.
4. Build concept tile contact sheets for LUAD, LUSC, KIRC high/low, tumor purity
   high/low, and MSI before final task selection.
5. Promote tumor purity or PRAD only after a pilot shows both visual coherence
   and classifier/evaluator movement.

