# Controlled Steering-Strength Sweeps

This workflow generates the paper panel:

`0% -> 20% -> 40% -> 60% -> 80% -> 100%`

while holding the following variables fixed:

- one or more seeded random `2048 x 2048` regions
- the target-cell mask for each region
- the PixCell seed and generation settings
- the SAE, target prototype, classifier, and edit policy

Only `prototype_strength` changes. This is distinct from an attention-budget
sweep, which changes the number of edited cells.

The default edit policy is
`configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json`,
shared with the canonical progressive steering runner.

The `0%` panel is a matched PixCell regeneration with
`prototype_strength=0`, not the untouched source image. This controls for the
generation pipeline itself. Each run directory also saves
`source_region_actual.png`; show it as a separate "Source" inset if space
allows.

If an input manifest contains several masks per region (for example attention
budgets and random-cell controls), resolve it first with
`--request-selector attention --request-budget 32`. The runner rejects
ambiguous duplicate requests rather than silently choosing a mask.

## Main entrypoints

- `scripts/run_steering_strength_sweep.py`: select, generate, evaluate, and plot.
- `scripts/plot_steering_strength_sweep.py`: rebuild figures from the tidy metrics CSV.
- `scripts/run_cell_fraction_sweep.py`: hold strength fixed while sweeping a
  deterministic nested fraction of valid tissue cells; the 0% point is the
  untouched source and empty feature cells are never selected.
- `scripts/plot_combined_control_sweeps.py`: combine matched steering-strength
  and valid-tile-fraction sweeps into one six-panel paper figure.
- `src/wsi_cf/paper/strength_sweep.py`: reusable selection and monotonicity logic.

All examples use the project environment:

```bash
PYTHON=/common/users/wq50/envs/pace/bin/python
```

## Reproducibility contract

Every run writes:

- `selected_edit_manifest.json`: the exact sampled regions and cell masks.
- `sweep_provenance.json`: seed, strengths, commands, manifest SHA-256, git commit, and dirty-tree state.
- `generated/strength_000` through `generated/strength_100`.
- `metrics/strength_*`: re-encoded UNI grids and classifier evaluations.
- `steering_strength_metrics.csv`: one tidy row per region and strength.
- `steering_strength_monotonicity.csv`: adjacent decreases and Spearman rho.
- `figures/steering_strength.{png,pdf,svg}`.

The two main responses are:

1. target-class probability after replacing the selected generated cells in
   the local region or full slide bag;
2. a target concept score, measured by re-encoding the generated RGB region
   with UNI and then the same SAE.

For concept-card tasks, the concept score is the target latent activation. The
legacy HPV workflow steers toward a full-code prototype whose selector latent
is not itself active in the prototype; for HPV, the score is therefore cosine
alignment to the exact target prototype vector. The raw selector-latent
activation is still retained in the CSV for diagnostics.

The figure does not force a monotonic fit or smooth the measurements.
Monotonicity is a result: report reversals, effect size, and Spearman rho.

The default plotted probability is the selected-cell replacement score because
it isolates the intended intervention. Use `--probability-source full_region`
when the figure should instead report the classifier score for every
re-encoded cell in the displayed generated image. Both values remain in the
metrics CSV.

## Task recipes

HNSCC HPV:

```bash
DEVICE=cuda:0 N_REGIONS=5 \
  examples/paper_strength_sweeps/01_run_hnscc_hpv.sh
```

LUAD to LUSC:

```bash
DEVICE=cuda:0 N_REGIONS=5 \
  examples/paper_strength_sweeps/02_run_luad_to_lusc.sh
```

KIRC low to high grade:

```bash
DEVICE=cuda:0 N_REGIONS=5 \
  examples/paper_strength_sweeps/03_run_kirc_low_to_high.sh
```

Normal to tumor for LUAD, COAD, BRCA, or KIRC:

```bash
TASK=coad_normal_tumor DEVICE=cuda:0 N_REGIONS=5 \
  examples/paper_strength_sweeps/04_run_normal_to_tumor.sh
```

For the matched KIRC valid-tile fraction sweep at fixed 90% strength:

```bash
DEVICE=cuda:5 bash examples/paper_strength_sweeps/05_run_kirc_cell_fraction.sh
```

Run any recipe with `DRY_RUN=1` first to resolve and record the selected
regions and all six generation commands without launching PixCell.

Set `ALL_REGION_CELLS=1` in a recipe, or pass `--all-region-cells` to the
Python entrypoint, to replace the input attention mask with the complete
region grid. For the canonical `2048 x 2048` region and 256-pixel grid step,
this records and steers all 64 cells in every selected region.

For regions containing missing or blank feature cells, pass
`--all-valid-feature-cells` to derive the target mask from
`valid_feature_mask.npy`. Use `--center-valid-feature-cells` for a contiguous
central block of tissue-valid cells; `--center-margin-cells 1` selects the
central `6 x 6` block of an `8 x 8` grid before invalid cells are removed.

Policy 09 commits complete generated windows, so its context passes can tint
invalid background cells even though they are not steering targets. Preserve
the original background with a soft tissue boundary by adding:

```bash
--runner-extra "--preserve-invalid-feature-cells --invalid-feature-feather-px 64"
```

This keeps the PixCell generation and steering schedule unchanged, then
restores cells excluded by `valid_feature_mask.npy` from the source image.
The feather avoids a hard 256-pixel tile boundary at the tissue edge.

For a localized block intervention, also restore all non-target cells after
the full-window generation pass:

```bash
--runner-extra "--preserve-invalid-feature-cells --invalid-feature-feather-px 64 \
  --preserve-outside-target-cells --target-cell-feather-px 64"
```

This makes the requested target-cell block the only intervention area and
softens its outer boundary while retaining policy 09's internal generation.

For PRAD morphology or grade-group tasks, use the same runner after preparing
the task's `region_bank.csv` and `progressive_edit_manifest.json`. Pass the
matching classifier bundle, target-label concept card, and label order. The
generic CLI intentionally does not hard-code tumor types.

## Paper recommendation

A single region is useful as a qualitative ladder, but it is not sufficient
evidence for a causal-control claim. For the main paper:

- sample at least 20 held-out regions from multiple patients per direction;
- show one pre-registered representative image ladder;
- plot per-region trajectories as faint lines and the mean with 95% CI;
- report the fraction of fully monotonic regions, median Spearman rho, and
  source-to-100% probability/activation change;
- repeat both directions when biologically meaningful;
- keep region selection independent of sweep outcomes.

Use a held-out test-region bank for final numbers. Do not select the displayed
region because it happened to be the most monotonic.
