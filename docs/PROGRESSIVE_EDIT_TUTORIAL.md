# Progressive Editing Tutorial

This document explains how to use the **canonical progressive editing pipeline** in `wsi_cf`.

The pipeline is implemented by:

- [scripts/run_progressive_region_edit.py](/common/users/wq50/wsi_cf/scripts/run_progressive_region_edit.py)

This is now the main region-level editing runner for manifest-driven experiments.

The same progressive editing logic is also the default editing backend for:

- [scripts/run_region_attention_classifier_eval.py](/common/users/wq50/wsi_cf/scripts/run_region_attention_classifier_eval.py)

It is designed to:

- take a prepared region bank entry
- take a manifest of target grid cells to edit
- automatically plan the needed overlapping `1024x1024` PixCell windows
- keep track of which cells were already visited or edited
- preserve previously visited context more strongly than fresh outer context
- save simple outputs by default, with detailed step outputs only when requested

## 1. What This Pipeline Does

At a high level, the pipeline works like this:

1. Load one real source region from `region_bank.csv`.
2. Load its aligned UNI feature grid from `region_zgrid.npy`.
3. Read a manifest that tells the pipeline which **global grid cells** should be edited.
4. Automatically choose the sequence of overlapping `4x4` PixCell windows needed to cover those target cells.
5. For each window:
   - edit only the target cells inside that window in SAE latent space
   - build a history-aware preservation map
   - run PixCell generation on that local window
   - write the generated window back into the evolving canvas
   - mark edited and visited cells in the run state
6. Save the final edited region plus a run manifest describing exactly what happened.

The main idea is:

- **target cells** tell the pipeline what we want to change
- **window planning** decides where to run PixCell
- **preservation** decides what should stay stable during each local generation step

## 2. When To Use This Runner

Use this runner when:

- you already have a prepared region bank
- you want to edit one or more cells inside a region
- you want the pipeline to decide the window order automatically
- you want a reproducible editing run described by a manifest

Do **not** use this runner when:

- you want the old case-based test runner with `random_two`, `block_2x2`, or row-strip experiment flags
- you want to sample regions from slides

Those are handled elsewhere. This runner is the **editing engine** itself.

## 3. Inputs You Need

You need two main inputs:

### `region_bank.csv`

This comes from the region-bank export workflow. Each row describes one saved region bundle, including:

- `region_id`
- `image_path`
- `feature_grid_path`
- `grid_step_px`
- region metadata such as label and slide key

The progressive editor uses:

- the real source image from `image_path`
- the aligned UNI grid from `feature_grid_path`

### Edit Manifest JSON

This tells the pipeline **which cells to edit**.

The manifest must be a JSON list. Each item describes one run.

Minimum required fields:

- `region_id`
- `target_cells`

Optional fields:

- `run_id`
- any extra metadata such as `source_method`, `attention_score`, `group`, `note`

If `run_id` is omitted, the runner creates a deterministic one from the region id and target cells.

## 4. Edit Manifest Format

### Minimal Example

```json
[
  {
    "region_id": "TCGA-CV-7263-01Z-00-DX1__mag_10p0__x_3868__y_33550",
    "target_cells": [
      {"gx": 1, "gy": 1},
      {"gx": 2, "gy": 1},
      {"gx": 5, "gy": 4}
    ]
  }
]
```

### Example With Metadata

```json
[
  {
    "run_id": "cv7263_attention_topcells",
    "region_id": "TCGA-CV-7263-01Z-00-DX1__mag_10p0__x_3868__y_33550",
    "target_cells": [
      {"gx": 1, "gy": 1},
      {"gx": 2, "gy": 1},
      {"gx": 5, "gy": 4}
    ],
    "source_method": "attention",
    "group": "pilot_1",
    "note": "top-attention positive edit"
  }
]
```

### Accepted Cell Formats

Each item in `target_cells` may be written as:

- `{"gx": 1, "gy": 2}`
- `[1, 2]`
- `"1,2"`

All three mean the same thing.

### Important Meaning Of `target_cells`

These are **global region-grid cells**, not local window cells.

For example, in a `2048x2048` region with `grid_step_px=256`, the region grid is `8x8`.

So:

- `gx` ranges from `0` to `7`
- `gy` ranges from `0` to `7`

The pipeline will automatically convert those global targets into local `4x4` window edits.

Important constraint:

- the pipeline now enforces that **only the center `2x2` cells of each local `4x4` window are editable**
- this means a target cell is valid only if it can appear as local `(1,1)`, `(2,1)`, `(1,2)`, or `(2,2)` in at least one planned window

For an `8x8` grid with `4x4` windows and stride `2`, this means:

- cells near the extreme border, such as `gx=0` or `gx=7`, are not directly editable in the current canonical pipeline
- the editable x and y coordinates are typically the interior coordinates covered by the center support of some window

## 5. How Automatic Planning Works

The runner does **not** need row-strip flags or manual window order.

It automatically:

1. builds all valid overlapping `4x4` windows over the region grid
2. finds which windows contain the requested target cells **inside their center `2x2` edit support**
3. starts from the top-most, then left-most reachable target window
4. continues with a deterministic nearest-window policy
5. stops as soon as all requested target cells are covered

This means:

- disconnected targets are allowed
- multiple windows may be used
- no extra windows are run after coverage is complete
- targets outside the center `2x2` support of every possible window are rejected

The planned windows are saved into `run_manifest.json` under `window_history`.

## 6. How Preservation Works In This Runner

This runner uses **history-aware preservation**.

Inside each active `1024x1024` window, the pipeline builds a spatial preserve-strength map with three regions:

- `edit_core`
  The selected target cells being edited in the current step.
  In the canonical pipeline, these are always a subset of the local center `2x2`.
- `visited_context`
  Cells inside the current window that were already visited by previous windows.
- `fresh_outer_context`
  Non-edit cells inside the current window that have not been visited yet.

The preservation strengths are:

- `preserve_edit_strength`
- `preserve_visited_strength`
- `preserve_fresh_context_strength`

The intended relationship is:

- `visited_context > fresh_outer_context`

This means:

- previously generated overlap is held more stable
- fresh context is allowed to adapt more freely
- edit cells can be fully free or partially preserved depending on your chosen setting

### Default Interpretation

Current defaults are:

- `preserve_edit_strength = 0.05`
- `preserve_visited_strength = 0.95`
- `preserve_fresh_context_strength = 0.35`
- `mid_steer_start_ratio = 0.5`
- `mid_steer_end_ratio = 1.0`
- `mid_steer_alpha_start = 0.5`
- `mid_steer_alpha_end = 1.0`
- `mid_steer_alpha_schedule = linear`

So by default:

- target edit cells are still mostly free to change, but no longer completely unconstrained
- previously visited context is strongly anchored
- untouched context can still move enough to make the transition smoother
- the edited conditioning starts later and ramps from moderate to strong influence

## 7. Minimal Example Command

```bash
/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/run_progressive_region_edit.py \
  --region-bank-csv /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_2048_sample4/region_bank.csv \
  --edit-manifest /common/users/wq50/wsi_cf/artifacts/edit_manifests/example_targets.json \
  --out-dir /common/users/wq50/wsi_cf/artifacts/progressive_edit_example \
  --direction hpv_pos \
  --device cuda:0
```

This uses:

- default SAE checkpoint and config
- default prototype bundle
- default PixCell-1024 and SD3 VAE
- default preservation settings
- default delayed mid-steer schedule
- default debug output mode

## 8. Debug Example Command

```bash
/common/users/wq50/envs/pace/bin/python /common/users/wq50/wsi_cf/scripts/run_progressive_region_edit.py \
  --region-bank-csv /common/users/wq50/wsi_cf/artifacts/hnscc_region_bank_10x_2048_sample4/region_bank.csv \
  --edit-manifest /common/users/wq50/wsi_cf/artifacts/edit_manifests/example_targets.json \
  --out-dir /common/users/wq50/wsi_cf/artifacts/progressive_edit_debug \
  --direction hpv_pos \
  --prototype-strength 0.8 \
  --steer-blend 1.0 \
  --preserve-edit-strength 0.0 \
  --preserve-visited-strength 0.98 \
  --preserve-fresh-context-strength 0.30 \
  --mid-steer-start-ratio 0.5 \
  --mid-steer-end-ratio 1.0 \
  --mid-steer-alpha-start 1.0 \
  --mid-steer-alpha-end 1.0 \
  --output-mode debug \
  --device cuda:0
```

This writes per-step debug outputs, including:

- local source window
- selected-cell overlay
- preserve-map preview
- steered local window

## 9. Output Structure

### Default `output_mode=minimal`

For each run, the runner saves:

- `source_region_actual.png`
- `generated.png`
- `run_manifest.json`

At the root output directory, it also saves:

- `experiment_args.json`
- `run_summary.json`

### `output_mode=debug`

In addition to the minimal outputs, the runner also saves:

- `source_targets_overlay.png`
- `generated_targets_overlay.png`
- `steps/step_XX/source_window.png`
- `steps/step_XX/selected_cells_overlay.png`
- `steps/step_XX/preserve_map.png`
- `steps/step_XX/steered_window.png`

## 10. `run_manifest.json` Contents

The per-run manifest is the main reproducibility record.

It includes:

- source region identifiers and paths
- grid shape and grid step
- requested target cells
- final edited cells
- final visited cells
- window history
- prototype settings
- preservation settings
- diffusion settings
- output mode
- full CLI args and exact command

This is the file to inspect if you want to know:

- which windows were used
- in what order they ran
- what was edited in each step
- which preservation settings were active

## 11. Argument Reference

This section explains **every argument** of the canonical runner.

### Core Inputs

`--region-bank-csv`

- Required.
- Path to the region bank CSV.
- Must contain rows with valid `region_id`, `image_path`, `feature_grid_path`, and `grid_step_px`.
- The runner uses `region_id` from the edit manifest to look up the correct source region in this file.

`--edit-manifest`

- Required.
- Path to the JSON edit manifest described above.
- This is the canonical editing input.
- One manifest item produces one run.

`--out-dir`

- Required.
- Directory where all run outputs are written.
- The runner creates one subdirectory per manifest item, using `run_id`.

### Run Selection

`--direction`

- Default: `hpv_pos`
- Choices: `hpv_pos`, `hpv_neg`
- Chooses which prototype direction to apply.
- `hpv_pos` means steer toward the positive-direction prototype.
- `hpv_neg` means steer toward the negative-direction prototype.

`--max-runs`

- Default: `0`
- `0` means run all manifest items.
- Positive values cap how many items from the manifest are executed.
- Useful for smoke tests.

`--skip-existing`

- Default: off
- If enabled, runs whose `generated.png` already exists are skipped.
- Useful when resuming a long output directory.

### Reproducibility And Runtime

`--seed`

- Default: `7`
- Global seed for deterministic planning and per-step generator seeds.
- The planner itself is deterministic even without randomness, but the diffusion process uses seeded generators.

`--device`

- Default: `cuda:0`
- Common values:
  - `cuda:0`
  - `auto`
  - `cpu`
- `cuda:0` is now the intended default for this pipeline.
- Use `auto` only if you specifically want automatic device selection.

`--dtype`

- Default: `fp16`
- Choices: `fp16`, `fp32`
- `fp16` is faster and usually the intended mode on GPU.
- `fp32` may be useful for debugging numerical issues.

### PixCell / Diffusion Models

`--pix-model-id`

- Default: `StonyBrook-CVLab/PixCell-1024`
- Hugging Face model id for the PixCell checkpoint.
- This runner is built around the `1024` model.

`--pix-pipeline-id`

- Default: `StonyBrook-CVLab/PixCell-pipeline`
- Custom pipeline id used with the PixCell checkpoint.

`--vae-model-id`

- Default: `stabilityai/stable-diffusion-3-medium-diffusers`
- VAE source used by PixCell generation.

`--vae-subfolder`

- Default: `vae`
- Subfolder inside the VAE checkpoint.

### Sampling Settings

`--steps`

- Default: `30`
- Number of diffusion denoising steps.
- Higher values usually increase compute time and can sometimes improve fidelity.

`--guidance`

- Default: `2.0`
- Guidance scale for conditional generation.
- Higher values usually make the edit conditioning stronger but can also increase drift or artifacts.

`--patch-batch`

- Default: `256`
- Number of MultiDiffusion latent windows processed together.
- Larger values may be faster but require more VRAM.

### SAE Editing Settings

`--prototype-strength`

- Default: `0.8`
- How strongly the SAE latent code moves toward the selected prototype.
- Higher values produce stronger feature-space edits.

`--steer-blend`

- Default: `1.0`
- How strongly the SAE-decoded UNI feature replaces the original UNI feature for the selected cells.
- Lower values soften the feature replacement.

### History-Aware Preservation Settings

`--preserve-edit-strength`

- Default: `0.05`
- Preservation strength inside the current target edit cells.
- `0.0` means the edit area is fully free.
- Higher values make the edit itself more conservative.
- The current default `0.05` gives a very light stabilizing pull without strongly suppressing the edit.

`--preserve-visited-strength`

- Default: `0.95`
- Preservation strength for cells in the current window that were already visited by earlier windows.
- This is the main overlap-stabilization control.
- Increasing this helps keep previously generated context stable.

`--preserve-fresh-context-strength`

- Default: `0.35`
- Preservation strength for non-edit, not-yet-visited context inside the active window.
- Lower values allow more adaptation and smoother transitions.
- Higher values make untouched context more rigid.

### Mid-Diffusion Conditioning Schedule

`--mid-steer-start-ratio`

- Default: `0.5`
- Fraction of the diffusion trajectory where edited conditioning begins.
- `0.0` means start using edited conditioning immediately.
- The current default delays the edited conditioning until the midpoint of the trajectory.

`--mid-steer-end-ratio`

- Default: `1.0`
- Fraction of the trajectory where the conditioning ramp ends.
- `1.0` means the schedule may continue until the last step.

`--mid-steer-alpha-start`

- Default: `0.5`
- Starting conditioning blend between base UNI grid and edited UNI grid.
- `1.0` means fully edited conditioning from the start of the active schedule.
- The current default starts with a moderate edit influence once the active schedule begins.

`--mid-steer-alpha-end`

- Default: `1.0`
- Ending conditioning blend.
- If both start and end are `1.0`, the edited conditioning is constant over the active schedule.

`--mid-steer-alpha-schedule`

- Default: `linear`
- Choices: `linear`, `cosine`
- Shape of the conditioning ramp between start and end ratios.

### SAE / Prototype Inputs

`--sae-ckpt`

- Default: `/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt`
- Path to the SAE checkpoint used for latent editing.

`--sae-cfg`

- Default: `/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json`
- Config used to reconstruct and load the SAE model correctly.

`--prototype-npz`

- Default: `/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz`
- Prototype bundle containing latent ids and prototype vectors.

`--prototype-key`

- Default: `prototype_median`
- Choices: `prototype_mean`, `prototype_median`
- Chooses which prototype vector set to use from the NPZ file.

`--pos-latent`

- Default: `2645`
- Preferred latent id for `hpv_pos`.
- If present in the prototype bundle, this latent is used.
- Otherwise the runner falls back to the first available latent matching the requested direction.

`--neg-latent`

- Default: `7036`
- Preferred latent id for `hpv_neg`.
- Behaves like `--pos-latent`, but for the negative direction.

### Output Controls

`--output-mode`

- Default: `debug`
- Choices: `minimal`, `debug`
- `minimal` saves only the main outputs.
- `debug` adds detailed per-step artifacts.
- The current default is `debug` so step-level inspection is available without rerunning.

## 12. Practical Recommendations

### For A First Sanity Check

Use:

- a small manifest with one run
- `2` to `4` target cells
- `output_mode=debug`
- `device cuda:0`

This lets you verify:

- window planning
- preserve-map behavior
- overlap stability
- final edit plausibility

### If The Edit Is Too Weak

Try:

- higher `prototype-strength`
- higher `guidance`
- lower `preserve-edit-strength`
- lower `preserve-fresh-context-strength`

### If Previously Edited Overlap Keeps Changing Too Much

Try:

- higher `preserve-visited-strength`

This is the main knob for stabilizing already-visited context.

### If The Transition Looks Too Rigid

Try:

- lower `preserve-visited-strength`
- lower `preserve-fresh-context-strength`
- later mid-steer start

## 13. Common Failure Modes

### Unknown `region_id`

Cause:

- the manifest references a region id that is not present in `region_bank.csv`

Fix:

- check `region_id` spelling
- confirm the runner is pointed at the correct bank CSV

### Target Cell Out Of Bounds

Cause:

- a target cell is outside the region grid size

Fix:

- check the region grid shape
- for `2048` with `grid_step_px=256`, valid cells are `0..7` in each axis

### Duplicate `run_id`

Cause:

- two manifest items use the same `run_id`

Fix:

- give them unique names or omit `run_id` and let the runner generate one

### Output Looks Too Different Outside The Edit Area

Cause:

- preservation is too weak, especially for visited context

Fix:

- increase `--preserve-visited-strength`
- optionally increase `--preserve-fresh-context-strength`

## 14. Relationship To Legacy Runner

The old script:

- [run_region_bank_10x_sae_cases.py](/common/users/wq50/wsi_cf/scripts/run_region_bank_10x_sae_cases.py)

is still useful for:

- old case-based experiments
- legacy reproduction
- direct comparison with older results

But for new progressive manifest-driven editing, prefer:

- [run_progressive_region_edit.py](/common/users/wq50/wsi_cf/scripts/run_progressive_region_edit.py)

## 15. Suggested Next Use

A good next workflow is:

1. select target cells upstream using attention or manual review
2. write them into a manifest JSON
3. run this canonical editor
4. inspect `generated.png` and `run_manifest.json`
5. rerun with `output_mode=debug` if you need step-level diagnostics

This keeps target selection and image editing cleanly separated, which is the intended design of the new pipeline.
