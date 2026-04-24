# Methods

This document describes the **current implemented steering method** in `wsi_cf`.

It is intentionally practical:

- what the canonical pipeline does
- what each component means intuitively
- the exact formulas used now
- the current limitations and caveats

For a step-by-step usage guide for the canonical progressive editor, see:

- [PROGRESSIVE_EDIT_TUTORIAL.md](/common/users/wq50/wsi_cf/docs/PROGRESSIVE_EDIT_TUTORIAL.md)

## Overview

The current canonical method is a **manifest-driven progressive region editor**.

It has six main stages:

1. start from a real source region from the prepared region bank
2. load the aligned UNI feature grid for that region
3. read a manifest of global target cells to edit
4. automatically plan overlapping local `4x4` PixCell windows that cover those targets
5. edit only the local center `2x2` cells of each active window using an SAE prototype
6. generate each local window with history-aware latent preservation and write it back into the evolving canvas

So the method is:

- **region-level generation**
- with **coarse cell-level control**
- with **progressive overlapping-window execution**

It is not:

- dense pixel-level editing
- arbitrary free-form masking
- whole-slide diffusion in one pass

The canonical runner for this method is:

- [run_progressive_region_edit.py](/common/users/wq50/wsi_cf/scripts/run_progressive_region_edit.py)

The region-level MIL attention evaluation runner now also uses the progressive editor by default for its editing stage:

- [run_region_attention_classifier_eval.py](/common/users/wq50/wsi_cf/scripts/run_region_attention_classifier_eval.py)

The older script:

- [run_region_bank_10x_sae_cases.py](/common/users/wq50/wsi_cf/scripts/run_region_bank_10x_sae_cases.py)

is still useful for legacy experiments, but it is no longer the best representation of the main progressive editing pipeline.

## Notation

Let:

- `I in R^(H x W x 3)` be the real source image
- `G_h, G_w` be the grid height and width
- `D` be the UNI feature dimension
- `L` be the SAE latent dimension
- `T subset {(g_x, g_y)}` be the global target-cell set from the edit manifest
- `W_k` be the `k`-th planned local `4x4` progressive window
- `S_k subset T` be the target cells edited in step `k`

For the default `1024x1024` local window:

- `H = W = 1024`
- `G_h = G_w = 4`
- `D = 1536`
- `L = 12288`

For a `2048x2048` source region with `grid_step_px = 256`, the full source grid is:

- `8 x 8`

The source UNI feature grid is:

- `X in R^(G_h x G_w x D)`

Flattened across cells:

- `X_flat in R^(N x D)`, where `N = G_h * G_w`

The SAE latent codes are:

- `Z in R^(N x L)`

In the canonical progressive editor there are two spatial levels:

- a **global region grid**, such as `8x8` for a `2048x2048` region
- a **local PixCell window grid**, always `4x4` for PixCell-1024

Only the local center `2x2` cells of each `4x4` window are editable. So each global target cell must be reachable as one of:

- local `(1,1)`
- local `(2,1)`
- local `(1,2)`
- local `(2,2)`

inside at least one planned window.

## Step 1: Real Source Region

### Intuition

We start from a **real tissue crop**, not a synthetic image.

This is the anchor for:

- the source UNI feature grid
- the visual comparison
- the source latent reference used for preservation

### Output

For each sampled source region we save:

- `source_region_actual.png`
- `region_zgrid.npy`

## Step 2: UNI Feature Grid Encoding

### Intuition

The source region is divided into coarse spatial cells. Each cell is encoded with UNI2-H into one feature vector.

For a `1024x1024` image and `256`-pixel step size:

- 16 spatial cells
- arranged as a `4x4` grid

For a `2048x2048` source region and the same step:

- 64 spatial cells
- arranged as an `8x8` grid

This means steering is applied at the level of **conditioning cells**, not arbitrary pixel masks.

### Formula

For each cell `(g_x, g_y)`, extract the corresponding patch and encode it:

- `x_(g_x,g_y) = UNI(patch_(g_x,g_y)) in R^D`

Stacking all cells gives:

- `X in R^(G_h x G_w x D)`

The progressive editor does not feed an entire large grid such as `8x8` to PixCell at once. Instead, it crops overlapping local `4x4` subgrids and runs PixCell window by window.

## Step 3: Manifest Targets And Progressive Window Planning

### Intuition

The edit manifest specifies **global target cells** on the source region grid.

The planner then:

1. builds all valid overlapping `4x4` local windows
2. keeps only windows whose local center `2x2` can represent at least one remaining target
3. starts from the top-most, then left-most reachable target window
4. continues with a deterministic nearest-window rule
5. stops when all requested target cells are covered

This means:

- target selection is upstream
- window planning is automatic
- the canonical editor never edits off-center local cells

### Center-Support Constraint

In the canonical progressive editor, the edited set `S_k` at step `k` is not allowed to be any arbitrary subset of the local `4x4` window.

It is explicitly constrained to the local center `2x2` support:

- `(1,1)`
- `(2,1)`
- `(1,2)`
- `(2,2)`

So if a manifest target cannot land inside the center `2x2` of any valid window, the run is rejected.

## Step 4: SAE Prototype Steering

### Intuition

We do **not** directly change pixels. We edit selected UNI feature vectors in **SAE latent space**.

The current method uses a **full latent prototype vector**:

- encode the UNI feature into SAE latent space
- move the SAE code toward a prototype
- decode back to UNI feature space
- replace only the selected cells

This is different from editing a single latent neuron.

### SAE Encoding

Flatten the active local UNI grid:

- `X_flat in R^(N x D)`

Encode with the SAE:

- `Z = SAE_enc(X_flat) in R^(N x L)`

### Prototype Interpolation

Let:

- `p in R^L` be the selected prototype vector
- `s in [0,1]` be `prototype_strength`

For selected cells `i in S_k`, the current implementation does:

- `Z_edit[i] = (1 - s) * Z[i] + s * p`

For non-selected cells:

- `Z_edit[i] = Z[i]`

So:

- `s = 0` means no latent edit
- `s = 1` means fully replace the selected SAE code with the prototype

### Decode Back To UNI Space

Decode edited SAE latents:

- `X_rec = SAE_dec(Z_edit) in R^(N x D)`

### UNI-Space Blending

Let:

- `b in [0,1]` be `steer_blend`

For selected cells:

- `X_new[i] = (1 - b) * X_flat[i] + b * X_rec[i]`

For non-selected cells with `keep_non_selected=True`:

- `X_new[i] = X_flat[i]`

Reshape back to grid form:

- `X_new_grid in R^(G_h x G_w x D)`

### What The Parameters Mean

`prototype_strength`

- controls how far the selected SAE code moves toward the prototype

`steer_blend`

- controls how strongly the decoded edited UNI feature replaces the original UNI feature

Interpretation:

- high `prototype_strength` + high `steer_blend` gives the strongest cell edit
- low values give softer edits

## Step 5: Diffusion Conditioning Schedule

### Intuition

PixCell does not have to use the edited conditioning for the whole diffusion trajectory.

We can:

- apply the edit from the beginning
- start the edit later
- ramp the edit from weak to strong

This is useful because strong early edits can cause global drift.

### Base And Edited Conditioning

Let:

- `X_base` be the original local UNI grid
- `X_edit` be the edited local UNI grid

At diffusion step `t`, define a scalar blending coefficient `alpha_t in [0,1]`.

Then the active conditioning grid for the current local window is:

- `X_t = (1 - alpha_t) * X_base + alpha_t * X_edit`

### Schedule

Let:

- `r_t` be the normalized diffusion step ratio in `[0,1]`
- `r_start` be `mid_steer_start_ratio`
- `r_end` be `mid_steer_end_ratio`
- `a_0` be `mid_steer_alpha_start`
- `a_1` be `mid_steer_alpha_end`

Then:

- if `r_t <= r_start`, `alpha_t = a_0`
- if `r_t >= r_end`, `alpha_t = a_1`
- otherwise interpolate between `a_0` and `a_1`

Linear schedule:

- `u = (r_t - r_start) / (r_end - r_start)`
- `alpha_t = (1 - u) * a_0 + u * a_1`

Cosine schedule:

- `u = (r_t - r_start) / (r_end - r_start)`
- `u_cos = 0.5 - 0.5 * cos(pi * u)`
- `alpha_t = (1 - u_cos) * a_0 + u_cos * a_1`

### Practical Meaning

If you want a delayed edit:

- `mid_steer_start_ratio = 0.5`
- `mid_steer_alpha_start = 0.5`
- `mid_steer_alpha_end = 1.0`

Then:

- early diffusion uses mostly original conditioning
- later diffusion uses progressively stronger edited conditioning

In the canonical progressive editor, this schedule is applied independently at each local window step.

## Step 6: History-Aware Latent Preservation

### Intuition

The conditioning edit says what should change. The preservation mechanism says what should stay the same.

This is done in **image latent space**, not in UNI space.

In practice, preservation acts like a soft tether to the original source image during denoising.

- the SAE-edited UNI features tell PixCell what morphology we want
- the preservation path tells the sampler not to drift too far from the original image where we want visual stability
- in progressive editing, previously visited overlap should usually be preserved more strongly than fresh context

### Canonical Progressive Preserve Map

For each local `4x4` window, the canonical progressive editor partitions the window into three conceptual regions:

- `edit_core`
  The selected target cells edited in the current step.
- `visited_context`
  Cells inside the current window that were already covered by previous windows.
- `fresh_context`
  Non-edit cells inside the current window that have not been visited yet.

These are controlled by three strengths:

- `lambda_edit = preserve_edit_strength`
- `lambda_visit = preserve_visited_strength`
- `lambda_fresh = preserve_fresh_context_strength`

The intended relationship is:

- `lambda_visit > lambda_fresh`

because previously visited overlap should be held more stable than untouched context.

### Preserve-Map Formula

Let:

- `Y_t` be the current denoised latent state at step `t`
- `R_t` be the original source latent re-noised to the scheduler state at step `t`
- `P in [0,1]^(H x W)` be the image-space preserve-strength map for the current local window
- `P_lat` be the latent-resolution version of that map

Then the preserved latent update is:

- `Y_t <- (1 - P_lat) * Y_t + P_lat * R_t`

So:

- `P_lat = 0` means no preservation pull
- `P_lat = 1` means fully snap to the source latent trajectory
- intermediate values mean partial preservation

The canonical defaults are:

- `preserve_edit_strength = 0.05`
- `preserve_visited_strength = 0.95`
- `preserve_fresh_context_strength = 0.35`

This means:

- the edit core is mostly free, but not completely unconstrained
- previously visited overlap is strongly stabilized
- fresh context still has room to adapt and blend

### Interpretation

This encourages:

- selected center-support cells to change
- previously visited overlap to remain stable
- fresh context to remain more flexible than visited context

More concretely:

- high visited-context preservation reduces accidental rewrites in overlap regions
- nonzero edit preservation can reduce over-strong edits inside the chosen center-support cells
- too much preservation can make the edit weak or make rectangular boundaries more visible
- too little preservation can let the whole window drift, even outside the intended edit cells

### Legacy Contrast

The legacy single-window runner still uses the older two-region formulation based on:

- `preserve_outside_strength`
- `preserve_edit_strength`

That older formulation is still useful, but it is no longer the best description of the canonical progressive editor.

## Step 7: Progressive Outputs

### Canonical Outputs

The canonical progressive editor saves:

- `source_region_actual.png`
- `generated.png`
- `run_manifest.json`
- `experiment_args.json` at the output root

If debug output is enabled, it also saves:

- per-step local source windows
- selected-cell overlays
- preserve-map previews
- per-step local generated windows

### Legacy Baseline Outputs

The older single-window and case-based scripts may also save files such as:

- `source_region_generated.png`
- `baseline_generated.png`

Those are still useful for legacy comparisons, but they are not the main output convention of the canonical progressive editor.

## What Is Being Edited Right Now

To be precise, the canonical progressive editor edits:

- selected **UNI conditioning cells**
- but only when those global cells land inside the local center `2x2` of the active `4x4` PixCell window

and it preserves:

- previously visited and fresh **image latent regions** with different strengths

It does **not** currently do:

- direct pixel editing
- direct selected-region VAE latent editing as the primary edit
- exact structure-boundary editing
- whole-slide diffusion in one pass

## Magnification Caveat

This is an important current limitation.

### Current Default Prototype Bundle

The repo-local prototype bundle currently used by default was built from the SAE at:

- `/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json`

That SAE was trained with:

- `magnification = 20x`

### Current 10x Local-Region Runners

The new local `10x` runners:

- sample actual `10x` image crops
- encode those crops directly with UNI2-H

So the current default `10x` steering setup uses:

- `20x` SAE concepts
- actual `10x` source features

This is a representation mismatch.

### Closer Existing Alternative

There is also an older SAE run at:

- `/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_10x_pool2x2/`

with:

- `magnification = 10x_pool2x2`

This is closer to the current `10x` workflow than the default `20x` SAE, but still not identical to actual optical `10x`.

## Honest Interpretation

The current method is best described as:

- **region-level generation with coarse spatial cell-level control under progressive overlapping context**

This is useful for studying:

- locality
- edit propagation
- plausibility under context
- overlap stability

But it should not be described as:

- dense fine-grained boundary editing
- exact arbitrary structure editing
- true whole-slide diffusion

## Current Implementation References

Prototype edit:

- `/common/users/wq50/SAE_path/utils/sae_edit.py`

PixCell scheduling and preservation:

- `/common/users/wq50/wsi_cf/src/wsi_cf/generation/pixcell.py`

Canonical progressive planner and manifest logic:

- `/common/users/wq50/wsi_cf/src/wsi_cf/steering/progressive.py`

Canonical progressive runner:

- `/common/users/wq50/wsi_cf/scripts/run_progressive_region_edit.py`

Legacy 10x bank SAE runner:

- `/common/users/wq50/wsi_cf/scripts/run_region_bank_10x_sae_cases.py`
