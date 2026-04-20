# Methods

This document describes the **current implemented steering method** in `wsi_cf`.

It is intentionally practical:

- what the pipeline does
- what each component means intuitively
- the exact formulas used now
- the current limitations and caveats

## Overview

The current local counterfactual pipeline has four main stages:

1. sample a real source region from a slide
2. encode that region into a UNI feature grid
3. edit selected UNI cells using an SAE prototype
4. generate an output image with PixCell, optionally preserving non-edited regions

For a `1024x1024` source region with `grid_step_px=256`, the UNI conditioning grid is `4x4`.

So the method is:

- **region-level generation**
- with **coarse cell-level control**

not dense pixel-level editing.

## Notation

Let:

- `I in R^(H x W x 3)` be the real source image
- `G_h, G_w` be the conditioning grid height and width
- `D` be the UNI feature dimension
- `L` be the SAE latent dimension

For the default `1024x1024` setup:

- `H = W = 1024`
- `G_h = G_w = 4`
- `D = 1536`
- `L = 12288`

The source UNI feature grid is:

- `X in R^(G_h x G_w x D)`

Flattened across cells:

- `X_flat in R^(N x D)`, where `N = G_h * G_w`

The SAE latent codes are:

- `Z in R^(N x L)`

The selected edited cells are encoded by a binary mask:

- `M in {0,1}^(G_h x G_w)`

Flattened:

- `m in {0,1}^N`

## Step 1: Real Source Region

### Intuition

We start from a **real tissue crop**, not a synthetic image.

This is the anchor for:

- the source UNI feature grid
- the visual comparison
- optional latent preservation outside the edited region

### Output

For each sampled source region we save:

- `source_region_actual.png`
- `region_zgrid.npy`

## Step 2: UNI Feature Grid Encoding

### Intuition

The source region is divided into coarse spatial cells.  
Each cell is encoded with UNI2-H into one feature vector.

For a `1024x1024` image and `256`-pixel step size:

- 16 spatial cells
- arranged as a `4x4` grid

This means steering is applied at the level of **conditioning cells**, not arbitrary pixel masks.

### Formula

For each cell `(g_x, g_y)`, we extract the corresponding `256x256` patch and encode it:

- `x_(g_x,g_y) = UNI(patch_(g_x,g_y)) in R^D`

Stacking all cells gives:

- `X in R^(G_h x G_w x D)`

## Step 3: SAE Prototype Steering

### Intuition

We do **not** directly change pixels.  
We edit the selected UNI feature vectors in **SAE latent space**.

The current method uses a **full latent prototype vector**:

- encode the UNI feature into SAE latent space
- move the SAE code toward a prototype
- decode back to UNI feature space
- replace only the selected cells

This is different from editing a single latent neuron.

### SAE Encoding

Flatten the UNI grid:

- `X_flat in R^(N x D)`

Encode with the SAE:

- `Z = SAE_enc(X_flat) in R^(N x L)`

### Prototype Interpolation

Let:

- `p in R^L` be the selected prototype vector
- `s in [0,1]` be `prototype_strength`

For selected cells `i in S`, the current implementation does:

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

## Step 4: Diffusion Conditioning Schedule

### Intuition

PixCell does not have to use the edited conditioning for the whole diffusion trajectory.

We can:

- apply the edit from the beginning
- start the edit later
- ramp the edit from weak to strong

This is useful because strong early edits can cause global drift.

### Base And Edited Conditioning

Let:

- `X_base` be the original UNI grid
- `X_edit` be the edited UNI grid

At diffusion step `t`, define a scalar blending coefficient `alpha_t in [0,1]`.

Then the active conditioning grid is:

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
- `mid_steer_alpha_start = 0.0`
- `mid_steer_alpha_end = 1.0`

Then:

- early diffusion uses mostly original conditioning
- later diffusion uses edited conditioning

## Step 5: Optional Preservation Outside The Edited Region

### Intuition

The conditioning edit says what should change.  
The preservation mechanism says what should stay the same.

This is done in **image latent space**, not in UNI space.

### Current Mechanism

1. Take the real source image `I`
2. Encode it into VAE latents:
   - `L_src`
3. Build a spatial mask from the selected edited cells
4. During diffusion, outside the edited region, pull the current latent state back toward the noised trajectory of `L_src`

### Mask

The current preserve mask is a hard rectangular mask induced by the selected grid cells.

Let:

- `Q in [0,1]^(H x W)` be the image-space edit mask

Currently `Q` is binary and grid-aligned:

- `Q = 1` inside selected cells
- `Q = 0` outside selected cells

This mask is then resized to latent-space resolution.

### Latent Preservation Formula

Let:

- `Y_t` be the current denoised latent state at step `t`
- `R_t` be the original source latent re-noised to the scheduler state at step `t`
- `lambda in [0,1]` be `preserve_outside_strength`
- `Q_lat` be the latent-resolution edit mask

Then the preserved latent update is:

- `W = lambda * (1 - Q_lat)`
- `Y_t <- (1 - W) * Y_t + W * R_t`

So:

- inside the edited region (`Q_lat = 1`): no preservation pull
- outside the edited region (`Q_lat = 0`): pull toward the original source latent trajectory

### Interpretation

This encourages:

- selected cells to change
- non-selected regions to remain close to the source image

### Current Limitation

Because the mask is currently hard and rectangular, strong preservation can create visible grid-aligned boundaries.

## Step 6: Baseline And Generated Outputs

### Actual Source

- `source_region_actual.png`

This is the real sampled source image.

### Generated Baseline

- `source_region_generated.png`
- `baseline_generated.png`

This is the diffusion-regenerated baseline from the same source region with no steering.

### Edited Outputs

Each case generates a steered image under the selected SAE edit and scheduling settings.

## What Is Being Edited Right Now

To be precise, the current method edits:

- selected **UNI conditioning cells**

and optionally preserves:

- non-selected **image latent regions**

It does **not** currently do:

- direct pixel editing
- direct selected-region VAE latent editing as the primary edit
- exact structure-boundary editing

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

- **region-level generation with coarse spatial cell-level control**

This is useful for studying:

- locality
- edit propagation
- plausibility under context

But it should not be described as:

- dense fine-grained boundary editing
- exact arbitrary structure editing

## Current Implementation References

Prototype edit:

- `/common/users/wq50/SAE_path/utils/sae_edit.py`

PixCell scheduling and preservation:

- `/common/users/wq50/wsi_cf/src/wsi_cf/generation/pixcell.py`

10x bank SAE runner:

- `/common/users/wq50/wsi_cf/scripts/run_region_bank_10x_sae_cases.py`

10x one-off SAE runner:

- `/common/users/wq50/wsi_cf/scripts/run_sae_10x_selected_cells_test.py`
