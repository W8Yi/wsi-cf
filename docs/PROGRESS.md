# Progress Log

This file is the running log for:

- completed experiments
- current findings
- failures / caveats
- decisions we made
- next experiments to run
- ideas worth keeping but not implementing yet

It is meant to be short, practical, and updated continuously.

---

## Current Focus

Primary project direction right now:

- attention-guided local counterfactual steering for HNSCC HPV
- local `1024x1024` steering at `10x`
- region-level and slide-scale stitching tests with `PixCell-1024`
- seam handling for large stitched outputs
- using SAE prototypes for controlled concept steering

Current paper framing:

- whole-slide attention chooses **where**
- local region generation shows **what changes**
- the main claim is **slide-aware local counterfactual steering**
- not true whole-slide one-shot generation

---

## Current Status

What is working now:

- one-tile and multi-tile steering with `PixCell-1024`
- donor-tile replacement and SAE prototype steering
- `10x` region-bank creation with aligned `4x4` UNI grids
- local attention-guided `1024` steering
- local before/after MIL scoring on the re-encoded steered region
- `4096x4096` stitching stress tests with overlapping `1024` windows
- central-area and seam-edge stress layouts for large stitched regions

What still needs work:

- better region proposal quality for attention-guided selection
- cleaner handling of fallback / no-op regions
- better seam suppression without washing out edits
- optional `256` seam-repair stage
- tighter evaluation metrics for locality and artifact rate
- magnification-consistent concept experiments

---

## Key Decisions

- Keep `256` and `1024` as the main scale comparison for the paper.
- Treat `1024` as the main pathology-grounded regional editing scale.
- Use attention for region selection, not as the final scale of generation.
- Evaluate local `1024` bags first; full-slide reinsertion is stage 2.
- Preserve outside latents by default for local attention-guided runs.
- Skip poor fallback regions by default in the local attention-guided runner.
- Prefer overlapping-window generation plus stitching for large outputs instead of claiming native `4096` generation.

---

## Failed But Useful Experiments

### Random SAE Top-Concept Center-2x2 Probe

Status:

- failed as a showcase result; keep as a diagnostic control

What was run:

- wrapper: `examples/sae_concepts/02_random_region_center2x2_10_top_concepts.sh`
- engine: `scripts/run_export_random_concept_steer.py`
- output: `paper_example/random_sae_top_concepts_center2x2/`
- source region: `TCGA-EA-A5O9-01Z-00-DX1__random1024__mag_20p0__gx_54__gy_41`
- edit mask: fixed central `2x2` cells in a `4x4` UNI grid
- SAE variant: `tcga_uni2_sae_relu_v1`
- concepts: random distinct concepts from the exported top-concept pool
- selected latents: `10832, 1256, 9686, 6778, 6334, 2911, 4689, 4250, 8831, 3499`
- settings: `default`, `prototype_strength=0.9`, `prototype_top_k=5`, `steps=30`, `seed=7`

Exact command:

```bash
PYTHON=/common/users/wq50/envs/pace/bin/python \
DEVICE=cuda:3 \
SEED=7 \
OUT_DIR=paper_example/random_sae_top_concepts_center2x2 \
examples/sae_concepts/02_random_region_center2x2_10_top_concepts.sh
```

Why it is retained:

- it is a useful negative/control experiment for checking whether arbitrary
  exported SAE concepts produce visually specific local edits
- it should not be treated as the default steering showcase
- PixCell emitted a model-load warning that `y_pos_embed.y_pos_embed` was newly
  initialized, which should be kept in mind when interpreting these outputs

Decision:

- use `examples/hnscc_hpv_showcase_smoothed28/01_run_progressive_edit.sh` with
  `configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json`
  as the default HNSCC steering
  example instead

---

## Default HNSCC Steering Reproduction

- script: `examples/hnscc_hpv_showcase_smoothed28/01_run_progressive_edit.sh`
- policy: `configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json`
- direction: `hpv_neg`
- SAE/prototype provenance: legacy `relu_sae_base` checkpoint with the existing HNSCC prototype bundle
- regenerated output: `paper_example/hnscc_hpv_showcase_smoothed28_default_regenerated_20260526/progressive/`
- manifest behavior: 28 historical smoothed targets requested; 14 center-support cells edited in 5 windows; 14 unsupported border targets dropped and recorded

---

## Experiment Timeline

### 1. PixCell Local Steering

Status:

- done

What we learned:

- one edited cell in `PixCell-1024` usually produces a soft local effect, not a hard quarter-image replacement
- full `4x4` edits are stronger but can over-change the region
- scheduled steering and outside-latent preservation are useful for reducing drift

Main takeaway:

- `1024` is better for region-level morphology than `256`, but local control is still coarse because steering happens on the `4x4` conditioning grid

### 2. 10x Region Bank

Status:

- done

What we built:

- random `10x` `1024x1024` region bank
- aligned `4x4` UNI feature grids
- per-region images, cells, and metadata

Why it matters:

- gives a fixed pool of real local regions for controlled steering tests

### 3. SAE Selected-Cells Tests

Status:

- done

What we learned:

- random, neighboring, and block edits give different locality / coherence tradeoffs
- isolated cell edits can produce too-small visible changes
- connected multi-cell edits often look more natural

### 4. Attention-Guided Local 1024 Pipeline

Status:

- implemented and run

What it does:

- computes MIL attention on the full slide bag
- selects one local `1024` region
- steers only selected high-attention cells
- re-encodes the steered image with UNI
- replaces selected local cells in the local bag
- reruns MIL on that local bag

Important confirmed behavior:

- the post-steer classifier score uses freshly re-encoded steered features, not stale source features

Current issue:

- some top-attention regions can still be low-information tissue unless quality-filtered

Latest fix:

- added region quality filtering and deduplicated candidate proposals
- local runner now skips poor fallback regions by default

### 5. Seam / Stitching Stress Tests

Status:

- implemented

What we built:

- `4096x4096` stress runner
- overlapping `1024x1024` windows
- edge-cell seam stress layouts
- central-area large-edit layout
- multiple stitch modes:
  - `hard_stitch`
  - `overlap_average`
  - `center_weighted_blend`
  - `trusted_center_only`

What we expect:

- `hard_stitch` is the baseline worst case
- too much overlap can dilute edits
- `center_weighted_blend` is likely the best compromise
- `trusted_center_only` suppresses seams strongly but may weaken edge edits

---

## Important Findings

### Re-encoding

- confirmed: after steering, we re-encode the generated image into a fresh UNI `4x4` grid
- confirmed: the local post-steer MIL score is computed on those re-encoded features

### Attention-guided region selection

- attention occupancy alone is not enough
- some selected regions are tissue-rich but not information-rich
- low-cellularity / low-contrast regions can still pass naive attention filters

### Preservation

- when preservation is off, global drift is expected
- preservation should remain default-on for local steering experiments
- hard rectangular preserve masks can still produce visible box artifacts

### Stitching

- overlap reduces seams but can soften edits
- if overlap is too large, the edit becomes diluted
- a future `256` repair model may be better as a seam-polishing stage than more overlap

---

## Active Problems

### 1. Region quality

Problem:

- high-attention anchors can still land in weak or low-cellularity regions

Current mitigation:

- quick image-quality filters:
  - tissue score
  - dark fraction
  - saturation fraction

Still open:

- maybe use better nucleus / hematoxylin / cell-density heuristics later

### 2. Coarse control

Problem:

- steering is still applied on the `4x4` grid, not on dense masks

Consequence:

- we should claim coarse local control, not exact structure-boundary editing

### 3. Seams at slide scale

Problem:

- edge edits can still create visible disagreement between neighboring `1024` windows

Current plan:

- test moderate overlap
- test center-weighted stitching
- test trusted-center-only stitching
- later test optional `256` seam repair

### 4. Magnification mismatch

Problem:

- some concept bundles were built from `20x` representations while some current experiments are on `10x`

Need:

- keep careful track of feature representation and magnification for each experiment

---

## Suggested Metrics

These are the metrics we likely need to support the paper claim.

### Prediction shift

- local MIL probability before vs after
- fraction of slides shifting in the intended direction
- mean / median shift on valid non-fallback runs

### Locality

- change inside selected cells vs outside selected cells
- image-space difference inside / outside edited region
- feature-space difference inside / outside edited region

### Preservation

- similarity between source and steered image outside edited region
- similarity between source and baseline-generated region

### Artifact rate

- visible rectangular border rate
- seam visibility score for stitched outputs
- proportion of low-quality or skipped regions

### Plausibility

- blinded review or pathologist rating if possible
- morphology-specific sanity checks for each disease task

---

## Good Next Experiments

### Immediate

- rerun attention-guided local `1024` with the new quality-filtered region selection
- compare valid non-fallback runs only
- inspect the strongest positive and strongest negative cases manually

### Local steering

- delayed steering:
  - `mid_steer_start_ratio = 0.5`
  - `mid_steer_alpha_start = 0.0`
  - `mid_steer_alpha_end = 1.0`
- compare against always-on steering

### Stitching

- hard stitch baseline with `stride = 1024`
- moderate overlap with `stride = 768`
- center-weighted blend at `stride = 768`
- trusted-center-only at `stride = 768`

### 20x large-area test

- run `4096x4096` at `20x`
- use `center_area` layout
- edit a central `2048x2048` area
- compare stitch modes

### Seam repair

- test `1024` main generation + optional `256` seam-repair stage

---

## Runs To Remember

Important artifact directories so far:

- `artifacts/attention_guided_local_1024`
- `artifacts/attention_guided_local_1024_preserve_default`
- `artifacts/attention_guided_region_bank_10x`
- `artifacts/hnscc_region_bank_10x_1024_sample4`
- `artifacts/hnscc_10x_concept_regions_from_20x_representatives`
- `artifacts/seam_stress_4096_*`

When adding a new important run, record:

- run path
- goal
- key settings
- what worked
- what failed

---

## Notes / Ideas Parking Lot

- use a `256` model for local seam repair, not as the main region editor
- test feathered preserve masks instead of hard rectangular preserve masks
- add automatic rectangular-artifact detection
- add feature-level trusted-center stitching for slide-scale reinsertion
- build magnification-matched `10x` concept prototypes
- extend the same framework to LUAD, MSI, KIRC, CESC, and tumor-vs-normal tasks

---

## Update Rule

Whenever we finish an experiment, add:

- one short description of what was run
- one short result summary
- one line on whether it changes the next plan

This file should stay concise and useful rather than becoming a long notebook dump.
