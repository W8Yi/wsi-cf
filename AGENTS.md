# AI Orientation

This file is the fast-start map for AI coding agents and new collaborators.
For deeper method details, read the linked docs after this overview.

## What This Repo Is

`wsi_cf` is a research codebase for whole-slide image counterfactual steering.
The canonical workflow is:

1. visualize attention on whole-slide feature bags,
2. find editable tissue regions,
3. load or build aligned UNI feature grids,
4. edit selected grid cells in SAE latent space,
5. run PixCell-1024 progressive region generation,
6. optionally re-encode and evaluate edited regions downstream.

The default task is HNSCC HPV status, but newer scripts support broader
classifier and concept-discovery workflows.

## Read First

Read these in order when getting oriented:

1. `README.md` - high-level workflow and default commands.
2. `docs/METHODS.md` - current implemented steering method.
3. `docs/SCRIPTS.md` - active entrypoint registry.
4. `docs/PROGRESSIVE_EDIT_TUTORIAL.md` - practical progressive-edit usage.
5. `src/wsi_cf/common/paths.py` - default resources, model variants, and external data paths.

Useful status/context docs:

- `docs/PROGRESS.md` - current findings, caveats, and active problems.
- `docs/MIGRATION_PLAN.md` - what has moved into this focused package.
- `docs/TEST.md` - historical region-bank and steering test protocol.

## Canonical Entrypoints

- `scripts/visualize_attention.py` - MIL/CLAM attention visualization.
- `scripts/find_regions.py` - attention-guided/manual region discovery and edit manifests.
- `scripts/run_progressive_region_edit.py` - main manifest-driven progressive editor.
- `scripts/train_attention_classifier.py` - repo-native attention MIL classifier training.
- `scripts/prepare_classifier_concept_associations.py` - bridge from trained classifiers to concept discovery.
- `scripts/find_label_concepts.py` - label-relevant SAE concept selection and representative tiles.

Core HNSCC HPV examples live in `examples/hnscc_hpv/`.

## Core Modules

- `src/wsi_cf/common/paths.py` - repository defaults and task resource paths.
- `src/wsi_cf/common/runtime.py` - runtime helpers such as device and seed handling.
- `src/wsi_cf/data/` - H5, slide, donor-pool, concept-bank, attention-proposal, and region-bank utilities.
- `src/wsi_cf/models/` - MIL, CLAM, and SAE model definitions.
- `src/wsi_cf/steering/progressive.py` - progressive window planning, target coverage, and edit history.
- `src/wsi_cf/steering/sae_edit.py` - SAE latent-space editing of UNI feature grids.
- `src/wsi_cf/steering/sae_runtime.py` - SAE load/encode/decode helpers.
- `src/wsi_cf/generation/pixcell.py` - PixCell windowing, conditioning, scheduling, preservation, and multidiffusion helpers.
- `src/wsi_cf/eval/` - local-region and HNSCC HPV classifier evaluation helpers.

Prefer putting reusable logic under `src/wsi_cf/` and keeping scripts as thin
CLI orchestration layers.

## Default Resources And Assumptions

- Default task: `hnscc_hpv`.
- Default SAE variant: `tcga_uni2_sae_relu_v1`.
- Default grid step: `256` px.
- Default progressive local PixCell window: `1024x1024`, represented as a `4x4` UNI grid.
- Default HNSCC HPV prototype latents:
  - HPV+: `2645`
  - HPV-: `7036`

Large raw slides and precomputed feature stores are external. Do not assume
they are vendored in this repository. External defaults are centralized in
`src/wsi_cf/common/paths.py` and `resources/tasks/hnscc_hpv.json`.

## Testing

Tests live in `tests/`.
They add `src/` to `sys.path` through `tests/conftest.py`; this repo does not
currently rely on a package install for tests.

Use the project `pace` conda environment for this repo. It has the expected
ML/CUDA/OpenSlide/PixCell dependencies:

```bash
conda activate pace
# or call it directly when activation is inconvenient:
/common/users/wq50/envs/pace/bin/python
```

Common commands:

```bash
pytest -q
pytest --collect-only -q
pytest tests/test_progressive_region_runner.py -q
```

Some tests import `torch` and require the correct ML/CUDA environment. If test
collection fails with `libcusparseLt.so.0`, switch to the project environment
above before treating it as a code failure. If CUDA is not visible inside the
agent shell, run GPU generation from a GPU-visible shell with the same `pace`
environment, for example `CUDA_VISIBLE_DEVICES=3 DEVICE=cuda:0 ...`.

## Working Tree Conventions

This repo often contains generated artifacts, experiment outputs, and local
work-in-progress changes. Before editing, check:

```bash
git status --short --branch
```

Do not remove or overwrite generated artifacts or untracked experiment outputs
unless the user explicitly asks. Keep edits scoped to the requested task.

## Practical Debugging Notes

- Start with `docs/SCRIPTS.md` when looking for the right runner.
- Start with `docs/METHODS.md` when reasoning about intended behavior.
- Start with `src/wsi_cf/common/paths.py` when defaults or missing files are confusing.
- Start with tests when changing planner, manifest, region-bank, or PixCell scheduling behavior.
- For progressive editing bugs, inspect `scripts/run_progressive_region_edit.py`,
  `src/wsi_cf/steering/progressive.py`, and `src/wsi_cf/generation/pixcell.py` together.
