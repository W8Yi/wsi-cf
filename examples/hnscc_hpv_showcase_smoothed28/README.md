# HNSCC HPV Showcase Smoothed-28 Tutorial

This folder is a reproducible tutorial for the current showcase counterfactual
case. It starts from the selected 2048 px HNSCC region and smoothed-28 cell
manifest, runs our progressive steering method, runs a matched naive baseline,
and writes data-only metrics for comparing the two outputs.

The example uses the smoothed-28 selection:

- attention-only top-23 seed cells,
- one 4-neighbor smoothing pass with border relaxation,
- pruning back to 28 selected cells,
- no SAE-similarity filtering in the selector.

## 1. Run Progressive Steering

```bash
PYTHON=/common/users/wq50/envs/pace/bin/python \
DEVICE=cuda:3 \
examples/hnscc_hpv_showcase_smoothed28/01_run_progressive_edit.sh
```

Output root:

```text
artifacts/hnscc_hpv_showcase_smoothed28_tutorial/progressive/
```

Policy:

```text
configs/edit_policies/original28_flip.json
```

This is the paper showcase setting: progressive PixCell windows with
history-aware latent preservation for visited and fresh-context regions.

## 2. Run Naive Baseline

```bash
PYTHON=/common/users/wq50/envs/pace/bin/python \
DEVICE=cuda:3 \
examples/hnscc_hpv_showcase_smoothed28/02_run_naive_baseline.sh
```

Output root:

```text
artifacts/hnscc_hpv_showcase_smoothed28_tutorial/naive_previous_settings/
```

Policy:

```text
configs/edit_policies/previous_naive_on_smoothed28.json
```

This baseline uses the same selected cells but removes all preservation:
`preserve_edit_strength=0`, `preserve_visited_strength=0`, and
`preserve_fresh_context_strength=0`.

## 3. Compute Data Metrics

```bash
PYTHON=/common/users/wq50/envs/pace/bin/python \
examples/hnscc_hpv_showcase_smoothed28/03_compute_metrics.sh
```

Metrics output:

```text
artifacts/hnscc_hpv_showcase_smoothed28_tutorial/metrics/
  classifier_predictions.csv
  classifier_predictions.json
  perturbation_summary.csv
  perturbation_summary.json
  perturbation_boxplot_stats.csv
  perturbation_boxplot_stats.json
  progressive_stage_prediction_trajectory.csv
  progressive_stage_prediction_trajectory.json
  verification.json
```

The metric script recomputes:

- source, progressive, and naive HPV predictions from saved re-encoded UNI2
  grids using `resources/models/classifiers/hnscc_hpv/mil_split0.pt`,
- per-pixel mean absolute RGB difference from the source image,
- Tukey boxplot statistics for the same RGB-difference distribution,
- optional progressive stage trajectory by rebuilding intermediate stage
  canvases from `run_manifest.json` and the committed `steered_window.png`
  files.

## One Command

```bash
PYTHON=/common/users/wq50/envs/pace/bin/python \
DEVICE=cuda:3 \
examples/hnscc_hpv_showcase_smoothed28/run_all.sh
```

## Useful Overrides

Use another output root:

```bash
OUT_ROOT=artifacts/my_showcase_reproduction \
examples/hnscc_hpv_showcase_smoothed28/run_all.sh
```

Use existing runs and compute metrics only:

```bash
PYTHON=/common/users/wq50/envs/pace/bin/python \
examples/hnscc_hpv_showcase_smoothed28/compute_metrics.py \
  --progressive-run-dir artifacts/hnscc_hpv_attention_only_top23_smooth4_prune28_flip_policy/case_clean_sae_expand_attn_seed_p90__attention_only_top23_smooth4_prune28 \
  --naive-run-dir artifacts/hnscc_hpv_attention_only_top23_smooth4_prune28_previous_naive_settings/case_clean_sae_expand_attn_seed_p90__attention_only_top23_smooth4_prune28 \
  --out-dir artifacts/hnscc_hpv_showcase_smoothed28_tutorial/metrics_existing_runs \
  --stage-mode reencode
```

Use `--stage-mode none` when only final classifier and perturbation metrics are
needed.
