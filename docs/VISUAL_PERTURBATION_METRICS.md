# Visual Perturbation Metrics

This benchmark compares the paper edit method against a deliberately poor naive
baseline. The goal is to report whether a classifier transition was achieved
with less visible image disruption.

## Methods Compared

- `ours`: the normal prediction-transition edit output, currently policy 09
  (`configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json`).
- `bad_naive`: `configs/edit_policies/bad_naive_no_preserve_no_sliding.json`.
  This baseline keeps padded center-2x2 support so all benchmark manifest cells
  are legal, but disables source-image preservation, disables stride-1 sliding
  refinement by using a non-overlapping 4-cell stride, commits the whole PixCell
  window, and applies full-strength steering for the full diffusion trajectory.

The naive baseline is intentionally visually harsh. It is meant to show that
the transition-aware method changes the classifier while perturbing the image
less, not to be a competitive editing policy.

## Primary RGB Metric

For each generated image, compare it to the same source region:

```text
mean_abs_rgb = mean_pixels(mean_channels(abs(after_rgb - source_rgb)))
```

Values are in raw RGB intensity units on the 0-255 scale. Lower means the
after-image is visually closer to the source.

The metric is reported over:

- full region,
- selected steered cells from the edit manifest,
- unselected cells,
- committed pixels from `run_manifest.json`,
- pixels outside committed regions.

## Reproducible Commands

Generate and score the bad-naive baseline in streaming mode. This is the
space-saving default: only `STREAM_CHUNK_SIZE` naive runs are stored at once,
metrics are written immediately, and the temporary naive images are removed
after each chunk.

```bash
cd /common/users/wq50/wsi_cf
DEVICE=cuda:0 \
TASKS="hnscc_hpv normal_tumor prad_morphology_group" \
RUN_NAIVE=1 \
examples/paper_metrics/03_run_visual_perturbation_naive_baseline.sh
```

Use `STREAM_CHUNK_SIZE=1` for minimum disk use, or `STREAM_CHUNK_SIZE=8/16`
to avoid reloading PixCell for every individual run.

If you intentionally want to keep the full naive image tree and score it later:

```bash
cd /common/users/wq50/wsi_cf
DEVICE=cuda:0 \
TASKS="hnscc_hpv normal_tumor prad_morphology_group" \
RUN_NAIVE=1 \
STREAM_NAIVE=0 \
RUN_VISUAL_METRICS=0 \
examples/paper_metrics/03_run_visual_perturbation_naive_baseline.sh

RUN_VISUAL_METRICS=1 \
STREAM_NAIVE=0 \
examples/paper_metrics/03_run_visual_perturbation_naive_baseline.sh
```

For a single direction:

```bash
cd /common/users/wq50/wsi_cf
/common/users/wq50/envs/pace/bin/python scripts/compare_edit_visual_perturbation.py \
  --ours-root paper_outputs/prediction_transition_benchmark_test_only_unbalanced/generated/hnscc_hpv/hpv_pos_to_hpv_neg \
  --naive-root paper_outputs/prediction_transition_benchmark_test_only_unbalanced_bad_naive_no_preserve_no_sliding/generated/hnscc_hpv/hpv_pos_to_hpv_neg \
  --manifest paper_outputs/prediction_transition_benchmark_test_only_unbalanced/manifests/hnscc_hpv/hpv_pos_to_hpv_neg/combined_manifest.json \
  --out-dir paper_outputs/prediction_transition_benchmark_test_only_unbalanced/metrics_visual/hnscc_hpv/hpv_pos_to_hpv_neg
```

By default the paper PRAD benchmark uses `prad_morphology_group`, the three
collapsed morphology classes. The older `prad_grade_group` directions still
exist, but they are opt-in with `TASKS="prad_grade_group"`.

## Outputs

Each task/direction writes:

```text
visual_perturbation_by_run.csv
visual_perturbation_per_cell.csv
visual_perturbation_summary.csv
visual_perturbation_paired_by_run.csv
visual_perturbation_paired_summary.csv
visual_perturbation_summary.json
visual_perturbation_summary.{png,pdf,svg}
```

Use `visual_perturbation_paired_summary.csv` for the compact paper statement:
positive `mean_naive_minus_ours_mean_abs_rgb` means the naive baseline perturbs
the image more than our method.

## Border Inconsistency Metric

The border metric measures whether generated images introduce artificial hard
tile boundaries. For each internal tile boundary, compare adjacent pixels across
the boundary:

```text
seam(image, boundary) =
  mean_pixels_on_boundary(mean_rgb_channels(abs(image_right_or_bottom - image_left_or_top)))
```

Because real tissue can naturally have edges, the reported paper metric is
source-normalized:

```text
seam_excess = seam(after_image, boundary) - seam(source_image, boundary)
```

Interpretation:

- `seam_excess <= 0`: the after-image is no more discontinuous than the source
  at that boundary.
- `seam_excess > 0`: the edit introduced extra boundary discontinuity.
- positive `naive_minus_ours_seam_excess_mean_abs_rgb` means the naive baseline
  has stronger artificial borders than our progressive method.

The script reports this over:

- `all_tile_boundaries`: every internal 256 px UNI tile boundary,
- `selected_cell_perimeter`: boundaries between selected and unselected cells,
- `edited_cell_perimeter`: boundaries between edited and unedited cells,
- `visited_cell_perimeter`: boundaries between visited and unvisited cells,
- `committed_window_edges`: PixCell commit-window edges from `run_manifest.json`.

Additional output files:

```text
border_inconsistency_by_run.csv
border_inconsistency_summary.csv
border_inconsistency_paired_by_run.csv
border_inconsistency_paired_summary.csv
```
