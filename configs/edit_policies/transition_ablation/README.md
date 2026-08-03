# Transition Ablation Edit Policies

These policies are intentionally small one-region visual ablations for testing
how to reduce rectangular PixCell/grid-cell artifacts.

They are meant to be run with:

```bash
examples/transition_ablation/01_run_kirc_one_region_transition_ablation.sh
```

The key rule for this folder is that the JSON policy should be the source of
truth. The experiment wrapper should not pass planning/commit overrides unless
you are deliberately overriding the policy.

Policy families:

- `01_support_only_current.json`: current support-cell commit baseline.
- `02_support_feather96.json`: softens support-cell edges with Gaussian feather.
- `03_soft_commit_halo_a035.json`: blends generated one-cell context halo at 0.35.
- `04_soft_commit_halo_a060.json`: stronger context-halo blend.
- `05_full_window_commit.json`: commits the whole generated 4x4 window.
- `06_full_window_strong_context_preserve.json`: whole-window commit with more context preservation.
- `07_soft_commit_halo_lower_steer.json`: softer steer with context halo commit.
- `08_full_window_lower_steer.json`: softer steer with whole-window commit.
- `09_full_window_regen_center_preserve_outer.json`: whole-window commit that also treats the full center 2x2 as editable in the preservation map, while strongly preserving the outer 4x4 context.
