# Getting started

[← Documentation](README.md) · [Project home](../README.md)

Start with the illustrated [pipeline walkthrough](../README.md#how-it-works)
to understand region selection, concept steering, progressive generation, and
prediction evaluation before running the code.

## Get the repository

Install Git and Git LFS, then clone the project:

```bash
git lfs install
git clone https://github.com/W8Yi/wsi-cf.git
cd wsi-cf
```

Model checkpoints are stored with Git LFS and can make the initial download
large. To browse the code and figures first, skip checkpoint downloads:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/W8Yi/wsi-cf.git
cd wsi-cf
```

Download model weights later with `git lfs pull`.

## Prepare the research environment

Generation uses the project's `pace` conda environment in the original research
setup. The repository does not currently provide a standalone environment lock
file. Use the configured research environment with PyTorch/CUDA, OpenSlide,
UNI, and the PixCell dependencies before running generation.

The editor defaults to `StonyBrook-CVLab/PixCell-1024` and
`StonyBrook-CVLab/PixCell-pipeline`. Model access and any additional encoder
dependencies must be available in your environment.

Large raw slides and precomputed feature stores are external. Default lab paths
are centralized in [`src/wsi_cf/common/paths.py`](../src/wsi_cf/common/paths.py)
and [`resources/tasks/`](../resources/tasks/). Configure the paths for your data
before using task-specific examples.

## Prepare the inputs

The progressive editor expects:

- A **region bank CSV** pointing to a source region image and its aligned UNI
  feature grid.
- An **edit manifest** specifying the region and target grid cells.
- The matching **SAE checkpoint and configuration**, plus the task resources
  needed for the chosen steering direction.

The default showcase script references the prepared
`TCGA-P3-A5QE-01Z-00-DX1_top_right_2048_attention_only_top23_smooth4_prune28`
bundle under `artifacts/showcase_regions/`. That exact bundle is not included
in the repository. Make it available at the referenced location, or pass your
own region bank and manifest directly to
[`scripts/run_progressive_region_edit.py`](../scripts/run_progressive_region_edit.py).
The [progressive editing tutorial](PROGRESSIVE_EDIT_TUTORIAL.md) describes the
input formats and preparation workflow.

## Run the showcase

With the research environment active and the prepared showcase inputs in place:

```bash
PYTHON=python \
DEVICE=cuda:0 \
SAE_CKPT=resources/models/sae/relu_sae_base/relu_final.pt \
SAE_CFG=resources/models/sae/relu_sae_base/run_config.json \
bash examples/hnscc_hpv_showcase_smoothed28/01_run_progressive_edit.sh
```

This uses the legacy `relu_sae_base` SAE required by the historical HNSCC
prototype bundle. The repository's general default SAE is
`tcga_uni2_sae_relu_v1`; the historical prototypes should not be paired with it
without regenerating the prototype bundle.

The showcase writes outputs under
`artifacts/hnscc_hpv_showcase_smoothed28_tutorial/progressive/`. To also run the
matched baseline and compute comparison metrics, substitute `run_all.sh` for
`01_run_progressive_edit.sh` above.

See the [showcase walkthrough](../examples/hnscc_hpv_showcase_smoothed28/README.md)
for policy settings and expected outputs, or the
[reproduction reference](REPRODUCTION.md) for the original lab commands.
