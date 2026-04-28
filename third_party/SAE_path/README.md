# Vendored SAE Resources

This directory is a hard copy of the SAE resources used by the WSI counterfactual
experiments so the repository can run without depending on a sibling
`/common/users/wq50/SAE_path` checkout.

Included resources:

- `metadata/labels/`: task label tables and slide-level metadata used for
  concept-label association experiments.
- `metadata/manifests/`: split files and feature manifests used by HNSCC HPV
  and TCGA SAE workflows.
- `models/`, `utils/`, `concept_steer/`, `scripts/`: SAE-side helper code used
  by the current steering and concept association scripts.
- `runs/relu_sae_base/relu_final.pt`: default ReLU SAE checkpoint.
- `runs/relu_sae_base/run_config.json`: default SAE config.
- `outputs/sae_prototypes/hnscc_hpv_split0_selected/`: selected HPV concept
  prototype vectors used by the current steering pipeline.

The repo path resolver in `src/wsi_cf/common/paths.py` prefers this vendored
directory and only falls back to the external SAE checkout if the vendored copy
is missing.

Large downstream classifier checkpoints are not included here. Scripts that need
MIL or CLAM checkpoints still expose explicit `--mil-ckpt` or `--clam-ckpt`
arguments.
