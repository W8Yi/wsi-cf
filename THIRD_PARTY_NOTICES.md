# Third-Party Notices

This repository includes model definitions that were migrated from prior local SAE, MIL, and CLAM research code used by the project team. They are now maintained as first-class `wsi_cf` modules:

- `src/wsi_cf/models/sae.py`
- `src/wsi_cf/models/mil.py`
- `src/wsi_cf/models/clam.py`
- `src/wsi_cf/steering/sae_runtime.py`
- `src/wsi_cf/steering/sae_edit.py`

Model weights, labels, manifests, and prototype vectors are stored as versioned project resources under `resources/`. Large raw slide data and precomputed feature stores remain external inputs and are not vendored into this repository.
