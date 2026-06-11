from __future__ import annotations

import json
from pathlib import Path
from typing import Any


WSI_CF_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = WSI_CF_ROOT / "src"
RESOURCES_ROOT = WSI_CF_ROOT / "resources"
TASKS_ROOT = RESOURCES_ROOT / "tasks"

DEFAULT_TASK = "hnscc_hpv"
DEFAULT_TASK_CONFIG = TASKS_ROOT / f"{DEFAULT_TASK}.json"

DEFAULT_SAE_VARIANT = "tcga_uni2_sae_relu_v1"
SAE_VARIANTS = {
    "tcga_uni2_sae_relu_v1": {
        "checkpoint": RESOURCES_ROOT / "models/sae/tcga_uni2_sae_relu_v1/relu_final.pt",
        "config": RESOURCES_ROOT / "models/sae/tcga_uni2_sae_relu_v1/run_config.json",
        "description": "Current default TCGA UNI2 ReLU SAE trained at 20x.",
    },
    "tcga_sae_batch_topk_20x_interp": {
        "checkpoint": RESOURCES_ROOT / "models/sae/tcga_sae_batch_topk_20x_interp/batch_topk_final.pt",
        "config": RESOURCES_ROOT / "models/sae/tcga_sae_batch_topk_20x_interp/run_config.json",
        "description": "TCGA UNI2 BatchTopK SAE trained at 20x for interpretability/concept extraction.",
    },
    "relu_sae_base": {
        "checkpoint": RESOURCES_ROOT / "models/sae/relu_sae_base/relu_final.pt",
        "config": RESOURCES_ROOT / "models/sae/relu_sae_base/run_config.json",
        "description": "Legacy ReLU SAE kept for reproducing earlier experiments.",
    },
}
DEFAULT_SAE_CKPT = SAE_VARIANTS[DEFAULT_SAE_VARIANT]["checkpoint"]
DEFAULT_SAE_CFG = SAE_VARIANTS[DEFAULT_SAE_VARIANT]["config"]

DEFAULT_HNSCC_SPLIT_TSV = RESOURCES_ROOT / "manifests/hnsc_hpv_5fold/split_0.tsv"
DEFAULT_HNSCC_SPLIT_JSON = RESOURCES_ROOT / "manifests/hnsc_hpv_5fold/split_0.json"
DEFAULT_HNSCC_PROTOTYPE_NPZ = RESOURCES_ROOT / "prototypes/hnscc_hpv/prototype_vectors_for_selected_sae.npz"
DEFAULT_HNSCC_PROTOTYPE_JSON = RESOURCES_ROOT / "prototypes/hnscc_hpv/prototype_vectors_for_selected_sae.json"

DEFAULT_HNSCC_MIL_CKPT = RESOURCES_ROOT / "models/classifiers/hnscc_hpv/mil_split0.pt"
DEFAULT_HNSCC_CLAM_CKPT = RESOURCES_ROOT / "models/classifiers/hnscc_hpv/clam_split0.pt"
DEFAULT_HNSCC_CLAM_DATASET_CSV = RESOURCES_ROOT / "models/classifiers/hnscc_hpv/HNSCC.csv"
DEFAULT_HNSCC_CLAM_SPLITS_CSV = RESOURCES_ROOT / "models/classifiers/hnscc_hpv/splits_0.csv"

# External data inputs. These are intentionally not vendored because they are large slide/feature stores.
DEFAULT_HNSCC_SLIDES_DIR = Path("/common/users/wq50/HNSCC/HNSCC_slides")
DEFAULT_HNSCC_FEATURES_ROOT = Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2")
DEFAULT_HNSCC_CLAM_FEATURES_PT_DIR = Path("/common/users/wq50/CLAM/features/HPV_UNI2_features/pt_files")
DEFAULT_HNSCC_CLAM_COORDS_H5_DIR = Path("/common/users/wq50/CLAM/HNSCC_cases/patches")
DEFAULT_HNSCC_CLAM_SLIDES_DIR = Path("/common/users/wq50/CLAM/HNSCC_slides")

DEFAULT_SHOWCASE_REGION_IMAGE = (
    WSI_CF_ROOT
    / "artifacts/showcase_regions/TCGA-P3-A5QE-01Z-00-DX1_top_right_2048/region_top_right_2048.png"
)
DEFAULT_SHOWCASE_EDIT_MANIFEST = RESOURCES_ROOT / "examples/showcase_hnscc_hpv/progressive_edit_manifest.json"
DEFAULT_SHOWCASE_OUT_DIR = WSI_CF_ROOT / "artifacts/showcase_progressive_edit_hnscc_hpv"


def resource_path(relative: str | Path) -> Path:
    """Resolve a resource path relative to the repository root when needed."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return WSI_CF_ROOT / path


def resolve_sae_paths(
    variant: str | None = None,
    checkpoint: str | Path | None = None,
    config: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve an SAE checkpoint/config pair from a named variant or explicit paths."""
    if checkpoint is not None or config is not None:
        if checkpoint is None or config is None:
            raise ValueError("Both SAE checkpoint and config must be provided when overriding paths.")
        return resource_path(checkpoint), resource_path(config)

    variant_name = variant or DEFAULT_SAE_VARIANT
    if variant_name not in SAE_VARIANTS:
        choices = ", ".join(sorted(SAE_VARIANTS))
        raise ValueError(f"Unknown SAE variant {variant_name!r}. Available variants: {choices}")
    record = SAE_VARIANTS[variant_name]
    return resource_path(record["checkpoint"]), resource_path(record["config"])


def read_task_config(task: str | Path = DEFAULT_TASK) -> dict[str, Any]:
    """Load a task config from ``resources/tasks`` or from an explicit JSON path."""
    task_path = Path(task)
    if not task_path.suffix:
        task_path = TASKS_ROOT / f"{task}.json"
    if not task_path.is_absolute():
        task_path = WSI_CF_ROOT / task_path
    with task_path.open("r") as f:
        config = json.load(f)
    config["task_config_path"] = str(task_path)
    return config
