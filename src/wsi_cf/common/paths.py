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

DEFAULT_SAE_CKPT = RESOURCES_ROOT / "models/sae/relu_sae_base/relu_final.pt"
DEFAULT_SAE_CFG = RESOURCES_ROOT / "models/sae/relu_sae_base/run_config.json"

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
