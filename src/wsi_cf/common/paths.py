from __future__ import annotations

import sys
from pathlib import Path


WSI_CF_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = WSI_CF_ROOT / "src"
VENDORED_SAE_ROOT = WSI_CF_ROOT / "third_party" / "SAE_path"
EXTERNAL_SAE_ROOT = WSI_CF_ROOT.parent / "SAE_path"


def _looks_like_sae_root(path: Path) -> bool:
    return (path / "utils").exists() and (path / "models").exists()


def _detect_legacy_repo_root() -> Path:
    candidates = [
        VENDORED_SAE_ROOT,
        EXTERNAL_SAE_ROOT,
        WSI_CF_ROOT.parent,
    ]
    for candidate in candidates:
        if _looks_like_sae_root(candidate):
            return candidate
    return candidates[0]


LEGACY_REPO_ROOT = _detect_legacy_repo_root()
DEFAULT_SAE_ROOT = LEGACY_REPO_ROOT
DEFAULT_SAE_CKPT = DEFAULT_SAE_ROOT / "runs/relu_sae_base/relu_final.pt"
DEFAULT_SAE_CFG = DEFAULT_SAE_ROOT / "runs/relu_sae_base/run_config.json"
DEFAULT_HNSCC_SPLIT_TSV = DEFAULT_SAE_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.tsv"
DEFAULT_HNSCC_SPLIT_JSON = DEFAULT_SAE_ROOT / "metadata/manifests/hnsc_hpv_5fold/split_0.json"
DEFAULT_HNSCC_PROTOTYPE_NPZ = (
    DEFAULT_SAE_ROOT / "outputs/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"
)


def ensure_legacy_repo_root_on_path() -> Path:
    """Allow Stage 1 modules to import vetted legacy helpers during migration."""
    path = str(LEGACY_REPO_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return LEGACY_REPO_ROOT
