from __future__ import annotations

import sys
from pathlib import Path


WSI_CF_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = WSI_CF_ROOT / "src"


def _detect_legacy_repo_root() -> Path:
    candidates = [
        WSI_CF_ROOT.parent / "SAE_path",
        WSI_CF_ROOT.parent,
    ]
    for candidate in candidates:
        if (candidate / "utils").exists() and (candidate / "models").exists():
            return candidate
    return candidates[0]


LEGACY_REPO_ROOT = _detect_legacy_repo_root()


def ensure_legacy_repo_root_on_path() -> Path:
    """Allow Stage 1 modules to import vetted legacy helpers during migration."""
    path = str(LEGACY_REPO_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return LEGACY_REPO_ROOT
