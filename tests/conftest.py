from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = TESTS_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_script_module(script_name: str):
    script_path = WSI_CF_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"wsi_cf_script_{script_name.replace('.', '_')}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
