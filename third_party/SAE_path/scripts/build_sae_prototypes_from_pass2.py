from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_steer.build_sae_prototypes_from_pass2 import main


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _WSI_SAE_SRC = Path(__file__).resolve().parents[2] / "wsi-sae" / "src"
    if _WSI_SAE_SRC.exists():
        sys.path.insert(0, str(_WSI_SAE_SRC))
        from wsi_sae.cli import main as _wsi_sae_main

        _wsi_sae_main(["build-prototypes", *sys.argv[1:]])
    else:
        main()
