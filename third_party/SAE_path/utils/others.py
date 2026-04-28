from __future__ import annotations

"""
Compatibility shim for legacy imports.

Prefer importing from:
- utils.clam
- utils.uni
- utils.sae
- utils.sae_edit
"""


def load_clam_mb(*args, **kwargs):
    from utils.clam import load_clam_mb as _impl

    return _impl(*args, **kwargs)


def clam_predict_bag(*args, **kwargs):
    from utils.clam import clam_predict_bag as _impl

    return _impl(*args, **kwargs)


def get_uni(*args, **kwargs):
    from utils.uni import get_uni as _impl

    return _impl(*args, **kwargs)


def load_sae_from_config(*args, **kwargs):
    from utils.sae import load_sae_from_config as _impl

    return _impl(*args, **kwargs)


def sae_encode_features(*args, **kwargs):
    from utils.sae import sae_encode_features as _impl

    return _impl(*args, **kwargs)


def _sae_decode_latents(*args, **kwargs):
    from utils.sae import sae_decode_latents as _impl

    return _impl(*args, **kwargs)


def edit_uni_z_grid_with_sae(*args, **kwargs):
    from utils.sae_edit import edit_uni_z_grid_with_sae as _impl

    return _impl(*args, **kwargs)
