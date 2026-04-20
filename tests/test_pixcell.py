from __future__ import annotations

import pytest
import torch

from wsi_cf.generation.pixcell import (
    align_uncond_embedding,
    infer_pixcell_cond_grid_side,
    infer_pixcell_native_patch_px,
    resolve_pixcell_window_config,
)


def test_pixcell_config_inference() -> None:
    assert infer_pixcell_native_patch_px("StonyBrook-CVLab/PixCell-256") == 256
    assert infer_pixcell_cond_grid_side("StonyBrook-CVLab/PixCell-256") == 1
    assert infer_pixcell_native_patch_px("StonyBrook-CVLab/PixCell-1024") == 1024
    assert infer_pixcell_cond_grid_side("StonyBrook-CVLab/PixCell-1024") == 4
    assert resolve_pixcell_window_config(
        pix_model_id="StonyBrook-CVLab/PixCell-1024",
        patch_px=0,
        stride_px=0,
    ) == (1024, 512, 4)


def test_align_uncond_embedding_supports_single_and_matching_tokens() -> None:
    cond = torch.zeros((2, 16, 1536), dtype=torch.float32)
    uncond_single = torch.zeros((2, 1, 1536), dtype=torch.float32)
    aligned = align_uncond_embedding(uncond_single, cond)
    assert aligned.shape == (2, 16, 1536)

    uncond_matching = torch.zeros((2, 16, 1536), dtype=torch.float32)
    assert align_uncond_embedding(uncond_matching, cond).shape == (2, 16, 1536)

    with pytest.raises(RuntimeError):
        align_uncond_embedding(torch.zeros((2, 4, 1536)), cond)
