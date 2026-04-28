from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def validate_min_image_size(width: int, height: int, min_size: int = 256) -> None:
    """Fail early when an input image is too small for the downstream pipeline."""
    if width < min_size or height < min_size:
        raise ValueError(f"Input image is too small: {(width, height)}. Minimum size is {min_size}x{min_size}.")


def validate_min_output_size(height: int, width: int, min_size: int = 256) -> None:
    """PixCell generation assumes at least 256x256 output dimensions."""
    if height < min_size or width < min_size:
        raise ValueError(f"Output size must be >= {min_size} in both dimensions, got {(height, width)}.")


def save_image_tensor_01(img_t: torch.Tensor, out_path: Path) -> None:
    """
    Save a tensor image in [0, 1] with shape [1, 3, H, W] as an RGB PNG/JPEG.

    The helper keeps script code focused on pipeline logic instead of repeated
    tensor-to-PIL conversion boilerplate.
    """
    arr = (img_t[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def pil_to_nchw_float01(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a [1, 3, H, W] float tensor in [0, 1]."""
    arr = np.asarray(img).transpose(2, 0, 1).copy()
    return torch.from_numpy(arr).unsqueeze(0).float() / 255.0
