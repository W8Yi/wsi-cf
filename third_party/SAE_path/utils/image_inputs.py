from __future__ import annotations

from pathlib import Path


def gather_image_paths(image: str | None, image_dir: str | None) -> list[Path]:
    """
    Resolve either a single image path or a directory of images into a file list.

    This keeps CLI scripts small and ensures consistent validation/messages across
    image-processing entrypoints.
    """
    if image and image_dir:
        raise ValueError("Use only one of --image or --image-dir")
    if image:
        return [Path(image)]
    if not image_dir:
        raise ValueError("Either --image or --image-dir is required")

    base = Path(image_dir)
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(base.glob(pat))
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No images found in {image_dir}")
    return files
