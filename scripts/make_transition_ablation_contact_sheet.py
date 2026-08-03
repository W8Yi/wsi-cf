#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a contact sheet for one-region transition ablations.")
    parser.add_argument("--out-root", type=Path, required=True, help="Root containing one subdirectory per policy.")
    parser.add_argument("--run-id", type=str, required=True, help="Run directory name inside each policy output.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None, help="Optional source image to place first.")
    parser.add_argument("--image-name", type=str, default="generated.png")
    parser.add_argument("--thumb-width", type=int, default=512)
    parser.add_argument("--cols", type=int, default=3)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    items: list[tuple[str, Path]] = []
    if args.source is not None:
        items.append(("source", args.source))
    for policy_dir in sorted(path for path in args.out_root.iterdir() if path.is_dir()):
        image_path = policy_dir / args.run_id / args.image_name
        if image_path.exists():
            items.append((policy_dir.name, image_path))
    if not items:
        raise ValueError(f"No images found under {args.out_root} for run_id={args.run_id}")

    thumbs: list[tuple[str, Image.Image]] = []
    for label, image_path in items:
        image = Image.open(image_path).convert("RGB")
        ratio = int(args.thumb_width) / float(image.width)
        thumbs.append((label, image.resize((int(args.thumb_width), int(round(image.height * ratio))), Image.Resampling.LANCZOS)))

    cols = max(1, int(args.cols))
    rows = (len(thumbs) + cols - 1) // cols
    pad = 12
    label_h = 44
    cell_h = max(image.height for _, image in thumbs) + label_h
    sheet = Image.new(
        "RGB",
        (cols * int(args.thumb_width) + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(22)
    for index, (label, image) in enumerate(thumbs):
        row, col = divmod(index, cols)
        x = pad + col * (int(args.thumb_width) + pad)
        y = pad + row * (cell_h + pad)
        draw.text((x + 4, y + 6), label, fill=(0, 0, 0), font=font)
        sheet.paste(image, (x, y + label_h))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()

