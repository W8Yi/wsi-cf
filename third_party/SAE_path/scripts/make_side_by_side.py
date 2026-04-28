#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a side-by-side before/after image with labels.")
    ap.add_argument("--before", type=str, required=True)
    ap.add_argument("--after", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--before-label", type=str, default="before")
    ap.add_argument("--after-label", type=str, default="after")
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--top-pad", type=int, default=20)
    ap.add_argument("--bg", type=str, default="255,255,255")
    args = ap.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    out_path = Path(args.out)

    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")

    w1, h1 = before.size
    w2, h2 = after.size

    bg_rgb = tuple(int(v) for v in args.bg.split(","))
    canvas = Image.new("RGB", (w1 + args.gap + w2, max(h1, h2) + args.top_pad), bg_rgb)
    canvas.paste(before, (0, args.top_pad))
    canvas.paste(after, (w1 + args.gap, args.top_pad))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((2, 2), args.before_label, fill=(0, 0, 0), font=font)
    draw.text((w1 + args.gap + 2, 2), args.after_label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
