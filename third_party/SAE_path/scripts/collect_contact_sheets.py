#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Collect per-axis contact sheets (axis_xxxxxx/{pos,neg}/contact_sheet.png) "
            "into flat pos/ and neg/ folders with axis-labeled filenames."
        )
    )
    ap.add_argument("--src-root", type=Path, required=True, help="Export root (e.g. outputs/uni_axis_top_tiles_export/test_top20)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output folder containing pos/ and neg/ subfolders")
    ap.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="Whether to copy files or create symlinks (default: copy).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing collected files.")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    src_root = args.src_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not src_root.exists():
        raise SystemExit(f"Source root not found: {src_root}")

    collected = 0
    skipped = 0

    for sign in ("pos", "neg"):
        (out_dir / sign).mkdir(parents=True, exist_ok=True)

    for p in sorted(src_root.glob("axis_*/[pn][oe][sg]/contact_sheet.png")):
        axis_dir = p.parent.parent.name  # axis_000123
        sign = p.parent.name             # pos / neg
        if sign not in {"pos", "neg"}:
            continue
        dst = out_dir / sign / f"{axis_dir}__{sign}.png"

        if dst.exists():
            if not args.overwrite:
                skipped += 1
                continue
            if dst.is_symlink() or dst.is_file():
                dst.unlink()

        if args.mode == "copy":
            shutil.copy2(p, dst)
        else:
            dst.symlink_to(p)
        collected += 1

    print(f"Collected: {collected}")
    print(f"Skipped:   {skipped}")
    print(f"Output:    {out_dir}")


if __name__ == "__main__":
    main()
