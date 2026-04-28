from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
LATENT_DIR_RE = re.compile(r"^latent_(\d+)$")
RUN_DIR_RE = re.compile(r"^(?P<mode>[a-z_]+)_strength_(?P<tag>[mp0-9]+)$")


@dataclass(frozen=True)
class SweepRun:
    latent_idx: int
    mode: str
    strength: float
    run_dir: Path


def _parse_strength_tag(tag: str) -> float:
    s = tag.replace("m", "-", 1) if tag.startswith("m") else tag
    s = s.replace("p", ".")
    return float(s)


def _parse_strength_csv(val: str) -> list[float]:
    if not val.strip():
        return []
    return [float(x.strip()) for x in val.split(",") if x.strip()]


def _gather_tiles(orig_dir: Path) -> list[Path]:
    files = sorted([p for p in orig_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not files:
        raise SystemExit(f"No tile images found in {orig_dir}")
    return files


def _discover_runs(sweep_root: Path) -> list[SweepRun]:
    runs: list[SweepRun] = []
    for latent_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir()]):
        m_lat = LATENT_DIR_RE.match(latent_dir.name)
        if not m_lat:
            continue
        latent_idx = int(m_lat.group(1))
        for run_dir in sorted([p for p in latent_dir.iterdir() if p.is_dir()]):
            m_run = RUN_DIR_RE.match(run_dir.name)
            if not m_run:
                continue
            runs.append(
                SweepRun(
                    latent_idx=latent_idx,
                    mode=m_run.group("mode"),
                    strength=_parse_strength_tag(m_run.group("tag")),
                    run_dir=run_dir,
                )
            )
    if not runs:
        raise SystemExit(f"No SAE sweep runs found under {sweep_root}")
    return runs


def _load_img_rgb(path: Path, thumb_size: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if thumb_size > 0 and im.size != (thumb_size, thumb_size):
        im = im.resize((thumb_size, thumb_size), resample=Image.BILINEAR)
    return im


def _abs_diff_img(orig: Image.Image, edit: Image.Image) -> Image.Image:
    if orig.size != edit.size:
        edit = edit.resize(orig.size, resample=Image.BILINEAR)
    return ImageChops.difference(orig, edit)


def _placeholder(size: tuple[int, int], text: str) -> Image.Image:
    im = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    d.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(180, 180, 180))
    d.text((6, 6), text, fill=(80, 80, 80), font=font)
    return im


def _make_labeled_grid(
    *,
    images: list[list[Image.Image]],
    row_labels: list[str],
    col_labels: list[str],
    out_path: Path,
    title: str,
    cell_gap: int = 6,
    header_h: int = 26,
    row_label_w: int = 180,
    title_h: int = 28,
) -> None:
    if not images or not images[0]:
        raise ValueError("Empty image grid")
    n_rows = len(images)
    n_cols = len(images[0])
    if any(len(row) != n_cols for row in images):
        raise ValueError("Inconsistent row lengths in grid")
    if len(row_labels) != n_rows or len(col_labels) != n_cols:
        raise ValueError("Label lengths do not match grid dimensions")

    cell_w, cell_h = images[0][0].size
    total_w = row_label_w + n_cols * cell_w + (n_cols - 1) * cell_gap
    total_h = title_h + header_h + n_rows * cell_h + (n_rows - 1) * cell_gap

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((8, 6), title, fill=(0, 0, 0), font=font)

    for c, label in enumerate(col_labels):
        x = row_label_w + c * (cell_w + cell_gap)
        draw.text((x + 4, title_h + 4), label, fill=(0, 0, 0), font=font)

    y_base = title_h + header_h
    for r, (row_label, row_imgs) in enumerate(zip(row_labels, images)):
        y = y_base + r * (cell_h + cell_gap)
        draw.text((8, y + 4), row_label, fill=(0, 0, 0), font=font)
        for c, im in enumerate(row_imgs):
            x = row_label_w + c * (cell_w + cell_gap)
            canvas.paste(im, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _find_edit_image(run_dir: Path, tile_stem: str) -> Path | None:
    p = run_dir / tile_stem / "sae_steer_edit.png"
    if p.exists():
        return p
    return None


def build_sae_sweep_summaries(
    *,
    orig_dir: Path,
    sweep_root: Path,
    out_dir: Path,
    tile_limit: int = 24,
    latent_limit: int = 0,
    strengths: list[float] | None = None,
    thumb_size: int = 192,
    include_diff: bool = False,
) -> dict[str, Any]:
    orig_tiles = _gather_tiles(orig_dir)
    if tile_limit and tile_limit > 0:
        orig_tiles = orig_tiles[: int(tile_limit)]

    runs = _discover_runs(sweep_root)

    # Group runs by latent.
    runs_by_latent: dict[int, dict[float, SweepRun]] = {}
    modes_seen: set[str] = set()
    for r in runs:
        modes_seen.add(r.mode)
        runs_by_latent.setdefault(r.latent_idx, {})[float(r.strength)] = r
    mode_name = sorted(modes_seen)[0] if modes_seen else "latent_delta"

    latent_ids = sorted(runs_by_latent.keys())
    if latent_limit and latent_limit > 0:
        latent_ids = latent_ids[: int(latent_limit)]

    discovered_strengths = sorted({s for lid in latent_ids for s in runs_by_latent[lid].keys()})
    strengths_use = sorted([s for s in (strengths or discovered_strengths) if s in set(discovered_strengths)])
    if not strengths_use:
        raise SystemExit("No matching strengths found for visualization.")

    out_dir.mkdir(parents=True, exist_ok=True)
    by_latent_dir = out_dir / "by_latent"
    by_tile_dir = out_dir / "by_tile"
    by_latent_diff_dir = out_dir / "by_latent_diff"
    by_tile_diff_dir = out_dir / "by_tile_diff"

    print(f"Tiles selected: {len(orig_tiles)}")
    print(f"Latents selected: {len(latent_ids)}")
    print(f"Strengths: {strengths_use}")

    orig_cache: dict[str, Image.Image] = {p.stem: _load_img_rgb(p, thumb_size) for p in orig_tiles}
    edit_cache: dict[tuple[int, str, float], Image.Image] = {}
    for lid in latent_ids:
        for s in strengths_use:
            run = runs_by_latent.get(lid, {}).get(s)
            if run is None:
                continue
            for tile_path in orig_tiles:
                p = _find_edit_image(run.run_dir, tile_path.stem)
                if p is not None:
                    edit_cache[(lid, tile_path.stem, s)] = _load_img_rgb(p, thumb_size)

    # By-latent: rows=tiles, cols=orig+strengths
    print("[1/3] Building by-latent grids...")
    col_labels = ["orig"] + [f"{s:+g}" for s in strengths_use]
    for lid in latent_ids:
        rows_rgb: list[list[Image.Image]] = []
        rows_diff: list[list[Image.Image]] = []
        row_labels: list[str] = []
        for tile_path in orig_tiles:
            stem = tile_path.stem
            orig_im = orig_cache[stem]
            row = [orig_im]
            row_diff = [_placeholder(orig_im.size, "n/a")]
            for s in strengths_use:
                edit_im = edit_cache.get((lid, stem, s))
                if edit_im is None:
                    row.append(_placeholder(orig_im.size, "missing"))
                    row_diff.append(_placeholder(orig_im.size, "missing"))
                else:
                    row.append(edit_im)
                    if include_diff:
                        row_diff.append(_abs_diff_img(orig_im, edit_im))
            rows_rgb.append(row)
            rows_diff.append(row_diff)
            row_labels.append(stem)

        _make_labeled_grid(
            images=rows_rgb,
            row_labels=row_labels,
            col_labels=col_labels,
            out_path=by_latent_dir / f"latent_{lid:06d}_{mode_name}.png",
            title=f"latent_{lid} | mode={mode_name} | rows=tiles | cols=orig+strengths",
        )
        if include_diff:
            _make_labeled_grid(
                images=rows_diff,
                row_labels=row_labels,
                col_labels=col_labels,
                out_path=by_latent_diff_dir / f"latent_{lid:06d}_{mode_name}_absdiff.png",
                title=f"latent_{lid} | ABS DIFF vs orig | mode={mode_name}",
            )

    # By-tile: rows=latents, cols=orig+strengths
    print("[2/3] Building by-tile grids...")
    for tile_path in orig_tiles:
        stem = tile_path.stem
        orig_im = orig_cache[stem]
        rows_rgb: list[list[Image.Image]] = []
        rows_diff: list[list[Image.Image]] = []
        row_labels: list[str] = []
        for lid in latent_ids:
            row = [orig_im]
            row_diff = [_placeholder(orig_im.size, "n/a")]
            for s in strengths_use:
                edit_im = edit_cache.get((lid, stem, s))
                if edit_im is None:
                    row.append(_placeholder(orig_im.size, "missing"))
                    row_diff.append(_placeholder(orig_im.size, "missing"))
                else:
                    row.append(edit_im)
                    if include_diff:
                        row_diff.append(_abs_diff_img(orig_im, edit_im))
            rows_rgb.append(row)
            rows_diff.append(row_diff)
            row_labels.append(f"latent_{lid}")

        _make_labeled_grid(
            images=rows_rgb,
            row_labels=row_labels,
            col_labels=col_labels,
            out_path=by_tile_dir / f"{stem}_{mode_name}.png",
            title=f"{stem} | mode={mode_name} | rows=latents | cols=orig+strengths",
        )
        if include_diff:
            _make_labeled_grid(
                images=rows_diff,
                row_labels=row_labels,
                col_labels=col_labels,
                out_path=by_tile_diff_dir / f"{stem}_{mode_name}_absdiff.png",
                title=f"{stem} | ABS DIFF vs orig | mode={mode_name}",
            )

    print("[3/3] Writing summary metadata...")
    summary = {
        "orig_dir": str(orig_dir),
        "sweep_root": str(sweep_root),
        "out_dir": str(out_dir),
        "mode": mode_name,
        "tile_count": len(orig_tiles),
        "latent_count": len(latent_ids),
        "strengths": strengths_use,
        "thumb_size": int(thumb_size),
        "include_diff": bool(include_diff),
        "tiles": [p.name for p in orig_tiles],
        "latents": latent_ids,
    }
    (out_dir / "summary_meta.json").write_text(json.dumps(summary, indent=2))
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build summary grids from SAE latent sweep outputs.")
    ap.add_argument("--orig-dir", type=Path, required=True)
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tile-limit", type=int, default=24)
    ap.add_argument("--latent-limit", type=int, default=0)
    ap.add_argument("--strengths", type=str, default="", help="Comma-separated strengths to include.")
    ap.add_argument("--thumb-size", type=int, default=192)
    ap.add_argument("--include-diff", action="store_true")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    strengths = _parse_strength_csv(args.strengths)
    summary = build_sae_sweep_summaries(
        orig_dir=args.orig_dir,
        sweep_root=args.sweep_root,
        out_dir=args.out_dir,
        tile_limit=int(args.tile_limit),
        latent_limit=int(args.latent_limit),
        strengths=strengths or None,
        thumb_size=int(args.thumb_size),
        include_diff=bool(args.include_diff),
    )
    print("Saved summary grids to:", summary["out_dir"])


if __name__ == "__main__":
    main()
