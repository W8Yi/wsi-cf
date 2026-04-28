#!/usr/bin/env python3
"""
wsi_downloader.py

Download TCGA WSIs using gdc-client + manifest_index_enriched.json where each slide key maps to a *single* GDC UUID.

Changes vs prior version:
- Assumes slide keys are unique -> no multi-candidate selection logic.
- Accepts multiple slides (CLI: many args or a text file with one slide per line).
- After download, moves the .svs/.tif into the output directory directly:
    out_dir/<SLIDE>.svs
  and removes the UUID folder created by gdc-client.

Index format expected (per slide key):
  enriched[slide_key]["gdc"] is either:
    - {"id": "...", "filename": "..."}   (dict)
    - [{"id": "...", "filename": "..." }] (list with one element)

Example CLI:
  python wsi_downloader.py TCGA-49-4488-01Z-00-DX1 TCGA-HT-7607-01Z-00-DX1 --out ./wsi

  python wsi_downloader.py --slides-file slides.txt --out ./wsi

Import usage:
  from wsi_downloader import download_many, download_one
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Slide parsing (must match your new unique slide key scheme)
# Supports DX10/DX11/... and TS1, etc.
# ----------------------------
SLIDE_RE = re.compile(
    r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]-\d{2}-(?:DX\d+|TS\d+))",
    re.IGNORECASE,
)

def parse_slide_key(x: str | Path) -> str:
    name = Path(x).name
    m = SLIDE_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse TCGA slide key from: {x}")
    return m.group(1).upper()


# ----------------------------
# Index loading
# ----------------------------
_INDEX_CACHE: Dict[Path, Dict[str, Any]] = {}

def load_enriched_index(index_json: Path) -> Dict[str, Any]:
    index_json = index_json.expanduser().resolve()
    if index_json not in _INDEX_CACHE:
        _INDEX_CACHE[index_json] = json.loads(index_json.read_text())
    return _INDEX_CACHE[index_json]

@dataclass(frozen=True)
class GdcRef:
    uuid: str
    filename: str

def _extract_single_gdc_ref(entry: Dict[str, Any], slide_key: str) -> GdcRef:
    gdc = entry.get("gdc")

    if gdc is None:
        raise KeyError(f"{slide_key}: missing 'gdc' field in enriched index entry")

    # Allow dict or single-element list
    if isinstance(gdc, dict):
        uuid = str(gdc.get("id", "")).strip()
        fname = str(gdc.get("filename", "")).strip()
        if not uuid or not fname:
            raise KeyError(f"{slide_key}: invalid gdc dict (need id+filename)")
        return GdcRef(uuid=uuid, filename=fname)

    if isinstance(gdc, list):
        if len(gdc) != 1:
            raise RuntimeError(
                f"{slide_key}: expected exactly 1 gdc candidate after uniqueness fix, got {len(gdc)}"
            )
        item = gdc[0]
        if not isinstance(item, dict):
            raise RuntimeError(f"{slide_key}: gdc[0] is not a dict")
        uuid = str(item.get("id", "")).strip()
        fname = str(item.get("filename", "")).strip()
        if not uuid or not fname:
            raise KeyError(f"{slide_key}: invalid gdc entry (need id+filename)")
        return GdcRef(uuid=uuid, filename=fname)

    raise RuntimeError(f"{slide_key}: unsupported gdc type: {type(gdc)}")


# ----------------------------
# Filesystem helpers
# ----------------------------
def _assert_executable(p: Path) -> Path:
    p = p.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"gdc-client not found: {p}")
    if not p.is_file():
        raise RuntimeError(f"gdc-client path is not a file: {p}")
    return p

def _find_downloaded_wsi(uuid_dir: Path) -> Path:
    """
    gdc-client downloads into out_dir/<UUID>/.../<FILE>
    In practice the file is usually directly under the UUID folder, but be robust.
    """
    if not uuid_dir.exists():
        raise FileNotFoundError(f"Expected UUID folder not found: {uuid_dir}")

    # Search for common WSI extensions
    exts = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
    hits: List[Path] = []
    for p in uuid_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            hits.append(p)

    if not hits:
        # If you also download .tar.gz or other packaging, you can extend here
        raise FileNotFoundError(f"No WSI file found under {uuid_dir}")

    if len(hits) > 1:
        # Pick the largest file if multiple WSIs appear (rare)
        hits.sort(key=lambda x: x.stat().st_size, reverse=True)
    return hits[0]

def _safe_move(src: Path, dst: Path, *, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    shutil.move(str(src), str(dst))


# ----------------------------
# Core download functions (importable)
# ----------------------------
def resolve_uuid_from_index(slide: str | Path, *, index_json: Path) -> GdcRef:
    slide_key = parse_slide_key(slide)
    idx = load_enriched_index(index_json)
    entry = idx.get(slide_key)
    if entry is None:
        raise KeyError(f"{slide_key}: not found in {index_json}")
    return _extract_single_gdc_ref(entry, slide_key)

def download_one(
    slide: str | Path,
    *,
    index_json: Path,
    out_dir: Path,
    gdc_client: Path,
    n_processes: int = 1,
    token_path: Optional[Path] = None,
    latest: bool = False,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Downloads one slide WSI and returns the final saved path out_dir/<SLIDE>.<ext>

    Behavior:
    - calls gdc-client download <uuid> -d out_dir [--latest] [-t token]
    - moves the downloaded WSI out of out_dir/<uuid>/... to out_dir/<slide>.<ext>
    - deletes the out_dir/<uuid> folder
    """
    slide_key = parse_slide_key(slide)
    ref = resolve_uuid_from_index(slide_key, index_json=index_json)

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose final extension from GDC filename
    ext = Path(ref.filename).suffix
    if not ext:
        # fallback
        ext = ".svs"
    final_path = out_dir / f"{slide_key}{ext}"

    if skip_if_exists and final_path.exists() and not overwrite:
        if verbose:
            print(f"[skip] {slide_key} exists: {final_path}")
        return final_path

    exe = _assert_executable(gdc_client)

    uuid_dir = out_dir / ref.uuid
    # Clean stale partial folders from interrupted previous runs.
    if uuid_dir.exists():
        shutil.rmtree(uuid_dir, ignore_errors=True)

    cmd = [str(exe), "download", ref.uuid, "-d", str(out_dir)]
    if int(n_processes) > 1:
        cmd += ["-n", str(int(n_processes))]
    if latest:
        cmd.append("--latest")
    if token_path is not None:
        cmd += ["-t", str(token_path)]

    if verbose:
        print(f"Slide: {slide_key}")
        print(f"UUID:  {ref.uuid}")
        print(f"GDC filename: {ref.filename}")
        print("Running:", " ".join(cmd), flush=True)

    subprocess.run(cmd, check=True)

    downloaded = _find_downloaded_wsi(uuid_dir)

    # Move to out_dir/<slide>.<ext>
    _safe_move(downloaded, final_path, overwrite=overwrite)

    # Clean up UUID folder
    # (it may now be empty, but remove defensively)
    shutil.rmtree(uuid_dir, ignore_errors=True)

    if verbose:
        print(f"[ok] saved: {final_path}")

    return final_path

def download_many(
    slides: Iterable[str | Path],
    *,
    index_json: Path,
    out_dir: Path,
    gdc_client: Path,
    n_processes: int = 1,
    token_path: Optional[Path] = None,
    latest: bool = False,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> List[Path]:
    saved: List[Path] = []
    for s in slides:
        try:
            p = download_one(
                s,
                index_json=index_json,
                out_dir=out_dir,
                gdc_client=gdc_client,
                n_processes=n_processes,
                token_path=token_path,
                latest=latest,
                overwrite=overwrite,
                skip_if_exists=skip_if_exists,
                verbose=verbose,
            )
            saved.append(p)
        except Exception as e:
            # keep going; caller can inspect failures in logs
            if verbose:
                print(f"[fail] {s}: {e}")
    return saved


# ----------------------------
# CLI
# ----------------------------
def _read_slides_file(p: Path) -> List[str]:
    slides: List[str] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        slides.append(line)
    return slides

def main():
    ap = argparse.ArgumentParser(
        description="Download TCGA WSIs using metadata/indexes/manifest_index.json (unique slide keys)."
    )
    ap.add_argument("slides", nargs="*", help="Slide keys or filenames starting with the slide key (.h5/.svs).")
    ap.add_argument("--slides-file", type=Path, default=None, help="Text file with one slide per line.")
    ap.add_argument("--index", type=Path, default=Path("metadata/indexes/manifest_index.json"))
    ap.add_argument("--out", type=Path, default=Path("./wsi_download"))
    ap.add_argument(
        "--gdc-client",
        dest="gdc_client",
        type=Path,
        default=Path("/common/users/wq50/SAE_path/gdc/gdc-client"),
    )
    ap.add_argument("--token", type=Path, default=None)
    ap.add_argument("--n-processes", type=int, default=1, help="gdc-client connections per file download.")
    ap.add_argument("--latest", action="store_true", help="Enable --latest for gdc-client download.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing out_dir/<slide>.<ext>")
    ap.add_argument("--no-skip", action="store_true", help="Do not skip if output exists")
    ap.add_argument("--quiet", action="store_true", help="Less printing")
    args = ap.parse_args()

    slides: List[str] = []
    if args.slides_file is not None:
        slides.extend(_read_slides_file(args.slides_file))
    slides.extend(args.slides or [])

    if not slides:
        raise SystemExit("No slides provided. Use positional slides and/or --slides-file.")

    failed = 0
    for s in slides:
        try:
            download_one(
                s,
                index_json=args.index,
                out_dir=args.out,
                gdc_client=args.gdc_client,
                n_processes=max(1, int(args.n_processes)),
                token_path=args.token,
                latest=bool(args.latest),
                overwrite=args.overwrite,
                skip_if_exists=not args.no_skip,
                verbose=not args.quiet,
            )
        except Exception as e:
            failed += 1
            print(f"[fail] {s}: {e}", flush=True)
    if failed > 0:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
