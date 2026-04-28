#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.common.paths import DEFAULT_HNSCC_SPLIT_TSV
from wsi_cf.data.slides import open_slide


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure exact GDC diagnostic SVS files exist for HNSCC attention visualization. "
            "The exact SVS filename is inferred from split_0.tsv h5_path, so downloaded slides "
            "match the feature H5 coordinate frame."
        )
    )
    parser.add_argument("--split-tsv", type=Path, default=DEFAULT_HNSCC_SPLIT_TSV)
    parser.add_argument("--features-root", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"))
    parser.add_argument("--slides-dir", type=Path, default=Path("/common/users/wq50/HNSCC/test"))
    parser.add_argument("--out-dir", type=Path, default=Path("/common/users/wq50/wsi_cf/artifacts/gdc_slide_downloads"))
    parser.add_argument("--split", type=str, default="test", help="Split to ensure, or 'all'.")
    parser.add_argument("--slide-key", action="append", default=[], help="Optional slide key filter. Can be passed multiple times.")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--max-slides", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    return parser


def read_requested_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    requested_slide_keys = {str(item) for item in args.slide_key}
    rows: list[dict[str, Any]] = []
    with args.split_tsv.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            slide_key = str(row.get("slide_key", "")).strip()
            if not slide_key:
                continue
            if requested_slide_keys and slide_key not in requested_slide_keys:
                continue
            if str(args.split).lower() != "all" and str(row.get("split", "")) != str(args.split):
                continue
            label = int(row.get("label", -1))
            if label not in (0, 1):
                continue
            if args.label is not None and label != int(args.label):
                continue
            h5_path = args.features_root / f"{slide_key}.h5"
            if not h5_path.exists():
                continue
            legacy_h5_name = Path(str(row.get("h5_path", ""))).name
            if legacy_h5_name.endswith(".h5"):
                svs_filename = legacy_h5_name[:-3] + ".svs"
            else:
                svs_filename = f"{slide_key}.svs"
            item = dict(row)
            item["label"] = label
            item["feature_h5_path"] = str(h5_path)
            item["svs_filename"] = svs_filename
            rows.append(item)
    rows.sort(key=lambda item: (int(item["label"]), str(item["slide_key"])))
    if int(args.max_slides) > 0:
        rows = rows[: int(args.max_slides)]
    return rows


def query_gdc_files(filenames: list[str], *, timeout: int) -> dict[str, dict[str, Any]]:
    if not filenames:
        return {}
    filters = {"op": "in", "content": {"field": "file_name", "value": sorted(set(filenames))}}
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,data_type,data_format,experimental_strategy,cases.project.project_id,cases.submitter_id",
        "format": "JSON",
        "size": str(max(1, len(set(filenames)))),
    }
    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=int(timeout))
    response.raise_for_status()
    hits = response.json()["data"]["hits"]
    return {str(hit["file_name"]): hit for hit in hits}


def read_h5_coord_extent(h5_path: Path) -> tuple[int, int, int]:
    with h5py.File(h5_path, "r") as handle:
        coords = handle["coords"][:]
        if coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords[0]
        coords = np.asarray(coords, dtype=np.int64)
        tile_size = infer_coord_tile_size(coords, fallback=512)
        extent_x = int(coords[:, 0].max()) + int(tile_size)
        extent_y = int(coords[:, 1].max()) + int(tile_size)
    return int(extent_x), int(extent_y), int(tile_size)


def infer_coord_tile_size(coords: np.ndarray, *, fallback: int) -> int:
    arr = np.asarray(coords, dtype=np.int64)
    candidates: list[int] = []
    for axis in (0, 1):
        vals = np.unique(arr[:, axis])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            candidates.append(int(np.min(diffs)))
    if not candidates:
        return int(fallback)
    return max(1, int(min(candidates)))


def check_slide_matches(slide_path: Path, *, h5_path: Path) -> tuple[bool, str, dict[str, Any]]:
    try:
        extent_x, extent_y, tile_size = read_h5_coord_extent(h5_path)
        slide = open_slide(slide_path)
        slide_w, slide_h = slide.dimensions
        slide.close()
    except Exception as exc:
        return False, f"check_failed:{exc}", {}
    ok = extent_x <= int(slide_w) * 1.05 and extent_y <= int(slide_h) * 1.05
    meta = {
        "h5_coord_extent_x": int(extent_x),
        "h5_coord_extent_y": int(extent_y),
        "h5_coord_tile_size": int(tile_size),
        "slide_width": int(slide_w),
        "slide_height": int(slide_h),
    }
    if ok:
        return True, "coords_match_slide_frame", meta
    return False, f"h5_coord_extent_{extent_x}x{extent_y}_exceeds_svs_{slide_w}x{slide_h}", meta


def download_gdc_file(file_id: str, out_path: Path, *, expected_size: int, timeout: int, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if force and out_path.exists():
        out_path.unlink()
    if out_path.exists() and out_path.stat().st_size == int(expected_size):
        return
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if force and tmp.exists():
        tmp.unlink()
    start = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start > 0 else {}
    mode = "ab" if start > 0 else "wb"
    with requests.get(f"{GDC_DATA_ENDPOINT}/{file_id}", stream=True, timeout=int(timeout), headers=headers) as response:
        response.raise_for_status()
        with tmp.open(mode) as handle:
            downloaded = start
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded // (250 * 1024 * 1024) != (downloaded - len(chunk)) // (250 * 1024 * 1024):
                    print(f"  {out_path.name}: {downloaded / 1024 / 1024:.1f} MiB", flush=True)
    tmp.rename(out_path)
    actual = out_path.stat().st_size
    if actual != int(expected_size):
        raise RuntimeError(f"Downloaded size mismatch for {out_path}: got {actual}, expected {expected_size}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()
    args.slides_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_requested_rows(args)
    if not rows:
        raise RuntimeError("No rows matched the requested filters and available H5 features.")
    gdc = query_gdc_files([str(row["svs_filename"]) for row in rows], timeout=int(args.timeout))

    manifest_rows: list[dict[str, Any]] = []
    total_gdc_size = 0
    for idx, row in enumerate(rows, start=1):
        svs_filename = str(row["svs_filename"])
        hit = gdc.get(svs_filename)
        slide_path = args.slides_dir / svs_filename
        status = "missing_from_gdc"
        reason = "missing_from_gdc"
        match_meta: dict[str, Any] = {}
        expected_size = int(hit.get("file_size", 0)) if hit else 0
        total_gdc_size += expected_size
        if hit is not None:
            needs_download = bool(args.force_redownload) or not slide_path.exists() or slide_path.stat().st_size != expected_size
            if not needs_download and slide_path.exists():
                ok, reason, match_meta = check_slide_matches(slide_path, h5_path=Path(str(row["feature_h5_path"])))
                needs_download = not ok
            if needs_download and bool(args.download):
                print(f"[{idx}/{len(rows)}] download {svs_filename}", flush=True)
                download_gdc_file(
                    str(hit["file_id"]),
                    slide_path,
                    expected_size=expected_size,
                    timeout=int(args.timeout),
                    force=bool(args.force_redownload),
                )
            if slide_path.exists():
                ok, reason, match_meta = check_slide_matches(slide_path, h5_path=Path(str(row["feature_h5_path"])))
                status = "matched" if ok else "mismatch"
            else:
                status = "missing_local"
        manifest_rows.append(
            {
                "slide_key": str(row["slide_key"]),
                "case_id": str(row.get("case_id", "")),
                "split": str(row.get("split", "")),
                "label": int(row["label"]),
                "svs_filename": svs_filename,
                "slide_path": str(slide_path),
                "feature_h5_path": str(row["feature_h5_path"]),
                "gdc_file_id": str(hit.get("file_id", "")) if hit else "",
                "gdc_file_size": int(expected_size),
                "status": status,
                "reason": reason,
                **match_meta,
            }
        )
    manifest_csv = args.out_dir / f"gdc_slide_manifest_{args.split}.csv"
    write_csv(manifest_csv, manifest_rows)
    summary = {
        "split": str(args.split),
        "n_requested": len(rows),
        "n_matched": sum(1 for row in manifest_rows if row["status"] == "matched"),
        "n_mismatch": sum(1 for row in manifest_rows if row["status"] == "mismatch"),
        "n_missing_gdc": sum(1 for row in manifest_rows if row["status"] == "missing_from_gdc"),
        "n_missing_local": sum(1 for row in manifest_rows if row["status"] == "missing_local"),
        "total_gdc_size_bytes": int(total_gdc_size),
        "total_gdc_size_gib": float(total_gdc_size / (1024**3)),
        "slides_dir": str(args.slides_dir),
        "manifest_csv": str(manifest_csv),
    }
    write_json(args.out_dir / f"gdc_slide_summary_{args.split}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["n_matched"] != len(rows):
        raise RuntimeError("Not all requested slides are matched. See manifest CSV for details.")


if __name__ == "__main__":
    main()
