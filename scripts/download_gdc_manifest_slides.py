#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import requests


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download public GDC slide files from a prepared GDC manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=0, help="Maximum files to download; 0 means all manifest rows.")
    parser.add_argument("--chunk-mb", type=int, default=4)
    parser.add_argument("--verify-md5", action=argparse.BooleanOptionalAction, default=True)
    return parser


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_is_valid(path: Path, expected_size: int, expected_md5: str, verify_md5: bool) -> bool:
    if not path.exists() or path.stat().st_size != expected_size:
        return False
    return not verify_md5 or not expected_md5 or md5sum(path).lower() == expected_md5.lower()


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if int(args.max_files) > 0:
        rows = rows[: int(args.max_files)]
    if not rows:
        raise ValueError(f"No GDC file rows in {args.manifest}")

    downloaded = 0
    reused = 0
    total_bytes = sum(int(row.get("size") or 0) for row in rows)
    print(f"[plan] files={len(rows)} expected_size_gb={total_bytes / 1e9:.2f} out_dir={args.out_dir}", flush=True)
    for index, row in enumerate(rows, start=1):
        file_id = str(row["id"])
        filename = str(row["filename"])
        expected_size = int(row.get("size") or 0)
        expected_md5 = str(row.get("md5") or "")
        slide_key = filename.split(".")[0]
        out_dir = args.out_dir / slide_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        if file_is_valid(out_path, expected_size, expected_md5, bool(args.verify_md5)):
            reused += 1
            print(f"[skip {index}/{len(rows)}] {filename}", flush=True)
            continue
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"[download {index}/{len(rows)}] {filename} size_gb={expected_size / 1e9:.2f}", flush=True)
        with requests.get(f"https://api.gdc.cancer.gov/data/{file_id}", stream=True, timeout=(60, 600)) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=max(1, int(args.chunk_mb)) * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if not file_is_valid(tmp_path, expected_size, expected_md5, bool(args.verify_md5)):
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file failed size or MD5 verification: {filename}")
        tmp_path.rename(out_path)
        downloaded += 1
    print(f"[ok] downloaded={downloaded} reused={reused} total={len(rows)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
