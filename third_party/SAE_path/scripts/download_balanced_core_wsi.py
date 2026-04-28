#!/usr/bin/env python3
"""
Download WSI files for the TCGA Balanced Core subset.

Reads balanced_core_slides.tsv, filters by split, and downloads each slide via
utils.wsi_downloader.download_one.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Allow `python scripts/...` execution without manual PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.wsi_downloader import download_one


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--slides_tsv",
        type=Path,
        default=Path("metadata/manifests/tcga_balanced_core_90_10/balanced_core_slides.tsv"),
        help="Balanced Core slide table TSV.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "test"],
        help="Subset split to download.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("wsi/tcga_balanced_core"),
        help="Destination folder for downloaded .svs files.",
    )
    ap.add_argument(
        "--index_json",
        type=Path,
        default=Path("metadata/indexes/manifest_index.json"),
        help="Manifest index containing GDC UUID mapping.",
    )
    ap.add_argument(
        "--gdc_client",
        type=Path,
        default=Path("gdc/gdc-client"),
        help="Path to gdc-client executable.",
    )
    ap.add_argument(
        "--token",
        type=Path,
        default=None,
        help="Optional GDC token file.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel slide downloads (threads).",
    )
    ap.add_argument(
        "--n_processes",
        type=int,
        default=4,
        help="gdc-client -n per slide.",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retries per slide on failure.",
    )
    ap.add_argument("--latest", action="store_true", help="Pass --latest to gdc-client.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing slide output.")
    ap.add_argument("--max_slides", type=int, default=0, help="Optional cap for quick test runs.")
    ap.add_argument("--quiet", action="store_true", help="Less per-slide logging.")
    return ap.parse_args()


def load_slide_keys(slides_tsv: Path, split: str) -> List[str]:
    rows = []
    with slides_tsv.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if split != "all" and row.get("split") != split:
                continue
            key = str(row.get("slide_key", "")).strip()
            if key:
                rows.append(key)
    # deterministic dedupe while preserving order
    seen = set()
    uniq = []
    for key in rows:
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")
    if args.n_processes <= 0:
        raise SystemExit("--n_processes must be > 0")
    if args.retries < 0:
        raise SystemExit("--retries must be >= 0")

    slide_keys = load_slide_keys(args.slides_tsv, args.split)
    if args.max_slides and args.max_slides > 0:
        slide_keys = slide_keys[: args.max_slides]
    if not slide_keys:
        raise SystemExit(f"No slide keys found for split={args.split} in {args.slides_tsv}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    done = 0
    ok = 0
    failed = []

    def run_one(slide_key: str):
        last_err = None
        for attempt in range(args.retries + 1):
            try:
                out_path = download_one(
                    slide=slide_key,
                    index_json=args.index_json,
                    out_dir=out_dir,
                    gdc_client=args.gdc_client,
                    n_processes=args.n_processes,
                    token_path=args.token,
                    latest=bool(args.latest),
                    overwrite=bool(args.overwrite),
                    skip_if_exists=not bool(args.overwrite),
                    verbose=not bool(args.quiet),
                )
                return {"slide_key": slide_key, "ok": True, "path": str(out_path), "attempt": attempt + 1}
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                if attempt < args.retries:
                    time.sleep(1.0)
        return {"slide_key": slide_key, "ok": False, "error": last_err}

    print(
        f"[start] slides={len(slide_keys)} split={args.split} out_dir={out_dir} "
        f"workers={args.workers} gdc_n={args.n_processes}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, k): k for k in slide_keys}
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res["ok"]:
                ok += 1
            else:
                failed.append(res)
            if done % 10 == 0 or done == len(slide_keys):
                print(f"[progress] {done}/{len(slide_keys)} ok={ok} fail={len(failed)}", flush=True)

    elapsed = time.time() - t0
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slides_tsv": str(args.slides_tsv.resolve()),
        "split": args.split,
        "out_dir": str(out_dir),
        "index_json": str(args.index_json.resolve()),
        "workers": args.workers,
        "n_processes": args.n_processes,
        "retries": args.retries,
        "latest": bool(args.latest),
        "overwrite": bool(args.overwrite),
        "counts": {
            "requested": len(slide_keys),
            "ok": ok,
            "failed": len(failed),
        },
        "failed": failed,
        "elapsed_sec": elapsed,
    }
    out_summary = out_dir / f"download_summary_{args.split}.json"
    out_summary.write_text(json.dumps(summary, indent=2))

    print(f"[done] ok={ok} failed={len(failed)} elapsed={elapsed:.1f}s", flush=True)
    print(f"[ok] wrote {out_summary}", flush=True)

    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
