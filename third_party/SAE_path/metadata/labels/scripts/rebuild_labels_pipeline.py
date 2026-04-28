#!/usr/bin/env python3
"""
One-command rebuild for metadata/labels with validation and run metadata.

This script runs the label pipeline in order:
1) fetch raw sources
2) build cBio slide catalog
3) fetch GDC slide labels
4) build per-target + master labels
5) validate expected outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: List[str], cwd: Path) -> Dict[str, object]:
    t0 = time.time()
    print(f"[run] {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd), check=False)
    dt = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "duration_sec": round(dt, 3),
    }


def require_nonempty_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected output: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Empty expected output: {path}")


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    raise RuntimeError(f"Could not locate repo root from: {start}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_index", type=Path, default=Path("metadata/indexes/manifest_index.json"))
    ap.add_argument("--labels_root", type=Path, default=Path("metadata/labels"))
    ap.add_argument("--refresh_sources", action="store_true")
    ap.add_argument("--refresh_cbio_download", action="store_true")
    args = ap.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    raw_dir = args.labels_root / "sources" / "raw"
    inter_dir = args.labels_root / "sources" / "intermediate"
    qc_dir = args.labels_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    run_meta: Dict[str, object] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "manifest_index": str(args.manifest_index),
        "labels_root": str(args.labels_root),
        "commands": [],
    }

    try:
        git_head = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or ""
        )
    except Exception:
        git_head = ""
    if git_head:
        run_meta["git_head"] = git_head

    py = sys.executable or "python"
    commands: List[List[str]] = [
        [py, "metadata/labels/scripts/fetch_label_sources.py", "--out_dir", str(raw_dir)],
        [
            py,
            "metadata/labels/scripts/build_tcga_label_catalog.py",
            "--manifest_index",
            str(args.manifest_index),
            "--out_dir",
            str(inter_dir),
            "--raw_dir",
            str(raw_dir),
        ],
        [
            py,
            "metadata/labels/scripts/fetch_gdc_slide_labels.py",
            "--manifest_index",
            str(args.manifest_index),
            "--out_dir",
            str(inter_dir),
            "--batch_size",
            "300",
            "--sleep_sec",
            "0.05",
        ],
        [
            py,
            "metadata/labels/scripts/build_master_labels.py",
            "--manifest_index",
            str(args.manifest_index),
            "--cbio_slide_tsv",
            str(inter_dir / "tcga_slide_label_catalog.tsv"),
            "--gdc_slide_tsv",
            str(inter_dir / "tcga_gdc_slide_labels.tsv"),
            "--out_dir",
            str(args.labels_root),
        ],
    ]

    if args.refresh_sources:
        commands[0].append("--refresh")
    if args.refresh_cbio_download:
        commands[1].append("--refresh_cbio_download")

    for cmd in commands:
        run_meta["commands"].append(run_cmd(cmd, cwd=repo_root))

    expected_outputs = [
        args.labels_root / "targets" / "hpv_status_case.tsv",
        args.labels_root / "targets" / "immune_subtype_case.tsv",
        args.labels_root / "targets" / "msi_status_case.tsv",
        args.labels_root / "targets" / "pam50_case.tsv",
        args.labels_root / "targets" / "tumor_grade_case.tsv",
        args.labels_root / "targets" / "stage_case.tsv",
        args.labels_root / "targets" / "survival_case.tsv",
        args.labels_root / "targets" / "mutation_tp53_kras_case.tsv",
        args.labels_root / "targets" / "tumor_purity_case.tsv",
        args.labels_root / "master" / "case_labels_master.tsv",
        args.labels_root / "master" / "slide_labels_master.tsv",
        args.labels_root / "qc" / "coverage_by_target.tsv",
        args.labels_root / "qc" / "conflicts.tsv",
        args.labels_root / "qc" / "build_summary.json",
        raw_dir / "source_download_manifest.json",
    ]
    for p in expected_outputs:
        require_nonempty_file(p)

    run_meta["outputs"] = {
        str(p): {
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        }
        for p in expected_outputs
    }
    run_meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_meta["status"] = "ok"

    out_run_meta = qc_dir / "build_pipeline_run.json"
    out_run_meta.write_text(json.dumps(run_meta, indent=2))
    print(f"[ok] wrote {out_run_meta}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
