#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# match suffixes like _001 or _001_001 at the end
GRID_SUFFIX_RE = re.compile(r"^(.*?)(?:_\d{3}(?:_\d{3})?)?$", re.IGNORECASE)
# match trailing "-ext 2" / "_ext2" / " ext 2"
EXT_RE = re.compile(r"^(.*?)(?:[\s_\-]*ext\s*\d+)?$", re.IGNORECASE)


def canonicalize(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = EXT_RE.match(s).group(1).strip()
    m = GRID_SUFFIX_RE.match(s)
    return m.group(1) if m else s


def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def build_h5_index(h5_dir: Path) -> Dict[str, Path]:
    """
    Index by:
      - exact stem
      - canonical(stem) (only if unique)
    """
    exact: Dict[str, Path] = {}
    canon_to_paths: Dict[str, List[Path]] = {}

    for p in h5_dir.glob("*.h5"):
        stem = p.stem
        exact[stem] = p
        canon_to_paths.setdefault(canonicalize(stem), []).append(p)

    merged = dict(exact)
    for c, paths in canon_to_paths.items():
        if c not in merged and len(paths) == 1:
            merged[c] = paths[0]
    return merged


def resolve_to_h5(sid: str, idx: Dict[str, Path]) -> Optional[Path]:
    sid = str(sid).strip()
    if sid in idx:
        return idx[sid]
    c = canonicalize(sid)
    if c in idx:
        return idx[c]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True, type=Path)
    ap.add_argument("--h5_dir", required=True, type=Path)
    ap.add_argument("--out_json", required=True, type=Path)
    ap.add_argument("--require_resolve", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.split_csv)

    id_col = df.columns[0]
    cols = {c.lower(): c for c in df.columns[1:]}
    if not {"train", "val", "test"}.issubset(cols.keys()):
        raise ValueError(f"Expected columns train,val,test; got {list(df.columns)}")

    idx = build_h5_index(args.h5_dir)

    out = {"train": [], "val": [], "test": []}
    missing = []

    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()

        targets = []
        if parse_bool(row[cols["train"]]): targets.append("train")
        if parse_bool(row[cols["val"]]):   targets.append("val")
        if parse_bool(row[cols["test"]]):  targets.append("test")
        if not targets:
            continue

        p = resolve_to_h5(sid, idx)
        if p is None:
            missing.append(sid)
            continue

        for t in targets:
            out[t].append(str(p))

    payload = {
        "meta": {
            "split_csv": str(args.split_csv),
            "h5_dir": str(args.h5_dir),
            "counts": {k: len(v) for k, v in out.items()},
            "missing_count": len(missing),
        },
        "train": out["train"],
        "val": out["val"],
        "test": out["test"],
        "missing_ids": missing,
    }

    if args.require_resolve and missing:
        raise RuntimeError(
            f"{len(missing)} slide ids could not be resolved. First few: {missing[:10]}"
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["meta"], indent=2))
    if missing:
        print(f"[warn] missing ids written to manifest: {len(missing)}")


if __name__ == "__main__":
    main()
