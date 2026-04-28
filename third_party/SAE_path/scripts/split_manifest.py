#!/usr/bin/env python3
"""
make_tcga_patient_split.py

Create TCGA patient-level splits (train/test = 0.9/0.1) across ALL TCGA cancers.
- Split is patient-level (TCGA-XX-YYYY).
- Stratified *within each cancer folder* so every cancer contributes ~10% patients to test.
- Outputs JSON manifests listing feature file paths for train/test, plus a CSV summary.

Assumptions:
- Feature files are .h5/.hdf5 under:
    /common/users/wq50/UNI2_features/extracted_features/TCGA-*
- Each feature filename stem contains a TCGA barcode like:
    TCGA-AB-1234-...
  We extract patient_id = "TCGA-AB-1234" (first 12 chars).
"""

from pathlib import Path
import re
import json
import hashlib
from collections import defaultdict
import pandas as pd

ROOT = Path("/common/users/wq50/UNI2_features/extracted_features")
OUT_JSON = ROOT / "sae_manifests_tcga_patient_train_test_90_10.json"
OUT_CSV  = ROOT / "sae_tcga_patient_split_summary.csv"

TEST_FRAC = 0.10
SUPPORTED_EXTS = (".h5", ".hdf5")

# TCGA patient id pattern: TCGA-XX-YYYY (2 alnum, 4 alnum)
TCGA_PAT_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def stable_hash_to_float(s: str) -> float:
    """Deterministic hash -> [0,1)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 16**8


def extract_patient_id(stem: str) -> str | None:
    """Extract TCGA patient id like 'TCGA-AB-1234' from filename stem."""
    m = TCGA_PAT_RE.search(stem)
    if not m:
        return None
    # normalize to uppercase for consistency
    return m.group(1).upper()


def list_feature_files(folder: Path):
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(folder.rglob(f"*{ext}"))
    return sorted(files)


def main():
    tcga_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("TCGA-")])
    if not tcga_dirs:
        raise SystemExit(f"No TCGA-* directories found under {ROOT}")

    # Per-cancer mapping: patient_id -> list[filepath]
    cancer_to_patient_files: dict[str, dict[str, list[str]]] = {}

    total_files = 0
    missing_patient = []

    for cancer_dir in tcga_dirs:
        patient_map = defaultdict(list)
        files = list_feature_files(cancer_dir)
        total_files += len(files)

        for p in files:
            pid = extract_patient_id(p.stem)
            if pid is None:
                missing_patient.append(str(p))
                continue
            patient_map[pid].append(str(p))

        cancer_to_patient_files[cancer_dir.name] = dict(patient_map)

    if missing_patient:
        print(f"[WARN] {len(missing_patient)} files missing TCGA patient barcode. "
              f"They will be excluded. Example:\n  {missing_patient[0]}")

    # Split patient IDs within each cancer deterministically
    train_files, test_files = [], []
    summary_rows = []

    for cancer, patient_map in sorted(cancer_to_patient_files.items()):
        patient_ids = sorted(patient_map.keys())
        if not patient_ids:
            continue

        # Deterministic assignment: hash(patient_id) < TEST_FRAC -> test
        test_pids = [pid for pid in patient_ids if stable_hash_to_float(pid) < TEST_FRAC]
        train_pids = [pid for pid in patient_ids if pid not in set(test_pids)]

        # Collect file paths
        for pid in train_pids:
            train_files.extend(patient_map[pid])
        for pid in test_pids:
            test_files.extend(patient_map[pid])

        # Summary
        n_pat = len(patient_ids)
        n_test_pat = len(test_pids)
        n_train_pat = len(train_pids)
        n_train_files = sum(len(patient_map[pid]) for pid in train_pids)
        n_test_files = sum(len(patient_map[pid]) for pid in test_pids)

        summary_rows.append({
            "cancer": cancer,
            "patients_total": n_pat,
            "patients_train": n_train_pat,
            "patients_test": n_test_pat,
            "files_total": n_train_files + n_test_files,
            "files_train": n_train_files,
            "files_test": n_test_files,
            "test_frac_patients": (n_test_pat / n_pat) if n_pat else 0.0,
        })

    # Save manifests
    manifests = {
        "tcga_train": sorted(train_files),
        "tcga_test": sorted(test_files),
        "meta": {
            "root": str(ROOT),
            "split": "patient_level_within_each_cancer",
            "train_test_ratio": "0.9/0.1",
            "test_frac_rule": f"md5(patient_id) < {TEST_FRAC}",
            "counts": {
                "tcga_train_files": len(train_files),
                "tcga_test_files": len(test_files),
                "excluded_files_missing_patient_id": len(missing_patient),
                "total_feature_files_scanned": total_files,
            },
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(manifests, f, indent=2)

    df = pd.DataFrame(summary_rows).sort_values("patients_total", ascending=False)
    df.to_csv(OUT_CSV, index=False)

    print("Saved manifests:", OUT_JSON)
    print("Saved summary:  ", OUT_CSV)
    print(json.dumps(manifests["meta"], indent=2))


if __name__ == "__main__":
    main()