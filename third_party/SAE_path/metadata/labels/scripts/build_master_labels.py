#!/usr/bin/env python3
"""
Build per-target case labels and a unified case/slide master label table.

Outputs (under metadata/labels by default):
- targets/*.tsv
- master/case_labels_master.tsv
- master/slide_labels_master.tsv
- qc/coverage_by_target.tsv
- qc/conflicts.tsv
- qc/build_summary.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple


def is_nonempty(v: object) -> bool:
    s = str(v or "").strip()
    return s != "" and s.lower() not in {"na", "nan", "none", "null", "not reported", "[not available]"}


def maybe_float(v: object) -> Optional[float]:
    if not is_nonempty(v):
        return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def mode_with_conflict(values: Iterable[str]) -> Tuple[str, bool]:
    vals = [str(v).strip() for v in values if is_nonempty(v)]
    if not vals:
        return "", False
    c = Counter(vals)
    top = c.most_common()
    selected = top[0][0]
    conflict = len(c) > 1
    return selected, conflict


def mode_float(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(median(vals))


def tsv_rows(path: Path) -> List[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name}: {path}\n"
            f"Run the source fetch/build scripts first."
        )


def parse_case_id_from_slide(slide_key: str) -> str:
    toks = str(slide_key).split("-")
    if len(toks) < 3:
        return ""
    return "-".join(toks[:3]).upper()


def parse_sample_id_from_slide(slide_key: str) -> str:
    m = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2})", str(slide_key).upper())
    return m.group(1) if m else ""


def project_from_meta(meta: dict) -> str:
    ds = str(meta.get("dataset", "") or "").strip()
    if ds:
        return ds
    h5 = str(meta.get("h5_path", "") or "")
    m = re.search(r"/extracted_features/([^/]+)/", h5)
    return m.group(1) if m else ""


def normalize_os_status(v: str) -> str:
    s = str(v or "").strip().upper()
    if s in {"LIVING", "ALIVE"}:
        return "Alive"
    if s in {"DECEASED", "DEAD"}:
        return "Dead"
    if s:
        return s.title()
    return ""


def normalize_stage(v: str) -> str:
    s = str(v or "").strip()
    if not s:
        return ""
    u = s.upper()
    if u.startswith("STAGE "):
        return "Stage " + s[6:].strip()
    return s


def normalize_grade(v: str) -> str:
    s = str(v or "").strip()
    return s.upper() if s else ""


def normalize_immune_subtype(v: str) -> str:
    s = str(v or "").strip().upper()
    if not s or s in {"NA", "NAN", "NONE", "NULL"}:
        return ""
    if s in {"1", "2", "3", "4", "5", "6"}:
        return f"C{s}"
    if len(s) == 2 and s[0] == "C" and s[1] in {"1", "2", "3", "4", "5", "6"}:
        return s
    return s


def msi_from_subtype(sub: str) -> Optional[str]:
    s = str(sub or "").strip()
    if not s:
        return None
    if "MSI" in s:
        return "MSI"
    non_msi = {
        "COAD_CIN",
        "COAD_GS",
        "COAD_POLE",
        "READ_CIN",
        "READ_GS",
        "READ_POLE",
        "STAD_CIN",
        "STAD_GS",
        "STAD_EBV",
        "STAD_POLE",
    }
    if s in non_msi:
        return "NonMSI"
    return None


def msi_from_pancan_selected(v: str) -> Optional[str]:
    s = str(v or "").strip()
    if s in {"GI.HM-indel", "GI.HM-SNV"}:
        return "MSI"
    if s in {"GI.CIN", "GI.GS", "GI.EBV"}:
        return "NonMSI"
    return None


def pam50_from_cbio_subtype(sub: str) -> Optional[str]:
    s = str(sub or "").strip()
    mp = {
        "BRCA_LUMA": "LumA",
        "BRCA_LUMB": "LumB",
        "BRCA_BASAL": "Basal",
        "BRCA_HER2": "Her2",
        "BRCA_NORMAL": "Normal",
    }
    return mp.get(s.upper())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_index", type=Path, default=Path("metadata/indexes/manifest_index.json"))
    ap.add_argument(
        "--cbio_slide_tsv",
        type=Path,
        default=Path("metadata/labels/sources/intermediate/tcga_slide_label_catalog.tsv"),
    )
    ap.add_argument(
        "--gdc_slide_tsv",
        type=Path,
        default=Path("metadata/labels/sources/intermediate/tcga_gdc_slide_labels.tsv"),
    )
    ap.add_argument(
        "--cesc_hpv_tsv",
        type=Path,
        default=Path("metadata/labels/sources/curated/hpv/nature2017_cesc_hpv_consensus.tsv"),
    )
    ap.add_argument(
        "--hnsc_hpv_tsv",
        type=Path,
        default=Path("metadata/labels/sources/curated/hpv/nature2015_hnsc_hpv_status.tsv"),
    )
    ap.add_argument(
        "--pancan_subtype_tsv",
        type=Path,
        default=Path("metadata/labels/sources/raw/TCGASubtype.20170308.tsv"),
    )
    ap.add_argument(
        "--viral_tsv",
        type=Path,
        default=Path("metadata/labels/sources/raw/viral.tsv"),
    )
    ap.add_argument(
        "--purity_tsv",
        type=Path,
        default=Path("metadata/labels/sources/raw/TCGA_mastercalls.abs_tables_JSedit.fixed.txt"),
    )
    ap.add_argument(
        "--mutation_tsv",
        type=Path,
        default=Path("metadata/labels/sources/raw/cbio_mutation_tp53_kras.tsv"),
    )
    ap.add_argument(
        "--immune_tsv",
        type=Path,
        default=Path("metadata/labels/sources/raw/immune_subtype_mclust.tsv.gz"),
    )
    ap.add_argument("--out_dir", type=Path, default=Path("metadata/labels"))
    args = ap.parse_args()

    require_file(args.manifest_index, "manifest_index")
    require_file(args.cbio_slide_tsv, "cbio_slide_tsv")
    require_file(args.gdc_slide_tsv, "gdc_slide_tsv")
    require_file(args.cesc_hpv_tsv, "cesc_hpv_tsv")
    require_file(args.hnsc_hpv_tsv, "hnsc_hpv_tsv")
    require_file(args.pancan_subtype_tsv, "pancan_subtype_tsv")
    require_file(args.viral_tsv, "viral_tsv")
    require_file(args.purity_tsv, "purity_tsv")
    require_file(args.mutation_tsv, "mutation_tsv")
    require_file(args.immune_tsv, "immune_tsv")

    targets_dir = args.out_dir / "targets"
    master_dir = args.out_dir / "master"
    qc_dir = args.out_dir / "qc"
    for d in [targets_dir, master_dir, qc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest_index.read_text())
    cbio_by_slide = {r["slide_key"]: r for r in tsv_rows(args.cbio_slide_tsv)}
    gdc_by_slide = {r["slide_key"]: r for r in tsv_rows(args.gdc_slide_tsv)}

    cesc_official = {r["case_id"]: r["hpv_status"] for r in tsv_rows(args.cesc_hpv_tsv)}
    hnsc_official = {r["case_id"]: r["hpv_status"] for r in tsv_rows(args.hnsc_hpv_tsv)}

    subtype_case: Dict[str, dict] = {}
    for r in tsv_rows(args.pancan_subtype_tsv):
        rr = {k.strip('"'): (v or "").strip('"') for k, v in r.items()}
        sid = rr.get("pan.samplesID", "")
        m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", sid.upper())
        if not m:
            continue
        subtype_case[m.group(1)] = rr

    viral_score: Dict[str, float] = {}
    for r in tsv_rows(args.viral_tsv):
        case = str(r.get("ParticipantBarcode", "")).strip().upper()
        v = maybe_float(r.get("HPV", ""))
        if case and v is not None:
            viral_score[case] = v

    purity_by_sample: Dict[str, float] = {}
    for r in tsv_rows(args.purity_tsv):
        sample = str(r.get("array", "")).strip().upper()
        p = maybe_float(r.get("purity", ""))
        if sample and p is not None:
            purity_by_sample[sample] = p

    mutation_by_sample: Dict[str, dict] = {}
    for r in tsv_rows(args.mutation_tsv):
        sample = str(r.get("sample_id", "")).strip().upper()
        if not sample:
            continue
        mutation_by_sample[sample] = r

    immune_by_sample: Dict[str, List[str]] = defaultdict(list)
    immune_by_case: Dict[str, List[str]] = defaultdict(list)
    for r in tsv_rows(args.immune_tsv):
        sample_barcode = str(r.get("SampleBarcode", "")).strip().upper().replace(".", "-")
        m = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2})", sample_barcode)
        sample_id = m.group(1) if m else ""
        case_id = "-".join(sample_id.split("-")[:3]) if sample_id else ""
        subtype = normalize_immune_subtype(r.get("ClusterModel1", "")) or normalize_immune_subtype(r.get("ClusterModel2", ""))
        if not subtype:
            continue
        if sample_id:
            immune_by_sample[sample_id].append(subtype)
        if case_id:
            immune_by_case[case_id].append(subtype)

    cases: Dict[str, dict] = {}
    for slide_key, meta in manifest.items():
        case_id = str(meta.get("patient_id", "") or parse_case_id_from_slide(slide_key)).upper()
        sample_id = parse_sample_id_from_slide(slide_key)
        proj = project_from_meta(meta)
        row = cases.setdefault(
            case_id,
            {
                "case_id": case_id,
                "slide_keys": [],
                "sample_ids": set(),
                "projects": [],
            },
        )
        row["slide_keys"].append(slide_key)
        if sample_id:
            row["sample_ids"].add(sample_id)
        if proj:
            row["projects"].append(proj)

    if not cases:
        raise RuntimeError("No cases parsed from manifest.")

    conflicts: List[dict] = []
    master_cases: List[dict] = []

    for case_id, ctx in sorted(cases.items()):
        slide_keys = ctx["slide_keys"]
        sample_ids = sorted(ctx["sample_ids"])
        project_dir, project_conflict = mode_with_conflict(ctx["projects"])
        if project_conflict:
            conflicts.append(
                {
                    "case_id": case_id,
                    "target": "project_dir",
                    "detail": f"multiple project_dir values: {sorted(set(ctx['projects']))}",
                }
            )

        cbio_subtypes = []
        cbio_grades = []
        cbio_stages = []
        cbio_os_statuses = []
        cbio_os_months = []
        cbio_pfs_statuses = []
        cbio_pfs_months = []
        gdc_grades = []
        gdc_stages = []
        gdc_vitals = []
        gdc_projects = []

        for sk in slide_keys:
            cb = cbio_by_slide.get(sk, {})
            gd = gdc_by_slide.get(sk, {})
            if is_nonempty(cb.get("cbio_subtype", "")):
                cbio_subtypes.append(str(cb["cbio_subtype"]).strip())
            if is_nonempty(cb.get("cbio_histologic_grade", "")):
                cbio_grades.append(normalize_grade(cb["cbio_histologic_grade"]))
            if is_nonempty(cb.get("cbio_ajcc_stage", "")):
                cbio_stages.append(normalize_stage(cb["cbio_ajcc_stage"]))
            if is_nonempty(cb.get("cbio_os_status", "")):
                cbio_os_statuses.append(normalize_os_status(cb["cbio_os_status"]))
            om = maybe_float(cb.get("cbio_os_months", ""))
            if om is not None:
                cbio_os_months.append(om)
            if is_nonempty(cb.get("cbio_pfs_status", "")):
                cbio_pfs_statuses.append(str(cb["cbio_pfs_status"]).strip().upper())
            pm = maybe_float(cb.get("cbio_pfs_months", ""))
            if pm is not None:
                cbio_pfs_months.append(pm)
            if is_nonempty(gd.get("gdc_tumor_grade", "")):
                gdc_grades.append(normalize_grade(gd["gdc_tumor_grade"]))
            if is_nonempty(gd.get("gdc_ajcc_pathologic_stage", "")):
                gdc_stages.append(normalize_stage(gd["gdc_ajcc_pathologic_stage"]))
            if is_nonempty(gd.get("gdc_vital_status", "")):
                gdc_vitals.append(normalize_os_status(gd["gdc_vital_status"]))
            if is_nonempty(gd.get("gdc_project_id", "")):
                gdc_projects.append(str(gd["gdc_project_id"]).strip())

        gdc_project_id, gdc_project_conf = mode_with_conflict(gdc_projects)
        if (not is_nonempty(project_dir)) and is_nonempty(gdc_project_id):
            # Fall back to GDC project when the project cannot be inferred from h5_path.
            project_dir = gdc_project_id
        if gdc_project_conf:
            conflicts.append(
                {
                    "case_id": case_id,
                    "target": "gdc_project_id",
                    "detail": f"multiple gdc_project_id values: {sorted(set(gdc_projects))}",
                }
            )

        cbio_subtype_mode, cbio_subtype_conf = mode_with_conflict(cbio_subtypes)

        # HPV
        hpv_status = "Unknown"
        hpv_source = "not_applicable"
        hpv_conflict = 0
        if project_dir == "TCGA-CESC":
            hpv_source = "nature2017_cesc"
            if case_id in cesc_official:
                hpv_status = cesc_official[case_id]
            else:
                hpv_status = "Unknown"
                hpv_source = "nature2017_cesc_missing_case"
        elif project_dir == "TCGA-HNSC":
            hpv_source = "cbio_subtype_hnsc"
            mp = {"HNSC_HPV+": "HPV+", "HNSC_HPV-": "HPV-"}
            vals = [mp[s] for s in cbio_subtypes if s in mp]
            if vals:
                hpv_status, c = mode_with_conflict(vals)
                hpv_conflict = int(c)
            elif case_id in hnsc_official:
                hpv_status = hnsc_official[case_id]
                hpv_source = "nature2015_hnsc_fallback"
            else:
                hpv_status = "Unknown"
                hpv_source = "cbio_subtype_hnsc_missing_case"
            if case_id in hnsc_official and hpv_status in {"HPV+", "HPV-"} and hpv_status != hnsc_official[case_id]:
                hpv_conflict = 1
                conflicts.append(
                    {
                        "case_id": case_id,
                        "target": "hpv_status",
                        "detail": f"cbio({hpv_status}) != official_hnsc({hnsc_official[case_id]})",
                    }
                )
        hpv_score = None
        hpv_score_pred = ""
        if project_dir in {"TCGA-CESC", "TCGA-HNSC"}:
            hpv_score = viral_score.get(case_id)
            if hpv_score is not None:
                hpv_score_pred = "HPV+" if hpv_score >= 6.0 else "HPV-"

        # Immune subtype (C1-C6) from mclust ensemble table.
        immune_subtype = "Unknown"
        immune_source = "missing"
        immune_conflict = 0
        vals = []
        for s in sample_ids:
            vals.extend(immune_by_sample.get(s, []))
        if not vals:
            vals = immune_by_case.get(case_id, [])
        if vals:
            immune_subtype, c = mode_with_conflict(vals)
            immune_conflict = int(c)
            immune_source = "immune_subtype_mclust_cluster_model1"

        # MSI
        msi_status = "Unknown"
        msi_source = "not_applicable"
        msi_conflict = 0
        if project_dir in {"TCGA-COAD", "TCGA-STAD"}:
            msi_source = "cbio_subtype"
            vals = [msi_from_subtype(s) for s in cbio_subtypes]
            vals = [v for v in vals if v is not None]
            if vals:
                msi_status, c = mode_with_conflict(vals)
                msi_conflict = int(c)
            else:
                pan_sel = subtype_case.get(case_id, {}).get("Subtype_Selected", "")
                pv = msi_from_pancan_selected(pan_sel)
                if pv is not None:
                    msi_status = pv
                    msi_source = "pancan_subtype_selected"
                else:
                    msi_status = "Unknown"
                    msi_source = "missing"

        # PAM50 (BRCA)
        pam50 = "Unknown"
        pam50_source = "not_applicable"
        pam50_conflict = 0
        if project_dir in {"TCGA-BRCA_IDC", "TCGA-BRCA_OTHERS", "TCGA-BRCA"}:
            vals = [pam50_from_cbio_subtype(s) for s in cbio_subtypes]
            vals = [v for v in vals if v is not None]
            pam50_source = "cbio_subtype"
            if vals:
                pam50, c = mode_with_conflict(vals)
                pam50_conflict = int(c)
            else:
                pam50 = "Unknown"
                pam50_source = "missing"

        # Grade
        grade = "Unknown"
        grade_source = "missing"
        grade_conflict = 0
        if gdc_grades:
            grade, c = mode_with_conflict(gdc_grades)
            grade_conflict = int(c)
            grade_source = "gdc_tumor_grade"
        elif cbio_grades:
            grade, c = mode_with_conflict(cbio_grades)
            grade_conflict = int(c)
            grade_source = "cbio_histologic_grade"

        # Stage
        stage = "Unknown"
        stage_source = "missing"
        stage_conflict = 0
        if gdc_stages:
            stage, c = mode_with_conflict(gdc_stages)
            stage_conflict = int(c)
            stage_source = "gdc_ajcc_pathologic_stage"
        elif cbio_stages:
            stage, c = mode_with_conflict(cbio_stages)
            stage_conflict = int(c)
            stage_source = "cbio_ajcc_stage"

        # Survival
        os_status = "Unknown"
        survival_source = "missing"
        survival_conflict = 0
        if cbio_os_statuses:
            os_status, c = mode_with_conflict(cbio_os_statuses)
            survival_conflict = int(c)
            survival_source = "cbio_os_status"
        elif gdc_vitals:
            os_status, c = mode_with_conflict(gdc_vitals)
            survival_conflict = int(c)
            survival_source = "gdc_vital_status"
        os_months = mode_float(cbio_os_months)
        pfs_status, pfs_conf = mode_with_conflict(cbio_pfs_statuses)
        pfs_months = mode_float(cbio_pfs_months)
        if pfs_conf:
            survival_conflict = 1

        # Mutations
        mut_rows = [mutation_by_sample[s] for s in sample_ids if s in mutation_by_sample]
        mutation_has_data = any(str(r.get("has_mutation_data", "")).strip() == "1" for r in mut_rows)
        tp53_mut = "Unknown"
        kras_mut = "Unknown"
        mutation_source = "cbio_mutation_tp53_kras"
        n_samples_mut_data = sum(1 for r in mut_rows if str(r.get("has_mutation_data", "")).strip() == "1")
        if mutation_has_data:
            tp53_mut = "1" if any(str(r.get("tp53_mutated", "")).strip() == "1" for r in mut_rows) else "0"
            kras_mut = "1" if any(str(r.get("kras_mutated", "")).strip() == "1" for r in mut_rows) else "0"

        # Purity
        pur_vals = [purity_by_sample[s] for s in sample_ids if s in purity_by_sample]
        purity = mode_float(pur_vals)
        purity_source = "absolute_mastercalls" if purity is not None else "missing"

        master_cases.append(
            {
                "case_id": case_id,
                "project_dir": project_dir,
                "project_dir_conflict": int(project_conflict),
                "gdc_project_id": gdc_project_id,
                "num_slides": len(slide_keys),
                "num_samples": len(sample_ids),
                "cbio_subtype_mode": cbio_subtype_mode,
                "cbio_subtype_conflict": int(cbio_subtype_conf),
                "hpv_status": hpv_status,
                "hpv_source": hpv_source,
                "hpv_conflict": hpv_conflict,
                "hpv_score_viral": f"{hpv_score:.6f}" if hpv_score is not None else "",
                "hpv_status_from_viral_threshold6": hpv_score_pred,
                "immune_subtype": immune_subtype,
                "immune_source": immune_source,
                "immune_conflict": immune_conflict,
                "msi_status": msi_status,
                "msi_source": msi_source,
                "msi_conflict": msi_conflict,
                "pam50_subtype": pam50,
                "pam50_source": pam50_source,
                "pam50_conflict": pam50_conflict,
                "tumor_grade": grade,
                "grade_source": grade_source,
                "grade_conflict": grade_conflict,
                "stage": stage,
                "stage_source": stage_source,
                "stage_conflict": stage_conflict,
                "os_status": os_status,
                "os_months": f"{os_months:.6f}" if os_months is not None else "",
                "pfs_status": pfs_status,
                "pfs_months": f"{pfs_months:.6f}" if pfs_months is not None else "",
                "survival_source": survival_source,
                "survival_conflict": survival_conflict,
                "tp53_mutated": tp53_mut,
                "kras_mutated": kras_mut,
                "mutation_source": mutation_source,
                "mutation_has_data": int(mutation_has_data),
                "mutation_n_samples_with_data": n_samples_mut_data,
                "tumor_purity": f"{purity:.6f}" if purity is not None else "",
                "purity_source": purity_source,
                "purity_n_samples": len(pur_vals),
            }
        )

    master_cases.sort(key=lambda r: r["case_id"])
    case_fields = list(master_cases[0].keys())
    write_tsv(master_dir / "case_labels_master.tsv", case_fields, master_cases)

    # Per-target case files
    def target_rows(name: str, cols: List[str], keep_fn) -> None:
        rows = [r for r in master_cases if keep_fn(r)]
        write_tsv(targets_dir / f"{name}.tsv", cols, rows)

    target_rows(
        "hpv_status_case",
        ["case_id", "project_dir", "hpv_status", "hpv_source", "hpv_conflict", "hpv_score_viral", "hpv_status_from_viral_threshold6"],
        lambda r: r["project_dir"] in {"TCGA-CESC", "TCGA-HNSC"},
    )
    target_rows(
        "immune_subtype_case",
        ["case_id", "project_dir", "immune_subtype", "immune_source", "immune_conflict"],
        lambda r: True,
    )
    target_rows(
        "msi_status_case",
        ["case_id", "project_dir", "msi_status", "msi_source", "msi_conflict"],
        lambda r: r["project_dir"] in {"TCGA-COAD", "TCGA-STAD"},
    )
    target_rows(
        "pam50_case",
        ["case_id", "project_dir", "pam50_subtype", "pam50_source", "pam50_conflict"],
        lambda r: r["project_dir"] in {"TCGA-BRCA", "TCGA-BRCA_IDC", "TCGA-BRCA_OTHERS"},
    )
    target_rows(
        "tumor_grade_case",
        ["case_id", "project_dir", "tumor_grade", "grade_source", "grade_conflict"],
        lambda r: True,
    )
    target_rows(
        "stage_case",
        ["case_id", "project_dir", "stage", "stage_source", "stage_conflict"],
        lambda r: True,
    )
    target_rows(
        "survival_case",
        ["case_id", "project_dir", "os_status", "os_months", "pfs_status", "pfs_months", "survival_source", "survival_conflict"],
        lambda r: True,
    )
    target_rows(
        "mutation_tp53_kras_case",
        [
            "case_id",
            "project_dir",
            "tp53_mutated",
            "kras_mutated",
            "mutation_has_data",
            "mutation_n_samples_with_data",
            "mutation_source",
        ],
        lambda r: True,
    )
    target_rows(
        "tumor_purity_case",
        ["case_id", "project_dir", "tumor_purity", "purity_n_samples", "purity_source"],
        lambda r: True,
    )

    # Slide master
    case_map = {r["case_id"]: r for r in master_cases}
    slide_rows: List[dict] = []
    for slide_key, meta in manifest.items():
        case_id = str(meta.get("patient_id", "") or parse_case_id_from_slide(slide_key)).upper()
        sample_id = parse_sample_id_from_slide(slide_key)
        proj = project_from_meta(meta)
        base = {
            "slide_key": slide_key,
            "case_id": case_id,
            "sample_id": sample_id,
            "project_dir": proj,
            "h5_path": str(meta.get("h5_path", "") or ""),
        }
        row = dict(base)
        row.update(case_map.get(case_id, {}))
        slide_rows.append(row)

    slide_rows.sort(key=lambda r: r["slide_key"])
    slide_fields = list(slide_rows[0].keys())
    write_tsv(master_dir / "slide_labels_master.tsv", slide_fields, slide_rows)

    # QC tables
    def coverage_count(col: str, unknown_tokens: set, rows: List[dict]) -> int:
        return sum(1 for r in rows if str(r.get(col, "")).strip() not in unknown_tokens)

    total_cases = len(master_cases)
    coverage_rows = []
    targets = [
        ("hpv_status", {"", "Unknown"}, lambda r: r["project_dir"] in {"TCGA-CESC", "TCGA-HNSC"}),
        ("immune_subtype", {"", "Unknown"}, lambda r: True),
        ("msi_status", {"", "Unknown"}, lambda r: r["project_dir"] in {"TCGA-COAD", "TCGA-STAD"}),
        ("pam50_subtype", {"", "Unknown"}, lambda r: r["project_dir"] in {"TCGA-BRCA", "TCGA-BRCA_IDC", "TCGA-BRCA_OTHERS"}),
        ("tumor_grade", {"", "Unknown"}, lambda r: True),
        ("stage", {"", "Unknown"}, lambda r: True),
        ("os_status", {"", "Unknown"}, lambda r: True),
        ("tp53_mutated", {"", "Unknown"}, lambda r: True),
        ("kras_mutated", {"", "Unknown"}, lambda r: True),
        ("tumor_purity", {""}, lambda r: True),
    ]
    for col, unknown, applicable_fn in targets:
        applicable_rows = [r for r in master_cases if applicable_fn(r)]
        n = coverage_count(col, unknown, applicable_rows)
        applicable_total = len(applicable_rows)
        coverage_rows.append(
            {
                "target": col,
                "labeled_cases": n,
                "total_cases": total_cases,
                "coverage": f"{(n / total_cases):.6f}" if total_cases else "0.000000",
                "applicable_cases": applicable_total,
                "coverage_applicable": f"{(n / applicable_total):.6f}" if applicable_total else "0.000000",
            }
        )
    write_tsv(qc_dir / "coverage_by_target.tsv", list(coverage_rows[0].keys()), coverage_rows)

    if conflicts:
        write_tsv(qc_dir / "conflicts.tsv", list(conflicts[0].keys()), conflicts)
    else:
        write_tsv(qc_dir / "conflicts.tsv", ["case_id", "target", "detail"], [])

    source_files = {
        "manifest_index": args.manifest_index,
        "cbio_slide_tsv": args.cbio_slide_tsv,
        "gdc_slide_tsv": args.gdc_slide_tsv,
        "cesc_hpv_tsv": args.cesc_hpv_tsv,
        "hnsc_hpv_tsv": args.hnsc_hpv_tsv,
        "pancan_subtype_tsv": args.pancan_subtype_tsv,
        "viral_tsv": args.viral_tsv,
        "purity_tsv": args.purity_tsv,
        "mutation_tsv": args.mutation_tsv,
        "immune_tsv": args.immune_tsv,
    }
    sources_meta = {}
    for k, p in source_files.items():
        st = p.stat()
        sources_meta[k] = {
            "path": str(p),
            "size_bytes": st.st_size,
            "sha256": file_sha256(p),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        }
    conflict_counts = Counter(c["target"] for c in conflicts)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "cases": total_cases,
            "slides": len(slide_rows),
            "conflicts": len(conflicts),
        },
        "conflicts_by_target": dict(conflict_counts),
        "coverage_by_target": coverage_rows,
        "outputs": {
            "targets_dir": str(targets_dir),
            "case_master": str(master_dir / "case_labels_master.tsv"),
            "slide_master": str(master_dir / "slide_labels_master.tsv"),
            "coverage_tsv": str(qc_dir / "coverage_by_target.tsv"),
            "conflicts_tsv": str(qc_dir / "conflicts.tsv"),
        },
        "sources": sources_meta,
    }
    (qc_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[ok] wrote {master_dir / 'case_labels_master.tsv'}")
    print(f"[ok] wrote {master_dir / 'slide_labels_master.tsv'}")
    print(f"[ok] wrote {qc_dir / 'coverage_by_target.tsv'}")
    print(f"[ok] wrote {qc_dir / 'conflicts.tsv'}")
    print(f"[ok] wrote {qc_dir / 'build_summary.json'}")


if __name__ == "__main__":
    main()
