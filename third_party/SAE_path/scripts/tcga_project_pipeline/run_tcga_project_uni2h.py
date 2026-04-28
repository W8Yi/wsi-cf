#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
INDEX_JSON_DEFAULT = ROOT_DIR / "metadata" / "indexes" / "manifest_index.json"
BASE_DATA_DIR_DEFAULT = Path("/research/projects/mllab/WSI/TCGA_features")
GDC_CLIENT_DEFAULT = ROOT_DIR / "gdc" / "gdc-client"
DATASET_ID_DEFAULT = "w8yi/tcga-wsi-uni2h-features"
WSI_EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")

PROJECTS = [
    "TCGA-ACC",
    "TCGA-BLCA",
    "TCGA-BRCA_IDC",
    "TCGA-BRCA_OTHERS",
    "TCGA-CESC",
    "TCGA-CHOL",
    "TCGA-COAD",
    "TCGA-DLBC",
    "TCGA-ESCA",
    "TCGA-GBM",
    "TCGA-HNSC",
    "TCGA-KICH",
    "TCGA-KIRC",
    "TCGA-KIRP",
    "TCGA-LGG",
    "TCGA-LIHC",
    "TCGA-LUAD",
    "TCGA-LUSC",
    "TCGA-MESO",
    "TCGA-OV",
    "TCGA-PAAD",
    "TCGA-PCPG",
    "TCGA-PRAD",
    "TCGA-READ",
    "TCGA-SARC",
    "TCGA-SKCM",
    "TCGA-STAD",
    "TCGA-TGCT",
    "TCGA-THCA",
    "TCGA-THYM",
    "TCGA-UCEC",
    "TCGA-UCS",
    "TCGA-UVM",
]


@dataclass
class FeatureCheckResult:
    project: str
    expected_total: int
    feature_total: int
    matched_total: int
    missing: list[str]
    extra: list[str]
    matched: bool
    project_dir: Path
    features_dir: Path


def parse_csv_upper(raw: str) -> set[str]:
    return {item.strip().upper() for item in raw.split(",") if item.strip()}


def normalize_project(project: str) -> str:
    out = project.strip().upper()
    if out and not out.startswith("TCGA-"):
        out = f"TCGA-{out}"
    return out


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_gpu_ids(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def has_single_valid_gdc(entry: dict) -> bool:
    gdc = entry.get("gdc")
    if isinstance(gdc, dict):
        gid = str(gdc.get("id", "")).strip()
        fname = str(gdc.get("filename", "")).strip()
        return bool(gid and fname)
    if isinstance(gdc, list):
        if len(gdc) != 1:
            return False
        item = gdc[0]
        if not isinstance(item, dict):
            return False
        gid = str(item.get("id", "")).strip()
        fname = str(item.get("filename", "")).strip()
        return bool(gid and fname)
    return False


def load_index(index_json: Path) -> dict:
    return json.loads(index_json.read_text())


def expected_slide_keys(index_payload: dict, project: str) -> list[str]:
    project_upper = project.upper()
    return sorted(
        key
        for key, value in index_payload.items()
        if str(value.get("dataset", "")).upper() == project_upper and has_single_valid_gdc(value)
    )


def feature_keys(features_dir: Path) -> set[str]:
    if not features_dir.exists():
        return set()
    return {path.stem for path in features_dir.glob("*.h5")}


def check_project(
    index_payload: dict,
    base_data_dir: Path,
    project: str,
    strict_match: bool = False,
) -> FeatureCheckResult:
    expected = expected_slide_keys(index_payload, project)
    expected_set = set(expected)

    project_dir = base_data_dir / project
    features_dir = project_dir / "features"
    found_set = feature_keys(features_dir)

    missing = sorted(expected_set - found_set)
    extra = sorted(found_set - expected_set)
    matched_total = len(expected_set & found_set)

    matched = bool(expected_set) and (not missing) and ((not strict_match) or (not extra))

    return FeatureCheckResult(
        project=project,
        expected_total=len(expected_set),
        feature_total=len(found_set),
        matched_total=matched_total,
        missing=missing,
        extra=extra,
        matched=matched,
        project_dir=project_dir,
        features_dir=features_dir,
    )


def write_pipeline_summary(result: FeatureCheckResult, strict_match: bool) -> Path:
    summary_path = result.project_dir / "pipeline_summary.json"
    payload = {
        "project": result.project,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "feature_check_only",
        "feature_check": {
            "strict_match": strict_match,
            "matched": result.matched,
            "expected_total": result.expected_total,
            "feature_total": result.feature_total,
            "matched_total": result.matched_total,
            "missing_total": len(result.missing),
            "extra_total": len(result.extra),
            "missing_files": result.missing,
            "extra_files_not_in_manifest": result.extra,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    return summary_path


def find_slide_file(slides_dir: Path, slide_key: str) -> Path | None:
    for ext in WSI_EXTS:
        p = slides_dir / f"{slide_key}{ext}"
        if p.exists():
            return p
    return None


def run_download_missing(
    missing_keys: list[str],
    *,
    index_json: Path,
    slides_dir: Path,
    python_bin: str,
    gdc_client: Path,
    gdc_token: str,
    download_jobs: int,
    download_n_processes: int,
) -> None:
    if not missing_keys:
        return

    slides_dir.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, int(download_jobs))
    nproc = max(1, int(download_n_processes))

    print(f"[work:download] missing={len(missing_keys)} workers={max_workers} gdc_n={nproc}", flush=True)

    def one(slide_key: str) -> tuple[str, int, str]:
        cmd = [
            python_bin,
            str(ROOT_DIR / "utils" / "wsi_downloader.py"),
            slide_key,
            "--index",
            str(index_json),
            "--out",
            str(slides_dir),
            "--gdc-client",
            str(gdc_client),
            "--n-processes",
            str(nproc),
            "--quiet",
        ]
        if gdc_token.strip():
            cmd.extend(["--token", gdc_token.strip()])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return (slide_key, 0, "")
        msg = (proc.stderr or proc.stdout or "download failed").strip()
        return (slide_key, proc.returncode, msg)

    failures: list[tuple[str, int, str]] = []
    done = 0
    total = len(missing_keys)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one, key) for key in missing_keys]
        for fut in cf.as_completed(futs):
            slide_key, rc, msg = fut.result()
            done += 1
            if rc == 0:
                print(f"[work:download {done}/{total}] ok {slide_key}", flush=True)
            else:
                print(f"[work:download {done}/{total}] fail {slide_key}: {msg}", flush=True)
                failures.append((slide_key, rc, msg))

    if failures:
        sample = ", ".join(x[0] for x in failures[:10])
        raise RuntimeError(f"download failed for {len(failures)} slide(s): {sample}")


def run_extract_missing(
    missing_keys: list[str],
    *,
    project: str,
    project_dir: Path,
    slides_dir: Path,
    features_dir: Path,
    coords_dir: Path,
    vis_dir: Path,
    python_bin: str,
    gpu_list_raw: str,
    batch_size: int,
    filter_workers: int,
    reader: str,
    loader_workers: int,
    loader_prefetch_factor: int,
    loader_pin_memory: bool,
    extract_dynamic_gpu: bool,
    tmp_work_root: Path,
) -> None:
    if not missing_keys:
        return

    tmp_work_root.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    missing_wsi_dir = Path(tempfile.mkdtemp(prefix=f"tcga_missing_wsi_{project}.", dir=str(tmp_work_root)))
    tmp_summary_dir = Path(tempfile.mkdtemp(prefix=f"tcga_summary_{project}.", dir=str(tmp_work_root)))
    try:
        for key in missing_keys:
            src = find_slide_file(slides_dir, key)
            if src is None:
                raise RuntimeError(f"missing downloaded slide for key: {key}")
            dst = missing_wsi_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                dst.symlink_to(src)
            except Exception:
                shutil.copy2(src, dst)

        cmd = [
            python_bin,
            str(ROOT_DIR / "scripts" / "extract_filtered_uni_from_wsi.py"),
            "--wsi_dir",
            str(missing_wsi_dir),
            "--out_dir",
            str(features_dir),
            "--summary_dir",
            str(tmp_summary_dir),
            "--coords_dir",
            str(coords_dir),
            "--viz_dir",
            str(vis_dir),
            "--reader",
            str(reader),
            "--batch_size",
            str(int(batch_size)),
            "--filter_workers",
            str(int(filter_workers)),
            "--loader_workers",
            str(int(loader_workers)),
            "--prefetch_factor",
            str(int(loader_prefetch_factor)),
            "--num_shards",
            "1",
            "--shard_index",
            "0",
        ]
        if loader_pin_memory:
            cmd.append("--pin_memory")

        gpu_txt = str(gpu_list_raw).strip().lower()
        env = os.environ.copy()
        log_file: Path

        if gpu_txt == "cpu":
            cmd.extend(["--device", "cpu"])
            log_file = project_dir / "shard_cpu.log"
            print(f"[work:extract] CPU mode for {len(missing_keys)} missing slide(s)", flush=True)
        else:
            gpu_ids = parse_gpu_ids(gpu_list_raw)
            if not gpu_ids:
                gpu_ids = ["0"]
            if len(gpu_ids) >= 2:
                cmd.extend(["--device", "cuda:0"])
                if extract_dynamic_gpu:
                    cmd.extend(["--gpu_list", ",".join(gpu_ids), "--dynamic_gpu"])
                    log_file = project_dir / "shard_dynamic.log"
                    print(f"[work:extract] dynamic GPUs={gpu_ids} for {len(missing_keys)} missing slide(s)", flush=True)
                else:
                    env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
                    log_file = project_dir / "shard_0.log"
                    print(
                        f"[work:extract] multiple GPUs provided but EXTRACT_DYNAMIC_GPU=0; using first GPU {gpu_ids[0]}",
                        flush=True,
                    )
            else:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
                cmd.extend(["--device", "cuda:0"])
                log_file = project_dir / "shard_0.log"
                print(f"[work:extract] GPU={gpu_ids[0]} for {len(missing_keys)} missing slide(s)", flush=True)

        with log_file.open("w") as lf:
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"extract failed (rc={proc.returncode}), see log: {log_file}")

        for coords_file in features_dir.glob("*.coords.csv"):
            coords_file.unlink(missing_ok=True)
    finally:
        shutil.rmtree(missing_wsi_dir, ignore_errors=True)
        shutil.rmtree(tmp_summary_dir, ignore_errors=True)


def cleanup_slides(project: str, base_data_dir: Path, slides_dir: Path) -> None:
    expected = base_data_dir / project / "slides"
    if slides_dir != expected:
        raise RuntimeError(f"slide cleanup safety check failed: {slides_dir} != {expected}")
    if slides_dir.exists():
        shutil.rmtree(slides_dir)
        print(f"[cleanup] deleted slides directory: {slides_dir}", flush=True)


def _copy_h5_only(src_features_dir: Path, dst_features_dir: Path) -> int:
    dst_features_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for h5 in sorted(src_features_dir.glob("*.h5")):
        shutil.copy2(h5, dst_features_dir / h5.name)
        count += 1
    return count


def run_upload_if_needed(
    *,
    args: argparse.Namespace,
    project: str,
    project_dir: Path,
    features_dir: Path,
    vis_dir: Path,
) -> None:
    if args.auto_upload != 1:
        print("[upload] disabled (AUTO_UPLOAD=0)", flush=True)
        return

    marker_path = project_dir / args.upload_marker_name
    if marker_path.exists():
        print(f"[upload] marker exists, skip upload: {marker_path}", flush=True)
        return

    if not features_dir.exists():
        raise RuntimeError(f"features directory not found: {features_dir}")

    args.tmp_work_root.mkdir(parents=True, exist_ok=True)

    if args.upload_mode == "large":
        staging = Path(tempfile.mkdtemp(prefix=f"hf_tcga_{project}.", dir=str(args.tmp_work_root)))
        try:
            staged_features = staging / project / "features"
            staged_vis = staging / project / "vis"
            h5_count = _copy_h5_only(features_dir, staged_features)
            if h5_count <= 0:
                raise RuntimeError(f"no .h5 files to upload in {features_dir}")
            if vis_dir.exists():
                shutil.copytree(vis_dir, staged_vis, dirs_exist_ok=True)
            else:
                staged_vis.mkdir(parents=True, exist_ok=True)
                print(f"[upload] vis directory missing, uploading empty vis folder: {vis_dir}", flush=True)

            cmd = [
                "hf",
                "upload-large-folder",
                args.dataset_id,
                str(staging),
                "--repo-type",
                "dataset",
                "--num-workers",
                str(args.num_upload_workers),
            ]
            print(f"[upload] {' '.join(cmd)}", flush=True)
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                raise RuntimeError(f"hf upload-large-folder failed (rc={proc.returncode})")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    else:
        h5_stage = Path(tempfile.mkdtemp(prefix=f"hf_h5_only_{project}.", dir=str(args.tmp_work_root)))
        try:
            h5_count = _copy_h5_only(features_dir, h5_stage)
            if h5_count <= 0:
                raise RuntimeError(f"no .h5 files to upload in {features_dir}")

            cmd_features = [
                "hf",
                "upload",
                args.dataset_id,
                str(h5_stage),
                f"{project}/features",
                "--repo-type",
                "dataset",
            ]
            print(f"[upload] {' '.join(cmd_features)}", flush=True)
            rc_features = subprocess.run(cmd_features).returncode
            if rc_features != 0:
                raise RuntimeError(f"hf upload features failed (rc={rc_features})")

            if vis_dir.exists():
                cmd_vis = [
                    "hf",
                    "upload",
                    args.dataset_id,
                    str(vis_dir),
                    f"{project}/vis",
                    "--repo-type",
                    "dataset",
                ]
                print(f"[upload] {' '.join(cmd_vis)}", flush=True)
                rc_vis = subprocess.run(cmd_vis).returncode
                if rc_vis != 0:
                    raise RuntimeError(f"hf upload vis failed (rc={rc_vis})")
            else:
                print(f"[upload] vis directory missing, skipped vis upload: {vis_dir}", flush=True)
        finally:
            shutil.rmtree(h5_stage, ignore_errors=True)

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.touch()
    print(f"[upload] completed; wrote marker: {marker_path}", flush=True)


def run_process_missing(args: argparse.Namespace, project: str, initial: FeatureCheckResult) -> None:
    if not initial.missing:
        print("[work] no missing feature keys to process.", flush=True)
        return

    project_dir = args.base_data_dir / project
    slides_dir = project_dir / "slides"
    features_dir = project_dir / "features"
    coords_dir = project_dir / "coords"
    vis_dir = project_dir / "vis"

    run_download_missing(
        initial.missing,
        index_json=args.index_json,
        slides_dir=slides_dir,
        python_bin=args.python_bin,
        gdc_client=args.gdc_client,
        gdc_token=args.gdc_token,
        download_jobs=args.download_jobs,
        download_n_processes=args.download_n_processes,
    )
    run_extract_missing(
        initial.missing,
        project=project,
        project_dir=project_dir,
        slides_dir=slides_dir,
        features_dir=features_dir,
        coords_dir=coords_dir,
        vis_dir=vis_dir,
        python_bin=args.python_bin,
        gpu_list_raw=args.gpu_list,
        batch_size=args.batch_size,
        filter_workers=args.filter_workers,
        reader=args.reader,
        loader_workers=args.loader_workers,
        loader_prefetch_factor=args.loader_prefetch_factor,
        loader_pin_memory=args.loader_pin_memory,
        extract_dynamic_gpu=args.extract_dynamic_gpu,
        tmp_work_root=args.tmp_work_root,
    )

    if args.delete_slides_after_project:
        cleanup_slides(project, args.base_data_dir, slides_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Check one TCGA project by validating features/*.h5 against manifest slide keys. "
            "Optional --process-missing can download/extract missing features."
        )
    )
    ap.add_argument("--project", help="Project name, e.g. TCGA-HNSC or HNSC.")
    ap.add_argument("--list-projects", action="store_true", help="Print supported projects and exit.")

    ap.add_argument(
        "--strict-match",
        action="store_true",
        default=parse_bool_env("STRICT_MATCH", False),
        help="Require exact equality (no extra .h5 files).",
    )
    ap.add_argument(
        "--base-data-dir",
        type=Path,
        default=Path(os.environ.get("BASE_DATA_DIR", str(BASE_DATA_DIR_DEFAULT))),
        help="Base folder containing TCGA project directories.",
    )
    ap.add_argument(
        "--index-json",
        type=Path,
        default=Path(os.environ.get("INDEX_JSON", str(INDEX_JSON_DEFAULT))),
        help="Path to manifest index JSON.",
    )
    ap.add_argument("--no-summary", action="store_true", help="Do not write pipeline_summary.json.")

    ap.add_argument(
        "--process-missing",
        action="store_true",
        default=parse_bool_env("PROCESS_MISSING", False),
        help="If mismatch is due to missing features, download/extract them and re-check.",
    )
    ap.add_argument(
        "--auto-upload",
        type=int,
        choices=[0, 1],
        default=1 if parse_bool_env("AUTO_UPLOAD", True) else 0,
        help="When 1, upload to Hugging Face if marker is missing.",
    )
    ap.add_argument(
        "--dataset-id",
        default=os.environ.get("DATASET_ID", DATASET_ID_DEFAULT),
        help="Hugging Face dataset repo id.",
    )
    ap.add_argument(
        "--upload-mode",
        default=os.environ.get("UPLOAD_MODE", "standard"),
        choices=["standard", "large"],
        help="Upload mode: standard or upload-large-folder.",
    )
    ap.add_argument(
        "--num-upload-workers",
        type=int,
        default=int(os.environ.get("NUM_UPLOAD_WORKERS", "8")),
        help="Worker count for upload-large-folder.",
    )
    ap.add_argument(
        "--upload-marker-name",
        default=os.environ.get("UPLOAD_MARKER_NAME", ".upload_complete"),
        help="Project-local marker file written when upload completes.",
    )
    ap.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable for helper scripts.",
    )
    ap.add_argument(
        "--gdc-client",
        type=Path,
        default=Path(os.environ.get("GDC_CLIENT", str(GDC_CLIENT_DEFAULT))),
        help="Path to gdc-client binary.",
    )
    ap.add_argument(
        "--gdc-token",
        default=os.environ.get("GDC_TOKEN", ""),
        help="Optional path to GDC token file.",
    )
    ap.add_argument("--download-jobs", type=int, default=int(os.environ.get("DOWNLOAD_JOBS", "4")))
    ap.add_argument("--download-n-processes", type=int, default=int(os.environ.get("DOWNLOAD_N_PROCESSES", "4")))

    ap.add_argument("--gpu-list", default=os.environ.get("GPU_LIST", "0"), help="Comma-separated GPU ids or 'cpu'.")
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "512")))
    ap.add_argument("--filter-workers", type=int, default=int(os.environ.get("FILTER_WORKERS", "4")))
    ap.add_argument("--reader", default=os.environ.get("READER", "auto"), choices=["auto", "openslide", "cucim"])
    ap.add_argument("--loader-workers", type=int, default=int(os.environ.get("LOADER_WORKERS", "0")))
    ap.add_argument(
        "--loader-prefetch-factor",
        type=int,
        default=int(os.environ.get("LOADER_PREFETCH_FACTOR", "2")),
    )
    ap.add_argument(
        "--loader-pin-memory",
        action="store_true",
        default=parse_bool_env("LOADER_PIN_MEMORY", True),
    )
    ap.add_argument(
        "--extract-dynamic-gpu",
        action="store_true",
        default=parse_bool_env("EXTRACT_DYNAMIC_GPU", False),
    )
    ap.add_argument(
        "--tmp-work-root",
        type=Path,
        default=Path(os.environ.get("TMP_WORK_ROOT", str(BASE_DATA_DIR_DEFAULT / ".tmp_work"))),
    )
    ap.add_argument(
        "--delete-slides-after-project",
        action="store_true",
        default=parse_bool_env("DELETE_SLIDES_AFTER_PROJECT", False),
    )

    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.list_projects:
        for project in PROJECTS:
            print(project)
        return 0

    if not args.project:
        ap.error("--project is required unless --list-projects is used.")

    project = normalize_project(args.project)
    if project not in PROJECTS:
        print(f"Unsupported project: {project}", file=sys.stderr)
        print("Use --list-projects to see valid choices.", file=sys.stderr)
        return 1

    try:
        index_payload = load_index(args.index_json)
    except Exception as exc:
        print(f"Failed to load index JSON: {args.index_json} ({exc})", file=sys.stderr)
        return 1

    result = check_project(index_payload, args.base_data_dir, project, strict_match=args.strict_match)
    if not args.no_summary:
        summary_path = write_pipeline_summary(result, strict_match=args.strict_match)
        print(f"[summary] wrote {summary_path}")

    print(
        f"[check] {project} expected={result.expected_total} features={result.feature_total} "
        f"matched={result.matched_total} missing={len(result.missing)} extra={len(result.extra)} "
        f"strict={1 if args.strict_match else 0}"
    )

    if result.expected_total == 0:
        print(f"[fail] {project} has 0 expected manifest slides.", file=sys.stderr)
        return 2

    if result.missing:
        print(f"[missing] first {min(10, len(result.missing))}/{len(result.missing)}: {' '.join(result.missing[:10])}")
    if result.extra:
        print(f"[extra] first {min(10, len(result.extra))}/{len(result.extra)}: {' '.join(result.extra[:10])}")

    if result.matched:
        try:
            run_upload_if_needed(
                args=args,
                project=project,
                project_dir=result.project_dir,
                features_dir=result.features_dir,
                vis_dir=result.project_dir / "vis",
            )
        except Exception as exc:
            print(f"[fail] upload failed for {project}: {exc}", file=sys.stderr)
            return 2
        print(f"[pass] {project} feature check matched; continuing is safe.")
        return 0

    if args.process_missing and result.missing:
        print(f"[work] mismatch detected -> start processing {len(result.missing)} missing slide(s)", flush=True)
        try:
            run_process_missing(args, project, result)
        except Exception as exc:
            print(f"[fail] processing failed for {project}: {exc}", file=sys.stderr)
            return 2

        result_after = check_project(index_payload, args.base_data_dir, project, strict_match=args.strict_match)
        if not args.no_summary:
            summary_path = write_pipeline_summary(result_after, strict_match=args.strict_match)
            print(f"[summary] wrote {summary_path}")

        print(
            f"[recheck] {project} expected={result_after.expected_total} features={result_after.feature_total} "
            f"matched={result_after.matched_total} missing={len(result_after.missing)} extra={len(result_after.extra)}"
        )
        if result_after.matched:
            try:
                run_upload_if_needed(
                    args=args,
                    project=project,
                    project_dir=result_after.project_dir,
                    features_dir=result_after.features_dir,
                    vis_dir=result_after.project_dir / "vis",
                )
            except Exception as exc:
                print(f"[fail] upload failed for {project}: {exc}", file=sys.stderr)
                return 2
            print(f"[pass] {project} recovered and now matches.")
            return 0

        print(f"[fail] {project} still mismatched after processing.", file=sys.stderr)
        return 2

    print(f"[fail] {project} feature check mismatch.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
