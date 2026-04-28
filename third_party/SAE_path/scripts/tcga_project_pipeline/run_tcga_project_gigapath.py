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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import h5py


ROOT_DIR = Path(__file__).resolve().parents[2]
INDEX_JSON_DEFAULT = ROOT_DIR / "metadata" / "indexes" / "manifest_index.json"
BASE_DATA_DIR_DEFAULT = Path("/research/projects/mllab/WSI/TCGA_features")
GDC_CLIENT_DEFAULT = ROOT_DIR / "gdc" / "gdc-client"
DATASET_ID_DEFAULT = "w8yi/tcga-wsi-gigapath-features"
FEATURES_SUBDIR_DEFAULT = "features_gigapath"
UPLOAD_FEATURES_SUBDIR_DEFAULT = "features_gigapath"
PROJECT_COMPLETE_FLAG_NAME_DEFAULT = ".seal_encode_complete"
WSI_EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")
LEGACY_FEATURES_SUBDIRS = ("gigapath_feature", "features_seal/gigapath_feature")

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
    pruned_invalid_total: int = 0
    pruned_invalid_samples: list[str] = field(default_factory=list)


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


def _validate_feature_h5(path: Path) -> tuple[bool, str]:
    try:
        with h5py.File(path, "r") as f:
            required = {"features", "coords", "coords_patching", "annots"}
            keys = set(f.keys())
            missing = sorted(required - keys)
            if missing:
                return False, f"missing datasets: {missing}"

            feat = f["features"]
            coords = f["coords"]
            coords_patch = f["coords_patching"]
            annots = f["annots"]

            if feat.ndim != 3:
                return False, f"features ndim={feat.ndim}, expected=3"
            if coords.ndim != 3:
                return False, f"coords ndim={coords.ndim}, expected=3"
            if coords_patch.ndim != 2:
                return False, f"coords_patching ndim={coords_patch.ndim}, expected=2"
            if annots.ndim != 3:
                return False, f"annots ndim={annots.ndim}, expected=3"

            if feat.shape[0] != 1 or coords.shape[0] != 1 or annots.shape[0] != 1:
                return False, f"leading dims invalid feat={feat.shape} coords={coords.shape} annots={annots.shape}"
            if coords.shape[2] != 2 or coords_patch.shape[1] != 2 or annots.shape[2] != 1:
                return False, f"tail dims invalid coords={coords.shape} coords_patching={coords_patch.shape} annots={annots.shape}"

            n_feat = int(feat.shape[1])
            n_coords = int(coords.shape[1])
            n_patch = int(coords_patch.shape[0])
            n_ann = int(annots.shape[1])
            if not (n_feat == n_coords == n_patch == n_ann):
                return False, f"row mismatch feat={n_feat} coords={n_coords} coords_patching={n_patch} annots={n_ann}"
            if n_feat <= 0:
                return False, "empty feature rows"

        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def feature_keys(features_dir: Path) -> tuple[set[str], list[str]]:
    if not features_dir.exists():
        return set(), []

    valid: set[str] = set()
    pruned_invalid: list[str] = []
    for path in sorted(features_dir.glob("*.h5")):
        ok, reason = _validate_feature_h5(path)
        if ok:
            valid.add(path.stem)
            continue

        try:
            path.unlink()
            pruned_invalid.append(f"{path.name}: {reason}")
        except Exception as exc:
            pruned_invalid.append(f"{path.name}: {reason}; unlink_failed={type(exc).__name__}: {exc}")

    return valid, pruned_invalid


def check_project(
    index_payload: dict,
    base_data_dir: Path,
    project: str,
    strict_match: bool = False,
    features_subdir: str = FEATURES_SUBDIR_DEFAULT,
) -> FeatureCheckResult:
    expected = expected_slide_keys(index_payload, project)
    expected_set = set(expected)

    project_dir = base_data_dir / project
    features_dir = project_dir / features_subdir
    found_set, pruned_invalid = feature_keys(features_dir)

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
        pruned_invalid_total=len(pruned_invalid),
        pruned_invalid_samples=pruned_invalid[:10],
    )


def maybe_migrate_legacy_features_subdir(project_dir: Path, target_subdir: str) -> str:
    target_dir = project_dir / target_subdir
    if target_dir.exists():
        return target_subdir

    for legacy_subdir in LEGACY_FEATURES_SUBDIRS:
        if not legacy_subdir or legacy_subdir == target_subdir:
            continue
        legacy_dir = project_dir / legacy_subdir
        if not legacy_dir.exists():
            continue
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy_dir.rename(target_dir)
            print(f"[migrate] features dir {legacy_dir} -> {target_dir}", flush=True)
            return target_subdir
        except Exception as exc:
            print(
                f"[warn] could not rename legacy features dir {legacy_dir} -> {target_dir}: {exc}. "
                f"Using legacy dir for this run.",
                flush=True,
            )
            return legacy_subdir
    return target_subdir


def ensure_project_structure(project_dir: Path, features_subdir: str = FEATURES_SUBDIR_DEFAULT) -> None:
    (project_dir / "slides").mkdir(parents=True, exist_ok=True)
    (project_dir / features_subdir).mkdir(parents=True, exist_ok=True)
    (project_dir / "coords").mkdir(parents=True, exist_ok=True)
    (project_dir / "vis").mkdir(parents=True, exist_ok=True)


def write_slide_keys_manifest(project_dir: Path, expected_keys: list[str]) -> Path:
    out_path = project_dir / "slide_keys.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{k}\n" for k in expected_keys), encoding="utf-8")
    return out_path


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
            "pruned_invalid_total": result.pruned_invalid_total,
            "pruned_invalid_samples": result.pruned_invalid_samples,
            "missing_total": len(result.missing),
            "extra_total": len(result.extra),
            "missing_files": result.missing,
            "extra_files_not_in_manifest": result.extra,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    return summary_path


def write_project_complete_flag(
    *,
    project_dir: Path,
    flag_name: str,
    encoder: str,
    phase: str,
    result: FeatureCheckResult,
) -> Path:
    flag_path = project_dir / flag_name
    payload = {
        "project": result.project,
        "encoder": encoder,
        "phase": phase,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matched": result.matched,
        "expected_total": result.expected_total,
        "feature_total": result.feature_total,
        "matched_total": result.matched_total,
        "missing_total": len(result.missing),
        "extra_total": len(result.extra),
    }
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[flag] wrote {flag_path} encoder={encoder} phase={phase} "
        f"matched={int(result.matched)} missing={len(result.missing)} extra={len(result.extra)}",
        flush=True,
    )
    return flag_path


def find_slide_file(slides_dir: Path, slide_key: str) -> Path | None:
    for ext in WSI_EXTS:
        p = slides_dir / f"{slide_key}{ext}"
        if p.exists():
            return p
    return None


def extract_summary_dir_for(project_dir: Path, features_subdir: str) -> Path:
    safe_name = str(features_subdir).strip().replace("/", "_") or "features"
    return project_dir / f"{safe_name}_summary"


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
) -> list[str]:
    if not missing_keys:
        return []

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
        print(f"[warn] download failed for {len(failures)} slide(s): {sample}", flush=True)
    return [x[0] for x in failures]


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
    summary_dir = extract_summary_dir_for(project_dir, features_dir.name)
    summary_dir.mkdir(parents=True, exist_ok=True)
    try:
        for key in missing_keys:
            src = find_slide_file(slides_dir, key)
            if src is None:
                print(f"[warn] missing downloaded slide for key: {key}", flush=True)
                continue
            dst = missing_wsi_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                dst.symlink_to(src)
            except Exception:
                shutil.copy2(src, dst)

        stage_feature_keys_before, _ = feature_keys(features_dir)

        base_cmd = [
            python_bin,
            str(ROOT_DIR / "scripts" / "extract_filtered_gigapath_from_wsi.py"),
            "--wsi_dir",
            str(missing_wsi_dir),
            "--out_dir",
            str(features_dir),
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
        ]
        if loader_pin_memory:
            base_cmd.append("--pin_memory")

        gpu_txt = str(gpu_list_raw).strip().lower()

        if gpu_txt == "cpu":
            cmd = base_cmd + [
            "--summary_dir",
            str(summary_dir),
                "--num_shards",
                "1",
                "--shard_index",
                "0",
                "--device",
                "cpu",
            ]
            env = os.environ.copy()
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            log_file = project_dir / "shard_cpu.log"
            print(f"[work:extract] CPU mode for {len(missing_keys)} missing slide(s)", flush=True)
            with log_file.open("w") as lf:
                proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                print(f"[warn] extract failed (rc={proc.returncode}), see log: {log_file}", flush=True)
        else:
            gpu_ids = parse_gpu_ids(gpu_list_raw)
            if not gpu_ids:
                gpu_ids = ["0"]
            if len(gpu_ids) >= 2 and extract_dynamic_gpu:
                print(
                    f"[work:extract] multi-GPU shard mode enabled across GPUs={gpu_ids} "
                    f"for {len(missing_keys)} missing slide(s)",
                    flush=True,
                )
                procs: list[tuple[int, str, Path, subprocess.Popen[bytes], object]] = []
                num_shards = len(gpu_ids)
                for shard_index, gpu_id in enumerate(gpu_ids):
                    shard_summary_dir = summary_dir / f"shard_{shard_index}"
                    shard_cmd = base_cmd + [
                        "--summary_dir",
                        str(shard_summary_dir),
                        "--num_shards",
                        str(num_shards),
                        "--shard_index",
                        str(shard_index),
                        "--device",
                        "cuda:0",
                    ]
                    shard_env = os.environ.copy()
                    shard_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    shard_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                    log_file = project_dir / f"shard_{shard_index}.log"
                    lf = log_file.open("w")
                    proc = subprocess.Popen(shard_cmd, env=shard_env, stdout=lf, stderr=subprocess.STDOUT)
                    procs.append((shard_index, str(gpu_id), log_file, proc, lf))

                failures: list[str] = []
                for shard_index, gpu_id, log_file, proc, lf in procs:
                    rc = proc.wait()
                    lf.close()
                    if rc != 0:
                        failures.append(f"shard={shard_index} gpu={gpu_id} rc={rc} log={log_file}")
                if failures:
                    print("[warn] extract failed in multi-GPU shard mode: " + " | ".join(failures), flush=True)
            else:
                cmd = base_cmd + [
                    "--summary_dir",
                    str(summary_dir),
                    "--num_shards",
                    "1",
                    "--shard_index",
                    "0",
                    "--device",
                    "cuda:0",
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                log_file = project_dir / "shard_0.log"
                if len(gpu_ids) >= 2 and not extract_dynamic_gpu:
                    print(
                        f"[work:extract] multiple GPUs provided but EXTRACT_DYNAMIC_GPU=0; "
                        f"using first GPU {gpu_ids[0]}",
                        flush=True,
                    )
                else:
                    print(f"[work:extract] GPU={gpu_ids[0]} for {len(missing_keys)} missing slide(s)", flush=True)

                with log_file.open("w") as lf:
                    proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
                if proc.returncode != 0:
                    print(f"[warn] extract failed (rc={proc.returncode}), see log: {log_file}", flush=True)

        for coords_file in features_dir.glob("*.coords.csv"):
            coords_file.unlink(missing_ok=True)
        stage_feature_keys_after, _ = feature_keys(features_dir)
        gained = sorted(stage_feature_keys_after - stage_feature_keys_before)
        print(
            f"[work:extract] summary_dir={summary_dir} newly_completed={len(gained)} "
            f"total_valid={len(stage_feature_keys_after)}",
            flush=True,
        )
    finally:
        shutil.rmtree(missing_wsi_dir, ignore_errors=True)


def cleanup_slides(project: str, base_data_dir: Path, slides_dir: Path) -> None:
    expected = base_data_dir / project / "slides"
    if slides_dir != expected:
        raise RuntimeError(f"slide cleanup safety check failed: {slides_dir} != {expected}")
    if slides_dir.exists():
        shutil.rmtree(slides_dir)
        print(f"[cleanup] deleted slides directory: {slides_dir}", flush=True)


def reuse_coords_from_base(
    *,
    project: str,
    needed_keys: list[str],
    dst_coords_dir: Path,
    reuse_coords_base: Path | None,
) -> int:
    if reuse_coords_base is None:
        return 0

    src_coords_dir = reuse_coords_base / project / "coords"
    if not src_coords_dir.exists():
        print(f"[coords-reuse] source coords dir not found, skip: {src_coords_dir}", flush=True)
        return 0

    dst_coords_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for key in needed_keys:
        src = src_coords_dir / f"{key}.coords.csv"
        dst = dst_coords_dir / f"{key}.coords.csv"
        if dst.exists() or not src.exists():
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(
        f"[coords-reuse] project={project} copied={copied} source={src_coords_dir} dest={dst_coords_dir}",
        flush=True,
    )
    return copied


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
            staged_features = staging / project / args.upload_features_subdir
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
                f"{project}/{args.upload_features_subdir}",
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
    features_dir = project_dir / args.features_subdir
    coords_dir = project_dir / "coords"
    vis_dir = project_dir / "vis"

    _ = reuse_coords_from_base(
        project=project,
        needed_keys=initial.missing,
        dst_coords_dir=coords_dir,
        reuse_coords_base=args.reuse_coords_base,
    )

    download_failures = run_download_missing(
        initial.missing,
        index_json=args.index_json,
        slides_dir=slides_dir,
        python_bin=args.python_bin,
        gdc_client=args.gdc_client,
        gdc_token=args.gdc_token,
        download_jobs=args.download_jobs,
        download_n_processes=args.download_n_processes,
    )
    extract_targets = [key for key in initial.missing if find_slide_file(slides_dir, key) is not None]
    if download_failures:
        print(
            f"[warn] skipping extraction for {len(download_failures)} slide(s) with failed downloads",
            flush=True,
        )
    if not extract_targets:
        print("[warn] no extracted targets available after download step", flush=True)
        return

    run_extract_missing(
        extract_targets,
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
            "Check one TCGA project by validating feature .h5 files against manifest slide keys. "
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
        "--features-subdir",
        default=os.environ.get("FEATURES_SUBDIR", FEATURES_SUBDIR_DEFAULT),
        help="Project-local subdirectory name holding local .h5 features.",
    )
    ap.add_argument(
        "--index-json",
        type=Path,
        default=Path(os.environ.get("INDEX_JSON", str(INDEX_JSON_DEFAULT))),
        help="Path to manifest index JSON.",
    )
    ap.add_argument(
        "--reuse-coords-base",
        type=Path,
        default=Path(os.environ.get("REUSE_COORDS_BASE", "/research/projects/mllab/WSI/TCGA_features")),
        help="If set, copy existing coords from <reuse-base>/<project>/coords before extraction.",
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
        "--upload-features-subdir",
        default=os.environ.get(
            "UPLOAD_FEATURES_SUBDIR",
            os.environ.get("FEATURES_SUBDIR", UPLOAD_FEATURES_SUBDIR_DEFAULT),
        ),
        help="Remote dataset subdirectory name used for uploaded feature .h5 files.",
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
        "--project-complete-flag-name",
        default=os.environ.get("PROJECT_COMPLETE_FLAG_NAME", PROJECT_COMPLETE_FLAG_NAME_DEFAULT),
        help="Project-local flag file written when the encode/check cycle finishes.",
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

    default_gpu_list = os.environ.get("GPU_LIST", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    default_gpu_list = str(default_gpu_list).strip() or "0"
    default_extract_dynamic = parse_bool_env(
        "EXTRACT_DYNAMIC_GPU",
        (str(default_gpu_list).lower() != "cpu") and (len(parse_gpu_ids(default_gpu_list)) >= 2),
    )

    ap.add_argument(
        "--gpu-list",
        default=default_gpu_list,
        help="Comma-separated GPU ids or 'cpu'. Defaults to GPU_LIST or CUDA_VISIBLE_DEVICES.",
    )
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "64")))
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
        default=default_extract_dynamic,
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

    project_dir = args.base_data_dir / project
    args.features_subdir = maybe_migrate_legacy_features_subdir(project_dir, args.features_subdir)
    ensure_project_structure(project_dir, args.features_subdir)
    expected_keys = expected_slide_keys(index_payload, project)
    manifest_path = write_slide_keys_manifest(project_dir, expected_keys)
    print(f"[manifest] wrote {len(expected_keys)} valid slide keys -> {manifest_path}")

    result = check_project(
        index_payload,
        args.base_data_dir,
        project,
        strict_match=args.strict_match,
        features_subdir=args.features_subdir,
    )
    if not args.no_summary:
        summary_path = write_pipeline_summary(result, strict_match=args.strict_match)
        print(f"[summary] wrote {summary_path}")

    print(
        f"[check] {project} expected={result.expected_total} features={result.feature_total} "
        f"matched={result.matched_total} missing={len(result.missing)} extra={len(result.extra)} "
        f"pruned_bad={result.pruned_invalid_total} "
        f"strict={1 if args.strict_match else 0}"
    )
    if result.pruned_invalid_samples:
        print(
            f"[pruned] first {min(10, len(result.pruned_invalid_samples))}/{result.pruned_invalid_total}: "
            f"{' | '.join(result.pruned_invalid_samples[:10])}"
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
            write_project_complete_flag(
                project_dir=result.project_dir,
                flag_name=args.project_complete_flag_name,
                encoder="gigapath",
                phase="matched_upload_failed",
                result=result,
            )
            print(f"[fail] upload failed for {project}: {exc}", file=sys.stderr)
            return 2
        write_project_complete_flag(
            project_dir=result.project_dir,
            flag_name=args.project_complete_flag_name,
            encoder="gigapath",
            phase="matched",
            result=result,
        )
        print(f"[pass] {project} feature check matched; continuing is safe.")
        return 0

    if args.process_missing and result.missing:
        print(f"[work] mismatch detected -> start processing {len(result.missing)} missing slide(s)", flush=True)
        try:
            run_process_missing(args, project, result)
        except Exception as exc:
            print(f"[fail] processing failed for {project}: {exc}", file=sys.stderr)
            return 2

        result_after = check_project(
            index_payload,
            args.base_data_dir,
            project,
            strict_match=args.strict_match,
            features_subdir=args.features_subdir,
        )
        if not args.no_summary:
            summary_path = write_pipeline_summary(result_after, strict_match=args.strict_match)
            print(f"[summary] wrote {summary_path}")

        print(
            f"[recheck] {project} expected={result_after.expected_total} features={result_after.feature_total} "
            f"matched={result_after.matched_total} missing={len(result_after.missing)} "
            f"extra={len(result_after.extra)} pruned_bad={result_after.pruned_invalid_total}"
        )
        if result_after.pruned_invalid_samples:
            print(
                f"[pruned] first {min(10, len(result_after.pruned_invalid_samples))}/"
                f"{result_after.pruned_invalid_total}: {' | '.join(result_after.pruned_invalid_samples[:10])}"
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
                write_project_complete_flag(
                    project_dir=result_after.project_dir,
                    flag_name=args.project_complete_flag_name,
                    encoder="gigapath",
                    phase="processed_matched_upload_failed",
                    result=result_after,
                )
                print(f"[fail] upload failed for {project}: {exc}", file=sys.stderr)
                return 2
            write_project_complete_flag(
                project_dir=result_after.project_dir,
                flag_name=args.project_complete_flag_name,
                encoder="gigapath",
                phase="processed_matched",
                result=result_after,
            )
            print(f"[pass] {project} recovered and now matches.")
            return 0

        write_project_complete_flag(
            project_dir=result_after.project_dir,
            flag_name=args.project_complete_flag_name,
            encoder="gigapath",
            phase="processed_finished_with_missing",
            result=result_after,
        )
        print(f"[fail] {project} still mismatched after processing.", file=sys.stderr)
        return 2

    print(f"[fail] {project} feature check mismatch.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
