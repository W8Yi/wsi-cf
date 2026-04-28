#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_tcga_project_virchow2 import (
    BASE_DATA_DIR_DEFAULT,
    INDEX_JSON_DEFAULT,
    PROJECTS,
    parse_bool_env,
    parse_csv_upper,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
VIRCHOW_SCRIPT = ROOT_DIR / "scripts" / "tcga_project_pipeline" / "run_tcga_project_virchow2.py"
GIGAPATH_SCRIPT = ROOT_DIR / "scripts" / "tcga_project_pipeline" / "run_tcga_project_gigapath.py"
VIRCHOW_STAGE_FLAG_NAME = ".virchow2_encode_complete"
GIGAPATH_STAGE_FLAG_NAME = ".gigapath_encode_complete"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Per-project sequential TCGA pipeline: download/check/process/upload Virchow2 first, "
            "then GigaPath, then optionally clean slides, then move to next project."
        )
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
    ap.add_argument(
        "--reuse-coords-base",
        type=Path,
        default=Path(os.environ.get("REUSE_COORDS_BASE", "/research/projects/mllab/WSI/TCGA_features")),
        help="Copy existing coords from <reuse-base>/<project>/coords before extraction.",
    )
    ap.add_argument(
        "--only-projects",
        default=os.environ.get("ONLY_PROJECTS", ""),
        help="Comma-separated allowlist of projects.",
    )
    ap.add_argument(
        "--skip-projects",
        default=os.environ.get("SKIP_PROJECTS", ""),
        help="Comma-separated blocklist of projects.",
    )
    ap.add_argument(
        "--strict-match",
        action="store_true",
        default=parse_bool_env("STRICT_MATCH", False),
        help="Require exact equality (no extra .h5 files).",
    )
    ap.add_argument(
        "--auto-process-mismatch",
        type=int,
        choices=[0, 1],
        default=1 if parse_bool_env("AUTO_PROCESS_MISMATCH", True) else 0,
        help="When 1, pass --process-missing to both encoder runners.",
    )
    ap.add_argument(
        "--continue-on-error",
        type=int,
        choices=[0, 1],
        default=1 if parse_bool_env("CONTINUE_ON_ERROR", True) else 0,
        help="When 1, keep processing remaining projects after a project failure.",
    )
    ap.add_argument(
        "--cleanup-slides-after-both",
        type=int,
        choices=[0, 1],
        default=1 if parse_bool_env("CLEANUP_SLIDES_AFTER_BOTH", True) else 0,
        help="When 1, delete project slides only after both Virchow2 and GigaPath succeed.",
    )
    ap.add_argument(
        "--virchow-auto-upload",
        type=int,
        choices=[0, 1],
        default=1 if parse_bool_env("VIRCHOW_AUTO_UPLOAD", True) else 0,
        help="Virchow2 stage auto upload.",
    )
    ap.add_argument(
        "--gigapath-auto-upload",
        type=int,
        choices=[0, 1],
        default=1 if parse_bool_env("GIGAPATH_AUTO_UPLOAD", True) else 0,
        help="GigaPath stage auto upload.",
    )
    ap.add_argument(
        "--virchow-dataset-id",
        default=os.environ.get("VIRCHOW_DATASET_ID", "w8yi/tcga-wsi-virchow2-features"),
        help="Hugging Face dataset id for Virchow2 features.",
    )
    ap.add_argument(
        "--gigapath-dataset-id",
        default=os.environ.get("GIGAPATH_DATASET_ID", "w8yi/tcga-wsi-gigapath-features"),
        help="Hugging Face dataset id for GigaPath features.",
    )
    ap.add_argument(
        "--virchow-features-subdir",
        default=os.environ.get("VIRCHOW_FEATURES_SUBDIR", "features_virchow2"),
        help="Project subdir for local Virchow2 .h5 files.",
    )
    ap.add_argument(
        "--gigapath-features-subdir",
        default=os.environ.get("GIGAPATH_FEATURES_SUBDIR", "features_gigapath"),
        help="Project subdir for local GigaPath .h5 files.",
    )
    ap.add_argument(
        "--virchow-upload-subdir",
        default=os.environ.get("VIRCHOW_UPLOAD_SUBDIR", "features_virchow2"),
        help="Remote HF subdir for Virchow2 .h5 files.",
    )
    ap.add_argument(
        "--gigapath-upload-subdir",
        default=os.environ.get("GIGAPATH_UPLOAD_SUBDIR", "features_gigapath"),
        help="Remote HF subdir for GigaPath .h5 files.",
    )
    ap.add_argument(
        "--virchow-upload-marker-name",
        default=os.environ.get("VIRCHOW_UPLOAD_MARKER_NAME", ".upload_complete_virchow2"),
        help="Project-local upload marker for Virchow2 stage.",
    )
    ap.add_argument(
        "--gigapath-upload-marker-name",
        default=os.environ.get("GIGAPATH_UPLOAD_MARKER_NAME", ".upload_complete_gigapath"),
        help="Project-local upload marker for GigaPath stage.",
    )
    ap.add_argument(
        "--project-complete-flag-name",
        default=os.environ.get("PROJECT_COMPLETE_FLAG_NAME", ".seal_encode_complete"),
        help="Project-local flag file written after both encoder stages finish, even if some slides remain missing.",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Forward --no-summary to both encoder runners.",
    )
    return ap


def selected_projects(only_raw: str, skip_raw: str) -> list[str]:
    only_projects = parse_csv_upper(only_raw)
    skip_projects = parse_csv_upper(skip_raw)

    out: list[str] = []
    for project in PROJECTS:
        pu = project.upper()
        if only_projects and pu not in only_projects:
            continue
        if skip_projects and pu in skip_projects:
            continue
        out.append(project)
    return out


def build_stage_cmd(
    *,
    script_path: Path,
    project: str,
    args: argparse.Namespace,
    dataset_id: str,
    features_subdir: str,
    upload_subdir: str,
    upload_marker_name: str,
    project_complete_flag_name: str,
    auto_upload: int,
    cleanup_slides_after_stage: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--project",
        project,
        "--base-data-dir",
        str(args.base_data_dir),
        "--index-json",
        str(args.index_json),
        "--reuse-coords-base",
        str(args.reuse_coords_base),
        "--dataset-id",
        str(dataset_id),
        "--features-subdir",
        str(features_subdir),
        "--upload-features-subdir",
        str(upload_subdir),
        "--upload-marker-name",
        str(upload_marker_name),
        "--project-complete-flag-name",
        str(project_complete_flag_name),
        "--auto-upload",
        str(int(auto_upload)),
    ]
    if args.auto_process_mismatch == 1:
        cmd.append("--process-missing")
    if args.strict_match:
        cmd.append("--strict-match")
    if args.no_summary:
        cmd.append("--no-summary")
    if cleanup_slides_after_stage:
        cmd.append("--delete-slides-after-project")
    return cmd


def run_stage(label: str, cmd: list[str]) -> int:
    print(f"[work:{label}] {' '.join(cmd)}")
    return subprocess.call(cmd)


def write_project_complete_flag(
    *,
    base_data_dir: Path,
    project: str,
    flag_name: str,
    virchow_rc: int,
    gigapath_rc: int,
) -> Path:
    flag_path = base_data_dir / project / flag_name
    payload = {
        "project": project,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "virchow2_gigapath_combined",
        "finished": True,
        "virchow2_exit_code": int(virchow_rc),
        "gigapath_exit_code": int(gigapath_rc),
        "both_ok": bool(virchow_rc == 0 and gigapath_rc == 0),
    }
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text(json.dumps(payload, indent=2) + "\n")
    return flag_path


def main() -> int:
    args = build_arg_parser().parse_args()

    run_list = selected_projects(args.only_projects, args.skip_projects)
    if not run_list:
        print("No projects selected after applying ONLY_PROJECTS/SKIP_PROJECTS filters.", file=sys.stderr)
        return 1

    print(f"Selected {len(run_list)} project(s): {' '.join(run_list)}")
    print(f"BASE_DATA_DIR={args.base_data_dir}")
    print(f"STRICT_MATCH={1 if args.strict_match else 0}")
    print(f"AUTO_PROCESS_MISMATCH={args.auto_process_mismatch}")
    print(f"CONTINUE_ON_ERROR={args.continue_on_error}")
    print(f"CLEANUP_SLIDES_AFTER_BOTH={args.cleanup_slides_after_both}")
    print(f"PROJECT_COMPLETE_FLAG_NAME={args.project_complete_flag_name}")

    passed = 0
    failed = 0

    for idx, project in enumerate(run_list, start=1):
        print("\n==============================")
        print(f"[project {idx}/{len(run_list)}] {project}")
        print("==============================")

        virchow_cmd = build_stage_cmd(
            script_path=VIRCHOW_SCRIPT,
            project=project,
            args=args,
            dataset_id=args.virchow_dataset_id,
            features_subdir=args.virchow_features_subdir,
            upload_subdir=args.virchow_upload_subdir,
            upload_marker_name=args.virchow_upload_marker_name,
            project_complete_flag_name=VIRCHOW_STAGE_FLAG_NAME,
            auto_upload=args.virchow_auto_upload,
            cleanup_slides_after_stage=False,
        )
        rc_v = run_stage("virchow2", virchow_cmd)
        if rc_v != 0:
            print(f"[fail] virchow2 stage finished nonzero rc={rc_v}", flush=True)
            print("[warn] continuing to gigapath so the combined project run can finish and write its final flag", flush=True)

        gigapath_cmd = build_stage_cmd(
            script_path=GIGAPATH_SCRIPT,
            project=project,
            args=args,
            dataset_id=args.gigapath_dataset_id,
            features_subdir=args.gigapath_features_subdir,
            upload_subdir=args.gigapath_upload_subdir,
            upload_marker_name=args.gigapath_upload_marker_name,
            project_complete_flag_name=GIGAPATH_STAGE_FLAG_NAME,
            auto_upload=args.gigapath_auto_upload,
            cleanup_slides_after_stage=(args.cleanup_slides_after_both == 1 and rc_v == 0),
        )
        rc_g = run_stage("gigapath", gigapath_cmd)
        if rc_g != 0:
            print(f"[fail] gigapath stage finished nonzero rc={rc_g}", flush=True)

        flag_path = write_project_complete_flag(
            base_data_dir=args.base_data_dir,
            project=project,
            flag_name=args.project_complete_flag_name,
            virchow_rc=rc_v,
            gigapath_rc=rc_g,
        )

        if rc_v == 0 and rc_g == 0:
            passed += 1
            print(f"[pass] combined project finished cleanly -> wrote {flag_path} -> continue")
            continue

        failed += 1
        print(
            f"[done] combined project finished with issues -> wrote {flag_path} "
            f"(virchow2_rc={rc_v} gigapath_rc={rc_g})",
            flush=True,
        )
        if args.continue_on_error != 1:
            print("[stop] CONTINUE_ON_ERROR=0")
            break

    print("\n[done] per-project sequential virchow2->gigapath sweep finished.")
    print(f"[summary] pass={passed} fail={failed} total={passed + failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
