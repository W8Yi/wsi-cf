#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_tcga_project_uni2h import (
    BASE_DATA_DIR_DEFAULT,
    INDEX_JSON_DEFAULT,
    PROJECTS,
    check_project,
    load_index,
    parse_bool_env,
    parse_csv_upper,
    write_pipeline_summary,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
ONE_PROJECT_SCRIPT = ROOT_DIR / "scripts" / "tcga_project_pipeline" / "run_tcga_project_uni2h.py"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run feature checks across selected TCGA projects. "
            "When AUTO_PROCESS_MISMATCH=1 (default), mismatches start processing automatically."
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
        "--continue-on-error",
        type=int,
        choices=[0, 1],
        default=1 if parse_bool_env("CONTINUE_ON_ERROR", True) else 0,
        help="When 1, keep checking remaining projects after a mismatch.",
    )
    ap.add_argument(
        "--auto-process-mismatch",
        type=int,
        choices=[0, 1],
        default=1 if parse_bool_env("AUTO_PROCESS_MISMATCH", True) else 0,
        help="When 1, start processing mismatched projects automatically.",
    )
    ap.add_argument(
        "--auto-upload",
        type=int,
        choices=[0, 1],
        default=1 if parse_bool_env("AUTO_UPLOAD", True) else 0,
        help="When 1, project-level script uploads if marker is missing.",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write per-project pipeline_summary.json files.",
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


def run_project_script(project: str, args: argparse.Namespace, process_missing: bool) -> int:
    cmd = [
        sys.executable,
        str(ONE_PROJECT_SCRIPT),
        "--project",
        project,
        "--base-data-dir",
        str(args.base_data_dir),
        "--index-json",
        str(args.index_json),
    ]
    if process_missing:
        cmd.append("--process-missing")
    if args.strict_match:
        cmd.append("--strict-match")
    if args.no_summary:
        cmd.append("--no-summary")
    cmd.extend(["--auto-upload", str(args.auto_upload)])

    mode = "process-missing" if process_missing else "check/upload"
    print(f"[work] run project script ({mode}) for {project}")
    return subprocess.call(cmd)


def main() -> int:
    args = build_arg_parser().parse_args()

    run_list = selected_projects(args.only_projects, args.skip_projects)
    if not run_list:
        print("No projects selected after applying ONLY_PROJECTS/SKIP_PROJECTS filters.", file=sys.stderr)
        return 1

    try:
        index_payload = load_index(args.index_json)
    except Exception as exc:
        print(f"Failed to load index JSON: {args.index_json} ({exc})", file=sys.stderr)
        return 1

    print(f"Selected {len(run_list)} project(s): {' '.join(run_list)}")
    print(f"BASE_DATA_DIR={args.base_data_dir}")
    print(f"STRICT_MATCH={1 if args.strict_match else 0}")
    print(f"AUTO_PROCESS_MISMATCH={args.auto_process_mismatch}")
    print(f"AUTO_UPLOAD={args.auto_upload}")
    print(f"CONTINUE_ON_ERROR={args.continue_on_error}")

    passed = 0
    failed = 0

    for idx, project in enumerate(run_list, start=1):
        print("\n==============================")
        print(f"[project {idx}/{len(run_list)}] {project}")
        print("==============================")

        result = check_project(index_payload, args.base_data_dir, project, strict_match=args.strict_match)
        if not args.no_summary:
            summary_path = write_pipeline_summary(result, strict_match=args.strict_match)
            print(f"[summary] wrote {summary_path}")

        print(
            f"[check] expected={result.expected_total} features={result.feature_total} "
            f"matched={result.matched_total} missing={len(result.missing)} extra={len(result.extra)}"
        )
        if result.missing:
            print(
                f"[missing] first {min(10, len(result.missing))}/{len(result.missing)}: "
                f"{' '.join(result.missing[:10])}"
            )
        if result.extra:
            print(f"[extra] first {min(10, len(result.extra))}/{len(result.extra)}: {' '.join(result.extra[:10])}")

        if result.matched:
            rc = run_project_script(project, args, process_missing=False)
            if rc == 0:
                passed += 1
                print("[pass] matched -> continue")
                continue
            print(f"[fail] check/upload failed rc={rc}")
            failed += 1
            if args.continue_on_error != 1:
                print("[stop] CONTINUE_ON_ERROR=0")
                break
            continue

        if args.auto_process_mismatch == 1 and result.missing:
            rc = run_project_script(project, args, process_missing=True)
            if rc == 0:
                passed += 1
                print("[pass] recovered after processing -> continue")
                continue
            print(f"[fail] processing failed rc={rc}")
        else:
            print("[fail] mismatch")

        failed += 1
        if args.continue_on_error != 1:
            print("[stop] CONTINUE_ON_ERROR=0")
            break

    print("\n[done] feature-check sweep finished.")
    print(f"[summary] pass={passed} fail={failed} total={passed + failed}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
