#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_outputs"
DEFAULT_RULES = REPO_ROOT / "paper" / "artifact_rules.json"
DEFAULT_INDEX_CSV = DEFAULT_OUTPUT_ROOT / "artifact_index.csv"
DEFAULT_CURRENT_VIEW = DEFAULT_OUTPUT_ROOT / "current"

INDEX_COLUMNS = [
    "path",
    "size_bytes",
    "size_human",
    "mtime",
    "artifact_class",
    "keep_level",
    "paper_status",
    "notes",
]


@dataclass(frozen=True)
class ClassifiedArtifact:
    path: Path
    size_bytes: int
    mtime: str
    artifact_class: str
    keep_level: str
    paper_status: str
    notes: str
    paper_view: str | None = None

    @property
    def rel_path(self) -> str:
        return display_path(self.path)

    def to_row(self) -> dict[str, str]:
        return {
            "path": self.rel_path,
            "size_bytes": str(self.size_bytes),
            "size_human": human_size(self.size_bytes),
            "mtime": self.mtime,
            "artifact_class": self.artifact_class,
            "keep_level": self.keep_level,
            "paper_status": self.paper_status,
            "notes": self.notes,
        }


def human_size(size: int) -> str:
    units = ["B", "K", "M", "G", "T", "P"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}B"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size}B"


def iso_mtime(path: Path) -> str:
    stat = path.stat()
    return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        if path.parent.name == "artifacts":
            return f"artifacts/{path.name}"
        return path.as_posix()


def directory_size(path: Path) -> int:
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size

    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in files:
            item = root_path / name
            try:
                total += item.lstat().st_size if item.is_symlink() else item.stat().st_size
            except OSError:
                continue
        for name in dirs:
            item = root_path / name
            if item.is_symlink():
                try:
                    total += item.lstat().st_size
                except OSError:
                    continue
    return total


def load_rules(path: Path) -> dict[str, Any]:
    with path.open() as f:
        rules = json.load(f)
    if "rules" not in rules or "default_rule" not in rules:
        raise ValueError(f"Invalid artifact rules file: {path}")
    return rules


def classify_path(path: Path, rules: dict[str, Any]) -> dict[str, Any]:
    rel = display_path(path)
    name = path.name
    rule_items = list(rules["rules"])

    exact_rules = [rule for rule in rule_items if not has_glob(str(rule["pattern"]))]
    scratch_rules = [
        rule
        for rule in rule_items
        if str(rule.get("keep_level")) == "scratch" and rule not in exact_rules
    ]
    remaining_rules = [rule for rule in rule_items if rule not in exact_rules and rule not in scratch_rules]

    for rule in [*exact_rules, *scratch_rules, *remaining_rules]:
        pattern = str(rule["pattern"])
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
            return rule
    return rules["default_rule"]


def has_glob(pattern: str) -> bool:
    return re.search(r"[*?\[]", pattern) is not None


def scan_artifacts(artifacts_root: Path, rules_path: Path) -> list[ClassifiedArtifact]:
    rules = load_rules(rules_path)
    if not artifacts_root.exists():
        return []

    artifacts: list[ClassifiedArtifact] = []
    for path in sorted(artifacts_root.iterdir(), key=lambda p: p.name):
        rule = classify_path(path, rules)
        artifacts.append(
            ClassifiedArtifact(
                path=path,
                size_bytes=directory_size(path),
                mtime=iso_mtime(path),
                artifact_class=str(rule["artifact_class"]),
                keep_level=str(rule["keep_level"]),
                paper_status=str(rule["paper_status"]),
                notes=str(rule.get("notes", "")),
                paper_view=rule.get("paper_view"),
            )
        )
    return artifacts


def write_index_csv(artifacts: list[ClassifiedArtifact], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(artifact.to_row())


def write_index_json(artifacts: list[ClassifiedArtifact], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([artifact.to_row() for artifact in artifacts], f, indent=2)
        f.write("\n")


def read_index_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def print_report(rows: list[dict[str, str]], *, limit: int) -> None:
    total_size = sum(int(row["size_bytes"]) for row in rows)
    print(f"Artifacts: {len(rows)}")
    print(f"Total indexed size: {human_size(total_size)}")
    print()

    for label in ["keep_level", "artifact_class", "paper_status"]:
        print(f"By {label}:")
        counts = Counter(row[label] for row in rows)
        sizes: Counter[str] = Counter()
        for row in rows:
            sizes[row[label]] += int(row["size_bytes"])
        for key, count in sorted(counts.items()):
            print(f"  {key}: {count} ({human_size(sizes[key])})")
        print()

    print(f"Largest {limit}:")
    largest = sorted(rows, key=lambda row: int(row["size_bytes"]), reverse=True)[:limit]
    for row in largest:
        print(f"  {row['size_human']:>8}  {row['keep_level']:<18}  {row['path']}")


def rows_from_scan_or_index(args: argparse.Namespace) -> list[dict[str, str]]:
    index = Path(args.index)
    if getattr(args, "from_index", False):
        return read_index_csv(index)
    artifacts = scan_artifacts(Path(args.artifacts_root), Path(args.rules))
    return [artifact.to_row() for artifact in artifacts]


def command_scan(args: argparse.Namespace) -> int:
    artifacts = scan_artifacts(Path(args.artifacts_root), Path(args.rules))
    if args.csv:
        write_index_csv(artifacts, Path(args.csv))
        print(f"[scan] wrote CSV index: {args.csv}")
    if args.json:
        write_index_json(artifacts, Path(args.json))
        print(f"[scan] wrote JSON index: {args.json}")
    print(f"[scan] indexed {len(artifacts)} top-level artifacts")
    return 0


def command_report(args: argparse.Namespace) -> int:
    rows = rows_from_scan_or_index(args)
    print_report(rows, limit=int(args.limit))
    return 0


def command_plan_archive(args: argparse.Namespace) -> int:
    rows = rows_from_scan_or_index(args)
    candidates = [
        row
        for row in rows
        if row["keep_level"] in {"archive_candidate", "scratch", "delete_candidate"}
    ]
    candidates.sort(key=lambda row: (row["keep_level"], -int(row["size_bytes"]), row["path"]))
    if not candidates:
        print("[plan-archive] no archive/delete candidates found")
        return 0

    print("# Non-destructive archive/delete plan")
    print("# Review these entries before running any move or delete command.")
    print()
    for row in candidates:
        bucket = row["keep_level"]
        archive_path = f"artifacts_archive/{bucket}/{Path(row['path']).name}"
        print(f"# {row['size_human']} | {row['artifact_class']} | {row['notes']}")
        if bucket == "delete_candidate":
            print(f"# delete candidate: {row['path']}")
        else:
            print(f"mkdir -p artifacts_archive/{bucket}")
            print(f"mv {row['path']} {archive_path}")
        print()
    return 0


def safe_symlink(source: Path, link: Path, *, dry_run: bool) -> str:
    source = source.resolve()
    link_parent = link.parent
    rel_source = os.path.relpath(source, start=link_parent.resolve() if link_parent.exists() else link_parent)
    if link.exists() or link.is_symlink():
        if link.is_symlink() and os.readlink(link) == rel_source:
            return f"[skip] existing link: {link}"
        return f"[skip] destination exists and will not be overwritten: {link}"
    if dry_run:
        return f"[dry-run] ln -s {rel_source} {link}"
    link_parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(rel_source)
    return f"[link] {link} -> {rel_source}"


def command_make_paper_view(args: argparse.Namespace) -> int:
    rules = load_rules(Path(args.rules))
    view_dirs = list(rules.get("paper_view_dirs", []))
    view_root = Path(args.view_root)
    artifacts = scan_artifacts(Path(args.artifacts_root), Path(args.rules))
    selected = [artifact for artifact in artifacts if artifact.paper_view]

    if args.dry_run:
        print(f"[dry-run] would prepare paper view: {view_root}")
    else:
        view_root.mkdir(parents=True, exist_ok=True)
    for dirname in view_dirs:
        target = view_root / dirname
        if args.dry_run:
            print(f"[dry-run] mkdir -p {target}")
        else:
            target.mkdir(parents=True, exist_ok=True)

    for artifact in selected:
        link = view_root / str(artifact.paper_view) / artifact.path.name
        print(safe_symlink(artifact.path, link, dry_run=bool(args.dry_run)))

    print(f"[paper-view] selected {len(selected)} artifacts")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage paper-facing artifact indexes and symlink views.")
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
        subparser.add_argument("--rules", type=Path, default=DEFAULT_RULES)

    scan = subparsers.add_parser("scan", help="Scan top-level artifacts and write an index.")
    add_common(scan)
    scan.add_argument("--csv", type=Path, default=DEFAULT_INDEX_CSV)
    scan.add_argument("--json", type=Path, default=None)
    scan.set_defaults(func=command_scan)

    report = subparsers.add_parser("report", help="Print a size and classification summary.")
    add_common(report)
    report.add_argument("--index", type=Path, default=DEFAULT_INDEX_CSV)
    report.add_argument("--from-index", action="store_true", help="Read an existing CSV index instead of rescanning.")
    report.add_argument("--limit", type=int, default=20)
    report.set_defaults(func=command_report)

    archive = subparsers.add_parser("plan-archive", help="Print a non-destructive archive/delete plan.")
    add_common(archive)
    archive.add_argument("--index", type=Path, default=DEFAULT_INDEX_CSV)
    archive.add_argument("--from-index", action="store_true", help="Read an existing CSV index instead of rescanning.")
    archive.set_defaults(func=command_plan_archive)

    view = subparsers.add_parser("make-paper-view", help="Create a symlink view of paper-relevant artifacts.")
    add_common(view)
    view.add_argument("--view-root", type=Path, default=DEFAULT_CURRENT_VIEW)
    view.add_argument("--dry-run", action="store_true")
    view.set_defaults(func=command_make_paper_view)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
