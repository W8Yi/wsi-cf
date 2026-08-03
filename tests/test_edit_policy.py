from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from wsi_cf.steering.edit_policy import DEFAULT_EDIT_POLICY, add_edit_policy_args, apply_edit_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_edit_policy_args(parser)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--patch-batch", type=int, default=256)
    parser.add_argument("--prototype-strength", type=float, default=0.9)
    parser.add_argument("--steer-blend", type=float, default=1.0)
    parser.add_argument("--mid-steer-start-ratio", type=float, default=0.55)
    parser.add_argument("--mid-steer-end-ratio", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-start", type=float, default=0.4)
    parser.add_argument("--mid-steer-alpha-end", type=float, default=1.0)
    parser.add_argument("--mid-steer-alpha-schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--steer-context-halo-weight", type=float, default=0.0)
    parser.add_argument("--steer-context-halo-radius-cells", type=int, default=1)
    parser.add_argument("--preserve-edit-strength", type=float, default=0.0)
    parser.add_argument("--preserve-visited-strength", type=float, default=0.84)
    parser.add_argument("--preserve-fresh-context-strength", type=float, default=0.22)
    parser.add_argument("--edit-support", type=str, default="padded_center_2x2", choices=["center_2x2", "padded_center_2x2", "border_relaxed"])
    parser.add_argument("--context-halo-cells", type=int, default=1)
    parser.add_argument("--window-stride-cells", type=int, default=2)
    parser.add_argument("--window-selection-mode", type=str, default="coverage", choices=["coverage", "overlap"])
    parser.add_argument("--steer-full-support", action="store_true")
    parser.add_argument("--preserve-full-support", action="store_true")
    parser.add_argument("--commit-mode", type=str, default="support_cells", choices=["full_window", "support_cells", "edit_cells"])
    parser.add_argument("--commit-feather-px", type=int, default=64)
    parser.add_argument("--commit-halo-cells", type=int, default=0)
    parser.add_argument("--commit-halo-alpha", type=float, default=0.0)
    return parser


def write_policy(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "policy_name": "test",
                "version": 1,
                "generation": {
                    "steps": 40,
                    "guidance": 3.0,
                    "patch_batch": 128,
                },
                "steering": {
                    "prototype_strength": 1.1,
                    "mid_steer_alpha_schedule": "cosine",
                    "steer_context_halo_weight": 0.25,
                    "steer_context_halo_radius_cells": 1,
                },
                "preservation": {
                    "preserve_visited_strength": 0.7,
                },
                "planning": {
                    "edit_support": "border_relaxed",
                    "context_halo_cells": 2,
                    "window_stride_cells": 1,
                    "window_selection_mode": "overlap",
                    "steer_full_support": True,
                    "preserve_full_support": True,
                    "commit_mode": "edit_cells",
                    "commit_feather_px": 96,
                    "commit_halo_cells": 1,
                    "commit_halo_alpha": 0.35,
                },
            }
        )
    )


def test_policy09_is_the_repository_default() -> None:
    parser = build_parser()
    assert parser.parse_args([]).edit_policy == DEFAULT_EDIT_POLICY
    assert parser.parse_args(["--no-edit-policy"]).edit_policy is None


def test_repository_default_policy09_applies() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    root = Path(__file__).resolve().parents[1]

    out = apply_edit_policy(args, parser=parser, argv=[], root=root)

    assert out.edit_policy == root / DEFAULT_EDIT_POLICY
    assert out.preserve_visited_strength == 0.95
    assert out.preserve_fresh_context_strength == 0.75
    assert out.preserve_full_support is True
    assert out.commit_mode == "full_window"


def test_apply_edit_policy_fills_values(tmp_path: Path) -> None:
    parser = build_parser()
    policy = tmp_path / "policy.json"
    write_policy(policy)
    argv = ["--edit-policy", str(policy)]
    args = parser.parse_args(argv)

    out = apply_edit_policy(args, parser=parser, argv=argv, root=tmp_path)

    assert out.steps == 40
    assert out.guidance == 3.0
    assert out.patch_batch == 128
    assert out.prototype_strength == 1.1
    assert out.mid_steer_alpha_schedule == "cosine"
    assert out.steer_context_halo_weight == 0.25
    assert out.steer_context_halo_radius_cells == 1
    assert out.preserve_visited_strength == 0.7
    assert out.edit_support == "border_relaxed"
    assert out.context_halo_cells == 2
    assert out.window_stride_cells == 1
    assert out.window_selection_mode == "overlap"
    assert out.steer_full_support is True
    assert out.preserve_full_support is True
    assert out.commit_mode == "edit_cells"
    assert out.commit_feather_px == 96
    assert out.commit_halo_cells == 1
    assert out.commit_halo_alpha == 0.35
    assert out.edit_policy_applied["steps"] == 40
    assert out.edit_policy_overridden_by_cli == {}


def test_cli_flags_override_edit_policy(tmp_path: Path) -> None:
    parser = build_parser()
    policy = tmp_path / "policy.json"
    write_policy(policy)
    argv = [
        "--edit-policy",
        str(policy),
        "--prototype-strength",
        "1.2",
        "--steps=12",
    ]
    args = parser.parse_args(argv)

    out = apply_edit_policy(args, parser=parser, argv=argv, root=tmp_path)

    assert out.prototype_strength == 1.2
    assert out.steps == 12
    assert out.guidance == 3.0
    assert out.edit_policy_overridden_by_cli["prototype_strength"] == 1.1
    assert out.edit_policy_overridden_by_cli["steps"] == 40


def test_edit_policy_rejects_unknown_keys(tmp_path: Path) -> None:
    parser = build_parser()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"generation": {"unknown": 1}}))
    argv = ["--edit-policy", str(policy)]
    args = parser.parse_args(argv)

    with pytest.raises(ValueError, match="Unknown edit policy key"):
        apply_edit_policy(args, parser=parser, argv=argv, root=tmp_path)
