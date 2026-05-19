from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from wsi_cf.steering.edit_policy import add_edit_policy_args, apply_edit_policy


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
    parser.add_argument("--preserve-edit-strength", type=float, default=0.0)
    parser.add_argument("--preserve-visited-strength", type=float, default=0.84)
    parser.add_argument("--preserve-fresh-context-strength", type=float, default=0.22)
    parser.add_argument("--edit-support", type=str, default="center_2x2", choices=["center_2x2", "border_relaxed"])
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
                },
                "preservation": {
                    "preserve_visited_strength": 0.7,
                },
                "planning": {
                    "edit_support": "border_relaxed",
                },
            }
        )
    )


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
    assert out.preserve_visited_strength == 0.7
    assert out.edit_support == "border_relaxed"
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
