from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_EDIT_POLICY = Path(
    "configs/edit_policies/transition_ablation/09_full_window_regen_center_preserve_outer.json"
)


EDIT_POLICY_FIELDS: dict[str, dict[str, str]] = {
    "generation": {
        "steps": "steps",
        "guidance": "guidance",
        "patch_batch": "patch_batch",
    },
    "steering": {
        "prototype_strength": "prototype_strength",
        "steer_blend": "steer_blend",
        "mid_steer_start_ratio": "mid_steer_start_ratio",
        "mid_steer_end_ratio": "mid_steer_end_ratio",
        "mid_steer_alpha_start": "mid_steer_alpha_start",
        "mid_steer_alpha_end": "mid_steer_alpha_end",
        "mid_steer_alpha_schedule": "mid_steer_alpha_schedule",
        "steer_context_halo_weight": "steer_context_halo_weight",
        "steer_context_halo_radius_cells": "steer_context_halo_radius_cells",
    },
    "preservation": {
        "preserve_edit_strength": "preserve_edit_strength",
        "preserve_visited_strength": "preserve_visited_strength",
        "preserve_fresh_context_strength": "preserve_fresh_context_strength",
    },
    "planning": {
        "edit_support": "edit_support",
        "context_halo_cells": "context_halo_cells",
        "window_stride_cells": "window_stride_cells",
        "window_selection_mode": "window_selection_mode",
        "steer_full_support": "steer_full_support",
        "preserve_full_support": "preserve_full_support",
        "commit_mode": "commit_mode",
        "commit_feather_px": "commit_feather_px",
        "commit_halo_cells": "commit_halo_cells",
        "commit_halo_alpha": "commit_halo_alpha",
    },
}


def add_edit_policy_args(
    parser: argparse.ArgumentParser,
    *,
    default: Path | None = DEFAULT_EDIT_POLICY,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--edit-policy",
        type=Path,
        dest="edit_policy",
        help=(
            f"JSON edit policy (default: {DEFAULT_EDIT_POLICY}). "
            "CLI flags explicitly provided after/before this option override policy values."
        ),
    )
    group.add_argument(
        "--no-edit-policy",
        action="store_const",
        const=None,
        dest="edit_policy",
        help="Disable the repository default edit policy and use raw CLI defaults.",
    )
    parser.set_defaults(edit_policy=default)


def explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        if action.dest == argparse.SUPPRESS:
            continue
        for option in action.option_strings:
            option_to_dest[str(option)] = str(action.dest)
    explicit: set[str] = set()
    for token in argv:
        text = str(token)
        option = text.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit.add(dest)
    return explicit


def coerce_policy_value(parser: argparse.ArgumentParser, dest: str, value: Any) -> Any:
    action = next((candidate for candidate in parser._actions if candidate.dest == dest), None)
    if action is None:
        raise ValueError(f"Edit policy maps to unknown CLI destination: {dest}")
    if action.choices is not None and value not in action.choices:
        raise ValueError(f"Invalid edit policy value for {dest}: {value!r}. Expected one of {list(action.choices)}")
    if action.type is not None and value is not None:
        return action.type(value)
    return value


def apply_edit_policy(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    argv: list[str],
    root: Path,
) -> argparse.Namespace:
    args.edit_policy_config = None
    args.edit_policy_applied = {}
    args.edit_policy_overridden_by_cli = {}
    if args.edit_policy is None:
        return args

    policy_path = Path(args.edit_policy)
    if not policy_path.is_absolute():
        policy_path = Path(root) / policy_path
    with policy_path.open("r") as handle:
        policy = json.load(handle)
    if not isinstance(policy, dict):
        raise ValueError(f"Edit policy must be a JSON object: {policy_path}")

    known_sections = set(EDIT_POLICY_FIELDS) | {"policy_name", "description", "version"}
    unknown_sections = sorted(str(key) for key in policy if str(key) not in known_sections)
    if unknown_sections:
        raise ValueError(f"Unknown edit policy section(s) in {policy_path}: {unknown_sections}")

    explicit_dests = explicit_cli_dests(parser, argv)
    applied: dict[str, Any] = {}
    overridden: dict[str, Any] = {}
    for section, fields in EDIT_POLICY_FIELDS.items():
        payload = policy.get(section, {})
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"Edit policy section {section!r} must be an object in {policy_path}")
        unknown_keys = sorted(str(key) for key in payload if str(key) not in fields)
        if unknown_keys:
            raise ValueError(f"Unknown edit policy key(s) in section {section!r}: {unknown_keys}")
        for key, dest in fields.items():
            if key not in payload:
                continue
            value = coerce_policy_value(parser, dest, payload[key])
            if dest in explicit_dests:
                overridden[dest] = value
                continue
            setattr(args, dest, value)
            applied[dest] = value

    args.edit_policy = policy_path
    args.edit_policy_config = policy
    args.edit_policy_applied = applied
    args.edit_policy_overridden_by_cli = overridden
    return args
