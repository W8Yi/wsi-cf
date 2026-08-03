from __future__ import annotations

import argparse
from pathlib import Path

from conftest import load_script_module


def test_streaming_cache_key_changes_when_policy_changes(tmp_path: Path) -> None:
    script = load_script_module("run_streamed_full_test_benchmark.py")
    policy_a = tmp_path / "policy_a.json"
    policy_b = tmp_path / "policy_b.json"
    policy_a.write_text('{"policy": "a"}')
    policy_b.write_text('{"policy": "b"}')
    args = argparse.Namespace(
        task_name="task",
        direction_name="a_to_b",
        ours_policy=policy_a,
        naive_policy=policy_b,
        sae_variant="relu_sae_base",
        steps=30,
        patch_batch=256,
        edit_support="padded_center_2x2",
        window_stride_cells=1,
        window_selection_mode="overlap",
        commit_mode="full_window",
    )
    request = {
        "run_id": "run_001",
        "region_id": "region_001",
        "selector": "attention",
        "repeat_id": -1,
        "budget": 8,
        "target_cells": [{"gx": 0, "gy": 0}],
    }

    first = script.request_cache_key(args, request, method="ours")
    args.ours_policy.write_text('{"policy": "a", "changed": true}')
    second = script.request_cache_key(args, request, method="ours")

    assert first != second
    assert script.request_cache_key(args, request, method="ours") != script.request_cache_key(args, request, method="bad_naive")


def test_add_provenance_preserves_visual_method(tmp_path: Path) -> None:
    script = load_script_module("run_streamed_full_test_benchmark.py")
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    args = argparse.Namespace(
        task_name="task",
        direction_name="a_to_b",
        ours_policy=policy,
        naive_policy=policy,
        sae_variant="relu_sae_base",
        steps=30,
        patch_batch=256,
        edit_support="padded_center_2x2",
        window_stride_cells=1,
        window_selection_mode="overlap",
        commit_mode="full_window",
    )
    rows = [{"run_id": "run_001", "method": "bad_naive", "budget": 8}]
    requests = {"run_001": {"run_id": "run_001", "target_cells": []}}

    out = script.add_provenance(args, rows, requests, method="")

    assert out[0]["method"] == "bad_naive"
    assert out[0]["generation_mode"] == "cumulative_checkpoint"
    assert out[0]["trajectory_cache_key"]


def test_copy_legacy_run_dirs_reuses_only_complete_runs(tmp_path: Path) -> None:
    script = load_script_module("run_streamed_full_test_benchmark.py")
    legacy = tmp_path / "legacy"
    tmp = tmp_path / "tmp"
    complete = legacy / "run_complete"
    incomplete = legacy / "run_incomplete"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    (complete / "generated.png").write_bytes(b"png")
    (complete / "run_manifest.json").write_text("{}")
    (incomplete / "generated.png").write_bytes(b"png")
    requests = [{"run_id": "run_complete"}, {"run_id": "run_incomplete"}, {"run_id": "run_missing"}]

    copied, missing = script.copy_legacy_run_dirs(legacy, tmp, requests)

    assert [row["run_id"] for row in copied] == ["run_complete"]
    assert [row["run_id"] for row in missing] == ["run_incomplete", "run_missing"]
    assert (tmp / "run_complete" / "generated.png").exists()


def test_denominator_audit_summarizes_region_bank_coverage(monkeypatch) -> None:
    script = load_script_module("build_full_test_benchmark_audit.py")

    monkeypatch.setattr(
        script,
        "direction_specs",
        lambda: [
            {
                "task_name": "toy",
                "direction": "a_to_b",
                "source_label": "a",
                "target_label": "b",
                "region_bank_csv": "toy_region_bank.csv",
            }
        ],
    )
    monkeypatch.setattr(
        script,
        "eligible_for_spec",
        lambda _spec: [
            {"task_name": "toy", "slide_key": "slide_1", "case_id": "case_1", "source_label": "a", "split": "test", "h5_path": ""},
            {"task_name": "toy", "slide_key": "slide_2", "case_id": "case_2", "source_label": "a", "split": "test", "h5_path": ""},
        ],
    )
    monkeypatch.setattr(
        script,
        "region_rows_by_slide",
        lambda _path: {"slide_1": [{"region_id": "r1"}, {"region_id": "r2"}]},
    )

    payload = script.build_audit({"toy/a_to_b"})

    assert payload["summary"][0]["eligible_test_slides"] == 2
    assert payload["summary"][0]["included_slides_with_regions"] == 1
    assert payload["summary"][0]["excluded_slides_without_regions"] == 1
    assert payload["summary"][0]["region_bank_rows"] == 2
    assert {row["audit_status"] for row in payload["audit"]} == {"included", "excluded"}
