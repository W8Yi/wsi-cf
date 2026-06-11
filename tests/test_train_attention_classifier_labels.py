from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts/train_attention_classifier.py"
    spec = importlib.util.spec_from_file_location("train_attention_classifier", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        if "libcusparseLt.so.0" in str(exc):
            pytest.skip("torch CUDA shared library is unavailable in this Python environment")
        raise
    return module


def test_filter_and_encode_labels_respects_explicit_label_order() -> None:
    script = load_script()
    rows = [
        {"case_id": "c1", "slide_key": "s1", "split": "train", "label_name": "high"},
        {"case_id": "c2", "slide_key": "s2", "split": "train", "label_name": "low"},
        {"case_id": "c3", "slide_key": "s3", "split": "test", "label_name": "high"},
        {"case_id": "c4", "slide_key": "s4", "split": "test", "label_name": "low"},
    ]

    encoded, label_to_id = script.filter_and_encode_labels(
        rows,
        min_slides_per_class=1,
        max_slides_per_class=0,
        max_train_slides=0,
        max_test_slides=0,
        seed=7,
        label_order=["low", "high"],
    )

    assert label_to_id == {"low": 0, "high": 1}
    assert {row["label_name"]: row["label_id"] for row in encoded} == {"low": 0, "high": 1}
