from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from wsi_cf.data.region_bank import (
    RegionBankRow,
    RegionRoleRow,
    build_experiment_manifest,
    build_region_pairs,
    assign_region_roles,
    export_region_bundle,
    make_region_bank_summary,
    parse_region_bank_csv,
    sample_random_tissue_region,
    sample_random_tissue_region_at_magnification,
    save_label_contact_sheets,
    validate_balanced_request,
    write_region_bank_csv,
)


class FakeSlide:
    def __init__(self, width: int = 4096, height: int = 3072, value: int = 120):
        self.dimensions = (width, height)
        self._img = Image.new("RGB", (width, height), (value, value, value))
        self.properties = {"openslide.objective-power": "20"}

    def read_region(self, loc, level, size):
        x0, y0 = loc
        w, h = size
        crop = self._img.crop((x0, y0, x0 + w, y0 + h)).convert("RGBA")
        return crop


def test_region_bank_row_parsing_preserves_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "region_bank.csv"
    rows = [
        {
            "region_id": "SLIDE__x_10__y_20",
            "split": "train",
            "label": 1,
            "hpv_status": "positive",
            "case_id": "CASE1",
            "slide_key": "SLIDE",
            "slide_path": "/tmp/slide.svs",
            "canonical_h5_path": "/tmp/slide.h5",
            "region_x": 10,
            "region_y": 20,
            "region_w": 1024,
            "region_h": 1024,
            "grid_step_px": 256,
            "feature_dim": 1536,
            "tissue_score": 0.9,
            "seed": 7,
            "image_path": "/tmp/region.png",
            "feature_grid_path": "/tmp/region_zgrid.npy",
            "cell_preview_path": "/tmp/region_cells.png",
            "region_dir": "/tmp/bundle",
        }
    ]
    write_region_bank_csv(csv_path, rows)
    parsed = parse_region_bank_csv(csv_path)
    assert parsed == [
        RegionBankRow(
            region_id="SLIDE__x_10__y_20",
            split="train",
            label=1,
            hpv_status="positive",
            case_id="CASE1",
            slide_key="SLIDE",
            slide_path="/tmp/slide.svs",
            canonical_h5_path="/tmp/slide.h5",
            region_x=10,
            region_y=20,
            region_w=1024,
            region_h=1024,
            grid_step_px=256,
            feature_dim=1536,
            tissue_score=0.9,
            seed=7,
            image_path="/tmp/region.png",
            feature_grid_path="/tmp/region_zgrid.npy",
            cell_preview_path="/tmp/region_cells.png",
            region_dir="/tmp/bundle",
        )
    ]


def test_region_sampler_returns_coordinates_within_bounds() -> None:
    slide = FakeSlide()
    rng = random.Random(7)
    x, y, score = sample_random_tissue_region(
        slide,
        region_size=1024,
        n_tries=8,
        min_tissue=0.35,
        rng=rng,
    )
    assert 0 <= x <= slide.dimensions[0] - 1024
    assert 0 <= y <= slide.dimensions[1] - 1024
    assert 0.0 <= score <= 1.0


def test_region_sampler_at_magnification_returns_coordinates_within_bounds() -> None:
    slide = FakeSlide()
    rng = random.Random(9)
    x, y, score, crop_w, crop_h = sample_random_tissue_region_at_magnification(
        slide,
        out_size=1024,
        target_magnification=10.0,
        n_tries=8,
        min_tissue=0.35,
        rng=rng,
    )
    assert 0 <= x <= slide.dimensions[0] - crop_w
    assert 0 <= y <= slide.dimensions[1] - crop_h
    assert crop_w == 2048
    assert crop_h == 2048
    assert 0.0 <= score <= 1.0


def test_balance_validation_and_role_assignment() -> None:
    eligible_rows = []
    for label in (0, 1):
        for idx in range(20):
            eligible_rows.append({"label": label, "slide_key": f"S{label}_{idx}"})
    validate_balanced_request(eligible_rows=eligible_rows, total_regions=40, regions_per_slide=1)

    rows = []
    for label in (0, 1):
        for idx in range(20):
            rows.append(
                {
                    "label": label,
                    "slide_key": f"S{label}_{idx:02d}",
                    "region_id": f"R{label}_{idx:02d}",
                }
            )
    roles = assign_region_roles(rows, sources_per_label=10, donors_per_label=10)
    assert sum(1 for row in roles if row["label"] == 0 and row["role"] == "source") == 10
    assert sum(1 for row in roles if row["label"] == 0 and row["role"] == "donor") == 10
    assert sum(1 for row in roles if row["label"] == 1 and row["role"] == "source") == 10
    assert sum(1 for row in roles if row["label"] == 1 and row["role"] == "donor") == 10


def test_region_pairing_and_experiment_manifest() -> None:
    rows = []
    for label in (0, 1):
        for rank in range(2):
            rows.append(
                RegionRoleRow(
                    region_id=f"SRC_{label}_{rank}",
                    split="train",
                    label=label,
                    hpv_status="positive" if label == 1 else "negative",
                    case_id=f"CASE_SRC_{label}_{rank}",
                    slide_key=f"SRC_SLIDE_{label}_{rank}",
                    slide_path=f"/tmp/src_{label}_{rank}.svs",
                    canonical_h5_path=f"/tmp/src_{label}_{rank}.h5",
                    region_x=0,
                    region_y=0,
                    region_w=1024,
                    region_h=1024,
                    grid_step_px=256,
                    feature_dim=1536,
                    tissue_score=0.8,
                    seed=7,
                    image_path=f"/tmp/src_{label}_{rank}.png",
                    feature_grid_path=f"/tmp/src_{label}_{rank}.npy",
                    cell_preview_path=f"/tmp/src_{label}_{rank}_cells.png",
                    region_dir=f"/tmp/src_{label}_{rank}",
                    role="source",
                    role_rank=rank,
                )
            )
            rows.append(
                RegionRoleRow(
                    region_id=f"DON_{label}_{rank}",
                    split="train",
                    label=label,
                    hpv_status="positive" if label == 1 else "negative",
                    case_id=f"CASE_DON_{label}_{rank}",
                    slide_key=f"DON_SLIDE_{label}_{rank}",
                    slide_path=f"/tmp/don_{label}_{rank}.svs",
                    canonical_h5_path=f"/tmp/don_{label}_{rank}.h5",
                    region_x=0,
                    region_y=0,
                    region_w=1024,
                    region_h=1024,
                    grid_step_px=256,
                    feature_dim=1536,
                    tissue_score=0.8,
                    seed=7,
                    image_path=f"/tmp/don_{label}_{rank}.png",
                    feature_grid_path=f"/tmp/don_{label}_{rank}.npy",
                    cell_preview_path=f"/tmp/don_{label}_{rank}_cells.png",
                    region_dir=f"/tmp/don_{label}_{rank}",
                    role="donor",
                    role_rank=rank,
                )
            )
    pairs = build_region_pairs(rows)
    assert len(pairs) == 4
    first_neg = next(row for row in pairs if row["source_label"] == 0 and row["source_role_rank"] == 0)
    assert first_neg["same_label_donor_region_id"] == "DON_0_0"
    assert first_neg["cross_label_donor_region_id"] == "DON_1_0"

    manifest = build_experiment_manifest(pairs[:1])
    assert len(manifest) == 5
    assert {row["condition"] for row in manifest} == {
        "baseline",
        "same_label_one_cell",
        "cross_label_one_cell",
        "same_label_full_grid",
        "cross_label_full_grid",
    }


def test_balance_validation_fails_when_request_cannot_be_satisfied() -> None:
    eligible_rows = [{"label": 0, "slide_key": f"N{i}"} for i in range(20)] + [{"label": 1, "slide_key": "P0"}]
    try:
        validate_balanced_request(eligible_rows=eligible_rows, total_regions=40, regions_per_slide=1)
    except ValueError as exc:
        assert "Cannot satisfy balanced export" in str(exc)
    else:
        raise AssertionError("Expected validate_balanced_request to fail")


def test_export_region_bundle_saves_aligned_artifacts_and_metadata(tmp_path: Path) -> None:
    region_img = Image.new("RGB", (1024, 1024), (140, 100, 120))
    row = {
        "split": "train",
        "label": 1,
        "hpv_status": "positive",
        "case_id": "CASE1",
        "slide_key": "SLIDE1",
        "slide_path": "/tmp/slide1.svs",
        "canonical_h5_path": "/tmp/slide1.h5",
    }

    def fake_build_feature_grid(img: Image.Image) -> np.ndarray:
        assert img.size == (1024, 1024)
        return np.zeros((4, 4, 1536), dtype=np.float32)

    meta = export_region_bundle(
        region_img=region_img,
        out_dir=tmp_path,
        region_id="SLIDE1__x_10__y_20",
        row=row,
        region_x=10,
        region_y=20,
        region_size=1024,
        grid_step_px=256,
        tissue_score=0.8,
        seed=7,
        build_feature_grid=fake_build_feature_grid,
    )
    assert Path(meta["image_path"]).exists()
    assert Path(meta["feature_grid_path"]).exists()
    assert Path(meta["cell_preview_path"]).exists()
    assert Path(meta["region_dir"]).exists()
    grid = np.load(meta["feature_grid_path"])
    assert grid.shape == (4, 4, 1536)
    meta_json = json.loads((Path(meta["region_dir"]) / "region_meta.json").read_text())
    assert meta_json["grid_step_px"] == 256
    assert meta_json["feature_dim"] == 1536


def test_region_summary_and_contact_sheets(tmp_path: Path) -> None:
    rows = []
    roles = []
    for label in (0, 1):
        for idx in range(2):
            img_path = tmp_path / f"label_{label}_{idx}.png"
            Image.new("RGB", (128, 128), (100 + 10 * idx, 90, 80)).save(img_path)
            row = {
                "region_id": f"R{label}_{idx}",
                "split": "train",
                "label": label,
                "hpv_status": "positive" if label == 1 else "negative",
                "case_id": f"C{label}_{idx}",
                "slide_key": f"S{label}_{idx}",
                "slide_path": f"/tmp/S{label}_{idx}.svs",
                "canonical_h5_path": f"/tmp/S{label}_{idx}.h5",
                "region_x": idx,
                "region_y": idx,
                "region_w": 1024,
                "region_h": 1024,
                "grid_step_px": 256,
                "feature_dim": 1536,
                "tissue_score": 0.75,
                "seed": 7,
                "image_path": str(img_path),
                "feature_grid_path": f"/tmp/S{label}_{idx}.npy",
                "cell_preview_path": f"/tmp/S{label}_{idx}_cells.png",
                "region_dir": f"/tmp/R{label}_{idx}",
            }
            rows.append(row)
            roles.append({**row, "role": "source" if idx == 0 else "donor", "role_rank": idx})
    out_csv = tmp_path / "region_bank.csv"
    out_roles_csv = tmp_path / "region_roles.csv"
    summary = make_region_bank_summary(rows=rows, roles=roles, out_csv=out_csv, out_roles_csv=out_roles_csv)
    assert summary["n_regions"] == 4
    assert summary["label_counts"] == {"0": 2, "1": 2}
    assert summary["role_counts"]["source_label_0"] == 1
    assert summary["role_counts"]["donor_label_1"] == 1
    save_label_contact_sheets(rows=rows, out_dir=tmp_path)
    assert (tmp_path / "label_0_contact_sheet.png").exists()
    assert (tmp_path / "label_1_contact_sheet.png").exists()
