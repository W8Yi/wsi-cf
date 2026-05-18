#!/usr/bin/env bash
set -euo pipefail

cd /common/users/wq50/wsi_cf

PY="${PY:-/common/users/wq50/envs/pace/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-17}"

OUT_DIR="${OUT_DIR:-artifacts/kirc_random_batch_topk_concepts_1024_relu_steer}"
PREP_DIR="${PREP_DIR:-${OUT_DIR}/prepared_random_1024}"

READY_SLIDE_ROOT="${READY_SLIDE_ROOT:-/research/projects/mllab/WSI/.tmp/ready_buffer/slides/TCGA-KIRC}"
TASK_MANIFEST="${TASK_MANIFEST:-artifacts/classifier_training/kirc_low_vs_high_grade/task_manifest.csv}"

BATCH_TOPK_SAE_CKPT="${BATCH_TOPK_SAE_CKPT:-/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_20x_interp/batch_topk_final.pt}"
BATCH_TOPK_SAE_CFG="${BATCH_TOPK_SAE_CFG:-/common/users/wq50/SAE_path/runs/tcga_sae_batch_topk_20x_interp/run_config.json}"
RELU_SAE_CKPT="${RELU_SAE_CKPT:-resources/models/sae/tcga_uni2_sae_relu_v1/relu_final.pt}"
RELU_SAE_CFG="${RELU_SAE_CFG:-resources/models/sae/tcga_uni2_sae_relu_v1/run_config.json}"

TARGET_MAG="${TARGET_MAG:-20}"
REGION_SIZE="${REGION_SIZE:-1024}"
GRID_STEP_PX="${GRID_STEP_PX:-256}"
N_CONCEPTS="${N_CONCEPTS:-10}"
TOP_TILES_PER_CONCEPT="${TOP_TILES_PER_CONCEPT:-5}"
RUN_MODE="${RUN_MODE:-per_concept}"

# For speed, scan only a small random slice of the KIRC feature manifest and a
# random pool of batch-TopK latents, then choose active concepts from that pool.
CONCEPT_SCAN_MAX_SLIDES="${CONCEPT_SCAN_MAX_SLIDES:-12}"
RANDOM_BATCH_TOPK_POOL="${RANDOM_BATCH_TOPK_POOL:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4096}"

STEPS="${STEPS:-8}"
PROTOTYPE_STRENGTH="${PROTOTYPE_STRENGTH:-0.4}"
OUTPUT_MODE="${OUTPUT_MODE:-debug}"

if [[ ! -f "${TASK_MANIFEST}" ]]; then
  echo "[error] missing task manifest: ${TASK_MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${READY_SLIDE_ROOT}" ]]; then
  echo "[error] missing staged slide root: ${READY_SLIDE_ROOT}" >&2
  exit 1
fi

mkdir -p "${PREP_DIR}"

"$PY" - <<PY
import csv
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/common/users/wq50/wsi_cf")
sys.path.insert(0, str(ROOT / "src"))

from wsi_cf.common.io import save_png
from wsi_cf.common.runtime import resolve_device, set_seed
from wsi_cf.data.region_bank import make_region_cells_preview
from wsi_cf.data.slides import open_slide, quick_region_quality_metrics, read_region_rgb_at_magnification
from wsi_cf.generation.pixcell import build_uni_grid_from_image, load_uni2
from wsi_cf.steering.sae_runtime import load_sae_from_config, sae_encode_features

seed = int("${SEED}")
set_seed(seed)
rng = random.Random(seed)
np_rng = np.random.default_rng(seed)
device = resolve_device("${DEVICE}")

prep_dir = Path("${PREP_DIR}")
prep_dir.mkdir(parents=True, exist_ok=True)
region_dir = prep_dir / "random_region_1024"
region_dir.mkdir(parents=True, exist_ok=True)

ready_root = Path("${READY_SLIDE_ROOT}")
slide_paths = sorted(ready_root.glob("*/*.svs")) + sorted(ready_root.glob("*.svs"))
if not slide_paths:
    raise SystemExit(f"No staged .svs slides found under {ready_root}")

def sample_tissue_region(slide_path: Path):
    slide = open_slide(slide_path)
    width, height = slide.dimensions
    best = None
    for _ in range(48):
        x = rng.randint(0, max(0, width - int("${REGION_SIZE}") * 2))
        y = rng.randint(0, max(0, height - int("${REGION_SIZE}") * 2))
        img, crop_w0, crop_h0 = read_region_rgb_at_magnification(
            slide,
            x0=x,
            y0=y,
            out_w=int("${REGION_SIZE}"),
            out_h=int("${REGION_SIZE}"),
            target_magnification=float("${TARGET_MAG}"),
        )
        q = quick_region_quality_metrics(img)
        score = q["tissue_score"] + 0.5 * q["dark_fraction"] + 0.25 * q["saturation_fraction"]
        if best is None or score > best[0]:
            best = (score, img, x, y, crop_w0, crop_h0, q)
        if q["tissue_score"] >= 0.45 and q["dark_fraction"] >= 0.04 and q["saturation_fraction"] >= 0.04:
            break
    slide.close()
    assert best is not None
    return best

slide_path = rng.choice(slide_paths)
score, region_img, region_x, region_y, crop_w0, crop_h0, quality = sample_tissue_region(slide_path)
slide_key = slide_path.name.split(".svs")[0]
if "-01Z-" in slide_key:
    slide_key = slide_key.split(".")[0]
case_id = slide_key[:12] if slide_key.startswith("TCGA-") else slide_key
region_id = f"{slide_key}__random1024__mag_{str('${TARGET_MAG}').replace('.', 'p')}__x_{region_x}__y_{region_y}"

region_png = region_dir / "region.png"
zgrid_npy = region_dir / "region_zgrid.npy"
preview_png = region_dir / "region_cells.png"
save_png(region_img, region_png)
save_png(make_region_cells_preview(region_img, grid_step_px=int("${GRID_STEP_PX}")), preview_png)

uni_model, uni_transform = load_uni2(device)
z_grid = build_uni_grid_from_image(
    region_img,
    uni_model=uni_model,
    uni_transform=uni_transform,
    grid_step_px=int("${GRID_STEP_PX}"),
    device=device,
    out_dtype=torch.float32,
).detach().cpu().numpy().astype(np.float32)
np.save(zgrid_npy, z_grid)
del uni_model
if device.type == "cuda":
    torch.cuda.empty_cache()

grid_h, grid_w, feature_dim = z_grid.shape
if (grid_h, grid_w) != (4, 4):
    raise SystemExit(f"Expected 1024/256 -> 4x4 grid, got {grid_h}x{grid_w}")

# Pick any 2x2 location inside the 4x4 grid. border_relaxed editing is used
# downstream so border 2x2 blocks are allowed for this probe.
gx0 = rng.randint(0, grid_w - 2)
gy0 = rng.randint(0, grid_h - 2)
target_cells = [
    {"gx": gx0, "gy": gy0},
    {"gx": gx0 + 1, "gy": gy0},
    {"gx": gx0, "gy": gy0 + 1},
    {"gx": gx0 + 1, "gy": gy0 + 1},
]

region_bank_csv = prep_dir / "region_bank.csv"
with region_bank_csv.open("w", newline="") as handle:
    fieldnames = [
        "region_id", "split", "label", "hpv_status", "case_id", "slide_key",
        "slide_path", "canonical_h5_path", "region_x", "region_y", "region_w",
        "region_h", "grid_step_px", "feature_dim", "tissue_score", "seed",
        "image_path", "feature_grid_path", "cell_preview_path", "region_dir",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        "region_id": region_id,
        "split": "random_probe",
        "label": 0,
        "hpv_status": "",
        "case_id": case_id,
        "slide_key": slide_key,
        "slide_path": str(slide_path),
        "canonical_h5_path": "",
        "region_x": region_x,
        "region_y": region_y,
        "region_w": int("${REGION_SIZE}"),
        "region_h": int("${REGION_SIZE}"),
        "grid_step_px": int("${GRID_STEP_PX}"),
        "feature_dim": feature_dim,
        "tissue_score": quality["tissue_score"],
        "seed": seed,
        "image_path": str(region_png),
        "feature_grid_path": str(zgrid_npy),
        "cell_preview_path": str(preview_png),
        "region_dir": str(region_dir),
    })

edit_manifest = prep_dir / "progressive_edit_manifest.json"
edit_manifest.write_text(json.dumps([
    {
        "run_id": f"{region_id}__random_2x2_gx{gx0}_gy{gy0}",
        "region_id": region_id,
        "target_cells": target_cells,
        "selection_mode": "random_2x2",
    }
], indent=2) + "\\n")

with Path("${TASK_MANIFEST}").open("r", newline="") as handle:
    manifest_rows = [r for r in csv.DictReader(handle) if Path(r.get("h5_path", "")).exists()]
rng.shuffle(manifest_rows)
manifest_rows = manifest_rows[: int("${CONCEPT_SCAN_MAX_SLIDES}")]
if not manifest_rows:
    raise SystemExit("No usable feature rows found for random concept scan.")

batch_model, _, d_latent = load_sae_from_config(Path("${BATCH_TOPK_SAE_CKPT}"), Path("${BATCH_TOPK_SAE_CFG}"), device=str(device))
relu_model, _, _ = load_sae_from_config(Path("${RELU_SAE_CKPT}"), Path("${RELU_SAE_CFG}"), device=str(device))
batch_model.eval()
relu_model.eval()

candidate_latents = np_rng.choice(d_latent, size=min(int("${RANDOM_BATCH_TOPK_POOL}"), d_latent), replace=False)
candidate_latents = np.sort(candidate_latents.astype(np.int64))
latent_to_pos = {int(lat): i for i, lat in enumerate(candidate_latents)}
top_values = {int(lat): [] for lat in candidate_latents}
top_rows = {int(lat): [] for lat in candidate_latents}

def read_features_coords(path: Path):
    with h5py.File(path, "r") as handle:
        feats = np.asarray(handle["features"][:], dtype=np.float32)
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    return feats, coords

with torch.no_grad():
    for row in manifest_rows:
        h5_path = Path(row["h5_path"])
        feats, coords = read_features_coords(h5_path)
        for start in range(0, feats.shape[0], int("${BATCH_SIZE}")):
            end = min(feats.shape[0], start + int("${BATCH_SIZE}"))
            x = torch.as_tensor(feats[start:end], dtype=torch.float32, device=device)
            z = sae_encode_features(batch_model, x).detach().cpu().numpy().astype(np.float32)
            z_sel = z[:, candidate_latents]
            local_k = min(int("${TOP_TILES_PER_CONCEPT}"), z_sel.shape[0])
            if local_k <= 0:
                continue
            part = np.argpartition(-z_sel, kth=local_k - 1, axis=0)[:local_k]
            for col, latent in enumerate(candidate_latents):
                vals = z_sel[part[:, col], col]
                for local_i, val in zip(part[:, col], vals):
                    global_i = int(start + local_i)
                    top_values[int(latent)].append(float(val))
                    top_rows[int(latent)].append({
                        "task": "random_batch_topk_probe",
                        "class_label": "random_batch_topk",
                        "latent_idx": int(latent),
                        "ranking_method": "attention_weighted",
                        "tile_rank": 0,
                        "activation": float(val),
                        "attention": 1.0,
                        "attention_norm": 1.0,
                        "attention_weighted_activation": float(val),
                        "case_id": row.get("case_id", ""),
                        "slide_key": row.get("slide_key", ""),
                        "project_dir": row.get("project_dir", ""),
                        "label": row.get("label_name", row.get("label", "")),
                        "h5_path": str(h5_path),
                        "tile_index": global_i,
                        "coord_x": int(coords[global_i, 0]) if coords.size else -1,
                        "coord_y": int(coords[global_i, 1]) if coords.size else -1,
                    })

active = []
for latent in candidate_latents:
    vals = np.asarray(top_values[int(latent)], dtype=np.float32)
    if vals.size and float(np.max(vals)) > 0.0:
        active.append(int(latent))
if len(active) < int("${N_CONCEPTS}"):
    raise SystemExit(f"Only found {len(active)} active random batch-TopK latents; increase CONCEPT_SCAN_MAX_SLIDES or RANDOM_BATCH_TOPK_POOL.")

chosen_batch_latents = rng.sample(active, int("${N_CONCEPTS}"))
concepts = []
rep_rows = []
mapping_rows = []
used_relu_latents = set()

for concept_rank, batch_latent in enumerate(chosen_batch_latents, start=1):
    rows = sorted(top_rows[batch_latent], key=lambda r: -float(r["activation"]))[: int("${TOP_TILES_PER_CONCEPT}")]
    feats = []
    for row in rows:
        feats_arr, _ = read_features_coords(Path(row["h5_path"]))
        feats.append(feats_arr[int(row["tile_index"])])
    feats_np = np.stack(feats, axis=0).astype(np.float32)
    with torch.no_grad():
        x = torch.as_tensor(feats_np, dtype=torch.float32, device=device)
        z_relu = sae_encode_features(relu_model, x).detach().cpu().numpy().astype(np.float32)
    mean_relu = z_relu.mean(axis=0)
    for relu_latent in np.argsort(-mean_relu):
        relu_latent = int(relu_latent)
        if relu_latent not in used_relu_latents and float(mean_relu[relu_latent]) > 0:
            break
    used_relu_latents.add(relu_latent)

    concepts.append({
        "task": "random_batch_topk_probe",
        "class_label": "random_batch_topk",
        "latent_idx": int(relu_latent),
        "concept_rank": int(concept_rank),
        "final_score": 1.0,
        "association_score": 1.0,
        "attention_support_score": 0.0,
        "steering_direction": "random_batch_topk_to_relu",
        "source_concept_sae": "batch_topk_20x",
        "source_latent_idx": int(batch_latent),
        "steering_sae": "tcga_uni2_sae_relu_v1",
        "relu_mean_activation_on_batch_topk_tiles": float(mean_relu[relu_latent]),
    })
    mapping_rows.append({
        "concept_rank": concept_rank,
        "source_batch_topk_latent_idx": batch_latent,
        "mapped_relu_latent_idx": relu_latent,
        "relu_mean_activation_on_batch_topk_tiles": float(mean_relu[relu_latent]),
        "n_tiles_used": len(rows),
    })
    for tile_rank, row in enumerate(rows, start=1):
        out = dict(row)
        out["latent_idx"] = int(relu_latent)
        out["source_latent_idx"] = int(batch_latent)
        out["source_concept_sae"] = "batch_topk_20x"
        out["steering_sae"] = "tcga_uni2_sae_relu_v1"
        out["tile_rank"] = int(tile_rank)
        out["activation"] = float(z_relu[tile_rank - 1, relu_latent])
        out["attention_weighted_activation"] = float(z_relu[tile_rank - 1, relu_latent])
        rep_rows.append(out)

concepts_json = prep_dir / "selected_random_batch_topk_concepts_mapped_to_relu.json"
concepts_json.write_text(json.dumps({
    "task": "random_batch_topk_probe",
    "class_label": "random_batch_topk",
    "mode": "random_batch_topk_find_relu_steer",
    "concepts": concepts,
}, indent=2) + "\\n")

rep_csv = prep_dir / "representative_tiles_random_batch_topk_mapped_to_relu.csv"
fields = []
for row in rep_rows:
    for key in row:
        if key not in fields:
            fields.append(key)
with rep_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rep_rows)

mapping_csv = prep_dir / "batch_topk_to_relu_mapping.csv"
with mapping_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0].keys()))
    writer.writeheader()
    writer.writerows(mapping_rows)

summary = {
    "region_id": region_id,
    "slide_path": str(slide_path),
    "region_x": region_x,
    "region_y": region_y,
    "target_magnification": float("${TARGET_MAG}"),
    "region_size": int("${REGION_SIZE}"),
    "target_cells": target_cells,
    "quality": quality,
    "chosen_batch_topk_latents": chosen_batch_latents,
    "mapped_relu_latents": [c["latent_idx"] for c in concepts],
    "run_mode": "${RUN_MODE}",
    "region_bank_csv": str(region_bank_csv),
    "edit_manifest": str(edit_manifest),
    "concepts_json": str(concepts_json),
    "representative_tiles_csv": str(rep_csv),
    "mapping_csv": str(mapping_csv),
}
(prep_dir / "prep_summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
print(json.dumps(summary, indent=2))
PY

run_editor() {
  local concepts_json="$1"
  local out_dir="$2"
  local max_concepts="$3"
  "$PY" scripts/run_progressive_region_edit.py \
    --task random_batch_topk_probe \
    --region-bank-csv "${PREP_DIR}/region_bank.csv" \
    --edit-manifest "${PREP_DIR}/progressive_edit_manifest.json" \
    --out-dir "${out_dir}" \
    --concepts-json "${concepts_json}" \
    --representative-tiles-csv "${PREP_DIR}/representative_tiles_random_batch_topk_mapped_to_relu.csv" \
    --concept-class-label random_batch_topk \
    --concept-ranking-method attention_weighted \
    --concept-target-stat median \
    --concept-target-top-k "${TOP_TILES_PER_CONCEPT}" \
    --max-concepts "${max_concepts}" \
    --max-runs 1 \
    --target-magnification "${TARGET_MAG}" \
    --edit-support border_relaxed \
    --prototype-strength "${PROTOTYPE_STRENGTH}" \
    --steer-blend 1.0 \
    --preserve-edit-strength 0.05 \
    --preserve-visited-strength 0.95 \
    --preserve-fresh-context-strength 0.35 \
    --mid-steer-start-ratio 0.5 \
    --mid-steer-end-ratio 1.0 \
    --mid-steer-alpha-start 0.5 \
    --mid-steer-alpha-end 1.0 \
    --mid-steer-alpha-schedule linear \
    --steps "${STEPS}" \
    --guidance 2.0 \
    --patch-batch 256 \
    --sae-ckpt "${RELU_SAE_CKPT}" \
    --sae-cfg "${RELU_SAE_CFG}" \
    --output-mode "${OUTPUT_MODE}" \
    --device "${DEVICE}"
}

if [[ "${RUN_MODE}" == "combined" ]]; then
  run_editor \
    "${PREP_DIR}/selected_random_batch_topk_concepts_mapped_to_relu.json" \
    "${OUT_DIR}/edit_random_batch_topk_relu_1024/combined_${N_CONCEPTS}_concepts" \
    "${N_CONCEPTS}"
elif [[ "${RUN_MODE}" == "per_concept" ]]; then
  "$PY" - <<PY
import json
from pathlib import Path

prep_dir = Path("${PREP_DIR}")
payload = json.loads((prep_dir / "selected_random_batch_topk_concepts_mapped_to_relu.json").read_text())
single_dir = prep_dir / "single_concept_jsons"
single_dir.mkdir(parents=True, exist_ok=True)
for concept in sorted(payload["concepts"], key=lambda c: int(c.get("concept_rank", 10**9))):
    rank = int(concept["concept_rank"])
    relu_latent = int(concept["latent_idx"])
    source_latent = int(concept.get("source_latent_idx", -1))
    out = dict(payload)
    out["concepts"] = [concept]
    path = single_dir / f"rank_{rank:02d}_batch_topk_{source_latent}_relu_{relu_latent}.json"
    path.write_text(json.dumps(out, indent=2) + "\\n")
    print(path)
PY
  mapfile -t SINGLE_JSONS < <(find "${PREP_DIR}/single_concept_jsons" -maxdepth 1 -type f -name 'rank_*.json' | sort)
  for single_json in "${SINGLE_JSONS[@]}"; do
    stem="$(basename "${single_json}" .json)"
    run_editor "${single_json}" "${OUT_DIR}/edit_random_batch_topk_relu_1024/${stem}" 1
  done
else
  echo "[error] RUN_MODE must be per_concept or combined, got ${RUN_MODE}" >&2
  exit 1
fi

echo "[ok] random 1024 batch-TopK concept probe written to ${OUT_DIR}"
echo "     prep: ${PREP_DIR}/prep_summary.json"
echo "     edit: ${OUT_DIR}/edit_random_batch_topk_relu_1024"
