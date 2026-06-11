#!/usr/bin/env python3
"""Build a curated concept package for morphologically obvious labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_concept_package import export_concept_package


DEFAULT_OUT_DIR = WSI_CF_ROOT / "artifacts/morphology_label_concept_review"

CONCEPT_ROOTS = [
    WSI_CF_ROOT / "artifacts/classifier_label_concepts_all_top10_top50",
    WSI_CF_ROOT / "artifacts/label_only_concepts_top10_top50",
    WSI_CF_ROOT / "artifacts/classifier_label_concepts_top10_top50",
]

CURATED_LABELS = [
    {
        "task": "luad_lusc",
        "class_label": "LUAD",
        "source_root": "artifacts/classifier_label_concepts_all_top10_top50",
        "priority": 1,
        "verdict": "best",
        "reason": "Lung adenocarcinoma has gland/lepidic/acinar morphology; paired with LUSC in the same organ, making it a strong steering target.",
    },
    {
        "task": "luad_lusc",
        "class_label": "LUSC",
        "source_root": "artifacts/classifier_label_concepts_all_top10_top50",
        "priority": 1,
        "verdict": "best",
        "reason": "Lung squamous morphology is visually distinct from LUAD and the classifier/concept signal is strong.",
    },
    {
        "task": "tumor_purity_low_high",
        "class_label": "high",
        "source_root": "artifacts/label_only_concepts_top10_top50",
        "priority": 2,
        "verdict": "best",
        "reason": "High purity should correspond to dense tumor-rich tiles, a direct morphology/content axis for local editing.",
    },
    {
        "task": "tumor_purity_low_high",
        "class_label": "low",
        "source_root": "artifacts/label_only_concepts_top10_top50",
        "priority": 2,
        "verdict": "best",
        "reason": "Low purity should enrich stromal/normal/immune/background admixture, useful for preservation/locality tests.",
    },
    {
        "task": "kirc_low_vs_high_grade",
        "class_label": "high",
        "source_root": "artifacts/classifier_label_concepts_all_top10_top50",
        "priority": 3,
        "verdict": "good",
        "reason": "High grade KIRC is morphologic, but the trained classifier is weaker than LUAD/LUSC; use after visual confirmation.",
    },
    {
        "task": "kirc_low_vs_high_grade",
        "class_label": "low",
        "source_root": "artifacts/classifier_label_concepts_all_top10_top50",
        "priority": 3,
        "verdict": "good",
        "reason": "Low grade KIRC is morphologic and paired within one cancer type; useful if representative tiles look coherent.",
    },
    {
        "task": "cancer_type_all_tcga",
        "class_label": "TCGA-LUAD",
        "source_root": "artifacts/classifier_label_concepts_all_top10_top50",
        "priority": 4,
        "verdict": "visualize_only",
        "reason": "Pan-cancer labels are morphologically obvious but mix tissue site and tumor type, so they are better for concept inspection than clean counterfactual steering.",
    },
    {
        "task": "cancer_type_all_tcga",
        "class_label": "TCGA-LUSC",
        "source_root": "artifacts/classifier_label_concepts_all_top10_top50",
        "priority": 4,
        "verdict": "visualize_only",
        "reason": "Pan-cancer LUSC concepts can validate squamous morphology, but LUAD/LUSC binary is cleaner for our method.",
    },
    {
        "task": "cancer_type_all_tcga",
        "class_label": "TCGA-KIRC",
        "source_root": "artifacts/classifier_label_concepts_all_top10_top50",
        "priority": 4,
        "verdict": "visualize_only",
        "reason": "KIRC has clear-cell morphology and strong pan-cancer signal; useful as a positive-control concept set.",
    },
    {
        "task": "msi_coad_stad",
        "class_label": "MSI",
        "source_root": "artifacts/classifier_label_concepts_all_top10_top50",
        "priority": 5,
        "verdict": "secondary",
        "reason": "MSI can show lymphocyte-rich morphology, but it is molecular and less directly local than histology/purity/grade.",
    },
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(str(key))
                    seen.add(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_label_path(label: str) -> str:
    return str(label).replace("/", "_").replace(" ", "_")


def concept_dir_for(root: Path, task: str, class_label: str) -> Path:
    return root / task / safe_label_path(class_label)


def summarize_concept_dir(concept_dir: Path) -> dict[str, Any]:
    cards_path = concept_dir / "concept_cards.csv"
    reps_path = concept_dir / "representative_tiles.csv"
    rows = read_csv_rows(cards_path)
    reps = read_csv_rows(reps_path) if reps_path.exists() else []
    top = rows[0] if rows else {}
    top5 = rows[:5]
    return {
        "n_concepts": len(rows),
        "n_representative_tiles": len(reps),
        "top_latent_idx": top.get("latent_idx", ""),
        "top_final_score": parse_float(top.get("final_score", "")),
        "top_association_score": parse_float(top.get("association_score", "")),
        "top_cohen_d": parse_float(top.get("cohen_d", "")),
        "top_abs_diff": parse_float(top.get("abs_diff", "")),
        "mean_top5_final_score": sum(parse_float(r.get("final_score", "")) for r in top5) / max(len(top5), 1),
        "mean_top5_association_score": sum(parse_float(r.get("association_score", "")) for r in top5) / max(len(top5), 1),
    }


def classifier_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = WSI_CF_ROOT / "artifacts/classifier_training"
    for summary_path in sorted(root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        task = summary_path.parent.name
        data = payload.get("data", {})
        metrics = payload.get("best_test_metrics", {})
        for label, n_train in data.get("train_counts", {}).items():
            rows.append(
                {
                    "source": "classifier_training",
                    "task": task,
                    "label": label,
                    "train_count": int(n_train),
                    "test_count": int(data.get("test_counts", {}).get(label, 0)),
                    "best_accuracy": metrics.get("accuracy", ""),
                    "best_balanced_accuracy": metrics.get("balanced_accuracy", ""),
                    "best_macro_f1": metrics.get("macro_f1", ""),
                    "best_auroc": "" if metrics.get("auroc") is None else metrics.get("auroc", ""),
                }
            )
    return rows


def target_label_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((WSI_CF_ROOT / "resources/labels/targets").glob("*_case.tsv")):
        table = read_csv_rows(path)
        if not table:
            continue
        columns = [key for key in table[0] if key not in {"case_id", "project_dir"} and not key.endswith("_source") and not key.endswith("_conflict")]
        for column in columns:
            counts: dict[str, int] = {}
            for row in table:
                value = str(row.get(column, "")).strip()
                if not value or value == "Unknown":
                    continue
                counts[value] = counts.get(value, 0) + 1
            for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                rows.append({"source": path.name, "task": path.stem.replace("_case", ""), "label": label, "case_count": int(count)})
    return rows


def scan_existing_concepts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in CONCEPT_ROOTS:
        if not root.exists():
            continue
        for cards_path in sorted(root.glob("*/*/concept_cards.csv")):
            concept_dir = cards_path.parent
            task = concept_dir.parent.name
            label = concept_dir.name
            summary = summarize_concept_dir(concept_dir)
            rows.append(
                {
                    "source_root": str(root.relative_to(WSI_CF_ROOT)),
                    "task": task,
                    "class_label": label,
                    "concept_dir": str(concept_dir.relative_to(WSI_CF_ROOT)),
                    **summary,
                }
            )
    rows.sort(key=lambda r: (-float(r["top_association_score"]), -float(r["top_cohen_d"]), str(r["task"]), str(r["class_label"])))
    return rows


def copy_source_files(concept_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["concept_cards.csv", "representative_tiles.csv", "selected_concepts.json", "summary.json"]:
        src = concept_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def build_review(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_inventory = classifier_inventory() + target_label_inventory()
    write_csv(out_dir / "label_inventory.csv", label_inventory)

    existing = scan_existing_concepts()
    write_csv(out_dir / "existing_concept_outputs.csv", existing)

    selected_rows: list[dict[str, Any]] = []
    top_concept_rows: list[dict[str, Any]] = []
    export_outputs: dict[str, Any] = {}
    for item in CURATED_LABELS:
        root = WSI_CF_ROOT / item["source_root"]
        concept_dir = concept_dir_for(root, item["task"], item["class_label"])
        if not concept_dir.exists():
            selected_rows.append({**item, "available": False, "concept_dir": str(concept_dir.relative_to(WSI_CF_ROOT))})
            continue
        summary = summarize_concept_dir(concept_dir)
        rel_name = f'{item["priority"]:02d}_{item["task"]}__{safe_label_path(item["class_label"])}'
        review_label_dir = out_dir / "selected" / rel_name
        copy_source_files(concept_dir, review_label_dir)
        export_dir = review_label_dir / "concept_export"
        outputs = export_concept_package(
            argparse.Namespace(
                concept_dir=concept_dir,
                out_dir=export_dir,
                target_magnification=float(args.target_magnification),
                tile_size_px=int(args.tile_size_px),
                coord_space="level0_h5_coords",
                feature_name="UNI2",
            )
        )
        export_outputs[rel_name] = outputs
        selected_rows.append(
            {
                **item,
                "available": True,
                "concept_dir": str(concept_dir.relative_to(WSI_CF_ROOT)),
                "review_dir": str(review_label_dir.relative_to(WSI_CF_ROOT)),
                "concept_export_dir": str(export_dir.relative_to(WSI_CF_ROOT)),
                **summary,
            }
        )
        for row in read_csv_rows(concept_dir / "concept_cards.csv")[: int(args.top_concepts_per_label)]:
            top_concept_rows.append(
                {
                    "priority": item["priority"],
                    "verdict": item["verdict"],
                    "task": item["task"],
                    "class_label": item["class_label"],
                    "reason": item["reason"],
                    **row,
                }
            )

    write_csv(out_dir / "selected_morphology_labels.csv", selected_rows)
    write_csv(out_dir / "top_concepts_for_selected_labels.csv", top_concept_rows)
    write_json(
        out_dir / "manifest.json",
        {
            "out_dir": str(out_dir),
            "selected_label_count": len([row for row in selected_rows if row.get("available")]),
            "selected_labels_csv": str(out_dir / "selected_morphology_labels.csv"),
            "top_concepts_csv": str(out_dir / "top_concepts_for_selected_labels.csv"),
            "label_inventory_csv": str(out_dir / "label_inventory.csv"),
            "existing_concept_outputs_csv": str(out_dir / "existing_concept_outputs.csv"),
            "export_outputs": export_outputs,
        },
    )
    (out_dir / "README.md").write_text(
        "# Morphology Label Concept Review\n\n"
        "This package curates label-associated SAE concepts that are likely to be visually/morphologically interpretable.\n\n"
        "Recommended first-pass labels for local visualization and steering:\n"
        "1. `luad_lusc/LUAD` and `luad_lusc/LUSC`\n"
        "2. `tumor_purity_low_high/high` and `tumor_purity_low_high/low`\n"
        "3. `kirc_low_vs_high_grade/high` and `kirc_low_vs_high_grade/low`\n\n"
        "Each `selected/*/concept_export/` folder is portable. Use `slide_path_map.template.csv` to fill in local SVS paths, then crop tiles from `representative_tiles.csv` at `coord_x,coord_y`.\n\n"
        "Files:\n"
        "- `label_inventory.csv`: labels available in resources/classifier tasks.\n"
        "- `existing_concept_outputs.csv`: concept outputs already present in artifacts.\n"
        "- `selected_morphology_labels.csv`: curated labels, reasons, and concept strength summary.\n"
        "- `top_concepts_for_selected_labels.csv`: top concept rows to inspect first.\n"
        "- `selected/*/concept_export/`: local-PC visualization manifests.\n"
    )
    print(json.dumps({"out_dir": str(out_dir), "selected": selected_rows}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-concepts-per-label", type=int, default=10)
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument("--tile-size-px", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    build_review(parse_args())


if __name__ == "__main__":
    main()
