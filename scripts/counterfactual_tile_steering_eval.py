#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
WSI_CF_ROOT = SCRIPT_DIR.parent
SRC_ROOT = WSI_CF_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wsi_cf.common.io import write_json
from wsi_cf.common.runtime import resolve_device
from wsi_cf.eval.hnsc_hpv import run_counterfactual_eval


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual tile steering on HNSC HPV test split: "
            "select top-attention tiles, steer toward HPV+ and HPV- prototypes, and measure MIL prediction changes."
        )
    )
    parser.add_argument("--split-json", type=Path, default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.json"))
    parser.add_argument("--split-tsv", type=Path, default=Path("/common/users/wq50/SAE_path/metadata/manifests/hnsc_hpv_5fold/split_0.tsv"))
    parser.add_argument("--features-root", type=Path, default=Path("/research/projects/mllab/WSI/TCGA_features/TCGA-HNSC/features_uni2"))
    parser.add_argument("--mil-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/hnsc_hpv_attention_mil_5fold_filtered_uni2h_80/split_0/final.pt"))
    parser.add_argument("--sae-ckpt", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/relu_final.pt"))
    parser.add_argument("--sae-cfg", type=Path, default=Path("/common/users/wq50/SAE_path/runs/relu_sae_base/run_config.json"))
    parser.add_argument("--prototype-npz", type=Path, default=Path("/common/users/wq50/wsi_cf/artifacts/sae_prototypes/hnscc_hpv_split0_selected/prototype_vectors_for_selected_sae.npz"))
    parser.add_argument("--prototype-key", type=str, default="prototype_median", choices=["prototype_mean", "prototype_median"])
    parser.add_argument("--pos-latent", type=int, default=2645)
    parser.add_argument("--neg-latent", type=int, default=7036)
    parser.add_argument("--top-tiles-per-slide", type=int, default=2)
    parser.add_argument("--prototype-strength", type=float, default=0.8)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=Path, default=WSI_CF_ROOT / "artifacts/counterfactual_eval")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    args.device = str(device)
    results, summary = run_counterfactual_eval(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "counterfactual_tile_results.csv"
    summary_path = args.out_dir / "counterfactual_summary.json"
    with csv_path.open("w", newline="") as handle:
        fieldnames = list(results[0].keys()) if results else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(results)
    write_json(summary_path, summary)
    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {summary_path}")


if __name__ == "__main__":
    main()
