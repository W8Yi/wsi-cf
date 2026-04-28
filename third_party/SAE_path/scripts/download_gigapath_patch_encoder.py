#!/usr/bin/env python3
"""
Download (cache) the Prov-GigaPath patch encoder from Hugging Face.

This uses timm loading path so the downloaded weights are directly usable by:
  timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
"""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--timm_model",
        type=str,
        default="hf_hub:prov-gigapath/prov-gigapath",
        help="timm model id for GigaPath patch encoder",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device for quick verification load ("cpu" or "cuda:0").',
    )
    ap.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Optional HF token (or set HF_TOKEN env var).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print("[auth] Hugging Face login complete")

    import timm
    import torch

    device = args.device
    print(f"[download] loading {args.timm_model} on {device} ...")
    model = timm.create_model(args.timm_model, pretrained=True).to(device).eval()
    embed_dim = getattr(model, "num_features", None)
    n_params = sum(p.numel() for p in model.parameters())

    print("[ok] GigaPath patch encoder downloaded and load-verified")
    print(f"      model: {args.timm_model}")
    print(f"      embed_dim: {embed_dim}")
    print(f"      params: {n_params}")
    print("      cache: ~/.cache/huggingface/hub (default)")


if __name__ == "__main__":
    main()
