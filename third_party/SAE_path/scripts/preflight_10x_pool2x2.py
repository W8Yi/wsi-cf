#!/usr/bin/env python3
"""
Quick 10x_pool2x2 preflight / smoke test for SAE feature manifests.

Use this before a long train run to validate:
- manifest split paths
- 10x pooling-map construction from coords
- pooled tile counts / feature dims
- optional runtime chunk reads (single-process or DataLoader workers)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.dataloader import (
    SlideTileConfig,
    SlideTileDataset,
    make_sae_loader,
    load_manifest_json,
    get_paths_from_manifest,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "test", "val"])
    ap.add_argument("--magnification", type=str, default="10x_pool2x2", choices=["20x", "10x_pool2x2"])
    ap.add_argument("--tiles_per_slide", type=int, default=512)
    ap.add_argument("--slide_batch_tiles", type=int, default=512)
    ap.add_argument("--sampling", type=str, default="slice", choices=["slice", "multislice", "indexed"])
    ap.add_argument("--multislice_chunks", type=int, default=4)
    ap.add_argument("--pool_map_cache_dir", type=str, default=None)
    ap.add_argument("--pool_allow_partial", action="store_true")
    ap.add_argument("--shapes_cache_json", type=str, default=None)
    ap.add_argument("--n_examples", type=int, default=5, help="Number of dataset draws to print/read.")
    ap.add_argument("--use_loader", action="store_true", help="Exercise DataLoader path instead of direct dataset indexing.")
    ap.add_argument("--batch_size", type=int, default=2, help="Used only with --use_loader.")
    ap.add_argument("--num_workers", type=int, default=0, help="Used only with --use_loader.")
    args = ap.parse_args()

    manifest = load_manifest_json(args.manifest)
    paths = get_paths_from_manifest(manifest, args.split)
    print(f"[Preflight10x] split={args.split} manifest_paths={len(paths)}")

    cfg = SlideTileConfig(
        tiles_per_slide=args.tiles_per_slide,
        slide_batch_tiles=args.slide_batch_tiles,
        seed=1337,
        shuffle_slides=False,
        sample_with_replacement=False,
        normalize=None,
        return_meta=True,
        sampling=args.sampling,
        multislice_chunks=args.multislice_chunks,
        shapes_cache_json=args.shapes_cache_json,
        magnification=args.magnification,
        pool_map_cache_dir=args.pool_map_cache_dir,
        pool2x2_require_complete=(not args.pool_allow_partial),
    )
    ds = SlideTileDataset(paths, cfg)

    n_arr = torch.as_tensor(ds.slide_n, dtype=torch.long)
    d_set = sorted(set(int(x) for x in ds.slide_d))
    print(
        f"[Preflight10x] usable_slides={len(ds.h5_paths)} draws={len(ds)} "
        f"tiles_total={int(n_arr.sum().item())} "
        f"tiles_per_slide(min/median/max)="
        f"{int(n_arr.min().item())}/{int(n_arr.median().item())}/{int(n_arr.max().item())} "
        f"feat_dim_set={d_set[:8]}",
        flush=True,
    )

    n_examples = max(0, min(int(args.n_examples), len(ds)))
    if n_examples == 0:
        print("[Preflight10x] No examples requested.")
        return

    if not args.use_loader:
        print("[Preflight10x] Reading examples via direct dataset indexing...")
        for i in range(n_examples):
            x, meta = ds[i]
            print(
                f"  idx={i} slide={Path(meta['h5_path']).name} "
                f"num_tiles_in_slide={meta['num_tiles_in_slide']} sampled_tiles={meta['sampled_tiles']} "
                f"x_shape={tuple(x.shape)}",
                flush=True,
            )
        return

    print(
        f"[Preflight10x] Reading examples via DataLoader batch_size={args.batch_size} num_workers={args.num_workers}...",
        flush=True,
    )
    loader = make_sae_loader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2,
    )

    seen = 0
    for batch_idx, batch in enumerate(loader):
        x, metas = batch
        print(
            f"  batch={batch_idx} x_shape={tuple(x.shape)} draws_in_batch={len(metas)} "
            f"sampled_tiles_each={[int(m['sampled_tiles']) for m in metas]}",
            flush=True,
        )
        for meta in metas[:2]:
            print(
                f"    slide={Path(meta['h5_path']).name} pooled_tiles={meta['num_tiles_in_slide']} "
                f"sampling={meta['sampling']} mag={meta['magnification']}",
                flush=True,
            )
        seen += len(metas)
        if seen >= n_examples:
            break


if __name__ == "__main__":
    main()
