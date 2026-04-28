#!/usr/bin/env python3
"""
train_sae.py

Supports:
1) Training a ReLUSparseSAE (L1 sparsity)
2) Training a TopKSAE (hard k-sparsity), optionally with a larger-k warmup before narrowing to the target k
3) Training a BatchTopKSAE (global minibatch sparsity budget)
4) Training a two-level SDF-SAE (Level-1 SAE + Level-2 dictionary factorization)

Resume/continue:
- Auto-resumes from <ckpt_prefix>_ckpt_last.pt in out_dir unless --no_auto_resume is given
- Can start from an explicit checkpoint via --resume_ckpt

Requires:
- sae_models.py providing: ReLUSparseSAE, TopKSAE, l2_recon_loss, l1_sparsity, activation_stats, InputNormWrapper
- sae_dataloader.py providing: SlideTileDataset, SlideTileConfig, make_sae_loader, load_manifest_json, get_paths_from_manifest

Manifest JSON keys expected:
- "tcga_train", "tcga_test"

Magnification option:
- --magnification 20x: train on stored features directly.
- --magnification 10x_pool2x2: derive 10x-like features by averaging each 2x2 neighbor block of 20x tiles
  using coords from each H5. This avoids writing separate 10x feature H5 files.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataloader import (
    SlideTileDataset,
    SlideTileConfig,
    make_sae_loader,
    load_manifest_json,
    get_paths_from_manifest,
)

from models.sae import (
    ReLUSparseSAE,
    TopKSAE,
    BatchTopKSAE,
    SDFSAE2Level,
    l2_recon_loss,
    l1_sparsity,
    activation_stats,
    InputNormWrapper,
)


@torch.inference_mode()
def eval_recon_mse(
    model: nn.Module,
    loader,
    device: str,
    max_batches: int = 200,
    amp_enabled: bool = False,
) -> float:
    model.eval()
    tot = 0.0
    n = 0
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.to(device, non_blocking=True).float()
        with torch.amp.autocast("cuda", enabled=(amp_enabled and device.startswith("cuda"))):
            x_hat, _, _ = model(x)
        mse = F.mse_loss(x_hat, x, reduction="mean").item()
        bs = x.shape[0]
        tot += mse * bs
        n += bs
    model.train()
    return tot / max(1, n)


def maybe_cuda_mem_stats(device: str) -> Dict[str, float]:
    if not device.startswith("cuda"):
        return {}
    return {
        "sys/gpu_mem_alloc_gb": torch.cuda.memory_allocated() / (1024**3),
        "sys/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
    }

@torch.no_grad()
def dead_unit_stats(z: torch.Tensor, eps: float = 0.0) -> Dict[str, float]:
    """
    z: (B, d_latent) activations for the current batch.
    Reports:
      - dead_frac_batch: fraction of latents that are exactly (or <=eps) zero for ALL samples in this batch
      - mean_act_per_latent: mean activation per latent over the batch (averaged then mean)
      - max_act_per_latent: maximum activation per latent over the batch (averaged then max)
    Note: this is a *batch proxy*; true dead-units should be measured over many batches.
    """
    if z.ndim != 2:
        return {}
    # active if any sample has |z|>eps
    active = (z.abs() > eps).any(dim=0).float()          # (d_latent,)
    dead_frac = 1.0 - active.mean().item()
    mean_act_lat = z.abs().mean(dim=0)
    return {
        "dead_frac_batch": float(dead_frac),
        "mean_abs_act_per_latent": float(mean_act_lat.mean().item()),
        "max_abs_act_per_latent": float(mean_act_lat.max().item()),
    }


def save_ckpt(
    path: Path,
    base_model: nn.Module,
    wrapper_layernorm: bool,
    opt: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    epoch: int,
    best_test: float,
    last_test: Optional[float],
    stage_start_step: int,
    extra: Optional[Dict[str, Any]] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "epoch": epoch,
        "best_test_mse": best_test,
        "last_test_mse": last_test,
        "model": base_model.state_dict(),  # base SAE only (no wrapper LN)
        "wrapper_layernorm": wrapper_layernorm,
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "stage_start_step": stage_start_step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def make_base_model(
    stage: str,
    d_in: int,
    d_latent: int,
    tied: bool,
    topk_k: int,
    topk_mode: str,
    topk_nonneg: bool,
    sdf_n_level2: int,
    sdf_coeff_nonneg: bool,
    sdf_coeff_simplex: bool,
    init_scale: float = 0.02,
) -> nn.Module:
    if stage in {"relu", "relu2topk"}:
        return ReLUSparseSAE(d_in=d_in, d_latent=d_latent, tied=tied, use_pre_bias=True, init_scale=init_scale)
    if stage == "topk":
        return TopKSAE(
            d_in=d_in,
            d_latent=d_latent,
            k=topk_k,
            tied=tied,
            use_pre_bias=True,
            topk_mode=topk_mode,      # "value" or "magnitude"
            nonneg=topk_nonneg,       # usually True
            init_scale=init_scale,
        )
    if stage == "batch_topk":
        return BatchTopKSAE(
            d_in=d_in,
            d_latent=d_latent,
            k=topk_k,
            tied=tied,
            use_pre_bias=True,
            topk_mode=topk_mode,
            nonneg=topk_nonneg,
            init_scale=init_scale,
        )
    if stage == "sdf2":
        return SDFSAE2Level(
            d_in=d_in,
            d_latent=d_latent,
            d_level2=sdf_n_level2,
            tied=tied,
            use_pre_bias=True,
            coeff_nonneg=sdf_coeff_nonneg,
            coeff_simplex=sdf_coeff_simplex,
            init_scale=init_scale,
        )
    raise ValueError(f"Unknown stage '{stage}'")


def _copy_state_tensor(
    *,
    dst_state: Dict[str, torch.Tensor],
    src_state: Dict[str, torch.Tensor],
    key: str,
    allow_missing: bool,
) -> None:
    if key not in src_state:
        if allow_missing:
            return
        raise KeyError(f"Init checkpoint is missing required tensor '{key}'.")
    if key not in dst_state:
        raise KeyError(f"Target model is missing tensor '{key}' needed for init.")
    src = src_state[key]
    dst = dst_state[key]
    if tuple(src.shape) != tuple(dst.shape):
        raise ValueError(
            f"Shape mismatch for '{key}': src={tuple(src.shape)} vs dst={tuple(dst.shape)}"
        )
    dst_state[key] = src.detach().clone()


def maybe_init_model_from_ckpt(
    *,
    base_model: nn.Module,
    stage: str,
    init_mode: str,
    init_from_ckpt: Optional[str],
    init_allow_missing: bool,
) -> None:
    if init_mode == "none":
        if init_from_ckpt:
            print("Info: --init_from_ckpt was provided but --init_mode=none; skipping init.")
        return

    if init_mode != "batch_topk_to_sdf2":
        raise ValueError(f"Unsupported init_mode='{init_mode}'.")
    if stage != "sdf2":
        raise ValueError("--init_mode=batch_topk_to_sdf2 requires --stage sdf2.")
    if not init_from_ckpt:
        raise ValueError("--init_mode=batch_topk_to_sdf2 requires --init_from_ckpt.")

    init_path = Path(init_from_ckpt)
    if not init_path.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {init_path}")

    payload = torch.load(str(init_path), map_location="cpu")
    src_state = payload.get("model")
    if not isinstance(src_state, dict):
        raise ValueError(f"Invalid init checkpoint payload at {init_path}: missing 'model' dict.")

    stage_name = str(payload.get("stage_name", ""))
    if "batch_topk" not in stage_name:
        print(
            f"Warning: init checkpoint stage_name='{stage_name}' does not look like batch_topk. "
            "Proceeding with tensor compatibility checks."
        )

    dst_state = base_model.state_dict()
    shared_keys: List[str] = ["b_pre", "enc.weight", "enc.bias"]
    if "dec.weight" in dst_state:
        shared_keys.extend(["dec.weight", "dec.bias"])
    elif "dec_bias" in dst_state:
        shared_keys.append("dec_bias")

    for key in shared_keys:
        _copy_state_tensor(
            dst_state=dst_state,
            src_state=src_state,
            key=key,
            allow_missing=init_allow_missing,
        )

    base_model.load_state_dict(dst_state, strict=True)
    print(
        f"Initialized SDF2 level-1 params from checkpoint: {init_path} "
        f"(mode={init_mode}, allow_missing={init_allow_missing})"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # wandb
    ap.add_argument("--project", type=str, default="SAE_pathology")
    ap.add_argument("--entity", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--tags", type=str, default="")
    ap.add_argument("--mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # stage selection
    ap.add_argument("--stage", type=str, default="topk", choices=["relu", "topk", "batch_topk", "sdf2"])

    # model dims
    ap.add_argument("--d_in", type=int, default=1536)
    ap.add_argument("--latent_dim", type=int, default=8 * 1536)  # 12288
    ap.add_argument("--tied", action="store_true", help="Tie decoder weight to encoder weight^T")
    ap.add_argument("--no_input_layernorm", action="store_true")

    # TopK controls
    ap.add_argument("--topk_k", type=int, default=512)
    ap.add_argument("--topk_mode", type=str, default="value", choices=["value", "magnitude"])
    ap.add_argument("--topk_nonneg", action="store_true", help="Apply ReLU before TopK (recommended)")
    ap.add_argument("--topk_k_warmup", type=int, default=None, help="Optional larger k for a warmup TopK phase before the target k")
    ap.add_argument("--topk_warmup_steps", type=int, default=0, help="Number of steps to train at warmup k before switching to target k")
    ap.add_argument("--batch_topk_k", type=int, default=None, help="If set, overrides topk_k for batch_topk stage")

    # SDF-SAE (two-level) controls
    ap.add_argument("--sdf_n_level2", type=int, default=512, help="Number of high-level prototypes U for SDF 2-level model")
    ap.add_argument("--sdf_alpha", type=float, default=1.0, help="Weight alpha on Level-2 factorization loss")
    ap.add_argument("--sdf_lambda_a", type=float, default=1e-4, help="L1 coefficient penalty on A")
    ap.add_argument("--sdf_lambda_u", type=float, default=1e-4, help="L2 coefficient penalty on U")
    ap.add_argument("--sdf_active_only", action="store_true", help="Apply dynamic masking: only active Level-1 atoms contribute to Level-2 loss")
    ap.add_argument("--sdf_active_eps", type=float, default=1e-6, help="Activity threshold eps for dynamic masking in sdf_active_only mode")
    ap.add_argument("--sdf_coeff_nonneg", action="store_true", help="Constrain SDF coefficients A to nonnegative (via softplus)")
    ap.add_argument("--sdf_coeff_simplex", action="store_true", help="Constrain each row of A to simplex (softmax)")
    ap.add_argument(
        "--sdf_parent_balance_lambda",
        type=float,
        default=0.0,
        help="Optional weight on parent-usage balance penalty (0 disables).",
    )

    # data
    ap.add_argument("--tiles_per_slide", type=int, default=2048)
    ap.add_argument("--slide_batch_tiles", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2, help="Number of slide-chunks concatenated per step")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--eval_slide_batch_tiles", type=int, default=None, help="Optional smaller slide_batch_tiles for eval/test loader.")
    ap.add_argument("--eval_batch_size", type=int, default=None, help="Optional smaller batch_size for eval/test loader.")
    ap.add_argument("--eval_num_workers", type=int, default=None, help="Optional num_workers for eval/test loader.")
    ap.add_argument(
        "--magnification",
        type=str,
        default="20x",
        choices=["20x", "10x_pool2x2"],
        help="Data view: original 20x tiles, or 10x made by pooling non-overlapping 2x2 neighboring 20x tiles.",
    )
    ap.add_argument(
        "--pool_map_cache_dir",
        type=str,
        default=None,
        help="Optional directory to cache 2x2 pooling index maps (small files, speeds repeated runs).",
    )
    ap.add_argument(
        "--pool_allow_partial",
        action="store_true",
        help="Allow partial 2x2 blocks at boundaries in 10x_pool2x2 mode (default drops incomplete blocks).",
    )
    ap.add_argument(
        "--shapes_cache_json",
        type=str,
        default=None,
        help="Optional JSON cache of per-slide [n,d] stats (keyed by magnification), for faster startup.",
    )

    # optimization (ReLU stage)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--l1_lambda", type=float, default=1e-4)

    # optimization (TopK stage - can override)
    ap.add_argument("--topk_lr", type=float, default=None, help="If set, overrides lr for TopK stage")

    # training length
    ap.add_argument("--max_steps", type=int, default=50_000, help="Number of steps for the selected stage (excluding warmup)")

    # logging/eval/prints
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_batches", type=int, default=200)
    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--dead_eps", type=float, default=0.0, help="|z|<=eps counted as zero for dead-unit proxy")
    
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")

    # resume / continuation
    ap.add_argument("--resume_ckpt", type=str, default=None, help="Explicit checkpoint path to resume from")
    ap.add_argument("--no_auto_resume", action="store_true", help="Disable auto-resume from <ckpt_prefix>_ckpt_last.pt in out_dir")
    ap.add_argument("--init_from_ckpt", type=str, default=None, help="Optional checkpoint for model parameter initialization.")
    ap.add_argument(
        "--init_mode",
        type=str,
        default="none",
        choices=["none", "batch_topk_to_sdf2"],
        help="Initialization strategy for loading shared params before training.",
    )
    ap.add_argument(
        "--init_allow_missing",
        action="store_true",
        help="Allow missing shared tensors during --init_mode mapping.",
    )
    ap.add_argument(
        "--preflight_only",
        action="store_true",
        help="Build train/test datasets (including 10x pool-map validation) and exit before DataLoader/model training.",
    )

    args = ap.parse_args()

    use_topk_warmup = (args.stage == "topk") and (args.topk_k_warmup is not None) and (args.topk_warmup_steps > 0)
    if args.stage == "topk":
        if args.topk_k_warmup is not None and args.topk_k_warmup <= args.topk_k:
            raise ValueError("topk_k_warmup must be > topk_k to provide a larger-k warmup phase.")
        if args.topk_warmup_steps > 0 and args.topk_k_warmup is None:
            raise ValueError("topk_warmup_steps > 0 requires --topk_k_warmup to be set.")
    if args.sdf_coeff_simplex and args.sdf_coeff_nonneg:
        print("Info: --sdf_coeff_simplex implies nonnegative coefficients; --sdf_coeff_nonneg is redundant.")
    if args.stage == "sdf2" and args.sdf_n_level2 >= args.latent_dim:
        raise ValueError("--sdf_n_level2 must be < latent_dim for SDF 2-level hierarchy.")
    if args.sdf_parent_balance_lambda < 0:
        raise ValueError("--sdf_parent_balance_lambda must be >= 0.")
    if args.init_mode != "none" and not args.init_from_ckpt:
        raise ValueError("--init_mode requires --init_from_ckpt.")
    if args.init_mode == "batch_topk_to_sdf2" and args.stage != "sdf2":
        raise ValueError("--init_mode=batch_topk_to_sdf2 requires --stage sdf2.")

    if args.slide_batch_tiles > args.tiles_per_slide:
        raise ValueError("slide_batch_tiles must be <= tiles_per_slide (recommended: equal).")
    if args.eval_slide_batch_tiles is not None and args.eval_slide_batch_tiles > args.tiles_per_slide:
        raise ValueError("eval_slide_batch_tiles must be <= tiles_per_slide.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_slide_batch_tiles = int(args.eval_slide_batch_tiles or args.slide_batch_tiles)
    eval_batch_size = int(args.eval_batch_size or args.batch_size)
    eval_num_workers = int(args.eval_num_workers if args.eval_num_workers is not None else max(0, min(args.num_workers, 2)))

    # data
    manifest = load_manifest_json(args.manifest)
    train_paths = get_paths_from_manifest(manifest, "train")
    test_paths = get_paths_from_manifest(manifest, "test")

    cfg_train = SlideTileConfig(
        tiles_per_slide=args.tiles_per_slide,
        slide_batch_tiles=args.slide_batch_tiles,
        seed=1337,
        shuffle_slides=True,
        sample_with_replacement=False,
        normalize=None,
        return_meta=False,
        magnification=args.magnification,
        pool_map_cache_dir=args.pool_map_cache_dir,
        pool2x2_require_complete=(not args.pool_allow_partial),
        shapes_cache_json=args.shapes_cache_json,
    )
    cfg_test = SlideTileConfig(
        tiles_per_slide=args.tiles_per_slide,
        slide_batch_tiles=eval_slide_batch_tiles,
        seed=2024,
        shuffle_slides=False,
        sample_with_replacement=False,
        normalize=None,
        return_meta=False,
        magnification=args.magnification,
        pool_map_cache_dir=args.pool_map_cache_dir,
        pool2x2_require_complete=(not args.pool_allow_partial),
        shapes_cache_json=args.shapes_cache_json,
    )

    train_ds = SlideTileDataset(train_paths, cfg_train)
    test_ds = SlideTileDataset(test_paths, cfg_test)

    print(
        f"Train dataset: slides={len(train_ds.h5_paths)} draws/epoch={len(train_ds)} "
        f"tiles_per_draw~{cfg_train.slide_batch_tiles} magnification={args.magnification}",
        flush=True,
    )
    print(
        f"Test dataset: slides={len(test_ds.h5_paths)} draws/epoch={len(test_ds)} "
        f"tiles_per_draw~{cfg_test.slide_batch_tiles} magnification={args.magnification}",
        flush=True,
    )

    if args.preflight_only:
        def _split_summary(name: str, ds: SlideTileDataset) -> None:
            n_arr = torch.as_tensor(ds.slide_n, dtype=torch.long)
            d_arr = torch.as_tensor(ds.slide_d, dtype=torch.long)
            print(
                f"[Preflight] {name}: slides={len(ds.h5_paths)} "
                f"tiles_total={int(n_arr.sum().item())} "
                f"tiles_per_slide(min/median/max)="
                f"{int(n_arr.min().item())}/{int(n_arr.median().item())}/{int(n_arr.max().item())} "
                f"feat_dim_set={sorted(set(int(x) for x in d_arr.tolist()))[:8]}",
                flush=True,
            )
            if len(set(int(x) for x in d_arr.tolist())) > 1:
                print(f"[Preflight] Warning: multiple feature dims detected in {name} split.", flush=True)

        _split_summary("train", train_ds)
        _split_summary("test", test_ds)
        print("[Preflight] Dataset/magnification validation complete. Exiting before training.", flush=True)
        return

    train_loader = make_sae_loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        # train_ds.set_epoch(...) mutates dataset order on the main process; persistent workers would
        # retain stale dataset copies across epochs and ignore those updates.
        persistent_workers=False,
        prefetch_factor=2,
    )
    test_loader = make_sae_loader(
        test_ds,
        batch_size=eval_batch_size,
        num_workers=eval_num_workers,
        # Keep eval loader light to avoid post-eval RAM pressure when train workers are also alive.
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )

    # wandb init
    import wandb

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    wb = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        mode=args.mode,
        tags=tags,
        config={
            **vars(args),
            "train_files": len(train_paths),
            "test_files": len(test_paths),
            "effective_tiles_per_step": args.slide_batch_tiles * args.batch_size,
            "device": device,
        },
    )
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(dict(wb.config), f, indent=2)

    wrapper_layernorm = not args.no_input_layernorm

    # Training state
    global_step = 0
    epoch = 0
    best_test = float("inf")

    def train(
        stage_name: str,
        base_model: nn.Module,
        steps_target: int,
        lr: float,
        l1_lambda: float,
        ckpt_prefix: str,
        resume_ckpt: Optional[str] = None,
        auto_resume: bool = True,
    ):
        nonlocal global_step, epoch, best_test

        model: nn.Module = base_model
        if wrapper_layernorm:
            model = InputNormWrapper(args.d_in, model)

        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and torch.cuda.is_available()))

        stage_best = float("inf")
        stage_start_step = global_step

        # Resume logic: prefer explicit path, else auto-discover last ckpt
        resolved_ckpt = Path(resume_ckpt) if resume_ckpt is not None else None
        if resolved_ckpt is not None and not resolved_ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resolved_ckpt}")
        if resolved_ckpt is None and auto_resume:
            candidate = out_dir / f"{ckpt_prefix}_ckpt_last.pt"
            if candidate.exists():
                resolved_ckpt = candidate

        if resolved_ckpt is not None:
            payload = torch.load(str(resolved_ckpt), map_location=device)
            base_model.load_state_dict(payload["model"])
            loaded_wrapper_ln = payload.get("wrapper_layernorm", wrapper_layernorm)
            if loaded_wrapper_ln != wrapper_layernorm:
                raise ValueError(
                    f"Checkpoint wrapper_layernorm={loaded_wrapper_ln} does not match current setting {wrapper_layernorm}."
                )
            opt.load_state_dict(payload["opt"])
            scaler.load_state_dict(payload["scaler"])

            global_step = int(payload.get("step", global_step))
            epoch = int(payload.get("epoch", epoch))
            best_test = float(payload.get("best_test_mse", best_test))
            stage_best = float(payload.get("stage_best_mse", stage_best))

            if "stage_start_step" in payload:
                stage_start_step = int(payload["stage_start_step"])
            else:
                print("Warning: checkpoint missing stage_start_step; defaulting to 0 (may slightly under-train stage).")
                stage_start_step = 0

            print(f"Resumed from {resolved_ckpt}: global_step={global_step}, epoch={epoch}, best_test={best_test:.6f}")

        train_ds.set_epoch(epoch)
        data_iter = iter(train_loader)

        # speed/ETA
        last_t = time.time()
        ema_dt = None  # seconds/step EMA

        print(f"\n=== Stage '{stage_name}' start: target_steps={steps_target}, lr={lr}, l1_lambda={l1_lambda} ===", flush=True)

        while (global_step - stage_start_step) < steps_target:
            try:
                x = next(data_iter)
            except StopIteration:
                epoch += 1
                train_ds.set_epoch(epoch)
                data_iter = iter(train_loader)
                x = next(data_iter)

            if isinstance(x, (tuple, list)):
                x = x[0]
            x = x.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)

            sdf_terms = None
            sdf_parent_balance_term = None
            sdf_parent_usage_stats = None
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                x_hat, z, _ = model(x)
                recon = l2_recon_loss(x, x_hat)
                loss = recon
                if l1_lambda > 0.0:
                    loss = loss + l1_lambda * l1_sparsity(z)
                if hasattr(base_model, "sdf_factorization_loss"):
                    sdf_terms = base_model.sdf_factorization_loss(
                        z=z,
                        active_only=args.sdf_active_only,
                        active_eps=args.sdf_active_eps,
                        lambda_a=args.sdf_lambda_a,
                        lambda_u=args.sdf_lambda_u,
                    )
                    loss = loss + args.sdf_alpha * sdf_terms["sdf_total"]
                    if hasattr(base_model, "coeff_matrix"):
                        A = base_model.coeff_matrix().float()  # (n1, n2)
                        usage = A.mean(dim=0)
                        usage = usage / usage.sum().clamp_min(1e-12)
                        uniform = torch.full_like(usage, 1.0 / float(max(1, usage.numel())))
                        sdf_parent_balance_term = F.mse_loss(usage, uniform, reduction="mean")
                        if args.sdf_parent_balance_lambda > 0.0:
                            loss = loss + args.sdf_parent_balance_lambda * sdf_parent_balance_term
                        norm = max(math.log(float(max(2, usage.numel()))), 1e-12)
                        entropy = -(usage * torch.log(usage.clamp_min(1e-12))).sum() / norm
                        sdf_parent_usage_stats = {
                            "entropy": float(entropy.item()),
                            "min": float(usage.min().item()),
                            "max": float(usage.max().item()),
                        }

            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1

            # per-step print
            if args.print_every > 0 and (global_step % args.print_every == 0):
                now = time.time()
                dt = now - last_t
                last_t = now
                ema_dt = dt if ema_dt is None else (0.9 * ema_dt + 0.1 * dt)

                remaining = steps_target - (global_step - stage_start_step)
                eta_h = (remaining * ema_dt) / 3600.0 if ema_dt is not None else float("nan")
                tiles = x.shape[0]
                stats = activation_stats(z)

                print(
                    f"[{stage_name}] step={global_step} (epoch={epoch}) "
                    f"tiles={tiles} dt={dt:.2f}s ema={ema_dt:.2f}s ETA={eta_h:.2f}h "
                    f"loss={loss.item():.6f} recon={recon.item():.6f} "
                    f"frac_nz={stats['frac_nonzero']:.4f} nz/sample={stats['mean_nonzero_per_sample']:.1f}",
                    flush=True,
                )
                if sdf_terms is not None:
                    print(
                        f"  sdf_total={sdf_terms['sdf_total'].item():.6f} "
                        f"sdf_recon={sdf_terms['sdf_recon'].item():.6f} "
                        f"active_atoms={int(sdf_terms['sdf_active_atoms'].item())}",
                        flush=True,
                    )
                if sdf_parent_usage_stats is not None:
                    print(
                        f"  parent_balance={float(sdf_parent_balance_term.item() if sdf_parent_balance_term is not None else 0.0):.6f} "
                        f"parent_entropy={sdf_parent_usage_stats['entropy']:.4f} "
                        f"parent_min={sdf_parent_usage_stats['min']:.6f} "
                        f"parent_max={sdf_parent_usage_stats['max']:.6f}",
                        flush=True,
                    )

            # wandb log
            if global_step % args.log_every == 0:
                stats = activation_stats(z)
                dead = dead_unit_stats(z, eps=args.dead_eps)
                log = {
                    "stage": stage_name,
                    "train/loss": float(loss.item()),
                    "train/recon_mse": float(recon.item()),
                    "train/act_l1": float(l1_sparsity(z).item()),
                    "train/frac_nonzero": stats["frac_nonzero"],
                    "train/nz_per_sample": stats["mean_nonzero_per_sample"],
                    "train/z_mean": stats["z_mean"],
                    "train/z_std": stats["z_std"],
                    "train/top1_top2_margin": stats["top1_top2_margin"],
                    "sys/step": global_step,
                    "sys/epoch": epoch,
                    
                    # dead-unit proxy (batch-level)
                    "train/dead_frac_batch": dead["dead_frac_batch"],
                    "train/mean_abs_act_per_latent": dead["mean_abs_act_per_latent"],
                    "train/max_abs_act_per_latent": dead["max_abs_act_per_latent"],
                }
                if sdf_terms is not None:
                    log.update(
                        {
                            "train/sdf_total": float(sdf_terms["sdf_total"].item()),
                            "train/sdf_recon": float(sdf_terms["sdf_recon"].item()),
                            "train/sdf_l1_a": float(sdf_terms["sdf_l1_a"].item()),
                            "train/sdf_l2_u": float(sdf_terms["sdf_l2_u"].item()),
                            "train/sdf_active_atoms": float(sdf_terms["sdf_active_atoms"].item()),
                        }
                    )
                if sdf_parent_usage_stats is not None:
                    log.update(
                        {
                            "train/sdf_parent_balance": float(
                                sdf_parent_balance_term.item() if sdf_parent_balance_term is not None else 0.0
                            ),
                            "train/sdf_parent_usage_entropy": float(sdf_parent_usage_stats["entropy"]),
                            "train/sdf_parent_usage_min": float(sdf_parent_usage_stats["min"]),
                            "train/sdf_parent_usage_max": float(sdf_parent_usage_stats["max"]),
                        }
                    )
                log.update(maybe_cuda_mem_stats(device))
                wandb.log(log, step=global_step)

            # eval + ckpt
            if global_step % args.eval_every == 0:
                test_mse = eval_recon_mse(
                    model,
                    test_loader,
                    device=device,
                    max_batches=args.eval_batches,
                    amp_enabled=(args.amp and torch.cuda.is_available()),
                )
                wandb.log(
                    {
                        "stage": stage_name,
                        "test/recon_mse": float(test_mse),
                        "sys/step": global_step,
                        "sys/epoch": epoch,
                    },
                    step=global_step,
                )

                improved = test_mse < best_test
                if improved:
                    best_test = test_mse
                    wandb.log({"test/best_recon_mse": float(best_test)}, step=global_step)

                stage_best = min(stage_best, test_mse)

                # save checkpoints (save base SAE weights, not wrapper LN)
                save_ckpt(
                    out_dir / f"{ckpt_prefix}_ckpt_last.pt",
                    base_model=base_model,
                    wrapper_layernorm=wrapper_layernorm,
                    opt=opt,
                    scaler=scaler,
                    step=global_step,
                    epoch=epoch,
                    best_test=best_test,
                    last_test=test_mse,
                    stage_start_step=stage_start_step,
                    extra={"stage_name": stage_name, "stage_best_mse": stage_best},
                )
                if test_mse <= stage_best + 1e-12:
                    save_ckpt(
                        out_dir / f"{ckpt_prefix}_ckpt_best.pt",
                        base_model=base_model,
                        wrapper_layernorm=wrapper_layernorm,
                        opt=opt,
                        scaler=scaler,
                        step=global_step,
                        epoch=epoch,
                        best_test=best_test,
                        last_test=test_mse,
                        stage_start_step=stage_start_step,
                        extra={"stage_name": stage_name, "stage_best_mse": stage_best},
                    )

        # final stage save
        save_ckpt(
            out_dir / f"{ckpt_prefix}_final.pt",
            base_model=base_model,
            wrapper_layernorm=wrapper_layernorm,
            opt=opt,
            scaler=scaler,
            step=global_step,
            epoch=epoch,
            best_test=best_test,
            last_test=None,
            stage_start_step=stage_start_step,
            extra={"stage_name": stage_name, "stage_best_mse": stage_best},
        )

        print(f"=== Stage '{stage_name}' done. stage_best_mse={stage_best:.6f} global_best_mse={best_test:.6f} ===", flush=True)

        return base_model  # trained base model (weights updated)

    # ----------------------------
    # Run selected training mode
    # ----------------------------

    stage_topk_k = args.topk_k
    if args.stage == "batch_topk" and args.batch_topk_k is not None:
        stage_topk_k = args.batch_topk_k

    base = make_base_model(
        stage=args.stage,
        d_in=args.d_in,
        d_latent=args.latent_dim,
        tied=args.tied,
        topk_k=(args.topk_k_warmup if (args.stage == "topk" and use_topk_warmup) else stage_topk_k),
        topk_mode=args.topk_mode,
        topk_nonneg=args.topk_nonneg,
        sdf_n_level2=args.sdf_n_level2,
        sdf_coeff_nonneg=args.sdf_coeff_nonneg,
        sdf_coeff_simplex=args.sdf_coeff_simplex,
    )
    maybe_init_model_from_ckpt(
        base_model=base,
        stage=args.stage,
        init_mode=args.init_mode,
        init_from_ckpt=args.init_from_ckpt,
        init_allow_missing=bool(args.init_allow_missing),
    )

    if args.stage == "relu":
        train(
            stage_name="relu",
            base_model=base,
            steps_target=args.max_steps,
            lr=args.lr,
            l1_lambda=args.l1_lambda,
            ckpt_prefix="relu",
            resume_ckpt=args.resume_ckpt,
            auto_resume=not args.no_auto_resume,
        )
    elif args.stage == "topk":
        # TopK with optional warmup k, then target k
        if use_topk_warmup:
            base = train(
                stage_name=f"topk_warm(k={args.topk_k_warmup})",
                base_model=base,
                steps_target=args.topk_warmup_steps,
                lr=(args.topk_lr if args.topk_lr is not None else args.lr),
                l1_lambda=0.0,
                ckpt_prefix="topk_warm",
                resume_ckpt=args.resume_ckpt,
                auto_resume=not args.no_auto_resume,
            )
            base.k = args.topk_k  # narrow k after warmup

        train(
            stage_name=f"topk(k={args.topk_k})",
            base_model=base,
            steps_target=args.max_steps,
            lr=(args.topk_lr if args.topk_lr is not None else args.lr),
            l1_lambda=0.0,
            ckpt_prefix="topk",
            resume_ckpt=(None if use_topk_warmup else args.resume_ckpt),
            auto_resume=not args.no_auto_resume,
        )
    elif args.stage == "batch_topk":
        k_eff = stage_topk_k
        train(
            stage_name=f"batch_topk(k={k_eff})",
            base_model=base,
            steps_target=args.max_steps,
            lr=(args.topk_lr if args.topk_lr is not None else args.lr),
            l1_lambda=0.0,
            ckpt_prefix="batch_topk",
            resume_ckpt=args.resume_ckpt,
            auto_resume=not args.no_auto_resume,
        )
    elif args.stage == "sdf2":
        train(
            stage_name=f"sdf2(n2={args.sdf_n_level2})",
            base_model=base,
            steps_target=args.max_steps,
            lr=(args.topk_lr if args.topk_lr is not None else args.lr),
            l1_lambda=args.l1_lambda,
            ckpt_prefix="sdf2",
            resume_ckpt=args.resume_ckpt,
            auto_resume=not args.no_auto_resume,
        )
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")

    wandb.finish()
    print("Done. Outputs in:", out_dir)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _WSI_SAE_SRC = Path(__file__).resolve().parents[2] / "wsi-sae" / "src"
    if _WSI_SAE_SRC.exists():
        sys.path.insert(0, str(_WSI_SAE_SRC))
        from wsi_sae.cli import main as _wsi_sae_main

        _wsi_sae_main(["train", *sys.argv[1:]])
    else:
        main()
