#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from models.diffusion import (
    FlowMatchingConfig,
    FlowMatchingTrainer,
    JITStudentConfig,
    PixCellConfig,
    TrajectoryDistillConfig,
    TrajectoryDistillationTrainer,
    UNIFeatureDataset,
    build_pixcell_pipeline,
    build_teacher_student,
    decode_latents_to_images,
    sample_student_flow,
    sample_student_trajectory,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)


def collect_feature_paths(args: argparse.Namespace) -> List[str]:
    paths: List[str] = []

    if args.feature_paths:
        paths.extend([p.strip() for p in args.feature_paths.split(",") if p.strip()])

    if args.feature_glob:
        for pat in args.feature_glob.split(","):
            pat = pat.strip()
            if not pat:
                continue
            paths.extend([str(p) for p in sorted(Path().glob(pat))])

    if args.manifest:
        with open(args.manifest, "r") as f:
            manifest = json.load(f)
        key = args.manifest_key
        if key not in manifest:
            raise KeyError(
                f"Manifest key '{key}' not found in {args.manifest}. Available keys: {list(manifest.keys())}"
            )
        vals = manifest[key]
        if not isinstance(vals, list):
            raise ValueError(f"Manifest[{key}] must be a list of paths")
        paths.extend([str(v) for v in vals])

    # dedupe while preserving order
    seen = set()
    out = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    if not out:
        raise ValueError(
            "No feature files found. Use --feature_paths, --feature_glob, or --manifest + --manifest_key"
        )

    missing = [p for p in out if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} feature paths do not exist, first: {missing[0]}")

    return out


@torch.no_grad()
def save_preview(
    out_dir: Path,
    step: int,
    trainer_mode: str,
    student,
    cond: torch.Tensor,
    pipeline,
    latent_channels: int,
    latent_size: int,
    sample_steps: int,
    guidance: float,
) -> None:
    cond = cond[:1]
    if trainer_mode == "traj":
        lat = sample_student_trajectory(
            student=student,
            cond=cond,
            pipeline=pipeline,
            latent_channels=latent_channels,
            latent_size=latent_size,
            steps=sample_steps,
            guidance_scale=guidance,
        )
    else:
        lat = sample_student_flow(
            student=student,
            cond=cond,
            pipeline=pipeline,
            latent_channels=latent_channels,
            latent_size=latent_size,
            steps=sample_steps,
            guidance_scale=guidance,
        )

    img = decode_latents_to_images(pipeline, lat)[0]
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(out_dir / "samples" / f"step_{step:07d}.png", format="PNG")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("train_diffusion.py")

    ap.add_argument("--train_mode", choices=["traj", "flow"], required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--feature_paths", type=str, default="", help="Comma-separated feature files")
    ap.add_argument("--feature_glob", type=str, default="", help="Comma-separated glob patterns")
    ap.add_argument("--manifest", type=str, default="")
    ap.add_argument("--manifest_key", type=str, default="tcga_train")
    ap.add_argument("--feature_norm", choices=["none", "l2", "layernorm"], default="none")

    ap.add_argument("--pix_model_id", type=str, default="StonyBrook-CVLab/PixCell-256")
    ap.add_argument("--pix_pipeline_id", type=str, default="StonyBrook-CVLab/PixCell-pipeline")
    ap.add_argument("--vae_model_id", type=str, default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--vae_subfolder", type=str, default="vae")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp16")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--latent_size", type=int, default=32, help="Latent H=W; 32 for 256x256 with scale=8")
    ap.add_argument("--teacher_steps", type=int, default=28)

    ap.add_argument("--traj_min_steps", type=int, default=4)
    ap.add_argument("--traj_max_steps", type=int, default=8)

    ap.add_argument("--guidance_teacher", type=float, default=3.0)
    ap.add_argument("--guidance_student", type=float, default=1.0)

    ap.add_argument("--init_student_from_teacher", action="store_true")
    ap.add_argument("--student_arch", choices=["pixcell", "jit"], default="pixcell")
    ap.add_argument("--jit_hidden_size", type=int, default=768)
    ap.add_argument("--jit_depth", type=int, default=8)
    ap.add_argument("--jit_heads", type=int, default=12)
    ap.add_argument("--jit_mlp_ratio", type=float, default=4.0)
    ap.add_argument("--jit_dropout", type=float, default=0.0)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--preview_every", type=int, default=1000)
    ap.add_argument("--preview_steps", type=int, default=6)

    ap.add_argument("--resume", type=str, default="")

    return ap


def main() -> None:
    args = build_argparser().parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    run_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)
    print(f"Using device={device} dtype={run_dtype}")
    feat_paths = collect_feature_paths(args)
    ds = UNIFeatureDataset(feat_paths, normalize=args.feature_norm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    loader_iter = itertools.cycle(loader)

    pix_cfg = PixCellConfig(
        pix_model_id=args.pix_model_id,
        pix_pipeline_id=args.pix_pipeline_id,
        vae_model_id=args.vae_model_id,
        vae_subfolder=args.vae_subfolder,
        dtype=run_dtype,
    )

    pipeline = build_pixcell_pipeline(pix_cfg, device=device)
    pipeline.transformer.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.transformer.eval()
    pipeline.vae.eval()
    print(f"Loaded pipeline with pix_model_id={args.pix_model_id} and vae_model_id={args.vae_model_id}")
    cond_dim = int(ds.feature_dim)
    teacher, student = build_teacher_student(
        pipeline,
        cond_dim=cond_dim,
        init_student_from_teacher=args.init_student_from_teacher,
        student_arch=args.student_arch,
        jit_cfg=JITStudentConfig(
            latent_size=args.latent_size,
            hidden_size=args.jit_hidden_size,
            depth=args.jit_depth,
            num_heads=args.jit_heads,
            mlp_ratio=args.jit_mlp_ratio,
            dropout=args.jit_dropout,
        ),
    )
    # Keep teacher in run_dtype for fast target generation, but keep student weights in fp32
    # for optimizer stability (autocast handles mixed-precision compute).
    teacher.to(device=device, dtype=run_dtype)
    student.to(device=device, dtype=torch.float32)

    latent_channels = int(pipeline.vae.config.latent_channels)

    if args.train_mode == "traj":
        trainer = TrajectoryDistillationTrainer(
            teacher=teacher,
            student=student,
            pipeline=pipeline,
            cfg=TrajectoryDistillConfig(
                latent_channels=latent_channels,
                latent_size=args.latent_size,
                teacher_steps=args.teacher_steps,
                student_min_steps=args.traj_min_steps,
                student_max_steps=args.traj_max_steps,
                guidance_teacher=args.guidance_teacher,
                guidance_student=args.guidance_student,
            ),
        )
    else:
        trainer = FlowMatchingTrainer(
            teacher=teacher,
            student=student,
            pipeline=pipeline,
            cfg=FlowMatchingConfig(
                latent_channels=latent_channels,
                latent_size=args.latent_size,
                teacher_steps=args.teacher_steps,
                guidance_teacher=args.guidance_teacher,
                guidance_student=args.guidance_student,
            ),
        )
    
    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_param_dtype = next(student.parameters()).dtype
    use_scaler = (device.type == "cuda" and run_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ema = EMA(student, decay=args.ema_decay) if args.ema else None
    
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        student.load_state_dict(ckpt["student"])
        optim.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0)) + 1
        if ema is not None and "ema" in ckpt:
            model_state = student.state_dict()
            new_shadow = {}
            for k, v in ckpt["ema"].items():
                tgt = model_state.get(k, None)
                if tgt is None:
                    continue
                new_shadow[k] = v.to(device=tgt.device, dtype=tgt.dtype)
            ema.shadow = new_shadow

    print(
        f"Training mode={args.train_mode} | features={len(ds)} rows | dim={cond_dim} | "
        f"latent={latent_channels}x{args.latent_size}x{args.latent_size}"
    )
    print(
        f"Teacher dtype={next(teacher.parameters()).dtype} | "
        f"Student param dtype={train_param_dtype} | GradScaler={scaler.is_enabled()}",
        flush=True,
    )

    for step in range(start_step, args.max_steps):
        feat = next(loader_iter)
        feat = feat.to(device=device, dtype=run_dtype, non_blocking=True)
        cond = feat.unsqueeze(1)

        optim.zero_grad(set_to_none=True)

        use_autocast = device.type == "cuda" and run_dtype in {torch.float16, torch.bfloat16}
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=run_dtype)
            if use_autocast
            else nullcontext()
        )
        with amp_ctx:
            out = trainer.loss(cond)
            loss = out["loss"]

        if not torch.isfinite(loss):
            print(
                f"step={step} non-finite loss ({loss.detach().item()}); skipping optimizer step",
                flush=True,
            )
            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.update()
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        def has_nonfinite_grad(module: torch.nn.Module) -> bool:
            for p in module.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    return True
            return False

        if args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)

        if has_nonfinite_grad(student):
            print(f"step={step} non-finite gradients detected; skipping optimizer step", flush=True)
            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.update()
            continue

        if scaler.is_enabled():
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()

        if ema is not None:
            ema.update(student)

        if step % args.log_every == 0:
            msg = f"step={step} loss={float(loss.item()):.6f}"
            if "student_steps" in out:
                msg += f" student_steps={float(out['student_steps'].item()):.1f}"
            print(msg, flush=True)

        if step > 0 and step % args.save_every == 0:
            payload = {
                "step": step,
                "student": student.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
            }
            if ema is not None:
                payload["ema"] = ema.shadow
            torch.save(payload, out_dir / "ckpt" / f"ckpt_{step:07d}.pt")

        if step > 0 and step % args.preview_every == 0:
            backup = None
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in student.state_dict().items()}
                ema.copy_to(student)

            student.eval()
            save_preview(
                out_dir=out_dir,
                step=step,
                trainer_mode=args.train_mode,
                student=student,
                cond=cond,
                pipeline=pipeline,
                latent_channels=latent_channels,
                latent_size=args.latent_size,
                sample_steps=args.preview_steps,
                guidance=args.guidance_student,
            )
            student.train()

            if backup is not None:
                student.load_state_dict(backup)

    final = {
        "step": args.max_steps - 1,
        "student": student.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
    if ema is not None:
        final["ema"] = ema.shadow
    torch.save(final, out_dir / "ckpt" / "final.pt")
    print(f"Saved final checkpoint: {out_dir / 'ckpt' / 'final.pt'}")


if __name__ == "__main__":
    main()
