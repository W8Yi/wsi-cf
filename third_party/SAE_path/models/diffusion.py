from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DiffusionPipeline
from data.dataloader import _resolve_h5_path


class _FeatureSource:
    def __init__(self, path: Path):
        self.path = path
        self.n_rows = 0
        self.dim = 0

    def get_row(self, row: int) -> np.ndarray:
        raise NotImplementedError


class _H5FeatureSource(_FeatureSource):
    def __init__(self, path: Path):
        resolved = Path(_resolve_h5_path(str(path)))
        super().__init__(resolved)
        self.requested_path = Path(path)
        with h5py.File(self.path, "r") as f:
            if "features" not in f:
                raise KeyError(f"Missing 'features' in {self.path}")
            feats = f["features"]
            if feats.ndim == 2:
                self.n_rows, self.dim = int(feats.shape[0]), int(feats.shape[1])
                self._slice3d = False
            elif feats.ndim == 3 and feats.shape[0] == 1:
                self.n_rows, self.dim = int(feats.shape[1]), int(feats.shape[2])
                self._slice3d = True
            else:
                raise ValueError(f"Unsupported features shape {feats.shape} in {self.path}")
        self._h5: Optional[h5py.File] = None

    def _get_dataset(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", libver="latest", swmr=True)
        return self._h5["features"]

    def get_row(self, row: int) -> np.ndarray:
        feats = self._get_dataset()
        x = feats[0, row, :] if self._slice3d else feats[row, :]
        return np.asarray(x, dtype=np.float32)


class _NpyFeatureSource(_FeatureSource):
    def __init__(self, path: Path):
        super().__init__(path)
        arr = np.load(path, mmap_mode="r")
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = "features" if "features" in arr else (list(arr.keys())[0] if arr.keys() else None)
            if key is None:
                raise ValueError(f"No arrays found in npz file: {path}")
            warnings.warn(
                f"{path} is .npz; random row access is slower than .npy/.h5 for large training sets.",
                RuntimeWarning,
                stacklevel=2,
            )
            arr = arr[key]
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D features (N,D), got {arr.shape} from {path}")
        self._arr = arr
        self.n_rows = int(arr.shape[0])
        self.dim = int(arr.shape[1])

    def get_row(self, row: int) -> np.ndarray:
        return np.asarray(self._arr[row], dtype=np.float32)


class _PtFeatureSource(_FeatureSource):
    def __init__(self, path: Path):
        super().__init__(path)
        warnings.warn(
            f"{path} is .pt/.pth and will be loaded into memory. Prefer .h5 or .npy for large datasets.",
            RuntimeWarning,
            stacklevel=2,
        )
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            obj = obj["features"] if "features" in obj else obj[next(iter(obj.keys()), None)]
            if obj is None:
                raise ValueError(f"Empty dict in tensor file: {path}")
        if not torch.is_tensor(obj):
            obj = torch.as_tensor(obj)
        arr = obj.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D features (N,D), got {arr.shape} from {path}")
        self._arr = np.asarray(arr, dtype=np.float32)
        self.n_rows = int(self._arr.shape[0])
        self.dim = int(self._arr.shape[1])

    def get_row(self, row: int) -> np.ndarray:
        return self._arr[row]


def _build_feature_source(path: Path) -> _FeatureSource:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return _H5FeatureSource(path)
    if suffix in {".npy", ".npz"}:
        return _NpyFeatureSource(path)
    if suffix in {".pt", ".pth"}:
        return _PtFeatureSource(path)
    raise ValueError(f"Unsupported feature file: {path}")


class UNIFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[str], normalize: str = "none"):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("UNIFeatureDataset received no feature paths")

        self.normalize = normalize.lower()
        if self.normalize not in {"none", "l2", "layernorm"}:
            raise ValueError("normalize must be one of: none, l2, layernorm")

        self._sources: List[_FeatureSource] = []
        self._cum_sizes: List[int] = []
        total = 0
        for p in self.paths:
            src = _build_feature_source(p)
            self._sources.append(src)
            total += int(src.n_rows)
            self._cum_sizes.append(total)

        if total <= 0:
            raise ValueError("No feature rows found across all feature files")
        self._len = total
        self.feature_dim = int(self._sources[0].dim)
        for src, p in zip(self._sources, self.paths):
            if int(src.dim) != self.feature_dim:
                raise ValueError(f"Feature dim mismatch in {p}: {src.dim} vs {self.feature_dim}")

    def __len__(self) -> int:
        return self._len

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return x
        if self.normalize == "l2":
            denom = float(np.linalg.norm(x)) + 1e-8
            return x / denom
        mean = float(x.mean())
        std = float(x.std()) + 1e-8
        return (x - mean) / std

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)

        file_id = int(np.searchsorted(self._cum_sizes, idx, side="right"))
        prev = 0 if file_id == 0 else self._cum_sizes[file_id - 1]
        row = idx - prev
        x = self._sources[file_id].get_row(row)
        x = self._normalize(x)
        return torch.from_numpy(np.asarray(x, dtype=np.float32))


@dataclass
class PixCellConfig:
    pix_model_id: str = "StonyBrook-CVLab/PixCell-256"
    pix_pipeline_id: str = "StonyBrook-CVLab/PixCell-pipeline"
    vae_model_id: str = "stabilityai/stable-diffusion-3.5-large"
    vae_subfolder: str = "vae"
    dtype: torch.dtype = torch.float16


@dataclass
class JITStudentConfig:
    latent_size: int = 32
    hidden_size: int = 768
    depth: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    half = dim // 2
    emb_y = _sinusoidal_timestep_embedding(yy, half)
    emb_x = _sinusoidal_timestep_embedding(xx, dim - half)
    return torch.cat([emb_y, emb_x], dim=-1)


class _JITBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x + ctx[:, None, :])
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x + ctx[:, None, :]))
        return x


class JITImageTransformerStudent(nn.Module):
    """
    Lightweight image-only transformer student for latent diffusion distillation.
    Uses timestep + UNI conditioning through additive context modulation.
    """

    def __init__(self, in_channels: int, cond_dim: int, cfg: JITStudentConfig):
        super().__init__()
        self.in_channels = int(in_channels)
        self.cond_dim = int(cond_dim)
        self.cfg = cfg
        hs = int(cfg.hidden_size)

        self.proj_in = nn.Conv2d(self.in_channels, hs, kernel_size=1, bias=True)
        self.time_mlp = nn.Sequential(
            nn.Linear(hs, hs * 4),
            nn.SiLU(),
            nn.Linear(hs * 4, hs),
        )
        self.cond_proj = nn.Linear(self.cond_dim, hs)
        self.blocks = nn.ModuleList(
            [_JITBlock(hs, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm_out = nn.LayerNorm(hs)
        self.proj_out = nn.Conv2d(hs, self.in_channels, kernel_size=1, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ):
        x = hidden_states
        b, _, h, w = x.shape
        feat = self.proj_in(x)  # [B, Hs, H, W]
        tok = feat.flatten(2).transpose(1, 2)  # [B, HW, Hs]

        pos = _build_2d_sincos_pos_embed(h, w, tok.shape[-1], device=tok.device).to(tok.dtype)
        tok = tok + pos[None, :, :]

        if encoder_hidden_states.ndim == 3:
            cond = encoder_hidden_states.mean(dim=1)
        else:
            cond = encoder_hidden_states
        cond_ctx = self.cond_proj(cond.to(tok.dtype))
        t_ctx = self.time_mlp(_sinusoidal_timestep_embedding(timestep, tok.shape[-1]).to(tok.dtype))
        ctx = cond_ctx + t_ctx

        for blk in self.blocks:
            tok = blk(tok, ctx)

        tok = self.norm_out(tok)
        feat = tok.transpose(1, 2).reshape(b, -1, h, w)
        out = self.proj_out(feat)
        if return_dict:
            return SimpleNamespace(sample=out)
        return (out,)


class PixCellDenoiser(nn.Module):
    def __init__(self, transformer: nn.Module, cond_dim: int):
        super().__init__()
        self.transformer = transformer
        self.cond_dim = int(cond_dim)

    @staticmethod
    def _to_model_timestep(t: torch.Tensor) -> torch.Tensor:
        # Accept either normalized [0,1] time or raw diffusion timesteps.
        if torch.is_floating_point(t):
            t_min = float(t.min().item())
            t_max = float(t.max().item())
            if t_min >= -1e-6 and t_max <= 1.0 + 1e-6:
                t = t.clamp(0.0, 1.0)
                return (t * 1000.0).round().long()
            return t.round().long()
        return t.long()

    def _forward_raw(self, latents: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        if cond.ndim != 3:
            raise ValueError(f"cond must be [B,D] or [B,1,D], got {tuple(cond.shape)}")

        tt = self._to_model_timestep(t)
        out = self.transformer(
            hidden_states=latents,
            encoder_hidden_states=cond,
            timestep=tt,
            return_dict=True,
        )
        pred = out.sample if hasattr(out, "sample") else out[0]

        c = latents.shape[1]
        if pred.shape[1] == 2 * c:
            pred = pred[:, :c]
        elif pred.shape[1] != c:
            raise RuntimeError(f"Unexpected model channels {pred.shape[1]} (expected {c} or {2*c})")
        return pred

    def forward(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        uncond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None].expand(latents.shape[0])
        elif t.ndim == 1 and t.shape[0] == 1:
            t = t.expand(latents.shape[0])

        if uncond is None or guidance_scale == 1.0:
            return self._forward_raw(latents, t, cond)

        hs = torch.cat([latents, latents], dim=0)
        cs = torch.cat([uncond, cond], dim=0)
        ts = torch.cat([t, t], dim=0)
        pred = self._forward_raw(hs, ts, cs)
        n = latents.shape[0]
        pu, pc = pred[:n], pred[n:]
        return pu + guidance_scale * (pc - pu)


def build_pixcell_pipeline(cfg: PixCellConfig, device: torch.device) -> DiffusionPipeline:
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_id, subfolder=cfg.vae_subfolder)
    pipe = DiffusionPipeline.from_pretrained(
        cfg.pix_model_id,
        vae=vae,
        custom_pipeline=cfg.pix_pipeline_id,
        torch_dtype=cfg.dtype,
        trust_remote_code=True,
    )
    return pipe.to(device)


def build_teacher_student(
    pipeline: DiffusionPipeline,
    cond_dim: int,
    init_student_from_teacher: bool = True,
    student_arch: str = "pixcell",
    jit_cfg: Optional[JITStudentConfig] = None,
) -> Tuple[PixCellDenoiser, PixCellDenoiser]:
    teacher_tf = pipeline.transformer
    teacher = PixCellDenoiser(teacher_tf, cond_dim=cond_dim)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    if student_arch == "pixcell":
        if init_student_from_teacher:
            student_tf = copy.deepcopy(teacher_tf)
        else:
            tf_cls = type(teacher_tf)
            if not hasattr(tf_cls, "from_config"):
                raise RuntimeError(
                    "Transformer class does not expose from_config(); set init_student_from_teacher=True"
                )
            student_tf = tf_cls.from_config(teacher_tf.config)
    elif student_arch == "jit":
        if init_student_from_teacher:
            warnings.warn(
                "init_student_from_teacher ignored for student_arch='jit'; building fresh JIT student.",
                RuntimeWarning,
                stacklevel=2,
            )
        cfg = jit_cfg if jit_cfg is not None else JITStudentConfig()
        in_channels = int(pipeline.vae.config.latent_channels)
        student_tf = JITImageTransformerStudent(in_channels=in_channels, cond_dim=cond_dim, cfg=cfg)
    else:
        raise ValueError(f"Unknown student_arch={student_arch}; expected 'pixcell' or 'jit'")

    student = PixCellDenoiser(student_tf, cond_dim=cond_dim)
    for p in student.parameters():
        p.requires_grad = True
    student.train()
    return teacher, student


def make_uncond_embedding(
    pipeline: DiffusionPipeline,
    batch_size: int,
    cond_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if hasattr(pipeline, "get_unconditional_embedding"):
        emb = pipeline.get_unconditional_embedding(batch_size)
        return emb.to(device=device, dtype=dtype)
    return torch.zeros((batch_size, 1, cond_dim), device=device, dtype=dtype)


def _new_scheduler(pipeline: DiffusionPipeline):
    return type(pipeline.scheduler).from_config(pipeline.scheduler.config)


def _interp_state(t_grid: torch.Tensor, states: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
    tg = t_grid
    st = states
    if tg[0] < tg[-1]:
        tg = torch.flip(tg, dims=[0])
        st = torch.flip(st, dims=[0])

    t = t_query.clamp(float(tg[-1]), float(tg[0]))
    if t >= tg[0]:
        return st[0]
    if t <= tg[-1]:
        return st[-1]

    for i in range(len(tg) - 1):
        t0 = tg[i]
        t1 = tg[i + 1]
        if t <= t0 and t >= t1:
            # descending-time linear interpolation
            w = (t0 - t) / (t0 - t1 + 1e-8)
            return st[i] * (1.0 - w) + st[i + 1] * w
    return st[-1]


def scheduler_rollout(
    model: PixCellDenoiser,
    pipeline: DiffusionPipeline,
    xT: torch.Tensor,
    cond: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scheduler = _new_scheduler(pipeline)
    scheduler.set_timesteps(num_steps, device=xT.device)
    timesteps = scheduler.timesteps
    has_scale = hasattr(scheduler, "scale_model_input")

    x = xT
    states = [x]
    b = x.shape[0]
    uncond = make_uncond_embedding(
        pipeline,
        batch_size=b,
        cond_dim=cond.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )

    for t in timesteps:
        x_in = scheduler.scale_model_input(x, t) if has_scale else x
        t_batch = (t if torch.is_tensor(t) else torch.tensor(t, device=x.device))
        if t_batch.ndim == 0:
            t_batch = t_batch[None]
        t_batch = t_batch.expand(b).to(device=x.device, dtype=x.dtype)

        model_out = model(
            latents=x_in,
            t=t_batch,
            cond=cond,
            uncond=uncond,
            guidance_scale=guidance_scale,
        )
        step_out = scheduler.step(model_out, t, x, return_dict=True)
        x = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]
        states.append(x)

    t_nodes = torch.cat(
        [
            timesteps.to(device=x.device, dtype=x.dtype),
            torch.zeros(1, device=x.device, dtype=x.dtype),
        ],
        dim=0,
    )
    return t_nodes, torch.stack(states, dim=0)


@dataclass
class TrajectoryDistillConfig:
    latent_channels: int
    latent_size: int
    teacher_steps: int = 28
    student_min_steps: int = 4
    student_max_steps: int = 8
    guidance_teacher: float = 3.0
    guidance_student: float = 1.0


class TrajectoryDistillationTrainer:
    def __init__(
        self,
        teacher: PixCellDenoiser,
        student: PixCellDenoiser,
        pipeline: DiffusionPipeline,
        cfg: TrajectoryDistillConfig,
    ):
        self.teacher = teacher
        self.student = student
        self.pipeline = pipeline
        self.cfg = cfg

    def _teacher_rollout(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = cond.shape[0]
        xT = torch.randn(
            (b, self.cfg.latent_channels, self.cfg.latent_size, self.cfg.latent_size),
            device=cond.device,
            dtype=cond.dtype,
        )
        with torch.no_grad():
            t_nodes, states = scheduler_rollout(
                self.teacher,
                pipeline=self.pipeline,
                xT=xT,
                cond=cond,
                num_steps=self.cfg.teacher_steps,
                guidance_scale=self.cfg.guidance_teacher,
            )
        return t_nodes, states

    def loss(self, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        t_teacher, s_teacher = self._teacher_rollout(cond)

        m = int(torch.randint(self.cfg.student_min_steps, self.cfg.student_max_steps + 1, (1,)).item())
        scheduler_s = _new_scheduler(self.pipeline)
        scheduler_s.set_timesteps(m, device=cond.device)
        timesteps_s = scheduler_s.timesteps
        t_student_nodes = torch.cat(
            [
                timesteps_s.to(device=cond.device, dtype=cond.dtype),
                torch.zeros(1, device=cond.device, dtype=cond.dtype),
            ],
            dim=0,
        )
        has_scale = hasattr(scheduler_s, "scale_model_input")

        uncond = make_uncond_embedding(
            self.pipeline,
            batch_size=cond.shape[0],
            cond_dim=cond.shape[-1],
            device=cond.device,
            dtype=cond.dtype,
        )

        loss_sum = torch.zeros((), device=cond.device, dtype=cond.dtype)
        n_seg = 0
        for i in range(m):
            thi = t_student_nodes[i]
            tlo = t_student_nodes[i + 1]

            x_hi = _interp_state(t_teacher, s_teacher, thi)
            x_lo_target = _interp_state(t_teacher, s_teacher, tlo)

            x_in = scheduler_s.scale_model_input(x_hi, timesteps_s[i]) if has_scale else x_hi
            t_batch = thi.expand(cond.shape[0])
            model_out = self.student(
                latents=x_in,
                t=t_batch,
                cond=cond,
                uncond=uncond,
                guidance_scale=self.cfg.guidance_student,
            )
            step_out = scheduler_s.step(model_out, timesteps_s[i], x_hi, return_dict=True)
            x_lo_pred = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

            loss_sum = loss_sum + F.mse_loss(x_lo_pred, x_lo_target)
            n_seg += 1

        return {
            "loss": loss_sum / max(1, n_seg),
            "student_steps": torch.tensor(float(m), device=cond.device),
        }


@dataclass
class FlowMatchingConfig:
    latent_channels: int
    latent_size: int
    teacher_steps: int = 28
    guidance_teacher: float = 3.0
    guidance_student: float = 1.0


class FlowMatchingTrainer:
    def __init__(
        self,
        teacher: PixCellDenoiser,
        student: PixCellDenoiser,
        pipeline: DiffusionPipeline,
        cfg: FlowMatchingConfig,
    ):
        self.teacher = teacher
        self.student = student
        self.pipeline = pipeline
        self.cfg = cfg

    def _teacher_sample_xdata(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = cond.shape[0]
        x_noise = torch.randn(
            (b, self.cfg.latent_channels, self.cfg.latent_size, self.cfg.latent_size),
            device=cond.device,
            dtype=cond.dtype,
        )
        with torch.no_grad():
            _, states = scheduler_rollout(
                self.teacher,
                pipeline=self.pipeline,
                xT=x_noise,
                cond=cond,
                num_steps=self.cfg.teacher_steps,
                guidance_scale=self.cfg.guidance_teacher,
            )
            x_data = states[-1]
        return x_noise, x_data

    def loss(self, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_noise, x_data = self._teacher_sample_xdata(cond)
        b = cond.shape[0]

        t = torch.rand((b,), device=cond.device, dtype=cond.dtype)
        view = (b, 1, 1, 1)
        x_t = (1.0 - t.view(view)) * x_noise + t.view(view) * x_data
        v_target = x_data - x_noise

        uncond = make_uncond_embedding(
            self.pipeline,
            batch_size=b,
            cond_dim=cond.shape[-1],
            device=cond.device,
            dtype=cond.dtype,
        )
        v_pred = self.student(
            latents=x_t,
            t=t,
            cond=cond,
            uncond=uncond,
            guidance_scale=self.cfg.guidance_student,
        )
        return {"loss": F.mse_loss(v_pred, v_target)}


def sample_student_trajectory(
    student: PixCellDenoiser,
    cond: torch.Tensor,
    pipeline: DiffusionPipeline,
    latent_channels: int,
    latent_size: int,
    steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    b = cond.shape[0]
    p0 = next(student.parameters())
    device = p0.device
    dtype = p0.dtype
    cond = cond.to(device=device, dtype=dtype)
    x = torch.randn((b, latent_channels, latent_size, latent_size), device=device, dtype=dtype)
    _, states = scheduler_rollout(
        student,
        pipeline=pipeline,
        xT=x,
        cond=cond,
        num_steps=steps,
        guidance_scale=guidance_scale,
    )
    return states[-1]


def sample_student_flow(
    student: PixCellDenoiser,
    cond: torch.Tensor,
    pipeline: DiffusionPipeline,
    latent_channels: int,
    latent_size: int,
    steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    b = cond.shape[0]
    p0 = next(student.parameters())
    device = p0.device
    dtype = p0.dtype
    cond = cond.to(device=device, dtype=dtype)

    x = torch.randn((b, latent_channels, latent_size, latent_size), device=device, dtype=dtype)
    t_grid = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
    uncond = make_uncond_embedding(
        pipeline,
        batch_size=b,
        cond_dim=cond.shape[-1],
        device=device,
        dtype=dtype,
    )

    for i in range(len(t_grid) - 1):
        ti = t_grid[i]
        tj = t_grid[i + 1]
        dt = (tj - ti).to(x.dtype)
        t_batch = ti.expand(b)
        v = student(
            latents=x,
            t=t_batch,
            cond=cond,
            uncond=uncond,
            guidance_scale=guidance_scale,
        )
        x = x + dt * v
    return x


@torch.no_grad()
def decode_latents_to_images(pipeline: DiffusionPipeline, latents: torch.Tensor) -> torch.Tensor:
    vae_param = next(pipeline.vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    if hasattr(pipeline.vae.config, "scaling_factor"):
        latents = latents / pipeline.vae.config.scaling_factor
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1e4, neginf=-1e4)
    img = pipeline.vae.decode(latents, return_dict=True).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    return img
