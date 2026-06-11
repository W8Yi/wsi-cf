from __future__ import annotations

import os
import random

import numpy as np
import torch


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_arg)
        if device.index is not None and device.index >= torch.cuda.device_count():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if torch.cuda.device_count() == 1:
                visible_msg = visible if visible else "<unset>"
                print(
                    f"[wsi_cf] Requested {device_arg}, but only one CUDA device is visible "
                    f"(CUDA_VISIBLE_DEVICES={visible_msg}); using cuda:0.",
                    flush=True,
                )
                return torch.device("cuda:0")
            raise RuntimeError(
                f"Requested {device_arg}, but torch sees only {torch.cuda.device_count()} CUDA device(s). "
                "Use a visible device index such as cuda:0, or set CUDA_VISIBLE_DEVICES before launching."
            )
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)
