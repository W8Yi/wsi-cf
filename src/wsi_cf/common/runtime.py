from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)
