from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProportionalOddsRisk(nn.Module):
    """Linear ordinal risk model with ordered cumulative-grade thresholds."""

    def __init__(self, n_features: int, n_thresholds: int = 3):
        super().__init__()
        if int(n_thresholds) < 1:
            raise ValueError("n_thresholds must be positive")
        self.linear = nn.Linear(int(n_features), 1, bias=False)
        self.threshold_base = nn.Parameter(torch.tensor(-1.0))
        if int(n_thresholds) > 1:
            initial_delta = torch.log(torch.expm1(torch.tensor(1.0)))
            self.threshold_delta_raw = nn.Parameter(initial_delta.repeat(int(n_thresholds) - 1))
        else:
            self.register_parameter("threshold_delta_raw", None)
        self.n_thresholds = int(n_thresholds)

    @property
    def thresholds(self) -> torch.Tensor:
        if self.threshold_delta_raw is None:
            return self.threshold_base.reshape(1)
        deltas = F.softplus(self.threshold_delta_raw)
        return torch.cat((self.threshold_base.reshape(1), self.threshold_base + torch.cumsum(deltas, dim=0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        severity = self.linear(x).reshape(-1)
        logits = severity[:, None] - self.thresholds[None, :]
        probabilities = torch.sigmoid(logits)
        risk = probabilities.mean(dim=1)
        return risk, probabilities, logits
