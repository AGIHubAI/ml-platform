"""Training losses (all on standardized targets)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.mse_loss(pred, target, reduction=reduction)


def gaussian_nll_loss(
    mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL, ``0.5 * (log s^2 + (y - mu)^2 / s^2)`` (+const dropped)."""
    nll = 0.5 * (log_var + (target - mean) ** 2 * torch.exp(-log_var))
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll
