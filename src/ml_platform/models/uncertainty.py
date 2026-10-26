"""Monte-Carlo dropout predictive uncertainty (Gal & Ghahramani, 2016).

Dropout stays active at inference; the spread of predictions across
stochastic forward passes approximates *epistemic* uncertainty. It is a
cheap, rough estimate — adequate for demonstrating uncertainty-aware active
learning, not a calibrated posterior.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn

from ml_platform.models.mlp import ResidualMLP


def _enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    X: ArrayLike | torch.Tensor,
    n_samples: int = 30,
    *,
    seed: int | None = 0,
    batch_size: int = 8192,
    device: torch.device | str | None = None,
    return_aleatoric: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(mean, std)`` over ``n_samples`` stochastic forward passes, shape ``(n,)``.

    Values are in the model's (standardized) output units. ``seed`` makes the
    dropout masks reproducible without disturbing the global RNG. If the model
    has a variance head and ``return_aleatoric`` is True, the std includes the
    predicted aleatoric variance (law of total variance).
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    Xt = X if isinstance(X, torch.Tensor) else torch.as_tensor(np.asarray(X), dtype=torch.float32)
    Xt = Xt.to(dtype=torch.float32)
    was_training = model.training
    has_var = isinstance(model, ResidualMLP) and model.predict_variance
    means = []
    alea = []
    devices = [dev] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(seed)
        _enable_dropout_only(model)
        try:
            for start in range(0, Xt.shape[0], batch_size):
                xb = Xt[start : start + batch_size].to(dev)
                samples = []
                var_samples = []
                for _ in range(n_samples):
                    if has_var:
                        mu, log_var = model.forward_with_log_var(xb)  # type: ignore[operator]
                        var_samples.append(torch.exp(log_var))
                    else:
                        mu = model(xb)
                    samples.append(mu)
                stacked = torch.stack(samples)  # (S, B)
                means.append(stacked)
                if has_var:
                    alea.append(torch.stack(var_samples).mean(0))
        finally:
            model.train(was_training)
    S = torch.cat(means, dim=1).double().cpu().numpy()  # (S, n)
    mean = S.mean(axis=0)
    var = S.var(axis=0, ddof=1)
    if has_var and return_aleatoric:
        var = var + torch.cat(alea).double().cpu().numpy()
    return mean, np.sqrt(np.maximum(var, 1e-12))
