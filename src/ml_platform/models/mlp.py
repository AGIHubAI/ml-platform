"""Residual MLP surrogate model (deliberately simple)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from ml_platform.config import PlatformConfig


class ResidualBlock(nn.Module):
    """``y = act(Linear(x)) -> dropout``, plus a (projected) skip connection."""

    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.linear = nn.Linear(d_in, d_out)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip: nn.Module = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.dropout(self.act(self.linear(self.norm(x))))


class ResidualMLP(nn.Module):
    """``in_dim -> hidden_dims... -> 1`` residual MLP with dropout (for MC dropout).

    ``forward(x)`` returns the predicted mean, shape ``(B,)``. With
    ``predict_variance=True`` a second head predicts log-variance (used by the
    Gaussian-NLL loss); read it with `forward_with_log_var`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.1,
        predict_variance: bool = False,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        self.in_dim = int(in_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.dropout_p = float(dropout)
        self.predict_variance = bool(predict_variance)

        self.stem = nn.Linear(self.in_dim, self.hidden_dims[0])
        dims = self.hidden_dims
        self.blocks = nn.ModuleList(
            ResidualBlock(dims[max(i - 1, 0)], dims[i], self.dropout_p) for i in range(len(dims))
        )
        self.head_norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], 2 if self.predict_variance else 1)

    def spec(self) -> dict[str, Any]:
        """Constructor spec stored in checkpoints so the model can be rebuilt."""
        return {
            "class": "ResidualMLP",
            "kwargs": {
                "in_dim": self.in_dim,
                "hidden_dims": list(self.hidden_dims),
                "dropout": self.dropout_p,
                "predict_variance": self.predict_variance,
            },
        }

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.head_norm(h)

    def forward_with_log_var(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.head(self._features(x))
        if self.predict_variance:
            return out[:, 0], out[:, 1].clamp(-10.0, 5.0)
        return out[:, 0], None

    def forward(self, x: torch.Tensor, return_log_var: bool = False) -> Any:
        """Mean prediction ``(B,)``; with ``return_log_var`` a ``(mean, log_var)`` tuple
        (goes through ``forward`` so DDP hooks still fire)."""
        mu, log_var = self.forward_with_log_var(x)
        return (mu, log_var) if return_log_var else mu


def build_model(cfg: PlatformConfig) -> ResidualMLP:
    """Default model factory: ``ResidualMLP`` from ``cfg.model`` / ``cfg.data``."""
    return ResidualMLP(
        in_dim=cfg.data.n_features,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        predict_variance=cfg.training.loss == "gaussian_nll",
    )


def model_from_spec(spec: dict[str, Any]) -> nn.Module:
    if spec.get("class") != "ResidualMLP":
        raise ValueError(f"unknown model class {spec.get('class')!r}")
    return ResidualMLP(**spec["kwargs"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
