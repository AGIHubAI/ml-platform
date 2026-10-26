"""PyTorch surrogate models (pure torch; no orchestration imports)."""

from ml_platform.models.losses import gaussian_nll_loss, mse_loss
from ml_platform.models.mlp import ResidualMLP, build_model, count_parameters, model_from_spec
from ml_platform.models.uncertainty import mc_dropout_predict

__all__ = [
    "ResidualMLP",
    "build_model",
    "count_parameters",
    "gaussian_nll_loss",
    "mc_dropout_predict",
    "model_from_spec",
    "mse_loss",
]
