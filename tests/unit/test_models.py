from __future__ import annotations

import numpy as np
import pytest
import torch

from ml_platform.config import PlatformConfig
from ml_platform.models import (
    ResidualMLP,
    build_model,
    count_parameters,
    gaussian_nll_loss,
    mc_dropout_predict,
    model_from_spec,
)


def test_forward_shape() -> None:
    m = ResidualMLP(32, [256, 256, 128], 0.1)
    out = m(torch.randn(7, 32))
    assert out.shape == (7,)
    mu, lv = m(torch.randn(3, 32), return_log_var=True)
    assert mu.shape == (3,) and lv is None


def test_variance_head() -> None:
    m = ResidualMLP(8, [16], 0.0, predict_variance=True)
    mu, log_var = m.forward_with_log_var(torch.randn(5, 8))
    assert mu.shape == (5,) and log_var is not None and log_var.shape == (5,)
    loss = gaussian_nll_loss(mu, log_var, torch.zeros(5))
    loss.backward()


def test_build_from_cfg_and_spec() -> None:
    cfg = PlatformConfig.for_tests()
    m = build_model(cfg)
    assert isinstance(m, ResidualMLP) and m.hidden_dims == [32, 32]
    rebuilt = model_from_spec(m.spec())
    assert count_parameters(rebuilt) == count_parameters(m)
    with pytest.raises(ValueError):
        model_from_spec({"class": "Nope", "kwargs": {}})


def test_mc_dropout_shapes_and_reproducibility() -> None:
    torch.manual_seed(0)
    m = ResidualMLP(4, [16, 16], 0.2).eval()
    X = np.random.default_rng(0).normal(size=(11, 4))
    mean, std = mc_dropout_predict(m, X, n_samples=8, seed=1)
    assert mean.shape == (11,) and std.shape == (11,)
    assert (std > 0).all()
    mean2, std2 = mc_dropout_predict(m, X, n_samples=8, seed=1)
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(std, std2)
    assert not m.training  # mode restored
