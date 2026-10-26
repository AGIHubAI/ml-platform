"""Regression and uncertainty metrics + bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

MetricFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


def _arr(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def mae(y: ArrayLike, yhat: ArrayLike) -> float:
    return float(np.mean(np.abs(_arr(y) - _arr(yhat))))


def rmse(y: ArrayLike, yhat: ArrayLike) -> float:
    return float(np.sqrt(np.mean((_arr(y) - _arr(yhat)) ** 2)))


def r2(y: ArrayLike, yhat: ArrayLike) -> float:
    ya, yh = _arr(y), _arr(yhat)
    sst = float(np.sum((ya - ya.mean()) ** 2))
    if sst == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((ya - yh) ** 2)) / sst


def nll_gaussian(y: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> float:
    """Mean Gaussian negative log-likelihood (including the 0.5*log(2*pi) constant)."""
    ya, m, s = _arr(y), _arr(mu), np.maximum(_arr(sigma), 1e-12)
    return float(np.mean(0.5 * np.log(2 * np.pi * s**2) + 0.5 * ((ya - m) / s) ** 2))


def coverage(y: ArrayLike, mu: ArrayLike, sigma: ArrayLike, z: float = 1.96) -> float:
    """Fraction of targets inside ``mu ± z*sigma`` (0.95 expected for z=1.96 if calibrated)."""
    ya, m, s = _arr(y), _arr(mu), _arr(sigma)
    return float(np.mean(np.abs(ya - m) <= z * s))


def bootstrap_ci(
    metric_fn: MetricFn,
    y: ArrayLike,
    yhat: ArrayLike,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap: ``(point_estimate, lo, hi)`` for ``metric_fn(y, yhat)``."""
    ya, yh = _arr(y), _arr(yhat)
    point = float(metric_fn(ya, yh))
    if len(ya) < 2 or n <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, len(ya), size=len(ya))
        stats[b] = metric_fn(ya[idx], yh[idx])
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def regression_metrics(y: ArrayLike, yhat: ArrayLike) -> dict[str, float]:
    return {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "r2": r2(y, yhat), "n": float(len(_arr(y)))}
