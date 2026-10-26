"""Acquisition functions: turn a surrogate's ``(mean, std)`` into a ranking.

The platform maximizes the experimental response. Every acquisition function
here trades off *exploitation* (high predicted mean) against *exploration*
(high predictive uncertainty):

* ``ucb`` — upper confidence bound, ``mu + beta * sigma`` (handoff §11). ``beta``
  is the explicit exploration knob: ``beta=0`` is greedy, large ``beta`` is
  close to pure uncertainty sampling.
* ``expected_improvement`` — closed-form EI over the incumbent best observed
  value (Gaussian predictive assumption).
* ``thompson`` — one draw from ``N(mu, sigma^2)`` per candidate (seeded), a
  cheap randomized strategy that naturally diversifies batches.

All functions are pure NumPy (no ray/dagster/mlflow) and deterministic.
Ranking breaks ties by ``candidate_id`` so the output never depends on input
row order or sort-algorithm details.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

Strategy = Literal["ucb", "ei", "thompson", "greedy", "uncertainty"]
STRATEGIES: tuple[str, ...] = ("ucb", "ei", "thompson", "greedy", "uncertainty")

FloatArray = NDArray[np.float64]


def _arrays(mu: ArrayLike, sigma: ArrayLike) -> tuple[FloatArray, FloatArray]:
    m = np.asarray(mu, dtype=np.float64).reshape(-1)
    s = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if m.shape != s.shape:
        raise ValueError(f"mu and sigma shapes differ: {m.shape} vs {s.shape}")
    if (s < 0).any():
        raise ValueError("sigma must be non-negative")
    return m, s


def ucb(mu: ArrayLike, sigma: ArrayLike, beta: float = 1.0) -> FloatArray:
    """Upper confidence bound ``mu + beta * sigma``."""
    m, s = _arrays(mu, sigma)
    return m + float(beta) * s


def expected_improvement(
    mu: ArrayLike, sigma: ArrayLike, best: float, xi: float = 0.0
) -> FloatArray:
    """Expected improvement over ``best`` (maximization), Gaussian predictive.

    ``EI = (mu - best - xi) * Phi(z) + sigma * phi(z)``, ``z = (mu - best - xi) / sigma``.
    """
    m, s = _arrays(mu, sigma)
    s_safe = np.maximum(s, 1e-12)
    imp = m - float(best) - float(xi)
    z = imp / s_safe
    ei = imp * norm.cdf(z) + s_safe * norm.pdf(z)
    return np.where(s > 0, np.maximum(ei, 0.0), np.maximum(imp, 0.0))


def thompson(mu: ArrayLike, sigma: ArrayLike, seed: int = 0) -> FloatArray:
    """One posterior draw per candidate from ``N(mu, sigma^2)`` (seeded)."""
    m, s = _arrays(mu, sigma)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7503]))
    return m + s * rng.standard_normal(m.shape[0])


def acquisition_scores(
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    strategy: Strategy = "ucb",
    beta: float = 1.0,
    best: float | None = None,
    seed: int = 0,
) -> FloatArray:
    """Dispatch to one acquisition function by name."""
    if strategy == "ucb":
        return ucb(mu, sigma, beta)
    if strategy == "greedy":
        return ucb(mu, sigma, 0.0)
    if strategy == "uncertainty":
        return _arrays(mu, sigma)[1].copy()
    if strategy == "ei":
        if best is None:
            raise ValueError("strategy='ei' needs `best` (incumbent best observed response)")
        return expected_improvement(mu, sigma, best)
    if strategy == "thompson":
        return thompson(mu, sigma, seed)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


def rank_candidates(
    frame: pd.DataFrame,
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    beta: float = 1.0,
    strategy: Strategy = "ucb",
    best: float | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Score and rank candidates (rank 1 = most attractive).

    ``frame`` must hold a ``candidate_id`` column aligned row-by-row with
    ``mu``/``sigma``. Returns a new frame with columns
    ``candidate_id, pred_mean, pred_std, score, rank`` sorted by rank. Ties in
    score are broken by ``candidate_id`` (ascending) for determinism.
    """
    m, s = _arrays(mu, sigma)
    if len(frame) != m.shape[0]:
        raise ValueError(f"frame has {len(frame)} rows but {m.shape[0]} predictions")
    score = acquisition_scores(m, s, strategy=strategy, beta=beta, best=best, seed=seed)
    out = pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"].astype(str).to_numpy(),
            "pred_mean": m,
            "pred_std": s,
            "score": score,
        }
    )
    # lexsort: last key is primary -> sort by -score, then candidate_id
    order = np.lexsort((out["candidate_id"].to_numpy(), -score))
    out = out.iloc[order].reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out
