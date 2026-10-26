"""Paired model comparison on the same evaluation examples.

Both models are scored on *identical* examples and every bootstrap resample
uses the same indices for both, so example difficulty cancels out and the
comparison is far more sensitive than comparing two independent CIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

Metric = Literal["rmse", "mae", "squared_error"]


@dataclass(frozen=True)
class ComparisonResult:
    """``diff = metric(a) - metric(b)``: positive means **b is better** (lower error).

    Convention: ``a`` = baseline / incumbent, ``b`` = candidate.
    """

    metric: str
    n: int
    a_value: float
    b_value: float
    diff: float
    ci_lo: float
    ci_hi: float
    relative_improvement: float
    p_b_not_better: float
    n_boot: int

    @property
    def b_better(self) -> bool:
        """Candidate significantly better: the whole CI of the difference is > 0."""
        return self.ci_lo > 0.0

    @property
    def a_better(self) -> bool:
        return self.ci_hi < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "b_better": self.b_better, "a_better": self.a_better}


def _score(err_a: np.ndarray, err_b: np.ndarray, metric: Metric) -> tuple[float, float]:
    if metric == "rmse":
        return float(np.sqrt(np.mean(err_a**2))), float(np.sqrt(np.mean(err_b**2)))
    if metric == "mae":
        return float(np.mean(np.abs(err_a))), float(np.mean(np.abs(err_b)))
    return float(np.mean(err_a**2)), float(np.mean(err_b**2))


def paired_compare(
    y: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    metric: Metric = "rmse",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Paired bootstrap comparison of two prediction vectors on the same targets."""
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    ea = np.asarray(pred_a, dtype=np.float64).reshape(-1) - ya
    eb = np.asarray(pred_b, dtype=np.float64).reshape(-1) - ya
    if not (len(ea) == len(eb) == len(ya)):
        raise ValueError("y, pred_a and pred_b must have the same length")
    a_val, b_val = _score(ea, eb, metric)
    diff = a_val - b_val
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(ya), size=len(ya))
        sa, sb = _score(ea[idx], eb[idx], metric)
        diffs[i] = sa - sb
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2]) if n_boot > 0 else (diff, diff)
    return ComparisonResult(
        metric=metric,
        n=len(ya),
        a_value=a_val,
        b_value=b_val,
        diff=diff,
        ci_lo=float(lo),
        ci_hi=float(hi),
        relative_improvement=diff / a_val if a_val > 0 else 0.0,
        p_b_not_better=float(np.mean(diffs <= 0)) if n_boot > 0 else float("nan"),
        n_boot=n_boot,
    )
