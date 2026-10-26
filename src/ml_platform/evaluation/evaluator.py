"""Model evaluation: metrics with CIs, paired baseline comparison, promotion gates.

Metrics used for gating are computed on *standardized* responses
``z = (y - mean) / std``. The yardstick is a `TargetScale` that belongs to
the **evaluation dataset**, not to the model being evaluated: callers pass the
target mean/std of the training split of the dataset the evaluation belongs to
(the closed loop uses the training split of ``union(round_000..round_r)``).
Candidate and baseline are therefore scored in identical units, and
``max_rmse`` means the same thing for every candidate evaluated on a round,
whatever statistics that candidate happened to be trained with. Only when no
scale is given does `evaluate` fall back to the candidate checkpoint's own
normalizer (recorded as such in ``metrics.json`` / ``report.md``).

Baseline: pass the incumbent (production) model's predictions on the same
``eval_frame`` as ``baseline_predictions``. Without an incumbent, the baseline
is a mean predictor using the *training* target mean of the scale (no
evaluation-set leakage).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ml_platform.config import EvaluationConfig, PlatformConfig
from ml_platform.data.normalization import Normalizer
from ml_platform.evaluation import metrics as M
from ml_platform.evaluation.comparison import ComparisonResult, paired_compare


class PredictorLike(Protocol):
    normalizer: Normalizer
    feature_columns: list[str]

    def predict(self, X: Any) -> NDArray[np.float64]: ...

    def predict_with_uncertainty(
        self,
        X: Any,
        n_samples: int | None = None,
        *,
        seed: int | None = 0,
        include_aleatoric: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def model_info(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TargetScale:
    """The fixed yardstick for standardized metrics (``z = (y - y_mean) / y_std``)."""

    y_mean: float
    y_std: float
    source: str
    n: int | None = None

    @classmethod
    def from_targets(cls, y: ArrayLike, source: str) -> TargetScale:
        """Mean/std (population, like `Normalizer.fit`) of training targets."""
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        if ya.size == 0:
            raise ValueError("cannot derive a target scale from an empty array")
        std = float(ya.std())
        return cls(float(ya.mean()), std if std > 1e-8 else 1.0, source, int(ya.size))

    @classmethod
    def from_normalizer(cls, norm: Normalizer, source: str) -> TargetScale:
        return cls(float(norm.y_mean), float(norm.y_std), source)

    def to_dict(self) -> dict[str, Any]:
        return {"y_mean": self.y_mean, "y_std": self.y_std, "source": self.source, "n": self.n}


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str]
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons), "checks": self.checks}


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    gate: GateDecision
    comparison: ComparisonResult
    predictions: pd.DataFrame
    slices: dict[str, dict[str, float]] = field(default_factory=dict)
    baseline_name: str = "mean_predictor"
    model_info: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    target_scale: dict[str, Any] = field(default_factory=dict)

    def write(self, out_dir: str | os.PathLike[str]) -> dict[str, Path]:
        """Write ``metrics.json``, ``report.md``, ``predictions.parquet``, ``comparison.json``."""
        from ml_platform.evaluation.reporting import write_evaluation

        return write_evaluation(self, Path(out_dir))


def check_gates(
    rmse_std: float, improvement: float, cfg: EvaluationConfig, baseline_name: str
) -> GateDecision:
    checks: dict[str, dict[str, Any]] = {
        "max_rmse": {
            "value": rmse_std,
            "threshold": cfg.max_rmse,
            "passed": bool(rmse_std <= cfg.max_rmse),
        },
        "min_improvement_vs_baseline": {
            "value": improvement,
            "threshold": cfg.min_improvement_vs_baseline,
            "baseline": baseline_name,
            "passed": bool(improvement >= cfg.min_improvement_vs_baseline),
        },
    }
    reasons = [
        (
            f"{'PASS' if checks['max_rmse']['passed'] else 'FAIL'}: standardized RMSE "
            f"{rmse_std:.4f} {'<=' if checks['max_rmse']['passed'] else '>'} max_rmse {cfg.max_rmse}"
        ),
        (
            f"{'PASS' if checks['min_improvement_vs_baseline']['passed'] else 'FAIL'}: "
            f"relative RMSE improvement vs {baseline_name} {improvement:+.2%} "
            f"{'>=' if checks['min_improvement_vs_baseline']['passed'] else '<'} "
            f"{cfg.min_improvement_vs_baseline:.2%}"
        ),
    ]
    return GateDecision(
        passed=all(c["passed"] for c in checks.values()), reasons=reasons, checks=checks
    )


def _slice_metrics(
    df: pd.DataFrame, key: str, y_std_units: str, pred_std_units: str
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if key not in df.columns:
        return out
    for value, g in df.groupby(key, sort=True):
        out[f"{key}={value}"] = M.regression_metrics(g[y_std_units], g[pred_std_units])
    return out


def evaluate(
    predictor: PredictorLike,
    eval_frame: pd.DataFrame,
    baseline_predictions: ArrayLike | None = None,
    cfg: PlatformConfig | EvaluationConfig | None = None,
    *,
    baseline_name: str | None = None,
    seed: int = 0,
    target_scale: TargetScale | None = None,
) -> EvaluationResult:
    """Evaluate ``predictor`` on ``eval_frame`` (needs feature columns + ``response``).

    ``target_scale``: the dataset's yardstick for standardized metrics (see the
    module docstring). Pass it whenever several models are compared or gated on
    the same data; ``None`` falls back to the candidate's checkpoint normalizer.
    """
    ecfg = (
        cfg.evaluation
        if isinstance(cfg, PlatformConfig)
        else (cfg if cfg is not None else EvaluationConfig())
    )
    frame = eval_frame.reset_index(drop=True)
    if len(frame) == 0:
        raise ValueError("eval_frame is empty")
    scale = target_scale or TargetScale.from_normalizer(
        predictor.normalizer, source="candidate checkpoint normalizer (no dataset scale given)"
    )
    y = frame["response"].to_numpy(dtype=np.float64)
    pred = predictor.predict(frame)
    mu_mc, sigma = predictor.predict_with_uncertainty(
        frame, ecfg.mc_samples, seed=seed, include_aleatoric=True
    )
    _, sigma_epi = predictor.predict_with_uncertainty(frame, ecfg.mc_samples, seed=seed)

    if baseline_predictions is None:
        base = np.full_like(y, scale.y_mean)
        baseline_name = baseline_name or "mean_predictor(train_mean)"
    else:
        base = np.asarray(baseline_predictions, dtype=np.float64).reshape(-1)
        if base.shape != y.shape:
            raise ValueError("baseline_predictions must align with eval_frame rows")
        baseline_name = baseline_name or "baseline_model"

    def z(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return (v - scale.y_mean) / scale.y_std

    ys, ps, bs = z(y), z(pred), z(base)
    rmse_pt, rmse_lo, rmse_hi = M.bootstrap_ci(M.rmse, ys, ps, n=ecfg.bootstrap_samples, seed=seed)
    comparison = paired_compare(ys, bs, ps, metric="rmse", n_boot=ecfg.bootstrap_samples, seed=seed)
    improvement = comparison.relative_improvement
    sig_s = sigma / scale.y_std

    metrics: dict[str, float] = {
        "rmse": rmse_pt,
        "rmse_ci_lo": rmse_lo,
        "rmse_ci_hi": rmse_hi,
        "mae": M.mae(ys, ps),
        "r2": M.r2(ys, ps),
        "rmse_raw": M.rmse(y, pred),
        "mae_raw": M.mae(y, pred),
        "nll": M.nll_gaussian(ys, z(mu_mc), sig_s),
        "coverage_95": M.coverage(ys, z(mu_mc), sig_s, z=1.96),
        "mean_predictive_std": float(np.mean(sig_s)),
        "mean_epistemic_std": float(np.mean(sigma_epi / scale.y_std)),
        "baseline_rmse": comparison.a_value,
        "improvement_vs_baseline": improvement,
        "improvement_ci_lo": comparison.ci_lo / comparison.a_value if comparison.a_value else 0.0,
        "improvement_ci_hi": comparison.ci_hi / comparison.a_value if comparison.a_value else 0.0,
        "n_eval": float(len(frame)),
    }
    gate = check_gates(rmse_pt, improvement, ecfg, baseline_name)

    predictions = pd.DataFrame(
        {
            "experiment_id": frame["experiment_id"].astype(str)
            if "experiment_id" in frame.columns
            else [str(i) for i in range(len(frame))],
            "round_id": frame["round_id"].to_numpy()
            if "round_id" in frame.columns
            else np.zeros(len(frame), dtype=np.int64),
            "y_true": y,
            "y_pred": pred,
            "y_pred_mc_mean": mu_mc,
            "y_pred_std": sigma,
            "y_pred_std_epistemic": sigma_epi,
            "baseline_pred": base,
            "residual": y - pred,
            "y_true_std_units": ys,
            "y_pred_std_units": ps,
        }
    )
    slices = _slice_metrics(predictions, "round_id", "y_true_std_units", "y_pred_std_units")
    return EvaluationResult(
        metrics=metrics,
        gate=gate,
        comparison=comparison,
        predictions=predictions,
        slices=slices,
        baseline_name=baseline_name,
        model_info=predictor.model_info(),
        config=ecfg.model_dump(mode="json"),
        target_scale=scale.to_dict(),
    )
