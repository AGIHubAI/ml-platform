from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_platform.config import EvaluationConfig
from ml_platform.evaluation import (
    bootstrap_ci,
    check_gates,
    coverage,
    evaluate,
    mae,
    nll_gaussian,
    paired_compare,
    r2,
    rmse,
)
from ml_platform.inference import Predictor


def test_metrics_hand_computed() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([1.0, 2.0, 4.0, 2.0])
    # errors: 0, 0, -1, 2
    assert mae(y, yhat) == pytest.approx(0.75)
    assert rmse(y, yhat) == pytest.approx(math.sqrt(5 / 4))
    # SST = 5, SSE = 5 -> R2 = 0
    assert r2(y, yhat) == pytest.approx(0.0)
    assert r2(y, y) == pytest.approx(1.0)
    sigma = np.ones(4)
    expected_nll = 0.5 * math.log(2 * math.pi) + 0.5 * (0 + 0 + 1 + 4) / 4
    assert nll_gaussian(y, yhat, sigma) == pytest.approx(expected_nll)
    assert coverage(y, yhat, sigma, z=1.96) == pytest.approx(0.75)
    assert coverage(y, yhat, sigma * 2, z=1.0) == pytest.approx(1.0)


def test_bootstrap_ci_contains_point_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    yhat = y + rng.normal(scale=0.5, size=200)
    pt, lo, hi = bootstrap_ci(rmse, y, yhat, n=300, seed=1)
    assert lo < pt < hi
    assert 0.4 < pt < 0.6
    assert bootstrap_ci(rmse, y, yhat, n=300, seed=1) == (pt, lo, hi)


def test_paired_compare_detects_better_model() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    worse = y + rng.normal(scale=1.0, size=300)
    better = y + rng.normal(scale=0.5, size=300)
    res = paired_compare(y, worse, better, metric="rmse", n_boot=500)
    assert res.b_better and not res.a_better
    assert res.diff > 0 and res.relative_improvement > 0.3
    assert res.p_b_not_better < 0.01
    rev = paired_compare(y, better, worse, metric="rmse", n_boot=500)
    assert rev.a_better and not rev.b_better
    same = paired_compare(y, worse, worse, metric="mae", n_boot=200)
    assert same.diff == 0 and not same.b_better
    assert set(res.to_dict()) >= {"b_better", "ci_lo", "ci_hi"}


def test_gates_pass_fail() -> None:
    cfg = EvaluationConfig(max_rmse=0.35, min_improvement_vs_baseline=0.02)
    ok = check_gates(0.30, 0.10, cfg, "mean")
    assert ok.passed and all(r.startswith("PASS") for r in ok.reasons)
    too_high = check_gates(0.40, 0.10, cfg, "mean")
    assert not too_high.passed and too_high.reasons[0].startswith("FAIL")
    no_gain = check_gates(0.30, 0.01, cfg, "prod")
    assert not no_gain.passed and no_gain.reasons[1].startswith("FAIL")
    assert not no_gain.checks["min_improvement_vs_baseline"]["passed"]


def test_evaluate_end_to_end_and_write(trained, tmp_path: Path) -> None:
    result, _, va = trained
    pred = Predictor.from_checkpoint(result.checkpoint_path)
    cfg = EvaluationConfig(
        max_rmse=5.0, min_improvement_vs_baseline=0.02, bootstrap_samples=100, mc_samples=5
    )
    ev = evaluate(pred, va, None, cfg)
    assert ev.baseline_name.startswith("mean_predictor")
    assert ev.metrics["rmse_ci_lo"] <= ev.metrics["rmse"] <= ev.metrics["rmse_ci_hi"]
    assert ev.metrics["rmse"] < ev.metrics["baseline_rmse"]
    assert ev.gate.passed  # loose max_rmse; model beats mean predictor
    assert 0.0 <= ev.metrics["coverage_95"] <= 1.0
    assert len(ev.predictions) == len(va)
    assert "round_id=0" in ev.slices

    paths = ev.write(tmp_path / "eval")
    for name in ("metrics.json", "report.md", "predictions.parquet", "comparison.json"):
        assert (tmp_path / "eval" / name).exists()
    metrics = json.loads(paths["metrics"].read_text())
    assert metrics["gate"]["passed"] is True
    assert "Gate decision" in paths["report"].read_text()
    assert len(pd.read_parquet(paths["predictions"])) == len(va)

    strict = evaluate(pred, va, None, cfg.model_copy(update={"max_rmse": 0.01}))
    assert not strict.gate.passed

    # an incumbent that is identical to the candidate yields no improvement -> gate fails
    same = evaluate(pred, va, pred.predict(va), cfg, baseline_name="production_v1")
    assert same.metrics["improvement_vs_baseline"] == pytest.approx(0.0)
    assert not same.gate.passed
    with pytest.raises(ValueError):
        evaluate(pred, va, np.zeros(3), cfg)


def test_predictor_uncertainty_and_info(trained) -> None:
    result, _, va = trained
    p = Predictor.from_checkpoint(result.checkpoint_path)
    X = va.filter(regex=r"^f\d+$").to_numpy()
    mean, std = p.predict_with_uncertainty(X, n_samples=10)
    assert mean.shape == std.shape == (len(va),)
    assert (std > 0).all()
    _, std_total = p.predict_with_uncertainty(X, n_samples=10, include_aleatoric=True)
    assert (std_total >= std).all()
    # MC mean close to the deterministic prediction; outputs are in raw units
    assert np.corrcoef(mean, p.predict(X))[0, 1] > 0.95
    info = p.model_info()
    assert info["epochs_trained"] == 5 and info["n_parameters"] > 0
    assert len(info["checkpoint_hash"]) == 64
    with pytest.raises(ValueError):
        p.predict(np.zeros((2, 3)))


def test_standardization_scale_belongs_to_the_dataset(trained, tmp_path: Path) -> None:
    """Gate units come from the evaluation dataset's scale, not the candidate's checkpoint."""
    from ml_platform.evaluation import TargetScale

    result, tr, va = trained
    pred = Predictor.from_checkpoint(result.checkpoint_path)
    cfg = EvaluationConfig(bootstrap_samples=50, mc_samples=3)

    fallback = evaluate(pred, va, None, cfg)
    assert fallback.target_scale["source"].startswith("candidate checkpoint")
    assert fallback.target_scale["y_std"] == pytest.approx(pred.normalizer.y_std)

    scale = TargetScale.from_targets(tr["response"], source="training split of test dataset")
    ev = evaluate(pred, va, None, cfg, target_scale=scale)
    assert ev.target_scale == scale.to_dict() and ev.target_scale["n"] == len(tr)
    # standardized RMSE = raw RMSE / dataset std, whatever the candidate's own stats are
    assert ev.metrics["rmse"] == pytest.approx(ev.metrics["rmse_raw"] / scale.y_std)
    wider = TargetScale(scale.y_mean, 2 * scale.y_std, "x2")
    ev2 = evaluate(pred, va, None, cfg, target_scale=wider)
    assert ev2.metrics["rmse"] == pytest.approx(ev.metrics["rmse"] / 2)
    # the mean baseline uses the scale's (training) mean, so improvement is scale-free
    assert ev2.metrics["improvement_vs_baseline"] == pytest.approx(
        ev.metrics["improvement_vs_baseline"]
    )

    paths = ev.write(tmp_path / "eval")
    written = json.loads(paths["metrics"].read_text())
    assert written["standardization"]["source"] == "training split of test dataset"
    assert "training split of test dataset" in paths["report"].read_text()
