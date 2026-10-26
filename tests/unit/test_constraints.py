"""Composable domain constraints and rejection accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_platform.active_learning import (
    Constraint,
    ExcludeIds,
    FeatureBounds,
    MaxCost,
    all_of,
    apply_constraints,
    select_batch,
)
from ml_platform.config import PlatformConfig


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e"],
            "f00": [0.0, 4.0, 0.5, -5.0, 1.0],
            "f01": [0.0, 0.0, 0.2, 0.0, 0.1],
            "cost": [1.0, 1.0, 9.0, 9.0, 2.0],
        }
    )


def test_individual_constraints(frame: pd.DataFrame) -> None:
    assert FeatureBounds(-3, 3)(frame).tolist() == [True, False, True, False, True]
    assert MaxCost(5.0)(frame).tolist() == [True, True, False, False, True]
    assert ExcludeIds(["e"])(frame).tolist() == [True, True, True, True, False]
    for c in (FeatureBounds(-3, 3), MaxCost(5.0), ExcludeIds([])):
        assert isinstance(c, Constraint)
        assert c.name and c.reason


def test_apply_constraints_counts(frame: pd.DataFrame) -> None:
    report = apply_constraints(frame, [FeatureBounds(-3, 3), MaxCost(5.0), ExcludeIds(["e"])])
    assert report.kept["candidate_id"].tolist() == ["a"]
    assert report.n_input == 5 and report.n_kept == 1
    assert report.rejected_by == {"feature_bounds": 2, "max_cost": 2, "exclude_observed": 1}
    # b: bounds only; c: cost only; d: both; e: excluded only
    assert report.rejected_only_by == {"feature_bounds": 1, "max_cost": 1, "exclude_observed": 1}
    s = report.summary()
    assert s["n_rejected"] == 4 and "reasons" in s


def test_all_of_is_a_constraint(frame: pd.DataFrame) -> None:
    combo = all_of(FeatureBounds(-3, 3), MaxCost(5.0))
    np.testing.assert_array_equal(combo(frame), [True, False, False, False, True])
    assert apply_constraints(frame, [combo]).n_kept == 2


def test_duplicate_names_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unique"):
        apply_constraints(frame, [MaxCost(1.0), MaxCost(2.0)])


def test_select_batch_uses_config_constraints(
    pool: pd.DataFrame, session_cfg: PlatformConfig
) -> None:
    sub = pool.head(300).reset_index(drop=True)
    preds = pd.DataFrame(
        {
            "candidate_id": sub["candidate_id"],
            "pred_mean": np.linspace(0, 1, len(sub)),
            "pred_std": np.full(len(sub), 0.1),
        }
    )
    cap = float(sub["cost"].median())
    cfg = session_cfg.with_overrides(
        **{"active_learning.max_cost": cap, "active_learning.feature_bounds": (-2.0, 2.0)}
    )
    res = select_batch(sub, preds, set(), cfg, round_id=1, batch_size=10)
    fcols = [c for c in res.selected.columns if c[:1] == "f" and c[1:].isdigit()]
    assert (res.selected["cost"] <= cap).all()
    assert res.selected[fcols].abs().to_numpy().max() <= 2.0
    rej = res.summary["constraints"]["rejected_by"]
    assert rej["max_cost"] > 0 and rej["feature_bounds"] > 0
