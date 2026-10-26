from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_platform.config import PlatformConfig
from ml_platform.data import (
    SyntheticOracle,
    content_hash,
    generate_candidate_pool,
    initial_observations,
    make_oracle,
    measure_candidates,
    records_to_frame,
)
from ml_platform.hashing import hash_dataframe


def test_pool_is_deterministic_and_bounded(session_cfg: PlatformConfig, pool: pd.DataFrame) -> None:
    again = generate_candidate_pool(session_cfg)
    pd.testing.assert_frame_equal(pool, again)
    assert list(pool.columns[:3]) == ["candidate_id", "f00", "f01"]
    assert pool.columns[-1] == "cost"
    assert len(pool) == session_cfg.data.pool_size
    X = pool.filter(regex=r"^f\d+$").to_numpy()
    assert X.shape[1] == 32
    assert X.min() >= -3.0 and X.max() <= 3.0
    assert (pool["cost"] > 0).all()
    assert pool["candidate_id"].is_unique


def test_pool_depends_on_seed(session_cfg: PlatformConfig, pool: pd.DataFrame) -> None:
    other = generate_candidate_pool(session_cfg.with_overrides(**{"data.pool_seed": 1}))
    assert hash_dataframe(other) != hash_dataframe(pool)


def test_oracle_deterministic_and_heteroscedastic(pool: pd.DataFrame) -> None:
    X = pool.filter(regex=r"^f\d+$").to_numpy()
    a, b = SyntheticOracle(0), SyntheticOracle(0)
    np.testing.assert_array_equal(a.mean(X), b.mean(X))
    assert not np.allclose(a.mean(X), SyntheticOracle(1).mean(X))
    std = a.noise_std(X)
    assert (std > 0).all() and std.max() / std.min() > 2.0  # genuinely heteroscedastic
    y1, s1 = a.measure(X[:50], np.random.default_rng(3))
    y2, s2 = a.measure(X[:50], np.random.default_rng(3))
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(s1, s2)
    assert (a.cost(X) > 0).all()
    # there is an optimum region above the typical response
    assert a.mean(a.optimum_hint())[0] > np.quantile(a.mean(X), 0.99)


def test_oracle_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError):
        SyntheticOracle(0).mean(np.zeros((2, 5)))


def test_initial_observations_deterministic(pool: pd.DataFrame, oracle: SyntheticOracle) -> None:
    r1 = initial_observations(pool, oracle, 100, seed=0)
    r2 = initial_observations(pool, oracle, 100, seed=0)
    r3 = initial_observations(pool, oracle, 100, seed=1)
    f1, f2, f3 = (records_to_frame(r) for r in (r1, r2, r3))
    assert content_hash(f1) == content_hash(f2)
    assert content_hash(f1) != content_hash(f3)
    assert len(r1) == 100 and all(r.status == "measured" and r.round_id == 0 for r in r1)
    assert set(f1["experiment_id"]).issubset(set(pool["candidate_id"]))


def test_measure_candidates_reproducible_per_round(
    pool: pd.DataFrame, oracle: SyntheticOracle
) -> None:
    rows = pool.iloc[:20]
    a = measure_candidates(rows, oracle, round_id=2, seed=0)
    b = measure_candidates(rows, oracle, round_id=2, seed=0)
    c = measure_candidates(rows, oracle, round_id=3, seed=0)
    assert [r.response for r in a] == [r.response for r in b]
    assert [r.response for r in a] != [r.response for r in c]


def test_make_oracle_uses_pool_seed() -> None:
    cfg = PlatformConfig.for_tests().with_overrides(**{"data.pool_seed": 7})
    assert make_oracle(cfg).seed == 7


def test_hash_split_is_stable_across_rounds(obs_frame: pd.DataFrame) -> None:
    from ml_platform.data import train_val_split

    small = obs_frame.iloc[:150]
    tr_s, va_s = train_val_split(small, 0.2, seed=0)
    tr_b, va_b = train_val_split(obs_frame, 0.2, seed=0)
    # adding rows never moves an existing experiment between train and val
    assert set(va_s["experiment_id"]) <= set(va_b["experiment_id"])
    assert set(tr_s["experiment_id"]) <= set(tr_b["experiment_id"])
    assert not set(tr_b["experiment_id"]) & set(va_b["experiment_id"])
    assert 0.1 < len(va_b) / len(obs_frame) < 0.3
    # order independent and deterministic
    _, va2 = train_val_split(obs_frame.sample(frac=1.0, random_state=3), 0.2, seed=0)
    pd.testing.assert_frame_equal(va2, va_b)
    assert set(train_val_split(obs_frame, 0.2, seed=1)[1]["experiment_id"]) != set(
        va_b["experiment_id"]
    )


def test_other_split_strategies(obs_frame: pd.DataFrame) -> None:
    from ml_platform.data import train_val_split

    tr, va = train_val_split(obs_frame, 0.2, seed=0, strategy="random")
    assert len(va) == 60 and len(tr) == 240
    two_rounds = obs_frame.assign(round_id=[0] * 200 + [1] * 100)
    tr, va = train_val_split(two_rounds, 0.2, seed=0, strategy="newest_round")
    assert (va["round_id"] == 1).all() and (tr["round_id"] == 0).all()
    with pytest.raises(ValueError):
        train_val_split(obs_frame, 1.5, seed=0)
