"""Acquisition functions, ranking, and deterministic batch selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_platform.active_learning import (
    expected_improvement,
    rank_candidates,
    read_selection,
    select_batch,
    thompson,
    ucb,
    write_selection,
)
from ml_platform.config import PlatformConfig


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"candidate_id": [f"c{i:03d}" for i in range(n)]})


def test_ucb_formula() -> None:
    np.testing.assert_allclose(ucb([1.0, 2.0], [0.5, 0.0], beta=2.0), [2.0, 2.0])


def test_higher_beta_favours_uncertain_candidates() -> None:
    frame = _frame(2)
    mu = np.array([1.0, 0.5])  # c000 exploits, c001 explores
    sigma = np.array([0.1, 1.0])
    low = rank_candidates(frame, mu, sigma, beta=0.1)
    high = rank_candidates(frame, mu, sigma, beta=5.0)
    assert low["candidate_id"].iloc[0] == "c000"
    assert high["candidate_id"].iloc[0] == "c001"


def test_rank_output_and_tie_break() -> None:
    frame = pd.DataFrame({"candidate_id": ["b", "a", "c"]})
    out = rank_candidates(frame, [1.0, 1.0, 2.0], [0.0, 0.0, 0.0], beta=1.0)
    assert list(out.columns) == ["candidate_id", "pred_mean", "pred_std", "score", "rank"]
    assert out["candidate_id"].tolist() == ["c", "a", "b"]  # tie a/b broken by id
    assert out["rank"].tolist() == [1, 2, 3]


def test_ranking_independent_of_row_order() -> None:
    rng = np.random.default_rng(0)
    frame = _frame(50)
    mu, sigma = rng.normal(size=50), rng.uniform(0.1, 1, size=50)
    perm = rng.permutation(50)
    a = rank_candidates(frame, mu, sigma, beta=1.0)
    b = rank_candidates(frame.iloc[perm].reset_index(drop=True), mu[perm], sigma[perm], beta=1.0)
    pd.testing.assert_frame_equal(a, b)


def test_expected_improvement_properties() -> None:
    ei = expected_improvement([0.0, 0.0, 2.0], [1.0, 0.1, 0.0], best=1.0)
    assert (ei >= 0).all()
    assert ei[0] > ei[1]  # more uncertainty -> more EI when below the incumbent
    assert ei[2] == pytest.approx(1.0)  # deterministic improvement


def test_thompson_seeded() -> None:
    a = thompson(np.zeros(10), np.ones(10), seed=3)
    np.testing.assert_array_equal(a, thompson(np.zeros(10), np.ones(10), seed=3))
    assert not np.array_equal(a, thompson(np.zeros(10), np.ones(10), seed=4))


# ----------------------------------------------------------------- select_batch
@pytest.fixture
def small_pool(pool: pd.DataFrame) -> pd.DataFrame:
    return pool.head(400).reset_index(drop=True)


@pytest.fixture
def preds(small_pool: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "candidate_id": small_pool["candidate_id"],
            "pred_mean": rng.normal(size=len(small_pool)),
            "pred_std": rng.uniform(0.05, 0.5, size=len(small_pool)),
        }
    )


def test_select_batch_excludes_observed_and_respects_size(
    small_pool: pd.DataFrame, preds: pd.DataFrame, session_cfg: PlatformConfig
) -> None:
    first = select_batch(small_pool, preds, set(), session_cfg, round_id=1)
    top = set(first.candidate_ids[:10])
    res = select_batch(small_pool, preds, top, session_cfg, round_id=1, batch_size=20)
    assert len(res.selected) == 20
    assert not top & set(res.candidate_ids)
    assert res.summary["constraints"]["rejected_by"]["exclude_observed"] == 10
    assert res.selected["rank"].is_monotonic_increasing
    assert (res.selected["round_id"] == 1).all()
    assert {"f00", "f31", "cost"} <= set(res.selected.columns)


def test_selection_deterministic_and_hash_stable(
    small_pool: pd.DataFrame, preds: pd.DataFrame, session_cfg: PlatformConfig
) -> None:
    a = select_batch(small_pool, preds, {"cand_000001"}, session_cfg, round_id=2, model_version="3")
    shuffled = preds.sample(frac=1.0, random_state=0).reset_index(drop=True)
    b = select_batch(
        small_pool.iloc[::-1].reset_index(drop=True),
        shuffled,
        ["cand_000001"],
        session_cfg,
        round_id=2,
        model_version="3",
    )
    pd.testing.assert_frame_equal(a.selected, b.selected)
    assert a.selection_hash == b.selection_hash
    # any change of provenance changes the artifact version
    c = select_batch(small_pool, preds, {"cand_000001"}, session_cfg, round_id=2, model_version="4")
    d = select_batch(small_pool, preds, {"cand_000001"}, session_cfg, round_id=2, beta=3.0)
    assert len({a.selection_hash, c.selection_hash, d.selection_hash}) == 3


def test_diversity_spreads_batch(
    small_pool: pd.DataFrame, preds: pd.DataFrame, session_cfg: PlatformConfig
) -> None:
    fcols = [c for c in small_pool.columns if c.startswith("f")]
    res = select_batch(
        small_pool, preds, set(), session_cfg, round_id=1, batch_size=10, diversity_radius=7.0
    )
    X = res.selected[fcols].to_numpy()
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert len(res.selected) == 10
    assert res.summary["n_diversity_skipped"] > 0
    assert d.min() >= 7.0
    plain = select_batch(small_pool, preds, set(), session_cfg, round_id=1, batch_size=10)
    assert set(plain.candidate_ids) != set(res.candidate_ids)


def test_write_and_read_selection(
    tmp_path: Path, small_pool: pd.DataFrame, preds: pd.DataFrame, session_cfg: PlatformConfig
) -> None:
    res = select_batch(small_pool, preds, set(), session_cfg, round_id=1, model_version="7")
    out = write_selection(res, tmp_path / "sel")
    frame, manifest = read_selection(out)
    pd.testing.assert_frame_equal(frame, res.selected)
    assert manifest["selection_hash"] == res.selection_hash
    assert manifest["model_version"] == "7"
    assert manifest["round_id"] == 1
    assert manifest["selected_ids"] == res.candidate_ids
    assert "rejected_by" in manifest["summary"]["constraints"]
