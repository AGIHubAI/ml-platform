from __future__ import annotations

import os

import pandas as pd
import pytest

from ml_platform.data import (
    ImmutableRoundError,
    RoundSequenceError,
    RoundStore,
    SyntheticOracle,
    initial_observations,
    measure_candidates,
)


def _round0(pool: pd.DataFrame, oracle: SyntheticOracle, n: int = 40):
    return initial_observations(pool, oracle, n, seed=0)


def _round_n(pool: pd.DataFrame, oracle: SyntheticOracle, round_id: int, start: int, n: int = 10):
    # candidates from the tail of the pool (never picked by the 40-row round 0 in these tests
    # because we exclude already observed ids)
    return measure_candidates(pool.iloc[start : start + n], oracle, round_id=round_id, seed=0)


def test_write_read_roundtrip(store: RoundStore, pool, oracle) -> None:
    recs = _round0(pool, oracle)
    m = store.write_round(0, recs, provenance={"source": "test"})
    assert m.round_id == 0 and m.n_records == 40 and m.parent_hashes == []
    df = store.read_round(0)
    assert len(df) == 40
    assert store.list_rounds() == [0] and store.latest_round_id() == 0
    back = store.read_records(0)
    assert {r.experiment_id for r in back} == {r.experiment_id for r in recs}
    assert store.read_manifest(0).provenance == {"source": "test"}


def test_idempotent_rewrite(store: RoundStore, pool, oracle) -> None:
    m1 = store.write_round(0, _round0(pool, oracle))
    # regenerated records have new created_at timestamps and shuffled order: same content
    again = list(reversed(_round0(pool, oracle)))
    m2 = store.write_round(0, again, provenance={"retry": True})
    assert m2 == m1
    assert store.list_rounds() == [0]


def test_different_content_raises(store: RoundStore, pool, oracle) -> None:
    store.write_round(0, _round0(pool, oracle))
    with pytest.raises(ImmutableRoundError):
        store.write_round(0, _round0(pool, oracle, n=41))
    other = initial_observations(pool, oracle, 40, seed=1)
    with pytest.raises(ImmutableRoundError):
        store.write_round(0, other)


def test_files_are_read_only(store: RoundStore, pool, oracle) -> None:
    store.write_round(0, _round0(pool, oracle))
    for f in store.round_dir(0).iterdir():
        assert not os.access(f, os.W_OK)


def test_round_sequence_enforced(store: RoundStore, pool, oracle) -> None:
    with pytest.raises(RoundSequenceError):
        store.write_round(1, _round_n(pool, oracle, 1, 1000))
    recs = _round0(pool, oracle)
    store.write_round(0, recs)
    # duplicate experiment ids across rounds are rejected
    dup = measure_candidates(
        pool[pool["candidate_id"] == recs[0].experiment_id], oracle, round_id=1, seed=0
    )
    with pytest.raises(RoundSequenceError):
        store.write_round(1, dup)
    with pytest.raises(ValueError):
        store.write_round(1, _round_n(pool, oracle, 2, 1000))  # wrong round_id in records


def test_union_hash_chain_and_training_frame(store: RoundStore, pool, oracle) -> None:
    observed = {r.experiment_id for r in _round0(pool, oracle)}
    fresh = pool[~pool["candidate_id"].isin(observed)]
    m0 = store.write_round(0, _round0(pool, oracle))
    h0 = store.dataset_hash()
    m1 = store.write_round(1, measure_candidates(fresh.iloc[:10], oracle, round_id=1, seed=0))
    m2 = store.write_round(2, measure_candidates(fresh.iloc[10:25], oracle, round_id=2, seed=0))
    assert m1.parent_hashes == [m0.manifest_hash]
    assert m2.parent_hashes == [m1.manifest_hash]
    assert store.verify_chain()

    tf = store.training_frame()
    assert len(tf) == 40 + 10 + 15
    assert tf["experiment_id"].is_unique
    assert list(tf["round_id"].unique()) == [0, 1, 2]
    assert len(store.training_frame(up_to_round=1)) == 50
    assert store.observed_ids(up_to_round=0) == observed

    # point-in-time reproducibility of dataset ids
    assert store.dataset_hash(up_to_round=0) == h0
    assert len({store.dataset_hash(r) for r in (0, 1, 2)}) == 3
    assert store.dataset_hash() == m2.manifest_hash
    with pytest.raises(FileNotFoundError):
        store.training_frame(up_to_round=7)


def test_training_frame_measured_only(store: RoundStore, pool, oracle) -> None:
    recs = _round0(pool, oracle, n=10)
    failed = recs[0].model_copy(update={"status": "failed", "response": None})
    store.write_round(0, [failed, *recs[1:]])
    assert len(store.training_frame()) == 9
    assert len(store.all_observations()) == 10


def test_dataset_hash_same_content_same_hash(tmp_path, pool, oracle) -> None:
    a, b = RoundStore(tmp_path / "a"), RoundStore(tmp_path / "b")
    a.write_round(0, _round0(pool, oracle))
    b.write_round(0, _round0(pool, oracle))
    assert a.dataset_hash() == b.dataset_hash()


def test_pool_write_once(store: RoundStore, pool) -> None:
    m = store.write_pool(pool, provenance={"seed": 0})
    assert store.has_pool() and m["n_candidates"] == len(pool)
    assert store.write_pool(pool) == m
    pd.testing.assert_frame_equal(store.read_pool(), pool)
    with pytest.raises(ImmutableRoundError):
        store.write_pool(pool.iloc[:-1])
