"""Ray actor-pool batch inference equals the in-process reference."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from ml_platform.inference.batch import predict_pool, predict_pool_local, shard_bounds
from ml_platform.training import TrainResult

pytestmark = pytest.mark.smoke

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module")
def ray_local() -> Iterator[None]:
    from ml_platform.ray_runtime.cluster import ensure_ray

    started = not ray.is_initialized()
    ensure_ray(num_cpus=2, include_dashboard=False)
    yield
    if started:
        ray.shutdown()


def test_shard_bounds() -> None:
    assert shard_bounds(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert shard_bounds(0, 3) == []


def test_predict_pool_matches_local(
    ray_local: None, trained: tuple[TrainResult, pd.DataFrame, pd.DataFrame], pool: pd.DataFrame
) -> None:
    result, _, _ = trained
    tiny = pool.head(700).reset_index(drop=True)
    kw = {"shard_size": 128, "mc_samples": 8, "seed": 5}
    local = predict_pool_local(result.checkpoint_path, tiny, **kw)
    out = predict_pool(result.checkpoint_path, tiny, n_actors=2, return_stats=True, **kw)
    assert isinstance(out, tuple)
    remote, stats = out
    assert list(remote.columns) == ["candidate_id", "pred_mean", "pred_std"]
    assert remote["candidate_id"].tolist() == tiny["candidate_id"].tolist()
    np.testing.assert_allclose(remote["pred_mean"], local["pred_mean"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(remote["pred_std"], local["pred_std"], rtol=1e-6, atol=1e-9)
    assert (remote["pred_std"] > 0).all()
    assert stats["n_shards"] == 6 and stats["n_actors"] == 2
    # both actors did work and each loaded the model exactly once
    assert all(a["shards_served"] > 0 for a in stats["actors"])
    assert sum(a["rows_served"] for a in stats["actors"]) == len(tiny)
    # different seed -> different MC masks -> different std
    other = predict_pool_local(result.checkpoint_path, tiny, shard_size=128, mc_samples=8, seed=6)
    assert not np.allclose(other["pred_std"], local["pred_std"])
