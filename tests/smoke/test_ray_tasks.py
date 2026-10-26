"""Fast Ray smoke test: tasks, object refs, backpressure, and the stateful actor."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import ray

from ml_platform.config import PlatformConfig
from ml_platform.data import generate_candidate_pool
from ml_platform.evaluation import bootstrap_ci
from ml_platform.ray_runtime.cluster import ensure_ray, shutdown_ray
from ml_platform.ray_runtime.resources import select_resources
from ml_platform.ray_runtime.tasks import (
    bounded_map,
    parallel_bootstrap_ci,
    square_sum_task,
    start_experiment_simulator,
)

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def ray_cluster() -> Iterator[dict[str, float]]:
    already = ray.is_initialized()
    resources = ensure_ray(num_cpus=2, include_dashboard=False)
    yield resources
    if not already:
        shutdown_ray()


def test_task_and_object_ref(ray_cluster: dict[str, float]) -> None:
    x = np.arange(10.0)
    ref = ray.put(x)
    assert ray.get(square_sum_task.remote(ref)) == pytest.approx(float((x**2).sum()))
    out = bounded_map(square_sum_task, [np.ones(k) for k in range(1, 7)], max_in_flight=2)
    assert out == [float(k) for k in range(1, 7)]


def test_parallel_bootstrap_matches_serial_distribution(ray_cluster: dict[str, float]) -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    yhat = y + rng.normal(scale=0.3, size=400)
    point, lo, hi = parallel_bootstrap_ci(y, yhat, n_resamples=400, n_tasks=2, seed=1)
    again = parallel_bootstrap_ci(y, yhat, n_resamples=400, n_tasks=2, seed=1)
    assert (point, lo, hi) == again  # deterministic => safe to retry
    s_point, s_lo, s_hi = bootstrap_ci(
        lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))), y, yhat, n=400
    )
    assert point == pytest.approx(s_point)
    assert lo < point < hi
    assert abs(lo - s_lo) < 0.03 and abs(hi - s_hi) < 0.03


def test_experiment_simulator_is_idempotent(ray_cluster: dict[str, float]) -> None:
    cfg = PlatformConfig.for_tests(data__pool_size=200)
    pool = generate_candidate_pool(cfg)
    sim = start_experiment_simulator(cfg, pool)
    ids = pool["candidate_id"].iloc[:5].tolist()
    first: pd.DataFrame = ray.get(sim.measure.remote(1, ids))
    retry: pd.DataFrame = ray.get(sim.measure.remote(1, ids))  # e.g. an orchestration retry
    pd.testing.assert_frame_equal(first, retry)
    stats = ray.get(sim.stats.remote())
    assert stats["n_physical_measurements"] == 5 and stats["n_cache_hits"] == 5
    assert first["status"].eq("measured").all() and first["response"].notna().all()
    ray.kill(sim)


def test_select_resources_is_cpu_safe(ray_cluster: dict[str, float]) -> None:
    cfg = PlatformConfig.for_tests().with_overrides(**{"distributed.use_gpu": "auto"})
    res = select_resources(cfg, num_workers=8)
    assert res.num_workers <= int(ray_cluster["CPU"])
    if not __import__("torch").cuda.is_available():
        assert res.use_gpu is False and res.remote_options()["num_gpus"] == 0
