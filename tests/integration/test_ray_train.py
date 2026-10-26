"""Two-worker Ray Train + DDP run on the tiny test config."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ray

from ml_platform.config import PlatformConfig
from ml_platform.data import (
    RoundStore,
    generate_candidate_pool,
    initial_observations,
    make_oracle,
)
from ml_platform.inference import Predictor
from ml_platform.ray_runtime.cluster import ensure_ray, shutdown_ray
from ml_platform.training.distributed import distributed_info, train_distributed

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ray_cluster() -> Iterator[None]:
    already = ray.is_initialized()
    ensure_ray(num_cpus=4, include_dashboard=False)
    yield
    if not already:
        shutdown_ray()


@pytest.fixture
def cfg_with_round(tmp_path: Path) -> PlatformConfig:
    cfg = PlatformConfig.for_tests(tmp_path)
    store = RoundStore(cfg.paths.data_dir)
    pool = generate_candidate_pool(cfg)
    store.write_pool(pool)
    store.write_round(0, initial_observations(pool, make_oracle(cfg), 300, seed=0))
    return cfg


def test_two_worker_ddp(ray_cluster: None, cfg_with_round: PlatformConfig) -> None:
    cfg = cfg_with_round
    result = train_distributed(cfg, round_id=0, num_workers=2, run_id="test-run")
    info = distributed_info(result)

    assert result.world_size == 2
    assert len(info["workers"]) == 2 and sorted(w["rank"] for w in info["workers"]) == [0, 1]
    assert len(set(info["pids"])) == 2, "expected two distinct worker processes"
    assert info["params_in_sync"], "DDP replicas diverged"
    assert result.epochs_completed == cfg.training.epochs
    assert len(result.history) == cfg.training.epochs
    store = RoundStore(cfg.paths.data_dir)
    assert result.dataset_hash == store.dataset_hash(0)
    # checkpoint persisted by Ray Train under the configured artifacts dir, loadable
    assert str(result.checkpoint_path).startswith(str(cfg.paths.artifacts_dir))
    assert result.checkpoint_hash
    predictor = Predictor.from_checkpoint(result.checkpoint_path)
    frame: pd.DataFrame = store.training_frame(0)
    pred = predictor.predict(frame.iloc[:10])
    assert pred.shape == (10,) and np.isfinite(pred).all()
    assert predictor.model_info()["trained_world_size"] == 2
