"""Checkpoint at epoch 3 -> simulated worker failure -> Ray Train restores -> finish."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import ray

from ml_platform.config import PlatformConfig
from ml_platform.data import (
    generate_candidate_pool,
    initial_observations,
    make_oracle,
    records_to_frame,
)
from ml_platform.inference import Predictor
from ml_platform.ray_runtime.cluster import ensure_ray, shutdown_ray
from ml_platform.ray_runtime.tasks import start_experiment_simulator
from ml_platform.training.distributed import distributed_info, train_distributed

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ray_cluster() -> Iterator[None]:
    already = ray.is_initialized()
    ensure_ray(num_cpus=4, include_dashboard=False)
    yield
    if not already:
        shutdown_ray()


def test_worker_failure_resumes_from_checkpoint(ray_cluster: None, tmp_path: Path) -> None:
    cfg = PlatformConfig.for_tests(tmp_path)
    pool = generate_candidate_pool(cfg)
    frame = records_to_frame(initial_observations(pool, make_oracle(cfg), 300, seed=0))

    result = train_distributed(
        cfg, train_frame=frame, num_workers=2, fail_at_epoch=3, max_failures=1
    )
    info = distributed_info(result)
    assert info["failures_recovered"] == 1
    assert info["restored_from_epoch"] == 3
    assert result.resumed_from is not None and result.resumed_from.endswith("epoch_0003")
    assert result.epochs_completed == cfg.training.epochs
    # history is continuous: epochs 1..3 restored from the checkpoint, 4..N re-run
    assert [int(h["epoch"]) for h in result.history] == list(range(1, cfg.training.epochs + 1))
    assert info["params_in_sync"]
    Predictor.from_checkpoint(result.checkpoint_path)

    # Recovery is exact: every rank restored its own RNG stream from the checkpoint,
    # so the recovered run ends with the same parameters as an uninterrupted one.
    straight = train_distributed(cfg, train_frame=frame, num_workers=2)
    s_info = distributed_info(straight)
    assert s_info["failures_recovered"] == 0
    assert {w["param_digest"] for w in info["workers"]} == {
        w["param_digest"] for w in s_info["workers"]
    }
    assert result.metrics == straight.metrics


def test_failure_without_retries_surfaces(ray_cluster: None, tmp_path: Path) -> None:
    cfg = PlatformConfig.for_tests(tmp_path).with_overrides(**{"training.epochs": 3})
    pool = generate_candidate_pool(cfg)
    frame = records_to_frame(initial_observations(pool, make_oracle(cfg), 200, seed=0))
    with pytest.raises(Exception, match=r"(?i)simulated worker failure|TrainingFailed"):
        train_distributed(cfg, train_frame=frame, num_workers=2, fail_at_epoch=2, max_failures=0)


def test_experiment_retry_returns_recorded_measurement(ray_cluster: None, tmp_path: Path) -> None:
    cfg = PlatformConfig.for_tests(tmp_path, **{"data.pool_size": 300})
    pool = generate_candidate_pool(cfg)
    journal = tmp_path / "lab_journal.jsonl"
    sim = start_experiment_simulator(cfg, pool, journal_path=journal)
    ids = pool["candidate_id"].iloc[:8].tolist()
    first: pd.DataFrame = ray.get(sim.measure.remote(1, ids))
    ray.kill(sim)  # the "lab" process dies; a new one reloads the journal
    sim2 = start_experiment_simulator(cfg, pool, journal_path=journal)
    again: pd.DataFrame = ray.get(sim2.measure.remote(1, ids))
    pd.testing.assert_frame_equal(first, again)
    assert ray.get(sim2.stats.remote())["n_physical_measurements"] == 0


def test_restartable_actors_take_no_object_store_constructor_args(
    ray_cluster: None, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """Simulator + predictor actors are rebuilt from durable / per-call inputs after a crash."""
    import time

    from ml_platform.inference.batch import predict_pool, predict_pool_local
    from ml_platform.training import Trainer

    cfg = PlatformConfig.for_tests(tmp_path, **{"data.pool_size": 300, "training.epochs": 1})
    pool = generate_candidate_pool(cfg)
    journal = tmp_path / "lab_journal.jsonl"
    sim = start_experiment_simulator(cfg, pool, journal_path=journal)
    ids = pool["candidate_id"].iloc[:4].tolist()
    first: pd.DataFrame = ray.get(sim.measure.remote(1, ids))
    old_pid = ray.get(sim.pid.remote())
    ray.kill(sim, no_restart=False)  # crash; Ray re-runs __init__ with the same (small) args
    deadline = time.time() + 60
    stats: dict = {"pid": old_pid}
    while stats["pid"] == old_pid:  # ray.kill is asynchronous: wait for the new process
        assert time.time() < deadline, "actor was not restarted"
        try:
            stats = ray.get(sim.stats.remote())
        except ray.exceptions.RayActorError:
            time.sleep(0.2)
    assert stats["pool_path"].endswith(".parquet") and stats["n_logged"] == len(ids)
    again: pd.DataFrame = ray.get(sim.measure.remote(1, ids))
    pd.testing.assert_frame_equal(first, again)
    assert ray.get(sim.stats.remote())["n_physical_measurements"] == 0
    ray.kill(sim)

    frame = records_to_frame(initial_observations(pool, make_oracle(cfg), 120, seed=0))
    from ml_platform.data import ArrayDataset

    ds = ArrayDataset.from_frame(frame, dataset_hash="t")
    ckpt = Trainer(cfg).train(ds, None, checkpoint_dir=tmp_path / "ckpt").checkpoint_path
    out = predict_pool(ckpt, pool, n_actors=2, shard_size=100, mc_samples=3, seed=1)
    ref = predict_pool_local(ckpt, pool, shard_size=100, mc_samples=3, seed=1)
    pd.testing.assert_frame_equal(out, ref)
    assert "has constructor arguments in the object store" not in capfd.readouterr().err
