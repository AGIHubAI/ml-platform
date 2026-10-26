"""Minimal end-to-end closed loop through Dagster (handoff §15).

Two rounds of the real asset graph — Ray Train DDP (2 workers), MLflow
(SQLite in tmp), gated registry, Ray actor batch inference, active learning,
ExperimentSimulator actor, write-once RoundStore — materialized with
``closed_loop_round_job`` against a temporary Dagster instance, on the tiny
test configuration.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import ray
from dagster import AssetKey, instance_for_test

from ml_platform.config import PlatformConfig
from ml_platform.data import ImmutableRoundError, RoundStore
from ml_platform.orchestration import pipeline as P
from ml_platform.orchestration.definitions import build_definitions
from ml_platform.orchestration.jobs import run_bootstrap, run_round
from ml_platform.orchestration.partitions import ROUNDS_PARTITION_NAME
from ml_platform.ray_runtime.cluster import ensure_ray, shutdown_ray
from ml_platform.tracking import ModelRegistry, Tracker

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def ray_cluster() -> Iterator[None]:
    already = ray.is_initialized()
    ensure_ray(num_cpus=4, include_dashboard=False)
    yield
    if not already:
        shutdown_ray()


def test_two_round_closed_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ray_cluster: None
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = PlatformConfig.for_tests(
        tmp_path,
        **{
            "training.epochs": 3,
            "evaluation.bootstrap_samples": 100,
            "evaluation.mc_samples": 5,
            "data.batch_size_per_round": 20,
        },
    )
    cfg_path = P.write_config(cfg, tmp_path / "config.yaml")
    defs = build_definitions(config_path=str(cfg_path), num_workers=2, n_inference_actors=1)

    (tmp_path / "dagster_home").mkdir()
    with instance_for_test(temp_dir=str(tmp_path / "dagster_home")) as instance:
        assert run_bootstrap(defs, instance).success
        results = [run_round(defs, instance, r) for r in (0, 1)]
        assert all(res.success for res in results)
        materialized = instance.get_materialized_partitions(AssetKey("new_experimental_results"))
        assert materialized == {"round_000", "round_001"}
        assert "round_002" in instance.get_dynamic_partitions(ROUNDS_PARTITION_NAME)

    # ---- immutable, hash-chained rounds
    store = RoundStore(cfg.paths.data_dir)
    assert store.list_rounds() == [0, 1, 2]
    assert store.verify_chain()
    for r in (1, 2):
        m = store.read_manifest(r)
        assert m.parent_hashes == [store.read_manifest(r - 1).manifest_hash]
        assert m.n_records == 20
        assert m.provenance["source"] == "active_learning"
    with pytest.raises(ImmutableRoundError):
        store.write_round(1, store.read_round(1).assign(response=0.0))
    assert len(store.observed_ids()) == cfg.data.initial_observations + 40  # no duplicates

    # ---- lineage values produced by the assets
    trained = [res.output_for_node("trained_model") for res in results]
    registered = [res.output_for_node("registered_model") for res in results]
    selections = [res.output_for_node("selected_experiments") for res in results]
    new_rounds = [res.output_for_node("new_experimental_results") for res in results]
    assert [t.dataset_hash for t in trained] == [store.dataset_hash(0), store.dataset_hash(1)]
    assert all(t.world_size == 2 and len(t.worker_pids) == 2 for t in trained)

    # ---- selection manifests (versioned by selection_hash, model role recorded)
    for sel, new in zip(selections, new_rounds, strict=True):
        manifest = json.loads((sel.path / "selection.json").read_text())
        assert manifest["selection_hash"] == sel.selection_hash
        assert manifest["metadata"]["model_role"] in (
            P.MODEL_ROLE_PRODUCTION,
            P.MODEL_ROLE_RESEARCH,
        )
        assert set(store.read_round(new.round_id)["experiment_id"]) == set(sel.candidate_ids)
        assert new.n_physical_measurements == 20 and not new.noop

    # ---- re-running the experiment step is an idempotent no-op (no new measurements)
    again = P.run_experiments(cfg, store, P.RayCompute(), selections[1])
    assert again.noop and again.n_physical_measurements == 0
    assert again.round.manifest_hash == new_rounds[1].round.manifest_hash

    # ---- MLflow: runs with dataset id, checkpoints and evaluation artifacts
    tracker = Tracker(cfg.tracking)
    for t in trained:
        run = tracker.get_run(t.run_id)
        assert run.data.tags["dataset_id"] == t.dataset_hash
        assert run.data.tags["dagster_partition"] in ("round_000", "round_001")
        artifacts = {a.path for a in tracker.client.list_artifacts(t.run_id)}
        assert {"checkpoint", "evaluation"} <= artifacts
        assert "eval_rmse" in run.data.metrics

    # ---- registry lifecycle consistent with the gates
    registry = ModelRegistry(tracker=tracker)
    versions = {v["version"]: v for v in registry.list_versions()}
    assert len(versions) == 2
    for reg in registered:
        tags = registry.version_tags(reg.version)
        assert tags["gate_passed"] == str(reg.gate_passed).lower()
        if not reg.gate_passed:
            assert tags["lifecycle"] == "candidate"
    prod = [v for v in versions.values() if "production" in v["aliases"]]
    assert len(prod) <= 1
    assert all(v["gate_passed"] == "true" for v in prod)
    for sel, reg in zip(selections, registered, strict=True):
        if reg.production_version is None:
            assert sel.model_role == P.MODEL_ROLE_RESEARCH and sel.model_version == reg.version
        else:
            assert sel.model_role == P.MODEL_ROLE_PRODUCTION

    # ---- reports written per round
    for t in trained:
        assert (cfg.paths.reports_dir / f"round_{t.round_id:03d}" / t.run_id / "report.md").exists()
    preds = pd.read_parquet(results[1].output_for_node("candidate_predictions").path)
    assert not set(preds["candidate_id"]) & store.observed_ids(1)
