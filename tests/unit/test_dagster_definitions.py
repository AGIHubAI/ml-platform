"""The Dagster code location loads and has the intended shape (no execution)."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from dagster import (
    AssetKey,
    DagsterInstance,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    Definitions,
    DynamicPartitionsDefinition,
    build_sensor_context,
)

from ml_platform.config import PlatformConfig
from ml_platform.data import RoundStore
from ml_platform.orchestration import pipeline as P
from ml_platform.orchestration.definitions import build_definitions, defs

CHAIN = [
    "candidate_pool",
    "observed_experiments",
    "training_dataset",
    "trained_model",
    "evaluation_report",
    "registered_model",
    "candidate_predictions",
    "selected_experiments",
    "new_experimental_results",
]


def _parents(d: Definitions, name: str) -> set[str]:
    node = d.resolve_asset_graph().get(AssetKey(name))
    return {k.to_user_string() for k in node.parent_keys}


def test_defs_load() -> None:
    assert isinstance(defs, Definitions)
    defs.resolve_all_job_defs()  # resolves every job and binds resources


FUSI_CHAIN = [
    "fusi_session",
    "fusi_split",
    "fusi_preprocessor",
    "fusi_dataset",
    "fusi_decoder",
    "fusi_evaluation",
    "fusi_registered_decoder",
]


def test_asset_graph_is_the_handoff_chain() -> None:
    keys = {k.to_user_string() for k in defs.resolve_asset_graph().get_all_asset_keys()}
    assert keys == set(CHAIN) | set(FUSI_CHAIN)
    for upstream, downstream in pairwise(CHAIN):
        assert upstream in _parents(defs, downstream), f"{upstream} -> {downstream} missing"
    assert _parents(defs, "candidate_pool") == set()
    assert _parents(defs, "evaluation_report") == {"trained_model"}
    assert {"trained_model", "evaluation_report"} <= _parents(defs, "registered_model")


def test_rounds_are_dynamic_partitions() -> None:
    graph = defs.resolve_asset_graph()
    assert graph.get(AssetKey("candidate_pool")).partitions_def is None
    for name in CHAIN[1:]:
        pdef = graph.get(AssetKey(name)).partitions_def
        assert isinstance(pdef, DynamicPartitionsDefinition)
        assert pdef.name == "rounds"


def test_retry_policies() -> None:
    def policy(name: str):  # type: ignore[no-untyped-def]
        return defs.resolve_assets_def(name).op.retry_policy

    # physical experiment: never retried automatically
    assert policy("new_experimental_results") is None
    for name in ("trained_model", "evaluation_report", "candidate_predictions"):
        rp = policy(name)
        assert rp is not None and rp.max_retries >= 1 and rp.backoff is not None


def test_jobs_sensor_schedule() -> None:
    for job in ("bootstrap_job", "closed_loop_round_job", "retrain_job"):
        assert defs.resolve_job_def(job) is not None
    round_job = defs.resolve_job_def("closed_loop_round_job")
    selected = {k.to_user_string() for k in round_job.asset_layer.executable_asset_keys}
    assert selected == set(CHAIN[1:])
    sensor = defs.resolve_sensor_def("new_round_sensor")
    assert sensor.default_status == DefaultSensorStatus.STOPPED
    schedule = defs.resolve_schedule_def("nightly_retrain_schedule")
    assert schedule.default_status == DefaultScheduleStatus.STOPPED
    assert schedule.job_name == "retrain_job"


def test_sensor_requests_runs_for_new_rounds(tmp_path: Path) -> None:
    cfg = PlatformConfig.for_tests(tmp_path)
    store = RoundStore(cfg.paths.data_dir)
    P.ensure_candidate_pool(cfg, store)
    P.observe_round(cfg, store, 0)
    d = build_definitions(config_path=str(P.write_config(cfg, tmp_path / "cfg.yaml")))
    sensor = d.resolve_sensor_def("new_round_sensor")
    with (
        DagsterInstance.ephemeral() as instance,
        build_sensor_context(instance=instance, resources=d.resources, definitions=d) as ctx,
    ):
        result = sensor(ctx)
        assert [r.partition_key for r in result.run_requests] == ["round_000"]
        assert result.cursor == "0"
        assert result.dynamic_partitions_requests[0].partition_keys == ["round_000"]
        # nothing new since the cursor -> skip
        ctx.update_cursor("0")
        assert sensor(ctx).__class__.__name__ == "SkipReason"


def test_round_zero_needs_no_lab_but_later_rounds_do(tmp_path: Path) -> None:
    cfg = PlatformConfig.for_tests(tmp_path)
    store = RoundStore(cfg.paths.data_dir)
    P.ensure_candidate_pool(cfg, store)
    first = P.observe_round(cfg, store, 0)
    assert first.n_records == cfg.data.initial_observations
    assert P.observe_round(cfg, store, 0) == first  # idempotent
    with pytest.raises(P.RoundNotMeasuredError):
        P.observe_round(cfg, store, 1)
    ds = P.build_training_dataset(cfg, store, 0)
    assert ds.dataset_hash == store.dataset_hash(0) and ds.validation["ok"]


def test_fusi_graph_partitions_checks_and_jobs() -> None:
    graph = defs.resolve_asset_graph()
    for upstream, downstream in pairwise(FUSI_CHAIN):
        assert upstream in _parents(defs, downstream), f"{upstream} -> {downstream} missing"
    pdef = graph.get(AssetKey("fusi_session")).partitions_def
    assert isinstance(pdef, DynamicPartitionsDefinition) and pdef.name == "fusi_sessions"
    assert all(graph.get(AssetKey(n)).partitions_def is None for n in FUSI_CHAIN[1:])
    checks = {(c.asset_key.to_user_string(), c.name) for c in graph.asset_check_keys}
    expected = {
        ("fusi_dataset", n)
        for n in (
            "sessions_disjoint",
            "subjects_disjoint",
            "temporal_gap",
            "no_window_overlap",
            "preprocessing_fit_on_train_only",
            "feature_schema",
        )
    } | {("fusi_evaluation", "test_set_access"), ("fusi_evaluation", "train_test_gap")}
    assert checks == expected
    for spec in defs.resolve_assets_def("fusi_dataset").check_specs:
        assert spec.blocking
    train = defs.resolve_job_def("fusi_train_job")
    selected = {k.to_user_string() for k in train.asset_layer.executable_asset_keys}
    assert selected == set(FUSI_CHAIN[1:])
    assert defs.resolve_job_def("fusi_sessions_job").partitions_def is not None
