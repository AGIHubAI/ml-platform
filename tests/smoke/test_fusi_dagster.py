"""fUSI workload end to end through Dagster on the tiny config (Ray Train DDP + Ray actors).

Run 1: every leakage check passes, the decoder is trained with 2 DDP workers, evaluated
with a real-time latency run and registered. Run 2 forces a failing blocking check
(test-set access budget exhausted) and asserts the registry asset is skipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import ray
from dagster import instance_for_test

from ml_platform.config import PlatformConfig
from ml_platform.orchestration import fusi_pipeline as FP
from ml_platform.orchestration import pipeline as P
from ml_platform.orchestration.definitions import build_definitions
from ml_platform.orchestration.fusi_assets import run_fusi
from ml_platform.ray_runtime.cluster import ensure_ray, shutdown_ray
from ml_platform.tracking.fusi import fusi_registry
from ml_platform.workloads.fusi_realtime.config import FusiConfig, TelemetryConfig

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def ray_cluster() -> Iterator[None]:
    already = ray.is_initialized()
    ensure_ray(num_cpus=4, include_dashboard=False)
    yield
    if not already:
        shutdown_ray()


def test_fusi_dagster_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ray_cluster: None
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = PlatformConfig.for_tests(tmp_path)
    cfg.fusi = FusiConfig.for_tests()
    # 30 s sessions are too small to generalise; the overfitting guard is exercised in
    # tests/unit/test_fusi_splits.py, here run 1 must reach the registry.
    cfg.fusi.split.max_train_test_gap = 1.0
    otel = tmp_path / "otel"
    cfg.fusi.telemetry = TelemetryConfig(exporter="file", file_dir=str(otel))
    defs = build_definitions(config_path=str(P.write_config(cfg, tmp_path / "config.yaml")))
    keys = FP.session_keys(cfg)
    (tmp_path / "dagster_home").mkdir()
    with instance_for_test(temp_dir=str(tmp_path / "dagster_home")) as instance:
        session_runs, train = run_fusi(defs, instance, keys)
        assert all(r.success for r in session_runs)
        assert train.success
        checks = {e.check_name: e.passed for e in train.get_asset_check_evaluations()}
        assert len(checks) == 8 and all(checks.values()), checks

        dec = train.output_for_node("fusi_decoder")
        assert dec.summary["world_size"] == 2 and dec.summary["params_in_sync"]
        ev = train.output_for_node("fusi_evaluation")
        assert ev.latency["n_completed"] > 0 and ev.test_accesses == 1
        report = json.loads((ev.report_path / "evaluation.json").read_text())
        assert report["split_hash"] and "test_frame_acc" in report["accuracy"]
        reg = train.output_for_node("fusi_registered_decoder")
        # whatever the gate said, the registry agrees with it
        assert reg.stage == ("production" if ev.gate_passed else "candidate")
        assert fusi_registry(cfg).stage(reg.version) == reg.stage

        # OpenTelemetry from every layer, one JSON-lines file per process
        spans = [json.loads(line) for f in otel.glob("spans-*.jsonl") for line in f.open()]
        names = {sp["name"] for sp in spans}
        assert {f"dagster.asset.{a}" for a in ("fusi_session", "fusi_decoder")} <= names
        assert {"fusi.train.epoch", "fusi.train.step"} <= names  # Ray Train workers
        assert {"fusi.trial", "fusi.frame", "fusi.decode"} <= names  # real-time actors
        frames = [sp for sp in spans if sp["name"] == "fusi.frame"]
        trials = {sp["context"]["trace_id"] for sp in spans if sp["name"] == "fusi.trial"}
        assert {sp["context"]["trace_id"] for sp in frames} <= trials  # one trace per trial

        # a failing blocking check stops the run before the registry
        cfg.fusi.split.max_test_accesses = 1  # already used once above
        defs2 = build_definitions(config_path=str(P.write_config(cfg, tmp_path / "config2.yaml")))
        _, blocked = run_fusi(defs2, instance, keys)
        checks2 = {e.check_name: e.passed for e in blocked.get_asset_check_evaluations()}
        assert checks2["test_set_access"] is False
        done = {e.step_key for e in blocked.all_events if e.event_type_value == "STEP_SUCCESS"}
        assert "fusi_decoder" in done and "fusi_registered_decoder" not in done
        assert len(fusi_registry(cfg).list_versions()) == 1  # nothing new registered
