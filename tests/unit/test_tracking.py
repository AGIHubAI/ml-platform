"""MLflow adapter + registry lifecycle against a throwaway SQLite store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_platform.config import REPO_ROOT, PlatformConfig
from ml_platform.evaluation import GateDecision
from ml_platform.inference import Predictor
from ml_platform.tracking import ModelRegistry, PromotionError, Tracker, resolve_tracking_uri
from ml_platform.training import TrainResult


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PlatformConfig:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return PlatformConfig.for_tests(tmp_path)


def test_uri_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri("sqlite:///mlflow.db") == f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
    assert resolve_tracking_uri(f"sqlite:///{tmp_path}/x.db") == f"sqlite:///{tmp_path}/x.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert resolve_tracking_uri("sqlite:///mlflow.db") == "http://mlflow:5000"


def test_tracker_logs_params_metrics_tags(
    cfg: PlatformConfig, tmp_path: Path, trained: tuple[TrainResult, object, object]
) -> None:
    result, train_frame, _ = trained
    tracker = Tracker(cfg.tracking)
    with tracker.start_run("unit", tags={"round_id": 0}) as run_id:
        tracker.log_params({"a": 1, "nested": {"b": "x"}})
        tracker.log_metrics({"m": 0.5, "bad": float("nan")}, step=1)
        tracker.set_tags({"t": "v"})
        tracker.log_train_result(result, cfg=cfg, dataset_frame=train_frame)  # type: ignore[arg-type]
    run = tracker.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["a"] == "1" and run.data.params["nested.b"] == "x"
    assert run.data.params["dataset_id"] == "test-dataset"
    assert run.data.metrics["m"] == 0.5 and "bad" not in run.data.metrics
    assert "val_rmse" in run.data.metrics and "duration_s" in run.data.metrics
    tags = run.data.tags
    assert tags["t"] == "v" and tags["round_id"] == "0"
    for key in ("dataset_hash", "config_hash", "checkpoint_hash", "torch_version", "ray_version"):
        assert tags[key], key
    assert tags["config_hash"] == cfg.config_hash()
    assert run.inputs.dataset_inputs, "dataset should be logged as a run input"
    history = tracker.client.get_metric_history(run_id, "val_rmse")
    assert len(history) == result.epochs_completed
    # artifacts live next to the sqlite db, not in CWD
    art = tracker.run_artifact_dir(run_id)
    assert art is not None and str(art).startswith(str(tmp_path / "mlartifacts"))
    assert (art / "checkpoint" / "model.pt").exists()
    assert (art / "config.json").exists() and (art / "config_scientific.json").exists()


def test_registry_lifecycle(
    cfg: PlatformConfig, trained: tuple[TrainResult, object, object]
) -> None:
    result = trained[0]
    tracker = Tracker(cfg.tracking)
    registry = ModelRegistry(tracker=tracker)
    assert registry.production_version() is None

    def new_version() -> str:
        with tracker.start_run("reg") as run_id:
            pass
        return registry.register(run_id, result.checkpoint_path)

    v1 = new_version()
    assert registry.stage(v1) == "candidate"
    # never promote without a (passing) gate
    with pytest.raises(PromotionError):
        registry.set_stage(v1, "production")
    with pytest.raises(PromotionError):
        registry.promote_if_passed(v1, None)

    failing = GateDecision(passed=False, reasons=["FAIL: rmse too high"])
    assert registry.promote_if_passed(v1, failing) == "candidate"
    assert registry.stage(v1) == "candidate"
    assert "rmse too high" in registry.version_tags(v1)["gate_reasons"]
    assert registry.production_version() is None

    v2 = new_version()
    assert (
        registry.promote_if_passed(v2, GateDecision(passed=True, reasons=["PASS"])) == "production"
    )
    prod = registry.production_version()
    assert prod is not None and prod.version == v2

    v3 = new_version()
    assert registry.promote_if_passed(v3, {"passed": True, "reasons": ["PASS"]}) == "production"
    assert registry.stage(v2) == "archived" and registry.stage(v3) == "production"
    assert registry.production_version().version == v3  # type: ignore[union-attr]

    # aliases are reported per version (MLflow's search API alone returns none on SQLite)
    listed = {d["version"]: d for d in registry.list_versions()}
    assert listed[v3]["aliases"] == ["candidate", "production", "validated"]
    assert listed[v1]["aliases"] == [] and listed[v2]["aliases"] == []
    assert [listed[v]["lifecycle"] for v in (v1, v2, v3)] == ["candidate", "archived", "production"]
    assert registry.aliases()["production"] == v3

    predictor = registry.load_production_predictor()
    ref = Predictor.from_checkpoint(result.checkpoint_path)
    X = np.zeros((3, ref.normalizer.n_features))
    np.testing.assert_allclose(predictor.predict(X), ref.predict(X))
    assert predictor.model_info()["model_version"] == v3
