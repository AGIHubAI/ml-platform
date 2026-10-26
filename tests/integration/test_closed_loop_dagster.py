"""`scripts/run_closed_loop.py` end to end on the tiny config (tmp data/MLflow/Dagster).

Exercises the script path (bootstrap detection, per-round Dagster runs against
a persistent instance, the summary table) and that a second invocation
*continues* the campaign from the latest round instead of redoing it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from dagster import DagsterInstance

from ml_platform.config import REPO_ROOT, PlatformConfig
from ml_platform.data import RoundStore
from ml_platform.orchestration import pipeline as P
from ml_platform.tracking import ModelRegistry

pytestmark = pytest.mark.integration


def _load_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "run_closed_loop", REPO_ROOT / "scripts" / "run_closed_loop.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_closed_loop"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_runs_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("DAGSTER_HOME", str(tmp_path / "dagster_home"))
    cfg = PlatformConfig.for_tests(
        tmp_path, **{"training.epochs": 2, "evaluation.bootstrap_samples": 100}
    )
    cfg_path = P.write_config(cfg, tmp_path / "config.yaml")
    script = _load_script()

    assert script.main(["--rounds", "2", "--config", str(cfg_path), "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "closed-loop summary" in out and "round_001" in out
    store = RoundStore(cfg.paths.data_dir)
    assert store.list_rounds() == [0, 1, 2]

    # second invocation continues at round_002 (no bootstrap, no rework)
    assert script.main(["--rounds", "1", "--config", str(cfg_path), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "starting at round_002" in out and "bootstrap:" not in out
    assert store.list_rounds() == [0, 1, 2, 3]
    assert store.verify_chain()

    with DagsterInstance.get() as instance:
        runs = instance.get_runs()
        assert {r.job_name for r in runs} == {"bootstrap_job", "closed_loop_round_job"}
        assert sum(r.job_name == "closed_loop_round_job" for r in runs) == 3
    assert len(ModelRegistry(cfg).list_versions()) == 3
