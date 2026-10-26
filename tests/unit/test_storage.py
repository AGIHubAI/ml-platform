"""External-drive storage: env interpolation in configs and the unmounted-volume guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_platform.config import PathsConfig, load_config
from ml_platform.storage import StorageNotMountedError, apply_runtime_env, require_mounted


def test_unmounted_volume_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAGSTER_HOME", raising=False)  # `make test` exports one; it would win
    with pytest.raises(StorageNotMountedError, match="not mounted"):
        require_mounted("/Volumes/__ml_no_such_drive__/data")
    assert require_mounted(tmp_path) == tmp_path  # not under /Volumes: allowed
    with pytest.raises(StorageNotMountedError):
        PathsConfig(data_dir=Path("/Volumes/__ml_no_such_drive__/data")).resolved()
    with pytest.raises(StorageNotMountedError):
        apply_runtime_env(
            dagster_home=Path("/Volumes/__ml_no_such_drive__/dh"),
            ray_temp_dir=None,
            ray_spill_dir=None,
            dask_temp_dir=None,
        )
    assert not Path("/Volumes/__ml_no_such_drive__").exists()  # nothing was created


def test_config_env_interpolation_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("ML_RAY_TMPDIR", str(tmp_path / "ray"))
    cfg = load_config("fusi-micro")
    p = cfg.paths.resolved()
    assert p.data_dir == tmp_path / "store" / "data"
    assert p.dagster_home == tmp_path / "store" / "dagster_home"
    assert cfg.tracking.tracking_uri == f"sqlite:///{tmp_path}/store/mlflow/mlflow.db"
    monkeypatch.setenv("ML_DATA_DIR", str(tmp_path / "elsewhere"))
    assert cfg.paths.resolved().data_dir == tmp_path / "elsewhere"
    for key in ("DAGSTER_HOME", "RAY_TMPDIR", "ML_RAY_SPILL_DIR", "DASK_TEMPORARY_DIRECTORY"):
        monkeypatch.setenv(key, "placeholder")  # registers the original value for restore
        monkeypatch.delenv(key)
    env = cfg.paths.apply_runtime_env()
    assert env["DAGSTER_HOME"] == str(tmp_path / "store" / "dagster_home")
    assert env["RAY_TMPDIR"] == str(tmp_path / "ray")
    assert (tmp_path / "store" / "ray-spill").is_dir()
