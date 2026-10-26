from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_platform.config import PlatformConfig, load_config
from ml_platform.hashing import hash_config, hash_dataframe, hash_file


@pytest.mark.parametrize("name", ["local", "distributed", "gpu"])
def test_repo_configs_load(name: str) -> None:
    cfg = load_config(name)
    assert cfg.data.n_features == 32
    assert cfg.evaluation.max_rmse == pytest.approx(0.35)


def test_local_matches_plan_defaults() -> None:
    cfg = load_config("local")
    assert cfg.data.pool_size == 100_000
    assert cfg.data.initial_observations == 500
    assert cfg.model.hidden_dims == [256, 256, 128]
    assert load_config("gpu").distributed.use_gpu is True
    assert load_config("distributed").distributed.num_workers == 2


def test_env_var_selects_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_CONFIG", "gpu")
    assert load_config().distributed.use_gpu is True


def test_overrides_and_for_tests(tmp_path: Path) -> None:
    cfg = PlatformConfig.for_tests(tmp_path, **{"training.epochs": 2})
    assert cfg.training.epochs == 2
    assert cfg.data.pool_size == 2000
    assert cfg.paths.data_dir == tmp_path / "data"
    assert "training.epochs" in cfg.to_flat_dict()


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("training: {epochs: 3, not_a_key: 1}\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_hash_dataframe_column_order_and_dtype_invariant() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 1.5, -0.0], "c": ["x", "y", "z"]})
    reordered = df[["c", "a", "b"]]
    as_int32 = df.astype({"a": "int32"})
    positive_zero = df.assign(b=[0.5, 1.5, 0.0])
    h = hash_dataframe(df)
    assert (
        h == hash_dataframe(reordered) == hash_dataframe(as_int32) == hash_dataframe(positive_zero)
    )
    assert h != hash_dataframe(df.assign(b=[0.5, 1.5, 1e-12]))
    assert h != hash_dataframe(df.iloc[::-1])
    assert h == hash_dataframe(df.iloc[::-1], sort_rows_by="a")


def test_hash_config_and_file(tmp_path: Path) -> None:
    assert hash_config({"a": 1, "b": [1, 2]}) == hash_config({"b": [1, 2], "a": 1})
    assert hash_config(PlatformConfig()) == hash_config(PlatformConfig())
    f = tmp_path / "x.bin"
    f.write_bytes(np.arange(10).tobytes())
    assert hash_file(f) == hash_file(f)
    assert len(hash_file(tmp_path)) == 64


def test_config_hash_ignores_environment_sections(tmp_path: Path) -> None:
    base = PlatformConfig.for_tests(tmp_path / "a")
    moved = PlatformConfig.for_tests(tmp_path / "somewhere" / "else")  # other data dir + MLflow db
    assert base.paths != moved.paths and base.tracking != moved.tracking
    env_only = base.with_overrides(
        **{
            "serve.port": 9999,
            "tracking.experiment": "other",
            "distributed.use_gpu": True,
            "distributed.cpus_per_worker": 4,
        }
    )
    assert base.config_hash() == moved.config_hash() == env_only.config_hash()
    # ...but anything that changes the science changes the hash
    for key, value in {
        "seed": 1,
        "data.pool_size": 123,
        "model.dropout": 0.2,
        "training.lr": 0.01,
        "evaluation.max_rmse": 0.5,
        "active_learning.beta": 2.0,
        "distributed.num_workers": 3,  # global batch size changes under DDP
    }.items():
        assert base.with_overrides(**{key: value}).config_hash() != base.config_hash(), key
    assert set(base.scientific_dump()) == {
        "seed",
        "data",
        "model",
        "training",
        "evaluation",
        "active_learning",
        "distributed",
    }
