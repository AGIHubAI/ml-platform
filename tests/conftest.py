"""Shared fixtures: a tiny config, a small pool/oracle and a trained checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("ML_LOG_LEVEL", "WARNING")

from ml_platform.config import PlatformConfig
from ml_platform.data import (
    ArrayDataset,
    RoundStore,
    SyntheticOracle,
    generate_candidate_pool,
    initial_observations,
    make_oracle,
    records_to_frame,
    train_val_split,
)
from ml_platform.training import Trainer, TrainResult


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> PlatformConfig:
    return PlatformConfig.for_tests(tmp_path)


@pytest.fixture(scope="session")
def session_cfg() -> PlatformConfig:
    return PlatformConfig.for_tests()


@pytest.fixture(scope="session")
def pool(session_cfg: PlatformConfig) -> pd.DataFrame:
    return generate_candidate_pool(session_cfg)


@pytest.fixture(scope="session")
def oracle(session_cfg: PlatformConfig) -> SyntheticOracle:
    return make_oracle(session_cfg)


@pytest.fixture(scope="session")
def obs_frame(pool: pd.DataFrame, oracle: SyntheticOracle) -> pd.DataFrame:
    return records_to_frame(initial_observations(pool, oracle, 300, seed=0))


@pytest.fixture
def datasets(
    obs_frame: pd.DataFrame, session_cfg: PlatformConfig
) -> tuple[ArrayDataset, ArrayDataset]:
    tr, va = train_val_split(obs_frame, session_cfg.training.val_fraction, seed=0)
    return (
        ArrayDataset.from_frame(tr, dataset_hash="test-dataset"),
        ArrayDataset.from_frame(va, dataset_hash="test-dataset"),
    )


@pytest.fixture(scope="session")
def trained(
    tmp_path_factory: pytest.TempPathFactory, obs_frame: pd.DataFrame, session_cfg: PlatformConfig
) -> tuple[TrainResult, pd.DataFrame, pd.DataFrame]:
    """A model trained once per session on the tiny config (returns result, train, val)."""
    tr, va = train_val_split(obs_frame, session_cfg.training.val_fraction, seed=0)
    ckpt_dir = tmp_path_factory.mktemp("trained") / "checkpoints"
    result = Trainer(session_cfg).train(
        ArrayDataset.from_frame(tr, dataset_hash="test-dataset"),
        ArrayDataset.from_frame(va, dataset_hash="test-dataset"),
        checkpoint_dir=ckpt_dir,
    )
    return result, tr, va


@pytest.fixture
def store(tmp_path: Path) -> RoundStore:
    return RoundStore(tmp_path / "data")
