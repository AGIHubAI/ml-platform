"""fUSI step functions up to the dataset (no Ray): storage, split, train-only fit, checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_platform.config import PlatformConfig
from ml_platform.orchestration import fusi_pipeline as FP
from ml_platform.workloads.fusi_realtime.config import FusiConfig


@pytest.fixture
def fusi_cfg(tmp_path: Path) -> PlatformConfig:
    cfg = PlatformConfig.for_tests(tmp_path)
    cfg.fusi = FusiConfig.for_tests()
    return cfg


def test_steps_until_dataset(fusi_cfg: PlatformConfig) -> None:
    keys = FP.session_keys(fusi_cfg)
    assert keys == [f"sub00_ses{i:02d}" for i in range(4)]
    sessions = [FP.simulate_session(fusi_cfg, k) for k in keys]
    assert FP.simulate_session(fusi_cfg, keys[0]) == sessions[0]  # write-once, deterministic
    split = FP.plan_split(fusi_cfg, sessions)
    assert split.sessions == {
        "train": keys[:2],
        "val": [keys[2]],
        "test": [keys[3]],
    }
    pre = FP.fit_preprocessing(fusi_cfg, sessions, split)
    assert pre.fit_sessions == tuple(keys[:2])  # never the val/test sessions
    ds = FP.build_dataset(fusi_cfg, sessions, split, pre)
    assert all(c["passed"] for c in ds.checks), ds.checks
    assert {c["name"] for c in ds.checks} >= {"temporal_gap", "feature_schema"}
    meta = json.loads((ds.path / "dataset.json").read_text())
    assert "truth" not in " ".join(meta["inputs"])
    arrays, windows, _ = FP.load_arrays(fusi_cfg, ds, ("train", "val"))
    assert set(arrays) == set(keys[:3])  # training code never loads the test session
    assert set(windows.split) == {"train", "val"}
