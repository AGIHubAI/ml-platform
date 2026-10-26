"""Resource selection is CPU-safe and never trusts Ray's Apple-silicon 'GPU'."""

from __future__ import annotations

import pytest

from ml_platform.config import PlatformConfig
from ml_platform.ray_runtime import resources as R


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R.torch.cuda, "is_available", lambda: False)


def _cfg(**overrides: object) -> PlatformConfig:
    return PlatformConfig.for_tests().with_overrides(**overrides)


def test_auto_gpu_ignores_metal_gpu(monkeypatch: pytest.MonkeyPatch, cpu_only: None) -> None:
    monkeypatch.setattr(
        R, "_cluster_resources", lambda: {"CPU": 10.0, "GPU": 1.0, "accelerator_type:M4": 1.0}
    )
    res = R.select_resources(_cfg(**{"distributed.use_gpu": "auto"}))
    assert res.use_gpu is False
    assert res.remote_options() == {"num_cpus": 1, "num_gpus": 0.0}
    assert res.scaling_config_kwargs()["resources_per_worker"] == {"CPU": 1.0}


def test_forced_gpu_without_cuda_falls_back(
    monkeypatch: pytest.MonkeyPatch, cpu_only: None
) -> None:
    monkeypatch.setattr(R, "_cluster_resources", lambda: {"CPU": 4.0})
    res = R.select_resources(_cfg(**{"distributed.use_gpu": True}))
    assert res.use_gpu is False and any("falling back" in n for n in res.notes)


def test_workers_capped_by_cpus(monkeypatch: pytest.MonkeyPatch, cpu_only: None) -> None:
    monkeypatch.setattr(R, "_cluster_resources", lambda: {"CPU": 3.0})
    res = R.select_resources(_cfg(**{"distributed.cpus_per_worker": 2}), num_workers=4)
    assert res.num_workers == 1 and res.cpus_per_worker == 2
    res = R.select_resources(_cfg(), num_workers=8)
    assert res.num_workers == 3


def test_cuda_gpus_one_worker_per_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(R.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(R, "_cluster_resources", lambda: {"CPU": 16.0, "GPU": 2.0})
    res = R.select_resources(_cfg(**{"distributed.use_gpu": "auto"}), num_workers=4)
    assert res.use_gpu and res.num_workers == 2 and res.gpus_per_worker == 1.0
    assert res.scaling_config_kwargs()["resources_per_worker"]["GPU"] == 1.0
