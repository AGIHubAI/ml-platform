"""Scale profiles: same code, config-only differences; KubeRay manifests point at real code."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from ml_platform.config import REPO_ROOT, load_config

PROFILES = ("fusi-micro", "fusi-gpu", "fusi-multinode")


@pytest.mark.parametrize("name", PROFILES)
def test_profile_loads(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ML_STORAGE_ROOT", str(tmp_path))
    cfg = load_config(name)
    assert cfg.fusi.training.batch_size % cfg.fusi.training.num_workers == 0
    assert cfg.paths.resolved().data_dir.is_relative_to(tmp_path)


def test_profiles_differ_only_in_scale() -> None:
    micro, multi = load_config("fusi-micro").fusi, load_config("fusi-multinode").fusi
    assert multi.training.strategy == "fsdp2" and micro.training.strategy == "ddp"
    # the scientific definition that must not drift between profiles
    for section in ("preprocess", "split"):
        assert getattr(micro, section) == getattr(multi, section)
    assert micro.gates.p99_budget_ms == multi.gates.p99_budget_ms


def test_kuberay_manifests_reference_real_code() -> None:
    job = yaml.safe_load((REPO_ROOT / "infra/k8s/fusi-rayjob.yaml").read_text())
    config = job["spec"]["entrypoint"].split("--config ")[1].split()[0]
    assert (REPO_ROOT / config).is_file()
    svc = yaml.safe_load((REPO_ROOT / "infra/k8s/fusi-rayservice.yaml").read_text())
    app = yaml.safe_load(svc["spec"]["serveConfigV2"])["applications"][0]
    module, attr = app["import_path"].split(":")
    assert callable(getattr(importlib.import_module(module), attr))
    assert (REPO_ROOT / app["args"]["config"]).is_file()


def test_serve_app_builder_requires_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml_platform.workloads.fusi_realtime.serve_decoder import app_builder

    monkeypatch.delenv("ML_FUSI_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="checkpoint"):
        app_builder({})
    app = app_builder({"checkpoint": "/nonexistent", "config": "fusi-micro"})
    assert type(app).__name__ == "Application"  # built lazily; replicas load the checkpoint
