"""Worker-count invariance of training (1/2/4 Ray Train workers) and Serve replica invariance."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ray
import torch

from ml_platform.config import PlatformConfig
from ml_platform.ray_runtime.cluster import ensure_ray, shutdown_ray
from ml_platform.workloads.fusi_realtime.config import FusiConfig
from ml_platform.workloads.fusi_realtime.dataset import SessionArrays
from ml_platform.workloads.fusi_realtime.models import (
    build_decoder,
    load_any,
    load_decoder,
    save_decoder,
)
from ml_platform.workloads.fusi_realtime.serve_decoder import (
    FusiDecoderDeployment,
    decode_windows,
    deploy_decoder,
    shutdown_decoder,
)
from ml_platform.workloads.fusi_realtime.train_ray import train_decoder_distributed

pytestmark = pytest.mark.smoke

GRID = (4, 4, 4)
ATOL = 1e-5


@pytest.fixture(scope="module")
def ray_cluster() -> Iterator[None]:
    already = ray.is_initialized()
    ensure_ray(num_cpus=6, include_dashboard=False)
    yield
    if not already:
        shutdown_ray()


def _tiny_cfg(tmp: Path) -> PlatformConfig:
    cfg = PlatformConfig.for_tests(tmp)
    cfg.fusi = FusiConfig.for_tests()
    cfg.fusi.decoder.dropout = 0.0  # per-rank dropout streams differ; invariance needs 0
    cfg.fusi.training.epochs = 2
    cfg.fusi.training.batch_size = 8
    cfg.fusi.training.input_noise = 0.5
    return cfg


def _data() -> tuple[dict[str, SessionArrays], pd.DataFrame]:
    rng = np.random.default_rng(0)
    arrays, rows = {}, []
    for s in ("a", "b"):
        labels = rng.integers(0, 3, 200)
        labels[:15] = -100
        arrays[s] = SessionArrays(rng.standard_normal((200, *GRID)).astype(np.float32), labels)
        for start in range(0, 180, 12):
            rows.append((s, start, start + 12, "val" if s == "b" and start >= 96 else "train"))
    return arrays, pd.DataFrame(rows, columns=["session", "start", "end", "split"])


def test_training_is_invariant_to_worker_count(tmp_path: Path, ray_cluster: None) -> None:
    arrays, windows = _data()
    states, histories = {}, {}
    for world in (1, 2, 4):
        cfg = _tiny_cfg(tmp_path / f"w{world}")
        cfg.fusi.training.num_workers = world
        out = tmp_path / f"out{world}"
        summary = train_decoder_distributed(
            cfg, arrays=arrays, windows=windows, grid=GRID, n_classes=3, out_dir=out,
            preprocessor_hash="p", dataset_hash="d",
        )  # fmt: skip
        assert summary["world_size"] == world and summary["params_in_sync"]
        states[world] = load_decoder(str(out / "model.pt"))[0].state_dict()
        histories[world] = summary["history"]
    for world in (2, 4):
        diff = max(float((states[1][k] - states[world][k]).abs().max()) for k in states[1])
        print(f"max |dW| 1 vs {world} workers: {diff:.3e}")
        assert diff < ATOL
        for h1, hw in zip(histories[1], histories[world], strict=True):
            for key in ("train_loss", "train_frame_acc", "val_loss", "val_frame_acc"):
                assert abs(h1[key] - hw[key]) < 1e-4, (key, h1[key], hw[key])


def test_serve_replica_invariance(tmp_path: Path, ray_cluster: None) -> None:
    cfg = _tiny_cfg(tmp_path)
    cfg.fusi.serve.max_batch_size = 4
    torch.manual_seed(1)
    model = build_decoder(cfg.fusi.decoder, GRID, 3)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_decoder(model, str(ckpt / "model.pt"))
    w = cfg.fusi.decoder.attn_window
    windows = [np.random.default_rng(i).standard_normal((w, *GRID)).astype(np.float32)
               for i in range(16)]  # fmt: skip
    _kind, direct = load_any(str(ckpt))
    assert hasattr(direct, "eval")
    with torch.no_grad():
        ref = np.stack(
            [torch.softmax(direct(torch.from_numpy(x)[None])[0, -1], -1).numpy()  # type: ignore[operator]
             for x in windows]
        )  # fmt: skip
    outs = {}
    try:
        for n in (1, 2):
            cfg.fusi.serve.num_replicas = n
            handle = deploy_decoder(cfg, str(ckpt), name=f"fusi_decoder_{n}")
            outs[n] = decode_windows(handle, windows)
            shutdown_decoder()
    finally:
        shutdown_decoder()
    print(f"max |dP| 1 vs 2 replicas: {np.abs(outs[1] - outs[2]).max():.3e}")
    print(f"max |dP| replica vs direct: {np.abs(outs[2] - ref).max():.3e}")
    np.testing.assert_allclose(outs[1], outs[2], atol=1e-6)
    np.testing.assert_allclose(outs[2], ref, atol=1e-6)
    assert FusiDecoderDeployment is not None
