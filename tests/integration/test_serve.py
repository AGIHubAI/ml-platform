"""Ray Serve end-to-end: deploy a tiny trained checkpoint and call every endpoint."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from ml_platform.config import PlatformConfig
from ml_platform.inference import Predictor
from ml_platform.training import TrainResult

pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray")
serve = pytest.importorskip("ray.serve")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server(
    trained: tuple[TrainResult, pd.DataFrame, pd.DataFrame],
) -> Iterator[tuple[str, Path]]:
    from ml_platform.inference.serve import run
    from ml_platform.ray_runtime.cluster import ensure_ray

    result, _, _ = trained
    started = not ray.is_initialized()
    ensure_ray(num_cpus=4, include_dashboard=False)
    port = _free_port()
    cfg = PlatformConfig.for_tests()
    run(
        result.checkpoint_path,
        "test-v1",
        cfg,
        port=port,
        blocking=False,
        max_batch_size=16,
        batch_wait_timeout_s=0.05,  # generous so concurrent requests coalesce
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    yield url, Path(result.checkpoint_path)
    serve.shutdown()
    if started:
        ray.shutdown()


def test_health_and_model_info(server: tuple[str, Path]) -> None:
    url, ckpt = server
    h = httpx.get(f"{url}/health", timeout=10).json()
    assert h["status"] == "ok" and h["model_version"] == "test-v1"
    info = httpx.get(f"{url}/model-info", timeout=10).json()
    local = Predictor.from_checkpoint(ckpt).model_info()
    assert info["checkpoint_hash"] == local["checkpoint_hash"]
    assert info["n_parameters"] == local["n_parameters"] > 0
    assert info["serving_config"]["max_batch_size"] == 16


def test_predict_and_predict_batch(server: tuple[str, Path], pool: pd.DataFrame) -> None:
    url, ckpt = server
    fcols = [c for c in pool.columns if c[:1] == "f" and c[1:].isdigit()]
    X = pool[fcols].head(5).to_numpy()
    r = httpx.post(f"{url}/predict", json={"features": X[0].tolist()}, timeout=10)
    assert r.status_code == 200, r.text
    one = r.json()
    assert one["model_version"] == "test-v1" and one["std"] > 0

    rb = httpx.post(f"{url}/predict_batch", json={"features": X.tolist()}, timeout=10)
    assert rb.status_code == 200, rb.text
    preds = rb.json()["predictions"]
    assert len(preds) == 5
    # the served mean agrees with the offline predictor (same checkpoint, same seed)
    mean, _ = Predictor.from_checkpoint(ckpt).predict_with_uncertainty(X, seed=0)
    np.testing.assert_allclose([p["mean"] for p in preds], mean, rtol=1e-5, atol=1e-6)

    bad = httpx.post(f"{url}/predict", json={"features": [1.0, 2.0]}, timeout=10)
    assert bad.status_code == 422


def test_concurrent_requests_are_batched(server: tuple[str, Path]) -> None:
    url, _ = server
    rng = np.random.default_rng(0)
    X = rng.standard_normal((48, 32))

    async def fire() -> list[int]:
        async with httpx.AsyncClient(timeout=30) as client:
            rs = await asyncio.gather(
                *(client.post(f"{url}/predict", json={"features": x.tolist()}) for x in X)
            )
        return [r.status_code for r in rs]

    assert set(asyncio.run(fire())) == {200}
    m = httpx.get(f"{url}/model-info", timeout=10).json()["metrics"]
    assert m["requests_by_route"]["/predict"] >= 48
    assert m["batch_size"]["max_recent"] > 1, m["batch_size"]
    assert m["latency_ms"]["p95"] is not None
