"""Ray Serve deployment of the fUSI decoder: stateless window decode, any replica serves any request.

The hot path keeps a KV-cache actor (``realtime.py``) because one stream = one cache.
Serving is the opposite trade: a request carries the last ``attn_window`` preprocessed
frames ``(W, x, y, z)`` and the replica recomputes the whole causal window, returning the
class probabilities of the last frame. Nothing is kept between requests, so replicas scale
horizontally and the answer does not depend on which replica (or batch) handled it.

Dynamic batching (``@serve.batch``) stacks concurrent requests of equal length into one
forward pass; ``max_batch_size`` / ``batch_wait_timeout_s`` come from ``fusi.serve``.
Every request runs in a ``fusi.serve.request`` span (``telemetry.py``). No HTTP: callers
use the returned `DeploymentHandle`.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle

from ml_platform.config import PlatformConfig, load_config
from ml_platform.logging import get_logger
from ml_platform.ray_runtime.cluster import ensure_ray
from ml_platform.workloads.fusi_realtime import telemetry
from ml_platform.workloads.fusi_realtime.config import FusiConfig
from ml_platform.workloads.fusi_realtime.models import FusiDecoder, load_any

log = get_logger(__name__)

ENV_CHECKPOINT = "ML_FUSI_CHECKPOINT"


@torch.no_grad()
def decode_last_frame(model: FusiDecoder, windows: Sequence[np.ndarray]) -> np.ndarray:
    """Probabilities ``(B, C)`` of the last frame of each ``(W, x, y, z)`` window.

    Windows of equal length are decoded in one forward pass; the rest in their own group.
    """
    out = np.zeros((len(windows), model.spec()["n_classes"]), dtype=np.float32)
    groups: dict[int, list[int]] = {}
    for i, w in enumerate(windows):
        groups.setdefault(int(w.shape[0]), []).append(i)
    for idx in groups.values():
        x = torch.from_numpy(np.stack([np.asarray(windows[i], dtype=np.float32) for i in idx]))
        probs = torch.softmax(model(x)[:, -1], dim=-1)
        out[idx] = probs.numpy()
    return out


@serve.deployment(name="FusiDecoder", max_ongoing_requests=128, graceful_shutdown_timeout_s=5)
class FusiDecoderDeployment:
    """One replica: a loaded transformer decoder behind a dynamic-batching queue."""

    def __init__(self, checkpoint_dir: str, fusi_cfg: dict[str, Any]) -> None:
        torch.set_num_threads(1)
        cfg = FusiConfig.model_validate(fusi_cfg)
        kind, model = load_any(checkpoint_dir)
        if not isinstance(model, FusiDecoder):
            raise ValueError(f"serving supports the transformer decoder, got {kind!r}")
        self.model = model.eval()
        self.tracer = telemetry.configure(cfg.telemetry, service="fusi-serve")
        # decorator values are defaults; apply the configured ones to this replica's queue
        self._decode_batch.set_max_batch_size(cfg.serve.max_batch_size)  # type: ignore[attr-defined]
        self._decode_batch.set_batch_wait_timeout_s(cfg.serve.batch_wait_timeout_s)  # type: ignore[attr-defined]

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.005)
    async def _decode_batch(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        # the forward pass runs in a thread so the event loop keeps filling the next batch
        probs = await asyncio.to_thread(decode_last_frame, self.model, windows)
        return list(probs)

    async def decode(self, window: np.ndarray) -> np.ndarray:
        """Class probabilities ``(C,)`` of the last frame of ``window`` ``(W, x, y, z)``."""
        with telemetry.span(self.tracer, "fusi.serve.request", n_frames=int(window.shape[0])):
            out: np.ndarray = await self._decode_batch(window)
            return out


def build_app(cfg: PlatformConfig, checkpoint_dir: str) -> Application:
    deployment: Any = FusiDecoderDeployment  # serve.deployment returns an untyped Deployment
    app: Application = deployment.options(num_replicas=cfg.fusi.serve.num_replicas).bind(
        str(checkpoint_dir), cfg.fusi.model_dump(mode="json")
    )
    return app


def app_builder(args: dict[str, str] | None = None) -> Application:
    """Serve builder for a RayService: ``serve run
    ml_platform.workloads.fusi_realtime.serve_decoder:app_builder checkpoint=DIR
    [config=PATH]``; ``checkpoint`` falls back to ``$ML_FUSI_CHECKPOINT``."""
    args = dict(args or {})
    ckpt = args.get("checkpoint") or os.environ.get(ENV_CHECKPOINT)
    if not ckpt:
        raise ValueError(f"no checkpoint: pass checkpoint=DIR or set {ENV_CHECKPOINT}")
    return build_app(load_config(args.get("config")), ckpt)


def deploy_decoder(
    cfg: PlatformConfig, checkpoint_dir: str, *, name: str = "fusi_decoder"
) -> DeploymentHandle:
    """Start the deployment locally (no HTTP) and return its handle once it is serving."""
    ensure_ray()
    app = build_app(cfg, checkpoint_dir)
    handle = serve.run(app, name=name, route_prefix=None, blocking=False)
    log.info("fusi.serve.running", app=name, replicas=cfg.fusi.serve.num_replicas)
    return handle


def decode_windows(handle: DeploymentHandle, windows: Sequence[np.ndarray]) -> np.ndarray:
    """Decode many windows concurrently (so the replicas can batch); returns ``(N, C)``."""
    responses = [handle.decode.remote(np.asarray(w, dtype=np.float32)) for w in windows]
    return np.stack([r.result() for r in responses])


def shutdown_decoder() -> None:
    """Tear down every Serve application (tests and demos call this)."""
    serve.shutdown()
