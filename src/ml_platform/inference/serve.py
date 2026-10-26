"""Online inference with Ray Serve: the surrogate model behind an HTTP API.

Endpoints (FastAPI ingress)::

    POST /predict        {"features": [32 floats]}         -> {mean, std, model_version}
    POST /predict_batch  {"features": [[32 floats], ...]}  -> {predictions: [...], model_version}
    GET  /health                                           -> {status, model_version}
    GET  /model-info     model version, checkpoint hash, #params, live metrics

Design notes (see also README "Serving"):

* **Model loading** — each replica loads the checkpoint once in ``__init__``
  (`Predictor.from_checkpoint`). The serving layer is handed a *checkpoint
  path + model version*; deciding which checkpoint is "production" is the
  tracking/registry layer's job, so this module never talks to MLflow. On a
  multi-node cluster the path must be on shared storage (NFS / object store
  mount) — or bake the checkpoint into the image.
* **Dynamic batching** — ``/predict`` takes one vector, but calls go through a
  ``@serve.batch`` method: concurrent single requests arriving within
  ``batch_wait_timeout_s`` are stacked into one forward pass of up to
  ``max_batch_size`` rows. This trades a few ms of latency for much higher
  throughput (one MC-dropout pass over B rows costs ~ the same as over 1).
  The model call runs in a worker thread so the replica's event loop keeps
  accepting (and batching) requests meanwhile.
* **Replicas & resources** — ``num_replicas`` from ``cfg.serve``; each replica
  is a Ray actor with ``num_cpus`` chosen from the cluster, and ``num_gpus``
  only when CUDA exists (Ray's Metal GPU resource on macOS is ignored).
* **Backpressure** — ``max_ongoing_requests`` caps concurrent requests per
  replica (must be >= ``max_batch_size`` so a full batch can form); excess
  requests queue at the proxy/handle, and beyond ``max_queued_requests``
  Serve rejects with HTTP 503 instead of letting latency grow unboundedly.
* **Autoscaling** — pass ``autoscaling=True`` (or set ``autoscaling_config``
  in a Serve config file) to let Serve scale replicas between min/max based
  on ``target_ongoing_requests`` per replica; it replaces ``num_replicas``.
* **Rolling model replacement** — redeploying the same application name with
  a new ``model_version``/checkpoint (``serve.run(build_app(new_ckpt, v2),
  name=...)`` or ``serve deploy`` with a new config) makes Serve start new
  replicas, wait for them to pass health checks, shift traffic, then drain
  and stop the old ones — no downtime; ``/model-info`` shows which version
  answered.
* **Metrics** — per-replica counters (requests, rows, batch sizes, latency
  ring buffer → p50/p95/p99) exposed on ``/model-info``; the same numbers are
  emitted as Ray metrics (``ray.serve.metrics`` Counter/Histogram) which Ray
  exports in Prometheus format for Grafana in production.

How to launch (three equivalent ways)::

    # 1. Python / `make serve` (blocks until Ctrl-C):
    uv run python -m ml_platform.inference.serve --checkpoint PATH --model-version 3
    # 2. Serve CLI with an application *builder* (args are key=value):
    serve run ml_platform.inference.serve:app_builder checkpoint=PATH model_version=3
    # 3. Serve CLI / RayService config with an import path and env vars:
    ML_SERVE_CHECKPOINT=PATH ML_SERVE_MODEL_VERSION=3 \\
        serve run ml_platform.inference.serve:app

``app`` is resolved lazily (module ``__getattr__``), so importing this module
never fails when the env vars are absent; only accessing ``app`` does.
"""

import asyncio
import collections
import logging
import os
import signal
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application

from ml_platform.config import PlatformConfig, load_config
from ml_platform.inference.predictor import Predictor
from ml_platform.logging import get_logger

log = get_logger(__name__)

ENV_CHECKPOINT = "ML_SERVE_CHECKPOINT"
ENV_MODEL_VERSION = "ML_SERVE_MODEL_VERSION"
APP_NAME = "surrogate"
LATENCY_WINDOW = 2048
BATCH_WINDOW = 256
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


# ----------------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="One feature vector (n_features floats)")


class PredictBatchRequest(BaseModel):
    features: list[list[float]] = Field(..., description="List of feature vectors")


class Prediction(BaseModel):
    mean: float
    std: float
    model_version: str


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    model_version: str


api = FastAPI(title="ml-platform surrogate", version="1")


# ----------------------------------------------------------------------------- metrics
def _percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values), q)) if values else None


class _ServeMetrics:
    """Replica-local counters + optional Ray metrics (Prometheus-exported)."""

    def __init__(self, model_version: str) -> None:
        self.started = time.time()
        self.requests: collections.Counter[str] = collections.Counter()
        self.errors = 0
        self.rows = 0
        self.batches = 0
        self.batch_hist: collections.Counter[int] = collections.Counter()
        self.last_batches: collections.deque[int] = collections.deque(maxlen=BATCH_WINDOW)
        self.latency_ms: collections.deque[float] = collections.deque(maxlen=LATENCY_WINDOW)
        self._ray: dict[str, Any] = {}
        try:
            from ray.serve import metrics as m

            tags = {"model_version": model_version}
            self._ray["requests"] = m.Counter(
                "ml_serve_requests",
                "Requests handled",
                tag_keys=("route", "model_version"),  # type: ignore[arg-type]
            )
            self._ray["batch"] = m.Histogram(
                "ml_serve_batch_size",
                "Rows per model forward pass",
                boundaries=list(BATCH_BUCKETS),
                tag_keys=("model_version",),
            )
            self._ray["latency"] = m.Histogram(
                "ml_serve_latency_ms",
                "End-to-end handler latency (ms)",
                boundaries=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                tag_keys=("route", "model_version"),  # type: ignore[arg-type]
            )
            for metric in self._ray.values():
                metric.set_default_tags(tags)
        except Exception:  # metrics are best-effort (e.g. outside a replica)
            self._ray = {}

    def record_request(self, route: str, latency_ms: float) -> None:
        self.requests[route] += 1
        self.latency_ms.append(latency_ms)
        if self._ray:
            self._ray["requests"].inc(tags={"route": route})
            self._ray["latency"].observe(latency_ms, tags={"route": route})

    def record_batch(self, size: int) -> None:
        self.batches += 1
        self.rows += size
        self.last_batches.append(size)
        bucket = next((b for b in BATCH_BUCKETS if size <= b), BATCH_BUCKETS[-1])
        self.batch_hist[bucket] += 1
        if self._ray:
            self._ray["batch"].observe(size)

    def snapshot(self) -> dict[str, Any]:
        lat = list(self.latency_ms)
        recent = list(self.last_batches)
        return {
            "uptime_s": round(time.time() - self.started, 3),
            "requests_total": sum(self.requests.values()),
            "requests_by_route": dict(self.requests),
            "errors_total": self.errors,
            "rows_predicted": self.rows,
            "batches_total": self.batches,
            "batch_size": {
                "last": recent[-32:],
                "mean_recent": float(np.mean(recent)) if recent else None,
                "max_recent": max(recent) if recent else None,
                "histogram_le": {str(k): v for k, v in sorted(self.batch_hist.items())},
            },
            "latency_ms": {
                "n": len(lat),
                "p50": _percentile(lat, 50),
                "p95": _percentile(lat, 95),
                "p99": _percentile(lat, 99),
            },
        }


# ----------------------------------------------------------------------------- deployment
@serve.deployment(
    name="SurrogateModel",
    # A request is "ongoing" from arrival until its response; the cap must allow
    # a full dynamic batch to form. Overridden per config in `build_app`.
    max_ongoing_requests=128,
    health_check_period_s=10,
    graceful_shutdown_timeout_s=10,
)
@serve.ingress(api)
class SurrogateModelDeployment:
    """One Serve replica: a loaded `Predictor` plus batching and metrics."""

    def __init__(
        self,
        checkpoint_path: str,
        model_version: str,
        max_batch_size: int = 64,
        batch_wait_timeout_s: float = 0.01,
        mc_samples: int | None = None,
        device: str = "cpu",
        num_threads: int | None = None,
    ) -> None:
        if num_threads:
            torch.set_num_threads(num_threads)
        t0 = time.perf_counter()
        self.predictor = Predictor.from_checkpoint(checkpoint_path, device=device)
        self.load_s = time.perf_counter() - t0
        self.model_version = str(model_version)
        self.mc_samples = int(mc_samples or self.predictor.mc_samples)
        self.n_features = self.predictor.normalizer.n_features
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_timeout_s = float(batch_wait_timeout_s)
        # Apply config to the @serve.batch queue (decorator values are defaults).
        self._predict_batched.set_max_batch_size(self.max_batch_size)  # type: ignore[attr-defined]
        self._predict_batched.set_batch_wait_timeout_s(self.batch_wait_timeout_s)  # type: ignore[attr-defined]
        # The model + torch's global RNG (seeded MC dropout) are not re-entrant.
        self._lock = threading.Lock()
        self.metrics = _ServeMetrics(self.model_version)
        try:
            self.replica_id = str(serve.get_replica_context().replica_id.unique_id)
        except Exception:
            self.replica_id = "local"
        # Logger created here (not the module-level one): Serve cloudpickles the
        # ingress class by value, and a bound structlog logger is not picklable.
        get_logger(__name__).info(
            "serve.replica_ready",
            model_version=self.model_version,
            checkpoint=checkpoint_path,
            load_s=round(self.load_s, 3),
            replica=self.replica_id,
        )

    # ------------------------------------------------------------------ model
    def _infer(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Seeded per call: a given batch is reproducible, but a row's MC std can
        # vary slightly with batch composition (dropout masks span the batch).
        with self._lock:
            return self.predictor.predict_with_uncertainty(X, self.mc_samples, seed=0)

    def _validate(self, rows: list[list[float]]) -> np.ndarray:
        if not rows:
            raise HTTPException(status_code=422, detail="no feature vectors given")
        X = np.asarray(rows, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise HTTPException(
                status_code=422,
                detail=f"expected vectors of {self.n_features} features, got shape {list(X.shape)}",
            )
        if not np.isfinite(X).all():
            raise HTTPException(status_code=422, detail="features must be finite")
        return X

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.01)
    async def _predict_batched(self, rows: list[np.ndarray]) -> list[tuple[float, float]]:
        X = np.stack(rows)
        mean, std = await asyncio.to_thread(self._infer, X)
        self.metrics.record_batch(len(rows))
        return list(zip(mean.tolist(), std.tolist(), strict=True))

    # ------------------------------------------------------------------ routes
    @api.post("/predict", response_model=Prediction)
    async def predict(self, req: PredictRequest) -> Prediction:
        t0 = time.perf_counter()
        try:
            x = self._validate([req.features])[0]
            mean, std = await self._predict_batched(x)
        except Exception:
            self.metrics.errors += 1
            raise
        self.metrics.record_request("/predict", (time.perf_counter() - t0) * 1e3)
        return Prediction(mean=mean, std=std, model_version=self.model_version)

    @api.post("/predict_batch", response_model=BatchPrediction)
    async def predict_batch(self, req: PredictBatchRequest) -> BatchPrediction:
        t0 = time.perf_counter()
        try:
            X = self._validate(req.features)
            # Already a batch: bypass the dynamic batcher, chunk to max_batch_size
            # so one huge request can't monopolize the replica's memory.
            means, stds = [], []
            for s in range(0, len(X), self.max_batch_size):
                m, sd = await asyncio.to_thread(self._infer, X[s : s + self.max_batch_size])
                self.metrics.record_batch(len(m))
                means.append(m)
                stds.append(sd)
            mean, std = np.concatenate(means), np.concatenate(stds)
        except Exception:
            self.metrics.errors += 1
            raise
        self.metrics.record_request("/predict_batch", (time.perf_counter() - t0) * 1e3)
        return BatchPrediction(
            predictions=[
                Prediction(mean=float(m), std=float(s), model_version=self.model_version)
                for m, s in zip(mean, std, strict=True)
            ],
            model_version=self.model_version,
        )

    @api.get("/health")
    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "model_version": self.model_version, "replica": self.replica_id}

    @api.get("/model-info")
    async def model_info(self) -> dict[str, Any]:
        info = self.predictor.model_info()
        return {
            "model_version": self.model_version,
            "checkpoint_hash": info.get("checkpoint_hash"),
            "checkpoint_path": info.get("checkpoint_path"),
            "n_parameters": info.get("n_parameters"),
            "n_features": self.n_features,
            "epochs_trained": info.get("epochs_trained"),
            "dataset_hash": info.get("dataset_hash"),
            "val_metrics": info.get("val_metrics"),
            "device": info.get("device"),
            "replica": self.replica_id,
            "load_s": round(self.load_s, 4),
            "serving_config": {
                "max_batch_size": self.max_batch_size,
                "batch_wait_timeout_s": self.batch_wait_timeout_s,
                "mc_samples": self.mc_samples,
                "torch_threads": torch.get_num_threads(),
            },
            "metrics": self.metrics.snapshot(),
        }

    def check_health(self) -> None:
        """Serve's periodic replica health check (raising marks it unhealthy)."""
        if self.predictor is None:  # pragma: no cover - defensive
            raise RuntimeError("model not loaded")


# ----------------------------------------------------------------------------- builders
def replica_resources(num_replicas: int) -> dict[str, float]:
    """``ray_actor_options`` per replica, chosen from the cluster's resources.

    CPUs: an even share of the available CPUs, clamped to [1, 4] (a small MLP
    gains little from more intra-op threads). GPUs: only when the process sees
    CUDA — never Ray's Apple-Metal ``GPU`` resource.
    """
    import ray

    avail = float(ray.available_resources().get("CPU", 1.0)) if ray.is_initialized() else 2.0
    cpus = float(min(4, max(1, int(avail // max(1, num_replicas)) - 1)))
    opts: dict[str, float] = {"num_cpus": cpus}
    if torch.cuda.is_available():
        opts["num_gpus"] = min(1.0, torch.cuda.device_count() / max(1, num_replicas))
    return opts


def build_app(
    checkpoint_path: str | os.PathLike[str],
    model_version: str,
    cfg: PlatformConfig | None = None,
    *,
    num_replicas: int | None = None,
    mc_samples: int | None = None,
    max_batch_size: int | None = None,
    batch_wait_timeout_s: float | None = None,
    autoscaling: bool = False,
    max_replicas: int = 4,
) -> Application:
    """Bind the surrogate deployment to a checkpoint (returns a Serve `Application`)."""
    cfg = cfg or load_config()
    path = str(Path(checkpoint_path).resolve())
    if not Path(path).exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    n_rep = int(num_replicas or cfg.serve.num_replicas)
    mbs = int(max_batch_size or cfg.serve.max_batch_size)
    wait = float(
        batch_wait_timeout_s if batch_wait_timeout_s is not None else cfg.serve.batch_wait_timeout_s
    )
    res = replica_resources(n_rep)
    options: dict[str, Any] = {
        "ray_actor_options": res,
        # >= max_batch_size so a full batch can form; a bit of headroom for
        # /health and /model-info calls while a batch is running.
        "max_ongoing_requests": max(2 * mbs, 32),
        # Beyond this queue depth Serve returns 503 (load shedding / backpressure).
        "max_queued_requests": 4096,
    }
    if autoscaling:
        options["autoscaling_config"] = {
            "min_replicas": 1,
            "initial_replicas": n_rep,
            "max_replicas": max(n_rep, max_replicas),
            "target_ongoing_requests": max(1, mbs // 2),
            "upscale_delay_s": 5,
            "downscale_delay_s": 30,
        }
    else:
        options["num_replicas"] = n_rep
    return SurrogateModelDeployment.options(**options).bind(  # type: ignore[attr-defined]
        path,
        str(model_version),
        max_batch_size=mbs,
        batch_wait_timeout_s=wait,
        mc_samples=mc_samples or cfg.evaluation.mc_samples,
        device="cuda" if "num_gpus" in res else "cpu",
        num_threads=int(res["num_cpus"]),
    )


def app_builder(args: dict[str, str] | None = None) -> Application:
    """Serve application builder: ``serve run ml_platform.inference.serve:app_builder
    checkpoint=PATH model_version=V [config=local] [num_replicas=N]``.

    Missing args fall back to ``ML_SERVE_CHECKPOINT`` / ``ML_SERVE_MODEL_VERSION``.
    """
    args = dict(args or {})
    ckpt = args.get("checkpoint") or os.environ.get(ENV_CHECKPOINT)
    if not ckpt:
        raise ValueError(
            f"no checkpoint: pass checkpoint=PATH or set {ENV_CHECKPOINT} "
            "(resolve the production checkpoint via the tracking registry first)"
        )
    version = args.get("model_version") or os.environ.get(ENV_MODEL_VERSION) or "unversioned"
    cfg = load_config(args.get("config"))
    n_rep = int(args["num_replicas"]) if "num_replicas" in args else None
    return build_app(ckpt, version, cfg, num_replicas=n_rep)


def __getattr__(name: str) -> Any:
    # Lazy module attribute: `serve run ml_platform.inference.serve:app`
    # builds from env vars at access time, while plain imports never fail.
    if name == "app":
        return app_builder({})
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ----------------------------------------------------------------------------- runners
class _DropServeReconnectNotice(logging.Filter):
    """Drop Serve's "Connecting to existing Serve app ..." INFO line.

    ``serve.run`` always calls Serve's internal ``serve_start`` (without HTTP
    options), which logs that notice whenever Serve is already up — i.e. on
    *every* ``serve.run`` after our explicit ``serve.start`` with a custom
    port, and on every rolling update. It carries no information here:
    ``run`` only calls ``serve.start`` when Serve is not running yet and
    checks the running HTTP address itself. Only that exact message is
    dropped; genuine option-mismatch warnings still come through.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Connecting to existing Serve app" not in record.getMessage()


_serve_logger = logging.getLogger("ray.serve")
if not any(isinstance(f, _DropServeReconnectNotice) for f in _serve_logger.filters):
    _serve_logger.addFilter(_DropServeReconnectNotice())


def _serve_http_config() -> Any | None:
    """HTTP options of the Serve instance in this Ray cluster, or None if none is running."""
    try:  # private but stable since Ray 2.0; serve.status() would *raise* instead
        from ray.serve.context import _get_global_client
    except ImportError:  # pragma: no cover - future Ray layouts: behave as "not running"
        return None
    client = _get_global_client(raise_if_no_controller_running=False)
    return client.http_config if client is not None else None


def run(
    checkpoint_path: str | os.PathLike[str],
    model_version: str,
    cfg: PlatformConfig | None = None,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    blocking: bool = True,
    name: str = APP_NAME,
    **build_kwargs: Any,
) -> Any:
    """Start Serve on ``cfg.serve.port`` and deploy the model.

    Returns the `DeploymentHandle` when ``blocking=False``; otherwise blocks
    until SIGINT/SIGTERM and then shuts Serve down. Calling ``run`` again
    with the same ``name`` and a new checkpoint/version performs a rolling
    replacement of the running replicas.
    """
    from ml_platform.ray_runtime.cluster import ensure_ray

    cfg = cfg or load_config()
    ensure_ray()
    port = int(port or cfg.serve.port)
    running = _serve_http_config()
    if running is None:
        serve.start(http_options={"host": host, "port": port})
    elif (running.host, running.port) != (host, port):
        # HTTP options are fixed for the lifetime of a Serve instance.
        log.warning(
            "serve.already_running_on_other_address",
            requested=f"{host}:{port}",
            running=f"{running.host}:{running.port}",
        )
        host, port = running.host, running.port
    app = build_app(checkpoint_path, model_version, cfg, **build_kwargs)
    handle = serve.run(app, name=name, route_prefix="/")
    log.info("serve.running", url=f"http://{host}:{port}", model_version=model_version, app=name)
    if not blocking:
        return handle
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        while not stop.wait(1.0):
            pass
    finally:
        log.info("serve.shutdown")
        serve.shutdown()
    return None


serve_forever = run


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Serve the surrogate model with Ray Serve")
    p.add_argument("--checkpoint", default=os.environ.get(ENV_CHECKPOINT))
    p.add_argument("--model-version", default=os.environ.get(ENV_MODEL_VERSION, "unversioned"))
    p.add_argument("--config", default=None, help="config file/name (default: ML_CONFIG)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--num-replicas", type=int, default=None)
    p.add_argument("--autoscaling", action="store_true")
    a = p.parse_args(argv)
    if not a.checkpoint:
        p.error(f"--checkpoint or {ENV_CHECKPOINT} is required")
    run(
        a.checkpoint,
        a.model_version,
        load_config(a.config),
        host=a.host,
        port=a.port,
        num_replicas=a.num_replicas,
        autoscaling=a.autoscaling,
    )


if __name__ == "__main__":
    main()
