"""Distributed batch scoring of the candidate pool with Ray actors.

Ray concepts demonstrated (handoff §7):

* **Object store** — the checkpoint bytes and the pool feature matrix are
  ``ray.put`` *once*. Every actor/call receives an ``ObjectRef``; on the same
  node NumPy arrays are read zero-copy from shared memory, across nodes Ray
  transfers them once per node. Nothing is re-pickled per shard.
* **Actors as a model cache** — `PredictorActor` deserializes the checkpoint
  on its first call and then serves many shards. Model loading (the
  expensive, stateful part) is paid once per actor, not once per task.
* **Resource requests** — ``num_cpus`` per actor is derived from the cluster's
  available CPUs; ``num_gpus`` is requested only when CUDA is available.
  (Ray on Apple Silicon advertises a Metal ``GPU`` resource that PyTorch-CUDA
  cannot use, so the decision is based on ``torch.cuda.is_available()``, never
  on Ray's GPU count.)
* **Backpressure** — at most ``max_in_flight`` shard requests are outstanding;
  the driver ``ray.wait``\\ s for one to finish before submitting the next. This
  bounds object-store memory for results and keeps actor mailboxes short
  (instead of enqueueing all shards up front).
* **Failure behaviour** — actors are created with ``max_restarts=1`` and calls
  with ``max_task_retries=1``: a crashed actor process is restarted and the
  shard re-run. Scoring is a pure function of (checkpoint, shard, seed), so
  retrying is safe — unlike re-running a physical experiment.
* **Restart-safe inputs** — the actor constructor takes only small scalars.
  The checkpoint travels with every ``predict_shard`` call as
  ``[ObjectRef]`` (a ref *nested* in a list, so Ray does not resolve/copy it
  per call) and the actor fetches + deserializes it once, on first use. A
  restarted actor simply reloads it from the ref carried by the retried call:
  Ray keeps the arguments of a pending/retryable task alive, whereas an
  ObjectRef passed to the *constructor* of a restartable actor is not pinned
  for restarts (Ray warns about exactly that, ray-project/ray#53727).

Determinism: MC-dropout masks are seeded per *shard* (``seed`` + shard index),
so results are independent of which actor scored which shard and of
completion order. `predict_pool_local` runs the identical shard/seed schedule
in-process and is the reference implementation used by tests.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ml_platform.data.schema import feature_columns
from ml_platform.inference.predictor import Predictor
from ml_platform.logging import get_logger
from ml_platform.training.checkpointing import MODEL_FILE, resolve_checkpoint

log = get_logger(__name__)

DEFAULT_SHARD_SIZE = 10_000


def shard_bounds(n: int, shard_size: int) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` covering ``range(n)`` in contiguous shards."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    return [(s, min(s + shard_size, n)) for s in range(0, n, shard_size)]


def shard_seed(seed: int, shard_index: int) -> int:
    """Deterministic per-shard MC-dropout seed (independent of actor assignment)."""
    return int(np.random.SeedSequence([seed, shard_index, 0xBA7C]).generate_state(1)[0])


def _pool_matrix(pool: pd.DataFrame, n_features: int | None = None) -> np.ndarray:
    cols = (
        feature_columns(n_features)
        if n_features
        else [c for c in pool.columns if c[:1] == "f" and c[1:].isdigit()]
    )
    return np.ascontiguousarray(pool[cols].to_numpy(dtype=np.float64))


def _score_shard(
    predictor: Predictor, X: np.ndarray, mc_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    return predictor.predict_with_uncertainty(X, mc_samples, seed=seed)


def _result_frame(ids: pd.Series, mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ids.astype(str).to_numpy(),
            "pred_mean": mean.astype(np.float64),
            "pred_std": std.astype(np.float64),
        }
    )


# ----------------------------------------------------------------------------- local
def predict_pool_local(
    checkpoint_path: str | os.PathLike[str],
    pool_frame: pd.DataFrame,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    mc_samples: int | None = None,
    seed: int = 0,
    predictor: Predictor | None = None,
) -> pd.DataFrame:
    """In-process reference implementation of `predict_pool` (same shards and seeds)."""
    pred = predictor or Predictor.from_checkpoint(checkpoint_path)
    n_mc = int(mc_samples or pred.mc_samples)
    X = _pool_matrix(pool_frame, pred.normalizer.n_features)
    means, stds = [], []
    for i, (a, b) in enumerate(shard_bounds(len(X), shard_size)):
        m, s = _score_shard(pred, X[a:b], n_mc, shard_seed(seed, i))
        means.append(m)
        stds.append(s)
    mean = np.concatenate(means) if means else np.zeros(0)
    std = np.concatenate(stds) if stds else np.zeros(0)
    return _result_frame(pool_frame["candidate_id"], mean, std)


# ----------------------------------------------------------------------------- ray
class PredictorActor:
    """Stateful model-cache worker: load the checkpoint once, score many shards.

    Wrapped with ``ray.remote`` at call time (see `_actor_class`) so importing
    this module never requires Ray. ``model_ref`` arguments are
    ``[ObjectRef(checkpoint bytes)]``; the model is loaded from the first one
    seen (and re-loaded only if a different checkpoint ref arrives).
    """

    def __init__(self, num_threads: int = 1, device: str = "cpu") -> None:
        torch.set_num_threads(max(1, num_threads))
        self.device = device
        self.predictor: Predictor | None = None
        self._model_key: str | None = None
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self.load_s = 0.0
        self.shards_served = 0
        self.rows_served = 0

    def _model(self, model_ref: list[Any]) -> Predictor:
        import ray

        (ref,) = model_ref
        key = ref.hex()
        if self.predictor is None or key != self._model_key:
            t0 = time.perf_counter()
            # Materialize the checkpoint locally from the object store: works on a
            # multi-node cluster without a shared filesystem.
            self._tmp = tempfile.TemporaryDirectory(prefix="ml_predictor_")
            path = Path(self._tmp.name) / MODEL_FILE
            path.write_bytes(ray.get(ref))
            self.predictor = Predictor.from_checkpoint(path, device=self.device)
            self._model_key = key
            self.load_s = time.perf_counter() - t0
        return self.predictor

    def predict_shard(
        self,
        model_ref: list[Any],
        X: np.ndarray,
        start: int,
        stop: int,
        mc_samples: int,
        seed: int,
        shard_index: int,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        mean, std = _score_shard(self._model(model_ref), X[start:stop], mc_samples, seed)
        self.shards_served += 1
        self.rows_served += stop - start
        return shard_index, mean, std

    def stats(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "load_s": self.load_s,
            "shards_served": self.shards_served,
            "rows_served": self.rows_served,
            "checkpoint_hash": (
                self.predictor.model_info().get("checkpoint_hash") if self.predictor else None
            ),
        }

    def default_mc_samples(self, model_ref: list[Any]) -> int:
        return int(self._model(model_ref).mc_samples)


def actor_resources(n_actors: int, available_cpus: float | None = None) -> dict[str, float]:
    """Per-actor resource request chosen from what the cluster actually has.

    CPU: split the available CPUs across actors (at least 1 each). GPU: only if
    the *driver* sees CUDA (Ray's Metal ``GPU`` resource on macOS is ignored);
    actors then share the GPUs fractionally.
    """
    if available_cpus is None:
        import ray

        available_cpus = float(ray.available_resources().get("CPU", 1.0))
    cpus = max(1.0, float(int(available_cpus // max(1, n_actors))))
    res: dict[str, float] = {"num_cpus": cpus}
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        res["num_gpus"] = min(1.0, n_gpu / max(1, n_actors))
    return res


def _actor_class() -> Any:
    import ray

    return ray.remote(max_restarts=1, max_task_retries=1)(PredictorActor)


def predict_pool(
    checkpoint_path: str | os.PathLike[str],
    pool_frame: pd.DataFrame,
    *,
    n_actors: int = 2,
    shard_size: int = DEFAULT_SHARD_SIZE,
    mc_samples: int | None = None,
    seed: int = 0,
    max_in_flight: int | None = None,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Score ``pool_frame`` with MC-dropout on a pool of Ray actors.

    Returns ``DataFrame[candidate_id, pred_mean, pred_std]`` in the row order
    of ``pool_frame``; with ``return_stats=True`` also a dict of execution
    statistics (actors, shards, per-actor load time/rows, wall time).
    Numerically equal to `predict_pool_local` with the same ``shard_size``,
    ``mc_samples`` and ``seed``. Initializes Ray via ``ensure_ray`` if needed.
    """
    import ray

    from ml_platform.ray_runtime.cluster import ensure_ray

    ensure_ray()
    t0 = time.perf_counter()
    ckpt_file = resolve_checkpoint(checkpoint_path)
    X = _pool_matrix(pool_frame)
    bounds = shard_bounds(len(X), shard_size)
    if not bounds:
        empty = _result_frame(pool_frame["candidate_id"], np.zeros(0), np.zeros(0))
        return (empty, {"n_shards": 0}) if return_stats else empty

    n_act = max(1, min(n_actors, len(bounds)))
    res = actor_resources(n_act)
    use_cuda = "num_gpus" in res
    # object store: put the large inputs once, pass refs everywhere. The
    # checkpoint ref is nested in a list: it rides along with every call (so a
    # restarted actor can reload it) without being resolved per call.
    model_ref = [ray.put(ckpt_file.read_bytes())]
    X_ref = ray.put(X)
    Actor = _actor_class()
    actors = [
        Actor.options(**res).remote(  # type: ignore[attr-defined]
            int(res["num_cpus"]), "cuda" if use_cuda else "cpu"
        )
        for _ in range(n_act)
    ]
    n_mc = int(mc_samples or ray.get(actors[0].default_mc_samples.remote(model_ref)))
    limit = max(1, max_in_flight or 2 * n_act)
    log.info(
        "batch_inference.start",
        n_rows=len(X),
        n_shards=len(bounds),
        n_actors=n_act,
        actor_resources=res,
        max_in_flight=limit,
    )

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    in_flight: list[Any] = []
    try:
        for i, (a, b) in enumerate(bounds):
            if len(in_flight) >= limit:  # backpressure: wait for a slot
                done, in_flight = ray.wait(in_flight, num_returns=1)
                idx, m, s = ray.get(done[0])
                results[idx] = (m, s)
            actor = actors[i % n_act]  # round-robin; seeds don't depend on this
            in_flight.append(
                actor.predict_shard.remote(model_ref, X_ref, a, b, n_mc, shard_seed(seed, i), i)
            )
        for idx, m, s in ray.get(in_flight):
            results[idx] = (m, s)
        stats: dict[str, Any] = {
            "n_rows": len(X),
            "n_shards": len(bounds),
            "n_actors": n_act,
            "actor_resources": res,
            "mc_samples": n_mc,
            "max_in_flight": limit,
            "actors": ray.get([a.stats.remote() for a in actors]),
        }
    finally:
        for a in actors:
            ray.kill(a)

    mean = np.concatenate([results[i][0] for i in range(len(bounds))])
    std = np.concatenate([results[i][1] for i in range(len(bounds))])
    out = _result_frame(pool_frame["candidate_id"], mean, std)
    stats["wall_s"] = time.perf_counter() - t0
    stats["rows_per_s"] = len(X) / max(stats["wall_s"], 1e-9)
    log.info(
        "batch_inference.done",
        **{k: v for k, v in stats.items() if k not in ("actors", "actor_resources")},
    )
    return (out, stats) if return_stats else out
