"""Generic Ray workloads: stateless tasks, a stateful actor, object refs, backpressure.

Ray concepts demonstrated here (see also notebook 03):

**Tasks** (``@ray.remote`` functions) are stateless units of work. Ray's
scheduler places each task on a node with enough free *logical* resources
(``num_cpus``/``num_gpus``); tasks are queued, not rejected, when the cluster
is busy. Tasks are the right tool for embarrassingly parallel computation
such as bootstrap resampling or scoring shards of a candidate pool.

**Object store / ObjectRefs.** ``ray.put(x)`` stores ``x`` once in the
node-local shared-memory object store and returns an ``ObjectRef``. Passing
the ref (not the array) to N tasks means the array is serialized once, and
numpy arrays are read *zero-copy* by workers on the same node. Passing the
raw array to each ``.remote()`` call would serialize it N times. Top-level
ObjectRef arguments are resolved to values before the task runs; refs nested
inside containers stay refs (the task must ``ray.get`` them).

**Backpressure.** ``.remote()`` returns immediately, so a naive loop can
submit millions of tasks and exhaust driver/object-store memory. :func:`bounded_map`
keeps at most ``max_in_flight`` tasks outstanding and uses ``ray.wait`` to
admit a new task only when one finishes.

**Failure / retry semantics.**

* Tasks: ``max_retries`` re-executes a task whose *worker process died*
  (system failure). Application exceptions are retried only if
  ``retry_exceptions`` is ``True`` or lists the exception types. This is safe
  only for **idempotent, side-effect-free** computation — bootstrap
  resampling with a fixed seed returns the same answer on every attempt.
* Actors: ``max_restarts`` restarts a dead actor process (its ``__init__``
  runs again, in-memory state is lost); ``max_task_retries`` controls whether
  in-flight method calls are re-sent to the restarted actor.

**Retrying computation vs re-running an experiment.** :class:`ExperimentSimulator`
stands in for a *physical* experiment (the synthetic oracle). A lab
measurement costs money and time and its outcome is not reproducible, so it
must never be blindly retried. The actor therefore (a) uses
``max_task_retries=0`` (at-most-once method execution after a crash), and
(b) makes ``measure`` idempotent: results are recorded per
``(round_id, candidate_id)`` in a measurement log (optionally journaled to
disk so it survives an actor restart), and a retried request returns the
recorded measurement instead of measuring again. Durable versioning happens
downstream in the write-once ``RoundStore``.

**Placement.** Tasks/actors are placed by resource availability (and
optionally ``scheduling_strategy="SPREAD"`` or placement groups, which Ray
Train uses to gang-schedule all DDP workers at once, ``PACK`` by default).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
from numpy.typing import ArrayLike, NDArray

from ml_platform.config import PlatformConfig
from ml_platform.data.datasets import records_to_frame
from ml_platform.data.generation import make_oracle, measure_candidates
from ml_platform.data.schema import ExperimentRecord
from ml_platform.evaluation import metrics as M
from ml_platform.hashing import hash_dataframe
from ml_platform.logging import get_logger

log = get_logger(__name__)

METRICS: dict[str, Callable[[NDArray[np.float64], NDArray[np.float64]], float]] = {
    "rmse": M.rmse,
    "mae": M.mae,
    "r2": M.r2,
}


class TransientTaskError(RuntimeError):
    """A retryable application error (e.g. flaky storage read) — see ``retry_exceptions``."""


def ray_job_id() -> str | None:
    """Current Ray job id (hex) for log correlation, or None outside Ray."""
    if not ray.is_initialized():
        return None
    try:
        return str(ray.get_runtime_context().get_job_id())
    except Exception:  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- tasks
@ray.remote(num_cpus=1, max_retries=3, retry_exceptions=[TransientTaskError])  # type: ignore[call-overload]
def _bootstrap_chunk(
    y: NDArray[np.float64],
    yhat: NDArray[np.float64],
    metric: str,
    n_resamples: int,
    seed_entropy: int,
    spawn_key: tuple[int, ...],
) -> NDArray[np.float64]:
    """Compute ``n_resamples`` bootstrap replicates of ``metric``.

    ``y``/``yhat`` arrive as ObjectRefs from the driver and are resolved by
    Ray (zero-copy, read-only views on the same node). Deterministic in its
    seed, so a retry after a worker crash gives the identical result.
    """
    fn = METRICS[metric]
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy, spawn_key=spawn_key))
    n = len(y)
    out = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        out[b] = fn(y[idx], yhat[idx])
    return out


def parallel_bootstrap_ci(
    y: ArrayLike,
    yhat: ArrayLike,
    *,
    metric: str = "rmse",
    n_resamples: int = 1000,
    n_tasks: int = 4,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI ``(point, lo, hi)`` with resamples split across Ray tasks.

    The arrays are ``ray.put`` **once**; each task receives the ObjectRefs.
    Each task gets an independent RNG stream (``SeedSequence.spawn``), so the
    result is deterministic for a given ``(seed, n_tasks)`` — though not
    bit-identical to the serial ``evaluation.metrics.bootstrap_ci``.
    """
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    yh = np.asarray(yhat, dtype=np.float64).reshape(-1)
    point = float(METRICS[metric](ya, yh))
    if len(ya) < 2 or n_resamples <= 0:
        return point, point, point
    y_ref, yhat_ref = ray.put(ya), ray.put(yh)
    n_tasks = max(1, min(n_tasks, n_resamples))
    sizes = [len(c) for c in np.array_split(np.arange(n_resamples), n_tasks)]
    children = np.random.SeedSequence(seed).spawn(n_tasks)
    refs = [
        _bootstrap_chunk.remote(y_ref, yhat_ref, metric, size, int(seed), tuple(child.spawn_key))
        for size, child in zip(sizes, children, strict=True)
    ]
    stats = np.concatenate(ray.get(refs))
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def bounded_map(
    remote_fn: Any,
    items: Iterable[Any],
    *,
    max_in_flight: int = 4,
    remote_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    """Run ``remote_fn.remote(item)`` for every item with at most ``max_in_flight`` pending.

    Backpressure: when the window is full the driver blocks in ``ray.wait``
    until one task finishes, so memory stays bounded no matter how many
    items there are. Results are returned in input order.
    """
    kwargs = remote_kwargs or {}
    pending: dict[Any, int] = {}
    results: dict[int, Any] = {}
    for i, item in enumerate(items):
        if len(pending) >= max_in_flight:
            done, _ = ray.wait(list(pending), num_returns=1)
            for ref in done:
                results[pending.pop(ref)] = ray.get(ref)
        pending[remote_fn.remote(item, **kwargs)] = i
    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        for ref in done:
            results[pending.pop(ref)] = ray.get(ref)
    return [results[i] for i in range(len(results))]


@ray.remote(num_cpus=1, max_retries=2)
def square_sum_task(x: NDArray[np.float64]) -> float:
    """Tiny task used by the smoke test and notebooks (``sum(x**2)``)."""
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


# --------------------------------------------------------------------------- actor
@ray.remote(num_cpus=0, max_restarts=1, max_task_retries=0)
class ExperimentSimulator:
    """Stateful stand-in for the physical experiment (owns the synthetic oracle).

    State: the oracle, the candidate pool, and a measurement log keyed by
    ``(round_id, candidate_id)``. The log makes ``measure`` idempotent: asking
    again for a measurement that was already performed returns the recorded
    value (``n_physical_measurements`` does not increase). This is the
    difference between retrying *computation* (cheap, deterministic, safe to
    repeat) and re-running a *scientific experiment* (expensive, noisy,
    irreversible — must be recorded once and re-used).

    ``max_restarts=1`` restarts a crashed actor, but ``max_task_retries=0``
    means Ray will *not* automatically re-send an in-flight ``measure`` call
    to the restarted actor; the caller must decide (the first call after a
    crash raises ``ActorUnavailableError``/``ActorDiedError``). With
    ``journal_path`` the log is appended to a JSONL file and reloaded in
    ``__init__`` so a restart does not forget which experiments were already run.

    Restart-safe construction: ``__init__`` runs again on every restart with
    the *same* arguments, so they must still be valid then. The candidate
    catalogue is therefore passed as a **parquet path** on durable storage
    (the RoundStore's ``candidate_pool/pool.parquet`` in the closed loop), not
    as an ObjectRef — Ray does not pin constructor ObjectRefs for restarts
    (ray-project/ray#53727), and the lab reads its catalogue from the same
    storage it journals to. A DataFrame is accepted for direct, in-process use.
    """

    def __init__(
        self,
        cfg: dict[str, Any] | PlatformConfig,
        pool: str | os.PathLike[str] | pd.DataFrame,
        *,
        journal_path: str | None = None,
    ) -> None:
        self.cfg = cfg if isinstance(cfg, PlatformConfig) else PlatformConfig.model_validate(cfg)
        self.oracle = make_oracle(self.cfg)
        self.pool_path = None if isinstance(pool, pd.DataFrame) else str(pool)
        frame = pool if isinstance(pool, pd.DataFrame) else pd.read_parquet(pool)
        self.pool = frame.set_index("candidate_id", drop=False)
        self.journal = Path(journal_path) if journal_path else None
        self._log: dict[tuple[int, str], dict[str, Any]] = {}
        self.n_physical_measurements = 0
        self.n_cache_hits = 0
        if self.journal is not None and self.journal.exists():
            for line in self.journal.read_text().splitlines():
                rec = json.loads(line)
                self._log[(int(rec["round_id"]), str(rec["experiment_id"]))] = rec
        self._log_ctx = get_logger("experiment_simulator", ray_job_id=ray_job_id(), pid=os.getpid())

    def measure(
        self, round_id: int, candidate_ids: Sequence[str], seed: int | None = None
    ) -> pd.DataFrame:
        """Measure candidates for ``round_id`` (idempotent per ``(round_id, candidate_id)``).

        Returns an observations frame (``records_to_frame`` layout). New
        candidates are measured together as one batch with the same RNG
        derivation as :func:`~ml_platform.data.generation.measure_candidates`.
        """
        seed = self.cfg.seed if seed is None else seed
        ids = [str(c) for c in candidate_ids]
        new = [c for c in ids if (round_id, c) not in self._log]
        self.n_cache_hits += len(ids) - len(new)
        if new:
            missing = [c for c in new if c not in self.pool.index]
            if missing:
                raise KeyError(f"unknown candidate ids: {missing[:3]}")
            records = measure_candidates(
                self.pool.loc[new],
                self.oracle,
                round_id=round_id,
                seed=seed,
                provenance={"source": "ExperimentSimulator", "round_id": round_id},
            )
            fresh = [json.loads(r.model_dump_json()) for r in records]
            self.n_physical_measurements += len(fresh)
            if self.journal is not None:
                self.journal.parent.mkdir(parents=True, exist_ok=True)
                with self.journal.open("a") as fh:
                    for rec in fresh:
                        fh.write(json.dumps(rec) + "\n")
            for rec in fresh:
                self._log[(round_id, rec["experiment_id"])] = rec
        self._log_ctx.info(
            "simulator.measure",
            round_id=round_id,
            requested=len(ids),
            measured=len(new),
            cached=len(ids) - len(new),
        )
        return records_to_frame(
            [ExperimentRecord.model_validate(self._log[(round_id, c)]) for c in ids]
        )

    def stats(self) -> dict[str, Any]:
        return {
            "n_physical_measurements": self.n_physical_measurements,
            "n_cache_hits": self.n_cache_hits,
            "n_logged": len(self._log),
            "pid": os.getpid(),
            "pool_path": self.pool_path,
        }

    def pid(self) -> int:
        return os.getpid()


def durable_pool_file(pool: pd.DataFrame, directory: str | os.PathLike[str] | None = None) -> Path:
    """Write ``pool`` once to a content-addressed parquet file and return its path.

    ``directory`` defaults to ``$TMPDIR/ml_platform_lab``. Identical pools
    map to the same file, so repeated calls cost one hash, not one write.
    """
    root = Path(directory) if directory else Path(tempfile.gettempdir()) / "ml_platform_lab"
    path = root / f"candidate_pool_{hash_dataframe(pool)[:16]}.parquet"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f".{path.name}.{os.getpid()}.tmp"
        pool.to_parquet(tmp, index=False)
        tmp.replace(path)
    return path


def start_experiment_simulator(
    cfg: PlatformConfig,
    pool: pd.DataFrame | str | os.PathLike[str],
    *,
    journal_path: str | os.PathLike[str] | None = None,
    name: str | None = None,
) -> Any:
    """Create an :class:`ExperimentSimulator` actor.

    ``pool`` is the candidate catalogue: a parquet path (preferred — e.g. the
    RoundStore's pool file) or a DataFrame, which is written once to a
    content-addressed parquet file next to the journal (or in the temp dir).
    Either way the actor's constructor receives only small, by-value
    arguments, so ``max_restarts`` can always rebuild it (no ObjectRefs that
    could go out of scope).

    With ``name`` the actor is registered (``get_if_exists``) so every
    component in the namespace talks to the same "lab".
    """
    if isinstance(pool, pd.DataFrame):
        pool_path = durable_pool_file(pool, Path(journal_path).parent if journal_path else None)
    else:
        pool_path = Path(pool)
    opts: dict[str, Any] = {}
    if name:
        opts = {"name": name, "get_if_exists": True}
    return ExperimentSimulator.options(**opts).remote(  # type: ignore[attr-defined]
        cfg.model_dump(mode="json"),
        str(pool_path),
        journal_path=str(journal_path) if journal_path else None,
    )
