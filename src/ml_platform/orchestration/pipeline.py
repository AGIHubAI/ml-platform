"""The closed-loop workflow as plain Python step functions (no Dagster import).

Each function below is one node of the handoff §8 asset graph::

    candidate_pool → observed_experiments → training_dataset → trained_model
      → evaluation_report → registered_model → candidate_predictions
      → selected_experiments → new_experimental_results (= round r+1)

The Dagster assets in :mod:`ml_platform.orchestration.assets` are thin
adapters around these functions: they resolve resources, bind correlation ids
and turn the returned ``*Info`` records into materialization metadata. Scripts
(``scripts/bootstrap.py``, ``scripts/run_evaluation.py``) call the same
functions directly, so there is exactly one implementation of every step and
all of it is testable without a Dagster instance.

Layering (handoff §22):

* this module decides *what* happens in a round and *where* its outputs live
  (RoundStore, MLflow run, ``reports/``, ``artifacts/``);
* the scientific packages (``data``, ``evaluation``, ``active_learning``)
  decide *how* (they import neither Dagster, MLflow nor Ray);
* :class:`RayCompute` is the only place here that touches Ray — it delegates
  to ``training.distributed`` (Ray Train + DDP), ``inference.batch`` (Ray
  actors) and ``ray_runtime.tasks`` (Ray tasks / the ``ExperimentSimulator``
  actor). Dagster wraps it in ``RayComputeResource``.

Every returned ``*Info`` record is small, picklable and path-based: the data
itself lives in the durable platform stores, never in the orchestrator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from ml_platform.active_learning.loop import read_selection, select_batch, write_selection
from ml_platform.config import PlatformConfig
from ml_platform.data import (
    ImmutableRoundError,
    RoundSequenceError,
    RoundStore,
    initial_observations,
    make_oracle,
    round_key,
    train_val_split,
    validate_frame,
)
from ml_platform.data.datasets import POOL_FILE
from ml_platform.evaluation import evaluate
from ml_platform.evaluation.evaluator import EvaluationResult, TargetScale
from ml_platform.inference.predictor import Predictor
from ml_platform.logging import get_logger
from ml_platform.tracking import ModelRegistry, Tracker
from ml_platform.training.trainer import TrainResult

log = get_logger(__name__)

MODEL_ROLE_PRODUCTION = "production"
MODEL_ROLE_RESEARCH = "research-candidate"
PREDICTIONS_FILE = "predictions.parquet"
LAB_JOURNAL = "lab_journal/measurements.jsonl"
MLFLOW_UI_ENV = "MLFLOW_UI_URL"
DEFAULT_MLFLOW_UI = "http://localhost:5000"


class RoundNotMeasuredError(FileNotFoundError):
    """The requested round has not been produced by the lab (oracle) yet."""


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class PoolInfo:
    path: Path
    pool_hash: str
    manifest_hash: str
    n_candidates: int
    n_features: int


@dataclass(frozen=True)
class RoundInfo:
    round_id: int
    path: Path
    n_records: int
    dataset_hash: str
    manifest_hash: str
    parent_hashes: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return round_key(self.round_id)


@dataclass(frozen=True)
class TrainingDatasetInfo:
    round_id: int
    dataset_hash: str
    n_rows: int
    rounds: tuple[int, ...]
    rows_per_round: dict[int, int]
    n_train: int
    n_val: int
    validation: dict[str, Any]


@dataclass(frozen=True)
class TrainedModelInfo:
    round_id: int
    run_id: str
    experiment_id: str
    checkpoint_path: Path
    checkpoint_hash: str
    dataset_hash: str
    world_size: int
    worker_pids: tuple[int, ...]
    params_in_sync: bool | None
    val_rmse: float
    metrics: dict[str, float]
    epochs: int
    duration_s: float
    device: str
    ray_train_run: str | None


@dataclass(frozen=True)
class EvaluationInfo:
    round_id: int
    run_id: str
    report_dir: Path
    metrics: dict[str, float]
    gate_passed: bool
    gate_reasons: tuple[str, ...]
    baseline_name: str
    report_markdown: str
    ray_rmse_ci: tuple[float, float, float] | None


@dataclass(frozen=True)
class RegisteredModelInfo:
    round_id: int
    version: str
    stage: str
    run_id: str
    gate_passed: bool
    gate_reasons: tuple[str, ...]
    production_version: str | None
    reused_existing_version: bool


@dataclass(frozen=True)
class PredictionsInfo:
    round_id: int
    path: Path
    model_version: str
    model_role: str
    checkpoint_path: Path
    n_scored: int
    n_excluded_observed: int
    pred_mean_avg: float
    pred_std_avg: float
    stats: dict[str, Any]


@dataclass(frozen=True)
class SelectionInfo:
    round_id: int
    next_round_id: int
    path: Path
    selection_hash: str
    n_selected: int
    model_version: str
    model_role: str
    top_score: float | None
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class NewRoundInfo:
    source_round_id: int
    round: RoundInfo
    noop: bool
    n_physical_measurements: int
    selection_hash: str

    @property
    def round_id(self) -> int:
        return self.round.round_id


# --------------------------------------------------------------------------- compute
class ComputeBackend(Protocol):
    """What the workflow needs from the distributed-compute layer."""

    def train(
        self,
        cfg: PlatformConfig,
        *,
        round_id: int,
        train_frame: pd.DataFrame,
        dataset_hash: str,
        run_id: str | None,
    ) -> TrainResult: ...

    def predict_pool(
        self,
        checkpoint_path: str | os.PathLike[str],
        pool_frame: pd.DataFrame,
        *,
        mc_samples: int,
        seed: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]: ...

    def bootstrap_ci(
        self, y: np.ndarray, yhat: np.ndarray, *, metric: str, n_resamples: int, seed: int
    ) -> tuple[float, float, float]: ...

    def run_experiments(
        self,
        cfg: PlatformConfig,
        pool: pd.DataFrame | Path,
        *,
        round_id: int,
        candidate_ids: list[str],
        journal_path: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]: ...


@dataclass
class RayCompute:
    """`ComputeBackend` on Ray: the stable API that orchestration calls.

    ``address`` None -> ``RAY_ADDRESS`` or a local cluster (see
    ``ray_runtime.cluster.ensure_ray``). Dagster never schedules individual
    workers; it asks for "train with N DDP workers" and Ray decides placement.
    """

    address: str | None = None
    num_workers: int | None = None
    n_inference_actors: int = 2
    bootstrap_tasks: int = 4
    # caller-side retries of the idempotent lab measurement (see run_experiments)
    measure_attempts: int = 3
    measure_retry_backoff_s: float = 0.5

    def ensure(self) -> dict[str, Any]:
        from ml_platform.ray_runtime.cluster import ensure_ray

        return ensure_ray(self.address)

    def train(
        self,
        cfg: PlatformConfig,
        *,
        round_id: int,
        train_frame: pd.DataFrame,
        dataset_hash: str,
        run_id: str | None,
    ) -> TrainResult:
        from ml_platform.training.distributed import train_distributed

        self.ensure()
        return train_distributed(
            cfg,
            round_id=round_id,
            train_frame=train_frame,
            dataset_hash=dataset_hash,
            num_workers=self.num_workers,
            run_id=run_id,
        )

    def predict_pool(
        self,
        checkpoint_path: str | os.PathLike[str],
        pool_frame: pd.DataFrame,
        *,
        mc_samples: int,
        seed: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        from ml_platform.inference.batch import predict_pool

        self.ensure()
        out = predict_pool(
            checkpoint_path,
            pool_frame,
            n_actors=self.n_inference_actors,
            mc_samples=mc_samples,
            seed=seed,
            return_stats=True,
        )
        assert isinstance(out, tuple)
        return out

    def bootstrap_ci(
        self, y: np.ndarray, yhat: np.ndarray, *, metric: str, n_resamples: int, seed: int
    ) -> tuple[float, float, float]:
        from ml_platform.ray_runtime.tasks import parallel_bootstrap_ci

        self.ensure()
        return parallel_bootstrap_ci(
            y, yhat, metric=metric, n_resamples=n_resamples, n_tasks=self.bootstrap_tasks, seed=seed
        )

    def run_experiments(
        self,
        cfg: PlatformConfig,
        pool: pd.DataFrame | Path,
        *,
        round_id: int,
        candidate_ids: list[str],
        journal_path: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Measure candidates on the `ExperimentSimulator` actor (the stand-in "lab").

        The actor journals every measurement to ``journal_path``; asking again
        for ``(round_id, candidate_id)`` returns the recorded value instead of
        running the experiment a second time.

        **Caller-side retry.** The actor has ``max_restarts=1`` but
        ``max_task_retries=0``: Ray restarts a crashed lab process but never
        re-sends the in-flight ``measure`` itself, so the first call after a
        crash fails with ``ActorUnavailableError`` (restarting) or
        ``ActorDiedError`` (restarts exhausted). Retrying *here* is the safe,
        deliberate exception to "never retry an experiment", because
        ``measure`` is idempotent per ``(round_id, candidate_id)`` through the
        journal: whatever was measured and journaled before the crash is
        replayed, never re-measured, and only candidates with no recorded
        result are measured (for the first recorded time). On
        ``ActorUnavailableError`` we retry the same handle; on
        ``ActorDiedError`` we start a fresh simulator, which reloads the same
        journal. (A crash between the oracle call and the journal append loses
        that unrecorded result; the simulator's seeded oracle then reproduces
        it — a real lab would need a two-phase "reserve, then record" protocol.)
        """
        import time

        import ray
        from ray.exceptions import ActorDiedError, ActorUnavailableError

        from ml_platform.ray_runtime.tasks import start_experiment_simulator

        self.ensure()
        actor = start_experiment_simulator(cfg, pool, journal_path=journal_path)
        restarts: list[str] = []
        try:
            for attempt in range(1, self.measure_attempts + 1):
                try:
                    frame = ray.get(actor.measure.remote(round_id, candidate_ids, cfg.seed))
                    stats = ray.get(actor.stats.remote())
                    break
                except (ActorUnavailableError, ActorDiedError) as exc:
                    if attempt == self.measure_attempts:
                        raise
                    restarts.append(type(exc).__name__)
                    log.warning(
                        "pipeline.lab_unavailable_retrying",
                        round_id=round_id,
                        attempt=attempt,
                        error=type(exc).__name__,
                    )
                    time.sleep(self.measure_retry_backoff_s * attempt)
                    if isinstance(exc, ActorDiedError):
                        ray.kill(actor)
                        actor = start_experiment_simulator(cfg, pool, journal_path=journal_path)
        finally:
            ray.kill(actor)
        return frame, {**stats, "lab_retries": restarts}


# --------------------------------------------------------------------------- helpers
def round_store(cfg: PlatformConfig) -> RoundStore:
    return RoundStore(cfg.paths.resolved().data_dir)


def write_config(cfg: PlatformConfig, path: str | os.PathLike[str]) -> Path:
    """Persist a config as YAML loadable by ``load_config`` (used to hand configs to Dagster)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return p


def mlflow_run_url(experiment_id: str, run_id: str) -> str:
    base = os.environ.get(MLFLOW_UI_ENV, DEFAULT_MLFLOW_UI).rstrip("/")
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def mlflow_model_url(name: str, version: str) -> str:
    base = os.environ.get(MLFLOW_UI_ENV, DEFAULT_MLFLOW_UI).rstrip("/")
    return f"{base}/#/models/{name}/versions/{version}"


def _round_info(store: RoundStore, round_id: int) -> RoundInfo:
    m = store.read_manifest(round_id)
    return RoundInfo(
        round_id=round_id,
        path=store.round_dir(round_id),
        n_records=m.n_records,
        dataset_hash=m.dataset_hash,
        manifest_hash=m.manifest_hash,
        parent_hashes=tuple(m.parent_hashes),
        provenance=dict(m.provenance),
    )


# --------------------------------------------------------------------------- steps
def ensure_candidate_pool(cfg: PlatformConfig, store: RoundStore) -> PoolInfo:
    """Generate the (deterministic) candidate pool once; write-once in the RoundStore."""
    if not store.has_pool():
        from ml_platform.data import generate_candidate_pool

        pool = generate_candidate_pool(cfg)
        store.write_pool(
            pool,
            provenance={"generator": "generate_candidate_pool", "pool_seed": cfg.data.pool_seed},
        )
        log.info("pipeline.pool_written", n_candidates=len(pool))
    m = store.pool_manifest()
    return PoolInfo(
        path=store.pool_dir,
        pool_hash=m["pool_hash"],
        manifest_hash=m["manifest_hash"],
        n_candidates=int(m["n_candidates"]),
        n_features=int(m["n_features"]),
    )


def observe_round(cfg: PlatformConfig, store: RoundStore, round_id: int) -> RoundInfo:
    """The immutable observations of ``round_id``.

    Round 0 is the random initial design and is created here (idempotently).
    Later rounds are *produced by the lab* (``run_experiments`` of round r-1,
    or an external process dropping a round into the store); this step only
    reads and verifies them.
    """
    if not store.exists(round_id):
        if round_id != 0:
            raise RoundNotMeasuredError(
                f"{round_key(round_id)} has not been measured yet; materialize "
                f"new_experimental_results for {round_key(round_id - 1)} first"
            )
        pool = store.read_pool()
        records = initial_observations(
            pool, make_oracle(cfg), cfg.data.initial_observations, seed=cfg.seed
        )
        store.write_round(
            0,
            records,
            provenance={"source": "initial_random_design", "seed": cfg.seed},
        )
        log.info("pipeline.round_written", round_id=0, n_records=len(records))
    return _round_info(store, round_id)


def build_training_dataset(
    cfg: PlatformConfig, store: RoundStore, round_id: int
) -> TrainingDatasetInfo:
    """Training dataset = union(round_000..round_r), validated; identified by its hash chain."""
    store.verify_chain()
    frame = store.training_frame(round_id)
    report = validate_frame(
        frame,
        kind="observations",
        n_features=cfg.data.n_features,
        feature_bounds=tuple(cfg.active_learning.feature_bounds),  # type: ignore[arg-type]
    )
    report.raise_if_failed()
    tr, va = train_val_split(frame, cfg.training.val_fraction, cfg.seed, cfg.training.val_strategy)
    counts = frame.groupby("round_id").size()
    return TrainingDatasetInfo(
        round_id=round_id,
        dataset_hash=store.dataset_hash(round_id),
        n_rows=len(frame),
        rounds=tuple(int(r) for r in counts.index),
        rows_per_round={
            int(r): int(n) for r, n in zip(counts.index, counts.to_numpy(), strict=True)
        },
        n_train=len(tr),
        n_val=len(va),
        validation=report.to_dict(),
    )


def train_model(
    cfg: PlatformConfig,
    store: RoundStore,
    tracker: Tracker,
    compute: ComputeBackend,
    dataset: TrainingDatasetInfo,
    *,
    tags: dict[str, Any] | None = None,
) -> TrainedModelInfo:
    """Distributed training on the round union, tracked as one MLflow run."""
    r = dataset.round_id
    frame = store.training_frame(r)
    if store.dataset_hash(r) != dataset.dataset_hash:  # pragma: no cover - immutability guard
        raise ImmutableRoundError(f"dataset for {round_key(r)} changed since it was built")
    run_tags = {
        "round_id": r,
        "dataset_id": dataset.dataset_hash,
        "stage": "training",
        **(tags or {}),
    }
    with tracker.start_run(f"train_{round_key(r)}", tags=run_tags) as run_id:
        result = compute.train(
            cfg,
            round_id=r,
            train_frame=frame,
            dataset_hash=dataset.dataset_hash,
            run_id=run_id,
        )
        tracker.log_train_result(
            result, cfg=cfg, dataset_frame=frame, dataset_name=f"rounds_000-{r:03d}"
        )
    info = result.device_info.get("distributed", {}) or {}
    return TrainedModelInfo(
        round_id=r,
        run_id=run_id,
        experiment_id=tracker.experiment_id,
        checkpoint_path=Path(result.checkpoint_path),
        checkpoint_hash=result.checkpoint_hash or "",
        dataset_hash=result.dataset_hash or dataset.dataset_hash,
        world_size=result.world_size,
        worker_pids=tuple(int(p) for p in info.get("pids", [])),
        params_in_sync=info.get("params_in_sync"),
        val_rmse=float(result.metrics.get("val_rmse", float("nan"))),
        metrics={k: float(v) for k, v in result.metrics.items()},
        epochs=result.epochs_completed,
        duration_s=float(result.duration_s),
        device=str(result.device),
        ray_train_run=info.get("ray_train_run"),
    )


def evaluate_checkpoint(
    cfg: PlatformConfig,
    store: RoundStore,
    registry: ModelRegistry,
    checkpoint_path: str | os.PathLike[str],
    round_id: int,
    *,
    candidate_run_id: str | None = None,
) -> EvaluationResult:
    """Evaluate a checkpoint on the validation split of union(round_000..round_id).

    Baseline: the current production model's predictions on the *same* rows
    (paired comparison), unless the candidate *is* the production model
    (same MLflow run); otherwise a train-mean predictor.

    Standardized metrics (and the ``max_rmse`` gate) use the target mean/std of
    the *training split of this round's dataset* — a property of the data,
    identical for the candidate and the baseline, not the candidate's own
    checkpoint statistics.
    """
    frame = store.training_frame(round_id)
    train, val = train_val_split(
        frame, cfg.training.val_fraction, cfg.seed, cfg.training.val_strategy
    )
    scale = TargetScale.from_targets(
        train["response"],
        source=(
            f"training split of union(round_000..{round_key(round_id)}) "
            f"[dataset {store.dataset_hash(round_id)[:12]}]"
        ),
    )
    predictor = Predictor.from_checkpoint(checkpoint_path)
    baseline, baseline_name = None, None
    prod = registry.production_version()
    if prod is not None:
        prod_run = registry.version_tags(prod.version).get("run_id")
        same = (candidate_run_id is not None and prod_run == candidate_run_id) or (
            Path(prod.checkpoint_path).resolve() == Path(checkpoint_path).resolve()
        )
        if not same:
            baseline = Predictor.from_checkpoint(prod.checkpoint_path).predict(val)
            baseline_name = f"production_v{prod.version}"
    return evaluate(
        predictor,
        val,
        baseline,
        cfg,
        baseline_name=baseline_name,
        seed=cfg.seed,
        target_scale=scale,
    )


def evaluate_model(
    cfg: PlatformConfig,
    store: RoundStore,
    tracker: Tracker,
    registry: ModelRegistry,
    compute: ComputeBackend | None,
    trained: TrainedModelInfo,
) -> EvaluationInfo:
    """Paired evaluation vs the production model (or a mean baseline), gates, report."""
    r = trained.round_id
    evaluation = evaluate_checkpoint(
        cfg, store, registry, trained.checkpoint_path, r, candidate_run_id=trained.run_id
    )
    out_dir = cfg.paths.resolved().reports_dir / round_key(r) / trained.run_id
    ray_ci: tuple[float, float, float] | None = None
    with tracker.start_run(run_id=trained.run_id):
        tracker.log_evaluation(evaluation, out_dir)
        if compute is not None:
            # Same CI, computed by parallel Ray tasks (demonstrates task fan-out in evaluation).
            p = evaluation.predictions
            ray_ci = compute.bootstrap_ci(
                p["y_true_std_units"].to_numpy(),
                p["y_pred_std_units"].to_numpy(),
                metric="rmse",
                n_resamples=cfg.evaluation.bootstrap_samples,
                seed=cfg.seed,
            )
            tracker.log_metrics(
                {"eval_rmse_ci_lo_ray": ray_ci[1], "eval_rmse_ci_hi_ray": ray_ci[2]}
            )
    report_md = (out_dir / "report.md").read_text() if (out_dir / "report.md").exists() else ""
    return EvaluationInfo(
        round_id=r,
        run_id=trained.run_id,
        report_dir=out_dir,
        metrics={k: float(v) for k, v in evaluation.metrics.items()},
        gate_passed=bool(evaluation.gate.passed),
        gate_reasons=tuple(evaluation.gate.reasons),
        baseline_name=evaluation.baseline_name,
        report_markdown=report_md,
        ray_rmse_ci=ray_ci,
    )


def register_model(
    registry: ModelRegistry, trained: TrainedModelInfo, evaluation: EvaluationInfo
) -> RegisteredModelInfo:
    """Register the round's model as a candidate; promote only if the gates passed.

    Idempotent per training run: re-materializing reuses the version already
    registered for ``trained.run_id`` instead of creating a duplicate.
    """
    existing = [v for v in registry.list_versions() if v["run_id"] == trained.run_id]
    if existing:
        version, reused = str(existing[-1]["version"]), True
    else:
        version = registry.register(
            trained.run_id,
            trained.checkpoint_path,
            tags={
                "round_id": trained.round_id,
                "dataset_hash": trained.dataset_hash,
                "eval_rmse": round(evaluation.metrics.get("rmse", float("nan")), 6),
            },
        )
        reused = False
    stage = registry.promote_if_passed(
        version, {"passed": evaluation.gate_passed, "reasons": list(evaluation.gate_reasons)}
    )
    prod = registry.production_version()
    return RegisteredModelInfo(
        round_id=trained.round_id,
        version=version,
        stage=stage,
        run_id=trained.run_id,
        gate_passed=evaluation.gate_passed,
        gate_reasons=evaluation.gate_reasons,
        production_version=prod.version if prod else None,
        reused_existing_version=reused,
    )


def choose_scoring_model(
    registry: ModelRegistry, trained: TrainedModelInfo, registered: RegisteredModelInfo
) -> tuple[Path, str, str]:
    """Model-selection policy for decisions: production if any, else this round's model.

    Returns ``(checkpoint_path, model_version, model_role)``. A non-production
    model may drive *research* decisions (which experiments to run next) and is
    flagged ``research-candidate``; it is never served.
    """
    prod = registry.production_version()
    if prod is not None:
        return prod.checkpoint_path, prod.version, MODEL_ROLE_PRODUCTION
    return trained.checkpoint_path, registered.version, MODEL_ROLE_RESEARCH


def score_candidates(
    cfg: PlatformConfig,
    store: RoundStore,
    registry: ModelRegistry,
    compute: ComputeBackend,
    trained: TrainedModelInfo,
    registered: RegisteredModelInfo,
) -> PredictionsInfo:
    """Distributed MC-dropout scoring of the not-yet-observed candidate pool."""
    r = trained.round_id
    ckpt, version, role = choose_scoring_model(registry, trained, registered)
    pool = store.read_pool()
    observed = store.observed_ids(r)
    unobserved = pool[~pool["candidate_id"].astype(str).isin(observed)].reset_index(drop=True)
    preds, stats = compute.predict_pool(
        ckpt, unobserved, mc_samples=cfg.evaluation.mc_samples, seed=cfg.seed + r
    )
    out_dir = cfg.paths.resolved().artifacts_dir / "predictions" / round_key(r) / f"v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PREDICTIONS_FILE
    tmp = out_dir / f".{PREDICTIONS_FILE}.tmp"
    preds.to_parquet(tmp, index=False)
    tmp.replace(path)
    clean_stats = {k: v for k, v in stats.items() if k != "actors"}
    clean_stats["actor_pids"] = sorted({a.get("pid") for a in stats.get("actors", []) if a})
    (out_dir / "stats.json").write_text(
        json.dumps(
            {"model_version": version, "model_role": role, **clean_stats}, indent=2, default=str
        )
    )
    return PredictionsInfo(
        round_id=r,
        path=path,
        model_version=version,
        model_role=role,
        checkpoint_path=Path(ckpt),
        n_scored=len(preds),
        n_excluded_observed=len(pool) - len(unobserved),
        pred_mean_avg=float(preds["pred_mean"].mean()) if len(preds) else float("nan"),
        pred_std_avg=float(preds["pred_std"].mean()) if len(preds) else float("nan"),
        stats=clean_stats,
    )


def select_experiments(
    cfg: PlatformConfig, store: RoundStore, predictions: PredictionsInfo
) -> SelectionInfo:
    """Constraints + acquisition -> the next experimental batch (versioned by its hash)."""
    r = predictions.round_id
    nxt = r + 1
    preds = pd.read_parquet(predictions.path)
    result = select_batch(
        store.read_pool(),
        preds,
        store.observed_ids(r),
        cfg,
        round_id=nxt,
        model_version=predictions.model_version,
    )
    out = cfg.paths.resolved().data_dir / "selections" / round_key(nxt) / result.selection_hash[:12]
    write_selection(
        result,
        out,
        extra_metadata={
            "model_role": predictions.model_role,
            "selected_in_round": r,
            "predictions_path": str(predictions.path),
        },
    )
    score = result.summary.get("score") or {}
    return SelectionInfo(
        round_id=r,
        next_round_id=nxt,
        path=out,
        selection_hash=result.selection_hash,
        n_selected=len(result.selected),
        model_version=predictions.model_version,
        model_role=predictions.model_role,
        top_score=float(score["max"]) if score else None,
        candidate_ids=tuple(result.candidate_ids),
    )


def run_experiments(
    cfg: PlatformConfig,
    store: RoundStore,
    compute: ComputeBackend,
    selection: SelectionInfo,
) -> NewRoundInfo:
    """Run the selected experiments and persist them as the next immutable round.

    This is the *physical experiment* step, so it is at-most-once by design:

    * if the next round already exists for exactly these candidates, nothing
      is measured again (idempotent no-op);
    * if it exists for a *different* selection, it refuses — a new selection
      after the lab has run means a new round, never an overwrite;
    * measurements go through a journal, so a crash between "measured" and
      "round written" replays the recorded values instead of re-measuring.
    """
    nxt = selection.next_round_id
    selected, _manifest = read_selection(selection.path)
    ids = selected["candidate_id"].astype(str).tolist()
    if store.exists(nxt):
        existing = set(store.read_round(nxt)["experiment_id"].astype(str))
        if existing == set(ids):
            log.info("pipeline.round_exists_noop", round_id=nxt)
            return NewRoundInfo(
                source_round_id=selection.round_id,
                round=_round_info(store, nxt),
                noop=True,
                n_physical_measurements=0,
                selection_hash=selection.selection_hash,
            )
        raise ImmutableRoundError(
            f"{round_key(nxt)} was already measured for a different selection "
            f"(this selection: {selection.selection_hash[:12]}); experiments are not re-run"
        )
    latest = store.latest_round_id()
    if latest != selection.round_id:
        raise RoundSequenceError(
            f"selection was made in {round_key(selection.round_id)} but the latest round is "
            f"{round_key(latest) if latest is not None else 'none'}"
        )
    journal = cfg.paths.resolved().data_dir / LAB_JOURNAL
    frame, stats = compute.run_experiments(
        cfg,
        store.pool_dir / POOL_FILE,  # durable catalogue: a restarted lab re-reads it
        round_id=nxt,
        candidate_ids=ids,
        journal_path=journal,
    )
    store.write_round(
        nxt,
        frame,
        provenance={
            "source": "active_learning",
            "selection_hash": selection.selection_hash,
            "selected_in_round": selection.round_id,
            "model_version": selection.model_version,
            "model_role": selection.model_role,
            "lab": "ExperimentSimulator",
        },
    )
    return NewRoundInfo(
        source_round_id=selection.round_id,
        round=_round_info(store, nxt),
        noop=False,
        n_physical_measurements=int(stats.get("n_physical_measurements", len(ids))),
        selection_hash=selection.selection_hash,
    )
