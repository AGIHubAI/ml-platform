"""Dagster software-defined assets for the closed loop (handoff §8).

::

    candidate_pool → observed_experiments → training_dataset → trained_model
      → evaluation_report → registered_model → candidate_predictions
      → selected_experiments → new_experimental_results  (writes round r+1)

Everything except ``candidate_pool`` is partitioned by experimental round
(dynamic partitions ``round_000``, ``round_001``, ...). The loop closes
*across* partitions: ``new_experimental_results[round_r]`` writes
``data/rounds/round_{r+1}`` and registers that partition; the
``new_round_sensor`` (or ``scripts/run_closed_loop.py``) then materializes
``observed_experiments[round_{r+1}]`` and the rest of the chain. A static
asset edge back to ``observed_experiments`` would be a cycle, so the RoundStore
is the hand-off point — exactly as it would be for a real lab.

**These assets are glue only.** Each body resolves resources, binds
correlation ids for structured logs, calls one step function from
:mod:`ml_platform.orchestration.pipeline` and converts the result into
materialization metadata. Scientific logic lives in ``data`` / ``models`` /
``training`` / ``evaluation`` / ``active_learning``; distributed execution
lives behind `RayComputeResource`.

Asset values are small path/hash records pickled by the IO manager; the data
itself stays in the durable platform stores (RoundStore, MLflow, ``reports/``,
``artifacts/``).
"""

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd
from dagster import (
    AssetExecutionContext,
    Backoff,
    Jitter,
    MetadataValue,
    Output,
    RetryPolicy,
    asset,
)

from ml_platform.data import round_key
from ml_platform.logging import bound_ids, get_logger
from ml_platform.orchestration import pipeline as P
from ml_platform.orchestration.partitions import (
    ROUNDS_PARTITION_NAME,
    partition_round_id,
    rounds_partitions,
)
from ml_platform.orchestration.resources import (
    PlatformConfigResource,
    RayComputeResource,
    RoundStoreResource,
    TrackingResource,
)

log = get_logger(__name__)

GROUP = "closed_loop"

# Computation is deterministic and side-effect free (or idempotent), so a
# transient failure (Ray worker lost, object-store pressure, flaky disk) is
# simply retried with exponential backoff.
COMPUTE_RETRY = RetryPolicy(
    max_retries=2, delay=10, backoff=Backoff.EXPONENTIAL, jitter=Jitter.PLUS_MINUS
)
LIGHT_RETRY = RetryPolicy(max_retries=2, delay=2, backoff=Backoff.EXPONENTIAL)


@contextmanager
def _ids(context: AssetExecutionContext, **ids: Any) -> Iterator[Any]:
    """Bind Dagster run id + round/dataset/model ids into every log line of the step."""
    round_id = partition_round_id(context.partition_key) if context.has_partition_key else None
    with bound_ids(run_id=context.run_id, round_id=round_id, **ids):
        yield get_logger(f"asset.{context.asset_key.to_user_string()}")


def _f(x: float | None) -> MetadataValue[Any]:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return MetadataValue.text("n/a")
    return MetadataValue.float(float(x))


def _md_list(items: tuple[str, ...] | list[str]) -> MetadataValue[Any]:
    return MetadataValue.md("\n".join(f"- {i}" for i in items) or "- (none)")


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


# --------------------------------------------------------------------------- data
@asset(
    group_name=GROUP,
    kinds={"python", "parquet"},
    description="Candidate experiments (features + cost); deterministic, write-once.",
)
def candidate_pool(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    round_store: RoundStoreResource,
) -> Output[P.PoolInfo]:
    with _ids(context) as alog:
        info = P.ensure_candidate_pool(platform_config.load(), round_store.store())
        alog.info("asset.candidate_pool", n_candidates=info.n_candidates)
    return Output(
        info,
        metadata={
            "dagster/row_count": MetadataValue.int(info.n_candidates),
            "n_features": MetadataValue.int(info.n_features),
            "pool_hash": MetadataValue.text(info.pool_hash),
            "manifest_hash": MetadataValue.text(info.manifest_hash),
            "path": MetadataValue.path(str(info.path)),
        },
    )


@asset(
    partitions_def=rounds_partitions,
    group_name=GROUP,
    kinds={"python", "parquet"},
    retry_policy=LIGHT_RETRY,
    description=(
        "Immutable observations of one experimental round. round_000 is the random "
        "initial design; later rounds are written by new_experimental_results."
    ),
)
def observed_experiments(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    round_store: RoundStoreResource,
    candidate_pool: P.PoolInfo,
) -> Output[P.RoundInfo]:
    r = partition_round_id(context.partition_key)
    with _ids(context) as alog:
        info = P.observe_round(platform_config.load(), round_store.store(), r)
        alog.info("asset.observed_experiments", n_records=info.n_records)
    return Output(
        info,
        metadata={
            "dagster/row_count": MetadataValue.int(info.n_records),
            "round": MetadataValue.text(info.key),
            "content_hash": MetadataValue.text(info.dataset_hash),
            "manifest_hash": MetadataValue.text(info.manifest_hash),
            "parent_hashes": MetadataValue.text(", ".join(info.parent_hashes) or "(root)"),
            "pool_hash": MetadataValue.text(candidate_pool.pool_hash),
            "provenance": MetadataValue.json(info.provenance),
            "path": MetadataValue.path(str(info.path)),
        },
    )


@asset(
    partitions_def=rounds_partitions,
    group_name=GROUP,
    kinds={"python", "pandas"},
    retry_policy=LIGHT_RETRY,
    description="Training dataset = union(round_000..round_r), validated; id = hash chain.",
)
def training_dataset(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    round_store: RoundStoreResource,
    observed_experiments: P.RoundInfo,
) -> Output[P.TrainingDatasetInfo]:
    r = partition_round_id(context.partition_key)
    with _ids(context, dataset_id=observed_experiments.manifest_hash) as alog:
        info = P.build_training_dataset(platform_config.load(), round_store.store(), r)
        alog.info("asset.training_dataset", n_rows=info.n_rows)
    return Output(
        info,
        metadata={
            "dagster/row_count": MetadataValue.int(info.n_rows),
            "dataset_hash": MetadataValue.text(info.dataset_hash),
            "rounds": MetadataValue.text(", ".join(round_key(x) for x in info.rounds)),
            "rows_per_round": MetadataValue.json(
                {round_key(k): v for k, v in info.rows_per_round.items()}
            ),
            "n_train": MetadataValue.int(info.n_train),
            "n_val": MetadataValue.int(info.n_val),
            "validation_ok": MetadataValue.bool(bool(info.validation.get("ok"))),
            "validation_report": MetadataValue.json(info.validation),
        },
    )


# --------------------------------------------------------------------------- model
@asset(
    partitions_def=rounds_partitions,
    group_name=GROUP,
    kinds={"ray", "pytorch", "mlflow"},
    retry_policy=COMPUTE_RETRY,
    description="Surrogate trained with Ray Train + PyTorch DDP; tracked as an MLflow run.",
)
def trained_model(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    round_store: RoundStoreResource,
    tracking: TrackingResource,
    ray_compute: RayComputeResource,
    training_dataset: P.TrainingDatasetInfo,
) -> Output[P.TrainedModelInfo]:
    with _ids(context, dataset_id=training_dataset.dataset_hash) as alog:
        info = P.train_model(
            platform_config.load(),
            round_store.store(),
            tracking.tracker(),
            ray_compute.client(),
            training_dataset,
            tags={"dagster_run_id": context.run_id, "dagster_partition": context.partition_key},
        )
        alog.info("asset.trained_model", mlflow_run_id=info.run_id, val_rmse=info.val_rmse)
    return Output(
        info,
        metadata={
            "mlflow_run_id": MetadataValue.text(info.run_id),
            "mlflow_run": MetadataValue.url(P.mlflow_run_url(info.experiment_id, info.run_id)),
            "checkpoint": MetadataValue.path(str(info.checkpoint_path)),
            "checkpoint_hash": MetadataValue.text(info.checkpoint_hash),
            "dataset_hash": MetadataValue.text(info.dataset_hash),
            "world_size": MetadataValue.int(info.world_size),
            "worker_pids": MetadataValue.text(", ".join(map(str, info.worker_pids))),
            "params_in_sync": MetadataValue.text(str(info.params_in_sync)),
            "val_rmse": _f(info.val_rmse),
            "epochs": MetadataValue.int(info.epochs),
            "train_duration_s": _f(round(info.duration_s, 2)),
            "device": MetadataValue.text(info.device),
        },
    )


@asset(
    partitions_def=rounds_partitions,
    group_name=GROUP,
    kinds={"python", "ray", "mlflow"},
    retry_policy=COMPUTE_RETRY,
    description="Paired evaluation vs the production model (or mean baseline) + promotion gates.",
)
def evaluation_report(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    round_store: RoundStoreResource,
    tracking: TrackingResource,
    ray_compute: RayComputeResource,
    trained_model: P.TrainedModelInfo,
) -> Output[P.EvaluationInfo]:
    with _ids(context, dataset_id=trained_model.dataset_hash) as alog:
        tracker = tracking.tracker()
        info = P.evaluate_model(
            platform_config.load(),
            round_store.store(),
            tracker,
            tracking.registry(tracker),
            ray_compute.client(),
            trained_model,
        )
        alog.info(
            "asset.evaluation_report", gate_passed=info.gate_passed, rmse=info.metrics.get("rmse")
        )
    m = info.metrics
    meta: dict[str, Any] = {
        "gate_passed": MetadataValue.bool(info.gate_passed),
        "gate_reasons": _md_list(info.gate_reasons),
        "baseline": MetadataValue.text(info.baseline_name),
        "rmse_std_units": _f(m.get("rmse")),
        "rmse_ci95": MetadataValue.text(
            f"[{m.get('rmse_ci_lo', math.nan):.4f}, {m.get('rmse_ci_hi', math.nan):.4f}]"
        ),
        "baseline_rmse": _f(m.get("baseline_rmse")),
        "improvement_vs_baseline": _f(m.get("improvement_vs_baseline")),
        "r2": _f(m.get("r2")),
        "coverage_95": _f(m.get("coverage_95")),
        "n_eval": MetadataValue.int(int(m.get("n_eval", 0))),
        "report_dir": MetadataValue.path(str(info.report_dir)),
        "mlflow_run_id": MetadataValue.text(info.run_id),
        "mlflow_run": MetadataValue.url(P.mlflow_run_url(trained_model.experiment_id, info.run_id)),
        "report": MetadataValue.md(info.report_markdown[:6000] or "(empty)"),
    }
    if info.ray_rmse_ci is not None:
        meta["rmse_ci95_ray_tasks"] = MetadataValue.text(
            f"[{info.ray_rmse_ci[1]:.4f}, {info.ray_rmse_ci[2]:.4f}]"
        )
    return Output(info, metadata=meta)


@asset(
    partitions_def=rounds_partitions,
    group_name=GROUP,
    kinds={"mlflow"},
    retry_policy=LIGHT_RETRY,
    description=(
        "Model registered as a candidate; promoted to production only if the "
        "evaluation gates passed (training finishing is never enough)."
    ),
)
def registered_model(
    context: AssetExecutionContext,
    tracking: TrackingResource,
    trained_model: P.TrainedModelInfo,
    evaluation_report: P.EvaluationInfo,
) -> Output[P.RegisteredModelInfo]:
    with _ids(context, dataset_id=trained_model.dataset_hash) as alog:
        registry = tracking.registry()
        info = P.register_model(registry, trained_model, evaluation_report)
        alog.info(
            "asset.registered_model",
            model_version=info.version,
            stage=info.stage,
            gate_passed=info.gate_passed,
        )
    return Output(
        info,
        metadata={
            "model_version": MetadataValue.text(info.version),
            "lifecycle_stage": MetadataValue.text(info.stage),
            "gate_passed": MetadataValue.bool(info.gate_passed),
            "gate_reasons": _md_list(info.gate_reasons),
            "production_version": MetadataValue.text(info.production_version or "none"),
            "reused_existing_version": MetadataValue.bool(info.reused_existing_version),
            "mlflow_model": MetadataValue.url(P.mlflow_model_url(registry.name, info.version)),
            "mlflow_run_id": MetadataValue.text(info.run_id),
        },
    )


# --------------------------------------------------------------------------- decision
@asset(
    partitions_def=rounds_partitions,
    group_name=GROUP,
    kinds={"ray", "pytorch", "parquet"},
    retry_policy=COMPUTE_RETRY,
    deps=[candidate_pool],
    description=(
        "MC-dropout predictions for every not-yet-observed candidate, scored by Ray "
        "actors. Uses the production model, else this round's model (research-candidate)."
    ),
)
def candidate_predictions(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    round_store: RoundStoreResource,
    tracking: TrackingResource,
    ray_compute: RayComputeResource,
    trained_model: P.TrainedModelInfo,
    registered_model: P.RegisteredModelInfo,
) -> Output[P.PredictionsInfo]:
    with _ids(context, dataset_id=trained_model.dataset_hash) as alog:
        info = P.score_candidates(
            platform_config.load(),
            round_store.store(),
            tracking.registry(),
            ray_compute.client(),
            trained_model,
            registered_model,
        )
        alog.info(
            "asset.candidate_predictions",
            model_version=info.model_version,
            model_role=info.model_role,
            n_scored=info.n_scored,
        )
    s = info.stats
    return Output(
        info,
        metadata={
            "dagster/row_count": MetadataValue.int(info.n_scored),
            "model_version": MetadataValue.text(info.model_version),
            "model_role": MetadataValue.text(info.model_role),
            "n_excluded_observed": MetadataValue.int(info.n_excluded_observed),
            "pred_mean_avg": _f(info.pred_mean_avg),
            "pred_std_avg": _f(info.pred_std_avg),
            "n_actors": MetadataValue.int(int(s.get("n_actors", 0))),
            "n_shards": MetadataValue.int(int(s.get("n_shards", 0))),
            "actor_pids": MetadataValue.text(", ".join(map(str, s.get("actor_pids", [])))),
            "rows_per_s": _f(s.get("rows_per_s")),
            "path": MetadataValue.path(str(info.path)),
            "checkpoint": MetadataValue.path(str(info.checkpoint_path)),
        },
    )


@asset(
    partitions_def=rounds_partitions,
    group_name=GROUP,
    kinds={"python", "parquet"},
    retry_policy=LIGHT_RETRY,
    description="Next experimental batch (constraints + UCB acquisition), versioned by selection_hash.",
)
def selected_experiments(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    round_store: RoundStoreResource,
    candidate_predictions: P.PredictionsInfo,
) -> Output[P.SelectionInfo]:
    with _ids(context, model_version=candidate_predictions.model_version) as alog:
        info = P.select_experiments(
            platform_config.load(), round_store.store(), candidate_predictions
        )
        alog.info(
            "asset.selected_experiments",
            selection_hash=info.selection_hash[:12],
            n_selected=info.n_selected,
        )
    preview = pd.read_parquet(info.path / "selected.parquet").head(10)
    cols = [c for c in ("rank", "candidate_id", "pred_mean", "pred_std", "score") if c in preview]
    return Output(
        info,
        metadata={
            "dagster/row_count": MetadataValue.int(info.n_selected),
            "selection_hash": MetadataValue.text(info.selection_hash),
            "for_round": MetadataValue.text(round_key(info.next_round_id)),
            "model_version": MetadataValue.text(info.model_version),
            "model_role": MetadataValue.text(info.model_role),
            "top_score": _f(info.top_score),
            "path": MetadataValue.path(str(info.path)),
            "top_10": MetadataValue.md(_md_table(preview[cols].round(4))),
        },
    )


@asset(
    partitions_def=rounds_partitions,
    group_name=GROUP,
    kinds={"ray", "parquet"},
    # Deliberately NO retry_policy: this step stands for a *physical experiment*
    # (money, time, irreproducible noise). A failure must be inspected by a
    # human, not blindly re-run. Safety instead comes from idempotency: the
    # ExperimentSimulator journals every measurement and the RoundStore is
    # write-once, so a manual re-materialization with the same selection is a
    # no-op, and one with a different selection is refused.
    description=(
        "Runs the selected experiments on the synthetic oracle (ExperimentSimulator actor) "
        "and writes them as the next immutable round; registers that round's partition."
    ),
)
def new_experimental_results(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    round_store: RoundStoreResource,
    ray_compute: RayComputeResource,
    selected_experiments: P.SelectionInfo,
) -> Output[P.NewRoundInfo]:
    with _ids(context, model_version=selected_experiments.model_version) as alog:
        info = P.run_experiments(
            platform_config.load(),
            round_store.store(),
            ray_compute.client(),
            selected_experiments,
        )
        new_key = info.round.key
        if not context.instance.has_dynamic_partition(ROUNDS_PARTITION_NAME, new_key):
            context.instance.add_dynamic_partitions(ROUNDS_PARTITION_NAME, [new_key])
        alog.info(
            "asset.new_experimental_results",
            new_round=new_key,
            noop=info.noop,
            n_physical_measurements=info.n_physical_measurements,
        )
    return Output(
        info,
        metadata={
            "new_round": MetadataValue.text(new_key),
            "dagster/row_count": MetadataValue.int(info.round.n_records),
            "content_hash": MetadataValue.text(info.round.dataset_hash),
            "manifest_hash": MetadataValue.text(info.round.manifest_hash),
            "parent_hashes": MetadataValue.text(", ".join(info.round.parent_hashes)),
            "selection_hash": MetadataValue.text(info.selection_hash),
            "idempotent_noop": MetadataValue.bool(info.noop),
            "n_physical_measurements": MetadataValue.int(info.n_physical_measurements),
            "provenance": MetadataValue.json(
                json.loads(json.dumps(info.round.provenance, default=str))
            ),
            "path": MetadataValue.path(str(info.round.path)),
        },
    )


ROUND_ASSETS = [
    observed_experiments,
    training_dataset,
    trained_model,
    evaluation_report,
    registered_model,
    candidate_predictions,
    selected_experiments,
    new_experimental_results,
]
ALL_ASSETS = [candidate_pool, *ROUND_ASSETS]
