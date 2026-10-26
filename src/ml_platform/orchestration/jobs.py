"""Asset jobs + programmatic execution of them against a Dagster instance.

* ``bootstrap_job`` — ``candidate_pool`` + ``observed_experiments[round_000]``.
* ``closed_loop_round_job`` — one full turn of the loop for one round partition:
  observed_experiments → … → new_experimental_results (writes round r+1).
* ``retrain_job`` — training_dataset → … → registered_model for a round
  (target of the nightly schedule: retrain/re-evaluate without new experiments).

`run_bootstrap` / `run_round` execute the *same* job definitions in-process
against a persistent `DagsterInstance`, which is how ``scripts/run_closed_loop.py``
drives the loop: every run, materialization, metadata entry and lineage edge
is recorded in ``$DAGSTER_HOME`` and shows up in ``make dagster`` afterwards.
"""

from __future__ import annotations

from typing import Any

from dagster import (
    AssetSelection,
    DagsterInstance,
    Definitions,
    ExecuteInProcessResult,
    define_asset_job,
)

from ml_platform.orchestration.assets import (
    ROUND_ASSETS,
    candidate_pool,
    evaluation_report,
    observed_experiments,
    registered_model,
    trained_model,
    training_dataset,
)
from ml_platform.orchestration.partitions import ensure_round_partition, rounds_partitions

BOOTSTRAP_JOB = "bootstrap_job"
ROUND_JOB = "closed_loop_round_job"
RETRAIN_JOB = "retrain_job"

bootstrap_job = define_asset_job(
    BOOTSTRAP_JOB,
    selection=AssetSelection.assets(candidate_pool, observed_experiments),
    partitions_def=rounds_partitions,
    description="Create the candidate pool and the initial random design (round_000).",
    tags={"ml/pipeline": "bootstrap"},
)

closed_loop_round_job = define_asset_job(
    ROUND_JOB,
    selection=AssetSelection.assets(*ROUND_ASSETS),
    partitions_def=rounds_partitions,
    description=(
        "One closed-loop round: data → Ray Train DDP → evaluation → gated registry → "
        "Ray batch inference → active learning → oracle → next immutable round."
    ),
    tags={"ml/pipeline": "closed_loop"},
)

retrain_job = define_asset_job(
    RETRAIN_JOB,
    selection=AssetSelection.assets(
        training_dataset, trained_model, evaluation_report, registered_model
    ),
    partitions_def=rounds_partitions,
    description="Retrain + re-evaluate + gate the model of a round (no new experiments).",
    tags={"ml/pipeline": "retrain"},
)

JOBS = [bootstrap_job, closed_loop_round_job, retrain_job]


def run_bootstrap(
    defs: Definitions, instance: DagsterInstance, *, tags: dict[str, Any] | None = None
) -> ExecuteInProcessResult:
    """Materialize ``bootstrap_job`` for ``round_000`` (idempotent: pool/round are write-once)."""
    key = ensure_round_partition(instance, 0)
    return defs.resolve_job_def(BOOTSTRAP_JOB).execute_in_process(
        partition_key=key, instance=instance, tags=tags
    )


def run_round(
    defs: Definitions,
    instance: DagsterInstance,
    round_id: int,
    *,
    tags: dict[str, Any] | None = None,
) -> ExecuteInProcessResult:
    """Materialize every per-round asset of ``round_id`` (one Dagster run)."""
    key = ensure_round_partition(instance, round_id)
    return defs.resolve_job_def(ROUND_JOB).execute_in_process(
        partition_key=key, instance=instance, tags=tags
    )
