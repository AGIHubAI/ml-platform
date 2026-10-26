"""Event-driven and scheduled triggers.

``new_round_sensor`` — scientific data arrives when it arrives. The sensor
polls the RoundStore; when a new ``data/rounds/round_XXX`` directory appears
(written by ``new_experimental_results`` or by an external lab process) it
registers the dynamic partition and requests ``closed_loop_round_job`` for
it. Enabled, it turns the platform into an autonomous campaign: round r's
results trigger round r+1, up to ``ML_MAX_ROUND`` (default 5) so a demo
cannot run away. ``run_key`` = round key + manifest hash, so a round is
processed once per content version. Default status: STOPPED (the closed-loop
script drives rounds itself; enable the sensor in the UI to go event-driven).

``nightly_retrain_schedule`` — retrain/re-evaluate the latest round every night
(e.g. after a code or config change); default STOPPED.
"""

import os

from dagster import (
    AssetKey,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    RunRequest,
    ScheduleEvaluationContext,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    schedule,
    sensor,
)

from ml_platform.data import round_key
from ml_platform.orchestration.jobs import closed_loop_round_job, retrain_job
from ml_platform.orchestration.partitions import ROUNDS_PARTITION_NAME, rounds_partitions
from ml_platform.orchestration.resources import RoundStoreResource

MAX_ROUND_ENV = "ML_MAX_ROUND"
DEFAULT_MAX_ROUND = 5


@sensor(
    job=closed_loop_round_job,
    minimum_interval_seconds=30,
    default_status=DefaultSensorStatus.STOPPED,
    description="Detects new immutable rounds in the RoundStore and runs the closed loop for them.",
)
def new_round_sensor(
    context: SensorEvaluationContext, round_store: RoundStoreResource
) -> SensorResult | SkipReason:
    store = round_store.store()
    rounds = store.list_rounds()
    cursor = int(context.cursor) if context.cursor else -1
    fresh = [r for r in rounds if r > cursor]
    if not fresh:
        return SkipReason(f"no new rounds in {store.rounds_dir} (last seen: {cursor})")
    max_round = int(os.environ.get(MAX_ROUND_ENV, DEFAULT_MAX_ROUND))
    known = set(context.instance.get_dynamic_partitions(ROUNDS_PARTITION_NAME))
    to_add = [round_key(r) for r in fresh if round_key(r) not in known]
    done = context.instance.get_materialized_partitions(AssetKey("new_experimental_results"))
    run_requests = [
        RunRequest(
            partition_key=round_key(r),
            run_key=f"{round_key(r)}:{store.read_manifest(r).manifest_hash[:12]}",
            tags={"ml/trigger": "new_round_sensor"},
        )
        for r in fresh
        if r < max_round and round_key(r) not in done
    ]
    return SensorResult(
        run_requests=run_requests,
        dynamic_partitions_requests=[rounds_partitions.build_add_request(to_add)] if to_add else [],
        cursor=str(max(rounds)),
    )


@schedule(
    job=retrain_job,
    cron_schedule="0 2 * * *",
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.STOPPED,
    description="Nightly retrain + re-evaluation of the latest round.",
)
def nightly_retrain_schedule(
    context: ScheduleEvaluationContext, round_store: RoundStoreResource
) -> RunRequest | SkipReason:
    latest = round_store.store().latest_round_id()
    if latest is None:
        return SkipReason("no experimental rounds yet")
    key = round_key(latest)
    if not context.instance.has_dynamic_partition(ROUNDS_PARTITION_NAME, key):
        return SkipReason(f"partition {key} not registered yet (enable new_round_sensor)")
    return RunRequest(partition_key=key, tags={"ml/trigger": "nightly_retrain"})


SENSORS = [new_round_sensor]
SCHEDULES = [nightly_retrain_schedule]
