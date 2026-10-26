"""Dagster code location: ``dagster dev -m ml_platform.orchestration.definitions``.

`build_definitions` wires assets, jobs, sensors, schedules and resources (the
closed loop plus the fUSI real-time workload in ``fusi_assets.py``);
``defs`` is the default instance (config from ``$ML_CONFIG`` or
``configs/local.yaml``, Ray from ``$RAY_ADDRESS`` or a local cluster, MLflow
from ``$MLFLOW_TRACKING_URI`` or the config). Tests and scripts call
`build_definitions` with explicit resources instead — dependency injection
at the orchestration boundary.

Executor: in-process. Steps of one run execute sequentially in one process
(the round graph is a chain anyway) and share one Ray driver connection;
parallelism comes from Ray, not from Dagster subprocesses.
"""

from __future__ import annotations

from dagster import Definitions, FilesystemIOManager, in_process_executor

from ml_platform.orchestration.assets import ALL_ASSETS
from ml_platform.orchestration.fusi_assets import FUSI_ASSETS, FUSI_CHECKS, FUSI_JOBS
from ml_platform.orchestration.jobs import JOBS
from ml_platform.orchestration.resources import (
    PlatformConfigResource,
    RayComputeResource,
    RoundStoreResource,
    TrackingResource,
)
from ml_platform.orchestration.sensors import SCHEDULES, SENSORS


def build_definitions(
    *,
    config_path: str | None = None,
    ray_address: str | None = None,
    num_workers: int | None = None,
    n_inference_actors: int = 2,
    tracking_uri: str | None = None,
) -> Definitions:
    platform_config = PlatformConfigResource(config_path=config_path)
    return Definitions(
        assets=[*ALL_ASSETS, *FUSI_ASSETS],
        asset_checks=FUSI_CHECKS,
        jobs=[*JOBS, *FUSI_JOBS],
        sensors=SENSORS,
        schedules=SCHEDULES,
        resources={
            "platform_config": platform_config,
            "round_store": RoundStoreResource(config=platform_config),
            "tracking": TrackingResource(config=platform_config, tracking_uri=tracking_uri),
            "ray_compute": RayComputeResource(
                address=ray_address, num_workers=num_workers, n_inference_actors=n_inference_actors
            ),
            # Asset values are small path/hash records; base_dir defaults to the
            # instance's storage directory ($DAGSTER_HOME/storage).
            "io_manager": FilesystemIOManager(),
        },
        executor=in_process_executor,
    )


defs = build_definitions()
