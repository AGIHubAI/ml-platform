"""Dagster resources: dependency injection at the orchestration boundary.

Assets never construct infrastructure clients themselves; they receive these
resources, so the same asset code runs against a laptop (local Ray, SQLite
MLflow, ``data/``), the Docker Compose stack or KubeRay by swapping resource
configuration only.

* `PlatformConfigResource` — which ``configs/*.yaml`` to load (``None`` ->
  ``$ML_CONFIG`` or ``configs/local.yaml``).
* `RoundStoreResource` — the immutable experimental-round store.
* `TrackingResource` — MLflow `Tracker` + gated `ModelRegistry`.
* `RayComputeResource` — the stable compute API (train / predict_pool /
  bootstrap_ci / run_experiments). Dagster decides *when* a step runs;
  Ray decides *where* its workers run (handoff §8: Dagster is not
  responsible for individual worker scheduling).
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from dagster import ConfigurableResource

from ml_platform.config import PlatformConfig, load_config
from ml_platform.data import RoundStore
from ml_platform.orchestration.pipeline import RayCompute
from ml_platform.tracking import ModelRegistry, Tracker
from ml_platform.training.trainer import TrainResult


class PlatformConfigResource(ConfigurableResource):  # type: ignore[type-arg]
    """Loads the typed `PlatformConfig` (YAML path or bare name like ``distributed``)."""

    config_path: str | None = None

    def load(self) -> PlatformConfig:
        return load_config(self.config_path)


class RoundStoreResource(ConfigurableResource):  # type: ignore[type-arg]
    """The write-once `RoundStore` under ``paths.data_dir``."""

    config: PlatformConfigResource

    def store(self) -> RoundStore:
        return RoundStore(self.config.load().paths.resolved().data_dir)


class TrackingResource(ConfigurableResource):  # type: ignore[type-arg]
    """MLflow adapter. ``tracking_uri`` overrides the config (``MLFLOW_TRACKING_URI`` wins over both)."""

    config: PlatformConfigResource
    tracking_uri: str | None = None

    def tracker(self) -> Tracker:
        return Tracker(self.config.load().tracking, tracking_uri=self.tracking_uri)

    def registry(self, tracker: Tracker | None = None) -> ModelRegistry:
        return ModelRegistry(tracker=tracker or self.tracker())


class RayComputeResource(ConfigurableResource):  # type: ignore[type-arg]
    """Distributed compute via Ray (Ray Train DDP, Ray actors, Ray tasks).

    ``address``: Ray cluster address (``None`` -> ``$RAY_ADDRESS`` or start a
    local cluster; ``auto`` -> a running ``ray start --head``;
    ``ray://host:10001`` -> Ray client). ``num_workers``: DDP world size
    (``None`` -> ``distributed.num_workers`` from the config).
    """

    address: str | None = None
    num_workers: int | None = None
    n_inference_actors: int = 2
    bootstrap_tasks: int = 4

    def client(self) -> RayCompute:
        return RayCompute(
            address=self.address or os.environ.get("RAY_ADDRESS") or None,
            num_workers=self.num_workers,
            n_inference_actors=self.n_inference_actors,
            bootstrap_tasks=self.bootstrap_tasks,
        )

    # Convenience pass-throughs (the stable API assets and notebooks call).
    def train(
        self,
        cfg: PlatformConfig,
        *,
        round_id: int,
        train_frame: pd.DataFrame,
        dataset_hash: str,
        run_id: str | None,
    ) -> TrainResult:
        return self.client().train(
            cfg,
            round_id=round_id,
            train_frame=train_frame,
            dataset_hash=dataset_hash,
            run_id=run_id,
        )

    def predict_pool(
        self, checkpoint_path: str, pool_frame: pd.DataFrame, *, mc_samples: int, seed: int
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        return self.client().predict_pool(
            checkpoint_path, pool_frame, mc_samples=mc_samples, seed=seed
        )

    def bootstrap_ci(
        self, y: np.ndarray, yhat: np.ndarray, *, metric: str, n_resamples: int, seed: int
    ) -> tuple[float, float, float]:
        return self.client().bootstrap_ci(
            y, yhat, metric=metric, n_resamples=n_resamples, seed=seed
        )
