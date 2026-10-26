"""Optional Weights & Biases mirror of a tracking run (off by default, never required).

MLflow stays the system of record and the registry. When ``fusi.tracking.wandb`` is
true *and* the ``wandb`` package is importable *and* it is configured (``WANDB_API_KEY``
or ``WANDB_MODE=offline``), params/metrics are mirrored to W&B; in every other case
every method is a no-op, so pipelines never depend on W&B being present.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Mapping
from typing import Any

from ml_platform.logging import get_logger

log = get_logger(__name__)


def wandb_available() -> bool:
    if importlib.util.find_spec("wandb") is None:
        return False
    return bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline")


class WandbMirror:
    def __init__(self, enabled: bool, *, project: str, run_name: str, config: Mapping[str, Any]):
        self._run: Any = None
        if not enabled:
            return
        if not wandb_available():
            log.info("wandb.disabled", reason="package missing or not configured")
            return
        wandb = importlib.import_module("wandb")
        self._run = wandb.init(project=project, name=run_name, config=dict(config), reinit=True)

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(dict(metrics), step=step)

    def summary(self, values: Mapping[str, Any]) -> None:
        if self._run is not None:
            self._run.summary.update(dict(values))

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
