"""Narrow MLflow adapter — the only module family (``tracking/``) that imports mlflow.

Everything else in the platform talks to experiment tracking through
:class:`Tracker`, so swapping MLflow for another tracker (or a hosted MLflow)
touches only this package.

Tracking URI resolution (see :func:`resolve_tracking_uri`):

1. ``MLFLOW_TRACKING_URI`` environment variable (e.g. the Docker Compose MLflow
   server ``http://localhost:5000``) wins over the config;
2. otherwise ``cfg.tracking.tracking_uri`` (default ``sqlite:///mlflow.db``).

A *relative* SQLite path is anchored at the repository root, not the current
working directory, so ``scripts/``, notebooks, Dagster and Ray workers all
share one database. For a local SQLite store, artifacts go to a sibling
``mlartifacts/`` directory next to the database file (never ``./mlruns`` in
whatever the CWD happens to be).

What a training run records (``log_train_result``):

* params: hyperparameters, seed, world size, dataset id, sizes;
* metrics: the per-epoch history (``step=epoch``) + duration;
* tags: ``dataset_hash``, ``config_hash``, ``git_sha``, ``checkpoint_hash`` and
  environment metadata (python/torch/ray/mlflow versions, device, platform).
  ``config_hash`` covers only the *scientific* config
  (``PlatformConfig.scientific_dump``: seed/data/model/training/evaluation/
  active_learning + number of DDP workers), so relocating the data dir or
  switching the tracking URI does not change it;
* inputs: the training dataset via ``mlflow.data`` (schema + digest, not rows);
* artifacts: the final checkpoint directory, ``config.json`` (the full config,
  environment sections included), ``config_scientific.json`` (what was hashed),
  ``train_result.json``.
"""

from __future__ import annotations

import json
import math
import os
import platform
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas

from ml_platform.config import REPO_ROOT, PlatformConfig, TrackingConfig
from ml_platform.hashing import git_sha, hash_config
from ml_platform.logging import get_logger
from ml_platform.storage import require_mounted

if TYPE_CHECKING:
    from ml_platform.evaluation.evaluator import EvaluationResult
    from ml_platform.training.trainer import TrainResult

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
ARTIFACTS_DIRNAME = "mlartifacts"
_SQLITE_PREFIX = "sqlite:///"

log = get_logger(__name__)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


# --------------------------------------------------------------------------- uri helpers
def resolve_tracking_uri(configured: str | None = None, *, base: Path | None = None) -> str:
    """Env var over config; relative sqlite/file paths anchored at ``base`` (repo root)."""
    uri = os.environ.get(TRACKING_URI_ENV) or configured or TrackingConfig().tracking_uri
    base = base or REPO_ROOT
    if uri.startswith(_SQLITE_PREFIX):
        path = uri[len(_SQLITE_PREFIX) :]
        if path and path != ":memory:" and not path.startswith("/"):
            return f"{_SQLITE_PREFIX}{(base / path).resolve()}"
        return uri
    if "://" not in uri:  # a bare local path (legacy file store)
        p = Path(uri)
        return str(p if p.is_absolute() else (base / p).resolve())
    return uri


def sqlite_path(uri: str) -> Path | None:
    """Database file of a ``sqlite:///`` URI (None for other stores)."""
    if uri.startswith(_SQLITE_PREFIX):
        path = uri[len(_SQLITE_PREFIX) :]
        return Path(path) if path.startswith("/") else None
    return None


def default_artifact_root(uri: str) -> Path | None:
    """Repo-local artifact root for a local SQLite store: ``<db dir>/mlartifacts``."""
    db = sqlite_path(uri)
    return db.parent / ARTIFACTS_DIRNAME if db is not None else None


def local_path_from_uri(uri: str) -> Path | None:
    """``file:///x`` or a plain path -> Path; None for remote URIs."""
    if uri.startswith("file://"):
        from urllib.parse import unquote, urlparse

        return Path(unquote(urlparse(uri).path))
    if "://" not in uri:
        return Path(uri)
    return None


def environment_metadata() -> dict[str, str]:
    """Library versions + host info recorded with every training run."""
    out = {"python_version": platform.python_version(), "platform": platform.platform()}
    for dist in ("torch", "ray", "mlflow", "numpy", "pandas"):
        try:
            out[f"{dist}_version"] = metadata.version(dist)
        except metadata.PackageNotFoundError:  # pragma: no cover
            out[f"{dist}_version"] = "unknown"
    return out


def _flat(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _flat(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = obj


def _param_value(v: Any) -> str:
    s = json.dumps(v, default=str) if isinstance(v, list | dict | tuple) else str(v)
    return s[:6000]  # MLflow param value limit


# --------------------------------------------------------------------------- tracker
class Tracker:
    """Thin experiment-tracking adapter around MLflow.

    Usage::

        tracker = Tracker(cfg.tracking)
        with tracker.start_run("train_round_003", tags={"round_id": "3"}) as run_id:
            tracker.log_params({...})
            tracker.log_train_result(result, cfg=cfg)
    """

    def __init__(
        self,
        cfg: TrackingConfig | PlatformConfig | None = None,
        *,
        tracking_uri: str | None = None,
    ) -> None:
        tcfg = cfg.tracking if isinstance(cfg, PlatformConfig) else (cfg or TrackingConfig())
        self.cfg = tcfg
        self.tracking_uri = (
            resolve_tracking_uri(tracking_uri)
            if tracking_uri
            else resolve_tracking_uri(tcfg.tracking_uri)
        )
        self.experiment_name = tcfg.experiment
        self.artifact_root = default_artifact_root(self.tracking_uri)
        db = sqlite_path(self.tracking_uri)
        if db is not None:
            require_mounted(db).parent.mkdir(parents=True, exist_ok=True)
        self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)
        self._experiment_id: str | None = None
        self._run_id: str | None = None

    # ------------------------------------------------------------------ setup
    def _activate(self) -> None:
        """Point the (process-global) fluent API at this tracker's store."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)

    @property
    def experiment_id(self) -> str:
        if self._experiment_id is None:
            exp = self.client.get_experiment_by_name(self.experiment_name)
            if exp is None:
                location = (
                    (self.artifact_root / self.experiment_name).resolve().as_uri()
                    if self.artifact_root is not None
                    else None
                )
                self._experiment_id = self.client.create_experiment(
                    self.experiment_name, artifact_location=location
                )
            else:
                self._experiment_id = exp.experiment_id
        return self._experiment_id

    @property
    def run_id(self) -> str | None:
        """The run opened by :meth:`start_run` (None outside a run)."""
        return self._run_id

    def _require_run(self) -> str:
        if self._run_id is None:
            raise RuntimeError("no active run: use `with tracker.start_run(...)`")
        return self._run_id

    # ------------------------------------------------------------------ runs
    @contextmanager
    def start_run(
        self,
        name: str | None = None,
        tags: Mapping[str, Any] | None = None,
        *,
        run_id: str | None = None,
    ) -> Iterator[str]:
        """Open (or, with ``run_id``, re-open) a run; yields the run id.

        The run is marked FAILED if the block raises, FINISHED otherwise.
        """
        self._activate()
        str_tags = {k: str(v) for k, v in (tags or {}).items() if v is not None}
        active = mlflow.start_run(
            run_id=run_id,
            experiment_id=None if run_id else self.experiment_id,
            run_name=None if run_id else name,
            tags=str_tags or None,
        )
        previous = self._run_id
        self._run_id = active.info.run_id
        log.info("tracking.run_started", run_id=self._run_id, run_name=name)
        status = "FINISHED"
        try:
            yield self._run_id
        except BaseException:
            status = "FAILED"
            raise
        finally:
            mlflow.end_run(status=status)
            self._run_id = previous

    # ------------------------------------------------------------------ primitives
    def log_params(self, params: Mapping[str, Any]) -> None:
        self._require_run()
        flat: dict[str, Any] = {}
        _flat("", dict(params), flat)
        mlflow.log_params({k: _param_value(v) for k, v in flat.items()})

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        self._require_run()
        clean = {}
        for k, v in metrics.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                clean[k] = f
        if clean:
            mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        self._require_run()
        mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})

    def log_artifacts(
        self, local_dir: str | os.PathLike[str], artifact_path: str | None = None
    ) -> None:
        self._require_run()
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def log_artifact(
        self, local_file: str | os.PathLike[str], artifact_path: str | None = None
    ) -> None:
        self._require_run()
        mlflow.log_artifact(str(local_file), artifact_path=artifact_path)

    def log_dict(self, data: Mapping[str, Any], artifact_file: str) -> None:
        self._require_run()
        mlflow.log_dict(json.loads(json.dumps(dict(data), default=str)), artifact_file)

    def log_dataset(
        self,
        frame: pd.DataFrame,
        *,
        name: str,
        digest: str,
        source: str | None = None,
        context: str = "training",
    ) -> None:
        """Record the dataset as a run *input* (schema + digest; rows are not uploaded)."""
        self._require_run()
        cols = [c for c in frame.columns if c not in ("provenance", "created_at")]
        ds = from_pandas(
            frame[cols],
            source=source,  # type: ignore[arg-type]  # None -> code-location source
            targets="response" if "response" in cols else None,
            name=name,
            digest=digest[:36],
        )
        mlflow.log_input(ds, context=context)
        mlflow.set_tags({"dataset_id": digest, "dataset_name": name})

    # ------------------------------------------------------------------ conveniences
    def log_train_result(
        self,
        result: TrainResult,
        *,
        cfg: PlatformConfig | None = None,
        dataset_frame: pd.DataFrame | None = None,
        dataset_name: str = "training_dataset",
        log_checkpoint: bool = True,
    ) -> None:
        """Log everything needed to reproduce / audit a training run."""
        self._require_run()
        params: dict[str, Any] = {
            **result.hyperparams,
            "seed": result.seed,
            "world_size": result.world_size,
            "device": result.device,
            "n_train": result.n_train,
            "n_val": result.n_val,
            "n_parameters": result.n_parameters,
            "epochs_completed": result.epochs_completed,
            "dataset_id": result.dataset_hash,
        }
        self.log_params(params)
        for i, row in enumerate(result.history, start=1):
            self.log_metrics(
                {k: v for k, v in row.items() if k != "epoch"}, step=int(row.get("epoch", i))
            )
        self.log_metrics(
            {
                "duration_s": result.duration_s,
                **{f"final_{k}": v for k, v in result.metrics.items()},
            }
        )
        # scientific config only: moving data dirs / the MLflow URI must not change it
        config_hash = cfg.config_hash() if cfg is not None else hash_config(result.hyperparams)
        info = dict(result.device_info)
        distributed = info.pop("distributed", None)
        tags: dict[str, Any] = {
            "dataset_hash": result.dataset_hash,
            "config_hash": config_hash,
            "git_sha": result.git_sha or git_sha(),
            "checkpoint_hash": result.checkpoint_hash,
            "world_size": result.world_size,
            "resumed_from": result.resumed_from,
            **environment_metadata(),
            **{f"device.{k}": v for k, v in info.items() if not isinstance(v, dict | list)},
        }
        if isinstance(distributed, Mapping):
            for k in ("ray_job_id", "ray_train_run", "failures_recovered", "restored_from_epoch"):
                if distributed.get(k) is not None:
                    tags[f"ray.{k}"] = distributed[k]
        self.set_tags(tags)
        if dataset_frame is not None and result.dataset_hash:
            self.log_dataset(dataset_frame, name=dataset_name, digest=result.dataset_hash)
        if cfg is not None:
            # the full config (incl. paths/tracking/serve) stays available as an artifact
            self.log_dict(cfg.model_dump(mode="json"), "config.json")
            self.log_dict(cfg.scientific_dump(), "config_scientific.json")
        self.log_dict(result.to_dict(), "train_result.json")
        if log_checkpoint and Path(result.checkpoint_path).exists():
            self.log_artifacts(result.checkpoint_path, artifact_path="checkpoint")

    def log_evaluation(
        self, evaluation: EvaluationResult, out_dir: str | os.PathLike[str]
    ) -> dict[str, Path]:
        """Write the evaluation report files to ``out_dir`` and log them + metrics + gate."""
        self._require_run()
        files = evaluation.write(out_dir)
        self.log_metrics({f"eval_{k}": v for k, v in evaluation.metrics.items()})
        self.set_tags(
            {
                "gate_passed": str(evaluation.gate.passed).lower(),
                "gate_reasons": " | ".join(evaluation.gate.reasons)[:5000],
                "eval_baseline": evaluation.baseline_name,
            }
        )
        self.log_artifacts(out_dir, artifact_path="evaluation")
        return files

    # ------------------------------------------------------------------ queries
    def get_run(self, run_id: str) -> Any:
        return self.client.get_run(run_id)

    def run_artifact_dir(self, run_id: str) -> Path | None:
        """Local directory holding a run's artifacts (None for remote stores)."""
        return local_path_from_uri(self.client.get_run(run_id).info.artifact_uri)
