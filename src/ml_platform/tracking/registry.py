"""Model registry with an explicit, gate-driven lifecycle.

Lifecycle (one ``lifecycle`` tag per model version + MLflow aliases)::

    register() ──> candidate ──(gate passed)──> validated ──> production
                       │                                         │
                       └──(gate failed: stays candidate,          └─(superseded)─> archived
                           ``gate_reasons`` tag)

* ``candidate``: registered after training; available for research, never served.
* ``validated``: passed the evaluation gates.
* ``production``: the serving model (alias ``production``, exactly one version).
* ``archived``: a former production model.

Aliases ``candidate`` / ``validated`` / ``production`` point at the newest
version in that state, so consumers resolve ``models:/<name>@production``.

**Nothing is promoted because training finished.** Promotion only happens via
:meth:`ModelRegistry.promote_if_passed` with an evaluation `GateDecision`;
``set_stage(v, "validated"|"production")`` refuses versions that have no
recorded passing gate.

Registered artifact: an ``mlflow.pyfunc`` model (:class:`SurrogatePyfunc`)
that wraps the framework-free :class:`~ml_platform.inference.Predictor`
and carries the raw training checkpoint as a model artifact, so the model is
loadable both through ``mlflow.pyfunc.load_model`` and directly with
``Predictor.from_checkpoint`` (what Ray Serve uses).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

from ml_platform.config import PlatformConfig, TrackingConfig
from ml_platform.inference.predictor import Predictor
from ml_platform.logging import get_logger
from ml_platform.tracking.mlflow_client import Tracker, local_path_from_uri
from ml_platform.training.checkpointing import MODEL_FILE, checkpoint_hash, resolve_checkpoint

Stage = Literal["candidate", "validated", "production", "archived"]
STAGES: tuple[str, ...] = ("candidate", "validated", "production", "archived")
LIFECYCLE_TAG = "lifecycle"
CHECKPOINT_ARTIFACT = "checkpoint"
MODEL_ARTIFACT_NAME = "model"

log = get_logger(__name__)


class PromotionError(RuntimeError):
    """Raised when a lifecycle transition is not allowed (e.g. no passing gate)."""


class NoProductionModelError(LookupError):
    """Raised when no version carries the ``production`` alias."""


class ProductionModel(NamedTuple):
    version: str
    checkpoint_path: Path


class SurrogatePyfunc(PythonModel):  # type: ignore[misc]
    """``mlflow.pyfunc`` wrapper around :class:`Predictor`.

    Input: a DataFrame with feature columns ``f00..`` (extra columns ignored).
    Output: DataFrame ``prediction`` (+ ``std`` when ``params={"uncertainty": True}``).
    """

    def load_context(self, context: Any) -> None:
        self.predictor = Predictor.from_checkpoint(context.artifacts[CHECKPOINT_ARTIFACT])

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        # ``pd.DataFrame`` is a type hint MLflow accepts without deriving a schema
        # from it (the explicit signature logged in ``register`` stays authoritative);
        # with no hint at all MLflow warns on every import/log.
        frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        if params and params.get("uncertainty"):
            mu, sd = self.predictor.predict_with_uncertainty(frame)
            return pd.DataFrame({"prediction": mu, "std": sd})
        return pd.DataFrame({"prediction": np.asarray(self.predictor.predict(frame))})


def _gate_fields(gate: Any) -> tuple[bool, list[str]]:
    """Accept an evaluation `GateDecision` or a mapping with ``passed``/``reasons``."""
    if gate is None:
        raise PromotionError("refusing to promote without an evaluation gate decision")
    if isinstance(gate, Mapping):
        return bool(gate["passed"]), [str(r) for r in gate.get("reasons", [])]
    return bool(gate.passed), [str(r) for r in gate.reasons]


def _signature(ckpt_file: Path) -> tuple[Any, pd.DataFrame]:
    """``(signature, input_example)``: feature columns in; prediction (+std) out; param ``uncertainty``."""
    predictor = Predictor.from_checkpoint(ckpt_file)
    example = pd.DataFrame(
        np.zeros((2, len(predictor.feature_columns))), columns=predictor.feature_columns
    )
    signature = infer_signature(
        example,
        pd.DataFrame({"prediction": predictor.predict(example)}),
        params={"uncertainty": False},
    )
    return signature, example


class ModelRegistry:
    """Lifecycle management for ``cfg.tracking.registered_model``."""

    def __init__(
        self,
        cfg: TrackingConfig | PlatformConfig | None = None,
        *,
        tracker: Tracker | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.tracker = tracker or Tracker(cfg)
        self.name = self.tracker.cfg.registered_model
        self.client = self.tracker.client
        self._cache_dir = Path(cache_dir) if cache_dir else None

    # ------------------------------------------------------------------ registration
    def register(
        self,
        run_id: str,
        checkpoint_dir: str | os.PathLike[str],
        *,
        tags: Mapping[str, Any] | None = None,
        python_model: PythonModel | None = None,
        signature: tuple[Any, pd.DataFrame] | None = None,
    ) -> str:
        """Log the checkpoint as a pyfunc model on ``run_id`` and register it as a *candidate*.

        ``python_model`` / ``signature`` default to the surrogate's (`SurrogatePyfunc`);
        other workloads (the fUSI decoder) pass their own wrapper and
        ``(signature, input_example)``.
        """
        ckpt_file = resolve_checkpoint(checkpoint_dir)
        self.tracker._activate()
        active = mlflow.active_run()
        reopen = active is None or active.info.run_id != run_id
        if reopen:
            mlflow.start_run(run_id=run_id, nested=active is not None)
        sig, input_example = signature if signature is not None else _signature(ckpt_file)
        try:
            with tempfile.TemporaryDirectory() as staging:
                # stage under a fixed name: MLflow keeps the source basename
                staged = Path(staging) / CHECKPOINT_ARTIFACT
                shutil.copytree(ckpt_file.parent, staged)
                info = mlflow.pyfunc.log_model(
                    name=MODEL_ARTIFACT_NAME,
                    python_model=python_model or SurrogatePyfunc(),
                    artifacts={CHECKPOINT_ARTIFACT: str(staged)},
                    # explicit requirements: skips MLflow's slow requirement inference
                    pip_requirements=["torch", "numpy", "pandas", "ml-platform"],
                    registered_model_name=self.name,
                    signature=sig,
                    input_example=input_example,
                )
        finally:
            if reopen:
                mlflow.end_run()
        version = str(info.registered_model_version)
        version_tags = {
            "run_id": run_id,
            "checkpoint_hash": checkpoint_hash(ckpt_file),
            "source_checkpoint": str(ckpt_file.parent),
            **{k: v for k, v in (tags or {}).items() if v is not None},
        }
        for k, v in version_tags.items():
            self.client.set_model_version_tag(self.name, version, k, str(v))
        self._set_lifecycle(version, "candidate")
        log.info("registry.registered", model=self.name, model_version=version, run_id=run_id)
        return version

    # ------------------------------------------------------------------ lifecycle
    def _set_lifecycle(self, version: str, stage: Stage) -> None:
        self.client.set_model_version_tag(self.name, version, LIFECYCLE_TAG, stage)
        if stage == "archived":
            for alias in ("candidate", "validated", "production"):
                if self._alias_version(alias) == version:
                    self.client.delete_registered_model_alias(self.name, alias)
        else:
            self.client.set_registered_model_alias(self.name, stage, version)

    def _alias_version(self, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(self.name, alias).version)
        except mlflow.exceptions.MlflowException:
            return None

    def stage(self, version: str) -> str | None:
        return self.version_tags(version).get(LIFECYCLE_TAG)

    def version_tags(self, version: str) -> dict[str, str]:
        return dict(self.client.get_model_version(self.name, version).tags)

    def set_stage(self, version: str, stage: Stage) -> None:
        """Move a version to ``stage``. Validated/production require a recorded passing gate."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        version = str(version)
        if (
            stage in ("validated", "production")
            and self.version_tags(version).get("gate_passed") != "true"
        ):
            raise PromotionError(
                f"version {version} has no passing evaluation gate; "
                "use promote_if_passed(version, gate)"
            )
        if stage == "production":
            previous = self._alias_version("production")
            if previous is not None and previous != version:
                self._set_lifecycle(previous, "archived")
                log.info("registry.archived", model=self.name, model_version=previous)
        self._set_lifecycle(version, stage)
        log.info("registry.stage", model=self.name, model_version=version, stage=stage)

    def promote_if_passed(self, version: str, gate: Any) -> str:
        """Apply an evaluation gate decision; returns the resulting lifecycle stage.

        Passing gate: candidate -> validated -> production (previous production
        -> archived). Failing gate: stays ``candidate`` with ``gate_reasons``.
        """
        version = str(version)
        passed, reasons = _gate_fields(gate)
        self.client.set_model_version_tag(self.name, version, "gate_passed", str(passed).lower())
        self.client.set_model_version_tag(
            self.name, version, "gate_reasons", " | ".join(reasons)[:5000] or "-"
        )
        if not passed:
            log.info(
                "registry.gate_failed", model=self.name, model_version=version, reasons=reasons
            )
            return self.stage(version) or "candidate"
        self.set_stage(version, "validated")
        self.set_stage(version, "production")
        return "production"

    # ------------------------------------------------------------------ lookup
    def aliases(self) -> dict[str, str]:
        """``alias -> version`` for the registered model (empty if not registered yet)."""
        try:
            rm = self.client.get_registered_model(self.name)
        except mlflow.exceptions.MlflowException:
            return {}
        return {str(a): str(v) for a, v in (rm.aliases or {}).items()}

    def list_versions(self) -> list[dict[str, Any]]:
        """All versions with lifecycle tag, gate result and aliases, oldest first.

        Aliases come from the *registered model* (one call), not from
        ``search_model_versions``: MLflow's SQL store does not populate
        ``ModelVersion.aliases`` in search results (only ``get_model_version``
        does), so relying on it would report ``aliases: []`` for every version.
        """
        by_version: dict[str, list[str]] = {}
        for alias, v in sorted(self.aliases().items()):
            by_version.setdefault(v, []).append(alias)
        out = []
        for mv in self.client.search_model_versions(f"name='{self.name}'"):
            version = str(mv.version)
            out.append(
                {
                    "version": version,
                    "run_id": mv.run_id,
                    "lifecycle": mv.tags.get(LIFECYCLE_TAG),
                    "gate_passed": mv.tags.get("gate_passed"),
                    "aliases": by_version.get(version, []),
                }
            )
        return sorted(out, key=lambda d: int(d["version"]))

    def checkpoint_path(self, version: str) -> Path:
        """Local path of the checkpoint directory stored with ``version``."""
        uri = self.client.get_model_version_download_uri(self.name, str(version))
        local = local_path_from_uri(uri)
        if local is not None and local.exists():
            model_dir = local
        else:  # remote artifact store: download once into a cache dir
            cache = self._cache_dir or (Path.home() / ".cache" / "ml_platform" / "models")
            dest = cache / self.name / str(version)
            if not dest.exists():
                tmp = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{self.name}/{version}",
                    tracking_uri=self.tracker.tracking_uri,
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(tmp, dest)
            model_dir = dest
        # the download URI is either the model dir or its ``artifacts/`` subdir
        for ckpt in (
            model_dir / "artifacts" / CHECKPOINT_ARTIFACT,
            model_dir / CHECKPOINT_ARTIFACT,
        ):
            if (ckpt / MODEL_FILE).exists():
                return ckpt
        raise FileNotFoundError(f"no checkpoint inside model version {version} at {model_dir}")

    def production_version(self) -> ProductionModel | None:
        """``(version, checkpoint_path)`` of the production model, or None."""
        version = self._alias_version("production")
        if version is None:
            return None
        return ProductionModel(version, self.checkpoint_path(version))

    def load_production_predictor(self, device: str = "cpu") -> Predictor:
        prod = self.production_version()
        if prod is None:
            raise NoProductionModelError(f"no production version of {self.name!r}")
        predictor = Predictor.from_checkpoint(prod.checkpoint_path, device=device)
        predictor._info["model_version"] = prod.version
        return predictor
