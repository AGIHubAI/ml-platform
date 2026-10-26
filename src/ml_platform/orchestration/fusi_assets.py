"""Dagster assets, blocking asset checks and jobs of the fUSI real-time workload.

::

    fusi_session[sub00_ses00, ...]   dynamic partitions, one per recording session
      -> fusi_split -> fusi_preprocessor -> fusi_dataset  ── 6 blocking leakage checks
      -> fusi_decoder -> fusi_evaluation                  ── 2 blocking checks
      -> fusi_registered_decoder (gate-driven promotion in the MLflow registry)

Glue only, like ``assets.py``: bodies call ``fusi_pipeline`` steps. Every guard is an
``@asset_check(blocking=True)``: when one fails, Dagster skips the downstream assets
of that run, so a leaky split or an overfit model never reaches the registry. The
registry gate itself (``promote_if_passed``) is the second, independent barrier.

No ``from __future__ import annotations`` here (Dagster inspects real annotations).
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from dagster import (
    AssetCheckResult,
    AssetChecksDefinition,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetSelection,
    DagsterInstance,
    Definitions,
    DynamicPartitionsDefinition,
    ExecuteInProcessResult,
    MetadataValue,
    Output,
    asset,
    asset_check,
    define_asset_job,
)

from ml_platform.config import PlatformConfig
from ml_platform.logging import bound_ids
from ml_platform.orchestration import fusi_pipeline as FP
from ml_platform.orchestration.resources import PlatformConfigResource
from ml_platform.workloads.fusi_realtime import telemetry

GROUP = "fusi_realtime"
FUSI_SESSIONS_PARTITION = "fusi_sessions"
fusi_sessions_partitions = DynamicPartitionsDefinition(name=FUSI_SESSIONS_PARTITION)

DATASET_CHECKS = (
    "sessions_disjoint",
    "subjects_disjoint",
    "temporal_gap",
    "no_window_overlap",
    "preprocessing_fit_on_train_only",
    "feature_schema",
)
EVALUATION_CHECKS = ("test_set_access", "train_test_gap")


@contextmanager
def _step(context: AssetExecutionContext, cfg: PlatformConfig, **ids: str) -> Iterator[None]:
    """Bind log ids and wrap the step in one ``dagster.asset.<name>`` span."""
    tracer = telemetry.configure(cfg.fusi.telemetry, service="fusi-dagster")
    attrs = {"dagster.run_id": context.run_id}
    if context.has_partition_key:
        attrs["dagster.partition"] = context.partition_key
    name = f"dagster.asset.{context.asset_key.to_user_string()}"
    try:
        with bound_ids(run_id=context.run_id, **ids), telemetry.span(tracer, name, **attrs):
            yield
    finally:
        telemetry.flush()  # a multiprocess-executor step exits right after the asset


@asset(
    partitions_def=fusi_sessions_partitions,
    group_name=GROUP,
    kinds={"python", "zarr"},
    description="One simulated fUSI recording session -> Zarr volumes + Parquet events.",
)
def fusi_session(
    context: AssetExecutionContext, platform_config: PlatformConfigResource
) -> Output[FP.FusiSessionInfo]:
    cfg = platform_config.load()
    with _step(context, cfg):
        info = FP.simulate_session(cfg, context.partition_key)
    return Output(
        info,
        metadata={
            "dagster/row_count": MetadataValue.int(info.n_frames),
            "subject": MetadataValue.text(info.subject),
            "events_hash": MetadataValue.text(info.events_hash),
            "variables": MetadataValue.text(", ".join(info.variables)),
            "path": MetadataValue.path(str(info.path)),
        },
    )


def _sessions(sessions: dict[str, FP.FusiSessionInfo]) -> list[FP.FusiSessionInfo]:
    return [sessions[k] for k in sorted(sessions)]


@asset(
    group_name=GROUP,
    kinds={"python", "parquet"},
    description="Leakage-safe split of sessions into train/val/test blocks.",
)
def fusi_split(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    fusi_session: dict[str, FP.FusiSessionInfo],
) -> Output[FP.FusiSplitInfo]:
    cfg = platform_config.load()
    with _step(context, cfg):
        info = FP.plan_split(cfg, _sessions(fusi_session))
    return Output(
        info,
        metadata={
            "split_hash": MetadataValue.text(info.split_hash),
            "hold_out": MetadataValue.text(info.hold_out),
            "gap_frames": MetadataValue.int(info.gap_frames),
            "sessions": MetadataValue.json(info.sessions),
        },
    )


@asset(
    group_name=GROUP,
    kinds={"dask", "zarr"},
    description="RETROICOR + z-score fitted on the TRAIN fold only (Dask over Zarr).",
)
def fusi_preprocessor(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    fusi_session: dict[str, FP.FusiSessionInfo],
    fusi_split: FP.FusiSplitInfo,
) -> Output[FP.FusiPreprocessorInfo]:
    cfg = platform_config.load()
    with _step(context, cfg):
        info = FP.fit_preprocessing(cfg, _sessions(fusi_session), fusi_split)
    return Output(
        info,
        metadata={
            "state_hash": MetadataValue.text(info.state_hash),
            "fit_sessions": MetadataValue.text(", ".join(info.fit_sessions)),
            "fit_n_frames": MetadataValue.int(info.fit_n_frames),
            "path": MetadataValue.path(str(info.path)),
        },
    )


@asset(
    group_name=GROUP,
    kinds={"dask", "zarr"},
    description="Frozen features for every session + windows; leakage checks attached.",
)
def fusi_dataset(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    fusi_session: dict[str, FP.FusiSessionInfo],
    fusi_split: FP.FusiSplitInfo,
    fusi_preprocessor: FP.FusiPreprocessorInfo,
) -> Output[FP.FusiDatasetInfo]:
    cfg = platform_config.load()
    with _step(context, cfg):
        info = FP.build_dataset(cfg, _sessions(fusi_session), fusi_split, fusi_preprocessor)
    return Output(
        info,
        metadata={
            "dataset_hash": MetadataValue.text(info.dataset_hash),
            "split_hash": MetadataValue.text(info.split_hash),
            "windows_hash": MetadataValue.text(info.windows_hash),
            "n_windows": MetadataValue.json(info.n_windows),
            "checks": MetadataValue.json({c["name"]: c["passed"] for c in info.checks}),
        },
    )


@asset(
    group_name=GROUP,
    kinds={"ray", "pytorch", "mlflow"},
    description="Candidate decoder: Ray Train + DDP transformer (or PCA+LDA); MLflow run.",
)
def fusi_decoder(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    fusi_dataset: FP.FusiDatasetInfo,
) -> Output[FP.FusiDecoderInfo]:
    cfg = platform_config.load()
    with _step(context, cfg, dataset_id=fusi_dataset.dataset_hash):
        info = FP.train_decoder(cfg, fusi_dataset)
    return Output(
        info,
        metadata={
            "mlflow_run_id": MetadataValue.text(info.run_id),
            "kind": MetadataValue.text(info.kind),
            "checkpoint": MetadataValue.path(str(info.checkpoint_dir)),
            "checkpoint_hash": MetadataValue.text(info.checkpoint_hash),
            "params_in_sync": MetadataValue.text(str(info.summary.get("params_in_sync"))),
        },
    )


@asset(
    group_name=GROUP,
    kinds={"ray", "mlflow"},
    description="Test accuracy vs production/baseline, real-time latency run, promotion gate.",
)
def fusi_evaluation(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    fusi_dataset: FP.FusiDatasetInfo,
    fusi_decoder: FP.FusiDecoderInfo,
) -> Output[FP.FusiEvaluationInfo]:
    cfg = platform_config.load()
    with _step(context, cfg, dataset_id=fusi_dataset.dataset_hash):
        info = FP.evaluate_decoder(cfg, fusi_dataset, fusi_decoder)
    e2e = info.latency["end_to_end"]
    return Output(
        info,
        metadata={
            "test_frame_acc": MetadataValue.float(info.accuracy["test_frame_acc"]),
            "reference": MetadataValue.text(info.reference),
            "reference_acc": MetadataValue.float(info.reference_acc),
            "e2e_p50_ms": MetadataValue.float(float(e2e["p50"])),
            "e2e_p99_ms": MetadataValue.float(float(e2e["p99"])),
            "dropped": MetadataValue.int(int(info.latency["n_dropped"])),
            "test_accesses": MetadataValue.int(info.test_accesses),
            "gate_passed": MetadataValue.bool(info.gate_passed),
            "gate_reasons": MetadataValue.md(
                "\n".join(f"- {r}" for r in info.gate_reasons) or "- (none)"
            ),
            "report": MetadataValue.path(str(info.report_path)),
        },
    )


@asset(
    group_name=GROUP,
    kinds={"mlflow"},
    description="Registered as candidate; promoted to production only if the gate passed.",
)
def fusi_registered_decoder(
    context: AssetExecutionContext,
    platform_config: PlatformConfigResource,
    fusi_decoder: FP.FusiDecoderInfo,
    fusi_evaluation: FP.FusiEvaluationInfo,
) -> Output[FP.FusiRegistrationInfo]:
    cfg = platform_config.load()
    with _step(context, cfg):
        info = FP.register_decoder(cfg, fusi_decoder, fusi_evaluation)
    return Output(
        info,
        metadata={
            "model_version": MetadataValue.text(info.version),
            "stage": MetadataValue.text(info.stage),
            "gate_passed": MetadataValue.bool(info.gate_passed),
        },
    )


# --------------------------------------------------------------------------- checks
def _check_factory(asset_def: Any, name: str, input_name: str) -> AssetChecksDefinition:
    """One blocking check that reads the named result recorded by the step."""

    def _result(value: Any) -> AssetCheckResult:
        c = value.check(name)
        return AssetCheckResult(
            passed=bool(c["passed"]),
            severity=AssetCheckSeverity.ERROR,
            description=c["detail"],
            metadata={"detail": MetadataValue.text(str(c["detail"]))},
        )

    fn: Callable[..., AssetCheckResult]
    if input_name == "fusi_dataset":

        def fn(fusi_dataset: FP.FusiDatasetInfo) -> AssetCheckResult:
            return _result(fusi_dataset)

    else:

        def fn(fusi_evaluation: FP.FusiEvaluationInfo) -> AssetCheckResult:
            return _result(fusi_evaluation)

    return asset_check(asset=asset_def, name=name, blocking=True, description=name)(fn)


FUSI_CHECKS: list[AssetChecksDefinition] = [
    *(_check_factory(fusi_dataset, n, "fusi_dataset") for n in DATASET_CHECKS),
    *(_check_factory(fusi_evaluation, n, "fusi_evaluation") for n in EVALUATION_CHECKS),
]

FUSI_TRAIN_ASSETS = [
    fusi_split,
    fusi_preprocessor,
    fusi_dataset,
    fusi_decoder,
    fusi_evaluation,
    fusi_registered_decoder,
]
FUSI_ASSETS = [fusi_session, *FUSI_TRAIN_ASSETS]

FUSI_SESSIONS_JOB = "fusi_sessions_job"
FUSI_TRAIN_JOB = "fusi_train_job"

fusi_sessions_job = define_asset_job(
    FUSI_SESSIONS_JOB,
    selection=AssetSelection.assets(fusi_session),
    partitions_def=fusi_sessions_partitions,
    description="Simulate (or ingest) one fUSI session partition into Zarr + Parquet.",
    tags={"ml/pipeline": "fusi_sessions"},
)
fusi_train_job = define_asset_job(
    FUSI_TRAIN_JOB,
    selection=AssetSelection.assets(*FUSI_TRAIN_ASSETS),
    description=(
        "split -> preprocess (train fold) -> dataset + leakage checks -> Ray Train DDP -> "
        "evaluation + latency + gate -> registry (blocking checks stop promotion)."
    ),
    tags={"ml/pipeline": "fusi_train"},
)
FUSI_JOBS = [fusi_sessions_job, fusi_train_job]


def ensure_session_partitions(instance: DagsterInstance, keys: Sequence[str]) -> None:
    missing = [k for k in keys if not instance.has_dynamic_partition(FUSI_SESSIONS_PARTITION, k)]
    if missing:
        instance.add_dynamic_partitions(FUSI_SESSIONS_PARTITION, missing)


def run_fusi(
    defs: Definitions, instance: DagsterInstance, keys: Sequence[str]
) -> tuple[list[ExecuteInProcessResult], ExecuteInProcessResult]:
    """Materialise every session partition, then the training job (in-process)."""
    ensure_session_partitions(instance, keys)
    sessions_job = defs.resolve_job_def(FUSI_SESSIONS_JOB)
    session_runs = [
        sessions_job.execute_in_process(partition_key=k, instance=instance) for k in keys
    ]
    train = defs.resolve_job_def(FUSI_TRAIN_JOB).execute_in_process(
        instance=instance, raise_on_error=False
    )
    return session_runs, train
