"""Distributed training of the surrogate on the immutable round union, tracked in MLflow.

Examples::

    uv run python scripts/run_training.py --config configs/distributed.yaml
    uv run python scripts/run_training.py --round 2 --workers 2 --register
    uv run python scripts/run_training.py --fail-at-epoch 3        # Ray Train auto-recovers
    uv run python scripts/run_training.py --resume artifacts/ray_train/<run>/epoch_0003

Training data is ``RoundStore.training_frame(round)`` = union(round_000..round_N)
from ``paths.data_dir``. Ray Train starts ``--workers`` DDP worker processes;
the run (params, per-epoch metrics, dataset id, hashes, environment,
checkpoint) is logged to MLflow (``MLFLOW_TRACKING_URI`` or
``tracking.tracking_uri``). With ``--register`` the model is evaluated on the
validation split, registered as a *candidate* and promoted only if it passes
the evaluation gates.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from ml_platform.config import PlatformConfig, load_config
from ml_platform.data import (
    RoundStore,
    generate_candidate_pool,
    initial_observations,
    make_oracle,
    train_val_split,
)
from ml_platform.evaluation import TargetScale, evaluate
from ml_platform.inference import Predictor
from ml_platform.logging import bind_ids, get_logger
from ml_platform.ray_runtime.cluster import ensure_ray, shutdown_ray
from ml_platform.ray_runtime.tasks import ray_job_id
from ml_platform.tracking import ModelRegistry, Tracker
from ml_platform.training.distributed import (
    distributed_info,
    load_training_frame,
    train_distributed,
)

log = get_logger("run_training")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config", default=None, help="config file (default: $ML_CONFIG or configs/local.yaml)"
    )
    p.add_argument(
        "--round", type=int, default=None, help="train on rounds 0..ROUND (default: latest)"
    )
    p.add_argument(
        "--workers", type=int, default=None, help="number of DDP workers (default: config)"
    )
    p.add_argument(
        "--fail-at-epoch",
        type=int,
        default=None,
        help="inject a worker failure after this epoch's checkpoint",
    )
    p.add_argument(
        "--max-failures", type=int, default=1, help="Ray Train FailureConfig.max_failures"
    )
    p.add_argument("--resume", default=None, help="resume from a checkpoint directory")
    p.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    p.add_argument("--data-dir", default=None, help="override paths.data_dir")
    p.add_argument("--artifacts-dir", default=None, help="override paths.artifacts_dir")
    p.add_argument("--reports-dir", default=None, help="override paths.reports_dir")
    p.add_argument(
        "--init-data",
        action="store_true",
        help="create candidate pool + round 0 if the store is empty",
    )
    p.add_argument(
        "--register", action="store_true", help="evaluate, register and gate-promote the model"
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> PlatformConfig:
    cfg = load_config(args.config)
    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides["training.epochs"] = args.epochs
    for key in ("data_dir", "artifacts_dir", "reports_dir"):
        value = getattr(args, key)
        if value is not None:
            overrides[f"paths.{key}"] = str(Path(value).resolve())
    return cfg.with_overrides(**overrides) if overrides else cfg


def init_data(cfg: PlatformConfig, store: RoundStore) -> None:
    if store.list_rounds():
        return
    pool = generate_candidate_pool(cfg)
    store.write_pool(
        pool, provenance={"generator": "generate_candidate_pool", "pool_seed": cfg.data.pool_seed}
    )
    records = initial_observations(
        pool, make_oracle(cfg), cfg.data.initial_observations, seed=cfg.seed
    )
    manifest = store.write_round(0, records, provenance={"source": "run_training --init-data"})
    print(f"initialized {store.root}: pool={len(pool)} round_000={manifest.n_records}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    paths = cfg.paths.resolved()
    store = RoundStore(paths.data_dir)
    if args.init_data:
        init_data(cfg, store)

    t0 = time.perf_counter()
    frame, dataset_id, round_id = load_training_frame(cfg, args.round)
    ensure_ray()
    tracker = Tracker(cfg.tracking)
    bind_ids(round_id=round_id, dataset_id=dataset_id, ray_job_id=ray_job_id())

    summary: dict[str, Any] = {}
    try:
        with tracker.start_run(
            f"train_round_{round_id:03d}",
            tags={
                "round_id": round_id,
                "dataset_id": dataset_id,
                "stage": "training",
                "ray_job_id": ray_job_id(),
            },
        ) as run_id:
            bind_ids(run_id=run_id)
            result = train_distributed(
                cfg,
                round_id=round_id,
                num_workers=args.workers,
                fail_at_epoch=args.fail_at_epoch,
                resume_from=args.resume,
                run_id=run_id,
                max_failures=args.max_failures,
            )
            tracker.log_train_result(
                result, cfg=cfg, dataset_frame=frame, dataset_name=f"rounds_000-{round_id:03d}"
            )
            info = distributed_info(result)
            summary = {
                "run_id": run_id,
                "round_id": round_id,
                "dataset_id": dataset_id[:16],
                "n_observations": len(frame),
                "world_size": result.world_size,
                "worker_pids": info.get("pids"),
                "params_in_sync": info.get("params_in_sync"),
                "failures_recovered": info.get("failures_recovered"),
                "restored_from_epoch": info.get("restored_from_epoch"),
                "epochs": result.epochs_completed,
                "val_rmse": round(result.metrics.get("val_rmse", float("nan")), 4),
                "val_r2": round(result.metrics.get("val_r2", float("nan")), 4),
                "train_duration_s": round(result.duration_s, 1),
                "checkpoint": str(result.checkpoint_path),
                "checkpoint_hash": (result.checkpoint_hash or "")[:16],
                "tracking_uri": tracker.tracking_uri,
            }
            if args.register:
                registry = ModelRegistry(tracker=tracker)
                train, val = train_val_split(
                    frame, cfg.training.val_fraction, cfg.seed, cfg.training.val_strategy
                )
                # dataset-owned yardstick for standardized metrics / the max_rmse gate
                scale = TargetScale.from_targets(
                    train["response"],
                    source=f"training split of rounds 000-{round_id:03d} "
                    f"[dataset {dataset_id[:12]}]",
                )
                predictor = Predictor.from_checkpoint(result.checkpoint_path)
                prod = registry.production_version()
                baseline = None
                if prod is not None:
                    baseline = Predictor.from_checkpoint(prod.checkpoint_path).predict(val)
                evaluation = evaluate(
                    predictor,
                    val,
                    baseline,
                    cfg,
                    baseline_name=f"production_v{prod.version}" if prod else None,
                    seed=cfg.seed,
                    target_scale=scale,
                )
                tracker.log_evaluation(
                    evaluation, paths.reports_dir / f"round_{round_id:03d}" / run_id
                )
                version = registry.register(
                    run_id,
                    result.checkpoint_path,
                    tags={"round_id": round_id, "dataset_hash": dataset_id},
                )
                stage = registry.promote_if_passed(version, evaluation.gate)
                summary.update(
                    {
                        "eval_rmse_std": round(evaluation.metrics["rmse"], 4),
                        "gate_passed": evaluation.gate.passed,
                        "model_version": version,
                        "lifecycle": stage,
                    }
                )
    finally:
        shutdown_ray()

    summary["total_elapsed_s"] = round(time.perf_counter() - t0, 1)
    width = max(len(k) for k in summary)
    print("\n=== training summary " + "=" * 50)
    for k, v in summary.items():
        print(f"  {k:<{width}}  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
