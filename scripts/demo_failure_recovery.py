"""Failure & recovery demo: retrying *computation* vs re-running a *scientific experiment*.

Part 1 — distributed training survives a worker failure::

    epoch 3 checkpoint persisted -> SimulatedWorkerFailure on the workers
    -> Ray Train (FailureConfig(max_failures=1)) restarts the worker group
    -> workers call ray.train.get_checkpoint() and resume at epoch 4
    -> training finishes; final checkpoint is loadable

Part 2 — experiments are not retried, they are recorded:

    ExperimentSimulator actor: a retried measure() returns the recorded
    measurement (no second "lab run"), even after the actor process dies;
    RoundStore: re-writing an identical round is a no-op, different content
    is rejected (rounds are write-once and versioned).

Usage::

    uv run python scripts/demo_failure_recovery.py [--config configs/distributed.yaml]
        [--fail-at-epoch 3] [--epochs 8] [--artifacts-dir /tmp/x]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("ML_LOG_LEVEL", "WARNING")

import pandas as pd
import ray

from ml_platform.config import load_config
from ml_platform.data import (
    ImmutableRoundError,
    RoundStore,
    generate_candidate_pool,
    initial_observations,
    make_oracle,
    records_to_frame,
)
from ml_platform.inference import Predictor
from ml_platform.ray_runtime.cluster import ensure_ray, shutdown_ray
from ml_platform.ray_runtime.tasks import start_experiment_simulator
from ml_platform.training.distributed import distributed_info, train_distributed


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def step(text: str) -> None:
    print(f"  -> {text}", flush=True)


def part1_training(args: argparse.Namespace) -> None:
    overrides: dict[str, object] = {"training.epochs": args.epochs}
    if args.artifacts_dir:
        overrides["paths.artifacts_dir"] = str(Path(args.artifacts_dir).resolve())
    cfg = load_config(args.config).with_overrides(**overrides)
    store = RoundStore(cfg.paths.resolved().data_dir)
    if store.list_rounds():
        frame = store.training_frame()
        source = f"RoundStore {store.root} (rounds {store.list_rounds()})"
    else:  # stay side-effect free: synthesize round 0 in memory, do not write data/
        pool = generate_candidate_pool(cfg)
        frame = records_to_frame(
            initial_observations(pool, make_oracle(cfg), cfg.data.initial_observations, cfg.seed)
        )
        source = "in-memory round 0 (no rounds on disk)"

    banner("PART 1 - retrying COMPUTATION: worker failure during distributed training")
    step(f"data: {len(frame)} observations from {source}")
    step(
        f"plan: {cfg.training.epochs} epochs, checkpoint every epoch, inject a failure right "
        f"after the epoch-{args.fail_at_epoch} checkpoint is committed"
    )
    step("Ray Train FailureConfig(max_failures=1): on failure, restart the whole worker group")
    t0 = time.perf_counter()
    result = train_distributed(
        cfg, train_frame=frame, fail_at_epoch=args.fail_at_epoch, max_failures=1, run_name=None
    )
    info = distributed_info(result)
    restored = info.get("restored_from_epoch")
    print()
    step(f"checkpoint written and persisted at epoch {args.fail_at_epoch}")
    step(
        f"SimulatedWorkerFailure raised after epoch {args.fail_at_epoch} "
        f"-> Ray Train recovered {info.get('failures_recovered')} failure(s)"
    )
    step(
        f"new worker group (pids {info.get('pids')}) resumed from epoch {restored}: {result.resumed_from}"
    )
    step(
        f"finished at epoch {result.epochs_completed}/{cfg.training.epochs} "
        f"(world_size={result.world_size}, replicas in sync: {info.get('params_in_sync')}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    epochs = [int(h["epoch"]) for h in result.history]
    step(f"history epochs {epochs}: 1..{restored} came from the checkpoint, the rest were re-run")
    m = result.metrics
    step(
        f"final metrics: val_rmse={m.get('val_rmse', float('nan')):.4f} "
        f"val_r2={m.get('val_r2', float('nan')):.4f} train_loss={m.get('train_loss', float('nan')):.4f}"
    )
    predictor = Predictor.from_checkpoint(result.checkpoint_path)
    step(
        f"final checkpoint {result.checkpoint_path} loads: {predictor.model_info()['epochs_trained']} epochs"
    )
    print(
        "\n  Why this retry is safe: training is deterministic computation over an immutable\n"
        "  dataset. Re-executing epochs 4..N from a committed checkpoint cannot change the\n"
        "  world; at worst it wastes compute. Retry freely (checkpoint often, bound retries)."
    )


def part2_experiments(args: argparse.Namespace) -> None:
    cfg = load_config(args.config).with_overrides(**{"data.pool_size": 1000})
    banner("PART 2 - NOT retrying an EXPERIMENT: idempotent measurements, write-once rounds")
    pool = generate_candidate_pool(cfg)
    batch = pool["candidate_id"].iloc[:5].tolist()
    with tempfile.TemporaryDirectory(prefix="ml_demo_") as tmp:
        journal = Path(tmp) / "lab_journal.jsonl"
        sim = start_experiment_simulator(cfg, pool, journal_path=journal)
        step(
            f"ExperimentSimulator actor (pid {ray.get(sim.pid.remote())}) owns the oracle + measurement log"
        )
        first: pd.DataFrame = ray.get(sim.measure.remote(1, batch))
        step(
            f"measure(round=1, {len(batch)} candidates): responses {first['response'].round(3).tolist()}"
        )
        retry: pd.DataFrame = ray.get(sim.measure.remote(1, batch))
        stats = ray.get(sim.stats.remote())
        step(
            "orchestrator retries the same call (e.g. after a timeout): identical result = "
            f"{first.equals(retry)}; physical measurements performed = {stats['n_physical_measurements']} "
            f"(cache hits {stats['n_cache_hits']})"
        )
        ray.kill(sim)
        sim2 = start_experiment_simulator(cfg, pool, journal_path=journal)
        again: pd.DataFrame = ray.get(sim2.measure.remote(1, batch))
        stats2 = ray.get(sim2.stats.remote())
        step(
            f"actor process killed; restarted actor (pid {stats2['pid']}) reloads the journal: "
            f"identical = {first.equals(again)}, new physical measurements = {stats2['n_physical_measurements']}"
        )
        ray.kill(sim2)

        store = RoundStore(Path(tmp) / "data")
        records = initial_observations(pool, make_oracle(cfg), 20, seed=cfg.seed)
        m1 = store.write_round(0, records)
        m2 = store.write_round(0, records)
        step(f"RoundStore.write_round(0) twice with identical content -> same manifest: {m1 == m2}")
        tampered = records_to_frame(records)
        tampered.loc[0, "response"] += 1.0
        try:
            store.write_round(0, tampered)
            step("UNEXPECTED: overwrite accepted")
        except ImmutableRoundError as exc:
            step(f"re-writing round 0 with DIFFERENT measurements -> rejected: {str(exc)[:70]}...")
    print(
        "\n  Why this matters: a lab measurement is expensive, noisy and irreversible. Blindly\n"
        "  re-running it on retry would spend budget and produce a *different* value for the\n"
        "  same experiment id. So experiments run at most once (actor max_task_retries=0),\n"
        "  every result is recorded under (round_id, candidate_id), retries read the record,\n"
        "  and rounds are materialized once as immutable, hash-versioned datasets."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="distributed")
    p.add_argument("--fail-at-epoch", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument(
        "--artifacts-dir",
        default=None,
        help="Ray Train storage root (default: config paths.artifacts_dir)",
    )
    args = p.parse_args(argv)
    if args.fail_at_epoch >= args.epochs:
        p.error("--fail-at-epoch must be < --epochs")
    ensure_ray()
    try:
        part1_training(args)
        part2_experiments(args)
    finally:
        shutdown_ray()
    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
