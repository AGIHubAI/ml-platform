"""Closed-loop demonstration: data → model → decision → new data → model (handoff §13).

Usage::

    uv run python scripts/run_closed_loop.py [--rounds 3] [--config configs/local.yaml] [--fresh]

For each round this materializes Dagster's ``closed_loop_round_job`` for the
round's partition, in-process, against the persistent instance in
``$DAGSTER_HOME`` (default ``./dagster_home``)::

    observed_experiments → training_dataset → trained_model (Ray Train DDP)
      → evaluation_report → registered_model (gated) → candidate_predictions
      (Ray actors) → selected_experiments → new_experimental_results (oracle)
      = the next immutable round

so every run, materialization and lineage edge is visible in ``make dagster``
afterwards. The loop bootstraps (pool + round_000) if needed and continues
from the latest round in the RoundStore, so running it again extends the
campaign instead of redoing it.

``--fresh`` first deletes the generated state (data/, reports/, artifacts/,
the MLflow SQLite db + mlartifacts/, Dagster run storage — keeping
dagster_home/dagster.yaml), refusing any path outside the repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from ml_platform.config import REPO_ROOT, PlatformConfig, load_config, resolve_config_path
from ml_platform.data import round_key
from ml_platform.logging import get_logger
from ml_platform.tracking.mlflow_client import (
    default_artifact_root,
    resolve_tracking_uri,
    sqlite_path,
)

log = get_logger("run_closed_loop")

DEFAULT_DAGSTER_HOME = REPO_ROOT / "dagster_home"
KEEP = {".gitkeep", "dagster.yaml", "workspace.yaml"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rounds", type=int, default=3, help="closed-loop iterations (default 3; 0 with --fresh)"
    )
    p.add_argument("--config", default=None, help="config file/name (default: $ML_CONFIG or local)")
    p.add_argument("--fresh", action="store_true", help="wipe generated state first")
    p.add_argument("--workers", type=int, default=None, help="DDP workers (default: config)")
    p.add_argument("--ray-address", default=None, help="Ray address (default: $RAY_ADDRESS/local)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- --fresh
def _inside_repo(path: Path) -> bool:
    p = path.resolve()
    return p != REPO_ROOT and p.is_relative_to(REPO_ROOT)


def _wipe(path: Path) -> None:
    """Delete ``path`` (file or directory contents, keeping KEEP files) — repo paths only."""
    if not path.exists():
        return
    if not _inside_repo(path):
        raise SystemExit(f"refusing to delete {path}: not inside {REPO_ROOT}")
    if path.is_file():
        path.unlink()
        return
    for child in path.iterdir():
        if child.name in KEEP:
            continue
        if child.is_dir() and not child.is_symlink():
            # rounds are made read-only; directories stay writable so rmtree works
            shutil.rmtree(child)
        else:
            child.unlink()
    print(f"  wiped {path.relative_to(REPO_ROOT)}")


def fresh_start(cfg: PlatformConfig, dagster_home: Path) -> None:
    paths = cfg.paths.resolved()
    targets: list[Path] = [paths.data_dir, paths.reports_dir, paths.artifacts_dir, dagster_home]
    uri = resolve_tracking_uri(cfg.tracking.tracking_uri)
    db = sqlite_path(uri)
    if db is not None:
        targets.append(db)
        art = default_artifact_root(uri)
        if art is not None:
            targets.append(art)
    else:
        print(f"  (tracking store {uri} is not a local SQLite db: MLflow state kept)")
    bad = [t for t in targets if t.exists() and not _inside_repo(t)]
    if bad:
        raise SystemExit(f"refusing --fresh: paths outside the repo: {bad}")
    print("--fresh: removing generated state")
    for t in targets:
        _wipe(t)


# --------------------------------------------------------------------------- summary
def _row(round_id: int, result: Any, elapsed: float) -> dict[str, Any]:
    ds = result.output_for_node("training_dataset")
    tm = result.output_for_node("trained_model")
    ev = result.output_for_node("evaluation_report")
    reg = result.output_for_node("registered_model")
    pr = result.output_for_node("candidate_predictions")
    sel = result.output_for_node("selected_experiments")
    new = result.output_for_node("new_experimental_results")
    return {
        "round": round_key(round_id),
        "obs": ds.n_rows,
        "model": f"v{reg.version} ({reg.stage})",
        "val_rmse": ev.metrics.get("rmse", float("nan")),
        "baseline": ev.baseline_name,
        "gate": "pass" if ev.gate_passed else "fail",
        "scored_by": f"v{pr.model_version} {pr.model_role}",
        "top_score": sel.top_score if sel.top_score is not None else float("nan"),
        "selected": sel.n_selected,
        "new_round": new.round.key + (" (noop)" if new.noop else ""),
        "workers": f"{tm.world_size} pids={','.join(map(str, tm.worker_pids))}",
        "elapsed_s": elapsed,
        "dagster_run": result.run_id[:8],
    }


def print_summary(rows: list[dict[str, Any]], total_s: float) -> None:
    headers = {
        "round": "round",
        "obs": "observations",
        "model": "model version",
        "val_rmse": "val RMSE*",
        "gate": "gate",
        "scored_by": "decisions by",
        "top_score": "top score",
        "selected": "#selected",
        "new_round": "wrote",
        "elapsed_s": "elapsed s",
        "dagster_run": "dagster run",
    }

    def fmt(k: str, v: Any) -> str:
        if k in ("val_rmse", "top_score"):
            return f"{v:.4f}"
        if k == "elapsed_s":
            return f"{v:.1f}"
        return str(v)

    table = [[fmt(k, r[k]) for k in headers] for r in rows]
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(headers.values())]
    line = "  ".join(h.ljust(w) for h, w in zip(headers.values(), widths, strict=True))
    print("\n=== closed-loop summary " + "=" * max(0, len(line) - 24))
    print(line)
    print("  ".join("-" * w for w in widths))
    for t in table:
        print("  ".join(c.ljust(w) for c, w in zip(t, widths, strict=True)))
    print(
        f"\n* validation RMSE in standardized units (the gated metric; gate: <= max_rmse and "
        f"improvement vs baseline). Total {total_s:.1f}s."
    )
    for r in rows:
        print(f"  {r['round']}: baseline={r['baseline']}  DDP world_size={r['workers']}")


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rounds < 0 or (args.rounds == 0 and not args.fresh):
        raise SystemExit("--rounds must be >= 1 (or 0 together with --fresh: only wipe)")
    dagster_home = Path(os.environ.setdefault("DAGSTER_HOME", str(DEFAULT_DAGSTER_HOME)))
    dagster_home.mkdir(parents=True, exist_ok=True)
    cfg_path = resolve_config_path(args.config).resolve()
    cfg = load_config(cfg_path)
    if args.fresh:
        fresh_start(cfg, dagster_home)
        if args.rounds == 0:
            return 0

    # Heavy imports after --fresh (Dagster opens its SQLite storage on import of the instance).
    from dagster import AssetKey, DagsterInstance

    from ml_platform.orchestration import pipeline as P
    from ml_platform.orchestration.definitions import build_definitions
    from ml_platform.orchestration.jobs import run_bootstrap, run_round
    from ml_platform.ray_runtime.cluster import ensure_ray, shutdown_ray

    defs = build_definitions(
        config_path=str(cfg_path), ray_address=args.ray_address, num_workers=args.workers
    )
    store = P.round_store(cfg)
    t_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with DagsterInstance.get() as instance:
        ensure_ray(args.ray_address)
        try:
            bootstrapped = "round_000" in instance.get_materialized_partitions(
                AssetKey("observed_experiments")
            )
            if not store.exists(0) or not bootstrapped:
                print("bootstrap: candidate_pool + observed_experiments[round_000]")
                run_bootstrap(defs, instance, tags={"ml/trigger": "run_closed_loop"})
            start = store.latest_round_id()
            assert start is not None
            print(
                f"closed loop: {args.rounds} round(s) starting at {round_key(start)} "
                f"(config={cfg_path.name}, DAGSTER_HOME={dagster_home})"
            )
            for r in range(start, start + args.rounds):
                t0 = time.perf_counter()
                print(f"\n>>> materializing closed_loop_round_job[{round_key(r)}] ...", flush=True)
                result = run_round(defs, instance, r, tags={"ml/trigger": "run_closed_loop"})
                rows.append(_row(r, result, time.perf_counter() - t0))
                row = rows[-1]
                print(
                    f"<<< {row['round']}: obs={row['obs']} model={row['model']} "
                    f"val_rmse={row['val_rmse']:.4f} gate={row['gate']} "
                    f"selected={row['selected']} -> {row['new_round']} "
                    f"({row['elapsed_s']:.1f}s)",
                    flush=True,
                )
        finally:
            shutdown_ray()
    print_summary(rows, time.perf_counter() - t_start)
    print(
        "\nInspect: `make dagster` (Assets → lineage, Runs), `make mlflow` (runs, registry), "
        f"reports in {cfg.paths.resolved().reports_dir}, rounds in {store.rounds_dir}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
