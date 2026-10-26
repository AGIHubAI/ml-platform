"""Create the candidate pool and the initial observations (round_000).

Usage::

    uv run python scripts/bootstrap.py [--config configs/local.yaml] [--plain]

By default this materializes Dagster's ``bootstrap_job`` (``candidate_pool`` +
``observed_experiments[round_000]``) against the persistent instance in
``$DAGSTER_HOME`` (default ``./dagster_home``), so the data shows up with
lineage in ``make dagster``. ``--plain`` calls the same pipeline functions
without Dagster. Both are idempotent: the pool and round 0 are write-once.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from ml_platform.config import REPO_ROOT, load_config, resolve_config_path
from ml_platform.orchestration import pipeline as P


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None, help="config file/name (default: $ML_CONFIG or local)")
    p.add_argument("--plain", action="store_true", help="run without Dagster")
    a = p.parse_args(argv)
    cfg_path = resolve_config_path(a.config).resolve()
    cfg = load_config(cfg_path)
    store = P.round_store(cfg)
    if a.plain:
        pool = P.ensure_candidate_pool(cfg, store)
        round0 = P.observe_round(cfg, store, 0)
    else:
        os.environ.setdefault("DAGSTER_HOME", str(REPO_ROOT / "dagster_home"))
        os.makedirs(os.environ["DAGSTER_HOME"], exist_ok=True)
        from dagster import DagsterInstance

        from ml_platform.orchestration.definitions import build_definitions
        from ml_platform.orchestration.jobs import run_bootstrap

        with DagsterInstance.get() as instance:
            result = run_bootstrap(build_definitions(config_path=str(cfg_path)), instance)
        pool = result.output_for_node("candidate_pool")
        round0 = result.output_for_node("observed_experiments")
    print(
        f"candidate pool : {pool.n_candidates} candidates  hash={pool.pool_hash[:16]}  {pool.path}"
    )
    print(
        f"round_000      : {round0.n_records} observations  "
        f"manifest={round0.manifest_hash[:16]}  {round0.path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
