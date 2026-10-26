"""fUSI real-time workload through Dagster: sessions -> ... -> gated registry.

Usage::

    uv run python scripts/run_fusi_pipeline.py [--config configs/local.yaml]

Materialises ``fusi_sessions_job`` once per session partition, then ``fusi_train_job``
(split -> preprocessor -> dataset + leakage checks -> Ray Train DDP decoder ->
evaluation + real-time latency + gate -> registry), in-process against the persistent
instance in ``$DAGSTER_HOME`` (default ``./dagster_home``). Prints the check results,
accuracies, latency table and the registry decision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from ml_platform.config import REPO_ROOT, load_config, resolve_config_path

DEFAULT_DAGSTER_HOME = REPO_ROOT / "dagster_home"
DEFAULT_CONFIG = "fusi-micro"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        default=None,
        help="config file/name (default: $ML_CONFIG or configs/fusi-micro.yaml)",
    )
    args = p.parse_args(argv)
    cfg_path = resolve_config_path(args.config or os.environ.get("ML_CONFIG") or DEFAULT_CONFIG)
    cfg = load_config(cfg_path)
    # DAGSTER_HOME / Ray temp + spill / Dask temp from the config (external drive for
    # fusi-micro) before Dagster or Ray start; fails if that drive is not mounted
    exported = cfg.paths.apply_runtime_env()
    os.environ.setdefault("DAGSTER_HOME", str(DEFAULT_DAGSTER_HOME))
    Path(os.environ["DAGSTER_HOME"]).mkdir(parents=True, exist_ok=True)
    print("storage:", {k: v for k, v in exported.items()} or "tool defaults")

    from dagster import DagsterInstance

    from ml_platform.orchestration import fusi_pipeline as FP
    from ml_platform.orchestration.definitions import build_definitions
    from ml_platform.orchestration.fusi_assets import FUSI_TRAIN_ASSETS, run_fusi

    defs = build_definitions(config_path=str(cfg_path))
    keys = FP.session_keys(cfg)
    t0 = time.perf_counter()
    with DagsterInstance.get() as instance:
        print(f"fusi: {len(keys)} sessions, grid {cfg.fusi.sessions.grid}, config {cfg_path.name}")
        _, train = run_fusi(defs, instance, keys)
        print(f"\nDagster run {train.run_id[:8]}  success={train.success}")
        for ev in train.get_asset_check_evaluations():
            print(f"  check {ev.check_name:<34} {'PASS' if ev.passed else 'FAIL'}")
        done = {e.step_key for e in train.all_events if e.event_type_value == "STEP_SUCCESS"}
        blocked = [
            a.key.to_user_string() for a in FUSI_TRAIN_ASSETS if a.key.to_user_string() not in done
        ]
        if blocked:
            print(f"  not materialised (failed step or blocking check): {', '.join(blocked)}")
        try:
            ev_info = train.output_for_node("fusi_evaluation")
        except Exception:
            ev_info = None
        if ev_info is not None:
            print("\naccuracy:", json.dumps({k: round(v, 3) for k, v in ev_info.accuracy.items()}))
            print(f"reference: {ev_info.reference} = {ev_info.reference_acc:.3f}")
            print((ev_info.report_path / "latency.txt").read_text())
            print(f"gate passed: {ev_info.gate_passed}")
            for r in ev_info.gate_reasons:
                print(f"  - {r}")
        try:
            reg = train.output_for_node("fusi_registered_decoder")
            print(f"registry: {cfg.fusi.tracking.registered_model} v{reg.version} -> {reg.stage}")
        except Exception:
            print("registry: not reached")
    from ml_platform.ray_runtime.cluster import shutdown_ray

    shutdown_ray()
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0 if train.success else 1


if __name__ == "__main__":
    sys.exit(main())
