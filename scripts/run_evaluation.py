"""Evaluate a model on the latest training data and write an evaluation report.

Usage::

    uv run python scripts/run_evaluation.py                     # production (else newest) model
    uv run python scripts/run_evaluation.py --model-version 3   # a registered version
    uv run python scripts/run_evaluation.py --checkpoint artifacts/ray_train/<run>/checkpoint_...
    uv run python scripts/run_evaluation.py --round 1 --no-mlflow

The model is evaluated on the validation split of union(round_000..round_N)
(default: latest round), paired against the production model when it is a
different model, else against a train-mean baseline. Writes ``metrics.json``,
``report.md``, ``predictions.parquet`` and ``comparison.json`` to
``reports/round_XXX/eval_<model>_<timestamp>/`` and logs them as an MLflow
``evaluation`` run. Evaluation never promotes a model — promotion happens in
the pipeline's ``registered_model`` step through the registry gates.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from ml_platform.config import load_config
from ml_platform.data import round_key
from ml_platform.orchestration import pipeline as P
from ml_platform.tracking import ModelRegistry, Tracker


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None, help="config file/name (default: $ML_CONFIG or local)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--model-version", default=None, help="registered model version")
    src.add_argument("--checkpoint", default=None, help="checkpoint directory or model.pt")
    p.add_argument(
        "--round", type=int, default=None, help="evaluate on rounds 0..ROUND (default: latest)"
    )
    p.add_argument("--no-mlflow", action="store_true", help="do not log an MLflow run")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    store = P.round_store(cfg)
    round_id = store.latest_round_id() if a.round is None else a.round
    if round_id is None:
        p.error(f"no experimental rounds in {store.root}; run `make generate-data` first")
    tracker = Tracker(cfg.tracking)
    registry = ModelRegistry(tracker=tracker)

    run_id: str | None = None
    if a.checkpoint:
        ckpt, label = Path(a.checkpoint), "checkpoint"
    else:
        version = a.model_version
        if version is None:
            prod = registry.production_version()
            versions = registry.list_versions()
            if prod is not None:
                version = prod.version
            elif versions:
                version = versions[-1]["version"]
            else:
                p.error("no registered models yet; run `make closed-loop` or `make train` first")
        ckpt, label = registry.checkpoint_path(str(version)), f"v{version}"
        run_id = registry.version_tags(str(version)).get("run_id")

    evaluation = P.evaluate_checkpoint(
        cfg, store, registry, ckpt, round_id, candidate_run_id=run_id
    )
    out_dir = (
        cfg.paths.resolved().reports_dir
        / round_key(round_id)
        / f"eval_{label}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    if a.no_mlflow:
        evaluation.write(out_dir)
    else:
        tags = {
            "stage": "evaluation",
            "round_id": round_id,
            "dataset_id": store.dataset_hash(round_id),
            "evaluated_model": label,
            "source_run_id": run_id,
        }
        with tracker.start_run(f"evaluate_{label}_{round_key(round_id)}", tags=tags):
            tracker.log_evaluation(evaluation, out_dir)

    m = evaluation.metrics
    print(f"\n=== evaluation of {label} on {round_key(round_id)} (n={int(m['n_eval'])}) ===")
    print(
        f"  rmse (std units)   {m['rmse']:.4f}  95% CI [{m['rmse_ci_lo']:.4f}, {m['rmse_ci_hi']:.4f}]"
    )
    print(f"  baseline           {evaluation.baseline_name}  rmse={m['baseline_rmse']:.4f}")
    print(f"  improvement        {m['improvement_vs_baseline']:+.3%}")
    print(f"  r2 / coverage_95   {m['r2']:.4f} / {m['coverage_95']:.3f}")
    print(f"  gate               {'PASSED' if evaluation.gate.passed else 'FAILED'}")
    for reason in evaluation.gate.reasons:
        print(f"    - {reason}")
    print(f"  report             {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
