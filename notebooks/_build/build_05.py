"""Generate notebooks/05_dagster_scientific_workflows.ipynb (notebooks-c).

    uv run python notebooks/_build/build_05.py            # write the .ipynb (no outputs)
    uv run python notebooks/_build/build_05.py --execute  # write + execute in place
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbhelpers_b import code, main, md, preamble  # noqa: E402

EXTRA = """
os.environ.pop("MLFLOW_TRACKING_URI", None); os.environ.pop("RAY_ADDRESS", None)   # stay fully local
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "fatal")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine")
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
"""

cells = [
    md(r"""
    # 05 · Dagster for scientific workflows

    **Goal.** Understand what a workflow orchestrator is *for* in a research platform, using
    Dagster's vocabulary — assets, ops, jobs, resources, partitions, sensors, schedules,
    materializations, lineage, retries — and see exactly how the repository's closed loop
    uses each one. We first build a toy pipeline small enough to read in one screen, then run
    the platform's real asset graph for two experimental rounds (Ray Train DDP, MLflow,
    gated registry, Ray actors) and interrogate the Dagster instance for lineage.

    | Section | Content |
    |---|---|
    | 1 | Conceptual model: assets vs ops/jobs; Dagster vs Ray |
    | 2 | Why an experimental dataset is an *asset*, not a job step |
    | 3 | Retry semantics — a small derivation |
    | 4 | Toy: resource + dynamic partitions + `materialize` + metadata |
    | 5 | Toy: a flaky asset under a `RetryPolicy` |
    | 6 | Toy: ops and jobs; a sensor evaluated with `build_sensor_context` |
    | 7 | The real graph: bootstrap + two rounds on a tiny config |
    | 8 | Lineage across rounds from the instance's event log |
    | 9 | Idempotent re-materialization of `new_experimental_results` |
    | 10 | The repo's sensor and schedule, evaluated against the live instance |
    | 11–13 | Connection to the repo · failure modes · exercise |
    """),
    md(r"""
    ## 1. Conceptual model

    **Ops and jobs** describe *computations*: a job is a DAG of ops, and running it is a
    *run*. That is the model of Airflow-style task schedulers: "execute step B after step A".

    **Software-defined assets** describe *things that should exist*: a dataset, a model, a
    report. An asset declares its upstream assets and a function that (re)computes it.
    Executing the function is a **materialization**, recorded with metadata in the instance's
    event log. Jobs still exist — `define_asset_job` selects a subset of assets — but the
    durable object is the asset and its history, not the run.

    ```text
    ops/jobs  (verbs):  run#41: load → clean → train → eval        "did the job succeed?"
    assets    (nouns):  training_dataset[round_002] ──► trained_model[round_002]
                           materialized by run#41, dataset_hash=9c1f…, 820 rows
                                                       "which model came from which data?"
    ```

    | Concept | In this repo |
    |---|---|
    | asset | 9 assets, `candidate_pool → … → new_experimental_results` (`orchestration/assets.py`) |
    | partition | dynamic partition set `rounds` = `round_000, round_001, …` (`partitions.py`) |
    | resource | `PlatformConfigResource`, `RoundStoreResource`, `TrackingResource`, `RayComputeResource` (`resources.py`) |
    | job | `bootstrap_job`, `closed_loop_round_job`, `retrain_job` (`jobs.py`) |
    | sensor | `new_round_sensor`: a new round appears in the RoundStore → run the loop for it |
    | schedule | `nightly_retrain_schedule`: `0 2 * * *` retrain/re-evaluate the latest round |
    | materialization metadata | dataset/manifest hashes, MLflow run id + URL, model version, gate decision, selection hash |
    | retry policy | exponential backoff on compute assets; **none** on `new_experimental_results` |

    ### Dagster scheduling vs Ray scheduling

    Both are "schedulers", at different altitudes:

    | | Dagster | Ray |
    |---|---|---|
    | decides | **what** to (re)compute and **when**; in which order; for which round | **where** each piece runs and on **how many** workers/CPUs/GPUs |
    | unit | asset materialization / step (minutes–hours) | task / actor method call (ms–minutes) |
    | state | durable: runs, events, lineage, partitions in `$DAGSTER_HOME` | ephemeral: object store, actor state; gone after `ray.shutdown()` |
    | failure handling | re-run a whole *step* (RetryPolicy), or a human re-materializes | restart a task/actor/worker group; Ray Train restores from the last checkpoint |
    | in this repo | `trained_model` says "train on `training_dataset[round_r]` with 2 workers" | `TorchTrainer` places 2 DDP workers, sets rank/world size, restarts on worker loss |

    Dagster calls Ray *through a resource* (`RayComputeResource.client()` → `RayCompute`), so
    no asset knows whether Ray is local, a Docker Compose head node or a KubeRay cluster.
    """),
    md(r"""
    ## 2. Why an experimental dataset is an asset, not a job step

    In a job-step design, "run experiments" is a function call inside run #41 and its output
    is an intermediate value. In the asset design, `round_003` is a named, partitioned,
    versioned object:

    * **Identity & lineage.** Every model, report and selection can point at
      `observed_experiments[round_003]` and its `manifest_hash`; the UI can answer "which
      models were trained on data that includes round 3?".
    * **Different lifecycles.** A round is produced once (expensive, irreproducible noise) but
      *consumed* many times — by every later training run, the nightly retrain, ad-hoc
      re-evaluation. Steps are ephemeral; datasets outlive the runs that made them.
    * **Arrival is external.** In a real lab, round 3 arrives when the plate reader finishes,
      possibly days later and from outside the platform. An asset can be materialized by
      whoever produces it; a sensor reacts to it appearing. A job step cannot "arrive".
    * **Staleness.** When code or config changes, Dagster can tell that `trained_model` is
      stale relative to its upstream — but it must *never* recompute `round_003` by re-running
      the experiment. Modelling the round as an asset with **no retry** and a write-once store
      makes that explicit.
    * **Partitions = rounds.** Dynamic partitions give "one materialization per round" for
      free: backfills, per-round status, and a natural key shared with `data/rounds/round_XXX`.

    The loop closes *across* partitions: `new_experimental_results[round_r]` writes
    `round_{r+1}` into the RoundStore and registers the new partition key. A static asset edge
    back to `observed_experiments` would be a cycle; the RoundStore is the hand-off point.
    """),
    md(r"""
    ## 3. Retry semantics — a small derivation

    Suppose a step fails transiently with probability $p$ per attempt, independently, and has a
    retry policy with $k$ retries. Then

    $$
    P(\text{success}) = 1 - p^{\,k+1}, \qquad
    \mathbb E[\text{attempts}] = \sum_{i=0}^{k} p^{i} = \frac{1 - p^{k+1}}{1 - p}.
    $$

    With $p = 0.1$ and $k = 2$ (the repo's `COMPUTE_RETRY`), success rises from 90 % to 99.9 %
    at an expected cost of 1.11 attempts. For a **pure, deterministic** computation (training
    on a fixed dataset with fixed seeds, scoring a pool) a retry is harmless: every attempt
    produces the same thing, so *at-least-once* execution is fine.

    A physical experiment is different. If a failure happens *after* the samples are consumed
    but *before* the results are persisted (probability $q$ per attempt), a blind retry
    policy runs the experiment again. The expected number of physical executions becomes
    $\sum_{i=0}^{k} q^{i}$, and each costs money, time and — worse — produces a *different*
    noisy measurement, so the recorded result depends on how many times it failed. The
    platform therefore wants **at-most-once** physical execution + **idempotent** persistence:

    * `new_experimental_results` has **no** `RetryPolicy` → a failure goes to a human;
    * measurements are journaled by the `ExperimentSimulator` actor, so a re-run replays the
      recorded values instead of re-measuring;
    * the RoundStore is write-once: same content → no-op, different content → refused.
    """),
    code(r"""
    for p in (0.05, 0.1, 0.3):
        for k in (0, 2):
            print(f"p={p:<4}  retries={k}:  P(success)={1 - p**(k + 1):.4f}   "
                  f"E[attempts]={(1 - p**(k + 1)) / (1 - p):.3f}")
    """),
    preamble("nb05"),
    code(EXTRA),
    md(r"""
    ## 4. Toy pipeline: a resource, dynamic partitions, `materialize`, metadata

    A self-contained miniature of the platform: a `Lab` **resource** (the only thing that knows
    how to "measure"), a **dynamic partition set** of rounds, and two assets. We materialize
    into an ephemeral `DagsterInstance` whose storage lives in our temp dir.
    """),
    code(r"""
    import dagster as dg
    from dagster import (AssetExecutionContext, AssetKey, AssetSelection, Backoff, ConfigurableResource,
                         DagsterInstance, DynamicPartitionsDefinition, MetadataValue, Output, RetryPolicy,
                         RunRequest, SensorEvaluationContext, SensorResult, SkipReason, asset,
                         build_sensor_context, job, materialize, op, sensor)

    print("dagster", dg.__version__)
    QUIET = {"python_logs": {"python_log_level": "WARNING"},          # keep DEBUG event logs out of the notebook
             "telemetry": {"enabled": False}}
    toy_instance = DagsterInstance.ephemeral(tempdir=str(WORK / "toy_dagster"), settings=QUIET)

    toy_rounds = DynamicPartitionsDefinition(name="toy_rounds")

    class ToyLab(ConfigurableResource):
        noise: float = 0.1
        def measure(self, round_id: int, n: int = 8) -> list[float]:
            rng = np.random.default_rng(round_id)
            return (np.sin(rng.uniform(-2, 2, n)) + self.noise * rng.normal(size=n)).round(4).tolist()

    @asset(partitions_def=toy_rounds, description="Immutable measurements of one round.")
    def toy_measurements(context: AssetExecutionContext, lab: ToyLab) -> Output[list]:
        r = int(context.partition_key.removeprefix("round_"))
        ys = lab.measure(r)
        return Output(ys, metadata={"dagster/row_count": len(ys),
                                    "mean": MetadataValue.float(float(np.mean(ys))),
                                    "content_hash": MetadataValue.text(str(hash(tuple(ys)) & 0xFFFFFFFF))})

    @asset(partitions_def=toy_rounds)
    def toy_model(context: AssetExecutionContext, toy_measurements: list) -> Output[float]:
        best = float(max(toy_measurements))
        return Output(best, metadata={"best": MetadataValue.float(best),
                                      "n_inputs": len(toy_measurements)})

    toy_instance.add_dynamic_partitions("toy_rounds", ["round_000", "round_001"])
    for key in ("round_000", "round_001"):
        res = materialize([toy_measurements, toy_model], partition_key=key, instance=toy_instance,
                          resources={"lab": ToyLab(noise=0.1)})
        print(key, "success:", res.success, "| toy_model value:", res.output_for_node("toy_model"))
    """),
    md(r"""
    Every materialization is an event in the instance, with the metadata we attached. This is
    the *lineage record*: which run produced which partition of which asset, with what content.
    """),
    code(r"""
    def materializations(instance, asset_name, limit=20):
        recs = instance.fetch_materializations(AssetKey(asset_name), limit=limit).records
        return [(r.asset_materialization.partition, r.run_id[:8],
                 {k: v.value for k, v in r.asset_materialization.metadata.items() if k != "path"})
                for r in sorted(recs, key=lambda r: r.storage_id)]

    for name in ("toy_measurements", "toy_model"):
        for partition, run, meta in materializations(toy_instance, name):
            print(f"{name:<17} {partition}  run={run}  {meta}")
    print("materialized partitions of toy_model:", sorted(toy_instance.get_materialized_partitions(AssetKey("toy_model"))))
    """),
    md(r"""
    ## 5. A flaky asset under a `RetryPolicy`

    `toy_fit` fails on its first two attempts (think: a Ray worker was pre-empted) and succeeds
    on the third. With `RetryPolicy(max_retries=3, backoff=EXPONENTIAL)` Dagster re-executes
    only the failed *step*, re-loading its inputs from the IO manager — the upstream
    measurements are not recomputed.
    """),
    code(r"""
    CALLS = {"toy_measurements_2": 0, "toy_fit": 0}

    @asset(partitions_def=toy_rounds, name="toy_measurements_2")
    def toy_measurements_2(context: AssetExecutionContext, lab: ToyLab) -> list:
        CALLS["toy_measurements_2"] += 1
        return lab.measure(int(context.partition_key.removeprefix("round_")))

    @asset(partitions_def=toy_rounds,
           retry_policy=RetryPolicy(max_retries=3, delay=0.2, backoff=Backoff.EXPONENTIAL))
    def toy_fit(context: AssetExecutionContext, toy_measurements_2: list) -> Output[float]:
        CALLS["toy_fit"] += 1
        if context.retry_number < 2:
            raise RuntimeError(f"transient failure on attempt {context.retry_number}")
        return Output(float(np.mean(toy_measurements_2)), metadata={"retry_number": context.retry_number})

    t0 = time.perf_counter()
    res = materialize([toy_measurements_2, toy_fit], partition_key="round_000", instance=toy_instance,
                      resources={"lab": ToyLab()})
    events = [e.event_type_value for e in res.all_events if e.step_key == "toy_fit"]
    print("success:", res.success, f"in {time.perf_counter() - t0:.1f}s | calls:", CALLS)
    print("toy_fit step events:", [e for e in events if e in ("STEP_START", "STEP_UP_FOR_RETRY", "STEP_RESTARTED",
                                                             "STEP_SUCCESS", "ASSET_MATERIALIZATION")])
    print("materialization metadata:", materializations(toy_instance, "toy_fit")[-1][2])
    """),
    code(r"""
    # Without a retry policy, the same failure fails the run -- and nothing is materialized.
    @asset(partitions_def=toy_rounds, name="toy_fit_no_retry")
    def toy_fit_no_retry(context: AssetExecutionContext, toy_measurements_2: list) -> float:
        raise RuntimeError("the plate reader jammed")

    import contextlib, io
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):                     # Dagster logs the full stack trace at ERROR
        res = materialize([toy_measurements_2, toy_fit_no_retry], partition_key="round_001",
                          instance=toy_instance, resources={"lab": ToyLab()}, raise_on_error=False)
    print("\n".join(l[:140] for l in captured.getvalue().splitlines() if " - ERROR - " in l))
    print("success:", res.success, "| failed steps:", [e.step_key for e in res.get_step_failure_events()])
    print("toy_fit_no_retry materialized partitions:",
          toy_instance.get_materialized_partitions(AssetKey("toy_fit_no_retry")) or "none")
    """),
    md(r"""
    ## 6. Ops and jobs; sensors

    For contrast, the same computation as an **op-based job**. It runs and succeeds, but the
    instance records a *run*, not an asset — there is nothing to ask "is `toy_model` for
    round 1 up to date?" about. In the repo, ops appear only implicitly (every asset is backed
    by an op), and jobs are *asset selections* (`define_asset_job`).
    """),
    code(r"""
    @op
    def measure_op() -> list:
        return ToyLab().measure(0)

    @op
    def fit_op(ys: list) -> float:
        return float(max(ys))

    @job
    def toy_op_job():
        fit_op(measure_op())

    res = toy_op_job.execute_in_process(instance=toy_instance)
    print("op job success:", res.success, "| fit_op output:", res.output_for_node("fit_op"))
    print("asset materializations recorded by the op job:", len(res.get_asset_materialization_events()))
    """),
    md(r"""
    A **sensor** is a function the Dagster daemon evaluates periodically. It inspects the
    world (here: the lab's output directory), and returns run requests — plus, for dynamic
    partitions, *requests to add partition keys*. The cursor makes it incremental. We can
    evaluate it directly with `build_sensor_context`, which is how the repo unit-tests its
    sensor.
    """),
    code(r"""
    LAB_DIR = WORK / "toy_lab_output"
    LAB_DIR.mkdir()

    @sensor(asset_selection=AssetSelection.assets(toy_measurements, toy_model))
    def toy_round_sensor(context: SensorEvaluationContext):
        seen = int(context.cursor) if context.cursor else -1
        arrived = sorted(int(p.stem.removeprefix("round_")) for p in LAB_DIR.glob("round_*.done"))
        fresh = [r for r in arrived if r > seen]
        if not fresh:
            return SkipReason(f"nothing new (cursor={seen})")
        keys = [f"round_{r:03d}" for r in fresh]
        return SensorResult(run_requests=[RunRequest(partition_key=k, run_key=k) for k in keys],
                            dynamic_partitions_requests=[toy_rounds.build_add_request(keys)],
                            cursor=str(max(fresh)))

    with build_sensor_context(instance=toy_instance, cursor="1") as ctx:
        print("tick 1:", toy_round_sensor(ctx))
    (LAB_DIR / "round_002.done").touch()
    with build_sensor_context(instance=toy_instance, cursor="1") as ctx:
        out = toy_round_sensor(ctx)
    print("tick 2: run requests", [(r.partition_key, r.run_key) for r in out.run_requests],
          "| add partitions", [r.partition_keys for r in out.dynamic_partitions_requests], "| cursor", out.cursor)
    """),
    md(r"""
    ## 7. The real graph: bootstrap + two closed-loop rounds

    Now the platform itself, through its own orchestration entry points
    (`src/ml_platform/orchestration/`):

    * `build_definitions(config_path=…, num_workers=2, n_inference_actors=1)` wires the 9 assets,
      3 jobs, sensor, schedule and the four resources (dependency injection: we point the
      config resource at a tiny YAML in our temp dir);
    * `run_bootstrap(defs, instance)` materializes `candidate_pool` + `observed_experiments[round_000]`;
    * `run_round(defs, instance, r)` registers the partition and executes
      `closed_loop_round_job` for `round_r` — one Dagster run per round.

    Everything (RoundStore, MLflow SQLite, reports, Ray Train storage, and a persistent
    SQLite-backed `DagsterInstance.local_temp` standing in for `$DAGSTER_HOME`) goes to the temp dir. Ray runs locally with 4 CPUs and a private, deletable session directory.
    """),
    code(r"""
    import json, shutil
    import ray
    from ml_platform.config import PlatformConfig
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):   # MLflow import-time warning
        from ml_platform.orchestration import pipeline as P
        from ml_platform.orchestration.definitions import build_definitions
    from ml_platform.orchestration.jobs import ROUND_JOB, run_bootstrap, run_round
    from ml_platform.ray_runtime.cluster import ensure_ray

    RAY_TMP = tempfile.mkdtemp(prefix="nbc_ray_", dir="/tmp")    # short path (unix sockets) we delete at the end
    ensure_ray(num_cpus=4, include_dashboard=False, log_to_driver=False,
               object_store_memory=300 * 1024**2, _temp_dir=RAY_TMP)

    cfg = PlatformConfig.for_tests(WORK / "platform", **{
        "training.epochs": 3, "evaluation.bootstrap_samples": 100,
        "evaluation.mc_samples": 5, "data.batch_size_per_round": 20})
    cfg_path = P.write_config(cfg, WORK / "platform" / "config.yaml")
    defs = build_definitions(config_path=str(cfg_path), num_workers=2, n_inference_actors=1)
    # a real, persistent (SQLite) instance -- like $DAGSTER_HOME -- but in our temp dir
    (WORK / "dagster_home").mkdir()
    instance = DagsterInstance.local_temp(tempdir=str(WORK / "dagster_home"), overrides=QUIET)

    logging.getLogger("mlflow").setLevel(logging.ERROR)
    RUN_LOG = io.StringIO()          # MLflow/Ray INFO chatter goes here instead of the notebook
    t0 = time.perf_counter()
    with contextlib.redirect_stderr(RUN_LOG):
        boot = run_bootstrap(defs, instance)
    print(f"bootstrap_job: success={boot.success} ({time.perf_counter() - t0:.1f}s)")
    results = {}
    for r in (0, 1):
        t1 = time.perf_counter()
        with contextlib.redirect_stderr(RUN_LOG):
            results[r] = run_round(defs, instance, r)
        print(f"closed_loop_round_job[round_{r:03d}]: success={results[r].success} "
              f"({time.perf_counter() - t1:.1f}s)")
    errors = [l for l in RUN_LOG.getvalue().splitlines() if "ERROR" in l or "Traceback" in l]
    print(f"captured {len(RUN_LOG.getvalue().splitlines())} log lines, {len(errors)} errors")
    print("dynamic partitions now:", instance.get_dynamic_partitions("rounds"))
    """),
    md(r"""
    `round_002` exists as a partition although we never ran round 2: `new_experimental_results[round_001]`
    wrote it to the RoundStore and registered the key — that is the hand-off the
    `new_round_sensor` would pick up in a deployed system.

    ## 8. Lineage across rounds, from the instance's event log

    Everything below comes from `instance.fetch_materializations` — i.e. what the Dagster UI
    shows — not from the Python return values.
    """),
    code(r"""
    def meta_of(asset_name):
        out = {}
        recs = instance.fetch_materializations(AssetKey(asset_name), limit=50).records
        for r in sorted(recs, key=lambda r: r.storage_id):          # oldest first -> latest wins
            m = r.asset_materialization
            out[m.partition] = {k: v.value for k, v in m.metadata.items()} | {"_run": r.run_id[:8]}
        return out

    M = {a: meta_of(a) for a in ("observed_experiments", "training_dataset", "trained_model", "evaluation_report",
                                 "registered_model", "candidate_predictions", "selected_experiments",
                                 "new_experimental_results")}
    short = lambda h: str(h)[:10]
    rows = []
    for key in ("round_000", "round_001"):
        rows.append({
            "round": key,
            "dagster run": M["trained_model"][key]["_run"],
            "observed manifest": short(M["observed_experiments"][key]["manifest_hash"]),
            "parent of observed": short(M["observed_experiments"][key]["parent_hashes"]),
            "training rows": M["training_dataset"][key]["dagster/row_count"],
            "dataset_hash": short(M["training_dataset"][key]["dataset_hash"]),
            "model dataset_hash": short(M["trained_model"][key]["dataset_hash"]),
            "mlflow run": short(M["trained_model"][key]["mlflow_run_id"]),
            "val_rmse": round(M["trained_model"][key]["val_rmse"], 3),
            "gate": M["evaluation_report"][key]["gate_passed"],
            "model version": M["registered_model"][key]["model_version"],
            "stage": M["registered_model"][key]["lifecycle_stage"],
            "scored with": f'v{M["candidate_predictions"][key]["model_version"]} ({M["candidate_predictions"][key]["model_role"]})',
            "selection_hash": short(M["selected_experiments"][key]["selection_hash"]),
            "wrote round": M["new_experimental_results"][key]["new_round"],
            "new round manifest": short(M["new_experimental_results"][key]["manifest_hash"]),
        })
    lineage = pd.DataFrame(rows).set_index("round").T
    lineage
    """),
    md(r"""
    With the tiny config (3 epochs, 300 rows) no model passes the absolute RMSE gate, so both
    versions stay `candidate` and the decision step records `model_role = research-candidate`:
    an unvalidated model may choose the *next experiments* (a research decision, recorded in the
    round's provenance) but is never promoted or served. With `configs/local.yaml` the gate
    typically passes after a round or two and scoring switches to the `production` alias.
    """),
    code(r"""
    L = {k: M["new_experimental_results"][k] for k in ("round_000", "round_001")}
    checks = {
        "model trained on exactly its dataset (per round)":
            all(M["trained_model"][k]["dataset_hash"] == M["training_dataset"][k]["dataset_hash"] for k in L),
        "round_001 observations are round_000's lab output":
            M["observed_experiments"]["round_001"]["manifest_hash"] == L["round_000"]["manifest_hash"],
        "round_001 chains to round_000 (parent hash)":
            M["observed_experiments"]["round_001"]["parent_hashes"] == M["observed_experiments"]["round_000"]["manifest_hash"],
        "the lab ran exactly the selected batch (selection_hash carried through)":
            all(L[k]["selection_hash"] == M["selected_experiments"][k]["selection_hash"] for k in L),
        "dataset grows by one batch per round":
            M["training_dataset"]["round_001"]["dagster/row_count"]
            == M["training_dataset"]["round_000"]["dagster/row_count"] + cfg.data.batch_size_per_round,
    }
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    assert all(checks.values())
    print("\nprovenance written into round_002's manifest by the lab asset:")
    print(json.dumps(L["round_001"]["provenance"], indent=1))
    """),
    md(r"""
    The same identifiers are on the MLflow side: the training run of each round is tagged with
    the Dagster run id and partition and with `dataset_id` = the RoundStore hash, so one can
    navigate Dagster → MLflow → checkpoint and back.
    """),
    code(r"""
    from ml_platform.tracking import Tracker

    tracker = Tracker(cfg.tracking)
    for key in ("round_000", "round_001"):
        run = tracker.get_run(M["trained_model"][key]["mlflow_run_id"])
        t = run.data.tags
        print(f"{key}: mlflow run {run.info.run_id[:10]}  dagster_partition={t.get('dagster_partition')}  "
              f"dagster_run_id={t.get('dagster_run_id', '')[:8]}  dataset_id={t.get('dataset_id', '')[:10]}  "
              f"eval_rmse={run.data.metrics.get('eval_rmse', float('nan')):.3f}")
    """),
    md(r"""
    ## 9. Re-materializing the experiment asset is a no-op

    Ask Dagster to materialize **only** `new_experimental_results[round_001]` again — the
    situation after an operator clicks "Materialize" on a step whose run crashed late. The
    upstream `selected_experiments[round_001]` is loaded from the IO manager, the RoundStore
    already contains `round_002` for exactly that selection, so no experiment is run and no
    data changes. A *different* selection for an already-measured round is refused.
    """),
    code(r"""
    import dataclasses
    from ml_platform.data import ImmutableRoundError

    store = P.round_store(cfg)
    before = store.read_manifest(2).manifest_hash
    journal = cfg.paths.resolved().data_dir / P.LAB_JOURNAL
    journal_lines = len(journal.read_text().splitlines())

    again = defs.resolve_job_def(ROUND_JOB).execute_in_process(
        partition_key="round_001", instance=instance, asset_selection=[AssetKey("new_experimental_results")])
    meta = meta_of("new_experimental_results")["round_001"]
    print("re-materialization success:", again.success, "| run", again.run_id[:8])
    print("  idempotent_noop =", meta["idempotent_noop"], "| n_physical_measurements =", meta["n_physical_measurements"])
    print("  round_002 manifest unchanged:", store.read_manifest(2).manifest_hash == before,
          "| lab journal lines unchanged:", len(journal.read_text().splitlines()) == journal_lines)
    print("  materializations of new_experimental_results[round_001]:",
          sum(1 for r in instance.fetch_materializations(AssetKey("new_experimental_results"), limit=50).records
              if r.asset_materialization.partition == "round_001"))

    # A *different* batch for round_002 (e.g. chosen by a model re-trained after the lab ran) is refused.
    from ml_platform.active_learning import select_batch, write_selection
    sel = results[1].output_for_node("selected_experiments")
    preds = pd.read_parquet(results[1].output_for_node("candidate_predictions").path)
    alt = select_batch(store.read_pool(), preds, store.observed_ids(1), cfg, round_id=2, beta=5.0,
                       model_version=sel.model_version)
    write_selection(alt, WORK / "alt_selection")
    alt_info = dataclasses.replace(sel, path=WORK / "alt_selection", selection_hash=alt.selection_hash,
                                   candidate_ids=tuple(alt.candidate_ids))
    print(f"  alternative selection {alt.selection_hash[:12]} overlaps the measured batch in "
          f"{len(set(alt.candidate_ids) & set(sel.candidate_ids))}/{len(sel.candidate_ids)} candidates")
    try:
        P.run_experiments(cfg, store, P.RayCompute(), alt_info)
    except ImmutableRoundError as e:
        print("  refused:", e)
    """),
    md(r"""
    ## 10. The repo's sensor and schedule against the live instance

    `new_round_sensor` scans the RoundStore; with the cursor at round 1 it sees `round_002`
    and requests a `closed_loop_round_job` run for it (the `run_key` includes the manifest hash,
    so a round is processed once per content version). `nightly_retrain_schedule` targets the
    latest registered round. Both are `STOPPED` by default in the deployed code location.
    """),
    code(r"""
    from dagster import build_schedule_context
    from ml_platform.orchestration.resources import PlatformConfigResource, RoundStoreResource
    from ml_platform.orchestration.sensors import new_round_sensor, nightly_retrain_schedule

    rs = {"round_store": RoundStoreResource(config=PlatformConfigResource(config_path=str(cfg_path)))}
    with build_sensor_context(instance=instance, resources=rs, cursor="1") as ctx:
        out = new_round_sensor(ctx)
    print("new_round_sensor (cursor=1):", [(r.partition_key, r.run_key, r.tags) for r in out.run_requests],
          "| next cursor:", out.cursor)
    with build_sensor_context(instance=instance, resources=rs, cursor="2") as ctx:
        print("new_round_sensor (cursor=2):", new_round_sensor(ctx))
    with build_schedule_context(instance=instance, resources=rs) as ctx:
        req = nightly_retrain_schedule(ctx)
    print("nightly_retrain_schedule ->", getattr(req, "partition_key", req), "| job: retrain_job",
          "| default status:", nightly_retrain_schedule.default_status.name)
    """),
    md(r"""
    ## 11. Connection to the repository

    | Concept | Where |
    |---|---|
    | asset graph, metadata, retry policies (`COMPUTE_RETRY`, `LIGHT_RETRY`, none on the lab) | `src/ml_platform/orchestration/assets.py` |
    | step functions the assets wrap (no Dagster import) | `orchestration/pipeline.py` → `observe_round`, `build_training_dataset`, `train_model`, `evaluate_model`, `register_model`, `score_candidates`, `select_experiments`, `run_experiments` |
    | dynamic round partitions | `orchestration/partitions.py` → `rounds_partitions`, `ensure_round_partition` |
    | resources (Dagster → Ray/MLflow/RoundStore) | `orchestration/resources.py` → `RayComputeResource`, `TrackingResource`, `RoundStoreResource`, `PlatformConfigResource` |
    | jobs + programmatic runs | `orchestration/jobs.py` → `bootstrap_job`, `closed_loop_round_job`, `retrain_job`, `run_bootstrap`, `run_round` |
    | sensor / schedule | `orchestration/sensors.py` → `new_round_sensor`, `nightly_retrain_schedule` |
    | code location | `orchestration/definitions.py` → `build_definitions`, `defs`; `make dagster`, `make validate` |
    | driving the loop | `scripts/run_closed_loop.py` (`make closed-loop`) |
    | idempotent lab + write-once rounds | `ray_runtime/tasks.py` → `ExperimentSimulator`; `data/datasets.py` → `RoundStore.write_round` |
    | tests | `tests/smoke/test_closed_loop.py`, `tests/integration/test_closed_loop_dagster.py`, `tests/unit/test_dagster_definitions.py` |

    ## 12. Failure modes

    * **Science in assets.** An asset body with the acquisition function inline is untestable
      without Dagster and couples scientific changes to orchestration deploys. Keep assets as
      glue (resolve resources → call one step function → return metadata).
    * **Retrying a physical experiment.** A `RetryPolicy` on the lab step turns a crash
      after measurement into a duplicate (and different) measurement. Use no retry + journal
      + write-once store (§3, §9).
    * **Passing data through the IO manager.** Pickling a 100 000-row pool per asset per
      partition into `$DAGSTER_HOME` duplicates data with no content identity. Pass paths and
      hashes; store data in a real store.
    * **Dagster as a GPU scheduler.** Using Dagster's multiprocess executor or op
      concurrency to place training workers duplicates Ray's job badly (no gang scheduling,
      no placement groups, no DDP restart). Dagster says "train with N workers"; Ray places them.
    * **Static partitions for rounds.** A fixed list of rounds forces a guess about the
      campaign length and breaks when a lab adds an unplanned round. Rounds are dynamic.
    * **Cycles in the graph.** "Results feed the next round" drawn as an asset edge creates a
      cycle. Close the loop across partitions through a durable store + sensor.
    * **Non-deterministic compute behind a retry.** Retries are only safe if the step is
      deterministic or idempotent; otherwise the metadata of attempt 3 does not describe
      attempts 1–2's side effects (e.g. MLflow runs left `RUNNING`).
    * **Sensor without a cursor / run key.** Every tick re-requests the same round. The repo
      uses a cursor *and* `run_key = round + manifest hash`.

    ## 13. Exercise

    1. In §5, change `toy_fit` to fail on attempts 0–3 with `max_retries=3`. What does the run
       report, and what is materialized? Now add `Backoff.EXPONENTIAL` with `delay=1` and
       predict the total wall time before running it.
    2. Using §8, add a column "production version at scoring time" from
       `registered_model.production_version` and explain why round 1 might score candidates with
       a model *older* than the one trained in round 1.
    3. Run `run_round(defs, instance, 2)` (≈15 s). Re-run the §8 checks for three rounds.
       Then evaluate `new_round_sensor` with `cursor="2"` again: what changed, and why?
    4. Write a sensor that triggers `retrain_job` when the config file changes (hint: put
       `hash_file(cfg_path)` in the cursor). Why is that a `retrain_job` and not a
       `closed_loop_round_job`?
    """),
    code(r"""
    # Shut down Ray, delete this notebook's Ray session dir and all artifacts.
    instance.dispose(); toy_instance.dispose()
    ray.shutdown()
    shutil.rmtree(RAY_TMP, ignore_errors=True)
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"ray initialized: {ray.is_initialized()} | removed {WORK} | total {time.perf_counter() - T0:.0f}s")
    """),
]

if __name__ == "__main__":
    main("05_dagster_scientific_workflows", cells)
