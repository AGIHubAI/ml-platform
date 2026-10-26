"""Generate notebooks/00_system_map.ipynb (notebooks-c).

    uv run python notebooks/_build/build_00.py            # write the .ipynb (no outputs)
    uv run python notebooks/_build/build_00.py --execute  # write + execute in place
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbhelpers_b import code, main, md, preamble  # noqa: E402

cells = [
    md(r"""
    # 00 · System map — how the platform fits together

    **Goal.** Before looking at any single framework, build a map of the *whole* model
    lifecycle and of which component owns which decision. Every later notebook zooms into
    one box of this map; this one checks, against the live repository, that the boxes
    really are separate.

    | Section | Content |
    |---|---|
    | 1 | The lifecycle: data → model → decision → new data |
    | 2 | Five planes: data · compute · workflow control · metadata · serving |
    | 3 | Who owns what (and explicitly does **not**) |
    | 4 | Architecture diagram (Mermaid + ASCII) |
    | 5 | Derivation: the lineage identity of a model is a hash chain |
    | 6 | Executable: import boundaries, proven with `ast` |
    | 7 | Executable: the Dagster asset graph, loaded from `defs` |
    | 8 | Executable: the configuration |
    | 9 | Executable: a tiny end-to-end slice without any orchestrator |
    | 10–12 | Connection to the repo · failure modes · exercise |
    """),
    md(r"""
    ## 1. The lifecycle

    The platform exists to run an *iterative scientific campaign*: experiments are expensive,
    so a surrogate model decides which experiments to run next, and every new round of
    measurements retrains the model.

    ```text
          ┌──────────────────────────────────────────────────────────────────────────┐
          │                                                                          │
          ▼                                                                          │
    scientific data ─► validation / features ─► distributed training ─► tracking     │
    (round_000..r)       (RoundStore, hash)       (Ray Train + DDP)       (MLflow)   │
                                                                             │       │
                                                                             ▼       │
    next round r+1 ◄─ experiments ◄─ batch selection ◄─ scoring ◄─ registry ◄─ evaluation
    (write-once)      (the "lab")    (acquisition)     (Ray actors) (gates)   (paired, CIs)
    ```

    The loop closes *through data*: the output of round $r$ is not a model, it is a new
    immutable dataset `round_{r+1}`. That single design choice drives most of the
    architecture — datasets need identity, lineage and write-once semantics, and the
    orchestrator must treat them as first-class objects (assets), not as intermediate
    values of a script.
    """),
    md(r"""
    ## 2. Five planes

    | Plane | Question it answers | Durable state | Implemented by |
    |---|---|---|---|
    | **Data plane** | *What* was measured, and what is it called? | `data/candidate_pool/`, `data/rounds/round_XXX/{observations.parquet, manifest.json}`, selections, lab journal | `ml_platform.data.RoundStore` (write-once, hash-chained) |
    | **Compute plane** | *Where* and on *how many* workers does work run? | none (ephemeral workers, object store) | Ray: Ray Train (DDP), Ray tasks, Ray actors |
    | **Workflow control plane** | *What* runs *when*, in which order, for which round, and what did it produce? | Dagster run/event storage (`$DAGSTER_HOME`) | Dagster assets, jobs, partitions, sensors, schedules |
    | **Metadata plane** | *Which* run produced which model with which params/metrics/artifacts, and which model is trusted? | MLflow tracking DB + artifact store, model registry | `ml_platform.tracking` (the only MLflow importer) |
    | **Serving plane** | *How* are predictions delivered at low latency to callers? | none (replicas reload from registry) | Ray Serve deployment `SurrogateModelDeployment` |

    The scientific code (`data`, `models`, `training/trainer.py`, `evaluation`,
    `active_learning`) sits *underneath* all five planes: it is plain Python/PyTorch and can
    be called from a unit test, a notebook, a Ray worker or a Dagster asset identically.
    """),
    md(r"""
    ## 3. Who owns what

    | Component | Owns | Explicitly does **not** own |
    |---|---|---|
    | **PyTorch** | model definition (`ResidualMLP`), loss, autograd, optimizer step, DDP gradient all-reduce, checkpoint *contents* | how many workers exist or where they run; which dataset version is trained on; whether the model is good enough |
    | **Ray** (core + Train) | processes/actors, placement of work on CPUs/GPUs, object store, worker failure detection, DDP process-group setup via `TorchTrainer` | *when* a pipeline step runs, dataset identity, lineage, promotion decisions, durable history (a Ray cluster is ephemeral) |
    | **Dagster** | the asset graph, *when* each asset is (re)materialized, partitions (= rounds), retries of whole steps, sensors/schedules, materialization metadata, lineage across runs | per-worker scheduling, GPU assignment, model semantics, the metrics' meaning; it never holds the data itself (asset values here are small path/hash records) |
    | **MLflow** | run identity, params/metrics/tags, artifacts (checkpoint, evaluation report), registered model versions + lifecycle aliases | deciding promotion (the evaluation gate decides, MLflow records it), running anything, dataset storage |
    | **Kubernetes** (KubeRay, `infra/k8s/`) | nodes, pods, resource requests/limits, GPU device plugins, node pools, restarts of *pods*, persistent volumes | Python-level scheduling of tasks inside the Ray cluster, experiment lineage, training logic |
    | **Ray Serve** | online inference: replicas, request routing, dynamic batching (`@serve.batch`), autoscaling of replicas | which model is production (it *reads* the registry's `production` alias), training, offline pool scoring (Ray actors do that) |

    A useful test for every new piece of code: *which row of this table is it answering?*
    If the answer is "two rows", the code is in the wrong place.
    """),
    md(r"""
    ## 4. Architecture diagram

    ```mermaid
    flowchart LR
      subgraph control["Workflow control plane — Dagster"]
        S[new_round_sensor / nightly schedule] --> J[closed_loop_round_job<br/>partition = round_r]
      end
      subgraph data["Data plane — RoundStore"]
        P[(candidate_pool)]
        R[(round_000 … round_r)]
      end
      subgraph compute["Compute plane — Ray"]
        T[Ray Train TorchTrainer<br/>N × PyTorch DDP workers]
        E[Ray tasks<br/>bootstrap CIs]
        A[Ray actors<br/>MC-dropout pool scoring]
        L[ExperimentSimulator actor<br/>the 'lab']
      end
      subgraph meta["Metadata plane — MLflow"]
        M[(runs · metrics · artifacts)] --> G[(registry<br/>candidate → validated → production)]
      end
      subgraph serving["Serving plane — Ray Serve"]
        SV[SurrogateModelDeployment]
      end
      J -->|training_dataset| R
      J -->|trained_model| T --> M
      J -->|evaluation_report| E
      J -->|registered_model| G
      J -->|candidate_predictions| A
      J -->|selected_experiments| AL[active_learning.select_batch]
      J -->|new_experimental_results| L --> R
      R -->|new round appears| S
      G -->|production alias| SV
    ```

    The same picture in plain text (what each arrow *carries*):

    ```text
                       ┌───────────── Dagster (what / when / lineage) ─────────────┐
                       │ observed → dataset → trained → eval → registered →        │
                       │ predictions → selected → new_experimental_results         │
                       └──┬──────────┬──────────┬──────────┬──────────┬────────────┘
             paths+hashes │  "train  │ "CI with │ register │ "score   │ "measure these ids"
                          ▼  N=2"    ▼  4 tasks"▼  version ▼  pool"   ▼
     RoundStore ◄──── Ray Train ── PyTorch DDP  Ray tasks   MLflow    Ray actors   ExperimentSimulator
     (parquet +       (placement,  (all-reduce, (bootstrap) (runs,    (MC dropout) (journaled, idempotent)
      manifests)       restarts)    ckpt)                    registry)                    │
          ▲                                                    │ production alias        │
          └──────────────────── round_{r+1} (write-once) ◄─────┼──────────────────────────┘
                                                               ▼
                                                    Ray Serve replicas (/predict)
    ```
    """),
    md(r"""
    ## 5. Derivation: the identity of a training dataset

    A model is only reproducible if the dataset it was trained on has a stable name. Rounds
    are appended, never edited, so the training set of round $r$ is
    $D_r = \bigcup_{i \le r} \text{round}_i$. The `RoundStore` names it with a hash chain:

    $$
    c_i = H(\text{content of round}_i), \qquad
    m_i = H\big(i,\; n_i,\; c_i,\; [\,m_{i-1}\,]\big), \qquad
    \text{dataset\_hash}(D_r) = m_r .
    $$

    Because $m_r$ contains $m_{r-1}$, which contains $m_{r-2}$, …, changing a single value in
    any earlier round changes $m_r$ (up to hash collisions). So the one string
    `dataset_hash` recorded on an MLflow run and on a Dagster materialization commits to
    *every* observation the model saw. Provenance (`created_at`, who ran it) is deliberately
    **not** hashed, so a retried materialization of the same measurements has the same
    identity. We verify both claims on real data in §9.
    """),
    preamble("nb00"),
    md(r"""
    ## 6. Import boundaries, proven

    The separation in §3 is only real if the source code respects it. Scan every module in
    `src/ml_platform` with `ast` (this catches imports nested inside functions too, e.g.
    the lazy Ray imports in `orchestration/pipeline.py`) and record which frameworks each
    module imports **directly**.
    """),
    code(r"""
    import ast
    import ml_platform

    SRC = Path(ml_platform.__file__).parent
    FRAMEWORKS = ("dagster", "mlflow", "ray", "torch")

    def framework_imports(path: Path) -> set[str]:
        found = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            found |= {n.split(".")[0] for n in names if n.split(".")[0] in FRAMEWORKS}
        return found

    rows = []
    for f in sorted(SRC.rglob("*.py")):
        rel = f.relative_to(SRC).as_posix()
        imp = framework_imports(f)
        rows.append({"module": rel, **{fw: ("x" if fw in imp else "") for fw in FRAMEWORKS}})
    imports = pd.DataFrame(rows).set_index("module")
    imports[(imports != "").any(axis=1)]
    """),
    code(r"""
    def modules_importing(fw: str) -> list[str]:
        return [m for m in imports.index if imports.loc[m, fw] == "x"]

    PURE = ("data/", "models/", "evaluation/", "active_learning/",
            "training/trainer.py", "training/checkpointing.py", "training/config.py",
            "inference/predictor.py", "config.py", "hashing.py")
    pure_mods = [m for m in imports.index if m.startswith(PURE)]

    rules = {
        "scientific code imports no dagster/mlflow/ray":
            all(imports.loc[m, ["dagster", "mlflow", "ray"]].eq("").all() for m in pure_mods),
        "only tracking/ imports mlflow":
            all(m.startswith("tracking/") for m in modules_importing("mlflow")),
        "only orchestration/ imports dagster":
            all(m.startswith("orchestration/") for m in modules_importing("dagster")),
        "Dagster assets never import ray or torch directly":
            imports.loc["orchestration/assets.py", ["ray", "torch"]].eq("").all(),
    }
    for rule, ok in rules.items():
        print(f"{'PASS' if ok else 'FAIL'}  {rule}")
    print(f"\n{len(pure_mods)} scientific modules checked")
    print("ray is imported by:", modules_importing("ray"))
    print("mlflow is imported by:", modules_importing("mlflow"))
    assert all(rules.values())
    """),
    md(r"""
    Ray appears in exactly the places that *place work on workers*: `ray_runtime/`,
    `training/distributed.py` (Ray Train wrapper around the pure `Trainer`),
    `inference/batch.py` + `inference/serve.py`, and — lazily, inside the `RayCompute`
    methods only — `orchestration/pipeline.py`. The repository's own unit test
    `tests/unit/test_logging_boundaries.py` enforces a similar rule in CI.

    ## 7. The workflow graph, loaded from the live code location

    `ml_platform.orchestration.definitions.defs` is the object `dagster dev` loads.
    Building it does not start Ray, MLflow or touch `data/` — resources are lazy — so we can
    introspect it freely.
    """),
    code(r"""
    import contextlib, io
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")               # silence an MLflow import-time UserWarning
        from ml_platform.orchestration.definitions import defs

    graph = defs.resolve_asset_graph()
    rows = []
    for key in graph.toposorted_asset_keys:
        node = graph.get(key)
        op = defs.get_assets_def(key).op
        rp = op.retry_policy
        rows.append({
            "asset": key.to_user_string(),
            "upstream": ", ".join(p.to_user_string() for p in sorted(node.parent_keys)) or "—",
            "partitioned": type(node.partitions_def).__name__ if node.partitions_def else "—",
            "retry": f"{rp.max_retries}x, {rp.delay}s {rp.backoff.name.lower() if rp.backoff else ''}" if rp else "NONE",
            "kinds": ",".join(sorted(node.kinds)),
            "resources": ",".join(sorted(k for k in op.required_resource_keys if k != "io_manager")),
        })
    pd.set_option("display.max_colwidth", 60)
    pd.DataFrame(rows).set_index("asset")
    """),
    code(r"""
    print("jobs:")
    for job in defs.resolve_all_job_defs():
        if job.name.startswith("__"):
            continue
        names = {n.name for n in job.graph.nodes}
        steps = [k.to_user_string() for k in graph.toposorted_asset_keys if k.to_user_string() in names]
        print(f"  {job.name:<24} {len(steps)} steps: {' → '.join(steps)}")
    print("sensors:  ", [s.name for s in defs.sensors])
    print("schedules:", [(s.name, s.cron_schedule) for s in defs.schedules])
    print("resources:", {k: type(v).__name__ for k, v in defs.resources.items()})
    """),
    md(r"""
    Reading the table: every asset except `candidate_pool` is partitioned by the dynamic
    `rounds` partition set; compute-heavy assets carry an exponential-backoff retry policy,
    and `new_experimental_results` has **none** — it stands for a physical experiment (see
    notebook 05 and 11). Dagster's resources are the only way assets reach Ray (`ray_compute`)
    and MLflow (`tracking`).

    ## 8. Configuration

    One typed `PlatformConfig` (pydantic) feeds every plane. `configs/local.yaml` is the
    default; `PlatformConfig.for_tests(tmp)` shrinks everything and redirects all paths
    (and the MLflow SQLite file) into a temp directory — which is what this notebook uses.
    """),
    code(r"""
    from ml_platform.config import PlatformConfig, load_config, resolve_config_path

    local = load_config()
    print("default config file:", resolve_config_path().relative_to(SRC.parents[1]))
    for section in ("data", "model", "training", "distributed", "evaluation", "active_learning", "tracking"):
        print(f"  {section:<16}", getattr(local, section).model_dump(mode="json"))

    cfg = PlatformConfig.for_tests(WORK, **{"training.epochs": 8, "data.batch_size_per_round": 20})
    diff = {f"{s}.{k}": (getattr(local, s).model_dump()[k], v)
            for s in ("data", "model", "training", "evaluation")
            for k, v in getattr(cfg, s).model_dump().items() if getattr(local, s).model_dump()[k] != v}
    print("\nfor_tests overrides (local -> tiny):")
    for k, (a, b) in diff.items():
        print(f"  {k:<32} {a!s:>18} -> {b}")
    print("paths ->", {k: str(v).replace(str(WORK), "$WORK") for k, v in cfg.paths.model_dump().items()})
    print("mlflow ->", cfg.tracking.tracking_uri.replace(str(WORK), "$WORK"))
    """),
    md(r"""
    ## 9. A tiny end-to-end slice, without any orchestrator

    The Dagster assets are thin adapters over plain step functions in
    `orchestration/pipeline.py`, which in turn call the scientific packages. Here we call
    those same functions directly, in one process, with no Dagster, no Ray, no MLflow:

    pool → round_000 → training dataset → **single-process** `Trainer` → `evaluate` (gates)
    → MC-dropout scoring → `select_batch` → measure → round_001.
    """),
    code(r"""
    from ml_platform.active_learning import select_batch
    from ml_platform.data import ArrayDataset, RoundStore, make_oracle, measure_candidates, train_val_split
    from ml_platform.evaluation import evaluate
    from ml_platform.inference import Predictor
    from ml_platform.inference.batch import predict_pool_local
    from ml_platform.orchestration import pipeline as P
    from ml_platform.training import Trainer

    t0 = time.perf_counter()
    store = RoundStore(cfg.paths.data_dir)
    pool_info = P.ensure_candidate_pool(cfg, store)                   # data plane: write-once pool
    r0 = P.observe_round(cfg, store, 0)                               # random initial design
    ds = P.build_training_dataset(cfg, store, 0)                      # union + validation + hash
    print(f"pool: {pool_info.n_candidates} candidates | round_000: {r0.n_records} obs | "
          f"dataset {ds.dataset_hash[:12]} ({ds.n_train} train / {ds.n_val} val) | validation ok={ds.validation['ok']}")

    frame = store.training_frame(0)
    tr, va = train_val_split(frame, cfg.training.val_fraction, cfg.seed, cfg.training.val_strategy)
    result = Trainer(cfg).train(ArrayDataset.from_frame(tr), ArrayDataset.from_frame(va),
                                checkpoint_dir=WORK / "ckpt")                  # plain PyTorch, world_size=1
    print(f"trained {result.epochs_completed} epochs on {result.device} (world_size={result.world_size}) "
          f"in {result.duration_s:.1f}s, val_rmse={result.metrics['val_rmse']:.3f}")

    predictor = Predictor.from_checkpoint(result.checkpoint_path)
    ev = evaluate(predictor, va, None, cfg, seed=cfg.seed)                      # baseline = train mean
    print(f"eval: rmse={ev.metrics['rmse']:.3f} (std units) vs baseline {ev.metrics['baseline_rmse']:.3f}; "
          f"gate passed={ev.gate.passed}")
    for reason in ev.gate.reasons:
        print("   gate:", reason)
    """),
    code(r"""
    pool = store.read_pool()
    observed = store.observed_ids(0)
    unobserved = pool[~pool["candidate_id"].astype(str).isin(observed)]
    preds = predict_pool_local(result.checkpoint_path, unobserved, mc_samples=cfg.evaluation.mc_samples, seed=0)
    sel = select_batch(pool, preds, observed, cfg, round_id=1, model_version="notebook-00")
    print(f"scored {len(preds)} unobserved candidates; selected {len(sel.selected)} "
          f"(selection_hash {sel.selection_hash[:12]})")
    display(sel.selected[[c for c in ("rank", "candidate_id", "pred_mean", "pred_std", "score") if c in sel.selected]].head(5))

    recs = measure_candidates(sel.selected, make_oracle(cfg), round_id=1, seed=cfg.seed)
    m1 = store.write_round(1, recs, provenance={"source": "notebook-00", "selection_hash": sel.selection_hash})
    print(f"round_001 written: {m1.n_records} records, parent = round_000 manifest? "
          f"{m1.parent_hashes == [store.read_manifest(0).manifest_hash]}")
    print(f"slice took {time.perf_counter() - t0:.1f}s")
    """),
    md(r"""
    Note the gate: the model clearly beats the mean baseline, yet it is **not** promotable,
    because the tiny config does not reach the absolute `max_rmse` bar. Training finishing,
    and even beating a baseline, is not the same as being good enough — in the full pipeline
    `registered_model` would register it as a `candidate` only.

    Now check the claims of §5 on this real store: the chain verifies, the dataset hash of
    $D_1$ differs from $D_0$, re-writing identical content is a no-op, and altering one
    measured value is refused.
    """),
    code(r"""
    from ml_platform.data import ImmutableRoundError

    print("chain verifies:", store.verify_chain())
    print("dataset_hash(D0):", store.dataset_hash(0)[:16], "| dataset_hash(D1):", store.dataset_hash(1)[:16])
    again = store.write_round(1, store.read_round(1), provenance={"source": "retry"})
    print("identical re-write -> same manifest hash:", again.manifest_hash == m1.manifest_hash)
    tampered = store.read_round(1)
    tampered.loc[0, "response"] += 1e-3
    try:
        store.write_round(1, tampered)
    except ImmutableRoundError as e:
        print("tampered re-write refused:", type(e).__name__)
    """),
    md(r"""
    ## 10. Connection to the repository

    | Concept on the map | Where it lives |
    |---|---|
    | data plane, write-once rounds, hash chain | `src/ml_platform/data/datasets.py` → `RoundStore.write_round`, `dataset_hash`, `verify_chain`; `data/schema.py` → `RoundManifest.manifest_hash` |
    | synthetic lab / oracle | `data/synthetic_oracle.py` → `SyntheticOracle`; `ray_runtime/tasks.py` → `ExperimentSimulator` actor |
    | pure training loop | `training/trainer.py` → `Trainer.train` (single-process and DDP-aware) |
    | compute plane: distributed training | `training/distributed.py` → `train_distributed` (Ray Train `TorchTrainer`) |
    | evaluation + gates | `evaluation/evaluator.py` → `evaluate`, `check_gates` |
    | decision | `active_learning/loop.py` → `select_batch`, `write_selection` |
    | the step functions (no Dagster import) | `orchestration/pipeline.py` → `ensure_candidate_pool`, `observe_round`, `build_training_dataset`, `train_model`, `evaluate_model`, `register_model`, `score_candidates`, `select_experiments`, `run_experiments`; `RayCompute` |
    | workflow control plane | `orchestration/assets.py`, `jobs.py` (`run_bootstrap`, `run_round`), `partitions.py`, `sensors.py`, `resources.py`, `definitions.py` (`build_definitions`, `defs`) |
    | metadata plane | `tracking/mlflow_client.py` → `Tracker`; `tracking/registry.py` → `ModelRegistry.promote_if_passed` |
    | serving plane | `inference/serve.py` → `SurrogateModelDeployment`; `make serve` |
    | Kubernetes mapping | `infra/k8s/raycluster.yaml`, `rayjob.yaml`, `rayservice.yaml`, `dagster-values.yaml` |
    | the whole loop | `scripts/run_closed_loop.py` / `make closed-loop`; `tests/smoke/test_closed_loop.py` |

    ## 11. Failure modes (architectural)

    * **Science inside Dagster assets.** If the acquisition function or the loss is written
      in an `@asset` body it cannot be unit-tested, reused from a notebook or run on a Ray
      worker, and every change to the science changes the orchestration code. Here assets are
      ~20 lines of glue each; the science is importable without Dagster (§6, §9).
    * **Cluster logic in model code.** `ray.get`, `num_gpus=` or `if rank == 0: mlflow.log…`
      inside the model/trainer couples the model to one runtime. The `Trainer` only knows
      `torch.distributed`; Ray Train sets up the process group from outside.
    * **Orchestrator as data store.** Passing DataFrames between assets through the IO
      manager duplicates data into `$DAGSTER_HOME` without identity. Asset values here are
      path/hash records; data lives in the RoundStore/MLflow.
    * **Mutable datasets.** Overwriting `round_003` in place silently invalidates every model
      and report that references it. The store refuses (§9); corrections are new rounds.
    * **Training finished ⇒ promoted.** A model that ran without error is not a better model.
      Promotion goes through evaluation gates; the registry records the decision.
    * **Ray as a scheduler of record.** A Ray cluster is ephemeral: its task history vanishes
      on restart. "What ran when, on which data" must live in Dagster + MLflow.
    * **Serving reads a file path.** Serving a checkpoint path instead of the registry's
      `production` alias bypasses the gate and breaks rollback.

    ## 12. Exercise

    1. Add a fake violation: in a scratch copy of the source list, pretend
       `active_learning/acquisition.py` imports `mlflow` (edit the `imports` DataFrame) and
       confirm the rule check in §6 fails. Then write the same rule as a pytest.
    2. Extend §7 to print, for each job, which assets it selects **and** which it does not
       (hint: `defs.resolve_job_def("retrain_job")`). Why does `retrain_job` stop at
       `registered_model`?
    3. Using §9, write `round_002` by hand with a *different* selection for round 1, and
       predict what `P.run_experiments` would do if a Dagster run later tried to materialize
       `new_experimental_results[round_000]` with the original selection.
    4. Place a hypothetical new component — a feature store for molecular descriptors — on the
       five-plane map. Which component should own its versioning, and which must not?
    """),
    code(r"""
    # Clean up: everything this notebook wrote lives under WORK.
    import shutil
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"removed {WORK}; total {time.perf_counter() - T0:.1f}s")
    """),
]

if __name__ == "__main__":
    main("00_system_map", cells)
