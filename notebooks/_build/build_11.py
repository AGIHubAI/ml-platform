"""Generator for notebooks/11_fault_tolerance_and_distributed_systems.ipynb."""

from nbhelpers_b import code, main, md, preamble

NAME = "11_fault_tolerance_and_distributed_systems"

cells = [
    md(
        r"""
        # 11 · Fault tolerance and distributed-systems semantics

        Every component of the platform will eventually fail mid-operation: a Ray worker is OOM-killed,
        a spot node disappears during epoch 7, the Dagster run is retried after a network blip, a lab
        instrument times out after half a plate. The question is never *whether* work is repeated,
        but **what happens when it is**.

        The central distinction of this notebook:

        > **Retrying a computation is cheap and safe if it is deterministic and side-effect free.
        > Retrying a physical experiment is neither** — it costs money and time, consumes material,
        > and returns a *different* noisy answer. The system must record each measurement exactly once
        > and re-use the record on every retry.
        """
    ),
    md(
        r"""
        ## Conceptual model

        **Delivery / execution semantics** for an operation that may be retried after a failure:

        | Semantics | How | Risk | Where it is acceptable |
        |---|---|---|---|
        | at-most-once | never retry | lost work | expensive, non-repeatable actions without dedup (a *physical* measurement request) |
        | at-least-once | retry until acknowledged | **duplicates** | pure, deterministic computation (scoring a shard, a bootstrap chunk) |
        | exactly-once (*effect*) | at-least-once **+ idempotent / deduplicated effect** | needs a stable key and durable record | writing an experimental round, recording a measurement |

        "Exactly-once" is not a network guarantee — no protocol can deliver it over an unreliable
        channel (the two-generals problem). It is a **system property** built from retries plus an
        idempotent sink:

        $$\text{apply}(k, v)\;\text{twice} \equiv \text{apply}(k, v)\;\text{once} \quad\text{for a stable key }k.$$

        If each attempt fails independently with probability $p$ and we retry up to $R$ times, the
        operation still fails with probability $p^{R+1}$ and the expected number of executions is
        $\sum_{i=0}^{R} p^i = \frac{1-p^{R+1}}{1-p}$ — i.e. *every* retry policy with $R\ge1$ executes some
        operations more than once. Duplicates are the normal case, not an edge case.

        **Durable vs ephemeral state** in this platform:

        ```text
        ephemeral (lost on crash)                    durable (survives, versioned)
        ───────────────────────────                  ──────────────────────────────────────────
        Ray worker memory, actor fields      ──►     RoundStore rounds (write-once parquet + manifest)
        object store (ObjectRefs)                    checkpoints: epoch_XXXX/ + `latest` (atomic)
        in-flight Serve requests                     ExperimentSimulator journal (JSONL)
        Trainer optimizer state (in RAM)             MLflow runs / registry, evaluation artifacts
        ```

        A crash may lose anything on the left; recovery = rebuild it from the right.
        """
    ),
    preamble("nb11"),
    code(
        r"""
        import json, stat
        import ray
        import torch

        from ml_platform.config import PlatformConfig
        from ml_platform.data.datasets import (ArrayDataset, ImmutableRoundError, RoundSequenceError, RoundStore,
                                                  train_val_split)
        from ml_platform.data.generation import generate_candidate_pool, initial_observations, make_oracle, measure_candidates
        from ml_platform.ray_runtime.cluster import ensure_ray
        from ml_platform.ray_runtime.tasks import TransientTaskError, start_experiment_simulator
        from ml_platform.training import SimulatedWorkerFailure, Trainer, load_checkpoint

        cfg = PlatformConfig.for_tests(WORK, **{"data.pool_size": 2_000, "training.epochs": 6})
        pool = generate_candidate_pool(cfg)
        oracle = make_oracle(cfg)
        ensure_ray(num_cpus=4, log_to_driver=False)
        print("Ray CPUs:", ray.cluster_resources().get("CPU"))
        """
    ),
    md(
        r"""
        ## 1 · Retries in Ray: application errors vs worker crashes

        Ray distinguishes two failure classes:

        * **Application exceptions** (your code raised). Retried only if `retry_exceptions=True` or the
          exception type is listed — e.g. `TransientTaskError` in `ray_runtime/tasks.py`, used by
          `_bootstrap_chunk(max_retries=3, retry_exceptions=[TransientTaskError])`.
        * **System failures** (the worker process died: OOM, segfault, node loss). Retried up to
          `max_retries` times automatically.
        """
    ),
    code(
        r"""
        @ray.remote(max_retries=3, retry_exceptions=[TransientTaskError])
        def flaky_read(counter_file: str, fail_first: int) -> str:
            # Simulates a flaky storage read: fails `fail_first` times, then succeeds.
            p = Path(counter_file)
            attempt = (int(p.read_text()) if p.exists() else 0) + 1
            p.write_text(str(attempt))
            if attempt <= fail_first:
                raise TransientTaskError(f"attempt {attempt}: storage timeout")
            return f"succeeded on attempt {attempt}"

        print(ray.get(flaky_read.remote(str(WORK / "c1"), fail_first=2)))
        try:
            ray.get(flaky_read.remote(str(WORK / "c2"), fail_first=10))
        except TransientTaskError as e:
            print("gave up after 1 + 3 retries:", str(e).splitlines()[-1])
        """
    ),
    code(
        r"""
        @ray.remote(max_retries=1)
        def score_and_log(marker: str, side_effect_log: str) -> str:
            # A task with a NON-idempotent side effect, whose worker dies on the first attempt.
            with open(side_effect_log, "a") as fh:
                fh.write(f"pid {os.getpid()} appended a row\n")         # side effect happens first...
            if not Path(marker).exists():
                Path(marker).touch()
                os._exit(1)                                               # ...then the process is killed
            return "ok"

        log_file = WORK / "side_effects.log"
        print("result:", ray.get(score_and_log.remote(str(WORK / "m1"), str(log_file))))
        print(log_file.read_text())
        print("-> at-least-once execution: the side effect happened", len(log_file.read_text().splitlines()), "times")

        @ray.remote(max_retries=0)
        def dies(marker: str) -> str:
            os._exit(1)

        try:
            ray.get(dies.remote("unused"))
        except ray.exceptions.RayError as e:
            print("max_retries=0 (at-most-once):", type(e).__name__)
        """
    ),
    md(
        r"""
        The fix is not "fewer retries"; it is making the side effect **idempotent**: key it by a stable
        id and make the write a no-op if the key exists (a primary key, a write-once file, a
        conditional put). The platform does exactly that for its two important side effects:
        experimental rounds and physical measurements.

        ## 2 · `RoundStore`: write-once, idempotent round materialisation

        `RoundStore.write_round(round_id, records)` (in `data/datasets.py`):

        * computes a **content hash** of the scientific content (ids, features, response, std, status —
          *not* `created_at`/provenance, which differ between attempts);
        * if the round exists with the same hash → returns the existing manifest (**idempotent no-op**);
        * if it exists with different content → `ImmutableRoundError` (**never overwrite**);
        * writes into a temp dir and renames atomically, then makes files read-only;
        * rejects gaps / duplicate experiment ids (`RoundSequenceError`) and chains manifests by hash.
        """
    ),
    code(
        r"""
        store = RoundStore(WORK / "data")
        r0 = initial_observations(pool, oracle, n=100, seed=0)
        m_first = store.write_round(0, r0, provenance={"attempt": 1})

        # An orchestrator retry re-runs the asset: the measurement is re-derived from the SAME seed/round,
        # so the content is identical (created_at differs) -> no-op, same manifest.
        r0_retry = initial_observations(pool, oracle, n=100, seed=0)
        m_retry = store.write_round(0, r0_retry, provenance={"attempt": 2})
        print("retry is a no-op:", m_retry.manifest_hash == m_first.manifest_hash,
              "| provenance kept from first write:", m_retry.provenance)

        # "Re-running the experiment" produces different noise -> the store refuses to overwrite history.
        r0_rerun = initial_observations(pool, oracle, n=100, seed=1)          # different measurement noise
        r0_rerun = [r.model_copy(update={"round_id": 0}) for r in r0_rerun]
        try:
            store.write_round(0, r0_rerun)
        except ImmutableRoundError as e:
            print("ImmutableRoundError:", e)

        try:                                                      # skipping round 1 breaks the chain
            store.write_round(2, measure_candidates(pool.iloc[1500:1510], oracle, round_id=2, seed=0))
        except RoundSequenceError as e:
            print(type(e).__name__ + ":", e)

        f = store.round_dir(0) / "observations.parquet"
        print("file mode:", stat.filemode(f.stat().st_mode), "| chain verifies:", store.verify_chain())
        """
    ),
    code(
        r"""
        # Atomicity: a crash while writing leaves no half-written round behind.
        # (_atomic_write_dir is the private helper write_round uses: temp dir + rename.)
        def crash_while_writing(path):
            path.write_text("half a parquet file")
            raise OSError("disk full / process killed")

        try:
            store._atomic_write_dir(store.round_dir(1), {"observations.parquet": crash_while_writing})
        except OSError as e:
            print("write failed:", e)
        print("round 1 visible?", store.exists(1), "| rounds:", store.list_rounds(),
              "| leftovers:", [p.name for p in store.rounds_dir.iterdir()])
        """
    ),
    md(
        r"""
        ## 3 · The `ExperimentSimulator` actor: never measure twice

        The oracle stands in for the lab. `ray_runtime.tasks.ExperimentSimulator` is a Ray **actor**
        (stateful) with a measurement log keyed by `(round_id, candidate_id)`:

        * `measure(round_id, ids)` performs *new* measurements only for keys not in the log; repeats
          are cache hits (`n_physical_measurements` does not grow) → **idempotent**;
        * with `journal_path` the log is appended to a JSONL file and replayed in `__init__`, so the
          dedup survives an actor restart (**durable**);
        * the actor is declared `max_restarts=1, max_task_retries=0`: Ray restarts a crashed actor, but
          does **not** automatically re-send an in-flight `measure` call — whether to re-request is a
          decision for the caller, who can do so safely because the call is idempotent.
        """
    ),
    code(
        r"""
        def stats_when_restarted(actor, old_pid, timeout=60):
            # poll until the actor answers from a NEW process (ray.kill is asynchronous)
            t0 = time.time()
            while time.time() - t0 < timeout:
                try:
                    st = ray.get(actor.stats.remote(), timeout=5)
                    if st["pid"] != old_pid:
                        return st
                except ray.exceptions.RayError:
                    pass
                time.sleep(0.2)
            raise TimeoutError("actor did not restart")

        def request_measurement(lab, round_id, ids, attempts=30):
            # Caller-side retry. Safe ONLY because measure() is idempotent per (round_id, candidate_id).
            errors = []
            for _ in range(attempts):
                try:
                    return ray.get(lab.measure.remote(round_id, ids)), errors
                except ray.exceptions.ActorUnavailableError as e:
                    errors.append(str(e).split(": ", 2)[-1][:90]); time.sleep(0.2)
            raise RuntimeError(errors[-1])

        ids = pool.candidate_id.iloc[500:520].tolist()
        rows = []
        for label, journal in (("with journal (durable)", WORK / "lab_journal.jsonl"), ("no journal (ephemeral)", None)):
            lab = start_experiment_simulator(cfg, pool, journal_path=journal)
            first, _ = request_measurement(lab, 1, ids)
            request_measurement(lab, 1, ids)                                  # a retried request -> cache hit
            s1 = ray.get(lab.stats.remote())
            ray.kill(lab, no_restart=False)                                   # crash the actor process
            s2 = stats_when_restarted(lab, s1["pid"])                         # Ray restarted it (max_restarts=1)
            after, errs = request_measurement(lab, 1, ids)                    # caller re-requests after the crash
            s3 = ray.get(lab.stats.remote())
            rows.append({"lab": label, "pid before/after": f"{s1['pid']}/{s3['pid']}",
                         "physical before crash": s1["n_physical_measurements"],
                         "cache hits before crash": s1["n_cache_hits"],
                         "log entries after restart": s2["n_logged"],
                         "physical after re-request": s3["n_physical_measurements"],
                         "same values": bool(np.allclose(first.response, after.response))})
            ray.kill(lab, no_restart=True)
        pd.DataFrame(rows).set_index("lab")
        """
    ),
    md(
        r"""
        * With the journal, the restarted actor replays its log: the re-request after the crash is
          served from the record — **zero** additional physical experiments.
        * Without it, the in-memory log died with the process, and the re-request **re-runs 20
          experiments**. The values happen to match only because the *simulated* oracle derives its
          noise from `(seed, round_id)`; a real instrument would return new noise, and the lab would
          have paid twice.

        ## 4 · Checkpointing: resuming training after a worker failure

        `Trainer.train(..., fail_at_epoch=k)` raises `SimulatedWorkerFailure` right *after* epoch $k$'s
        checkpoint is written (atomically: `epoch_XXXX/` + a `latest` pointer).
        `Trainer.train(..., resume_from=dir)` restores model, optimizer, epoch counter, history,
        normaliser **and all RNG states**, so the resumed run is bit-identical to an uninterrupted one.
        Ray Train does the same across processes (`training/distributed.py` →
        `train_distributed(fail_at_epoch=..., max_failures=1)`).

        The cost of a failure is bounded by the checkpoint interval: with interval $c$ epochs, epoch time
        $t$ and checkpoint cost $w$, the expected loss per failure is about $c\,t/2$ of recomputation,
        while checkpointing costs $w/c$ per epoch — the classic Young/Daly trade-off
        $c^\* \approx \sqrt{2\,w\,\text{MTBF}}/t$.
        """
    ),
    code(
        r"""
        frame = store.training_frame()
        tr, va = train_val_split(frame, 0.2, cfg.seed)
        ds = lambda: (ArrayDataset.from_frame(tr), ArrayDataset.from_frame(va))

        try:
            Trainer(cfg).train(*ds(), checkpoint_dir=WORK / "run_crash", fail_at_epoch=3)
        except SimulatedWorkerFailure as e:
            print("crashed:", e)
        print("on disk:", sorted(p.name for p in (WORK / "run_crash").iterdir()))

        resumed = Trainer(cfg).train(*ds(), checkpoint_dir=WORK / "run_crash", resume_from=WORK / "run_crash")
        clean = Trainer(cfg).train(*ds(), checkpoint_dir=WORK / "run_clean")
        print(f"resumed from {Path(resumed.resumed_from).name}, finished {resumed.epochs_completed} epochs")

        a = load_checkpoint(resumed.checkpoint_path)["model_state"]
        b = load_checkpoint(clean.checkpoint_path)["model_state"]
        identical = all(torch.equal(a[k], b[k]) for k in a)
        print("final weights identical to uninterrupted run:", identical)
        pd.DataFrame({"resumed val_rmse": [h["val_rmse"] for h in resumed.history],
                      "uninterrupted val_rmse": [h["val_rmse"] for h in clean.history]},
                     index=pd.Index(range(1, cfg.training.epochs + 1), name="epoch"))
        """
    ),
    md(
        r"""
        ## 5 · Partial failure

        A fan-out of N tasks can partially fail. `ray.get(list_of_refs)` raises on the first error and
        hides which items succeeded; for long jobs, collect per-item outcomes and re-submit only
        the failures (which, again, must be idempotent to be safe to re-run).
        """
    ),
    code(
        r"""
        @ray.remote(max_retries=0)
        def score_shard(i: int) -> float:
            if i == 5:
                raise ValueError(f"shard {i}: corrupt input")
            return float(i) ** 2

        refs = {i: score_shard.remote(i) for i in range(8)}
        ok, failed = {}, {}
        for i, ref in refs.items():
            try:
                ok[i] = ray.get(ref)
            except ray.exceptions.RayTaskError as e:
                failed[i] = type(e.cause).__name__ if getattr(e, "cause", None) else type(e).__name__
        print(f"succeeded: {sorted(ok)}  failed: {failed}")
        """
    ),
    md(
        r"""
        ## 6 · Stragglers and speculative execution

        A synchronous stage finishes when its **slowest** task finishes. With $N$ tasks whose durations
        have a heavy tail, $\mathbb E[\max_i T_i]$ grows with $N$ even if the mean is fine; in DDP one
        slow rank stalls every all-reduce. Mitigations: smaller shards (over-partitioning), backup
        (speculative) copies of late tasks, or excluding a bad node. **Speculation duplicates work** —
        it is only allowed for idempotent computation, never for a physical experiment.
        """
    ),
    code(
        r"""
        @ray.remote(num_cpus=1, max_retries=0)
        def shard_work(i: int, attempt: int) -> tuple[int, int]:
            # Shard 7's first attempt lands on a "sick" node and is 12x slower.
            time.sleep(1.2 if (i == 7 and attempt == 0) else 0.1)
            return i, attempt

        t0 = time.perf_counter()
        ray.get([shard_work.remote(i, 0) for i in range(8)])
        t_plain = time.perf_counter() - t0

        t0 = time.perf_counter()
        pending = {shard_work.remote(i, 0): i for i in range(8)}
        done_shards, backups = {}, 0
        ready, _ = ray.wait(list(pending), num_returns=len(pending), timeout=0.35)   # deadline ~3x median
        for r in ready:
            done_shards[pending.pop(r)] = 0
        for ref, i in list(pending.items()):                                      # launch backup copies
            pending[shard_work.remote(i, 1)] = i; backups += 1
        while len(done_shards) < 8:
            (r,), _ = ray.wait(list(pending), num_returns=1)
            i = pending.pop(r)
            if i not in done_shards:
                done_shards[i] = ray.get(r)[1]
                for other, j in list(pending.items()):                           # cancel the loser
                    if j == i:
                        ray.cancel(other, force=True); pending.pop(other)
        t_spec = time.perf_counter() - t0
        print(f"no mitigation: {t_plain:.2f}s   speculative backup ({backups} backup task): {t_spec:.2f}s")
        print("winning attempt per shard:", dict(sorted(done_shards.items())))
        """
    ),
    md(
        r"""
        ## Connection to this repository

        | Mechanism | Where | Semantics |
        |---|---|---|
        | write-once rounds, content hash, atomic rename, read-only files | `src/ml_platform/data/datasets.py` → `RoundStore.write_round`, `ImmutableRoundError`, `RoundSequenceError`, `verify_chain` | exactly-once *effect* |
        | reproducible measurement noise per `(oracle seed, round, seed)` | `src/ml_platform/data/generation.py` → `measure_candidates` | makes a retried materialisation identical |
        | measurement log + journal, `max_task_retries=0` | `src/ml_platform/ray_runtime/tasks.py` → `ExperimentSimulator`, `start_experiment_simulator` | at-most-once physical execution, idempotent requests |
        | transient-error retries | `ray_runtime/tasks.py` → `_bootstrap_chunk` (`retry_exceptions=[TransientTaskError]`), `parallel_bootstrap_ci` | at-least-once, deterministic |
        | model-cache actors with restarts | `src/ml_platform/inference/batch.py` → `PredictorActor` (`max_restarts=1, max_task_retries=1`) | at-least-once scoring (pure) |
        | atomic checkpoints, `latest` pointer, RNG state | `src/ml_platform/training/checkpointing.py` → `save_checkpoint`, `resolve_checkpoint`; `trainer.py` → `Trainer.train(fail_at_epoch, resume_from)`, `SimulatedWorkerFailure` | resume = replay from durable state |
        | Ray Train restarts from the last reported checkpoint | `src/ml_platform/training/distributed.py` → `train_distributed(max_failures=...)` | |
        | versioned selections | `src/ml_platform/active_learning/loop.py` → `select_batch` (`selection_hash`), `write_selection` | idempotent rewrite |
        | orchestration retries | Dagster retry policies on assets in `src/ml_platform/orchestration/` — safe *because* the sinks above are idempotent | |

        ## Failure modes

        * **Retrying a side effect** — at-least-once execution duplicates appends, emails, instrument
          runs (§1). Key every side effect and make it idempotent.
        * **Overwriting history** — "just re-run round 3" silently replaces the data a registered model
          was trained on; lineage breaks. `ImmutableRoundError` makes this impossible.
        * **Non-atomic writes** — a crash mid-write leaves a half-file that a later reader trusts.
          Write to temp + rename; publish a pointer (`latest`, manifest) last.
        * **Ephemeral dedup state** — a dedup table in memory is lost exactly when you need it (after a
          crash, §3). Dedup state must be durable.
        * **Resuming onto different data** — `Trainer.load_checkpoint` refuses a checkpoint whose
          `dataset_hash` differs from the current dataset.
        * **Speculative duplicates of non-idempotent work** — backup tasks are for computation only.
        * **Retry storms** — synchronized retries amplify an outage; use backoff with jitter and caps.

        ## Exercise

        1. Make `score_and_log` in §1 idempotent: write its row to `WORK/effects/<key>.txt` via a temp
           file + `os.replace`, skipping if the file exists. Show the side effect happens once despite
           the crash.
        2. Crash the `ExperimentSimulator` *between* the physical measurement and the journal append
           (e.g. subclass it). What is lost? How would you order the operations — or add a two-phase
           "requested → measured" record — so a real lab never repeats or loses a measurement?
        3. Using the Young/Daly formula, pick a checkpoint interval for a 2-hour DDP job on spot nodes
           with an MTBF of 3 hours, 90 s per epoch and 20 s per checkpoint.
        """
    ),
    code(
        r"""
        ray.shutdown()
        import shutil
        for p in WORK.rglob("*"):                        # round files are read-only by design
            if p.is_file():
                p.chmod(0o644)
        shutil.rmtree(WORK, ignore_errors=True)
        print(f"done in {time.perf_counter() - T0:.1f}s; Ray shut down, workspace removed")
        """
    ),
]

if __name__ == "__main__":
    main(NAME, cells)
