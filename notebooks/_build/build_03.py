"""Generate notebooks/03_ray_distributed_execution.ipynb."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbhelpers_a import code, md, setup_cell, write  # noqa: E402

cells = [
    md(r"""
    # 03 · Ray distributed execution

    **Goal.** Learn Ray's execution model — tasks, actors, ObjectRefs and the object store,
    scheduling on logical resources, serialization, backpressure, and failure handling —
    and see where each shows up in this platform. We implement the same workload
    (scoring a candidate pool with a trained surrogate) twice: once with **stateless
    tasks**, once with **stateful actors**, and compare with the repository's
    `inference.batch.predict_pool`.

    Division of labour in this platform: *Dagster* decides **when** and **what** runs
    (assets, lineage); *Ray* decides **where** computation runs and moves the data;
    *PyTorch* defines the model math.
    """),
    md(r"""
    ## 1. Conceptual model

    ```text
      driver (this notebook)                          one Ray node (your laptop)
      ─────────────────────                ┌────────────────────────────────────────────────┐
      f.remote(x) ──► ObjectRef  ──────────►  raylet: local scheduler + resource accounting  │
      Actor.remote() ─► ActorHandle        │     │ leases worker processes                   │
      ray.get(ref) ◄── value ◄─────────┐   │     ▼                                            │
                                       │   │  worker 1   worker 2   worker 3   actor A (pid) │
      GCS (global control store):      │   │     │ results / ray.put                          │
      cluster membership, actors,      └───┼─ plasma object store (shared memory, zero-copy) │
      placement groups, job table          └────────────────────────────────────────────────┘
    ```

    | Primitive | What it is | State | Typical use here |
    |---|---|---|---|
    | **task** `@ray.remote def f` | a function call executed in some worker process | none (pure) | bootstrap resamples, shard scoring |
    | **actor** `@ray.remote class C` | a dedicated worker process holding an object | in-memory, private | model cache, the "lab" simulator |
    | **ObjectRef** | a future / pointer to an immutable value in the object store | — | pass data by reference |
    | **resources** | *logical* counters (`CPU`, `GPU`, custom) a task must reserve to be scheduled | — | fit work to hardware |

    Resources are **logical**: `num_cpus=1` is a scheduling reservation, not a CPU
    pin — Ray does not stop a task from using 8 threads.
    """),
    setup_cell("nb03"),
    code(r"""
    import numpy as np
    import ray
    from ml_platform.ray_runtime.cluster import ensure_ray

    # ensure_ray = the platform's single entry point to Ray (local, RAY_ADDRESS, or KubeRay).
    # It also disables Ray's uv-run runtime-env hook (which would rebuild a venv per worker).
    RAY_TMP = tempfile.mkdtemp(prefix="nba_ray_", dir="/tmp")   # short path (unix sockets) we can delete afterwards
    resources = ensure_ray(num_cpus=4, include_dashboard=False, log_to_driver=False,
                           object_store_memory=300 * 1024**2, _temp_dir=RAY_TMP)
    print("cluster resources:", {k: v for k, v in resources.items() if not k.startswith("node:")})
    """),
    md(r"""
    ## 2. Tasks and ObjectRefs

    `.remote()` returns **immediately** with an `ObjectRef`; the work runs asynchronously in
    a worker process. `ray.get` blocks for the value. Four 0.5-second tasks on 4 CPUs take
    ~0.5 s, not 2 s; eight take ~1 s (two waves — tasks are *queued*, never rejected).
    """),
    code(r"""
    @ray.remote(num_cpus=1)
    def slow_square(x: float) -> tuple[float, int]:
        time.sleep(0.5)
        return x * x, os.getpid()

    slow_square.remote(0.0)  # warm up a worker so timings exclude process start-up
    ray.get([slow_square.remote(i) for i in range(4)])
    for n in (4, 8):
        t0 = time.perf_counter()
        refs = [slow_square.remote(i) for i in range(n)]
        submit = time.perf_counter() - t0
        out = ray.get(refs)
        print(f"{n} tasks: submitted in {1e3*submit:.1f} ms, finished in {time.perf_counter()-t0:.2f}s, "
              f"on {len({pid for _, pid in out})} worker processes; first ref = {refs[0]}")
    """),
    md(r"""
    ## 3. The object store: `ray.put` once, pass references

    Arguments to `.remote()` are serialized into the object store. Passing a large array
    directly to $N$ tasks serializes it $N$ times; `ray.put` stores it **once** and every task
    receives the same ref. Ray resolves **top-level** ObjectRef arguments to values before
    the task runs — NumPy arrays come back as **zero-copy, read-only** views of shared
    memory on the same node. Refs nested inside a list/dict are *not* resolved.
    """),
    code(r"""
    big = np.random.default_rng(0).normal(size=(500_000, 4))        # 16 MB
    print(f"array size: {big.nbytes/1e6:.0f} MB")

    @ray.remote
    def col_mean(a: np.ndarray, j: int) -> float:
        return float(a[:, j].mean())

    t0 = time.perf_counter()
    by_value = ray.get([col_mean.remote(big, j % 4) for j in range(8)])    # 8 serializations
    t_value = time.perf_counter() - t0
    t0 = time.perf_counter()
    big_ref = ray.put(big)                                                  # 1 serialization
    by_ref = ray.get([col_mean.remote(big_ref, j % 4) for j in range(8)])
    t_ref = time.perf_counter() - t0
    print(f"pass array by value x8: {t_value:.2f}s | ray.put once + pass ref x8: {t_ref:.2f}s | same result: {by_value == by_ref}")

    @ray.remote
    def inspect(a, nested):
        return type(a).__name__, bool(a.flags.writeable), type(nested[0]).__name__

    print("top-level arg -> (type, writeable), nested arg type:", ray.get(inspect.remote(big_ref, [big_ref])))
    """),
    md(r"""
    The repository uses exactly this pattern:
    `ray_runtime.tasks.parallel_bootstrap_ci` puts `y`/`yhat` once and fans out resampling
    chunks, each with an independent `SeedSequence` stream, so the result is deterministic
    for a given `(seed, n_tasks)`:
    """),
    code(r"""
    from ml_platform.evaluation.metrics import bootstrap_ci, rmse
    from ml_platform.ray_runtime.tasks import parallel_bootstrap_ci

    rng = np.random.default_rng(1)
    y = rng.normal(size=5000); yhat = y + rng.normal(scale=0.3, size=5000)
    t0 = time.perf_counter(); serial = bootstrap_ci(rmse, y, yhat, n=2000, seed=0); t_s = time.perf_counter() - t0
    t0 = time.perf_counter(); par = parallel_bootstrap_ci(y, yhat, metric="rmse", n_resamples=2000, n_tasks=4, seed=0); t_p = time.perf_counter() - t0
    again = parallel_bootstrap_ci(y, yhat, metric="rmse", n_resamples=2000, n_tasks=4, seed=0)
    print(f"serial   (point, lo, hi) = {tuple(round(v, 4) for v in serial)}  {t_s:.2f}s")
    print(f"4 tasks  (point, lo, hi) = {tuple(round(v, 4) for v in par)}  {t_p:.2f}s   deterministic: {par == again}")
    """),
    md(r"""
    Note the timings: Ray is *slower* than the serial loop here. The whole job is ~40 ms of
    NumPy, while each task pays scheduling + serialization overhead (≈ 0.1–1 ms) and the
    first call pays for importing `ml_platform` in fresh worker processes. Distribute
    work only when per-task compute is much larger than that overhead: for an evaluation
    set this small the serial `evaluation.metrics.bootstrap_ci` is the better tool; the
    Ray version pays off for large evaluation sets and many resamples.

    ## 4. Scheduling and resources

    A task is scheduled only on a node with enough **free** logical resources for its
    request. Consequences:

    * `num_cpus=2` tasks on a 4-CPU node run at most 2 at a time;
    * a request that no node can *ever* satisfy is **infeasible** and stays pending forever
      (Ray logs a warning, it does not raise);
    * requests can be chosen at run time with `.options(...)`.

    **GPU caveat (Apple silicon).** Ray may advertise a `GPU` resource for an M-series
    Metal GPU, but PyTorch DDP/NCCL needs **CUDA**. The platform therefore never trusts
    Ray's raw GPU count: `ray_runtime.resources.cuda_gpu_count` returns 0 when the only
    accelerators are Apple (`accelerator_type:M…`), and `select_resources(cfg)` derives the
    request from what is actually usable.
    """),
    code(r"""
    @ray.remote(num_cpus=2)
    def two_cpu_job() -> float:
        time.sleep(0.5); return time.time()

    t0 = time.time(); ray.get([two_cpu_job.remote() for _ in range(4)])
    print(f"4 tasks x 2 CPUs on a 4-CPU cluster: {time.time() - t0:.1f}s  (2 waves of 2)")
    t0 = time.time(); ray.get([two_cpu_job.options(num_cpus=1).remote() for _ in range(4)])   # dynamic override
    print(f"same tasks with .options(num_cpus=1): {time.time() - t0:.1f}s  (1 wave of 4)")

    # infeasible request: pending forever, not an error
    @ray.remote(resources={"TPU": 1})
    def needs_tpu():
        return "ran"
    ref = needs_tpu.remote()
    ready, not_ready = ray.wait([ref], timeout=1.0)
    print("task requesting a TPU after 1 s -> ready:", len(ready), "pending:", len(not_ready))
    ray.cancel(ref, force=True)
    """),
    code(r"""
    from ml_platform.config import PlatformConfig
    from ml_platform.ray_runtime.resources import cuda_gpu_count, run_accelerated_example, select_resources

    cfg = PlatformConfig.for_tests(WORK)
    print("Ray GPU resource      :", ray.cluster_resources().get("GPU", 0),
          "| accelerator types:", [k for k in ray.cluster_resources() if k.startswith("accelerator_type")])
    print("CUDA GPUs usable      :", cuda_gpu_count())
    res = select_resources(cfg.with_overrides(**{"distributed.use_gpu": "auto"}), num_workers=8)
    print("select_resources(...) :", res.to_dict())
    print("remote_options()      :", res.remote_options())
    print("run_accelerated_example:", run_accelerated_example(cfg))
    """),
    md(r"""
    `select_resources` capped the 8 requested workers to the 4 CPUs of this Ray cluster and
    chose CPU execution; `run_accelerated_example` shows the pattern of a task that is
    *statically* annotated `num_gpus=1` (`gpu_matmul_benchmark`) being re-requested with
    `.options(num_gpus=0)` so it still runs on a CPU-only machine.

    ## 5. Serialization

    Ray serializes arguments, return values and the function/class itself with
    **cloudpickle** (plus zero-copy buffers for NumPy/Arrow). Anything captured by a
    closure is shipped too — a closure over a 1 GB array ships 1 GB with every function
    definition. Some objects cannot be serialized at all (locks, sockets, open files,
    CUDA contexts, DB connections): create them *inside* the task/actor instead.
    """),
    code(r"""
    import threading
    from ray.util import inspect_serializability

    lock = threading.Lock()
    def uses_lock():
        with lock:
            return 1

    ok, failures = inspect_serializability(uses_lock, name="uses_lock", print_file=open(os.devnull, "w"))
    print("closure over a threading.Lock serializable?", ok, "| offending objects:", [type(f.obj).__name__ for f in failures])
    try:
        ray.put(lock)
    except Exception as e:
        print("ray.put(lock) ->", type(e).__name__)
    """),
    md(r"""
    ## 6. Backpressure

    Because `.remote()` returns instantly, a loop over a million items submits a million
    tasks: their arguments and results pile up in driver memory and the object store.
    **Backpressure** bounds the number of in-flight tasks: submit up to $W$, then
    `ray.wait(..., num_returns=1)` for one to finish before submitting the next. Memory is
    $O(W)$ instead of $O(n)$, and throughput is unchanged as long as $W \ge$ the number of
    workers. The repo implements this as `ray_runtime.tasks.bounded_map`, and
    `inference.batch.predict_pool` uses the same window over actor calls.
    """),
    code(r"""
    from ml_platform.ray_runtime.tasks import bounded_map, square_sum_task

    @ray.remote(num_cpus=1)
    def work(i: int) -> int:
        time.sleep(0.05); return i

    def run_windowed(n: int, window: int) -> tuple[int, float]:
        pending, peak, t0 = [], 0, time.perf_counter()
        for i in range(n):
            if len(pending) >= window:
                _, pending = ray.wait(pending, num_returns=1)     # block until one finishes
            pending.append(work.remote(i))
            peak = max(peak, len(pending))
        ray.get(pending)
        return peak, time.perf_counter() - t0

    t0 = time.perf_counter(); refs = [work.remote(i) for i in range(80)]; ray.get(refs)
    print(f"unbounded : peak in-flight = 80, {time.perf_counter()-t0:.2f}s")
    for W in (2, 4, 8):
        peak, dt = run_windowed(80, W)
        print(f"window {W:>2}: peak in-flight = {peak:>2}, {dt:.2f}s")

    arrays = [np.full(10, float(i)) for i in range(20)]
    print("bounded_map(square_sum_task, ...) == local:",
          bounded_map(square_sum_task, arrays, max_in_flight=4) == [float((a**2).sum()) for a in arrays])
    """),
    md(r"""
    Window 2 halves throughput (only 2 of 4 CPUs busy); windows ≥ 4 match the unbounded run
    while holding at most $W$ results.

    ## 7. Fault handling

    Two different kinds of failure:

    | Failure | Tasks | Actors |
    |---|---|---|
    | **system**: worker process dies (OOM-kill, segfault, node loss) | re-executed up to `max_retries` (default 3) | process restarted up to `max_restarts`; `__init__` runs again, **in-memory state is lost**; in-flight calls re-sent only if `max_task_retries > 0` |
    | **application**: the function raises | *not* retried unless `retry_exceptions=True` or a list of exception types | the exception is returned to the caller; the actor keeps running |

    Retrying is only safe for **idempotent** work. Recomputing a bootstrap resample is
    free and deterministic; re-running a *physical experiment* costs money and gives a
    different noisy answer. The repo encodes this difference explicitly.
    """),
    code(r"""
    COUNTER_DIR = WORK / "attempts"; COUNTER_DIR.mkdir()

    def bump(name: str) -> int:                       # attempt counter that survives process death
        p = COUNTER_DIR / name
        n = int(p.read_text()) + 1 if p.exists() else 1
        p.write_text(str(n)); return n

    @ray.remote(max_retries=3)
    def crashes_once(tag: str) -> str:
        n = bump(tag)
        if n == 1:
            os._exit(1)                               # simulate the worker process dying
        return f"succeeded on attempt {n} (pid {os.getpid()})"

    class FlakyStorage(RuntimeError):
        pass

    @ray.remote(max_retries=3, retry_exceptions=[FlakyStorage])
    def flaky_read(tag: str) -> str:
        n = bump(tag)
        if n < 3:
            raise FlakyStorage(f"transient error on attempt {n}")
        return f"succeeded on attempt {n}"

    @ray.remote(max_retries=3)                        # retry_exceptions defaults to False
    def buggy(tag: str) -> str:
        bump(tag); raise ValueError("a real bug")

    print("worker crash, max_retries=3        :", ray.get(crashes_once.remote("crash")))
    print("app exception, retry_exceptions=[..]:", ray.get(flaky_read.remote("flaky")))
    try:
        ray.get(buggy.remote("bug"))
    except ValueError as e:
        print("app exception, no retry_exceptions  : raised", type(e).__name__, "after",
              (COUNTER_DIR / "bug").read_text(), "attempt(s)")
    """),
    code(r"""
    @ray.remote(max_restarts=1, max_task_retries=0)
    class Counter:
        def __init__(self):
            self.n = 0
        def incr(self) -> tuple[int, int]:
            self.n += 1; return self.n, os.getpid()
        def crash(self):
            os._exit(1)                                # simulate the actor process dying

    def call_when_back(method, *args, retries: int = 100):
        # actor calls fail while the actor is restarting; the *caller* decides to retry
        for _ in range(retries):
            try:
                return ray.get(method.remote(*args))
            except (ray.exceptions.RayActorError, ray.exceptions.ActorUnavailableError):
                time.sleep(0.1)
        raise TimeoutError

    c = Counter.remote()
    print("before crash :", [ray.get(c.incr.remote()) for _ in range(3)])
    try:
        ray.get(c.crash.remote())
    except ray.exceptions.RayActorError as e:
        print("crash call   :", type(e).__name__, "(max_task_retries=0 -> not re-sent)")
    print("after restart:", call_when_back(c.incr), "<- new pid, counter restarted from 0 (in-memory state lost)")
    ray.kill(c)
    """),
    md(r"""
    ### Retrying computation vs re-running an experiment

    `ray_runtime.tasks.ExperimentSimulator` is the platform's stand-in for the physical lab
    (it owns the synthetic oracle). It is created with `max_restarts=1, max_task_retries=0`
    — a crashed lab is restarted, but Ray never *automatically* re-sends a `measure` call —
    and `measure` is idempotent per `(round_id, candidate_id)`: repeated requests return
    the recorded measurement. With a `journal_path` the log is persisted, so a restart does
    not forget which experiments were already run.
    """),
    code(r"""
    from ml_platform.data import generate_candidate_pool
    from ml_platform.ray_runtime.tasks import start_experiment_simulator

    pool = generate_candidate_pool(cfg)
    lab = start_experiment_simulator(cfg, pool, journal_path=WORK / "lab_journal.jsonl")
    ids = pool["candidate_id"].head(5).tolist()
    first = ray.get(lab.measure.remote(1, ids))
    again = ray.get(lab.measure.remote(1, ids))            # e.g. an orchestrator retry
    print("identical responses on repeat:", first["response"].equals(again["response"]))
    print("stats:", ray.get(lab.stats.remote()))

    old_pid = ray.get(lab.pid.remote())
    ray.kill(lab, no_restart=False)                        # the lab process crashes (async kill)
    while True:
        st = call_when_back(lab.stats)
        if st["pid"] != old_pid:
            break
        time.sleep(0.1)
    print("after restart:", st, "<- fresh process, journal reloaded")
    third = ray.get(lab.measure.remote(1, ids))
    print("re-request after restart -> same values:", first["response"].equals(third["response"]),
          "| stats:", ray.get(lab.stats.remote()))
    ray.kill(lab)
    """),
    md(r"""
    After the restart `n_physical_measurements` is 0 and all 5 requests are cache hits: no
    experiment was run twice.

    ## 8. One workload, two implementations: tasks vs actors

    Workload: score a 2 000-row candidate pool with MC dropout using a trained
    checkpoint. Loading the model is the expensive, *stateful* part.

    * **Task-based**: each shard task receives the checkpoint bytes (by ref) and the feature
      matrix (by ref), **loads the model**, scores its slice. Simple, stateless, trivially
      retryable — but pays model loading per shard.
    * **Actor-based**: $k$ actors load the model **once** in `__init__` and then score many
      shards. Amortizes loading; needs care on restart (state is rebuilt from the object store).
    """),
    code(r"""
    import io
    from ml_platform.data import ArrayDataset, initial_observations, make_oracle, records_to_frame, train_val_split
    from ml_platform.training import Trainer, resolve_checkpoint
    from ml_platform.inference import Predictor

    frame = records_to_frame(initial_observations(pool, make_oracle(cfg), 300, seed=0))
    tr, va = train_val_split(frame, 0.2, seed=0)
    result = Trainer(cfg).train(ArrayDataset.from_frame(tr), ArrayDataset.from_frame(va), checkpoint_dir=WORK / "ckpt")
    ckpt_bytes = resolve_checkpoint(result.checkpoint_path).read_bytes()
    fcols = [c for c in pool.columns if c.startswith("f") and c[1:].isdigit()]
    X = np.ascontiguousarray(pool[fcols].to_numpy(np.float64))
    SHARD, MC = 250, 20
    bounds = [(a, min(a + SHARD, len(X))) for a in range(0, len(X), SHARD)]

    LOAD_COST_S = 0.25   # pretend the model is large (GBs of weights / CUDA init); our tiny one loads in ms

    def load_predictor(b: bytes) -> Predictor:
        time.sleep(LOAD_COST_S)
        p = WORK / f"ckpt_copy_{os.getpid()}" / "model.pt"
        p.parent.mkdir(exist_ok=True); p.write_bytes(b)
        return Predictor.from_checkpoint(p)

    print(f"checkpoint {len(ckpt_bytes)/1e3:.0f} kB, {len(bounds)} shards of {SHARD} rows")
    """),
    code(r"""
    # --- task-based ---------------------------------------------------------------------------
    @ray.remote(num_cpus=1)
    def score_shard_task(ckpt: bytes, X: np.ndarray, a: int, b: int, seed: int):
        t0 = time.perf_counter(); pred = load_predictor(ckpt); t_load = time.perf_counter() - t0
        mu, sd = pred.predict_with_uncertainty(X[a:b], MC, seed=seed)
        return a, mu, sd, t_load

    ckpt_ref, X_ref = ray.put(ckpt_bytes), ray.put(X)
    t0 = time.perf_counter()
    out = ray.get([score_shard_task.remote(ckpt_ref, X_ref, a, b, i) for i, (a, b) in enumerate(bounds)])
    t_tasks = time.perf_counter() - t0
    mu_tasks = np.concatenate([m for _, m, _, _ in sorted(out, key=lambda r: r[0])])
    print(f"tasks : {t_tasks:.2f}s, model loaded {len(out)} times (total {sum(r[3] for r in out):.2f}s of loading)")

    # --- actor-based --------------------------------------------------------------------------
    @ray.remote(num_cpus=1, max_restarts=1, max_task_retries=1)
    class ScoringActor:
        def __init__(self, ckpt: bytes):
            t0 = time.perf_counter(); self.pred = load_predictor(ckpt); self.t_load = time.perf_counter() - t0
            self.n = 0
        def score(self, X: np.ndarray, a: int, b: int, seed: int):
            self.n += 1
            mu, sd = self.pred.predict_with_uncertainty(X[a:b], MC, seed=seed)
            return a, mu, sd
        def stats(self):
            return {"pid": os.getpid(), "load_s": round(self.t_load, 3), "shards": self.n}

    t0 = time.perf_counter()
    actors = [ScoringActor.remote(ckpt_ref) for _ in range(2)]
    ray.get([a.stats.remote() for a in actors])            # wait until both actors are up and loaded
    t_start = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = ray.get([actors[i % 2].score.remote(X_ref, a, b, i) for i, (a, b) in enumerate(bounds)])
    t_actors = time.perf_counter() - t0
    print(f"actors: start-up (new processes + 1 load each) {t_start:.2f}s, scoring {t_actors:.2f}s")
    print("        per-actor stats:", ray.get([a.stats.remote() for a in actors]))
    mu_actors = np.concatenate([m for _, m, _ in sorted(out, key=lambda r: r[0])])
    print("same predictions (per-shard seeds):", np.allclose(mu_tasks, mu_actors))
    for a in actors:
        ray.kill(a)
    """),
    md(r"""
    Tasks paid the (simulated) load cost 8 times but ran 4-wide on warm workers; the 2
    actors paid it twice but also paid process start-up, and score only 2-wide. Which wins
    depends on *load cost × number of shards* versus *start-up + parallelism* — actors pay
    off for expensive, reusable state (large models, GPU contexts, DB connections) and
    long-lived services (Ray Serve, notebook 08).

    The `core_worker ... has constructor arguments in the object store and max_restarts > 0`
    line above is Ray warning about `ScoringActor.remote(ckpt_ref)`: on a restart Ray re-runs
    `__init__` with the *same* ObjectRef, but it does not pin constructor refs for that
    (ray-project/ray#53727). Here `ckpt_ref` stays alive in the notebook's globals, so a
    restart would still work. The repository's `PredictorActor` avoids the issue: its
    constructor takes only scalars and every `predict_shard` call carries `[ckpt_ref]` (a
    nested ref, not resolved per call); the actor loads the model once on first use, and a
    retried call brings the ref with it. `ExperimentSimulator` reads its candidate catalogue
    from a parquet path instead.

    Because MC-dropout seeds are attached to the **shard** (not to the worker), both
    implementations give identical numbers regardless of which process scored which shard
    — a requirement for retries to be safe. The repository's production version is
    `inference.batch.predict_pool` (actors + object store + `ray.wait` window), with the
    in-process reference `predict_pool_local` using the identical shard/seed schedule:
    """),
    code(r"""
    from ml_platform.inference.batch import predict_pool, predict_pool_local

    t0 = time.perf_counter()
    scored, stats = predict_pool(result.checkpoint_path, pool, n_actors=2, shard_size=500,
                                 mc_samples=MC, seed=0, return_stats=True)
    t_pp = time.perf_counter() - t0
    local = predict_pool_local(result.checkpoint_path, pool, shard_size=500, mc_samples=MC, seed=0)
    print(f"predict_pool: {len(scored)} rows in {t_pp:.2f}s; equals predict_pool_local:",
          np.allclose(scored["pred_mean"], local["pred_mean"]) and np.allclose(scored["pred_std"], local["pred_std"]))
    print("stats keys:", sorted(stats))
    scored.head(3)
    """),
    md(r"""
    ## 9. Connection to this repository

    | Concept | Where |
    |---|---|
    | connecting to Ray (local / `RAY_ADDRESS` / KubeRay), runtime env | `src/ml_platform/ray_runtime/cluster.py` — `ensure_ray`, `shutdown_ray`, `_disable_uv_run_hook` |
    | resource selection, CUDA vs Metal GPU, dynamic `.options` | `src/ml_platform/ray_runtime/resources.py` — `select_resources`, `cuda_gpu_count`, `WorkerResources.remote_options`, `run_accelerated_example` |
    | tasks + object store + retries | `src/ml_platform/ray_runtime/tasks.py` — `parallel_bootstrap_ci`, `_bootstrap_chunk` (`retry_exceptions=[TransientTaskError]`) |
    | backpressure | `ray_runtime/tasks.py::bounded_map`, `inference/batch.py::predict_pool` (`max_in_flight`) |
    | stateful actor, at-most-once semantics, journal | `ray_runtime/tasks.py` — `ExperimentSimulator`, `start_experiment_simulator` |
    | actor model cache | `src/ml_platform/inference/batch.py` — `PredictorActor`, `actor_resources`, `shard_seed` |
    | Ray Train on top of these primitives | `src/ml_platform/training/distributed.py` (notebook 04) |

    ## 10. Failure modes

    * **Infeasible resource requests** (`num_gpus=1` on a CPU cluster, a custom resource no node has) pend forever — no error. Choose requests dynamically.
    * **Trusting Ray's GPU count on Apple silicon** — the Metal GPU is not CUDA; DDP/NCCL will fail.
    * **Unbounded submission** — driver/object-store memory blow-up, object spilling to disk; use a `ray.wait` window.
    * **Calling `ray.get` inside a loop right after each `.remote()`** — serializes the work (no parallelism).
    * **Passing large objects by value** to many tasks, or capturing them in closures — repeated serialization.
    * **Mutating zero-copy arrays** — they are read-only; copy first.
    * **Retrying non-idempotent work** (`retry_exceptions=True` on a task that writes or triggers an experiment) — duplicates side effects.
    * **Actor restarts lose in-memory state** — persist what matters (journal, object store, checkpoints).
    * **Oversubscription** — `num_cpus` is logical; a task that spawns 16 threads on a `num_cpus=1` slot fights its neighbours (set `torch.set_num_threads`).
    * **Named/detached actors outliving the job**, and leftover Ray processes if `ray.shutdown()` is skipped.

    ## 11. Exercise

    1. Rewrite `score_shard_task` so that shards are submitted through `bounded_map`; then vary `SHARD` (50, 250, 1000)
       and plot wall time for tasks vs actors. Where does model-loading overhead stop mattering?
    2. Make `ScoringActor.score` crash (`os._exit(1)`) on its 3rd call. With `max_restarts=1, max_task_retries=1`, do you
       still get all shards scored? Are the numbers identical? What changes with `max_task_retries=0`?
    3. Extend `ExperimentSimulator`-style idempotency to the task world: write a task that "runs an experiment"
       and records its result in a journal keyed by `(round_id, candidate_id)`, then make it safe to call with
       `max_retries=3`.
    """),
    code(r"""
    ray.shutdown()
    shutil.rmtree(RAY_TMP, ignore_errors=True)       # this notebook's Ray session logs/sockets
    shutil.rmtree(WORK, ignore_errors=True)
    print("ray initialized:", ray.is_initialized(), "| cleaned up", WORK)
    """),
]

if __name__ == "__main__":
    write("03_ray_distributed_execution", cells)
