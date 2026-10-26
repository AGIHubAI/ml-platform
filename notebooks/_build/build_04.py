"""Generate notebooks/04_ray_train_and_resource_management.ipynb."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbhelpers_a import code, md, setup_cell, write  # noqa: E402

cells = [
    md(r"""
    # 04 · Ray Train and resource management

    **Goal.** Connect notebook 02 (what happens *inside* DDP processes) with notebook 03
    (how Ray places work on resources). Ray Train is the layer that **creates the DDP
    process group on Ray-managed workers**: it reserves resources for all ranks at once,
    starts one worker process per rank, wires up `torch.distributed`, persists
    checkpoints, and restarts the group on failure.

    > *Ray determines where workers execute. PyTorch DDP determines how model replicas
    > synchronize gradients.* — `src/ml_platform/training/distributed.py`

    We (1) look at placement groups, (2) run a minimal `TorchTrainer` whose workers report
    what they observe (rank, world size, device, pid), (3) run the platform's
    `train_distributed` with 2 workers and inspect its distributed metadata, and (4) kill a
    worker mid-training and watch Ray Train recover from the last persisted checkpoint.
    """),
    md(r"""
    ## 1. Conceptual model — Ray Train ↔ DDP

    ```text
     driver: TorchTrainer(train_loop_per_worker, ScalingConfig(num_workers=N, use_gpu, resources_per_worker),
                          RunConfig(storage_path, FailureConfig(max_failures), CheckpointConfig(num_to_keep)))
        │ .fit()
        ▼
     Train controller (actor) ── reserves ONE placement group with N bundles {CPU:c, GPU:g}  (gang scheduling)
        │                        all N bundles or nothing -> no half-started DDP job holding resources
        ▼
     worker group: N Ray actors, one OS process per rank  ─┐
        rank 0 ─┐                                           │ Ray sets RANK, WORLD_SIZE, LOCAL_RANK,
        rank 1 ─┼── torch.distributed.init_process_group    │ MASTER_ADDR/PORT, CUDA_VISIBLE_DEVICES
        ...     │   (gloo on CPU / nccl on CUDA)            │ and calls init_process_group for you
        rank N-1┘                                          ─┘
          each runs train_loop_per_worker(config):  Trainer (DDP) ... ray.train.report(metrics, checkpoint)
                                                                         │ rank 0 attaches checkpoint dir
                                                                         ▼
                                                  storage_path/<run>/checkpoint_... (num_to_keep newest)
     on worker failure: tear down the whole group -> new placement group -> new processes
                        -> ray.train.get_checkpoint() returns the latest persisted checkpoint -> resume
    ```

    | Concern | PyTorch DDP alone (`torchrun`, nb 02) | Ray Train |
    |---|---|---|
    | who starts rank processes | `torchrun` on every node (you ssh / k8s) | Ray, onto any node with free resources |
    | rendezvous env vars | `torchrun` | Ray Train's backend setup |
    | resource reservation | none (you pick hosts) | placement group, all-or-nothing |
    | GPU assignment | `LOCAL_RANK` → `cuda:<local_rank>` | Ray sets `CUDA_VISIBLE_DEVICES`; `ray.train.torch.get_device()` |
    | checkpoint persistence | your code | `ray.train.report(checkpoint=...)` → `storage_path` |
    | failure handling | torchrun elastic agent restarts local procs | `FailureConfig(max_failures)`: restart the group, hand back the latest checkpoint |
    | data movement | each rank reads files | object store (`ray.put`) or `ray.data` streaming shards |
    | gradient sync | DDP | still DDP — Ray Train does not touch the math |
    """),
    setup_cell("nb04"),
    code(r"""
    import numpy as np
    import pandas as pd
    import ray
    import ray.train
    import torch
    import torch.distributed as dist
    from ml_platform.ray_runtime.cluster import ensure_ray

    RAY_TMP = tempfile.mkdtemp(prefix="nba_ray_", dir="/tmp")    # short path we can delete afterwards
    res = ensure_ray(num_cpus=4, include_dashboard=False, log_to_driver=False,
                     object_store_memory=300 * 1024**2, _temp_dir=RAY_TMP)
    print({k: v for k, v in res.items() if not k.startswith("node:")})
    """),
    md(r"""
    ## 2. Placement groups — gang scheduling

    A DDP job with $N$ ranks is useless until *all* $N$ processes exist (the first
    collective blocks until everyone joins). Scheduling ranks one by one risks a deadlock:
    two jobs each grab half the GPUs and wait forever for the rest. A **placement group**
    reserves a set of resource *bundles* atomically; strategies control locality:
    `PACK` (as few nodes as possible — fast NCCL/shared memory; Ray Train's default),
    `SPREAD` (fault isolation), and the `STRICT_` variants that fail rather than compromise.
    """),
    code(r"""
    from ray.util.placement_group import placement_group, remove_placement_group

    pg = placement_group([{"CPU": 1}] * 2, strategy="PACK")
    print("2 x 1-CPU bundles ready:", pg.wait(timeout_seconds=10))
    print("available CPUs while reserved:", ray.available_resources().get("CPU"))

    too_big = placement_group([{"CPU": 1}] * 5, strategy="PACK")    # 5 CPUs on a 4-CPU cluster
    print("5 x 1-CPU bundles ready  :", too_big.wait(timeout_seconds=2), "-> none of it is reserved (all-or-nothing)")
    print("available CPUs           :", ray.available_resources().get("CPU"))
    remove_placement_group(too_big); remove_placement_group(pg)
    """),
    md(r"""
    ## 3. What a worker observes

    Inside `train_loop_per_worker`, `ray.train.get_context()` exposes the worker's place in
    the group; by the time the function runs, `torch.distributed` is already initialized.
    We collect what each worker sees (written to files in the run's storage, since
    `Result.metrics` only carries rank 0's report, and only when it has a checkpoint).
    """),
    code(r"""
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    def observe_loop(config: dict) -> None:
        import json, os, socket
        import ray.train, ray.train.torch, torch, torch.distributed as dist
        ctx = ray.train.get_context()
        t = torch.tensor([float(ctx.get_world_rank())])
        dist.all_reduce(t)                                   # the process group is live
        info = {
            "world_rank": ctx.get_world_rank(), "local_rank": ctx.get_local_rank(),
            "world_size": ctx.get_world_size(), "node_rank": ctx.get_node_rank(),
            "pid": os.getpid(), "host": socket.gethostname(),
            "backend": dist.get_backend(), "device": str(ray.train.torch.get_device()),
            "env": {k: os.environ.get(k) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "CUDA_VISIBLE_DEVICES")},
            "sum_of_ranks_via_all_reduce": float(t),
        }
        with open(os.path.join(config["out"], f"rank{ctx.get_world_rank()}.json"), "w") as f:
            json.dump(info, f)
        ray.train.report({"sum_of_ranks": float(t)})

    OBS = WORK / "observe"; OBS.mkdir()
    t0 = time.perf_counter()
    result = TorchTrainer(
        observe_loop,
        train_loop_config={"out": str(OBS)},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        run_config=RunConfig(name="observe", storage_path=str(WORK / "ray_results")),
    ).fit()
    print(f"fit() took {time.perf_counter() - t0:.1f}s (placement group + 2 new worker processes + gloo rendezvous)")
    obs = pd.DataFrame([json.loads(p.read_text()) for p in sorted(OBS.glob("rank*.json"))])
    for e in obs.pop("env"):
        print("env set by Ray Train:", e)
    obs
    """),
    md(r"""
    Two processes, two pids, one gloo process group; `RANK`/`WORLD_SIZE`/`MASTER_ADDR`
    were set by Ray Train, exactly the variables `torchrun` set in notebook 02.

    **GPU assignment.** With `use_gpu=True` each bundle also reserves `{"GPU": 1}`; Ray sets
    `CUDA_VISIBLE_DEVICES` per worker process and `ray.train.torch.get_device()` returns
    the right `cuda:i`, so the training code never hard-codes device indices. On this
    machine `CUDA_VISIBLE_DEVICES` is unset and the device is `cpu`. Note the Apple-silicon
    trap from notebook 03: Ray advertises a Metal `GPU` resource, so `use_gpu=True` *would
    schedule* here and then fail in NCCL. `ray_runtime.resources.select_resources` decides
    `use_gpu` from CUDA availability, never from Ray's GPU count:
    """),
    code(r"""
    from ml_platform.config import PlatformConfig
    from ml_platform.ray_runtime.resources import select_resources

    cfg = PlatformConfig.for_tests(WORK).with_overrides(**{"training.epochs": 4, "distributed.use_gpu": "auto"})
    wr = select_resources(cfg, num_workers=2)
    print("Ray GPU resource:", ray.cluster_resources().get("GPU", 0), "| chosen:", wr.to_dict())
    print("ScalingConfig kwargs:", wr.scaling_config_kwargs())
    """),
    md(r"""
    ## 4. The platform's `train_distributed`

    `ml_platform.training.distributed.train_distributed(cfg, round_id=...)`:

    1. **Data movement** — reads the immutable round union from the `RoundStore` *once* on the
       driver, splits train/validation deterministically, and `ray.put`s both frames; every
       worker `ray.get`s the refs (zero-copy on the same node), so all ranks see byte-identical
       data with the same dataset hash.
    2. **Placement** — `select_resources` → `ScalingConfig`.
    3. **Workers** — `_train_loop_per_worker` runs the ordinary `Trainer` (DDP inside), and its
       `on_epoch_end` hook calls `ray.train.report` on every rank, attaching rank 0's
       checkpoint directory.
    4. **Summary** — at the end ranks `all_gather` their pid/host/parameter digest; the driver
       returns them in `TrainResult.device_info["distributed"]` (read with `distributed_info`).

    First create a tiny immutable round store in the scratch directory:
    """),
    code(r"""
    from ml_platform.data import RoundStore, generate_candidate_pool, initial_observations, make_oracle

    store = RoundStore(cfg.paths.data_dir)                         # WORK/data
    pool = generate_candidate_pool(cfg)
    store.write_pool(pool)
    manifest = store.write_round(0, initial_observations(pool, make_oracle(cfg), cfg.data.initial_observations, cfg.seed))
    print(f"round_000: {manifest.n_records} records, dataset id {store.dataset_hash(0)[:16]}…")
    """),
    code(r"""
    from ml_platform.training.distributed import distributed_info, ray_train_storage, train_distributed

    t0 = time.perf_counter()
    base = train_distributed(cfg, round_id=0, num_workers=2, run_name="nb04_baseline")
    print(f"train_distributed: {time.perf_counter() - t0:.1f}s wall, world_size={base.world_size}, "
          f"epochs={base.epochs_completed}, val_rmse={base.metrics['val_rmse']:.4f}")
    info = distributed_info(base)
    print("pids:", info["pids"], "| params_in_sync:", info["params_in_sync"], "| backend:", info["backend"])
    print("global batch size:", base.hyperparams["distributed.global_batch_size"],
          "=", cfg.training.batch_size, "per worker x", info["resources"]["num_workers"], "workers")
    pd.DataFrame(info["workers"])[["rank", "local_rank", "world_size", "pid", "device", "n_train_rows_seen_per_epoch", "param_digest"]] \
      .assign(param_digest=lambda d: d.param_digest.str[:16])
    """),
    code(r"""
    run_dir = ray_train_storage(cfg) / "nb04_baseline"
    print("storage:", run_dir)
    for p in sorted(run_dir.iterdir()):
        print("  ", p.name, "/" if p.is_dir() else "", sorted(q.name for q in p.iterdir()) if p.is_dir() else "")
    print("final checkpoint:", base.checkpoint_path.relative_to(ray_train_storage(cfg)))
    """),
    md(r"""
    Only the newest `num_to_keep=2` checkpoints survive in storage (`CheckpointConfig`);
    the per-epoch metric history lives *inside* the checkpoint, so nothing is lost.
    Identical `param_digest`s on both ranks = DDP kept the replicas in sync; each rank saw
    half of the training rows per epoch.

    ## 5. Failure and recovery

    `fail_at_epoch=2` makes the `Trainer` raise `SimulatedWorkerFailure` right after epoch
    2's checkpoint has been reported (and — via `ray.train.get_all_reported_checkpoints` —
    registered by the controller, so it is durable). With `max_failures=1` Ray Train:

    1. sees the worker error and tears down **the whole worker group** (a DDP group cannot
       continue with a missing rank);
    2. re-reserves resources and starts *new* worker processes;
    3. hands them the latest persisted checkpoint through `ray.train.get_checkpoint()`;
    4. `Trainer.load_checkpoint` restores model, optimizer, epoch, history and **each
       rank's own** RNG state, and training continues at epoch 3.
    """),
    code(r"""
    t0 = time.perf_counter()
    rec = train_distributed(cfg, round_id=0, num_workers=2, fail_at_epoch=2, max_failures=1, run_name="nb04_failure")
    rinfo = distributed_info(rec)
    print(f"{time.perf_counter() - t0:.1f}s wall | failures_recovered={rinfo['failures_recovered']} "
          f"| restored_from_epoch={rinfo['restored_from_epoch']} | epochs_completed={rec.epochs_completed}")
    print("worker pids after recovery:", rinfo["pids"], "(first attempt used", info["pids"], "in the baseline run)")
    print("resumed from:", Path(rec.resumed_from).name if rec.resumed_from else None)
    pd.DataFrame({"epoch": [int(h["epoch"]) for h in base.history],
                  "val_rmse uninterrupted": [h["val_rmse"] for h in base.history],
                  "val_rmse failed+recovered": [h["val_rmse"] for h in rec.history]})
    """),
    code(r"""
    from ml_platform.inference import Predictor

    Xq = pool.head(500)
    p_base = Predictor.from_checkpoint(base.checkpoint_path).predict(Xq)
    p_rec = Predictor.from_checkpoint(rec.checkpoint_path).predict(Xq)
    print("max |prediction difference| uninterrupted vs recovered:", float(np.abs(p_base - p_rec).max()))
    print("params_in_sync after recovery:", rinfo["params_in_sync"])
    same = [h1["val_rmse"] == h2["val_rmse"] for h1, h2 in zip(base.history, rec.history)]
    print("per-epoch val_rmse bit-identical to the uninterrupted run:", same)
    assert all(same) and float(np.abs(p_base - p_rec).max()) == 0.0, "DDP resume should be bit-exact"
    """),
    md(r"""
    Recovery worked mechanically — new processes, resumed at epoch 3 from the persisted
    epoch-2 checkpoint, all 4 epochs in the history, replicas in sync — **and exactly**:
    every epoch's `val_rmse` is bit-identical to the uninterrupted run and the final
    predictions match to the last bit (difference `0.0`).

    That is not automatic. Besides model/optimizer state, each rank carries its own RNG
    stream: rank r's dropout masks (and the `DataLoader`'s base seed) come from rank r's
    torch generator, seeded differently per rank. The data order needs no saved state
    (`DistributedSampler.set_epoch` makes it a pure function of `(seed, epoch)`), but the
    RNG streams do. So `Trainer.save_checkpoint` `all_gather`s **every rank's** RNG state
    into rank 0's payload (`rng_states[rank]`), and on resume each rank restores *its own*.
    An earlier version saved only rank 0's state and re-seeded ranks > 0; rank 1's dropout
    masks then diverged after the restart, its gradients differed, and — through the
    all-reduce — so did the model: recovery "worked" but was not reproducible.
    (Resuming with a *different* world size cannot be exact — the data shards change — so
    the `Trainer` falls back to a deterministic re-seed and logs
    `training.resume_not_bit_exact`.)

    Lesson: "deterministic" and "bit-reproducible after failure" are different claims;
    test the second one explicitly by comparing against an uninterrupted run, as done here.

    """),
    md(r"""
    ## 6. Retries, elasticity and their boundaries

    * **What is retried.** Ray Train restarts the *whole group* (DDP cannot lose a rank);
      `max_failures` counts group restarts (`-1` = forever). Work since the last persisted
      checkpoint is recomputed — checkpoint frequency trades I/O for lost work.
      Deterministic bugs fail again on every retry: retries fix *transient* failures only.
    * **Elasticity.** DDP's world size is fixed for the lifetime of a process group. Changing
      the number of workers means tearing the group down and re-forming it — i.e. a restart
      from a checkpoint. The Trainer resumes fine with a different world size, but the
      effective batch size changes (notebook 02) and there is no saved RNG stream for new
      ranks, so the continued run is no longer bit-comparable (it warns
      `training.resume_not_bit_exact`). Growing/shrinking on the fly is possible (Ray Train supports a
      min/max worker range in newer APIs; torchrun has `--nnodes=min:max`), but always at a
      re-rendezvous boundary.
    * **Data movement.** Here the dataset is a few hundred rows, so `ray.put` once is the
      right choice. For data that does not fit in memory, `ray.data` streams shards to
      workers (`ray.train.get_dataset_shard`), overlapping I/O and training.
    * **Placement constraints.** A 4-GPU `PACK` job needs 4 free GPUs *at once*; on a busy
      cluster it waits (queueing) rather than starting partially.

    ## 7. Connection to this repository

    | Concept | Where |
    |---|---|
    | driver: data `ray.put`, `ScalingConfig`, `RunConfig`, `FailureConfig`, `CheckpointConfig(num_to_keep)` | `src/ml_platform/training/distributed.py::train_distributed` |
    | worker loop: `get_context()`, `get_checkpoint()`, `report()`, commit barrier, all-gather summary | `training/distributed.py::_train_loop_per_worker` |
    | reading the result | `training/distributed.py::distributed_info`, `param_digest`, `ray_train_storage` |
    | DDP math, resume, failure injection | `src/ml_platform/training/trainer.py` — `Trainer.train(resume_from=, fail_at_epoch=, on_epoch_end=)`, `SimulatedWorkerFailure` |
    | resource choice (CUDA vs Metal) | `src/ml_platform/ray_runtime/resources.py::select_resources`, `WorkerResources.scaling_config_kwargs` |
    | Ray init (runtime env, faster Train health checks) | `src/ml_platform/ray_runtime/cluster.py::ensure_ray` (`RAY_TRAIN_HEALTH_CHECK_INTERVAL_S`) |
    | tests | `tests/integration/test_ray_train.py`, `tests/integration/test_failure_recovery.py` |
    | on Kubernetes | `infra/k8s/raycluster.yaml`, `infra/k8s/rayjob.yaml` (notebook 12) |

    ## 8. Failure modes

    * **Requesting GPUs that are not CUDA** (Apple Metal) — schedules, then fails in NCCL/`cuda:` device calls.
    * **Infeasible `ScalingConfig`** (more workers × CPUs than the cluster has) — the job waits forever for its placement group.
    * **Only rank 0 calling `ray.train.report`** — the report is a collective; the other ranks hang.
    * **Worker-local checkpoint paths** — on a multi-node cluster `storage_path` must be shared storage (NFS/S3); a path on one node is invisible after the group moves.
    * **Failure injected before the checkpoint is durable** — recovery restarts from the *previous* epoch (why `_train_loop_per_worker` waits on `get_all_reported_checkpoints` before failing).
    * **Retrying deterministic bugs** — `max_failures` burns time and GPU hours; fail fast on non-transient errors.
    * **Every rank reading the dataset from disk / a database** — N× I/O and possibly N different snapshots; read once and broadcast via the object store, or use versioned data.
    * **Changing world size on resume** and expecting identical results.
    * **Too many kept checkpoints** filling the disk (`num_to_keep`).

    ## 9. Exercise

    1. Run `train_distributed(cfg, round_id=0, num_workers=1)` and compare `val_rmse` and `global_batch_size` with the
       2-worker baseline. Then set `training.batch_size` so the *effective* batch size matches and compare again.
    2. Call `train_distributed(..., fail_at_epoch=2, max_failures=0)`. What exception reaches the driver, and what is
       left in `ray_train_storage(cfg)`? Resume manually with `resume_from=` pointing at the persisted checkpoint.
    3. Request `num_workers=8` with `distributed.cpus_per_worker=1` on this 4-CPU Ray cluster. What does
       `select_resources` do? What would happen if you bypassed it and passed `ScalingConfig(num_workers=8)` directly?
    """),
    code(r"""
    ray.shutdown()
    shutil.rmtree(RAY_TMP, ignore_errors=True)
    shutil.rmtree(WORK, ignore_errors=True)
    print("ray initialized:", ray.is_initialized(), "| cleaned up", WORK)
    """),
]

if __name__ == "__main__":
    write("04_ray_train_and_resource_management", cells)
