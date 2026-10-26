"""Generate notebooks/02_distributed_training_ddp.ipynb."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbhelpers_a import code, md, setup_cell, write  # noqa: E402

DDP_SCRIPT = r'''
"""Launched by torchrun: one OS process per rank (written by notebook 02)."""
import json, os, sys, time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

out_dir = Path(sys.argv[1])
dist.init_process_group("gloo")      # env:// rendezvous: torchrun sets RANK, WORLD_SIZE, MASTER_ADDR/PORT
rank, world = dist.get_rank(), dist.get_world_size()
torch.set_num_threads(1)


def make_model():
    torch.manual_seed(0)             # identical init everywhere (DDP also broadcasts rank 0's params)
    return nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 1))


# every rank builds the same GLOBAL batch and takes its own contiguous shard of it
g = torch.Generator().manual_seed(123)
X = torch.randn(64, 32, generator=g)
y = torch.randn(64, generator=g)
lo, hi = rank * 64 // world, (rank + 1) * 64 // world

model = DDP(make_model())
loss = ((model(X[lo:hi]).squeeze(-1) - y[lo:hi]) ** 2).mean()
local_loss = loss.item()
loss.backward()                      # <-- DDP all-reduces (averages) gradients inside backward
grad = torch.cat([p.grad.flatten() for p in model.parameters()])
all_grads = [torch.empty_like(grad) for _ in range(world)]
dist.all_gather(all_grads, grad)     # collect every rank's post-backward gradient

opt = torch.optim.SGD(model.parameters(), lr=0.1)
opt.step()                           # same grads + same params -> same update on every rank
digest = float(sum(p.detach().double().sum() for p in model.parameters()))

# communication vs computation: a 1M-parameter model, with and without the all-reduce
big = DDP(nn.Sequential(*[nn.Linear(512, 512) for _ in range(4)]))
xb = torch.randn(32, 512)
t_sync, t_nosync = [], []
for _ in range(7):
    t0 = time.perf_counter(); big(xb).sum().backward(); t_sync.append(time.perf_counter() - t0)
with big.no_sync():                  # no_sync(): backward WITHOUT all-reduce (gradient accumulation)
    for _ in range(7):
        t0 = time.perf_counter(); big(xb).sum().backward(); t_nosync.append(time.perf_counter() - t0)

# straggler: the last rank is 0.5 s slow; the all-reduce is a barrier, so everyone waits
dist.barrier()
t0 = time.perf_counter()
if rank == world - 1:
    time.sleep(0.5)
dist.all_reduce(torch.ones(1))
t_wait = time.perf_counter() - t0

ref_diff = None
if rank == 0:                        # reference: ONE process, full global batch, no DDP
    ref = make_model()
    ((ref(X).squeeze(-1) - y) ** 2).mean().backward()
    ref_grad = torch.cat([p.grad.flatten() for p in ref.parameters()])
    ref_diff = float((grad - ref_grad).abs().max())

med = lambda v: 1e3 * sorted(v)[len(v) // 2]
(out_dir / f"grad_rank{rank}.json").write_text(json.dumps({
    "rank": rank, "world_size": world, "pid": os.getpid(), "rows": [lo, hi],
    "local_loss": local_loss,
    "max_grad_diff_between_ranks": float(max((a - all_grads[0]).abs().max() for a in all_grads)),
    "max_grad_diff_vs_single_process_full_batch": ref_diff,
    "param_sum_after_step": digest,
    "step_ms_with_allreduce": med(t_sync), "step_ms_no_sync": med(t_nosync),
    "straggler_wait_s": t_wait, "big_model_params": sum(p.numel() for p in big.parameters()),
}))
dist.destroy_process_group()
'''

TRAINER_SCRIPT = r'''
"""The repository Trainer under torchrun (written by notebook 02)."""
import hashlib, json, os, sys
from pathlib import Path

import torch
import torch.distributed as dist

from ml_platform.config import PlatformConfig
from ml_platform.data import (ArrayDataset, generate_candidate_pool, initial_observations,
                                 make_oracle, records_to_frame, train_val_split)
from ml_platform.training import Trainer

out_dir = Path(sys.argv[1])
torch.set_num_threads(1)
if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    dist.init_process_group("gloo")
cfg = PlatformConfig.for_tests(out_dir).with_overrides(**{"training.epochs": 5})
pool = generate_candidate_pool(cfg)
frame = records_to_frame(initial_observations(pool, make_oracle(cfg), 300, seed=0))
tr, va = train_val_split(frame, 0.2, seed=0)
trainer = Trainer(cfg)
result = trainer.train(ArrayDataset.from_frame(tr, dataset_hash="nb02"),
                       ArrayDataset.from_frame(va, dataset_hash="nb02"),
                       checkpoint_dir=out_dir / f"ckpt_ws{trainer.world_size}")
h = hashlib.sha256()
for k, v in trainer.base_model.state_dict().items():
    h.update(k.encode() + v.detach().numpy().tobytes())
sampler_idx = list(iter(trainer._train_loader(1).sampler))
(out_dir / f"trainer_ws{trainer.world_size}_rank{trainer.rank}.json").write_text(json.dumps({
    "rank": trainer.rank, "world_size": trainer.world_size, "pid": os.getpid(),
    "val_rmse": result.metrics["val_rmse"], "train_loss": result.metrics["train_loss"],
    "param_digest": h.hexdigest()[:16], "n_indices_epoch1": len(sampler_idx),
    "first_indices_epoch1": sampler_idx[:6],
    "checkpoint_written_by_me": trainer.rank == 0,
}))
if dist.is_initialized():
    dist.destroy_process_group()
'''

cells = [
    md(r"""
    # 02 · Distributed training and DDP

    **Goal.** Understand *data-parallel* training precisely enough to reason about its
    correctness (why the result equals large-batch SGD), its cost (communication vs
    computation), and its failure modes (stragglers, hangs, divergent replicas). We run
    a real 2-process DDP job with `torchrun` from this notebook, verify numerically that
    DDP's averaged gradient equals the single-process full-batch gradient, and then run
    the repository's `Trainer` under DDP.

    Notebook 04 shows how Ray Train *places* these processes; this notebook is about
    what happens *inside* them.
    """),
    md(r"""
    ## 1. Conceptual model

    ### Processes, not threads
    Each DDP replica is a separate **OS process** with its own Python interpreter, its own
    copy of the model and optimizer, and (on GPU hosts) its own device. Threads would
    share one interpreter and fight over the GIL for every Python-level op; processes
    avoid that and map 1:1 onto GPUs and onto machines. The price: nothing is shared
    implicitly — processes cooperate only through explicit **collectives**.

    ### Vocabulary
    | Term | Meaning |
    |---|---|
    | **world size** $N$ | number of processes in the job |
    | **rank** $k \in \{0..N-1\}$ | global id of a process; rank 0 conventionally does I/O (checkpoints, logs) |
    | **local rank** | id within one machine → which GPU to use (`cuda:<local_rank>`) |
    | **process group** | the set of ranks that take part in a collective, plus a backend: **gloo** (CPU, TCP) or **nccl** (CUDA GPUs) |
    | **rendezvous** | how processes find each other: `MASTER_ADDR`/`MASTER_PORT` + `RANK`/`WORLD_SIZE` env vars (`env://`) |
    | **collective** | an operation all ranks must call: `all_reduce`, `all_gather`, `broadcast`, `barrier` |

    ```text
                        ┌──────────── global batch of B = N·b rows ────────────┐
    DistributedSampler  │ shard 0 (b rows) │ shard 1 (b rows) │ … │ shard N-1 │
                        └────────┬─────────┴────────┬─────────┴───┴─────┬─────┘
                                 ▼                  ▼                   ▼
                        rank 0: fwd/bwd     rank 1: fwd/bwd     rank N-1: fwd/bwd     (replicas θ identical)
                           g_0                 g_1                 g_{N-1}
                             └────────── all_reduce(sum) / N ──────────┘              (inside loss.backward())
                                               ḡ  (same on all ranks)
                        rank k: θ ← opt.step(θ, ḡ)   → replicas stay identical, no parameter broadcast needed
    ```
    """),
    md(r"""
    ## 2. Derivation — why DDP **averages** gradients

    Let the global batch $\mathcal{B}$ of size $B$ be split into $N$ disjoint shards
    $\mathcal{B}_k$ of equal size $b = B/N$. The per-shard mean loss is
    $L_k(\theta) = \frac{1}{b}\sum_{i \in \mathcal{B}_k} \ell_i(\theta)$. The full-batch mean loss is

    $$
    L(\theta) = \frac{1}{B}\sum_{i\in\mathcal{B}} \ell_i(\theta)
              = \frac{1}{N}\sum_{k=1}^{N} \frac{1}{b}\sum_{i\in\mathcal{B}_k}\ell_i(\theta)
              = \frac{1}{N}\sum_{k=1}^{N} L_k(\theta).
    $$

    Differentiation is linear, so

    $$
    \boxed{\nabla L(\theta) = \frac{1}{N}\sum_{k=1}^{N}\nabla L_k(\theta)}
    $$

    Each rank computes $\nabla L_k$ locally; `all_reduce(SUM)` followed by division by $N$
    (what DDP does) yields exactly the gradient of **one process training on the whole
    global batch**. Consequences:

    * **Synchronous data-parallel SGD = large-batch SGD** with *effective batch size*
      $B_\text{eff} = b \times N$ (per-worker batch × world size). In this repo
      `training.batch_size` is **per worker**; `train_distributed` logs
      `distributed.global_batch_size = batch_size × num_workers`.
    * If DDP *summed* instead, the step would be $N\times$ larger — changing the world size
      would silently change the learning rate.
    * Changing $N$ at fixed $b$ changes $B_\text{eff}$, hence the optimization trajectory.
      The common heuristic is the **linear scaling rule**: $\eta_N = N\,\eta_1$ (keep
      $\eta / B_\text{eff}$ constant; valid while the gradient noise dominates), usually
      with warm-up; for Adam-type optimizers a $\sqrt{N}$ rule is often closer. Either way it
      is a heuristic, which is why world size is recorded as run metadata.
    * Exactness needs **equal shard sizes** (unequal shards weight rows unequally — hence
      `DistributedSampler` pads) and no cross-sample coupling in the loss (BatchNorm uses
      *per-shard* statistics unless you use `SyncBatchNorm`; dropout masks differ per rank).
    """),
    setup_cell("nb02"),
    code(r"""
    import numpy as np
    import torch
    from torch import nn

    torch.manual_seed(0)
    print("torch", torch.__version__, "| gloo available:", torch.distributed.is_gloo_available(),
          "| nccl available:", torch.distributed.is_nccl_available())
    """),
    md(r"""
    ### Numerical check in one process

    Before involving multiple processes, simulate $N$ workers inside one process: compute
    each shard's gradient separately, average them, and compare with the full-batch gradient.
    """),
    code(r"""
    def grads(model, X, y):
        model.zero_grad(set_to_none=True)
        ((model(X).squeeze(-1) - y) ** 2).mean().backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 1))
    X, y = torch.randn(96, 32), torch.randn(96)
    g_full = grads(net, X, y)
    for N in (2, 3, 4):
        shards = torch.arange(96).chunk(N)                    # equal shards of 96/N rows
        g_avg = torch.stack([grads(net, X[s], y[s]) for s in shards]).mean(0)
        g_sum = torch.stack([grads(net, X[s], y[s]) for s in shards]).sum(0)
        print(f"N={N}: max|mean_k g_k - g_full| = {float((g_avg - g_full).abs().max()):.2e}   "
              f"(sum instead of mean is {float(g_sum.norm() / g_full.norm()):.1f}x too large)")

    # unequal shards break the identity (rows in the small shard get more weight)
    s1, s2 = torch.arange(0, 80), torch.arange(80, 96)
    g_uneq = (grads(net, X[s1], y[s1]) + grads(net, X[s2], y[s2])) / 2
    print(f"unequal shards (80/16): max diff = {float((g_uneq - g_full).abs().max()):.2e}")
    """),
    md(r"""
    ## 3. `DistributedSampler` — index partitioning

    `DistributedSampler(dataset, num_replicas=N, rank=k, shuffle=True, seed=s)`:

    1. permutes `range(n)` with a generator seeded by `seed + epoch` (identical on all ranks — no communication needed);
    2. **pads** by repeating indices so the length is divisible by $N$ (`drop_last=False`), or truncates (`drop_last=True`);
    3. gives rank $k$ the strided slice `indices[k::N]`.

    It needs no process group, so we can inspect it directly:
    """),
    code(r"""
    from torch.utils.data import DistributedSampler

    ds = list(range(10))                      # any sized object works
    for epoch in (0, 1):
        parts = []
        for rank in range(3):
            s = DistributedSampler(ds, num_replicas=3, rank=rank, shuffle=True, seed=0, drop_last=False)
            s.set_epoch(epoch)                # forget this and every epoch has the same order
            parts.append(list(s))
        flat = sum(parts, [])
        print(f"epoch {epoch}: " + " | ".join(f"rank{r}={p}" for r, p in enumerate(parts)),
              f"-> {len(flat)} slots, duplicated: {sorted({i for i in flat if flat.count(i) > 1})}")

    s = DistributedSampler(ds, num_replicas=3, rank=0, shuffle=False, drop_last=True)
    print("drop_last=True, rank 0 (no shuffle):", list(s), "-> each rank gets", len(s), "indices, 1 row dropped")
    """),
    md(r"""
    With 10 rows and 3 ranks, padding duplicates 2 rows per epoch — a slight re-weighting
    that is harmless for large $n$ but visible for tiny scientific datasets. The repo's
    `Trainer._train_loader` uses `drop_last=False` for training, and its `validate()`
    deliberately does **not** use the sampler: it shards validation rows as
    `rank::world_size` *without padding* and all-reduces sufficient statistics, so
    validation metrics are exact.

    ## 4. A real 2-process DDP job

    Notebooks and `torch.multiprocessing.spawn` do not mix well (the child must re-import
    the worker function, which does not exist in a module when defined in a notebook). The
    robust pattern — and the one production uses — is a **script** launched by
    `torchrun`, which spawns one process per rank and sets `RANK`, `LOCAL_RANK`,
    `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`. We write the script into the scratch dir.

    The script: every rank builds the same global batch of 64 rows, takes its shard of 32,
    runs forward/backward through `DistributedDataParallel`, then gathers the gradients of
    all ranks. Rank 0 also computes the single-process full-batch gradient for comparison.
    It also times a step with and without the all-reduce (`no_sync()`), and injects a
    0.5 s straggler.
    """),
    code(r"""
    import socket, subprocess

    SCRIPTS = WORK / "scripts"; SCRIPTS.mkdir()
    (SCRIPTS / "ddp_grad_check.py").write_text(DDP_SCRIPT)
    (SCRIPTS / "trainer_ddp.py").write_text(TRAINER_SCRIPT)
    TORCHRUN = Path(sys.executable).parent / "torchrun"      # the venv's torchrun

    def free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

    def torchrun(script: Path, nproc: int, *args: str) -> float:
        cmd = [str(TORCHRUN), "--nnodes", "1", "--nproc_per_node", str(nproc),
               "--master_addr", "127.0.0.1", "--master_port", str(free_port()), str(script), *args]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore"}
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(proc.stdout[-3000:], proc.stderr[-3000:]); raise RuntimeError("torchrun failed")
        return time.perf_counter() - t0

    print("launch:", " ".join(["torchrun", "--nproc_per_node", "2", "ddp_grad_check.py", "<out_dir>"]))
    elapsed = torchrun(SCRIPTS / "ddp_grad_check.py", 2, str(WORK))
    ranks = [json.loads((WORK / f"grad_rank{r}.json").read_text()) for r in range(2)]
    print(f"torchrun finished in {elapsed:.1f}s")
    for r in ranks:
        print(f"rank {r['rank']} (pid {r['pid']}): rows {r['rows']}, local loss {r['local_loss']:.4f}, "
              f"param sum after SGD step {r['param_sum_after_step']:.6f}")
    print("max |grad_rank_k - grad_rank_0| after backward      :", ranks[0]["max_grad_diff_between_ranks"])
    print("max |DDP grad - single-process full-batch grad|      :", ranks[0]["max_grad_diff_vs_single_process_full_batch"])
    """),
    md(r"""
    The two ranks computed **different local losses** on different rows, yet after
    `backward()` they hold **identical** gradients (difference exactly 0: all-reduce hands
    every rank the same bytes), and those equal the single-process full-batch gradient up
    to float32 round-off (~1e-8: the reduction order differs). After `optimizer.step()`
    the parameters are therefore identical on both ranks — DDP never broadcasts
    parameters after initialization; it relies on "same start + same gradients + same
    optimizer ⇒ same parameters".

    ## 5. Communication / computation ratio

    DDP overlaps the all-reduce with the backward pass (gradients are bucketed, ~25 MB per
    bucket, and a bucket is reduced as soon as all its gradients are ready). A **ring
    all-reduce** of $P$ parameters sends about $2\frac{N-1}{N}P$ values per worker —
    nearly independent of $N$ — so per step

    $$
    T_\text{step} \approx T_\text{compute}(b) + \max\!\big(0,\; T_\text{comm}(P) - T_\text{overlap}\big),
    \qquad T_\text{comm} \approx 2\tfrac{N-1}{N}\,\frac{4P\ \text{bytes}}{\text{bandwidth}} + \text{latency}.
    $$

    Compute shrinks with the per-worker batch $b$ while communication depends only on the
    model size $P$. Small models with small per-worker batches (like this repo's surrogate)
    are therefore **communication-bound**: scaling out makes each step *slower*, and data
    parallelism only pays when $T_\text{compute}(b) \gg T_\text{comm}(P)$.
    """),
    code(r"""
    r0 = ranks[0]
    P = r0["big_model_params"]
    ratio = (r0["step_ms_with_allreduce"] - r0["step_ms_no_sync"]) / r0["step_ms_no_sync"]
    print(f"model: {P/1e6:.2f}M params = {4*P/1e6:.1f} MB of fp32 gradients, per-worker batch 32")
    print(f"fwd+bwd without all-reduce (no_sync): {r0['step_ms_no_sync']:.2f} ms")
    print(f"fwd+bwd with all-reduce             : {r0['step_ms_with_allreduce']:.2f} ms")
    print(f"communication overhead / compute    : {ratio:.1f}x   (gloo over loopback TCP)")
    """),
    md(r"""
    Even on one machine over loopback, synchronizing 4 MB of gradients costs several times
    the compute for this batch. On GPUs with NCCL over NVLink/InfiniBand the bandwidth is
    100–1000× higher, but so is the compute speed, so the *ratio* is what you tune: larger
    per-worker batches, gradient accumulation (`no_sync()` for $k-1$ micro-batches, then
    one synchronized backward), fp16/bf16 gradient compression, or fewer, bigger nodes.

    ## 6. Stragglers

    Every collective is an implicit **barrier**: the step time is the time of the
    *slowest* rank, $T_\text{step} = \max_k T_k$. One slow worker (thermal throttling, a
    noisy neighbour, a slow data shard, GC pause) slows everyone; a *dead* worker makes
    the others block until the collective timeout (default 30 min for gloo/nccl!) unless
    something (torchrun's elastic agent, Ray Train's controller) tears the group down.
    """),
    code(r"""
    for r in ranks:
        print(f"rank {r['rank']}: time until all_reduce returned = {r['straggler_wait_s']:.3f}s"
              + ("   <- the straggler (slept 0.5 s)" if r["rank"] == 1 else "   <- idle, waiting for rank 1"))
    """),
    md(r"""
    ## 7. The repository `Trainer` under DDP

    `ml_platform.training.trainer.Trainer` needs no code change to run distributed: it
    checks `torch.distributed.is_initialized()` and then wraps the model in
    `DistributedDataParallel`, uses `DistributedSampler` (with `set_epoch`), all-reduces
    its loss/metric sufficient statistics, and lets only rank 0 write checkpoints (the
    other ranks wait at a `barrier`). We run the same script with 1 and 2 processes.
    """),
    code(r"""
    t1 = torchrun(SCRIPTS / "trainer_ddp.py", 1, str(WORK))
    t2 = torchrun(SCRIPTS / "trainer_ddp.py", 2, str(WORK))
    rows = [json.loads(p.read_text()) for p in sorted(WORK.glob("trainer_ws*_rank*.json"))]
    print(f"wall time incl. process start-up: world_size=1 {t1:.1f}s, world_size=2 {t2:.1f}s\n")
    print(f"{'ws':>2} {'rank':>4} {'pid':>6} {'val_rmse':>9} {'param_digest':>17} {'#idx/epoch':>10}  first sampler indices")
    for r in rows:
        print(f"{r['world_size']:>2} {r['rank']:>4} {r['pid']:>6} {r['val_rmse']:>9.4f} {r['param_digest']:>17} "
              f"{r['n_indices_epoch1']:>10}  {r['first_indices_epoch1']}")
    print("\ncheckpoint dirs:", sorted(p.name for p in WORK.glob("ckpt_ws*")),
          "| ws=2 epochs saved:", len(list((WORK / "ckpt_ws2").glob("epoch_*"))), "(rank 0 only)")
    """),
    md(r"""
    What to read from the table:

    * **Within** the world-size-2 run both ranks report the **same parameter digest** (the
      replicas stayed in sync) and the **same validation RMSE** (metrics were all-reduced,
      not rank-local), while each saw a different half of the indices. The halves are the
      strided slices `perm[0::2]` and `perm[1::2]` of the *same* epoch permutation the
      single-process run used (both samplers are seeded from `(seed, epoch)`).
    * **Across** world sizes the result differs: with per-worker batch 64, world size 2 means
      an effective batch of 128 → half as many optimizer steps per epoch and a different
      trajectory. Reproducing a run therefore requires the same world size — which is why
      the repo records `world_size` in checkpoints (`Trainer._payload`) and in MLflow.

    ## 8. Connection to this repository

    | Concept | Where |
    |---|---|
    | DDP wrap, `DistributedSampler`, `set_epoch`, rank-0 checkpoints, `barrier` | `src/ml_platform/training/trainer.py` — `Trainer.setup`, `Trainer._train_loader`, `Trainer.save_checkpoint` |
    | exact distributed validation (no padding, all-reduced sufficient stats) | `Trainer.validate`, `Trainer._all_reduce` |
    | rank / world size discovery | `trainer.py::_dist_info` (reads `torch.distributed`, `LOCAL_RANK`) |
    | per-rank dropout streams | `Trainer.setup` (`torch.manual_seed(seed + 7919*(rank+1))`) |
    | device per local rank | `src/ml_platform/training/config.py::resolve_device` (`cuda:<local_rank>`) |
    | effective batch size recorded | `src/ml_platform/training/distributed.py::train_distributed` (`distributed.global_batch_size`) |
    | 2-process gloo test without Ray | `tests/integration/test_ddp_trainer.py` |
    | who launches the processes | Ray Train (`training/distributed.py`, notebook 04) — or `torchrun`, as here |

    ## 9. Failure modes

    * **Hangs** — one rank skips a collective (e.g. an `if rank == 0:` around code that calls `all_reduce`, or ranks with different numbers of batches) → everyone else blocks until the timeout. The repo calls `ray.train.report` on *every* rank for this reason.
    * **Divergent replicas** — different init seeds without DDP's broadcast, a parameter updated outside autograd, or unused parameters (`find_unused_parameters`).
    * **Silent LR change** when world size changes (effective batch size), or when gradients are summed instead of averaged.
    * **Forgetting `sampler.set_epoch(epoch)`** — identical shuffle every epoch.
    * **Padding/duplication** in `DistributedSampler` biasing tiny datasets; using it for validation double-counts rows.
    * **BatchNorm** statistics are per-rank unless `SyncBatchNorm`.
    * **Stragglers** — the slowest rank sets the pace; a dead rank blocks the others until the collective timeout.
    * **Every rank writing the same checkpoint/log file** — races and corruption; only rank 0 writes.
    * **Port collisions / rendezvous failures** — two jobs on the same `MASTER_PORT`.
    * **Communication-bound scaling** — adding workers to a small model makes it slower (§5).

    ## 10. Exercise

    1. Edit `DDP_SCRIPT` so rank 1 draws a *different* initialization (remove `torch.manual_seed(0)` in `make_model`)
       and *wrap* after changing it. Are gradients still identical across ranks? Why (hint: DDP's constructor broadcasts rank 0's state)?
       What happens if you instead modify a parameter in-place *after* wrapping?
    2. Launch `trainer_ddp.py` with `--nproc_per_node 3`. Compute the effective batch size and number of optimizer
       steps per epoch, and explain the change in `val_rmse`. Then apply the linear LR-scaling rule via
       `training.lr` and see whether it brings the result closer to world size 1.
    3. Measure `step_ms_with_allreduce / step_ms_no_sync` for per-worker batch sizes 8, 32, 128, 512. Plot the
       communication/computation ratio and find the batch size where DDP starts to pay off on this machine.
    """),
    code(r"""
    shutil.rmtree(WORK, ignore_errors=True)
    print("cleaned up", WORK)
    """),
]

# inject the worker scripts as string constants into the notebook
cells.insert(
    next(i for i, c in enumerate(cells) if "import socket, subprocess" in c.source),
    code("DDP_SCRIPT = r'''" + DDP_SCRIPT + "'''\n\nTRAINER_SCRIPT = r'''" + TRAINER_SCRIPT + "'''"),
)

if __name__ == "__main__":
    write("02_distributed_training_ddp", cells)
