"""Resource selection: decide *what to ask Ray for* so CPU-only machines keep working.

Ray schedules a task/actor/Train worker onto a node only when the node has the
requested logical resources free (``num_cpus``, ``num_gpus``, custom
resources). Asking for a GPU on a CPU-only cluster does not fail — the request
just stays *pending forever*. Therefore resource requests are chosen at run
time from what actually exists, never hard-coded:

.. code-block:: python

    # static declaration: only schedulable where a GPU exists
    @ray.remote(num_cpus=2, num_gpus=1)
    def score_shard(...): ...

    # dynamic override chosen from the machine/cluster (works CPU-only too)
    res = select_resources(cfg)
    ref = score_shard.options(**res.remote_options()).remote(...)

GPU detection caveat (Apple silicon): Ray advertises ``GPU: 1`` on an M-series
Mac (Metal), but PyTorch DDP/NCCL needs **CUDA**. ``use_gpu: auto`` therefore
means "CUDA is usable", decided by ``torch.cuda.is_available()`` on the
driver, or — when attached to a remote cluster whose GPUs the driver cannot
see — by probing a GPU worker. Ray's raw ``GPU`` count is never trusted alone.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import ray
import torch

from ml_platform.config import DistributedConfig, PlatformConfig
from ml_platform.logging import get_logger

log = get_logger(__name__)

_APPLE_ACCELERATOR = re.compile(r"^accelerator_type:M\d", re.IGNORECASE)


@dataclass(frozen=True)
class WorkerResources:
    """Per-worker resource request + worker count for Ray Train / tasks."""

    num_workers: int
    use_gpu: bool
    cpus_per_worker: int
    gpus_per_worker: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def remote_options(self) -> dict[str, Any]:
        """Kwargs for ``fn.options(**...)`` / ``Actor.options(**...)``."""
        return {"num_cpus": self.cpus_per_worker, "num_gpus": self.gpus_per_worker}

    def resources_per_worker(self) -> dict[str, float]:
        """``ScalingConfig(resources_per_worker=...)`` (GPU is added by ``use_gpu``)."""
        return {"CPU": float(self.cpus_per_worker)}

    def scaling_config_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "use_gpu": self.use_gpu,
            "resources_per_worker": {
                **self.resources_per_worker(),
                **({"GPU": self.gpus_per_worker} if self.use_gpu else {}),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d


# --------------------------------------------------------------------------- probing
def _cluster_resources() -> dict[str, float]:
    return dict(ray.cluster_resources()) if ray.is_initialized() else {}


@ray.remote(num_gpus=1, num_cpus=0, max_retries=0)
def _probe_cuda() -> int:
    import torch as _torch

    return _torch.cuda.device_count() if _torch.cuda.is_available() else 0


def cuda_gpu_count(*, probe_timeout_s: float = 30.0) -> int:
    """Number of CUDA GPUs usable by workers (0 on Apple silicon / CPU-only)."""
    if torch.cuda.is_available():
        local = torch.cuda.device_count()
        return max(local, int(_cluster_resources().get("GPU", 0)))
    res = _cluster_resources()
    ray_gpus = int(res.get("GPU", 0))
    if ray_gpus == 0:
        return 0
    accel = [k for k in res if k.startswith("accelerator_type:")]
    if accel and all(_APPLE_ACCELERATOR.match(k) for k in accel):
        return 0  # Metal GPU advertised by Ray; not usable for CUDA/NCCL
    # remote cluster: ask a GPU worker (bounded wait so a busy cluster cannot hang us)
    ref = _probe_cuda.remote()
    ready, _ = ray.wait([ref], timeout=probe_timeout_s)
    if not ready:
        ray.cancel(ref, force=True)
        log.warning("resources.cuda_probe_timeout", ray_gpus=ray_gpus)
        return 0
    return ray_gpus if int(ray.get(ready[0])) > 0 else 0


def available_cpus() -> int:
    res = _cluster_resources()
    return int(res.get("CPU", 0)) or (os.cpu_count() or 1)


# --------------------------------------------------------------------------- selection
def select_resources(
    cfg: PlatformConfig | DistributedConfig, *, num_workers: int | None = None
) -> WorkerResources:
    """Choose worker count and per-worker resources for the current machine/cluster.

    * ``use_gpu: auto`` -> GPU iff CUDA is usable (see module docstring);
      ``use_gpu: true`` without CUDA degrades to CPU with a note;
    * GPU runs use one GPU per worker and at most one worker per GPU;
    * the worker count is capped so ``num_workers * cpus_per_worker`` fits
      the available CPUs (a 4-core laptop still runs the 2-worker demo).
    """
    dcfg = cfg.distributed if isinstance(cfg, PlatformConfig) else cfg
    notes: list[str] = []
    requested = max(1, int(num_workers or dcfg.num_workers))
    total_cpus = available_cpus()
    cpus_per_worker = max(1, min(int(dcfg.cpus_per_worker), total_cpus))
    if cpus_per_worker != dcfg.cpus_per_worker:
        notes.append(f"cpus_per_worker capped to {cpus_per_worker} (cluster has {total_cpus})")

    want_gpu = dcfg.use_gpu
    n_gpus = cuda_gpu_count() if want_gpu else 0
    use_gpu = n_gpus > 0
    if want_gpu is True and not use_gpu:
        notes.append("use_gpu=true but no CUDA GPU is available; falling back to CPU")

    n = requested
    if use_gpu and n > n_gpus:
        n = n_gpus
        notes.append(f"num_workers capped to {n} (one worker per CUDA GPU)")
    max_by_cpu = max(1, total_cpus // cpus_per_worker)
    if n > max_by_cpu:
        n = max_by_cpu
        notes.append(f"num_workers capped to {n} by {total_cpus} available CPUs")

    res = WorkerResources(
        num_workers=n,
        use_gpu=use_gpu,
        cpus_per_worker=cpus_per_worker,
        gpus_per_worker=1.0 if use_gpu else 0.0,
        notes=tuple(notes),
    )
    log.info(
        "resources.selected",
        **{k: v for k, v in res.to_dict().items() if k != "notes"},
        notes=notes,
    )
    return res


# --------------------------------------------------------------------------- example
@ray.remote(num_cpus=2, num_gpus=1)
def gpu_matmul_benchmark(n: int = 512) -> dict[str, Any]:
    """Example of a *statically* GPU-annotated task.

    Scheduling it as-is on a CPU-only cluster would leave it pending forever;
    call it via :func:`run_accelerated_example`, which overrides the request
    with ``.options(**select_resources(cfg).remote_options())``.
    """
    import time

    import torch as _torch

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    a = _torch.randn(n, n, device=device)
    t0 = time.perf_counter()
    for _ in range(5):
        a = (a @ a).tanh()
    if device.type == "cuda":
        _torch.cuda.synchronize()
    return {"device": str(device), "seconds": time.perf_counter() - t0, "pid": os.getpid()}


def run_accelerated_example(cfg: PlatformConfig, n: int = 256) -> dict[str, Any]:
    """Run :func:`gpu_matmul_benchmark` with resources chosen for this machine."""
    res = select_resources(cfg, num_workers=1)
    opts = {"num_cpus": 1, "num_gpus": res.gpus_per_worker}
    out: dict[str, Any] = ray.get(gpu_matmul_benchmark.options(**opts).remote(n))
    return {**out, "requested": opts}
