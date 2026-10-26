"""Distributed data-parallel training: Ray Train (placement) + PyTorch DDP (synchronization).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Ray determines where workers execute.                                   │
    │ PyTorch DDP determines how model replicas synchronize gradients.        │
    │                                                                         │
    │ Ray Train: reserves resources for N workers as one placement group      │
    │ (gang scheduling: all or nothing), starts one worker *process* per      │
    │ rank on whichever nodes have the CPUs/GPUs, sets MASTER_ADDR/PORT,      │
    │ RANK, WORLD_SIZE, LOCAL_RANK and calls                                  │
    │ ``torch.distributed.init_process_group`` (gloo on CPU, nccl on CUDA).   │
    │ It also persists reported checkpoints to ``storage_path`` and restarts  │
    │ the whole worker group on failure (``FailureConfig``).                  │
    │                                                                         │
    │ PyTorch DDP (inside ``training.trainer.Trainer``): one model replica    │
    │ per process, ``DistributedSampler`` shards the data, and every          │
    │ ``loss.backward()`` all-reduces (averages) gradients, so after each     │
    │ optimizer step all replicas hold identical parameters.                  │
    └─────────────────────────────────────────────────────────────────────────┘

The training code itself (`Trainer`) knows nothing about Ray: the very same
class runs single-process, under ``torchrun``, or here. This module only
adds the Ray-specific glue:

* **Data movement**: the driver reads the immutable round union once, splits
  train/validation deterministically, and ``ray.put``s both frames into the
  object store. Workers ``ray.get`` the refs (a zero-copy read on the same
  node, one transfer per node otherwise). The dataset is tiny (hundreds to
  thousands of rows), so this beats re-reading parquet in every worker and
  guarantees every rank sees byte-identical data (same dataset hash). For
  datasets that do not fit in memory, ``ray.data`` streaming shards would
  replace this.
* **Checkpoints**: `Trainer` writes checkpoints on rank 0 into a
  worker-local directory; the ``on_epoch_end`` hook reports metrics on every
  rank and, on rank 0, a ``ray.train.Checkpoint`` of that directory, which
  Ray copies to ``<artifacts>/ray_train/<run>/epoch_XXXX``. Only the newest
  ``num_to_keep`` persisted checkpoints are kept (disk is precious).
* **Recovery**: at start-up each worker asks ``ray.train.get_checkpoint()``.
  After a failure Ray restarts the worker group and hands back the latest
  persisted checkpoint; the Trainer resumes from it (model, optimizer,
  epoch, history and *every rank's own* RNG state), so the recovered run ends
  with parameters bit-identical to an uninterrupted run with the same number
  of workers (``tests/integration/test_failure_recovery.py``).
* **Observability**: every log line carries ``rank``, ``world_size``,
  ``pid``, ``run_id``, ``round_id``, ``dataset_id`` and ``ray_job_id``.
  At the end the ranks ``all_gather`` their pid/host/parameter digest; the
  driver returns them in ``TrainResult.device_info["distributed"]`` (identical
  digests on all ranks = DDP kept the replicas in sync).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import ray
import ray.train
import ray.train.torch
import torch
import torch.distributed as dist
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from ml_platform.config import PlatformConfig
from ml_platform.data.datasets import ArrayDataset, RoundStore, content_hash, train_val_split
from ml_platform.logging import bound_ids, get_logger
from ml_platform.ray_runtime.cluster import ensure_ray
from ml_platform.ray_runtime.resources import WorkerResources, select_resources
from ml_platform.training import checkpointing as ckpt
from ml_platform.training.trainer import Trainer, TrainResult

log = get_logger(__name__)

DEFAULT_NUM_TO_KEEP = 2
SUMMARY_FILE = "worker_summary.json"


def param_digest(model: torch.nn.Module) -> str:
    """SHA-256 over a module's state dict (identical across DDP ranks after sync)."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode() + v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _ray_job_id() -> str | None:
    try:
        return str(ray.get_runtime_context().get_job_id())
    except Exception:  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- worker side
def _train_loop_per_worker(config: dict[str, Any]) -> None:
    """Runs in each Ray Train worker process (one per DDP rank)."""
    ctx = ray.train.get_context()
    rank, world_size = ctx.get_world_rank(), ctx.get_world_size()
    cfg = PlatformConfig.model_validate(config["cfg"])
    ids = {
        "run_id": config.get("run_id"),
        "round_id": config.get("round_id"),
        "dataset_id": config["dataset_hash"],
        "ray_job_id": _ray_job_id(),
        "rank": rank,
        "world_size": world_size,
        "pid": os.getpid(),
    }
    wlog = get_logger("ray_train.worker", **ids)
    device = ray.train.torch.get_device()

    # Data from the object store (the driver put it there exactly once).
    train_frame: pd.DataFrame = ray.get(config["train_ref"])
    val_frame: pd.DataFrame = ray.get(config["val_ref"])
    train_ds = ArrayDataset.from_frame(train_frame, dataset_hash=config["dataset_hash"])
    val_ds = ArrayDataset.from_frame(val_frame, dataset_hash=config["dataset_hash"])

    local_root = Path(tempfile.mkdtemp(prefix=f"ml_ray_train_rank{rank}_"))
    try:
        # Recovery: Ray hands back the latest *persisted* checkpoint after a restart.
        restored = ray.train.get_checkpoint()
        resume_from: str | None = config.get("resume_from")
        resumed_source: str | None = resume_from
        fail_at_epoch = config.get("fail_at_epoch")
        if restored is not None:
            resume_from = restored.to_directory(str(local_root / "restored"))
            resumed_source = restored.path
            # the failure was injected on the first attempt; never re-inject after recovery
            fail_at_epoch = None
        restored_epoch = (
            int(ckpt.load_checkpoint(resume_from)["epoch"]) if resume_from is not None else None
        )
        wlog.info(
            "ray_train.worker_start",
            device=str(device),
            hostname=socket.gethostname(),
            n_train=len(train_ds),
            restored_from_epoch=restored_epoch,
            restored_by_ray=restored is not None,
        )

        ckpt_dir = local_root / "checkpoints"

        def on_epoch_end(epoch: int, metrics: dict[str, float], path: Path | None) -> None:
            # Called on every rank (ray.train.report is a collective); rank 0 attaches the checkpoint.
            checkpoint = ray.train.Checkpoint.from_directory(str(path)) if path else None
            ray.train.report(
                {k: float(v) for k, v in metrics.items()},
                checkpoint=checkpoint,
                checkpoint_dir_name=ckpt.epoch_dir_name(epoch) if checkpoint else None,
                delete_local_checkpoint_after_upload=False,
            )
            if fail_at_epoch is not None and epoch == fail_at_epoch:
                # Commit barrier: block until the controller has registered this
                # checkpoint, so the injected failure happens *after* it is durable
                # (otherwise the controller may see the error first and restore
                # from the previous epoch).
                ray.train.get_all_reported_checkpoints(timeout_s=120)
            if path is not None:  # keep only the newest local checkpoint
                for old in ckpt_dir.glob("epoch_*"):
                    if old.name != path.name:
                        shutil.rmtree(old, ignore_errors=True)

        with bound_ids(**ids):
            trainer = Trainer(cfg, device=device, run_id=config.get("run_id"))
            result = trainer.train(
                train_ds,
                val_ds,
                checkpoint_dir=ckpt_dir,
                resume_from=resume_from,
                fail_at_epoch=fail_at_epoch,
                on_epoch_end=on_epoch_end,
            )

        me = {
            "rank": rank,
            "local_rank": ctx.get_local_rank(),
            "world_size": world_size,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "device": str(device),
            "param_digest": param_digest(trainer.base_model),
            "n_train_rows_seen_per_epoch": len(range(rank, len(train_ds), world_size)),
        }
        workers: list[Any] = [me]
        if dist.is_available() and dist.is_initialized():
            workers = [None] * world_size
            dist.all_gather_object(workers, me)
        wlog.info("ray_train.worker_done", param_digest=me["param_digest"][:12])
        if rank == 0:
            # Result.metrics only carries the metrics reported *with* the latest
            # checkpoint, so the end-of-run summary goes to a file in the run's
            # storage directory (shared storage is required for multi-node Ray Train anyway).
            summary = {
                "result": result.to_dict(),
                "workers": workers,
                "restored_from_epoch": restored_epoch,
                "recovered_by_ray": restored is not None,
                "resumed_source": resumed_source,
            }
            out = Path(config["summary_path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(summary, default=str, indent=2))
            tmp.replace(out)
    finally:
        shutil.rmtree(local_root, ignore_errors=True)


# --------------------------------------------------------------------------- driver side
def load_training_frame(cfg: PlatformConfig, round_id: int | None) -> tuple[pd.DataFrame, str, int]:
    """Union of rounds ``0..round_id`` from the RoundStore + its dataset id."""
    store = RoundStore(cfg.paths.resolved().data_dir)
    rid = store.latest_round_id() if round_id is None else round_id
    if rid is None:
        raise FileNotFoundError(f"no experimental rounds in {store.root}")
    return store.training_frame(rid), store.dataset_hash(rid), rid


def ray_train_storage(cfg: PlatformConfig) -> Path:
    """Absolute Ray Train ``storage_path`` (under the repo/config artifacts dir)."""
    return (cfg.paths.resolved().artifacts_dir / "ray_train").resolve()


def train_distributed(
    cfg: PlatformConfig,
    *,
    round_id: int | None = None,
    train_frame: pd.DataFrame | None = None,
    num_workers: int | None = None,
    fail_at_epoch: int | None = None,
    resume_from: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    max_failures: int = 1,
    num_to_keep: int = DEFAULT_NUM_TO_KEEP,
    run_name: str | None = None,
    dataset_hash: str | None = None,
) -> TrainResult:
    """Train the surrogate with Ray Train + DDP; returns a driver-side `TrainResult`.

    ``train_frame`` defaults to ``RoundStore.training_frame(round_id)`` (latest
    round if None). ``fail_at_epoch`` injects `SimulatedWorkerFailure` right
    after that epoch's checkpoint (first attempt only); with
    ``max_failures >= 1`` Ray restarts the workers, which resume from the
    persisted checkpoint. ``result.checkpoint_path`` is the persisted final
    checkpoint (loadable with ``Predictor.from_checkpoint``).
    ``result.device_info["distributed"]`` has per-rank pids / hosts /
    parameter digests and recovery info.
    """
    t0 = time.perf_counter()
    if train_frame is None:
        frame, ds_hash, round_id = load_training_frame(cfg, round_id)
        dataset_hash = dataset_hash or ds_hash
    else:
        frame = train_frame
        dataset_hash = dataset_hash or content_hash(frame)
    tc = cfg.training
    tr, va = train_val_split(frame, tc.val_fraction, cfg.seed, tc.val_strategy)

    ensure_ray()
    job_id = _ray_job_id()
    res: WorkerResources = select_resources(cfg, num_workers=num_workers)
    storage = ray_train_storage(cfg)
    storage.mkdir(parents=True, exist_ok=True)
    rid = f"r{round_id:03d}" if round_id is not None else "adhoc"
    name = run_name or f"train_{rid}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    dlog = get_logger(
        "ray_train.driver",
        run_id=run_id,
        round_id=round_id,
        dataset_id=dataset_hash,
        ray_job_id=job_id,
    )
    dlog.info(
        "ray_train.launch",
        name=name,
        storage_path=str(storage),
        num_workers=res.num_workers,
        use_gpu=res.use_gpu,
        cpus_per_worker=res.cpus_per_worker,
        n_train=len(tr),
        n_val=len(va),
        fail_at_epoch=fail_at_epoch,
        max_failures=max_failures,
        notes=list(res.notes),
    )

    loop_config = {
        "cfg": cfg.model_dump(mode="json"),
        "train_ref": ray.put(tr),
        "val_ref": ray.put(va),
        "dataset_hash": dataset_hash,
        "fail_at_epoch": fail_at_epoch,
        "resume_from": str(resume_from) if resume_from is not None else None,
        "run_id": run_id,
        "round_id": round_id,
        "summary_path": str(storage / name / SUMMARY_FILE),
    }
    trainer = TorchTrainer(
        _train_loop_per_worker,
        train_loop_config=loop_config,
        scaling_config=ScalingConfig(**res.scaling_config_kwargs()),
        run_config=RunConfig(
            name=name,
            storage_path=str(storage),
            failure_config=FailureConfig(max_failures=max_failures),
            checkpoint_config=CheckpointConfig(num_to_keep=num_to_keep),
        ),
    )
    ray_result = trainer.fit()
    if ray_result.checkpoint is None:
        raise RuntimeError(f"Ray Train run {name} finished without a checkpoint")

    summary = json.loads((storage / name / SUMMARY_FILE).read_text())
    rank0 = summary["result"]
    workers = summary["workers"]
    final_ckpt = Path(ray_result.checkpoint.path)
    restored_epoch = summary.get("restored_from_epoch")
    distributed_info = {
        "backend": "ray_train+torch_ddp",
        "ray_job_id": job_id,
        "ray_train_run": str(Path(ray_result.path)),
        "workers": workers,
        "pids": sorted({w["pid"] for w in workers}),
        "params_in_sync": len({w["param_digest"] for w in workers}) == 1,
        "failures_recovered": int(bool(summary.get("recovered_by_ray"))),
        "restored_from_epoch": restored_epoch,
        "resources": res.to_dict(),
    }
    result = TrainResult(
        checkpoint_path=final_ckpt,
        metrics={k: float(v) for k, v in rank0["metrics"].items()},
        epochs_completed=int(rank0["epochs_completed"]),
        duration_s=time.perf_counter() - t0,
        world_size=int(rank0["world_size"]),
        device=rank0["device"],
        seed=int(rank0["seed"]),
        dataset_hash=rank0["dataset_hash"],
        git_sha=rank0.get("git_sha"),
        hyperparams={
            **rank0["hyperparams"],
            "distributed.num_workers": res.num_workers,
            "distributed.use_gpu": res.use_gpu,
            "distributed.cpus_per_worker": res.cpus_per_worker,
            "distributed.global_batch_size": tc.batch_size * res.num_workers,
        },
        device_info={**rank0["device_info"], "distributed": distributed_info},
        history=[dict(h) for h in rank0["history"]],
        resumed_from=summary.get("resumed_source") or None,
        checkpoint_hash=ckpt.checkpoint_hash(final_ckpt),
        n_parameters=int(rank0["n_parameters"]),
        n_train=int(rank0["n_train"]),
        n_val=int(rank0["n_val"]),
    )
    dlog.info(
        "ray_train.done",
        epochs=result.epochs_completed,
        world_size=result.world_size,
        pids=distributed_info["pids"],
        params_in_sync=distributed_info["params_in_sync"],
        val_rmse=result.metrics.get("val_rmse"),
        failures_recovered=distributed_info["failures_recovered"],
        checkpoint=str(final_ckpt),
        duration_s=round(result.duration_s, 2),
    )
    return result


def distributed_info(result: TrainResult) -> dict[str, Any]:
    """The Ray/DDP section of a distributed `TrainResult` (empty for single-process runs)."""
    info = result.device_info.get("distributed", {})
    return dict(info) if isinstance(info, dict) else {}
