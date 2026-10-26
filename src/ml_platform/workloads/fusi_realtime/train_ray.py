"""Ray Train + PyTorch DDP training of the fUSI decoder (same path as ``training/distributed.py``).

The driver puts the per-session feature arrays and the leakage-checked window table into
the object store once; each Ray Train worker builds its `WindowDataset`, and
``train.fit_decoder`` shards each global batch across ranks (`GlobalBatchSampler`) and wraps the model in DDP
(or FSDP2 when ``fusi.training.strategy == "fsdp2"``). Placement comes from
``ray_runtime.resources.select_resources`` (CPU-only on a Mac: Ray's Metal "GPU" is
ignored because DDP needs CUDA). Rank 0 writes the final checkpoint and a summary.
"""

from __future__ import annotations

import json
import os
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
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from ml_platform.config import PlatformConfig
from ml_platform.logging import get_logger
from ml_platform.ray_runtime.cluster import ensure_ray
from ml_platform.ray_runtime.resources import select_resources
from ml_platform.training.distributed import param_digest
from ml_platform.workloads.fusi_realtime import telemetry
from ml_platform.workloads.fusi_realtime.dataset import SessionArrays, WindowDataset
from ml_platform.workloads.fusi_realtime.models import build_decoder, save_decoder
from ml_platform.workloads.fusi_realtime.train import fit_decoder

log = get_logger(__name__)

CHECKPOINT_FILE = "model.pt"
SUMMARY_FILE = "fusi_summary.json"


def _loop_per_worker(config: dict[str, Any]) -> None:
    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    cfg = PlatformConfig.model_validate(config["cfg"]).fusi
    torch.set_num_threads(1)
    arrays: dict[str, SessionArrays] = ray.get(config["arrays_ref"])
    windows: pd.DataFrame = ray.get(config["windows_ref"])
    train_ds = WindowDataset(arrays, windows[windows.split == "train"])
    val_ds = WindowDataset(arrays, windows[windows.split == "val"])
    tracer = telemetry.configure(cfg.telemetry, service="fusi-train")
    # identical init on every rank (and every world size) before any DDP/FSDP wrapping
    torch.manual_seed(int(config["seed"]))
    model = build_decoder(cfg.decoder, tuple(config["grid"]), int(config["n_classes"]))  # type: ignore[arg-type]
    wlog = get_logger("fusi.ray_train.worker", rank=rank, world_size=world, pid=os.getpid())
    wlog.info("fusi.train.start", n_train=len(train_ds), n_val=len(val_ds))
    result = fit_decoder(
        model,
        train_ds,
        val_ds,
        cfg.training,
        seed=int(config["seed"]),
        # placement decides the device: the worker's CUDA GPU when use_gpu, else CPU
        device=ray.train.torch.get_device(),
        on_epoch_end=lambda e, m: ray.train.report({k: float(v) for k, v in m.items()}),
        tracer=tracer,
    )
    telemetry.flush()
    if result.strategy == "fsdp2":
        # parameters are sharded DTensors: gather the full state (collective) onto rank 0
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        full = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        model = build_decoder(cfg.decoder, tuple(config["grid"]), int(config["n_classes"]))  # type: ignore[arg-type]
        if rank == 0:
            model.load_state_dict(full)
    digest = param_digest(model) if (rank == 0 or result.strategy != "fsdp2") else "sharded"
    me = {"rank": rank, "pid": os.getpid(), "param_digest": digest}
    workers: list[Any] = [me]
    if dist.is_available() and dist.is_initialized():
        workers = [None] * world
        dist.all_gather_object(workers, me)
    if rank == 0:
        out = Path(config["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        save_decoder(
            model,
            str(out / CHECKPOINT_FILE),
            preprocessor_hash=config["preprocessor_hash"],
            dataset_hash=config["dataset_hash"],
            history=result.history,
            best_epoch=result.best_epoch,
        )
        summary = {
            "history": result.history,
            "best_epoch": result.best_epoch,
            "world_size": result.world_size,
            "strategy": result.strategy,
            "n_parameters": result.n_parameters,
            "workers": workers,
        }
        (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, default=str))


def train_decoder_distributed(
    cfg: PlatformConfig,
    *,
    arrays: dict[str, SessionArrays],
    windows: pd.DataFrame,
    grid: tuple[int, int, int],
    n_classes: int,
    out_dir: Path,
    preprocessor_hash: str,
    dataset_hash: str,
) -> dict[str, Any]:
    """Train with ``fusi.training.num_workers`` Ray Train workers; returns the rank-0 summary."""
    t0 = time.perf_counter()
    ensure_ray()
    res = select_resources(cfg, num_workers=cfg.fusi.training.num_workers)
    storage = (cfg.paths.resolved().artifacts_dir / "ray_train").resolve()
    storage.mkdir(parents=True, exist_ok=True)
    name = f"fusi_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    trainer = TorchTrainer(
        _loop_per_worker,
        train_loop_config={
            "cfg": cfg.model_dump(mode="json"),
            "arrays_ref": ray.put(arrays),
            "windows_ref": ray.put(windows),
            "grid": list(grid),
            "n_classes": n_classes,
            "seed": cfg.fusi.sim.seed,
            "out_dir": str(out_dir),
            "preprocessor_hash": preprocessor_hash,
            "dataset_hash": dataset_hash,
        },
        scaling_config=ScalingConfig(**res.scaling_config_kwargs()),
        run_config=RunConfig(name=name, storage_path=str(storage)),
    )
    trainer.fit()
    summary: dict[str, Any] = json.loads((out_dir / SUMMARY_FILE).read_text())
    digests = {w["param_digest"] for w in summary["workers"]} - {"sharded"}
    summary["params_in_sync"] = len(digests) == 1  # DDP: all replicas equal; FSDP2: rank 0 full
    summary["duration_s"] = time.perf_counter() - t0
    summary["resources"] = res.to_dict()
    summary["checkpoint"] = str(out_dir / CHECKPOINT_FILE)
    log.info(
        "fusi.train.done",
        world_size=summary["world_size"],
        strategy=summary["strategy"],
        params_in_sync=summary["params_in_sync"],
        duration_s=round(summary["duration_s"], 1),
    )
    return summary
