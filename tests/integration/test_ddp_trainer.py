"""Two-process torch.distributed (gloo) test of the Trainer — no Ray involved.

Proves the Trainer's DDP path on its own: DistributedSampler sharding,
gradient synchronization (replicas end identical), all-reduced metrics
(identical on every rank), rank-0-only checkpointing, and fail/resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ml_platform.config import PlatformConfig
from ml_platform.data import (
    ArrayDataset,
    generate_candidate_pool,
    initial_observations,
    make_oracle,
    records_to_frame,
    train_val_split,
)
from ml_platform.inference import Predictor
from ml_platform.training import SimulatedWorkerFailure, Trainer, latest_checkpoint

pytestmark = pytest.mark.integration

WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _datasets(cfg: PlatformConfig) -> tuple[ArrayDataset, ArrayDataset]:
    pool = generate_candidate_pool(cfg)
    frame = records_to_frame(initial_observations(pool, make_oracle(cfg), 301, seed=0))
    tr, va = train_val_split(frame, 0.2, seed=0, strategy="random")
    return ArrayDataset.from_frame(tr, dataset_hash="ddp"), ArrayDataset.from_frame(
        va, dataset_hash="ddp"
    )


def _param_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode() + v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _worker(rank: int, port: int, out_dir: str, fail_at: int | None) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        ML_LOG_LEVEL="WARNING",
    )
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        cfg = PlatformConfig.for_tests().with_overrides(**{"training.epochs": 4})
        out = Path(out_dir)
        ckpt_dir = out / "ckpt"
        seen_shards: list[int] = []
        record: dict[str, object] = {"rank": rank}
        trainer = Trainer(cfg)
        try:
            result = trainer.train(
                *_datasets(cfg),
                checkpoint_dir=ckpt_dir,
                fail_at_epoch=fail_at,
                on_epoch_end=lambda e, m, p: seen_shards.append(e),
            )
        except SimulatedWorkerFailure as exc:
            # "restart the worker": a fresh Trainer resumes from the shared checkpoint dir
            record["failed_at"] = exc.epoch
            trainer = Trainer(cfg)
            result = trainer.train(*_datasets(cfg), checkpoint_dir=ckpt_dir, resume_from=ckpt_dir)
        result_metrics = result.history[-1]
        record.update(
            epochs=result.epochs_completed,
            world_size=result.world_size,
            resumed_from=result.resumed_from,
            checkpoint_path=str(result.checkpoint_path),
        )
        n_local = len(list(trainer._train_loader(1).sampler))  # DistributedSampler shard
        record.update(
            metrics=result_metrics,
            params=_param_digest(trainer.base_model),
            is_ddp=isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel),
            n_local=n_local,
            callbacks=seen_shards,
        )
        (out / f"rank{rank}.json").write_text(json.dumps(record))
    finally:
        dist.destroy_process_group()


def _run(tmp_path: Path, fail_at: int | None) -> list[dict]:
    mp.start_processes(
        _worker,
        args=(_free_port(), str(tmp_path), fail_at),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    return [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(WORLD_SIZE)]


def test_ddp_two_workers(tmp_path: Path) -> None:
    r0, r1 = _run(tmp_path, fail_at=None)
    assert r0["is_ddp"] and r1["is_ddp"]
    assert r0["world_size"] == r1["world_size"] == 2
    assert r0["epochs"] == r1["epochs"] == 4
    # DistributedSampler shards 241 training rows across 2 ranks (padded to 2 x 121)
    assert r0["n_local"] == r1["n_local"] == 121
    # gradients were synchronized: replicas are bit-identical
    assert r0["params"] == r1["params"]
    # metrics were all-reduced: every rank reports the same global numbers
    assert r0["metrics"] == r1["metrics"]
    assert r0["metrics"]["n_val"] == 60
    assert r0["callbacks"] == r1["callbacks"] == [1, 2, 3, 4]
    # only rank 0 wrote checkpoints; they load into a plain single-process Predictor
    ckpt_dir = tmp_path / "ckpt"
    latest = latest_checkpoint(ckpt_dir)
    assert latest is not None and latest.name == "epoch_0004"
    assert sorted(p.name for p in ckpt_dir.iterdir() if p.is_dir()) == [
        f"epoch_{e:04d}" for e in range(1, 5)
    ]
    assert not list(ckpt_dir.glob(".tmp_*"))
    pred = Predictor.from_checkpoint(latest)
    assert pred.model_info()["trained_world_size"] == 2


def test_ddp_fail_and_resume_is_bit_exact(tmp_path: Path) -> None:
    (tmp_path / "resumed").mkdir()
    (tmp_path / "straight").mkdir()
    r0, r1 = _run(tmp_path / "resumed", fail_at=2)
    assert r0["failed_at"] == r1["failed_at"] == 2
    assert r0["epochs"] == r1["epochs"] == 4
    assert r0["resumed_from"].endswith("epoch_0002")
    assert r0["params"] == r1["params"]
    assert r0["metrics"] == r1["metrics"]
    # every rank restored its *own* RNG stream (dropout masks), so the resumed
    # run is bit-identical to an uninterrupted run with the same world size
    u0, _ = _run(tmp_path / "straight", fail_at=None)
    assert r0["params"] == u0["params"]
    assert r0["metrics"] == u0["metrics"]
    payload = torch.load(
        tmp_path / "resumed" / "ckpt" / "epoch_0002" / "model.pt", weights_only=True
    )
    assert len(payload["rng_states"]) == WORLD_SIZE
    assert not torch.equal(payload["rng_states"][0]["torch"], payload["rng_states"][1]["torch"]), (
        "ranks should have distinct dropout RNG streams"
    )
