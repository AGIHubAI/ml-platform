"""Decoder training loop (pure torch; DDP- and FSDP2-aware, Ray-free).

Runs single-process or inside a ``torch.distributed`` process group, the same way
``training/trainer.py`` does for the surrogate: when a process group exists the
model is wrapped (``strategy="ddp"`` -> `DistributedDataParallel`, ``"fsdp2"`` ->
``torch.distributed.fsdp.fully_shard`` per temporal block + root), the training
windows are sharded with `DistributedSampler`, and epoch metrics are all-reduced.
Ray Train only provides the process group (``train_ray.py``).

Loss: cross-entropy on labelled frames of every window (dense supervision; frames
before the signal can be visible carry label -100 and are ignored).

Worker-count invariance: ``batch_size`` is the GLOBAL batch. `GlobalBatchSampler` yields
for step k the same global batch for any world size (rank r takes slice r), the loss is
normalised by the global labelled-frame count (so DDP's gradient averaging gives the
global mean) and augmentation noise is per sample (``dataset.py``). Final weights then
agree across 1/2/4 workers up to float reduction order. Dropout masks come from per-rank
RNG streams, so exact invariance holds with ``dropout=0``; with dropout the runs are
equal only in distribution (deriving per-sample masks would need a custom dropout op in
the model, not done).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from opentelemetry.trace import Tracer
from torch import nn
from torch.utils.data import DataLoader, Sampler

from ml_platform.workloads.fusi_realtime import telemetry
from ml_platform.workloads.fusi_realtime.config import FusiTrainingConfig
from ml_platform.workloads.fusi_realtime.dataset import WindowDataset
from ml_platform.workloads.fusi_realtime.models import FusiDecoder


@dataclass
class FitResult:
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    world_size: int = 1
    strategy: str = "single"
    n_parameters: int = 0


def _dist() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _wrap(model: FusiDecoder, strategy: str, device: torch.device) -> tuple[nn.Module, str]:
    _, world = _dist()
    if world == 1:
        return model, "single"
    if strategy == "fsdp2":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        # explicit mesh: the default would probe an accelerator (MPS on a Mac) that gloo lacks
        mesh = init_device_mesh(device.type, (world,))
        for blk in model.blocks:
            fully_shard(blk, mesh=mesh)
        fully_shard(model, mesh=mesh)
        return model, "fsdp2"
    ids = [device.index] if device.type == "cuda" else None
    return nn.parallel.DistributedDataParallel(model, device_ids=ids), "ddp"


def global_batches(n: int, global_batch: int, seed: int, epoch: int) -> list[list[int]]:
    """Global batches of one epoch: a seeded permutation cut into full ``global_batch`` chunks.

    The permutation depends only on (seed, epoch, n), never on the world size; the
    ragged tail is dropped so every step has the same size on every rank.
    """
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 1_000_003 * epoch))
    return [
        perm[k * global_batch : (k + 1) * global_batch].tolist() for k in range(n // global_batch)
    ]


class GlobalBatchSampler(Sampler[list[int]]):
    """Batch sampler: rank r gets slice r of every global batch (see `global_batches`)."""

    def __init__(self, n: int, global_batch: int, *, world: int, rank: int, seed: int) -> None:
        if global_batch % world:
            raise ValueError(
                f"fusi.training.batch_size={global_batch} is the GLOBAL batch and must be "
                f"divisible by the world size {world}"
            )
        if n < global_batch:
            raise ValueError(f"{n} training windows are fewer than the global batch {global_batch}")
        self.n, self.gb, self.world, self.rank, self.seed = n, global_batch, world, rank, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n // self.gb

    def __iter__(self) -> Iterator[list[int]]:
        per = self.gb // self.world
        for batch in global_batches(self.n, self.gb, self.seed, self.epoch):
            yield batch[self.rank * per : (self.rank + 1) * per]


def _allreduce(values: list[float], device: torch.device) -> list[float]:
    _, world = _dist()
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(t)
    return [float(v) for v in t.cpu()]


def fit_decoder(
    model: FusiDecoder,
    train_ds: WindowDataset,
    val_ds: WindowDataset | None,
    cfg: FusiTrainingConfig,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    tracer: Tracer | None = None,
) -> FitResult:
    dev = torch.device(device)
    torch.manual_seed(seed)
    rank, world = _dist()
    tracer = tracer or telemetry.configure("none")  # no-op tracer unless the caller set one
    sampler = GlobalBatchSampler(len(train_ds), cfg.batch_size, world=world, rank=rank, seed=seed)
    model.to(dev)
    net, strategy = _wrap(model, cfg.strategy, dev)
    torch.manual_seed(seed + rank)  # dropout streams differ per rank (after identical init)
    train_ds.noise, train_ds.seed = cfg.input_noise, seed
    loader = DataLoader(train_ds, batch_sampler=sampler)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    result = FitResult(
        world_size=world, strategy=strategy, n_parameters=sum(p.numel() for p in model.parameters())
    )
    # Early stopping on the validation fold (never the test fold): keep the weights of
    # the best validation epoch. Validation runs on the full val set on every rank, so
    # all ranks pick the same epoch. Under FSDP2 parameters are sharded DTensors and
    # the last epoch is kept instead (TODO: gather the full state for selection).
    best_acc, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        with telemetry.span(tracer, "fusi.train.epoch", epoch=epoch, rank=rank, world=world):
            sampler.set_epoch(epoch)
            train_ds.epoch = epoch
            net.train()
            loss_sum = correct = count = 0.0
            for step, (x, y) in enumerate(loader):
                with telemetry.span(tracer, "fusi.train.step", epoch=epoch, step=step):
                    x, y = x.to(dev), y.to(dev)
                    logits = net(x)
                    m = y >= 0
                    n = int(m.sum())
                    g_n = int(_allreduce([float(n)], dev)[0])  # labelled frames in the GLOBAL batch
                    # local sum / global count * world: DDP averages gradients over ranks, so the
                    # result is the gradient of the global mean CE, whatever the world size. A step
                    # with no labelled frame anywhere still runs backward (collectives must match)
                    # with an exact zero loss instead of CE's NaN.
                    ce_sum = (
                        F.cross_entropy(
                            logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum"
                        )
                        if n
                        else logits.sum() * 0.0
                    )
                    loss = ce_sum * world / max(g_n, 1)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    loss_sum += float(ce_sum.detach())
                    correct += float((logits.argmax(-1)[m] == y[m]).sum())
                    count += n
            sched.step()
            loss_sum, correct, count = _allreduce([loss_sum, correct, count], dev)
            metrics = {
                "epoch": float(epoch),
                "train_loss": loss_sum / max(count, 1),
                "train_frame_acc": correct / max(count, 1),
            }
            if val_ds is not None and len(val_ds):
                metrics.update(_val_metrics(net, val_ds, dev, cfg.batch_size))
            result.history.append(metrics)
            if strategy != "fsdp2" and metrics.get("val_frame_acc", -1.0) > best_acc:
                best_acc, result.best_epoch = metrics["val_frame_acc"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if on_epoch_end is not None:
                on_epoch_end(epoch, metrics)
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        result.best_epoch = cfg.epochs - 1
    return result


@torch.no_grad()
def _val_metrics(
    net: nn.Module, ds: WindowDataset, dev: torch.device, batch_size: int
) -> dict[str, float]:
    """Validation on every rank over the full set (small), so no gather is needed."""
    net.eval()
    loss_sum = correct = count = 0.0
    for x, y in DataLoader(ds, batch_size=batch_size):
        x, y = x.to(dev), y.to(dev)
        logits = net(x)
        m = y >= 0
        n = int(m.sum())
        if n == 0:
            continue
        loss_sum += float(F.cross_entropy(logits[m], y[m], reduction="sum"))
        correct += float((logits.argmax(-1)[m] == y[m]).sum())
        count += n
    return {"val_loss": loss_sum / max(count, 1), "val_frame_acc": correct / max(count, 1)}


@torch.no_grad()
def predict_session(model: FusiDecoder, features: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Class probabilities ``(T, C)`` for a whole session, causally.

    One full band-masked pass over the session equals streaming it frame by frame,
    but ``T x T`` scores get large, so long sessions run in overlapping chunks with
    ``n_layers * (attn_window - 1)`` frames of left context (the exact receptive field).
    """
    model.eval()
    t_total = features.shape[0]
    spec = model.spec()
    ctx = spec["n_layers"] * (spec["attn_window"] - 1)
    out = np.zeros((t_total, spec["n_classes"]), dtype=np.float32)
    for start in range(0, t_total, chunk):
        lo = max(0, start - ctx)
        end = min(t_total, start + chunk)
        x = torch.from_numpy(np.ascontiguousarray(features[lo:end], dtype=np.float32))[None]
        pos = torch.arange(lo, end)
        probs = torch.softmax(model(x, pos), dim=-1)[0, start - lo :]
        out[start:end] = probs.numpy()
    return out


def unwrap(model: nn.Module) -> Any:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
