"""PyTorch trainer that works single-process *and* inside a torch.distributed group.

The same `Trainer` runs:

* in a plain Python process (no process group) — unit tests, notebooks;
* under ``torchrun`` / ``torch.multiprocessing.spawn`` with an initialized
  process group (see ``tests/integration/test_ddp_trainer.py``);
* inside Ray Train workers (``training/distributed.py``): Ray decides *where*
  the worker processes run and sets up the process group; PyTorch DDP decides
  *how* the model replicas synchronize gradients (all-reduce in backward).

Distributed semantics when ``torch.distributed.is_initialized()``:

* the model is wrapped in ``DistributedDataParallel`` (one replica per rank);
* a ``DistributedSampler`` shards the training set (``set_epoch`` each epoch
  so shuffles differ per epoch but are identical across ranks);
  ``training.batch_size`` is *per worker*, the effective global batch is
  ``batch_size * world_size``;
* validation shards rows ``rank::world_size`` (no padding) and the sufficient
  statistics (loss sum, SSE, SAE, count, ...) are ``all_reduce``-d so every
  rank reports the exact global metrics;
* only rank 0 writes checkpoints; other ranks wait at a barrier;
* logs carry ``rank``/``world_size``; per-epoch logs are emitted by rank 0.

Determinism: all RNGs are seeded, ``torch.use_deterministic_algorithms`` is
enabled (``warn_only``) and the data order is a pure function of
``(seed, epoch)`` (``DistributedSampler.set_epoch`` / a ``RandomSampler``
generator re-seeded every epoch), so it needs no checkpointed state. What
*does* carry state from epoch to epoch is each rank's global RNG stream —
the CPU (and CUDA) torch generator drives the per-rank dropout masks and the
``DataLoader`` base seed, and Python/NumPy are captured for completeness.
Every rank's RNG state is ``all_gather``-ed into the checkpoint
(``rng_states[rank]``) and each rank restores *its own* state on resume, so a
run resumed from epoch k reproduces the uninterrupted run bit-for-bit — for
the same world size (tested under torch.distributed and under Ray Train
failure/recovery). Resuming with a *different* world size cannot be exact
(the data shards change anyway): ranks without a saved state fall back to a
deterministic re-seed from ``(seed, rank, epoch)`` and a warning is logged.

This module intentionally does not import ray, dagster, or mlflow.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from ml_platform.config import PlatformConfig
from ml_platform.data.datasets import ArrayDataset
from ml_platform.data.normalization import Normalizer
from ml_platform.hashing import git_sha
from ml_platform.logging import get_logger
from ml_platform.models.losses import gaussian_nll_loss, mse_loss
from ml_platform.models.mlp import ResidualMLP, build_model, count_parameters
from ml_platform.training import checkpointing as ckpt
from ml_platform.training.config import (
    configure_determinism,
    device_info,
    get_rng_state,
    resolve_device,
    seed_everything,
    set_rng_state,
)

ModelFactory = Callable[[PlatformConfig], nn.Module]
EpochCallback = Callable[[int, dict[str, float], Path | None], None]


class SimulatedWorkerFailure(RuntimeError):
    """Raised on purpose (``fail_at_epoch``) *after* that epoch's checkpoint is written."""

    def __init__(self, epoch: int, checkpoint_path: Path | None) -> None:
        super().__init__(
            f"simulated worker failure after epoch {epoch} (checkpoint: {checkpoint_path})"
        )
        self.epoch = epoch
        self.checkpoint_path = checkpoint_path

    def __reduce__(self) -> tuple[type, tuple[int, Path | None]]:
        # picklable across process boundaries (Ray ships worker exceptions to the driver)
        return (type(self), (self.epoch, self.checkpoint_path))


@dataclass
class TrainResult:
    checkpoint_path: Path
    metrics: dict[str, float]
    epochs_completed: int
    duration_s: float
    world_size: int
    device: str
    seed: int
    dataset_hash: str | None
    git_sha: str | None = None
    hyperparams: dict[str, Any] = field(default_factory=dict)
    device_info: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    resumed_from: str | None = None
    checkpoint_hash: str | None = None
    n_parameters: int = 0
    n_train: int = 0
    n_val: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["checkpoint_path"] = str(self.checkpoint_path)
        return d


def _dist_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) — (0, 1, 0) without a process group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


class Trainer:
    def __init__(
        self,
        cfg: PlatformConfig,
        model_factory: ModelFactory | None = None,
        *,
        device: torch.device | str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.model_factory: ModelFactory = model_factory or build_model
        self.rank, self.world_size, self.local_rank = _dist_info()
        self.device = (
            torch.device(device) if device is not None else resolve_device(cfg, self.local_rank)
        )
        self.log = get_logger("trainer", run_id=run_id, rank=self.rank, world_size=self.world_size)
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.normalizer: Normalizer | None = None
        self.epoch = 0
        self.history: list[dict[str, float]] = []
        self.train_ds: ArrayDataset | None = None
        self.val_ds: ArrayDataset | None = None
        self.dataset_hash: str | None = None
        self._last_ckpt: Path | None = None

    # ------------------------------------------------------------------ properties
    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1 or (dist.is_available() and dist.is_initialized())

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def base_model(self) -> nn.Module:
        """The underlying module (unwrapped from DDP)."""
        assert self.model is not None, "call train() or setup() first"
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    # ------------------------------------------------------------------ setup
    def setup(self, train_ds: ArrayDataset, val_ds: ArrayDataset | None = None) -> None:
        """Seed, fit the normalizer on the training split, build model + optimizer."""
        tc = self.cfg.training
        seed_everything(self.cfg.seed)
        configure_determinism(tc.deterministic)

        self.train_ds, self.val_ds = train_ds, val_ds
        self.dataset_hash = train_ds.dataset_hash
        self.normalizer = Normalizer.fit(train_ds.X, train_ds.y)
        self._apply_normalizer()

        model = self.model_factory(self.cfg).to(self.device)
        if self.is_distributed:
            model = DistributedDataParallel(
                model, device_ids=[self.device.index] if self.device.type == "cuda" else None
            )
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay
        )
        # decorrelate dropout masks across ranks (init was identical on all ranks)
        torch.manual_seed(self.cfg.seed + 7919 * (self.rank + 1))
        self.epoch = 0
        self.history = []

    def _apply_normalizer(self) -> None:
        assert self.normalizer is not None
        for ds in (self.train_ds, self.val_ds):
            if ds is not None:
                ds.set_normalized(
                    self.normalizer.transform_x(ds.X), self.normalizer.transform_y(ds.y)
                )

    # ------------------------------------------------------------------ loaders
    def _train_loader(self, epoch: int) -> DataLoader:
        assert self.train_ds is not None
        tc = self.cfg.training
        sampler: DistributedSampler | RandomSampler
        if self.is_distributed:
            sampler = DistributedSampler(
                self.train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.cfg.seed,
                drop_last=False,
            )
            sampler.set_epoch(epoch)
        else:
            g = torch.Generator()
            g.manual_seed(self.cfg.seed * 1_000_003 + epoch)
            sampler = RandomSampler(self.train_ds, generator=g)
        return DataLoader(
            self.train_ds,
            batch_size=tc.batch_size,
            sampler=sampler,
            num_workers=tc.num_dataloader_workers,
            drop_last=False,
        )

    # ------------------------------------------------------------------ steps
    def _loss(self, xb: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.model is not None
        if self.cfg.training.loss == "gaussian_nll":
            # call through the (possibly DDP) wrapper so gradients are synchronized
            mu, log_var = self.model(xb, return_log_var=True)
            assert log_var is not None
            return gaussian_nll_loss(mu, log_var, yb), mu
        pred = self.model(xb)
        return mse_loss(pred, yb), pred

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        if self.is_distributed:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def _train_epoch(self, epoch: int) -> float:
        assert self.model is not None and self.optimizer is not None
        self.model.train()
        stats = torch.zeros(2, dtype=torch.float64, device=self.device)
        for xb, yb in self._train_loader(epoch):
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss, _ = self._loss(xb, yb)
            loss.backward()  # DDP all-reduces gradients here
            self.optimizer.step()
            stats[0] += loss.detach().double() * xb.shape[0]
            stats[1] += xb.shape[0]
        self._all_reduce(stats)
        return float(stats[0] / stats[1].clamp(min=1))

    @torch.no_grad()
    def validate(self, ds: ArrayDataset | None = None) -> dict[str, float]:
        """Global validation metrics on standardized targets (all-reduced across ranks)."""
        ds = ds if ds is not None else self.val_ds
        assert self.model is not None and self.normalizer is not None
        if ds is None or len(ds) == 0:
            return {}
        if ds is not self.val_ds and ds is not self.train_ds:
            ds.set_normalized(self.normalizer.transform_x(ds.X), self.normalizer.transform_y(ds.y))
        model = self.base_model
        model.eval()
        idx = torch.arange(self.rank, len(ds), self.world_size)
        X, y = ds._Xt[idx].to(self.device), ds._yt[idx].to(self.device)
        # [loss_sum, sse, sae, n, sum_y, sum_y2]
        stats = torch.zeros(6, dtype=torch.float64, device=self.device)
        bs = 4096
        for s in range(0, X.shape[0], bs):
            xb, yb = X[s : s + bs], y[s : s + bs]
            if isinstance(model, ResidualMLP) and model.predict_variance:
                mu, log_var = model.forward_with_log_var(xb)
                assert log_var is not None
                batch_loss = gaussian_nll_loss(mu, log_var, yb, reduction="sum")
            else:
                mu = model(xb)
                batch_loss = mse_loss(mu, yb, reduction="sum")
            err = (mu - yb).double()
            yd = yb.double()
            stats += torch.stack(
                [
                    batch_loss.double(),
                    (err**2).sum(),
                    err.abs().sum(),
                    torch.tensor(float(xb.shape[0]), dtype=torch.float64, device=self.device),
                    yd.sum(),
                    (yd**2).sum(),
                ]
            )
        self._all_reduce(stats)
        loss_sum, sse, sae, n, sy, sy2 = (float(v) for v in stats.tolist())
        sst = sy2 - sy * sy / n
        return {
            "val_loss": loss_sum / n,
            "val_rmse": math.sqrt(sse / n),
            "val_mae": sae / n,
            "val_r2": 1.0 - sse / sst if sst > 0 else float("nan"),
            "val_rmse_raw": math.sqrt(sse / n) * self.normalizer.y_std,
            "n_val": n,
        }

    # ------------------------------------------------------------------ checkpoints
    def _gather_rng_states(self) -> list[dict[str, Any]]:
        """Every rank's RNG state, in rank order (collective: call on all ranks)."""
        mine = get_rng_state()
        if not self.is_distributed:
            return [mine]
        states: list[Any] = [None] * self.world_size
        # pickled CPU tensors; the checkpoint is tiny compared to the model state
        dist.all_gather_object(states, mine)
        return states

    def _payload(
        self, metrics: dict[str, float], rng_states: list[dict[str, Any]]
    ) -> dict[str, Any]:
        assert self.optimizer is not None and self.normalizer is not None
        base = self.base_model
        spec = base.spec() if isinstance(base, ResidualMLP) else {"class": type(base).__name__}
        return {
            "model_state": base.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "metrics": dict(metrics),
            "history": [dict(h) for h in self.history],
            "normalizer": self.normalizer.to_dict(),
            "config": self.cfg.model_dump(mode="json"),
            # rank 0's state (single-process format) + one state per rank
            "rng_state": rng_states[0],
            "rng_states": rng_states,
            "model_spec": spec,
            "seed": self.cfg.seed,
            "dataset_hash": self.dataset_hash,
            "world_size": self.world_size,
        }

    def save_checkpoint(
        self, checkpoint_dir: str | Path, metrics: dict[str, float] | None = None
    ) -> Path:
        """Rank 0 writes ``checkpoint_dir/epoch_XXXX``; all ranks sync and get the path.

        Collective: every rank contributes its RNG state (``all_gather``).
        """
        root = Path(checkpoint_dir)
        path = root / ckpt.epoch_dir_name(self.epoch)
        rng_states = self._gather_rng_states()
        if self.is_main:
            path = ckpt.save_checkpoint(root, self._payload(metrics or {}, rng_states))
        if self.is_distributed:
            dist.barrier()
        self._last_ckpt = path
        return path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore model, optimizer, epoch, history, normalizer and RNG state."""
        assert self.model is not None and self.optimizer is not None
        payload = ckpt.load_checkpoint(path, map_location=self.device)
        if (
            self.dataset_hash
            and payload.get("dataset_hash")
            and payload["dataset_hash"] != self.dataset_hash
        ):
            raise ValueError(
                "refusing to resume: checkpoint was trained on dataset "
                f"{payload['dataset_hash'][:12]}, current dataset is {self.dataset_hash[:12]}"
            )
        self.base_model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.epoch = int(payload["epoch"])
        self.history = [dict(h) for h in payload.get("history", [])]
        self.normalizer = Normalizer.from_dict(payload["normalizer"])
        self._apply_normalizer()
        self._restore_rng(payload)
        return payload

    def _restore_rng(self, payload: dict[str, Any]) -> None:
        """Restore this rank's own RNG stream (exact resume for the same world size)."""
        states: list[dict[str, Any]] | None = payload.get("rng_states")
        if states is None and "rng_state" in payload:  # checkpoints before per-rank states
            states = [payload["rng_state"]] if int(payload.get("world_size") or 1) == 1 else None
        if states is not None and len(states) == self.world_size:
            set_rng_state(states[self.rank])
            return
        # World size changed (or an old multi-rank checkpoint with only rank 0's
        # state): a bit-exact continuation is impossible anyway because the
        # DistributedSampler shards differ. Fall back to a deterministic,
        # rank-distinct stream derived from (seed, rank, epoch).
        seed_everything(self.cfg.seed + 7919 * (self.rank + 1) + self.epoch)
        if self.is_main:
            self.log.warning(
                "training.resume_not_bit_exact",
                reason="world size changed" if states is not None else "no per-rank RNG states",
                checkpoint_world_size=payload.get("world_size"),
                world_size=self.world_size,
                epoch=self.epoch,
            )

    # ------------------------------------------------------------------ main loop
    def train(
        self,
        train_ds: ArrayDataset,
        val_ds: ArrayDataset | None,
        *,
        checkpoint_dir: str | Path,
        resume_from: str | Path | None = None,
        fail_at_epoch: int | None = None,
        on_epoch_end: EpochCallback | None = None,
    ) -> TrainResult:
        """Train for ``cfg.training.epochs`` epochs.

        ``on_epoch_end(epoch, metrics, checkpoint_path)`` is called on *every*
        rank after each epoch (``checkpoint_path`` is the directory written
        this epoch, or ``None``) — Ray Train's ``report`` must be called by
        all workers. ``fail_at_epoch`` raises `SimulatedWorkerFailure` right
        after that epoch's checkpoint is written; ``resume_from`` continues a
        run from a checkpoint directory (or a training dir with ``latest``).
        """
        tc = self.cfg.training
        t0 = time.perf_counter()
        self.setup(train_ds, val_ds)
        resumed_from = None
        if resume_from is not None:
            payload = self.load_checkpoint(resume_from)
            resumed_from = str(ckpt.resolve_checkpoint(resume_from).parent)
            self.log.info("training.resumed", from_checkpoint=resumed_from, epoch=payload["epoch"])
        sha = git_sha() if self.is_main else None
        if self.is_main:
            self.log.info(
                "training.start",
                n_train=len(train_ds),
                n_val=len(val_ds) if val_ds is not None else 0,
                epochs=tc.epochs,
                start_epoch=self.epoch + 1,
                device=str(self.device),
                dataset_hash=self.dataset_hash,
            )

        metrics: dict[str, float] = dict(self.history[-1]) if self.history else {}
        for epoch in range(self.epoch + 1, tc.epochs + 1):
            train_loss = self._train_epoch(epoch)
            self.epoch = epoch
            metrics = {"epoch": float(epoch), "train_loss": train_loss, **self.validate()}
            self.history.append(metrics)
            should_ckpt = (
                epoch % max(tc.checkpoint_every, 1) == 0
                or epoch == tc.epochs
                or epoch == fail_at_epoch
            )
            path = self.save_checkpoint(checkpoint_dir, metrics) if should_ckpt else None
            if self.is_main:
                self.log.info(
                    "training.epoch_end",
                    epoch=epoch,
                    train_loss=round(train_loss, 5),
                    val_rmse=round(metrics.get("val_rmse", float("nan")), 5),
                    checkpoint=str(path) if path else None,
                )
            if on_epoch_end is not None:
                on_epoch_end(epoch, dict(metrics), path if self.is_main else None)
            if fail_at_epoch is not None and epoch == fail_at_epoch:
                self.log.warning("training.simulated_failure", epoch=epoch, checkpoint=str(path))
                raise SimulatedWorkerFailure(epoch, path)

        if self._last_ckpt is None or ckpt.epoch_dir_name(self.epoch) != self._last_ckpt.name:
            # e.g. resuming a run that had already finished: make sure a final checkpoint exists
            self.save_checkpoint(checkpoint_dir, metrics)
        assert self._last_ckpt is not None
        duration = time.perf_counter() - t0
        final_metrics = {k: float(v) for k, v in metrics.items() if k != "epoch"}
        result = TrainResult(
            checkpoint_path=self._last_ckpt,
            metrics=final_metrics,
            epochs_completed=self.epoch,
            duration_s=duration,
            world_size=self.world_size,
            device=str(self.device),
            seed=self.cfg.seed,
            dataset_hash=self.dataset_hash,
            git_sha=sha,
            hyperparams={
                **{f"training.{k}": v for k, v in tc.model_dump(mode="json").items()},
                **{f"model.{k}": v for k, v in self.cfg.model.model_dump(mode="json").items()},
                "seed": self.cfg.seed,
            },
            device_info=device_info(self.device),
            history=[dict(h) for h in self.history],
            resumed_from=resumed_from,
            checkpoint_hash=ckpt.checkpoint_hash(self._last_ckpt) if self.is_main else None,
            n_parameters=count_parameters(self.base_model),
            n_train=len(train_ds),
            n_val=len(val_ds) if val_ds is not None else 0,
        )
        if self.is_main:
            self.log.info(
                "training.done",
                epochs=self.epoch,
                duration_s=round(duration, 2),
                val_rmse=final_metrics.get("val_rmse"),
                checkpoint=str(self._last_ckpt),
            )
        return result


def predict_array(trainer: Trainer, X: np.ndarray) -> np.ndarray:
    """Deterministic (dropout off) predictions in raw units from a trained `Trainer`."""
    assert trainer.normalizer is not None
    model = trainer.base_model
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(trainer.normalizer.transform_x(X)).to(trainer.device)
        out = model(xt).double().cpu().numpy()
    return trainer.normalizer.inverse_y(out)
