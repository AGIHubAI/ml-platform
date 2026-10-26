"""Training: config/runtime helpers, DDP-aware Trainer, checkpoint I/O.

``training.distributed`` (Ray Train wrapper) is intentionally not imported here
so that importing the scientific training code never pulls in Ray.
"""

from ml_platform.training.checkpointing import (
    checkpoint_hash,
    latest_checkpoint,
    load_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)
from ml_platform.training.trainer import (
    SimulatedWorkerFailure,
    Trainer,
    TrainResult,
    predict_array,
)

__all__ = [
    "SimulatedWorkerFailure",
    "TrainResult",
    "Trainer",
    "checkpoint_hash",
    "latest_checkpoint",
    "load_checkpoint",
    "predict_array",
    "resolve_checkpoint",
    "save_checkpoint",
]
