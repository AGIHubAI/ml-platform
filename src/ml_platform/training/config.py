"""Training-time runtime settings: device selection, seeding, determinism."""

from __future__ import annotations

import os
import platform
import random
from typing import Any

import numpy as np
import torch

from ml_platform.config import PlatformConfig, TrainingConfig

__all__ = [
    "TrainingConfig",
    "configure_determinism",
    "device_info",
    "get_rng_state",
    "resolve_device",
    "seed_everything",
    "set_rng_state",
]


def resolve_device(cfg: PlatformConfig | TrainingConfig, local_rank: int = 0) -> torch.device:
    """``auto`` -> ``cuda:<local_rank>`` if CUDA is available, else CPU.

    MPS is never chosen automatically: it is not bit-reproducible and DDP does
    not support it; request it explicitly with ``training.device: mps``.
    """
    tc = cfg.training if isinstance(cfg, PlatformConfig) else cfg
    want = tc.device
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda but CUDA is not available")
        return torch.device("cuda", local_rank % max(torch.cuda.device_count(), 1))
    return torch.device(want)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    """Ask PyTorch for deterministic kernels where available (warn otherwise)."""
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)


def get_rng_state() -> dict[str, Any]:
    """Snapshot all RNGs in a ``torch.load(weights_only=True)``-safe structure."""
    np_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": _tuple_to_lists(random.getstate()),
        "numpy": {
            "kind": str(np_state[0]),
            "keys": torch.from_numpy(np.asarray(np_state[1], dtype=np.int64)),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(_lists_to_tuple(state["python"]))
    n = state["numpy"]
    np.random.set_state(
        (
            n["kind"],
            n["keys"].numpy().astype(np.uint32),
            n["pos"],
            n["has_gauss"],
            n["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _tuple_to_lists(x: Any) -> Any:
    if isinstance(x, tuple | list):
        return [_tuple_to_lists(v) for v in x]
    return x


def _lists_to_tuple(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(_lists_to_tuple(v) for v in x)
    return x


def device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["cuda_version"] = torch.version.cuda
    return info
