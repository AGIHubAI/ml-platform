"""Checkpoint format and I/O.

A checkpoint is a directory containing ``model.pt`` (a torch-serialized dict
loadable with ``weights_only=True``)::

    {format_version, model_state, optimizer_state, epoch, metrics, history,
     normalizer, config, rng_state, rng_states, model_spec, seed, dataset_hash,
     world_size}

plus a human-readable ``meta.json``. ``rng_state`` is rank 0's RNG snapshot;
``rng_states`` holds one snapshot per rank (index = rank) so a distributed
resume restores every rank's own stream. Under a training ``checkpoint_dir`` the
layout is ``epoch_0003/`` per saved epoch and a ``latest`` text file naming
the newest complete checkpoint. Writes are atomic (temp dir + rename) so a
crash mid-write never leaves a half checkpoint behind.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch

from ml_platform.hashing import hash_file

FORMAT_VERSION = 1
MODEL_FILE = "model.pt"
META_FILE = "meta.json"
LATEST_FILE = "latest"


def epoch_dir_name(epoch: int) -> str:
    return f"epoch_{epoch:04d}"


def save_checkpoint(checkpoint_dir: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    """Atomically write ``payload`` to ``checkpoint_dir/epoch_XXXX`` and update ``latest``."""
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root / epoch_dir_name(int(payload["epoch"]))
    tmp = root / f".tmp_{final.name}_{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        torch.save({"format_version": FORMAT_VERSION, **payload}, tmp / MODEL_FILE)
        meta = {
            k: payload.get(k)
            for k in ("epoch", "metrics", "seed", "dataset_hash", "world_size", "model_spec")
        }
        (tmp / META_FILE).write_text(json.dumps(meta, indent=2, default=str))
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    latest_tmp = root / f".{LATEST_FILE}.{uuid.uuid4().hex[:8]}"
    latest_tmp.write_text(final.name)
    latest_tmp.replace(root / LATEST_FILE)
    return final


def resolve_checkpoint(path: str | os.PathLike[str]) -> Path:
    """Accept a ``model.pt`` file, a checkpoint dir, or a training dir with ``latest``."""
    p = Path(path)
    if p.is_file():
        return p
    if (p / MODEL_FILE).exists():
        return p / MODEL_FILE
    if (p / LATEST_FILE).exists():
        return p / (p / LATEST_FILE).read_text().strip() / MODEL_FILE
    raise FileNotFoundError(f"no checkpoint found at {p}")


def latest_checkpoint(checkpoint_dir: str | os.PathLike[str]) -> Path | None:
    """Directory of the newest complete checkpoint under ``checkpoint_dir`` (or None)."""
    try:
        return resolve_checkpoint(checkpoint_dir).parent
    except FileNotFoundError:
        return None


def load_checkpoint(
    path: str | os.PathLike[str], map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(
        resolve_checkpoint(path), map_location=map_location, weights_only=True
    )
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {payload.get('format_version')}")
    return payload


def checkpoint_hash(path: str | os.PathLike[str]) -> str:
    """SHA-256 of the ``model.pt`` file (the model-artifact hash)."""
    return hash_file(resolve_checkpoint(path))
