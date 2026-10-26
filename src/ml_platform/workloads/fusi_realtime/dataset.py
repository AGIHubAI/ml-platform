"""Decoder datasets: per-session feature arrays + labels, indexed by leakage-checked windows.

Inputs are exactly the variables allowed by ``SplitConfig.allowed_features`` (features
derived from voxel data and physiology phases); labels come from task events. The
simulator's ``truth.parquet`` is never read here (`DECODER_INPUTS` is what the
feature-schema check inspects).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# What the decoder consumes, recorded for the feature-schema check.
DECODER_INPUTS: tuple[str, ...] = ("doppler", "cardiac_phase", "respiratory_phase")


@dataclass
class SessionArrays:
    features: np.ndarray  # (T, x, y, z) float32
    labels: np.ndarray  # (T,) int64, -100 = no label


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """``(x (W, x, y, z), y (W,))`` sequences cut from session arrays by a windows table."""

    def __init__(
        self,
        sessions: dict[str, SessionArrays],
        windows: pd.DataFrame,
        *,
        noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.sessions = sessions
        self.windows = windows.reset_index(drop=True)[["session", "start", "end"]]
        # Augmentation lives here, not in the training loop: the noise of sample ``i`` in
        # epoch ``e`` is a pure function of (seed, e, i), so it is identical whichever
        # rank loads the sample (worker-count invariance). ``fit_decoder`` bumps ``epoch``.
        self.noise = float(noise)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s, a, b = self.windows.iloc[i]
        arr = self.sessions[str(s)]
        x = torch.from_numpy(np.array(arr.features[int(a) : int(b)], copy=True))  # writable
        y = torch.from_numpy(np.array(arr.labels[int(a) : int(b)], copy=True))
        if self.noise > 0:  # features are z-scored: noise is in std units
            rng = np.random.default_rng([self.seed, self.epoch, int(i)])
            x = x + self.noise * torch.from_numpy(rng.standard_normal(x.shape, dtype=np.float32))
        return x, y


def frame_accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy over labelled frames (label >= 0)."""
    m = labels >= 0
    return float((pred[m] == labels[m]).mean()) if m.any() else float("nan")


def trial_accuracy(pred: np.ndarray, labels: np.ndarray, events: pd.DataFrame) -> float:
    """One decision per trial: the prediction at the last labelled frame of that trial."""
    df = pd.DataFrame({"trial": events["trial"].to_numpy(), "y": labels, "p": pred})
    df = df[df.y >= 0]
    if df.empty:
        return float("nan")
    last = df.groupby("trial").tail(1)
    return float((last.p == last.y).mean())
