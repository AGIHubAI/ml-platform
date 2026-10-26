"""Replay of recorded fUS runs and a simulator fitted to recordings.

`ReplaySource` feeds a `FusiRecording` (real Caltech or ``caltech_like``) frame by frame at
its native rate or resampled; `FittedSimulator` learns per-direction mean responses and
residual noise from TRAIN recordings and generates new trials. Both are pure numpy.

Preprocessing shared by the bandit (`movie_log_features`): log power Doppler minus a causal
EMA baseline (`preprocess.detrended_log`), the same unfitted steps as the decoders use.

Feedback effects in `FittedSimulator` are ASSUMPTIONS, not estimates: recordings made
under one fixed feedback policy cannot identify how feedback changes the brain signal.
`sample_trial(..., amplitude=a)` scales the evoked (mean) part by ``a`` and leaves the
residual noise unchanged; callers choose how a feedback gain maps to ``a``
(see `bandit.gain_amplitude`). Residual noise is a per-voxel AR(1) process with one shared
coefficient, which understates spatial correlation of real noise.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from ml_platform.logging import get_logger
from ml_platform.workloads.fusi_realtime.config import PreprocessConfig
from ml_platform.workloads.fusi_realtime.preprocess import detrended_log
from ml_platform.workloads.fusi_realtime.realdata import FusiRecording

log = get_logger(__name__)

DEFAULT_TRIAL_WINDOW_S = 4.0  # cue -> end of target acquire in the caltech_like structure


@dataclass(frozen=True)
class TrialWindow:
    """Frames ``[start, stop)`` of one trial, aligned to its onset (cue)."""

    trial: int
    direction: int
    start: int
    stop: int


def trial_windows(
    rec: FusiRecording, window_s: float = DEFAULT_TRIAL_WINDOW_S
) -> list[TrialWindow]:
    """Complete, labelled trial windows of ``rec`` (truncated or unlabelled ones dropped)."""
    n_t = rec.doppler.shape[0]
    width = max(round(window_s * rec.fs_hz), 1)
    out: list[TrialWindow] = []
    if "direction" not in rec.events:
        return out
    ev = rec.events
    for trial, onset, direction in zip(
        ev["trial"].tolist(), ev["onset_s"].tolist(), ev["direction"].tolist(), strict=True
    ):
        if not np.isfinite(onset) or direction is None or isinstance(direction, list):
            continue
        start = round(onset * rec.fs_hz)
        if start < 0 or start + width > n_t:
            continue
        out.append(TrialWindow(int(trial), int(direction), start, start + width))
    return out


def movie_log_features(rec: FusiRecording, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """``(T, V)`` drift-free log power Doppler of a whole recording (causal)."""
    return detrended_log(rec.doppler, rec.fs_hz, cfg or PreprocessConfig())


@dataclass(frozen=True)
class ReplayFrame:
    index: int
    t_s: float
    doppler: np.ndarray  # (x, y, z) float32 power Doppler
    trial: int  # -1 outside any trial window
    direction: int  # -1 outside any trial window


class ReplaySource:
    """Iterate a recording's frames at native rate or resampled to ``rate_hz``.

    Resampling is linear interpolation in time between the two neighbouring recorded
    frames (no extrapolation: the last output time is <= the last recorded time). It adds
    no information; a 10 Hz stream of a 2 Hz recording is a timing stress test only.
    Labels come from the event table: a frame belongs to the trial whose window
    ``[onset, onset + window_s)`` contains its time.
    """

    def __init__(
        self,
        recording: FusiRecording,
        rate_hz: float | None = None,
        window_s: float = DEFAULT_TRIAL_WINDOW_S,
    ) -> None:
        self.recording = recording
        self.rate_hz = float(rate_hz) if rate_hz else recording.fs_hz
        self.window_s = window_s
        n_t = recording.doppler.shape[0]
        duration = (n_t - 1) / recording.fs_hz
        self.n_frames = int(np.floor(duration * self.rate_hz + 1e-9)) + 1
        ev = recording.events
        ok = np.isfinite(ev["onset_s"].to_numpy(dtype=float)) if len(ev) else np.array([], bool)
        self._onsets = ev["onset_s"].to_numpy(dtype=float)[ok] if len(ev) else np.array([])
        self._dirs = ev["direction"].to_numpy()[ok] if len(ev) and "direction" in ev else None
        self._trials = ev["trial"].to_numpy()[ok] if len(ev) else np.array([])

    def __len__(self) -> int:
        return self.n_frames

    def _label(self, t: float) -> tuple[int, int]:
        if self._dirs is None or not len(self._onsets):
            return -1, -1
        inside = np.flatnonzero((self._onsets <= t + 1e-9) & (t < self._onsets + self.window_s))
        if not len(inside):
            return -1, -1
        i = int(inside[-1])
        return int(self._trials[i]), int(self._dirs[i])

    def _volume(self, t: float) -> np.ndarray:
        x = t * self.recording.fs_hz
        lo = min(int(np.floor(x + 1e-9)), self.recording.doppler.shape[0] - 1)
        hi = min(lo + 1, self.recording.doppler.shape[0] - 1)
        w = float(np.clip(x - lo, 0.0, 1.0))
        d = self.recording.doppler
        if w < 1e-9 or hi == lo:
            return d[lo]
        return ((1.0 - w) * d[lo] + w * d[hi]).astype(np.float32)

    def __iter__(self) -> Iterator[ReplayFrame]:
        for i in range(self.n_frames):
            t = i / self.rate_hz
            trial, direction = self._label(t)
            yield ReplayFrame(i, round(t, 6), self._volume(t), trial, direction)


def paced(frames: Iterable[ReplayFrame], rate_hz: float) -> Iterator[ReplayFrame]:
    """Yield ``frames`` no faster than ``rate_hz`` (wall-clock sleeps; optional helper)."""
    t0 = time.perf_counter()
    for k, frame in enumerate(frames):
        wait = t0 + k / rate_hz - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        yield frame


@dataclass
class FittedSimulator:
    """Per-direction mean response and residual noise fitted on TRAIN recordings only.

    Works in the `movie_log_features` domain, window-aligned to trial onset:
    ``mean_[d, w, v]`` and ``noise_std_[v]``. ``seen`` lists the fitted recording names so
    callers can assert that no validation/test run was used.
    """

    window_s: float = DEFAULT_TRIAL_WINDOW_S
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    directions_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    noise_std_: np.ndarray | None = None
    ar1_: float = 0.0
    seen: list[str] = field(default_factory=list)

    def fit(self, recordings_train: list[FusiRecording]) -> FittedSimulator:
        blocks: dict[int, list[np.ndarray]] = {}
        for rec in recordings_train:
            feats = movie_log_features(rec, self.preprocess)
            for w in trial_windows(rec, self.window_s):
                blocks.setdefault(w.direction, []).append(feats[w.start : w.stop])
            self.seen.append(rec.name)
        if not blocks:
            raise ValueError("no labelled trial windows in the training recordings")
        dirs = np.array(sorted(blocks))
        means = np.stack([np.mean(blocks[d], axis=0) for d in dirs])  # (D, W, V)
        resid = np.concatenate([np.stack(blocks[d]) - means[i] for i, d in enumerate(dirs)])
        self.directions_ = dirs
        self.mean_ = means
        self.noise_std_ = resid.std(axis=(0, 1)) + 1e-9  # (V,)
        z = resid / self.noise_std_
        num = float((z[:, 1:] * z[:, :-1]).mean())
        self.ar1_ = float(np.clip(num / max(float((z**2).mean()), 1e-12), 0.0, 0.95))
        log.info("fusi.replay.fitted", n_dirs=len(dirs), recordings=self.seen)
        return self

    @property
    def window_frames(self) -> int:
        assert self.mean_ is not None
        return int(self.mean_.shape[1])

    def sample_noise(self, rng: np.random.Generator) -> np.ndarray:
        """Standardised AR(1) noise ``(W, V)`` (unit marginal variance)."""
        assert self.mean_ is not None
        w, v = self.mean_.shape[1:]
        e = rng.standard_normal((w, v))
        out = np.empty_like(e)
        out[0] = e[0]
        s = np.sqrt(1.0 - self.ar1_**2)
        for k in range(1, w):
            out[k] = self.ar1_ * out[k - 1] + s * e[k]
        return out

    def compose(self, direction: int, amplitude: float, noise: np.ndarray) -> np.ndarray:
        """``amplitude * mean[direction] + noise_std * noise``; lets callers share noise."""
        assert self.mean_ is not None and self.noise_std_ is not None
        assert self.directions_ is not None
        idx = int(np.flatnonzero(self.directions_ == direction)[0])
        out: np.ndarray = amplitude * self.mean_[idx] + self.noise_std_ * noise
        return out

    def sample_trial(
        self, direction: int, rng: np.random.Generator, amplitude: float = 1.0
    ) -> np.ndarray:
        """One generated trial ``(W, V)`` for ``direction`` (see module note on feedback)."""
        return self.compose(direction, amplitude, self.sample_noise(rng))
