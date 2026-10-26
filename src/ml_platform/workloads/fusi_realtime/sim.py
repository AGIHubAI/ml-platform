"""Synthetic volumetric functional-ultrasound (fUSI) source with a simulated user.

What one frame (one volume, default 32^3 voxels at 10 Hz) contains::

    iq        (n_ensemble, x, y, z) complex64   compound frames: tissue clutter + blood + noise
    doppler   (x, y, z) float32                 power Doppler (only in emit="doppler" mode)
    phases    cardiac / respiratory phase       recorded physiology (allowed decoder input)
    events    trial / phase / cue_direction     task events (training labels)
    truth     latent intent, engagement, lapse  EVALUATION ONLY, never a decoder input

Signal model (all published-basis, parameters are this simulator's choices):

* A "planning region" (ellipsoid, ``planning_fraction`` of voxels) has cosine-tuned
  voxels with smoothly varying preferred directions, standing in for the PPC planning
  activity decoded by Norman et al. 2021 / Griggs et al. 2024 (up to 8 directions).
* ``hemodynamic``: neural drive convolved with a gamma HRF peaking ``hrf_peak_s``
  (blood-volume lag 1-2 s). ``reporter``: first-order kinetics with ``reporter_tau_s``
  (an ASSUMPTION; no in-vivo constant in the sources read).
* Power Doppler = vascular map x (1 + blood-volume change) x (1 + cardiac pulsatility
  + respiratory motion) x (1 + drift). The cardiac waveform is non-sinusoidal so its
  harmonics matter for RETROICOR (Glover 2000).
* IQ ensemble: blood = random-phase samples whose power averages to the power Doppler
  with Gamma(``speckle_looks``) speckle, independent across compound frames; tissue clutter = ``clutter_db`` stronger,
  fully coherent across the ensemble apart from one global motion phase ramp, so it
  is exactly rank 1 and an SVD clutter filter (Demene 2015) removes it.

Closed loop: `FusiSimulator.feedback` takes the decoded cursor velocity. The simulated
user ends a trial early on success and adapts engagement (response amplitude) to the
running success rate, so decoder quality changes the signal it will see next.

Determinism: everything is drawn from generators seeded by ``(sim.seed, subject,
session)``; identical seeds and identical feedback give bit-identical frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ml_platform.workloads.fusi_realtime.config import SimConfig

# Columns that describe the simulator's hidden state. The feature-schema check
# (splits.check_feature_schema) refuses any decoder input containing these names.
LATENT_COLUMNS: tuple[str, ...] = ("latent_intent", "engagement", "lapsed", "neural_drive")
EVENT_COLUMNS: tuple[str, ...] = (
    "frame",
    "t_s",
    "trial",
    "phase",
    "cue_direction",
    "cardiac_phase",
    "respiratory_phase",
)

Emit = Literal["iq", "doppler"]


@dataclass(frozen=True)
class FrameTruth:
    """Hidden simulator state for one frame -- evaluation only."""

    latent_intent: int  # -1 = no intent (rest or lapse)
    engagement: float
    lapsed: bool
    neural_drive: float


@dataclass(frozen=True)
class Frame:
    index: int
    t_s: float
    trial: int
    phase: str  # "rest" | "plan"
    cue_direction: int  # instructed target, -1 during rest (a task event, legal label)
    cardiac_phase: float
    respiratory_phase: float
    iq: np.ndarray | None  # (ens, x, y, z) complex64
    doppler: np.ndarray | None  # (x, y, z) float32
    truth: FrameTruth = field(repr=False)

    def event_row(self) -> dict[str, float | int | str]:
        return {
            "frame": self.index,
            "t_s": self.t_s,
            "trial": self.trial,
            "phase": self.phase,
            "cue_direction": self.cue_direction,
            "cardiac_phase": self.cardiac_phase,
            "respiratory_phase": self.respiratory_phase,
        }

    def truth_row(self) -> dict[str, float | int | bool]:
        return {
            "frame": self.index,
            "latent_intent": self.truth.latent_intent,
            "engagement": self.truth.engagement,
            "lapsed": self.truth.lapsed,
            "neural_drive": self.truth.neural_drive,
        }


def direction_vectors(n: int) -> np.ndarray:
    """Unit vectors of the ``n`` centre-out targets, shape (n, 2)."""
    ang = 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def label_delay_s(cfg: SimConfig) -> float:
    """How long after cue onset a frame carries a usable label.

    The decoder can only see intent once the measured signal has risen: ~2/3 of the
    HRF peak for blood volume, ~1.5 tau for a fast reporter.
    """
    if cfg.signal_mode == "hemodynamic":
        return round(cfg.hrf_peak_s * 2 / 3, 3)
    return round(1.5 * cfg.reporter_tau_s, 3)


def gamma_hrf(dt: float, peak_s: float, width_s: float, length_s: float = 8.0) -> np.ndarray:
    """Gamma-shaped hemodynamic kernel sampled at ``dt``, unit area, peak at ``peak_s``."""
    t = np.arange(0.0, length_s, dt)
    shape = (peak_s / width_s) ** 2 + 1.0  # mode = (shape-1) * scale = peak_s
    scale = peak_s / (shape - 1.0)
    h = np.where(t > 0, t ** (shape - 1.0) * np.exp(-t / scale), 0.0)
    return h / h.sum()


def _smooth_field(rng: np.random.Generator, grid: tuple[int, int, int], sigma: float) -> np.ndarray:
    """Spatially smooth Gaussian random field (FFT low-pass), zero mean / unit std."""
    noise = rng.standard_normal(grid)
    freqs = np.meshgrid(*[np.fft.fftfreq(n) for n in grid], indexing="ij")
    k2 = sum(f**2 for f in freqs)
    filt = np.exp(-2 * (np.pi**2) * (sigma**2) * k2)
    out = np.real(np.fft.ifftn(np.fft.fftn(noise) * filt))
    return (out - out.mean()) / (out.std() + 1e-12)


@dataclass(frozen=True)
class SubjectAnatomy:
    """Per-subject maps (fixed across that subject's sessions)."""

    vascular: np.ndarray  # (V,) baseline power Doppler
    planning: np.ndarray  # (V,) bool
    tuning: np.ndarray  # (V, n_dirs) rectified cosine tuning x strength
    cardiac_map: np.ndarray  # (V,) pulsatility
    respiratory_map: np.ndarray  # (V,) motion sensitivity
    drift_map: np.ndarray  # (V,)
    tissue: np.ndarray  # (V,) complex tissue reflectivity

    @classmethod
    def build(cls, cfg: SimConfig, grid: tuple[int, int, int], seed: int) -> SubjectAnatomy:
        rng = np.random.default_rng(seed)
        nv = int(np.prod(grid))
        smooth = max(grid) / 8
        vascular = np.exp(0.6 * _smooth_field(rng, grid, smooth)).ravel()
        vascular = vascular / vascular.mean()
        # planning region: ellipsoid around an off-centre point, sized to the requested fraction
        coords = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in grid], indexing="ij"), -1)
        centre = np.array([0.25, -0.2, 0.1])
        dist = np.sqrt((((coords - centre) / np.array([1.0, 0.7, 0.8])) ** 2).sum(-1)).ravel()
        planning = dist <= np.quantile(dist, cfg.planning_fraction)
        theta = np.angle(
            _smooth_field(rng, grid, 1.6 * smooth) + 1j * _smooth_field(rng, grid, 1.6 * smooth)
        ).ravel()
        strength = np.clip(0.6 + 0.4 * _smooth_field(rng, grid, smooth).ravel(), 0.2, 1.0)
        dirs = 2 * np.pi * np.arange(cfg.n_directions) / cfg.n_directions
        tuning = np.maximum(np.cos(dirs[None, :] - theta[:, None]), 0.0) * strength[:, None]
        tuning[~planning] = 0.0
        cardiac_map = np.clip(vascular * (0.5 + 0.5 * rng.random(nv)), 0, None)
        respiratory_map = 0.5 + 0.5 * _smooth_field(rng, grid, smooth).ravel()
        drift_map = _smooth_field(rng, grid, smooth * 2).ravel()
        tissue = (rng.standard_normal(nv) + 1j * rng.standard_normal(nv)) / np.sqrt(2)
        return cls(
            vascular=vascular.astype(np.float64),
            planning=planning,
            tuning=tuning.astype(np.float64),
            cardiac_map=cardiac_map,
            respiratory_map=respiratory_map,
            drift_map=drift_map,
            tissue=tissue.astype(np.complex128),
        )


class SimulatedUser:
    """Attends the cue (or lapses) and adapts engagement to feedback (closes the loop)."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.engagement = 1.0

    def choose_intent(self, cue: int) -> tuple[int, bool]:
        """Latent intent for a new trial: the cue, or with ``lapse_rate`` no intent at all."""
        if self.rng.random() < self.cfg.lapse_rate:
            return -1, True
        return cue, False

    def trial_outcome(self, success: bool) -> None:
        lr = self.cfg.feedback_learning_rate
        if success:
            self.engagement += lr * (self.cfg.max_engagement - self.engagement)
        else:
            self.engagement -= lr * (self.engagement - 0.5)


class FusiSimulator:
    """Frame-by-frame fUSI source. ``next_frame()`` advances one volume (1 / frame_rate_hz)."""

    def __init__(
        self,
        cfg: SimConfig,
        *,
        subject: int = 0,
        session: int = 0,
        grid: tuple[int, int, int] | None = None,
        emit: Emit = "iq",
        closed_loop: bool = False,
    ) -> None:
        self.cfg = cfg
        self.grid: tuple[int, int, int] = tuple(grid or cfg.grid)  # type: ignore[assignment]
        self.nv = int(np.prod(self.grid))
        self.emit = emit
        self.closed_loop = closed_loop
        self.dt = 1.0 / cfg.frame_rate_hz
        self.anatomy = SubjectAnatomy.build(cfg, self.grid, seed=cfg.seed * 1000 + subject)
        self.rng = np.random.default_rng([cfg.seed, subject, session])
        self.user = SimulatedUser(cfg, np.random.default_rng([cfg.seed, subject, session, 1]))
        self.dir_vecs = direction_vectors(cfg.n_directions)
        # session-level variability
        self.gain = float(np.clip(1.0 + 0.08 * self.rng.standard_normal(), 0.8, 1.2))
        self.f_card = cfg.cardiac_hz * (1 + 0.05 * self.rng.standard_normal())
        self.f_resp = cfg.respiratory_hz * (1 + 0.05 * self.rng.standard_normal())
        self.phi_c = float(self.rng.uniform(0, 2 * np.pi))
        self.phi_r = float(self.rng.uniform(0, 2 * np.pi))
        self.drift_sign = float(self.rng.choice([-1.0, 1.0]))
        # response state: per-direction drive history (the response is low rank in direction)
        self.kernel = gamma_hrf(self.dt, cfg.hrf_peak_s, cfg.hrf_width_s)
        self.history = np.zeros((len(self.kernel), cfg.n_directions))
        self.reporter_state = np.zeros(cfg.n_directions)
        self.clutter_amp = np.sqrt(self.anatomy.vascular * 10 ** (cfg.clutter_db / 10))
        # task state
        self.index = 0
        self.trial = -1
        self.phase = "rest"
        self.phase_left = round(cfg.rest_s / self.dt)
        self.cue = -1
        self.intent = -1
        self.lapsed = False
        self.progress = 0.0  # cursor displacement towards the target in the current trial
        self.outcomes: list[bool] = []

    # ------------------------------------------------------------------ task
    def _advance_task(self) -> None:
        self.phase_left -= 1
        if self.phase_left > 0:
            return
        if self.phase == "rest":
            self._start_trial()
        else:
            self._end_trial(success=False)

    def _start_trial(self) -> None:
        self.trial += 1
        self.phase = "plan"
        self.phase_left = round(self.cfg.plan_s / self.dt)
        self.cue = int(self.rng.integers(self.cfg.n_directions))
        self.intent, self.lapsed = self.user.choose_intent(self.cue)
        self.progress = 0.0

    def _end_trial(self, *, success: bool) -> None:
        if self.closed_loop:
            self.user.trial_outcome(success)
            self.outcomes.append(success)
        self.phase = "rest"
        self.phase_left = round(self.cfg.rest_s / self.dt)
        self.cue = -1
        self.intent = -1
        self.lapsed = False

    def feedback(self, velocity: np.ndarray) -> None:
        """Decoded cursor velocity (2,) for the latest frame; success ends the trial early."""
        if not self.closed_loop or self.phase != "plan" or self.cue < 0:
            return
        self.progress += float(np.dot(velocity, self.dir_vecs[self.cue])) * self.dt
        if self.progress >= 1.0:
            self._end_trial(success=True)

    # ------------------------------------------------------------------ signal
    def _response(self) -> np.ndarray:
        """Per-direction response weights (n_dirs,) after HRF / reporter kinetics."""
        drive = np.zeros(self.cfg.n_directions)
        if self.phase == "plan" and self.intent >= 0:
            drive[self.intent] = self.user.engagement
        if self.cfg.signal_mode == "hemodynamic":
            self.history = np.roll(self.history, 1, axis=0)
            self.history[0] = drive
            weights: np.ndarray = (
                self.kernel @ self.history
            )  # unit-area kernel: steady state = drive
            return weights
        a = np.exp(-self.dt / self.cfg.reporter_tau_s)
        self.reporter_state = a * self.reporter_state + (1 - a) * drive
        return self.reporter_state.copy()

    def _power_doppler(self, t: float, cphase: float, rphase: float) -> np.ndarray:
        an, cfg = self.anatomy, self.cfg
        cbv = cfg.signal_gain * self.gain * (an.tuning @ self._response())
        # non-sinusoidal pulse (fast systolic upstroke): normalised exp(cos), zero mean
        pulse = (np.exp(np.cos(cphase)) - 1.2661) / 0.7
        physio = cfg.cardiac_amp * an.cardiac_map * pulse + cfg.respiratory_amp * (
            an.respiratory_map * np.sin(rphase)
        )
        drift = self.drift_sign * cfg.drift_per_min * (t / 60.0) * an.drift_map
        pd_true: np.ndarray = an.vascular * (1 + cbv) * (1 + physio) * (1 + drift)
        return np.clip(pd_true, 1e-6, None)

    def next_frame(self) -> Frame:
        cfg = self.cfg
        t = self.index * self.dt
        cphase = float((2 * np.pi * self.f_card * t + self.phi_c) % (2 * np.pi))
        rphase = float((2 * np.pi * self.f_resp * t + self.phi_r) % (2 * np.pi))
        pd_true = self._power_doppler(t, cphase, rphase)
        drive = self.user.engagement if (self.phase == "plan" and self.intent >= 0) else 0.0
        iq = doppler = None
        if self.emit == "iq":
            iq = self._iq(pd_true, rphase)
        else:
            looks = cfg.speckle_looks
            speckle = self.rng.gamma(looks, 1.0 / looks, size=self.nv)
            noise = cfg.thermal_noise * np.abs(self.rng.standard_normal(self.nv))
            doppler = (pd_true * speckle + noise).reshape(self.grid).astype(np.float32)
        frame = Frame(
            index=self.index,
            t_s=round(t, 6),
            trial=self.trial if self.phase == "plan" else -1,
            phase=self.phase,
            cue_direction=self.cue,
            cardiac_phase=cphase,
            respiratory_phase=rphase,
            iq=iq,
            doppler=doppler,
            truth=FrameTruth(
                latent_intent=self.intent,
                engagement=float(self.user.engagement),
                lapsed=self.lapsed,
                neural_drive=float(drive),
            ),
        )
        self.index += 1
        self._advance_task()
        return frame

    def _iq(self, pd_true: np.ndarray, rphase: float) -> np.ndarray:
        """Compound-frame ensemble: rank-1 moving tissue + white blood signal + thermal noise."""
        n_ens, nv = self.cfg.n_ensemble, self.nv
        rng = self.rng
        # Each stored compound frame stands for k = looks / n_ens decorrelated blood
        # samples (real ensembles hold ~100-300), so the power estimate over the stored
        # ensemble has Gamma(looks) speckle -- the same statistics as emit="doppler".
        k = max(self.cfg.speckle_looks / n_ens, 1.0)
        amp = np.sqrt(pd_true * rng.gamma(k, 1.0 / k, size=(n_ens, nv)))
        blood = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n_ens, nv)))
        # tissue moves with respiration: one global phase ramp across the ensemble
        ramp = np.exp(1j * 0.3 * np.cos(rphase) * np.arange(n_ens))[:, None]
        clutter = ramp * (self.clutter_amp * self.anatomy.tissue)[None, :]
        th = self.cfg.thermal_noise / np.sqrt(2)
        noise = th * (rng.standard_normal((n_ens, nv)) + 1j * rng.standard_normal((n_ens, nv)))
        iq = (blood + clutter + noise).astype(np.complex64)
        return iq.reshape((n_ens, *self.grid))

    # ------------------------------------------------------------------ batch helpers
    def run(self, n_frames: int) -> list[Frame]:
        return [self.next_frame() for _ in range(n_frames)]


def frames_to_tables(frames: list[Frame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(events, truth)`` tables. ``truth`` must never reach a decoder input."""
    events = pd.DataFrame([f.event_row() for f in frames], columns=list(EVENT_COLUMNS))
    truth = pd.DataFrame([f.truth_row() for f in frames])
    return events, truth


def frame_labels(events: pd.DataFrame, cfg: SimConfig) -> np.ndarray:
    """Per-frame training label: cue direction once the signal can be visible, else -100.

    Uses only task events (the instructed target), never the latent intent.
    """
    delay = round(label_delay_s(cfg) * cfg.frame_rate_hz)
    labels = np.full(len(events), -100, dtype=np.int64)
    plan = (events["phase"] == "plan").to_numpy()
    cue = events["cue_direction"].to_numpy()
    trial = events["trial"].to_numpy()
    since = np.zeros(len(events), dtype=np.int64)
    for i in range(1, len(events)):
        since[i] = since[i - 1] + 1 if plan[i] and trial[i] == trial[i - 1] else 0
    ok = plan & (since >= delay)
    labels[ok] = cue[ok]
    return labels
