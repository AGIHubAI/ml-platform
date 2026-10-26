"""Simulator mode ``caltech_like``: the existing simulator shaped like the public fUS-BMI data.

Sources: Griggs et al., Nature Neuroscience 2024, doi 10.1038/s41593-023-01500-7; authors'
MATLAB code https://github.com/wsgriggs2/rt_fUS_BMI; dataset
https://data.caltech.edu/records/pa710-cdn95; hemodynamic lag from Norman et al.,
Neuron 109(9):1554-1566.e4, 2021, doi 10.1016/j.neuron.2021.03.003.

No extracted Caltech ``.mat`` was available when this module was written
(``/Volumes/T7/ml-platform-data/caltech`` absent; the dataset is 34 GB and is not
downloaded here), so no value below was read from real files. Calibration table:

=========================  ==================  ============================================
quantity                   value               source
=========================  ==================  ============================================
frame rate                 2 Hz                dataset description (confirmed)
imaging geometry           2-D plane, y = 1    dataset description / ``dop`` is 3-D (confirmed)
plane size                 128 x 96 voxels    assumption (lateral x depth)
state names                trialstart, initialfixation, fixationhold, cue, memory,
                           target_acquire, target_hold, reward, iti
                                               rt_fUS_BMI ``behavior`` fields (confirmed)
state durations            fix 1.0, cue 1.0,   assumption
                           memory 2.0, acquire
                           1.0, hold 1.0,
                           reward 0.5, iti 2.0
number of directions       8 (2 variant)       paper (8 targets); 2-direction is an option
hemodynamic lag            peak 1.5 s          Norman et al. 2021 (1-2 s)
signal gain                0.2                 assumption
speckle looks              16                  assumption (noisier than simulator default)
physiology amplitudes      0.02 / 0.02         assumption (pulsatility averaged in a frame)
lapse rate                 0.05                assumption
=========================  ==================  ============================================

Mapping onto the simulator's two-phase task: ``plan`` = cue + memory + target_acquire
(4.0 s, 8 frames); ``rest`` = target_hold + reward + iti + the next trial's
initialfixation (4.5 s, 9 frames). Tuning maps, HRF shape and noise are the simulator's
(not fitted to data); only the geometry, rate and timing follow the dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_platform.logging import get_logger
from ml_platform.workloads.fusi_realtime.config import CaltechLikeConfig, SimConfig
from ml_platform.workloads.fusi_realtime.realdata import FusiRecording
from ml_platform.workloads.fusi_realtime.sim import FusiSimulator

log = get_logger(__name__)

STATE_FIELDS: tuple[str, ...] = (
    "trialstart",
    "initialfixation",
    "fixationhold",
    "cue",
    "memory",
    "target_acquire",
    "target_hold",
    "reward",
    "iti",
)
# Wall-clock offset of the first frame ("10:00:00,000"): the loader subtracts t0.
CLOCK_BASE_S = 36000.0


def plan_s(cfg: CaltechLikeConfig) -> float:
    """Simulator ``plan`` phase: cue + memory + target acquisition."""
    return cfg.cue_s + cfg.memory_s + cfg.target_acquire_s


def rest_s(cfg: CaltechLikeConfig) -> float:
    """Simulator ``rest`` phase: hold + reward + ITI + next trial's fixation."""
    return cfg.target_hold_s + cfg.reward_s + cfg.iti_s + cfg.fixation_s


def caltech_like_sim_config(cfg: CaltechLikeConfig) -> SimConfig:
    """Simulator settings for a ``(lateral, 1, depth)`` plane at the dataset's frame rate."""
    return SimConfig(
        grid=(cfg.n_lateral, 1, cfg.n_depth),
        frame_rate_hz=cfg.frame_rate_hz,
        n_directions=cfg.n_directions,
        signal_mode="hemodynamic",
        hrf_peak_s=cfg.hrf_peak_s,
        hrf_width_s=cfg.hrf_width_s,
        signal_gain=cfg.signal_gain,
        speckle_looks=cfg.speckle_looks,
        cardiac_amp=cfg.cardiac_amp,
        respiratory_amp=cfg.respiratory_amp,
        plan_s=plan_s(cfg),
        rest_s=rest_s(cfg),
        lapse_rate=cfg.lapse_rate,
        seed=cfg.seed,
    )


def simulate_caltech_like_recording(
    cfg: CaltechLikeConfig, session: int, n_trials: int
) -> FusiRecording:
    """Power-Doppler movie ``(t, x, 1, z)`` with ``n_trials`` cue-aligned trials.

    ``events.onset_s`` is the cue time (the loader's first-choice onset state) and
    ``direction`` the instructed target. ``meta['trial_structure']`` keeps the durations so
    `write_mat` can emit every state time.
    """
    sim_cfg = caltech_like_sim_config(cfg)
    sim = FusiSimulator(sim_cfg, subject=0, session=session, emit="doppler")
    dt = 1.0 / cfg.frame_rate_hz
    n_plan, n_rest = round(plan_s(cfg) / dt), round(rest_s(cfg) / dt)
    frames = sim.run(n_rest + n_trials * (n_plan + n_rest))
    movie = np.stack([f.doppler for f in frames if f.doppler is not None])  # (t, x, 1, z)
    rows = []
    seen: set[int] = set()
    for f in frames:
        if f.phase == "plan" and f.trial not in seen:
            seen.add(f.trial)
            rows.append({"trial": f.trial, "onset_s": f.t_s, "direction": f.cue_direction})
    events = pd.DataFrame(rows, columns=["trial", "onset_s", "direction"])
    log.info("fusi.caltech_like.simulated", session=session, shape=list(movie.shape))
    return FusiRecording(
        name=f"caltech_like_S{session}",
        doppler=movie.astype(np.float32),
        fs_hz=cfg.frame_rate_hz,
        events=events,
        meta={"trial_structure": cfg.model_dump(), "synthetic": True, "session": session},
    )


def _fmt_time(seconds: float) -> str:
    """Seconds -> ``'HH:MM:SS,mmm'`` (the rt_fUS_BMI behaviour time format)."""
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def write_mat(recording: FusiRecording, path: Path, cfg: CaltechLikeConfig | None = None) -> Path:
    """Write ``recording`` with the rt_fUS_BMI variable names.

    ``dop`` is (depth, lateral, time); ``timestamps`` is n x 3 with column 1 in seconds;
    ``behavior`` is a struct array (one element per trial) with the state fields as
    ``'HH:MM:SS,mmm'`` strings plus ``direction``. The direction field name is an
    assumption shared with `realdata.DIRECTION_KEYS`.
    """
    from scipy.io import savemat

    struct = recording.meta.get("trial_structure")
    c = cfg or (CaltechLikeConfig(**struct) if struct else CaltechLikeConfig())
    n_t = recording.doppler.shape[0]
    dop = np.transpose(recording.doppler[:, :, 0, :], (2, 1, 0)).astype(np.float32)
    secs = CLOCK_BASE_S + np.arange(n_t) / recording.fs_hz
    timestamps = np.stack([secs, np.zeros(n_t), np.zeros(n_t)], axis=1)
    dtype = [(name, "O") for name in STATE_FIELDS] + [("direction", "O")]
    beh = np.empty((len(recording.events),), dtype=dtype)
    ev = recording.events
    for i, (onset, direction) in enumerate(
        zip(ev["onset_s"].tolist(), ev["direction"].tolist(), strict=True)
    ):
        cue = CLOCK_BASE_S + float(onset)
        memory = cue + c.cue_s
        acquire = memory + c.memory_s
        hold = acquire + c.target_acquire_s
        reward = hold + c.target_hold_s
        iti = reward + c.reward_s
        start = cue - c.fixation_s
        times = {
            "trialstart": start,
            "initialfixation": start,
            "fixationhold": start + c.fixation_s / 2,
            "cue": cue,
            "memory": memory,
            "target_acquire": acquire,
            "target_hold": hold,
            "reward": reward,
            "iti": iti,
        }
        beh[i] = (*(_fmt_time(times[n]) for n in STATE_FIELDS), int(direction))
    path = Path(path)
    savemat(path, {"dop": dop, "timestamps": timestamps, "behavior": beh})
    log.info("fusi.caltech_like.write_mat", path=str(path), shape=list(dop.shape))
    return path
