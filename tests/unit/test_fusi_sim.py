"""fUSI simulator: determinism, shapes, signal modes, closed loop, label hygiene."""

from __future__ import annotations

import numpy as np

from ml_platform.workloads.fusi_realtime.config import SimConfig
from ml_platform.workloads.fusi_realtime.sim import (
    EVENT_COLUMNS,
    LATENT_COLUMNS,
    FusiSimulator,
    frame_labels,
    frames_to_tables,
    label_delay_s,
)

CFG = SimConfig(grid=(8, 8, 8), n_ensemble=4, plan_s=2.0, rest_s=1.0)


def test_same_seed_same_frames_bit_identical() -> None:
    a = FusiSimulator(CFG, session=3).run(25)
    b = FusiSimulator(CFG, session=3).run(25)
    for fa, fb in zip(a, b, strict=True):
        assert fa.iq is not None and fb.iq is not None
        assert np.array_equal(fa.iq, fb.iq)
        assert fa.event_row() == fb.event_row()
    c = FusiSimulator(CFG, session=4).run(25)
    assert not np.array_equal(a[-1].iq, c[-1].iq)  # type: ignore[arg-type]


def test_shapes_and_dtypes() -> None:
    f = FusiSimulator(CFG).next_frame()
    assert f.iq is not None and f.iq.shape == (4, 8, 8, 8) and f.iq.dtype == np.complex64
    d = FusiSimulator(CFG, emit="doppler", grid=(4, 6, 8)).next_frame()
    assert d.doppler is not None and d.doppler.shape == (4, 6, 8) and d.doppler.dtype == np.float32
    assert 0 <= f.cardiac_phase < 2 * np.pi and 0 <= f.respiratory_phase < 2 * np.pi


def test_event_table_has_no_latent_state() -> None:
    events, truth = frames_to_tables(FusiSimulator(CFG).run(60))
    assert tuple(events.columns) == EVENT_COLUMNS
    assert not set(LATENT_COLUMNS) & set(events.columns)
    assert set(LATENT_COLUMNS) <= set(truth.columns)
    labels = frame_labels(events, CFG)
    # labels only during planning, from the cue, after the signal delay
    plan = (events.phase == "plan").to_numpy()
    assert (labels[~plan] == -100).all()
    assert set(labels[labels >= 0]) <= set(events.cue_direction[plan])


def _response_peak_s(mode: str) -> float:
    cfg = SimConfig(
        grid=(8, 8, 8), signal_mode=mode, lapse_rate=0.0, plan_s=3.0, rest_s=1.0, seed=1
    )
    sim = FusiSimulator(cfg, emit="doppler")
    resp = []
    for _ in range(int(4.0 * cfg.frame_rate_hz)):
        sim.next_frame()
        resp.append(_drive_response(sim))
    onset = int(cfg.rest_s * cfg.frame_rate_hz)
    rise = np.asarray(resp[onset:])
    return float(np.argmax(rise >= 0.5 * rise.max()) / cfg.frame_rate_hz)


def _drive_response(sim: FusiSimulator) -> float:
    if sim.cfg.signal_mode == "hemodynamic":
        return float((sim.kernel @ sim.history).max())
    return float(sim.reporter_state.max())


def test_reporter_mode_is_faster_than_hemodynamic() -> None:
    t_hemo = _response_peak_s("hemodynamic")
    t_rep = _response_peak_s("reporter")
    assert t_rep < 0.5 < 0.8 < t_hemo
    assert label_delay_s(SimConfig(signal_mode="reporter")) < label_delay_s(SimConfig())


def test_closed_loop_feedback_ends_trials_and_changes_engagement() -> None:
    cfg = SimConfig(grid=(8, 8, 8), n_ensemble=4, lapse_rate=0.0, plan_s=3.0, rest_s=1.0)
    good = FusiSimulator(cfg, closed_loop=True, emit="doppler")
    bad = FusiSimulator(cfg, closed_loop=True, emit="doppler")
    from ml_platform.workloads.fusi_realtime.sim import direction_vectors

    dirs = direction_vectors(cfg.n_directions)
    for _ in range(200):
        fg, fb = good.next_frame(), bad.next_frame()
        if fg.cue_direction >= 0:
            good.feedback(3.0 * dirs[fg.cue_direction])
        if fb.cue_direction >= 0:
            bad.feedback(-3.0 * dirs[fb.cue_direction])
    assert good.outcomes and all(good.outcomes)
    assert bad.outcomes and not any(bad.outcomes)
    assert good.user.engagement > 1.0 > bad.user.engagement
    assert len(good.outcomes) > len(bad.outcomes)  # successful trials end early
