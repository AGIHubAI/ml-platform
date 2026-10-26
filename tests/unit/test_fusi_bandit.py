"""Tests for caltech_like simulation, replay, fitted simulator and the closed-loop bandit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_platform.workloads.fusi_realtime import bandit as B
from ml_platform.workloads.fusi_realtime import caltech_like as cl
from ml_platform.workloads.fusi_realtime import realdata as rd
from ml_platform.workloads.fusi_realtime import replay as R
from ml_platform.workloads.fusi_realtime.config import BanditConfig, CaltechLikeConfig
from ml_platform.workloads.fusi_realtime.realdata import FusiRecording

CC = CaltechLikeConfig(n_lateral=32, n_depth=24, signal_gain=0.4, speckle_looks=32.0)
BC = BanditConfig()
N_TRIALS = 24


@pytest.fixture(scope="module")
def recs() -> list[FusiRecording]:
    return [cl.simulate_caltech_like_recording(CC, s, N_TRIALS) for s in range(6)]


@pytest.fixture(scope="module")
def arms(recs: list[FusiRecording]) -> dict[str, B.DecoderArm]:
    return B.fit_arms(recs[:4], BC, n_classes=8)


@pytest.fixture(scope="module")
def replay_env(recs: list[FusiRecording], arms: dict[str, B.DecoderArm]) -> B.ReplayEnv:
    return B.ReplayEnv(arms, B.build_trial_bank(recs[4:], BC), BC)


def test_mat_round_trip(recs: list[FusiRecording], tmp_path: Path) -> None:
    path = cl.write_mat(recs[0], tmp_path / "rt_fUS_data_S1_R1.mat")
    back = rd.load_caltech_session(path)
    assert back.doppler.shape == (recs[0].doppler.shape[0], 32, 1, 24)
    assert back.fs_hz == pytest.approx(2.0)
    np.testing.assert_allclose(back.doppler, recs[0].doppler, rtol=1e-6)
    assert len(back.events) == N_TRIALS
    assert set(back.events["direction"]) == set(range(8))
    np.testing.assert_allclose(back.events["onset_s"], recs[0].events["onset_s"], atol=1e-3)


def test_replay_native_and_resampled(recs: list[FusiRecording]) -> None:
    rec = recs[0]
    n_t = rec.doppler.shape[0]
    native = list(R.ReplaySource(rec))
    assert len(native) == n_t
    assert native[3].t_s == pytest.approx(1.5)
    np.testing.assert_array_equal(native[5].doppler, rec.doppler[5])
    fast = R.ReplaySource(rec, rate_hz=10.0)
    frames = list(fast)
    assert len(frames) == len(fast) == int((n_t - 1) / 2.0 * 10.0) + 1
    assert frames[7].t_s == pytest.approx(0.7)
    # t = 0.5 s lands on recorded frame 1; t = 0.6 s is 20% of the way to frame 2
    np.testing.assert_allclose(frames[5].doppler, rec.doppler[1])
    expect = 0.8 * rec.doppler[1] + 0.2 * rec.doppler[2]
    np.testing.assert_allclose(frames[6].doppler, expect, rtol=1e-5)
    onset = float(rec.events["onset_s"].iloc[0])
    labelled = [f for f in native if f.trial == 0]
    assert len(labelled) == 8
    assert labelled[0].t_s == pytest.approx(onset)
    assert labelled[0].direction == int(rec.events["direction"].iloc[0])
    assert native[0].trial == -1


def test_fitted_simulator_uses_train_only(recs: list[FusiRecording]) -> None:
    sim = R.FittedSimulator().fit(recs[:4])
    assert sim.seen == [r.name for r in recs[:4]]
    assert recs[4].name not in sim.seen
    assert sim.mean_ is not None and sim.mean_.shape[:2] == (8, 8)
    trial = sim.sample_trial(3, np.random.default_rng(0))
    assert trial.shape == (8, 32 * 24)
    again = sim.sample_trial(3, np.random.default_rng(0))
    np.testing.assert_array_equal(trial, again)


def _regret(policy: B.Policy, env: B.ReplayEnv, n: int = 120) -> float:
    return B.run_bandit(policy, env, n, seed=3).final_regret


def test_bandit_deterministic(replay_env: B.ReplayEnv) -> None:
    n_a = len(replay_env.actions)
    a = B.run_bandit(B.LinUCB(n_a, B.CONTEXT_DIM, 1.0, 5), replay_env, 60, seed=1)
    b = B.run_bandit(B.LinUCB(n_a, B.CONTEXT_DIM, 1.0, 5), replay_env, 60, seed=1)
    np.testing.assert_array_equal(a.chosen, b.chosen)
    t1 = B.run_bandit(B.LinearThompson(n_a, B.CONTEXT_DIM, 0.5, 5), replay_env, 60, seed=1)
    t2 = B.run_bandit(B.LinearThompson(n_a, B.CONTEXT_DIM, 0.5, 5), replay_env, 60, seed=1)
    np.testing.assert_array_equal(t1.chosen, t2.chosen)


def test_learners_beat_uniform(replay_env: B.ReplayEnv) -> None:
    n_a = len(replay_env.actions)
    uniform = _regret(B.UniformPolicy(0), replay_env)
    assert _regret(B.LinUCB(n_a, B.CONTEXT_DIM, 1.0, 0), replay_env) < uniform
    assert _regret(B.LinearThompson(n_a, B.CONTEXT_DIM, 0.5, 0), replay_env) < uniform


def test_disallowed_actions_never_selected(replay_env: B.ReplayEnv) -> None:
    mask = replay_env.mask
    labels = {a.label for a, m in zip(replay_env.actions, mask, strict=True) if not m}
    assert "candidate_unvalidated@1" in labels  # unvalidated decoder
    assert "lda_w3_k32@2" in labels  # gain above max_gain
    n_a = len(replay_env.actions)
    for pol in (
        B.LinUCB(n_a, B.CONTEXT_DIM, 50.0, 0),  # huge bonus tempts it to try everything
        B.LinearThompson(n_a, B.CONTEXT_DIM, 5.0, 0),
        B.UniformPolicy(0),
    ):
        res = B.run_bandit(pol, replay_env, 80, seed=0)
        assert mask[res.chosen].all()
    with pytest.raises(B.SafetyViolation):
        masked = int(np.flatnonzero(~mask)[0])
        B.run_bandit(B.FixedPolicy(masked), replay_env, 5, seed=0)


def test_fitted_sim_env_runs(recs: list[FusiRecording], arms: dict[str, B.DecoderArm]) -> None:
    sim = R.FittedSimulator().fit(recs[:4])
    env = B.FittedSimEnv(arms, sim, BC)
    res = B.run_bandit(B.LinUCB(len(env.actions), B.CONTEXT_DIM, 1.0, 0), env, 30, seed=0)
    assert env.mask[res.chosen].all()
    assert np.isfinite(res.cum_regret).all()
    _, hind = B.best_fixed_in_hindsight(res)
    assert hind[-1] >= 0.0


def test_offline_stub_and_trajectory_log(replay_env: B.ReplayEnv, tmp_path: Path) -> None:
    n_a = len(replay_env.actions)
    res = B.run_bandit(B.UniformPolicy(0), replay_env, 10, seed=0)
    assert res.trajectory[0].propensity == pytest.approx(1.0 / replay_env.mask.sum())
    path = B.write_trajectories(tmp_path / "traj.jsonl", res.trajectory)
    assert len(path.read_text().splitlines()) == 10
    assert n_a == len(res.actions)
    with pytest.raises(NotImplementedError):
        B.OfflineLearnerStub().fit(res.trajectory)
