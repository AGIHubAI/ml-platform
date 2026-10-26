"""fUSI preprocessing: clutter filter, stream == batch, fit on the train fold only, Dask == numpy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_platform.workloads.fusi_realtime import batch as B
from ml_platform.workloads.fusi_realtime import preprocess as PP
from ml_platform.workloads.fusi_realtime.config import PreprocessConfig, SimConfig
from ml_platform.workloads.fusi_realtime.sim import FusiSimulator

SIM = SimConfig(grid=(8, 8, 8), n_ensemble=6, plan_s=2.0, rest_s=1.0)
PRE = PreprocessConfig(chunk_t=16)


def _session(session: int, n: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = FusiSimulator(SIM, session=session).run(n)
    iq = np.stack([f.iq for f in frames])  # type: ignore[arg-type]
    card = np.array([f.cardiac_phase for f in frames])
    resp = np.array([f.respiratory_phase for f in frames])
    return iq, PP.clutter_filter(iq, PRE.clutter_rank), card, resp


def _fit(blocks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> PP.PreprocessorState:
    rows = []
    for key, power, card, resp in blocks:
        y = PP.detrended_log(power, SIM.frame_rate_hz, PRE)
        rows.append((key, 0, len(power), y, card, resp))
    return PP.fit_preprocessor(rows, grid=SIM.grid, frame_rate_hz=SIM.frame_rate_hz, cfg=PRE)


def test_svd_clutter_filter_removes_rank_one_tissue() -> None:
    sim = FusiSimulator(SIM, session=0)
    frame = sim.next_frame()
    assert frame.iq is not None
    truth = FusiSimulator(SIM, session=0)._power_doppler(
        0.0, frame.cardiac_phase, frame.respiratory_phase
    )
    unfiltered = PP.clutter_filter(frame.iq, 0).ravel()
    filtered = PP.clutter_filter(frame.iq, 1).ravel()
    assert np.median(unfiltered / truth) > 100  # clutter is ~30 dB above blood
    assert 0.7 < np.median(filtered / truth) < 1.3
    # batched (t, ens, ...) == per-volume
    iq2 = np.stack([frame.iq, frame.iq])
    assert np.allclose(PP.clutter_filter(iq2, 1)[1], PP.clutter_filter(frame.iq, 1))


def test_streaming_equals_batch_transform() -> None:
    iq, power, card, resp = _session(0)
    state = _fit([("s0", power, card, resp)])
    batch = PP.batch_transform(power, card, resp, state)
    stream = PP.StreamingPreprocessor(state)
    online = np.stack([stream.step(iq[i], card[i], resp[i]) for i in range(len(iq))])
    np.testing.assert_allclose(online, batch, atol=1e-5)
    assert batch.dtype == np.float32 and batch.shape == power.shape


def test_retroicor_removes_physiological_variance() -> None:
    # near noise-free Doppler, no task signal: what remains is physiology + drift
    cfg = SimConfig(grid=(8, 8, 8), speckle_looks=1e6, thermal_noise=0.0, signal_gain=0.0)
    frames = FusiSimulator(cfg, emit="doppler").run(300)
    power = np.stack([f.doppler for f in frames])  # type: ignore[arg-type]
    card = np.array([f.cardiac_phase for f in frames])
    resp = np.array([f.respiratory_phase for f in frames])
    y = PP.detrended_log(power, cfg.frame_rate_hz, PRE)
    state = PP.fit_preprocessor(
        [("s0", 0, 300, y, card, resp)], grid=cfg.grid, frame_rate_hz=cfg.frame_rate_hz, cfg=PRE
    )
    after = y - PP.retroicor_regressors(card, resp, PRE.retroicor_order) @ state.coef
    assert after.var() < 0.4 * y.var()  # the rest is the drift-baseline warm-up transient


def test_fit_uses_train_fold_only(tmp_path: Path) -> None:
    def stores(root: Path, test_session: int) -> dict[str, B.SessionStore]:
        out = {}
        for key, session in (("train", 0), ("test", test_session)):
            out[key] = B.write_simulated_session(
                root,
                FusiSimulator(SIM, session=session),
                key=key,
                subject="sub00",
                n_frames=40,
                chunk_t=PRE.chunk_t,
            )
        return out

    a, b = stores(tmp_path / "a", 1), stores(tmp_path / "b", 7)  # different test data
    with B.dask_threads(2):
        state = B.fit_preprocessor_dask(a, [("train", 0, 40)], PRE)
        same = B.fit_preprocessor_dask(b, [("train", 0, 40)], PRE)
        leaky = B.fit_preprocessor_dask(a, [("train", 0, 40), ("test", 0, 40)], PRE)
    # what the test fold contains cannot change the fitted state ...
    assert state.state_hash() == same.state_hash()
    # ... unless it is (wrongly) part of the fit, which the state records
    assert leaky.state_hash() != state.state_hash()
    assert state.fit_blocks == (("train", 0, 40),) and state.fit_n_frames == 40
    assert leaky.fit_sessions == ("test", "train")
    _, p0, c0, r0 = _session(0)
    state = _fit([("train", p0, c0, r0)])
    # test-fold features are not standardised to ~N(0, 1) by construction
    train_feats = PP.batch_transform(p0, c0, r0, state)
    assert abs(float(train_feats.mean())) < 0.05
    # versioned artifact round-trips and detects tampering
    d = state.save(tmp_path / "pre")
    loaded = PP.PreprocessorState.load(d)
    assert loaded.state_hash() == state.state_hash()
    arrays = dict(np.load(d / PP.STATE_ARRAYS))
    arrays["mean"] = arrays["mean"] + 1.0
    np.savez(d / PP.STATE_ARRAYS, **arrays)
    with pytest.raises(ValueError, match="hash"):
        PP.PreprocessorState.load(d)


@pytest.mark.parametrize("store_iq", [True, False])
def test_dask_zarr_pipeline_matches_numpy_reference(tmp_path: Path, store_iq: bool) -> None:
    sessions = {}
    for i in range(2):
        sim = FusiSimulator(SIM, session=i)
        sessions[f"s{i}"] = B.write_simulated_session(
            tmp_path,
            sim,
            key=f"s{i}",
            subject="sub00",
            n_frames=48,
            chunk_t=PRE.chunk_t,
            store_iq=store_iq,
        )
    assert ("iq" in sessions["s0"].variables) == store_iq
    with B.dask_threads(2):
        state = B.fit_preprocessor_dask(sessions, [("s0", 0, 40)], PRE)
        ds = B.open_raw(sessions["s0"])
        power = B.power_doppler(ds, PRE.clutter_rank).values
        card, resp = ds.cardiac_phase.values, ds.respiratory_phase.values
        y = PP.detrended_log(power, SIM.frame_rate_hz, PRE)
        ref = PP.fit_preprocessor(
            [("s0", 0, 40, y[:40], card[:40], resp[:40])],
            grid=SIM.grid,
            frame_rate_hz=SIM.frame_rate_hz,
            cfg=PRE,
        )
        np.testing.assert_allclose(state.coef, ref.coef, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(state.std, ref.std, rtol=1e-4)
        lazy = B.transform_session(sessions["s1"], state)
        ds1 = B.open_raw(sessions["s1"])
        expected = PP.batch_transform(
            B.power_doppler(ds1, PRE.clutter_rank).values,
            ds1.cardiac_phase.values,
            ds1.respiratory_phase.values,
            state,
        )
        np.testing.assert_allclose(lazy.values, expected, atol=1e-5)
        assert lazy.dims == ("t", "x", "y", "z")
        path = B.write_features(sessions["s1"], state, tmp_path / "features", PRE.chunk_t)
        np.testing.assert_allclose(B.read_features(path), expected, atol=1e-5)
    # write-once: a second write returns the existing store unchanged
    again = B.write_simulated_session(
        tmp_path,
        FusiSimulator(SIM, session=0),
        key="s0",
        subject="sub00",
        n_frames=48,
        chunk_t=PRE.chunk_t,
        store_iq=store_iq,
    )
    assert again == sessions["s0"]
    assert "latent_intent" in B.read_truth(again).columns
    assert "latent_intent" not in B.read_events(again).columns
