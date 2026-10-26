"""Leakage guards: a correct split passes every check, deliberately leaky ones fail."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_platform.workloads.fusi_realtime import splits as S
from ml_platform.workloads.fusi_realtime.config import SplitConfig
from ml_platform.workloads.fusi_realtime.dataset import DECODER_INPUTS

FS = 10.0
CFG = SplitConfig(window_frames=20, window_stride=10, gap_s=10.0)
SESSIONS = [S.SessionMeta(f"sub00_ses{i:02d}", "sub00", 300) for i in range(5)]


def _all(plan: S.SplitPlan, windows: pd.DataFrame, fit_blocks=None) -> dict[str, bool]:  # type: ignore[no-untyped-def]
    fit = (
        fit_blocks
        if fit_blocks is not None
        else [(b.session, b.start, b.end) for b in plan.blocks_of("train")]
    )
    checks = S.split_checks(plan, windows, fit_blocks=fit, inputs=DECODER_INPUTS, cfg=CFG)
    return {c.name: c.passed for c in checks}


def test_session_split_passes_all_checks() -> None:
    plan = S.make_split(SESSIONS, CFG, FS)
    windows = S.make_windows(plan, CFG.window_frames, CFG.window_stride)
    assert plan.sessions("test") == ["sub00_ses04"] and plan.sessions("val") == ["sub00_ses03"]
    assert all(_all(plan, windows).values())
    assert plan.split_hash() == S.make_split(list(reversed(SESSIONS)), CFG, FS).split_hash()


def test_time_block_split_has_gaps_and_passes() -> None:
    cfg = CFG.model_copy(update={"hold_out": "time_block"})
    plan = S.make_split(SESSIONS[:1], cfg, FS)
    windows = S.make_windows(plan, cfg.window_frames, cfg.window_stride)
    res = _all(plan, windows)
    assert all(res.values()), res
    gap = S.check_temporal_gap(windows, plan.gap_frames)
    assert gap.metadata["min_gap_frames"] >= 100


def test_random_frame_level_split_is_caught() -> None:
    """The classic leak: windows from one session assigned to splits at random."""
    plan = S.make_split(SESSIONS[:1], CFG.model_copy(update={"hold_out": "time_block"}), FS)
    rng = np.random.default_rng(0)
    starts = np.arange(0, 280, 5)
    leaky = pd.DataFrame(
        {
            "session": "sub00_ses00",
            "subject": "sub00",
            "split": rng.choice(["train", "val", "test"], size=len(starts)),
            "start": starts,
            "end": starts + 20,
        }
    )
    res = _all(plan, leaky)
    assert not res["temporal_gap"]
    assert not res["no_window_overlap"]


def test_session_listed_in_two_splits_is_caught() -> None:
    plan = S.make_split(SESSIONS, CFG, FS)
    bad_block = S.Block("sub00_ses04", "sub00", 0, 100, "train")  # the test session, again
    leaky = S.SplitPlan(plan.hold_out, plan.gap_frames, (*plan.blocks, bad_block))
    assert not S.check_sessions_disjoint(leaky).passed


def test_preprocessing_fitted_on_test_data_is_caught() -> None:
    plan = S.make_split(SESSIONS, CFG, FS)
    windows = S.make_windows(plan, CFG.window_frames, CFG.window_stride)
    fit_all = [(s.key, 0, s.n_frames) for s in SESSIONS]  # normalisation fitted on everything
    assert not _all(plan, windows, fit_blocks=fit_all)["preprocessing_fit_on_train_only"]


def test_latent_simulator_state_in_features_is_caught() -> None:
    ok = S.check_feature_schema(DECODER_INPUTS, CFG.allowed_features)
    assert ok.passed
    bad = S.check_feature_schema([*DECODER_INPUTS, "latent_intent"], CFG.allowed_features)
    assert not bad.passed and "latent_intent" in bad.detail
    assert not S.check_feature_schema(["doppler", "cue_direction"], CFG.allowed_features).passed


def test_subject_hold_out() -> None:
    sessions = [
        S.SessionMeta(f"sub{s:02d}_ses{i:02d}", f"sub{s:02d}", 300)
        for s in range(2)
        for i in range(3)
    ]
    cfg = CFG.model_copy(update={"hold_out": "subject"})
    plan = S.make_split(sessions, cfg, FS)
    assert {b.subject for b in plan.blocks_of("test")} == {"sub01"}
    assert S.check_subjects_disjoint(plan).passed
    leaky = S.SplitPlan(
        "subject", plan.gap_frames, (*plan.blocks, S.Block("x", "sub01", 0, 10, "train"))
    )
    assert not S.check_subjects_disjoint(leaky).passed
    with pytest.raises(ValueError):
        S.make_split(SESSIONS, cfg, FS)  # one subject only


def test_windows_never_cross_blocks() -> None:
    plan = S.make_split(SESSIONS[:1], CFG.model_copy(update={"hold_out": "time_block"}), FS)
    windows = S.make_windows(plan, 20, 7)
    for w in windows.itertuples(index=False):
        b = next(b for b in plan.blocks if b.split == w.split)
        assert b.start <= w.start and w.end <= b.end


def test_evaluation_level_guards() -> None:
    assert S.check_test_access(3, 5).passed and not S.check_test_access(6, 5).passed
    assert S.check_train_test_gap(0.6, 0.5, 0.35).passed
    assert not S.check_train_test_gap(1.0, 0.4, 0.35).passed
