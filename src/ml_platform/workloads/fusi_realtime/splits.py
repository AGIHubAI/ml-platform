"""Leakage-safe train/val/test splits for neural time series, and the checks that guard them.

Why this module exists: neighbouring fUSI frames are nearly identical and blood volume
stays correlated for seconds, so a random frame-level split puts near-duplicates on
both sides and inflates accuracy (Kapoor & Narayanan 2023, leakage taxonomy; Roberts
et al. 2017, block cross-validation). The split unit is therefore a *block* of frames:

* ``hold_out="session"`` -- whole sessions go to val/test (default);
* ``hold_out="subject"`` -- whole subjects go to test (headline cross-subject score);
* ``hold_out="time_block"`` -- one session cut into train | gap | val | gap | test
  blocks, for recordings with too few sessions; the gaps are ``gap_s`` long.

Every guarantee is a pure ``check_*`` function returning a `CheckResult`; Dagster
wraps them as blocking asset checks, so a failing check stops promotion. The checks
take the split *as data* (blocks + windows), so a deliberately leaky split built by
hand fails them (``tests/unit/test_fusi_splits.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ml_platform.hashing import hash_config, hash_dataframe
from ml_platform.workloads.fusi_realtime.config import SplitConfig
from ml_platform.workloads.fusi_realtime.sim import LATENT_COLUMNS

SplitName = Literal["train", "val", "test"]
SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class SessionMeta:
    key: str  # e.g. "s00_sess03"
    subject: str
    n_frames: int


@dataclass(frozen=True)
class Block:
    """Contiguous frames ``[start, end)`` of one session assigned to one split."""

    session: str
    subject: str
    start: int
    end: int
    split: SplitName


@dataclass(frozen=True)
class SplitPlan:
    hold_out: str
    gap_frames: int
    blocks: tuple[Block, ...]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([b.__dict__ for b in self.blocks])

    def split_hash(self) -> str:
        return hash_dataframe(self.frame())

    def sessions(self, split: str) -> list[str]:
        return sorted({b.session for b in self.blocks if b.split == split})

    def blocks_of(self, split: str) -> list[Block]:
        return [b for b in self.blocks if b.split == split]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- planning
def make_split(
    sessions: Sequence[SessionMeta], cfg: SplitConfig, frame_rate_hz: float
) -> SplitPlan:
    """Deterministic split. Held-out units are the *last* ones (sorted), like a prospective test."""
    gap = int(np.ceil(cfg.gap_s * frame_rate_hz))
    ss = sorted(sessions, key=lambda s: s.key)
    blocks: list[Block] = []
    if cfg.hold_out == "session":
        n_test, n_val = cfg.n_test_sessions, cfg.n_val_sessions
        if len(ss) < n_test + n_val + 1:
            raise ValueError(f"need >= {n_test + n_val + 1} sessions for a session split")
        for i, s in enumerate(ss):
            split: SplitName = (
                "test"
                if i >= len(ss) - n_test
                else "val"
                if i >= len(ss) - n_test - n_val
                else "train"
            )
            blocks.append(Block(s.key, s.subject, 0, s.n_frames, split))
    elif cfg.hold_out == "subject":
        subjects = sorted({s.subject for s in ss})
        if len(subjects) < 2:
            raise ValueError("hold_out='subject' needs at least two subjects")
        test_subject = subjects[-1]
        rest = [s for s in ss if s.subject != test_subject]
        for s in ss:
            if s.subject == test_subject:
                split = "test"
            else:
                split = "val" if rest and s.key == rest[-1].key and len(rest) > 1 else "train"
            blocks.append(Block(s.key, s.subject, 0, s.n_frames, split))
    else:  # time_block: train 60% | gap | val 20% | gap | test 20% of each session
        for s in ss:
            usable = s.n_frames - 2 * gap
            if usable < 10:
                raise ValueError(f"session {s.key} too short for two {cfg.gap_s}s gaps")
            a = int(usable * 0.6)
            b = a + gap + int(usable * 0.2)
            blocks += [
                Block(s.key, s.subject, 0, a, "train"),
                Block(s.key, s.subject, a + gap, b, "val"),
                Block(s.key, s.subject, b + gap, s.n_frames, "test"),
            ]
    return SplitPlan(hold_out=cfg.hold_out, gap_frames=gap, blocks=tuple(blocks))


def make_windows(plan: SplitPlan, window: int, stride: int) -> pd.DataFrame:
    """Sequence windows ``[start, end)`` that lie entirely inside one block.

    Columns ``session, subject, split, start, end``. Windows never cross a block edge,
    so no window mixes frames of two splits.
    """
    rows = []
    for b in plan.blocks:
        last = b.end - window
        for start in range(b.start, last + 1, stride):
            rows.append((b.session, b.subject, b.split, start, start + window))
        # keep the block tail when the stride skips it
        if last >= b.start and (last - b.start) % stride:
            rows.append((b.session, b.subject, b.split, last, b.end))
    return pd.DataFrame(rows, columns=["session", "subject", "split", "start", "end"])


def windows_hash(windows: pd.DataFrame) -> str:
    return hash_dataframe(windows.reset_index(drop=True))


# --------------------------------------------------------------------------- checks
def check_sessions_disjoint(plan: SplitPlan) -> CheckResult:
    """No session contributes to more than one split (unless the plan splits in time)."""
    owners = plan.frame().groupby("session")["split"].nunique()
    shared = sorted(owners[owners > 1].index)
    if plan.hold_out == "time_block":
        return CheckResult(
            "sessions_disjoint",
            True,
            f"time_block split: {len(shared)} sessions shared by design; guarded by temporal_gap",
            {"shared_sessions": shared},
        )
    return CheckResult(
        "sessions_disjoint",
        not shared,
        "ok" if not shared else f"sessions in several splits: {shared}",
        {"shared_sessions": shared},
    )


def check_subjects_disjoint(plan: SplitPlan) -> CheckResult:
    """Required only for ``hold_out='subject'``: no test subject appears in train/val."""
    df = plan.frame()
    test = set(df.loc[df.split == "test", "subject"])
    seen = set(df.loc[df.split != "test", "subject"])
    leaked = sorted(test & seen)
    required = plan.hold_out == "subject"
    passed = (not leaked) or not required
    detail = (
        "ok"
        if not leaked
        else f"test subjects also in train/val: {leaked}" + ("" if required else " (not required)")
    )
    return CheckResult(
        "subjects_disjoint", passed, detail, {"required": required, "leaked": leaked}
    )


def _intervals(windows: pd.DataFrame, split: str, session: str) -> np.ndarray:
    w = windows[(windows.split == split) & (windows.session == session)]
    return w[["start", "end"]].to_numpy()


def check_temporal_gap(windows: pd.DataFrame, gap_frames: int) -> CheckResult:
    """Within a session, every train window is >= ``gap_frames`` away from any val/test window."""
    worst: int | None = None
    where = ""
    for session in windows.session.unique():
        tr = _intervals(windows, "train", session)
        if not len(tr):
            continue
        for other in ("val", "test"):
            ot = _intervals(windows, other, session)
            if not len(ot):
                continue
            # gap between [a0,a1) and [b0,b1) = max(b0 - a1, a0 - b1); negative = overlap
            g = np.maximum(ot[None, :, 0] - tr[:, None, 1], tr[:, None, 0] - ot[None, :, 1])
            m = int(g.min())
            if worst is None or m < worst:
                worst, where = m, f"{session} train vs {other}"
    passed = worst is None or worst >= gap_frames
    detail = (
        "no session shared between train and val/test"
        if worst is None
        else f"min gap {worst} frames ({where}); required {gap_frames}"
    )
    return CheckResult(
        "temporal_gap", passed, detail, {"min_gap_frames": worst, "required": gap_frames}
    )


def check_no_window_overlap(windows: pd.DataFrame, plan: SplitPlan) -> CheckResult:
    """No frame belongs to windows of two splits, and every window sits inside one block."""
    problems: list[str] = []
    for session in windows.session.unique():
        owner: dict[int, str] = {}
        for w in windows[windows.session == session].itertuples(index=False):
            for f in range(int(w.start), int(w.end)):
                prev = owner.setdefault(f, str(w.split))
                if prev != w.split:
                    problems.append(f"{session} frame {f}: {prev} and {w.split}")
                    break
            if len(problems) > 5:
                break
    blocks = plan.frame()
    for w in windows.itertuples(index=False):
        inside = blocks[
            (blocks.session == w.session)
            & (blocks.split == w.split)
            & (blocks.start <= w.start)
            & (blocks.end >= w.end)
        ]
        if inside.empty:
            problems.append(f"{w.session} window [{w.start},{w.end}) crosses a block boundary")
            if len(problems) > 10:
                break
    return CheckResult(
        "no_window_overlap",
        not problems,
        "ok" if not problems else "; ".join(problems[:5]),
        {"n_problems": len(problems)},
    )


def check_preprocessing_fit_on_train(
    fit_blocks: Iterable[tuple[str, int, int]], plan: SplitPlan
) -> CheckResult:
    """Every frame the preprocessor was fitted on lies inside a TRAIN block."""
    train = plan.blocks_of("train")
    bad = [
        (s, a, b)
        for s, a, b in fit_blocks
        if not any(t.session == s and t.start <= a and b <= t.end for t in train)
    ]
    return CheckResult(
        "preprocessing_fit_on_train_only",
        not bad,
        "ok" if not bad else f"fitted on non-train frames: {bad[:5]}",
        {"non_train_blocks": [list(b) for b in bad]},
    )


def check_feature_schema(inputs: Iterable[str], allowed: Iterable[str]) -> CheckResult:
    """Decoder inputs are voxel data + physiology phases only; never simulator latent state."""
    cols = list(inputs)
    latent = sorted(c for c in cols if c in LATENT_COLUMNS or c.startswith("truth"))
    unknown = sorted(set(cols) - set(allowed) - set(latent))
    ok = not latent and not unknown
    detail = "ok" if ok else f"latent: {latent}; not allowed: {unknown}"
    return CheckResult("feature_schema", ok, detail, {"inputs": cols, "latent": latent})


def check_test_access(n_accesses: int, max_accesses: int) -> CheckResult:
    """The test fold is touched at most ``max_accesses`` times per split hash."""
    return CheckResult(
        "test_set_access",
        n_accesses <= max_accesses,
        f"{n_accesses} accesses (max {max_accesses})",
        {"n_accesses": n_accesses, "max": max_accesses},
    )


def check_train_test_gap(train_acc: float, test_acc: float, max_gap: float) -> CheckResult:
    """Overfitting guard: train accuracy may exceed test accuracy by at most ``max_gap``."""
    gap = float(train_acc - test_acc)
    return CheckResult(
        "train_test_gap",
        gap <= max_gap,
        f"train {train_acc:.3f} - test {test_acc:.3f} = {gap:.3f} (max {max_gap})",
        {"gap": gap, "max": max_gap},
    )


def split_checks(
    plan: SplitPlan,
    windows: pd.DataFrame,
    *,
    fit_blocks: Iterable[tuple[str, int, int]],
    inputs: Iterable[str],
    cfg: SplitConfig,
) -> list[CheckResult]:
    """All dataset-level leakage checks (run before training; any failure blocks promotion)."""
    return [
        check_sessions_disjoint(plan),
        check_subjects_disjoint(plan),
        check_temporal_gap(windows, plan.gap_frames),
        check_no_window_overlap(windows, plan),
        check_preprocessing_fit_on_train(fit_blocks, plan),
        check_feature_schema(inputs, cfg.allowed_features),
    ]


def checks_hash(checks: Sequence[CheckResult]) -> str:
    return hash_config({c.name: c.passed for c in checks})
