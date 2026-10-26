"""Closed-loop contextual bandit over (decoder arm, feedback gain) actions.

Staged design: (1) `ReplayEnv` on recorded trials, where the brain data do not depend on
the chosen decoder, so every arm's prediction is computable and rewards are fully
counterfactual (exact regret); (2) `FittedSimEnv` on trials from a `FittedSimulator`,
where the gain additionally changes the simulated evoked amplitude by an ASSUMED rule
(`gain_amplitude`); (3) offline RL on logged trajectories (stub only, see
`OfflinePolicyLearner`).

Safety is a HARD constraint implemented as an action mask (`action_mask`): an action with
a gain above ``max_gain`` or a decoder that is not ``validated`` is removed before any
policy scores it, and `run_bandit` raises `SafetyViolation` if a policy returns a masked
action anyway. It is never expressed as a reward penalty, because a penalty can be
out-weighed by reward and lets a learner trade safety for performance while exploring.

Reward = 1 if the arm's decoded direction (argmax at the last labelled frame of the trial
window) equals the instructed direction, else 0. Decoder arms are PCA+LDA fitted on TRAIN
recordings only; test trials come from other recordings.

Gain has no effect in `ReplayEnv` (recorded data were produced under one fixed
feedback): gain-variants of an arm are indistinguishable there, which is a property of
replay, not a bug.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ml_platform.logging import get_logger
from ml_platform.workloads.fusi_realtime.config import (
    BanditConfig,
    DecoderArmConfig,
    PreprocessConfig,
)
from ml_platform.workloads.fusi_realtime.models import PcaLdaDecoder, softmax_np
from ml_platform.workloads.fusi_realtime.realdata import FusiRecording
from ml_platform.workloads.fusi_realtime.replay import (
    FittedSimulator,
    movie_log_features,
    trial_windows,
)

log = get_logger(__name__)

LABEL_DELAY_S = 1.0  # frames earlier than this after the cue carry no visible signal
CONTEXT_DIM = 4  # bias, recent success rate, normalised trial index, previous confidence


class SafetyViolation(RuntimeError):
    """A policy selected an action outside the allowed set."""


# ---------------------------------------------------------------------------- actions
@dataclass(frozen=True)
class Action:
    arm: str
    gain: float

    @property
    def label(self) -> str:
        return f"{self.arm}@{self.gain:g}"


def action_set(cfg: BanditConfig) -> list[Action]:
    """Full universe: every configured arm x every configured gain (masked separately)."""
    return [Action(a.name, g) for a in cfg.arms for g in cfg.gains]


def action_mask(actions: Sequence[Action], cfg: BanditConfig) -> np.ndarray:
    """Boolean mask of allowed actions: validated decoder AND ``gain <= max_gain``."""
    validated = {a.name for a in cfg.arms if a.validated}
    return np.array([a.arm in validated and a.gain <= cfg.max_gain for a in actions], dtype=bool)


def gain_amplitude(gain: float, cfg: BanditConfig) -> float:
    """ASSUMPTION: evoked-amplitude multiplier of feedback gain ``g`` (inverted U).

    ``1 + sensitivity*(g-1) - overload*(g-1)^2``: moderate gain raises engagement, large
    gain overwhelms it. No published estimate; chosen only to give the learner something
    to find in `FittedSimEnv`.
    """
    d = gain - 1.0
    return max(1.0 + cfg.gain_sensitivity * d - cfg.gain_overload * d * d, 0.0)


# ------------------------------------------------------------------------ decoder arms
@dataclass
class DecoderArm:
    cfg: DecoderArmConfig
    model: PcaLdaDecoder


def _labelled_rows(rec: FusiRecording, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    delay = int(np.ceil(LABEL_DELAY_S * rec.fs_hz))
    idx: list[int] = []
    lab: list[int] = []
    for w in trial_windows(rec, window_s):
        idx.extend(range(w.start + delay, w.stop))
        lab.extend([w.direction] * (w.stop - w.start - delay))
    return np.array(idx, dtype=int), np.array(lab, dtype=int)


def fit_arms(
    train: Sequence[FusiRecording],
    cfg: BanditConfig,
    n_classes: int,
    preprocess: PreprocessConfig | None = None,
) -> dict[str, DecoderArm]:
    """Fit every configured arm on ``train`` recordings only."""
    feats: dict[str, list[np.ndarray]] = {a.name: [] for a in cfg.arms}
    labels: dict[str, list[np.ndarray]] = {a.name: [] for a in cfg.arms}
    for rec in train:
        movie = movie_log_features(rec, preprocess)
        idx, lab = _labelled_rows(rec, cfg.trial_window_s)
        if not len(idx):
            continue
        cache: dict[int, np.ndarray] = {}
        for a in cfg.arms:
            if a.n_frames not in cache:
                cache[a.n_frames] = PcaLdaDecoder.frame_features(movie, a.n_frames)[idx]
            feats[a.name].append(cache[a.n_frames])
            labels[a.name].append(lab)
    arms: dict[str, DecoderArm] = {}
    for a in cfg.arms:
        model = PcaLdaDecoder(a.n_components, a.shrinkage, a.n_frames, n_classes)
        model.fit(np.concatenate(feats[a.name]), np.concatenate(labels[a.name]))
        arms[a.name] = DecoderArm(a, model)
    log.info("fusi.bandit.arms_fitted", arms=list(arms), recordings=[r.name for r in train])
    return arms


def _decide(arm: DecoderArm, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class and softmax confidence for feature rows."""
    dec = arm.model.decision(feats)
    return dec.argmax(1), softmax_np(dec).max(1)


@dataclass
class TrialBank:
    """Held-out trials: directions plus last-frame features per distinct ``n_frames``."""

    directions: np.ndarray  # (N,)
    feats: dict[int, np.ndarray]  # n_frames -> (N, V)
    source: list[str] = field(default_factory=list)


def build_trial_bank(
    recs: Sequence[FusiRecording],
    cfg: BanditConfig,
    preprocess: PreprocessConfig | None = None,
) -> TrialBank:
    dirs: list[int] = []
    rows: dict[int, list[np.ndarray]] = {a.n_frames: [] for a in cfg.arms}
    for rec in recs:
        movie = movie_log_features(rec, preprocess)
        wins = trial_windows(rec, cfg.trial_window_s)
        for nf in rows:
            ff = PcaLdaDecoder.frame_features(movie, nf)
            rows[nf].append(ff[[w.stop - 1 for w in wins]])
        dirs.extend(w.direction for w in wins)
    return TrialBank(
        np.array(dirs, dtype=int),
        {nf: np.concatenate(v) for nf, v in rows.items()},
        [r.name for r in recs],
    )


# ----------------------------------------------------------------------- environments
@dataclass(frozen=True)
class TrialOutcome:
    """Counterfactual outcome of one trial: reward / confidence for every action.

    Masked actions hold NaN: they are never evaluated, so they cannot leak into regret.
    """

    rewards: np.ndarray
    confidence: np.ndarray


class BanditEnv(Protocol):
    name: str
    actions: list[Action]
    mask: np.ndarray

    def reset(self, seed: int) -> None: ...

    def trial(self, t: int) -> TrialOutcome: ...


def _expand(
    per_arm_reward: dict[str, float],
    per_arm_conf: dict[str, float],
    actions: Sequence[Action],
    mask: np.ndarray,
) -> TrialOutcome:
    r = np.array(
        [
            per_arm_reward.get(a.arm, np.nan) if m else np.nan
            for a, m in zip(actions, mask, strict=True)
        ]
    )
    c = np.array(
        [
            per_arm_conf.get(a.arm, np.nan) if m else np.nan
            for a, m in zip(actions, mask, strict=True)
        ]
    )
    return TrialOutcome(r, c)


class ReplayEnv:
    """Held-out recorded trials in a seeded order (permutation cycles).

    Each arm's decision on a recorded trial is fixed by the data, so the reward table is
    computed once for all arms (counterfactual) and regret is exact.
    """

    name = "replay"

    def __init__(self, arms: dict[str, DecoderArm], bank: TrialBank, cfg: BanditConfig) -> None:
        self.actions = action_set(cfg)
        self.mask = action_mask(self.actions, cfg)
        allowed = {a.arm for a, m in zip(self.actions, self.mask, strict=True) if m}
        self._reward: dict[str, np.ndarray] = {}
        self._conf: dict[str, np.ndarray] = {}
        for name, arm in arms.items():
            if name not in allowed:
                continue  # masked arms are never evaluated
            pred, conf = _decide(arm, bank.feats[arm.cfg.n_frames])
            self._reward[name] = (pred == bank.directions).astype(float)
            self._conf[name] = conf
        self.n_pool = len(bank.directions)
        self._order = np.arange(self.n_pool)

    def reset(self, seed: int) -> None:
        self._seed = seed
        self._cycles: list[np.ndarray] = []

    def _index(self, t: int) -> int:
        c = t // self.n_pool
        while len(self._cycles) <= c:
            rng = np.random.default_rng([self._seed, len(self._cycles)])
            self._cycles.append(rng.permutation(self.n_pool))
        return int(self._cycles[c][t % self.n_pool])

    def trial(self, t: int) -> TrialOutcome:
        i = self._index(t)
        return _expand(
            {k: float(v[i]) for k, v in self._reward.items()},
            {k: float(v[i]) for k, v in self._conf.items()},
            self.actions,
            self.mask,
        )


class FittedSimEnv:
    """Trials generated by a `FittedSimulator`; gain scales the evoked amplitude.

    Per trial one direction, one noise draw and one lapse draw are shared by all actions
    (common random numbers), so differences between actions are due to arm and gain only.
    A lapse (probability ``lapse_rate``) removes the evoked signal for all actions.
    """

    name = "fitted_sim"

    def __init__(
        self, arms: dict[str, DecoderArm], sim: FittedSimulator, cfg: BanditConfig
    ) -> None:
        self.arms, self.sim, self.cfg = arms, sim, cfg
        self.actions = action_set(cfg)
        self.mask = action_mask(self.actions, cfg)
        assert sim.directions_ is not None
        self._dirs = sim.directions_
        self._memo: dict[int, TrialOutcome] = {}

    def reset(self, seed: int) -> None:
        if getattr(self, "_seed", None) != seed:
            self._memo = {}  # outcomes are a pure function of (seed, t)
        self._seed = seed

    def trial(self, t: int) -> TrialOutcome:
        if t not in self._memo:
            self._memo[t] = self._generate(t)
        return self._memo[t]

    def _generate(self, t: int) -> TrialOutcome:
        rng = np.random.default_rng([self._seed, t])  # depends on t only, never on actions
        direction = int(self._dirs[rng.integers(len(self._dirs))])
        noise = self.sim.sample_noise(rng)
        lapse = rng.random() < self.cfg.lapse_rate
        per_arm: dict[tuple[str, float], tuple[float, float]] = {}
        for a, ok in zip(self.actions, self.mask, strict=True):
            if not ok or (a.arm, a.gain) in per_arm:
                continue
            amp = 0.0 if lapse else gain_amplitude(a.gain, self.cfg)
            x = self.sim.compose(direction, amp, noise)
            arm = self.arms[a.arm]
            feats = PcaLdaDecoder.frame_features(x, arm.cfg.n_frames)[-1:]
            pred, conf = _decide(arm, feats)
            per_arm[(a.arm, a.gain)] = (float(pred[0] == direction), float(conf[0]))
        nan = (np.nan, np.nan)
        pairs = [
            per_arm.get((a.arm, a.gain), nan) if m else nan
            for a, m in zip(self.actions, self.mask, strict=True)
        ]
        return TrialOutcome(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))


# --------------------------------------------------------------------------- policies
class Policy(Protocol):
    name: str

    def select(self, context: np.ndarray, mask: np.ndarray) -> int: ...

    def update(self, action: int, context: np.ndarray, reward: float) -> None: ...

    def propensity(self, action: int, context: np.ndarray, mask: np.ndarray) -> float: ...


def _check_mask(mask: np.ndarray) -> None:
    if not mask.any():
        raise SafetyViolation("no allowed action")


class _LinearModel:
    """Per-action ridge statistics ``A = I + sum x x^T``, ``b = sum r x``."""

    def __init__(self, n_actions: int, dim: int) -> None:
        self.a = np.stack([np.eye(dim) for _ in range(n_actions)])
        self.b = np.zeros((n_actions, dim))

    def inv(self) -> np.ndarray:
        out: np.ndarray = np.linalg.inv(self.a)
        return out

    def update(self, action: int, x: np.ndarray, reward: float) -> None:
        self.a[action] += np.outer(x, x)
        self.b[action] += reward * x


class LinUCB:
    """Disjoint LinUCB (Li et al. 2010): ``theta.x + alpha*sqrt(x^T A^-1 x)``.

    Deterministic given the history; ties are broken by a seeded jitter.
    """

    def __init__(self, n_actions: int, dim: int, alpha: float, seed: int) -> None:
        self.name = f"linucb(a={alpha:g})"
        self.alpha = alpha
        self.model = _LinearModel(n_actions, dim)
        self.rng = np.random.default_rng(seed)

    def select(self, context: np.ndarray, mask: np.ndarray) -> int:
        _check_mask(mask)
        ainv = self.model.inv()
        theta = np.einsum("kij,kj->ki", ainv, self.model.b)
        width = np.sqrt(np.einsum("i,kij,j->k", context, ainv, context))
        score = theta @ context + self.alpha * width + 1e-9 * self.rng.random(len(mask))
        return int(np.argmax(np.where(mask, score, -np.inf)))

    def update(self, action: int, context: np.ndarray, reward: float) -> None:
        self.model.update(action, context, reward)

    def propensity(self, action: int, context: np.ndarray, mask: np.ndarray) -> float:
        """1.0: the choice is a deterministic function of history (no logging randomness)."""
        return 1.0


class LinearThompson:
    """Linear Thompson sampling: ``theta ~ N(A^-1 b, v^2 A^-1)`` per action, seeded."""

    N_MC = 64

    def __init__(self, n_actions: int, dim: int, v: float, seed: int) -> None:
        self.name = f"thompson(v={v:g})"
        self.v = v
        self.model = _LinearModel(n_actions, dim)
        self.rng = np.random.default_rng(seed)
        self._prop_rng = np.random.default_rng([seed, 99])  # keeps select() reproducible

    def _chol(self) -> tuple[np.ndarray, np.ndarray]:
        ainv = self.model.inv()
        ainv = 0.5 * (ainv + np.transpose(ainv, (0, 2, 1)))
        return np.einsum("kij,kj->ki", ainv, self.model.b), np.linalg.cholesky(ainv)

    def _draw(
        self, rng: np.random.Generator, n: int, mean: np.ndarray, chol: np.ndarray
    ) -> np.ndarray:
        z = rng.standard_normal((n, *mean.shape))
        out: np.ndarray = mean[None] + self.v * np.einsum("kij,nkj->nki", chol, z)
        return out

    def select(self, context: np.ndarray, mask: np.ndarray) -> int:
        _check_mask(mask)
        mean, chol = self._chol()
        score = self._draw(self.rng, 1, mean, chol)[0] @ context
        return int(np.argmax(np.where(mask, score, -np.inf)))

    def update(self, action: int, context: np.ndarray, reward: float) -> None:
        self.model.update(action, context, reward)

    def propensity(self, action: int, context: np.ndarray, mask: np.ndarray) -> float:
        """Monte-Carlo estimate (64 draws, floored at 1/64) of P(select == action)."""
        mean, chol = self._chol()
        score = self._draw(self._prop_rng, self.N_MC, mean, chol) @ context
        picks = np.argmax(np.where(mask[None], score, -np.inf), axis=1)
        return float(max(np.mean(picks == action), 1.0 / self.N_MC))


class UniformPolicy:
    """Uniform over allowed actions (the random baseline)."""

    name = "uniform"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, context: np.ndarray, mask: np.ndarray) -> int:
        _check_mask(mask)
        return int(self.rng.choice(np.flatnonzero(mask)))

    def update(self, action: int, context: np.ndarray, reward: float) -> None:
        return None

    def propensity(self, action: int, context: np.ndarray, mask: np.ndarray) -> float:
        return 1.0 / float(mask.sum())


class FixedPolicy:
    """Always plays one action; raises `SafetyViolation` if it is masked."""

    def __init__(self, action: int, label: str = "") -> None:
        self.action = action
        self.name = f"fixed({label or action})"

    def select(self, context: np.ndarray, mask: np.ndarray) -> int:
        if not mask[self.action]:
            raise SafetyViolation(f"fixed action {self.action} is masked")
        return self.action

    def update(self, action: int, context: np.ndarray, reward: float) -> None:
        return None

    def propensity(self, action: int, context: np.ndarray, mask: np.ndarray) -> float:
        return 1.0


def make_policy(cfg: BanditConfig, n_actions: int, seed: int | None = None) -> Policy:
    s = cfg.seed if seed is None else seed
    if cfg.policy == "linucb":
        return LinUCB(n_actions, CONTEXT_DIM, cfg.alpha, s)
    if cfg.policy == "thompson":
        return LinearThompson(n_actions, CONTEXT_DIM, cfg.thompson_v, s)
    return UniformPolicy(s)


# ------------------------------------------------------------------------ offline hook
@dataclass(frozen=True)
class LoggedTrajectory:
    """One logged decision for off-policy evaluation / learning."""

    trial: int
    context: tuple[float, ...]
    action: int
    propensity: float
    reward: float


class OfflinePolicyLearner(Protocol):
    def fit(self, trajectories: Sequence[LoggedTrajectory]) -> None: ...


class OfflineLearnerStub:
    """Placeholder for the offline-RL stage."""

    def fit(self, trajectories: Sequence[LoggedTrajectory]) -> None:
        """Intended use: off-policy evaluation of a candidate policy pi on logs from pi_0.

        With propensities ``p_0(a|x)`` logged, the inverse-propensity estimate
        ``V(pi) = mean(1[pi(x)=a] * r / p_0(a|x))`` (and doubly robust variants) scores a
        candidate gain/decoder policy without deploying it, and can gate promotion of
        a learned policy. Valid only if logging was stochastic (uniform or Thompson, not
        greedy LinUCB with propensity 1) and the candidate stays inside the safety mask.
        Not implemented in this stage.
        """
        raise NotImplementedError("offline policy learning is a stub; see docstring")


def write_trajectories(path: Path, trajectories: Sequence[LoggedTrajectory]) -> Path:
    """Write trajectories as JSON lines."""
    path = Path(path)
    with path.open("w") as fh:
        for tr in trajectories:
            fh.write(json.dumps(tr.__dict__) + "\n")
    return path


# ---------------------------------------------------------------------------- the loop
@dataclass
class BanditResult:
    policy: str
    env: str
    seed: int
    actions: list[Action]
    chosen: np.ndarray  # (T,) action index
    rewards: np.ndarray  # (T,)
    oracle: np.ndarray  # (T,) best allowed action's reward on that trial
    reward_matrix: np.ndarray  # (T, A) counterfactual rewards, NaN where masked
    trajectory: list[LoggedTrajectory]

    @property
    def cum_regret(self) -> np.ndarray:
        out: np.ndarray = np.cumsum(self.oracle - self.rewards)
        return out

    @property
    def final_regret(self) -> float:
        return float(self.cum_regret[-1])

    def selection_frequency(self) -> np.ndarray:
        return np.bincount(self.chosen, minlength=len(self.actions)) / len(self.chosen)


def run_bandit(
    policy: Policy, env: BanditEnv, n_trials: int, seed: int, window: int = 10
) -> BanditResult:
    """Play ``n_trials`` against ``env``; the trial sequence depends on ``seed`` only."""
    env.reset(seed)
    n_a = len(env.actions)
    chosen = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    matrix = np.full((n_trials, n_a), np.nan)
    traj: list[LoggedTrajectory] = []
    prev_conf = 0.5
    for t in range(n_trials):
        recent = rewards[max(t - window, 0) : t]
        ctx = np.array([1.0, float(recent.mean()) if t else 0.5, t / n_trials, prev_conf])
        a = policy.select(ctx, env.mask)
        if not env.mask[a]:
            raise SafetyViolation(
                f"policy {policy.name} chose masked action {env.actions[a].label}"
            )
        out = env.trial(t)
        prop = policy.propensity(a, ctx, env.mask)
        policy.update(a, ctx, float(out.rewards[a]))
        chosen[t], rewards[t], matrix[t] = a, out.rewards[a], out.rewards
        prev_conf = float(out.confidence[a])
        traj.append(LoggedTrajectory(t, tuple(map(float, ctx)), a, prop, float(out.rewards[a])))
    oracle = np.nanmax(matrix, axis=1)
    return BanditResult(
        policy.name, env.name, seed, env.actions, chosen, rewards, oracle, matrix, traj
    )


def best_fixed_in_hindsight(result: BanditResult) -> tuple[int, np.ndarray]:
    """Best single action over the run (uses the counterfactual matrix) and its regret."""
    totals = np.nansum(result.reward_matrix, axis=0)
    totals[np.isnan(result.reward_matrix).all(axis=0)] = -np.inf
    best = int(np.argmax(totals))
    return best, np.cumsum(result.oracle - result.reward_matrix[:, best])


def run_baselines(env: BanditEnv, n_trials: int, seed: int) -> dict[str, BanditResult]:
    """Uniform random plus every allowed action played as a fixed policy."""
    out = {"uniform": run_bandit(UniformPolicy(seed), env, n_trials, seed)}
    for i, a in enumerate(env.actions):
        if env.mask[i]:
            out[f"fixed:{a.label}"] = run_bandit(FixedPolicy(i, a.label), env, n_trials, seed)
    return out
