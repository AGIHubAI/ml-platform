"""fUSI preprocessing: SVD clutter filter -> power Doppler -> drift -> RETROICOR -> z-score.

One causal chain, two execution forms that must agree (``tests/unit/test_fusi_preprocess.py``):

* `StreamingPreprocessor` -- one volume at a time on the hot path (state: drift baseline);
* `batch_transform` -- the same arithmetic vectorised over a session (``(T, V)`` arrays),
  used by the Dask/Zarr batch job in ``batch.py``.

Steps per volume:

1. **SVD clutter filter** (Demene et al. 2015): the compound-frame ensemble is a
   Casorati matrix ``M`` (ensemble x voxels). Tissue is spatially extended and
   coherent across the ensemble, so it lives in the top singular vectors; blood
   decorrelates. The ensemble-side singular vectors are the eigenvectors of the small
   Gram matrix ``M M^H`` (n_ens x n_ens), so the filter costs O(n_ens^2 V), not a
   full SVD. Removing ``clutter_rank`` components and averaging ``|.|^2`` over the
   ensemble gives power Doppler (rescaled by ``n_ens / (n_ens - rank)``).
2. ``log`` (multiplicative physiology/speckle become additive).
3. Causal EMA drift baseline (``drift_tau_s``) subtracted -- causal so that the
   offline features equal what the hot path computes.
4. **RETROICOR** (Glover, Li, Ress 2000): per-voxel regression on
   ``cos/sin(m * phase)`` of the recorded cardiac and respiratory phases, m = 1..M.
5. Per-voxel z-score.

Steps 4-5 have fitted parameters. They are fitted on the TRAIN fold only
(`fit_preprocessor`) and stored as a versioned, hashed artifact
(`PreprocessorState.save`) that records which sessions/frames it saw; the leakage
check ``preprocessing_fit_on_train_only`` reads that record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import lfilter

from ml_platform.hashing import hash_config
from ml_platform.workloads.fusi_realtime.config import PreprocessConfig

PREPROCESSOR_VERSION = 1
STATE_ARRAYS = "state.npz"
STATE_META = "state.json"


# --------------------------------------------------------------------------- pure kernels
def clutter_filter(iq: np.ndarray, rank: int) -> np.ndarray:
    """Power Doppler from an IQ ensemble ``(..., ens, *space)`` -> ``(..., *space)`` float32.

    Leading dims are batch dims (e.g. time). The ensemble axis is the first axis
    after them: pass ``iq`` with shape ``(ens, x, y, z)`` or ``(t, ens, x, y, z)``.
    """
    single = iq.ndim == 4
    batch = iq[None] if single else iq
    t, n_ens = batch.shape[:2]
    space = batch.shape[2:]
    m: np.ndarray = batch.reshape(t, n_ens, -1).astype(np.complex64, copy=False)
    gram = m @ np.conj(np.swapaxes(m, 1, 2))  # (t, ens, ens), Hermitian
    _, vecs = np.linalg.eigh(gram)  # ascending eigenvalues
    if rank > 0:
        u = vecs[:, :, -rank:]  # dominant (tissue) subspace, (t, ens, rank)
        m = m - u @ (np.conj(np.swapaxes(u, 1, 2)) @ m)
    power = (np.abs(m) ** 2).mean(axis=1) * (n_ens / max(n_ens - rank, 1))
    out = power.reshape(t, *space).astype(np.float32)
    return out[0] if single else out


def retroicor_regressors(cardiac: np.ndarray, respiratory: np.ndarray, order: int) -> np.ndarray:
    """``(T, 4*order)`` Fourier basis of the physiological phases (Glover 2000, eq. 1)."""
    cols = []
    for phase in (np.atleast_1d(cardiac), np.atleast_1d(respiratory)):
        for m in range(1, order + 1):
            cols += [np.cos(m * phase), np.sin(m * phase)]
    return np.stack(cols, axis=1).astype(np.float64)


def ema_alpha(dt: float, tau_s: float) -> float:
    return float(1.0 - np.exp(-dt / tau_s))


def ema_detrend(y: np.ndarray, alpha: float) -> np.ndarray:
    """``y - baseline`` along axis 0 with a causal EMA baseline started at ``y[0]``."""
    zi = ((1 - alpha) * y[0])[None, ...]
    base, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], y, axis=0, zi=zi)
    out: np.ndarray = y - base
    return out


def log_doppler(power: np.ndarray, eps: float) -> np.ndarray:
    out: np.ndarray = np.log(power.astype(np.float64) + eps)
    return out


# --------------------------------------------------------------------------- fitted state
def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


@dataclass(frozen=True)
class PreprocessorState:
    """The fitted, versioned preprocessing artifact (frozen after fitting on train)."""

    coef: np.ndarray  # (4*order, V) RETROICOR betas
    mean: np.ndarray  # (V,)
    std: np.ndarray  # (V,)
    grid: tuple[int, int, int]
    frame_rate_hz: float
    params: dict[str, Any]
    fit_blocks: tuple[tuple[str, int, int], ...]  # (session, start_frame, end_frame) seen
    fit_n_frames: int
    version: int = PREPROCESSOR_VERSION
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def fit_sessions(self) -> tuple[str, ...]:
        return tuple(sorted({b[0] for b in self.fit_blocks}))

    def state_hash(self) -> str:
        return hash_config(
            {
                "params": self.params,
                "fit_blocks": [list(b) for b in self.fit_blocks],
                "grid": list(self.grid),
                "version": self.version,
                "arrays": [_digest(a) for a in (self.coef, self.mean, self.std)],
            }
        )

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / STATE_ARRAYS, coef=self.coef, mean=self.mean, std=self.std)
        meta = {
            "grid": list(self.grid),
            "frame_rate_hz": self.frame_rate_hz,
            "params": self.params,
            "fit_blocks": [list(b) for b in self.fit_blocks],
            "fit_n_frames": self.fit_n_frames,
            "version": self.version,
            "state_hash": self.state_hash(),
            "meta": self.meta,
        }
        (d / STATE_META).write_text(json.dumps(meta, indent=2, default=str))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> PreprocessorState:
        d = Path(directory)
        meta = json.loads((d / STATE_META).read_text())
        arrays = np.load(d / STATE_ARRAYS)
        state = cls(
            coef=arrays["coef"],
            mean=arrays["mean"],
            std=arrays["std"],
            grid=tuple(meta["grid"]),  # type: ignore[arg-type]
            frame_rate_hz=float(meta["frame_rate_hz"]),
            params=dict(meta["params"]),
            fit_blocks=tuple((str(s), int(a), int(b)) for s, a, b in meta["fit_blocks"]),
            fit_n_frames=int(meta["fit_n_frames"]),
            version=int(meta["version"]),
            meta=dict(meta.get("meta", {})),
        )
        if state.state_hash() != meta["state_hash"]:
            raise ValueError(f"preprocessor state in {d} does not match its recorded hash")
        return state


def preprocess_params(cfg: PreprocessConfig) -> dict[str, Any]:
    return asdict(_Params.from_cfg(cfg))


@dataclass(frozen=True)
class _Params:
    clutter_rank: int
    drift_tau_s: float
    retroicor_order: int
    eps: float

    @classmethod
    def from_cfg(cls, cfg: PreprocessConfig) -> _Params:
        return cls(cfg.clutter_rank, cfg.drift_tau_s, cfg.retroicor_order, cfg.eps)


def detrended_log(power: np.ndarray, frame_rate_hz: float, cfg: PreprocessConfig) -> np.ndarray:
    """Session ``(T, *space)`` power Doppler -> ``(T, V)`` drift-free log power (unfitted steps)."""
    y = log_doppler(power.reshape(power.shape[0], -1), cfg.eps)
    return ema_detrend(y, ema_alpha(1.0 / frame_rate_hz, cfg.drift_tau_s))


def fit_preprocessor(
    blocks: list[tuple[str, int, int, np.ndarray, np.ndarray, np.ndarray]],
    *,
    grid: tuple[int, int, int],
    frame_rate_hz: float,
    cfg: PreprocessConfig,
) -> PreprocessorState:
    """Fit RETROICOR + z-score on TRAIN blocks only.

    ``blocks``: ``(session, start, end, y_detrended (T,V), cardiac (T,), respiratory (T,))``
    for the training fold. Normal equations are accumulated block by block, so the
    whole fold never has to be in memory at once.
    """
    if not blocks:
        raise ValueError("fit_preprocessor needs at least one training block")
    order = cfg.retroicor_order
    nr = 4 * order
    nv = blocks[0][3].shape[1]
    rtr = np.zeros((nr, nr))
    rty = np.zeros((nr, nv))
    for _, _, _, y, card, resp in blocks:
        r = retroicor_regressors(card, resp, order)
        rtr += r.T @ r
        rty += r.T @ y
    coef = np.linalg.solve(rtr + 1e-6 * np.eye(nr), rty)
    s1 = np.zeros(nv)
    s2 = np.zeros(nv)
    n = 0
    for _, _, _, y, card, resp in blocks:
        res = y - retroicor_regressors(card, resp, order) @ coef
        s1 += res.sum(0)
        s2 += (res**2).sum(0)
        n += res.shape[0]
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 1e-12))
    return PreprocessorState(
        coef=coef.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        grid=grid,
        frame_rate_hz=frame_rate_hz,
        params=preprocess_params(cfg),
        fit_blocks=tuple((s, int(a), int(b)) for s, a, b, *_ in blocks),
        fit_n_frames=n,
    )


def apply_fitted(
    y: np.ndarray, cardiac: np.ndarray, respiratory: np.ndarray, state: PreprocessorState
) -> np.ndarray:
    """Frozen RETROICOR + z-score on drift-free log power ``(T, V)`` -> ``(T, V)`` float32."""
    order = int(state.params["retroicor_order"])
    r = retroicor_regressors(cardiac, respiratory, order)
    res = y - r @ state.coef.astype(np.float64)
    out: np.ndarray = ((res - state.mean) / state.std).astype(np.float32)
    return out


def batch_transform(
    power: np.ndarray, cardiac: np.ndarray, respiratory: np.ndarray, state: PreprocessorState
) -> np.ndarray:
    """Whole-session power Doppler ``(T, *space)`` -> features ``(T, *space)`` float32."""
    cfg = PreprocessConfig(**state.params)
    y = detrended_log(power, state.frame_rate_hz, cfg)
    return apply_fitted(y, cardiac, respiratory, state).reshape(power.shape)


# --------------------------------------------------------------------------- hot path
class StreamingPreprocessor:
    """Per-volume preprocessing with a causal drift state; equals `batch_transform`."""

    def __init__(self, state: PreprocessorState) -> None:
        self.state = state
        self.rank = int(state.params["clutter_rank"])
        self.eps = float(state.params["eps"])
        self.order = int(state.params["retroicor_order"])
        self.alpha = ema_alpha(1.0 / state.frame_rate_hz, float(state.params["drift_tau_s"]))
        self.coef = state.coef.astype(np.float64)
        self.mean = state.mean.astype(np.float64)
        self.std = state.std.astype(np.float64)
        self.baseline: np.ndarray | None = None

    def reset(self) -> None:
        self.baseline = None

    def power(self, iq: np.ndarray) -> np.ndarray:
        return clutter_filter(iq, self.rank)

    def step_power(self, power: np.ndarray, cardiac: float, respiratory: float) -> np.ndarray:
        y = np.log(power.reshape(-1).astype(np.float64) + self.eps)
        if self.baseline is None:
            self.baseline = y.copy()
        else:
            self.baseline = self.baseline + self.alpha * (y - self.baseline)
        r = retroicor_regressors(np.array([cardiac]), np.array([respiratory]), self.order)[0]
        res = (y - self.baseline) - r @ self.coef
        out: np.ndarray = ((res - self.mean) / self.std).astype(np.float32)
        return out.reshape(power.shape)

    def step(self, iq: np.ndarray, cardiac: float, respiratory: float) -> np.ndarray:
        return self.step_power(self.power(iq), cardiac, respiratory)
