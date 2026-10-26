"""Batch preprocessing of 4-D fUSI sessions with Dask + xarray over Zarr.

Session store layout (one directory per session, written once)::

    <session>/raw.zarr        iq_re, iq_im (t, ens, x, y, z) float32   simulator output
                              or doppler   (t, x, y, z) float32        real power Doppler
                              cardiac_phase, respiratory_phase (t)
    <session>/events.parquet  task events (labels): frame, t_s, trial, phase, cue_direction
    <session>/truth.parquet   simulator latent state -- EVALUATION ONLY
    <session>/manifest.json   sizes + content hashes

Dimension names are ``x, y, z, t`` (stored ``t``-major so time chunks stream).

Execution plan:

1. clutter filter + power Doppler: ``apply_ufunc`` per time chunk (embarrassingly
   parallel, the expensive step);
2. log + causal drift removal: rechunk to full time x voxel blocks, EMA per voxel;
3. RETROICOR + z-score fit on TRAIN blocks only: normal equations ``R^T R`` (tiny) and
   ``R^T Y`` (dask reduction over time), then residual moments -- per voxel, so the fit
   never materialises a whole session;
4. frozen transform of every session -> ``features.zarr`` (float32, ``t, x, y, z``).

The numpy reference for steps 3-4 is ``preprocess.fit_preprocessor`` /
``preprocess.batch_transform``; ``tests/unit/test_fusi_batch.py`` checks they agree.

Daft is not used: events/labels are a few thousand rows per session, which pandas +
Parquet handle directly; a second dataframe engine would add a dependency without
removing any work at this scale.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from ml_platform.hashing import hash_dataframe
from ml_platform.workloads.fusi_realtime.config import PreprocessConfig
from ml_platform.workloads.fusi_realtime.preprocess import (
    PreprocessorState,
    clutter_filter,
    ema_alpha,
    ema_detrend,
    preprocess_params,
    retroicor_regressors,
)
from ml_platform.workloads.fusi_realtime.sim import FusiSimulator, frames_to_tables

RAW = "raw.zarr"
FEATURES = "features.zarr"
EVENTS = "events.parquet"
TRUTH = "truth.parquet"
MANIFEST = "manifest.json"
SPACE = ("x", "y", "z")


@dataclass(frozen=True)
class SessionStore:
    key: str
    subject: str
    path: Path
    n_frames: int
    frame_rate_hz: float
    grid: tuple[int, int, int]
    events_hash: str
    variables: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> SessionStore:
        m = json.loads((path / MANIFEST).read_text())
        return cls(
            key=m["key"],
            subject=m["subject"],
            path=path,
            n_frames=int(m["n_frames"]),
            frame_rate_hz=float(m["frame_rate_hz"]),
            grid=tuple(m["grid"]),  # type: ignore[arg-type]
            events_hash=m["events_hash"],
            variables=tuple(m["variables"]),
        )


@contextmanager
def dask_threads(n: int | None = None) -> Iterator[None]:
    """Local threaded scheduler (numpy/BLAS release the GIL); no distributed cluster needed."""
    with dask.config.set(scheduler="threads", num_workers=n):
        yield


def write_simulated_session(
    root: Path,
    sim: FusiSimulator,
    *,
    key: str,
    subject: str,
    n_frames: int,
    chunk_t: int,
    store_iq: bool = True,
    clutter_rank: int = 1,
) -> SessionStore:
    """Stream ``n_frames`` from ``sim`` into ``root/key`` chunk by chunk (bounded memory).

    With an IQ-emitting simulator and ``store_iq=False`` the SVD clutter filter runs
    here, per chunk, and only power Doppler is persisted (scanner-side filtering).
    """
    out = root / key
    if (out / MANIFEST).exists():
        return SessionStore.load(out)  # write-once: same key, same content (deterministic sim)
    out.mkdir(parents=True, exist_ok=True)
    store = out / RAW
    all_frames = []
    first = True
    for start in range(0, n_frames, chunk_t):
        frames = [sim.next_frame() for _ in range(min(chunk_t, n_frames - start))]
        t = np.array([f.t_s for f in frames])
        data: dict[str, Any] = {
            "cardiac_phase": ("t", np.array([f.cardiac_phase for f in frames])),
            "respiratory_phase": ("t", np.array([f.respiratory_phase for f in frames])),
        }
        if sim.emit == "iq" and not store_iq:
            iq = np.stack([f.iq for f in frames if f.iq is not None])
            data["doppler"] = (("t", *SPACE), clutter_filter(iq, clutter_rank))
        elif sim.emit == "iq":
            iq = np.stack([f.iq for f in frames if f.iq is not None])
            data["iq_re"] = (("t", "ens", *SPACE), iq.real.astype(np.float32))
            data["iq_im"] = (("t", "ens", *SPACE), iq.imag.astype(np.float32))
        else:
            dop = np.stack([f.doppler for f in frames if f.doppler is not None])
            data["doppler"] = (("t", *SPACE), dop)
        ds = xr.Dataset(data, coords={"t": t})
        if first:
            ds.attrs.update({"frame_rate_hz": sim.cfg.frame_rate_hz, "key": key})
            # explicit chunks: one time chunk holds whole volumes (the clutter filter's unit)
            enc = {
                v: {"chunks": (chunk_t, *ds[v].shape[1:])}
                for v in ds.data_vars
                if str(v).startswith(("iq", "dop"))
            }
            ds.to_zarr(store, mode="w", consolidated=False, encoding=enc)
            first = False
        else:
            ds.to_zarr(store, append_dim="t", consolidated=False)
        all_frames += [replace(f, iq=None, doppler=None) for f in frames]
    events, truth = frames_to_tables(all_frames)
    events.to_parquet(out / EVENTS, index=False)
    truth.to_parquet(out / TRUTH, index=False)
    stored_iq = sim.emit == "iq" and store_iq
    variables = ("iq" if stored_iq else "doppler", "cardiac_phase", "respiratory_phase")
    manifest = {
        "key": key,
        "subject": subject,
        "n_frames": n_frames,
        "frame_rate_hz": sim.cfg.frame_rate_hz,
        "grid": list(sim.grid),
        "variables": list(variables),
        "events_hash": hash_dataframe(events),
        "truth_file": TRUTH,
        "truth_policy": "evaluation only; never a decoder input",
        "source": "simulator",
        "clutter_filter": "batch" if stored_iq else "acquisition" if sim.emit == "iq" else "none",
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return SessionStore.load(out)


def open_raw(session: SessionStore) -> xr.Dataset:
    return xr.open_zarr(session.path / RAW, consolidated=False)


def power_doppler(ds: xr.Dataset, rank: int) -> xr.DataArray:
    """Lazy power Doppler ``(t, x, y, z)``; SVD clutter filter per time chunk when IQ is stored."""
    if "doppler" in ds:
        return ds["doppler"]

    def _filter(re: np.ndarray, im: np.ndarray) -> np.ndarray:
        return clutter_filter((re + 1j * im).astype(np.complex64), rank)

    whole = {d: -1 for d in ("ens", *SPACE)}
    out: xr.DataArray = xr.apply_ufunc(
        _filter,
        ds["iq_re"].chunk(whole),
        ds["iq_im"].chunk(whole),
        input_core_dims=[["ens", *SPACE], ["ens", *SPACE]],
        output_core_dims=[list(SPACE)],
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    return out.rename("doppler")


def detrended_log(power: xr.DataArray, frame_rate_hz: float, cfg: PreprocessConfig) -> xr.DataArray:
    """log + causal EMA drift removal; time must be one chunk, so rechunk voxels instead."""
    alpha = ema_alpha(1.0 / frame_rate_hz, cfg.drift_tau_s)
    logp: xr.DataArray = np.log(power.astype(np.float64) + cfg.eps)  # type: ignore[assignment]
    logp = logp.chunk({"t": -1, "x": max(1, power.sizes["x"] // 4)})

    def _ema_last(y: np.ndarray) -> np.ndarray:
        return np.moveaxis(ema_detrend(np.moveaxis(y, -1, 0), alpha), 0, -1)

    out: xr.DataArray = xr.apply_ufunc(
        _ema_last,
        logp,
        input_core_dims=[["t"]],
        output_core_dims=[["t"]],
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    return out.transpose("t", *SPACE)


def _regressors(ds: xr.Dataset, order: int, sl: slice) -> np.ndarray:
    card = ds["cardiac_phase"].isel(t=sl).values
    resp = ds["respiratory_phase"].isel(t=sl).values
    return retroicor_regressors(card, resp, order)


def fit_preprocessor_dask(
    sessions: dict[str, SessionStore],
    train_blocks: list[tuple[str, int, int]],
    cfg: PreprocessConfig,
) -> PreprocessorState:
    """Fit RETROICOR + z-score on ``train_blocks`` ``(session, start, end)`` only."""
    if not train_blocks:
        raise ValueError("no training blocks to fit the preprocessor on")
    order, nr = cfg.retroicor_order, 4 * cfg.retroicor_order
    first = sessions[train_blocks[0][0]]
    nv = int(np.prod(first.grid))
    ys: list[tuple[da.Array, np.ndarray]] = []
    for key, a, b in train_blocks:
        s = sessions[key]
        ds = open_raw(s)
        y = detrended_log(power_doppler(ds, cfg.clutter_rank), s.frame_rate_hz, cfg)
        ys.append(
            (y.isel(t=slice(a, b)).data.reshape(b - a, nv), _regressors(ds, order, slice(a, b)))
        )
    rtr = sum((r.T @ r for _, r in ys), np.zeros((nr, nr)))
    rty = sum((da.from_array(r.T) @ y for y, r in ys), da.zeros((nr, nv)))
    coef = np.linalg.solve(rtr + 1e-6 * np.eye(nr), np.asarray(rty.compute()))
    res = [y - da.from_array(r) @ coef for y, r in ys]
    s1 = sum((x.sum(0) for x in res), da.zeros(nv))
    s2 = sum(((x**2).sum(0) for x in res), da.zeros(nv))
    n = sum(b - a for _, a, b in train_blocks)
    m1, m2 = (np.asarray(v) for v in da.compute(s1, s2))
    mean = m1 / n
    std = np.sqrt(np.maximum(m2 / n - mean**2, 1e-12))
    return PreprocessorState(
        coef=coef.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        grid=first.grid,
        frame_rate_hz=first.frame_rate_hz,
        params=preprocess_params(cfg),
        fit_blocks=tuple((k, int(a), int(b)) for k, a, b in train_blocks),
        fit_n_frames=n,
    )


def transform_session(session: SessionStore, state: PreprocessorState) -> xr.DataArray:
    """Lazy frozen transform of a whole session -> features ``(t, x, y, z)`` float32."""
    cfg = PreprocessConfig(**state.params)
    ds = open_raw(session)
    y = detrended_log(power_doppler(ds, cfg.clutter_rank), session.frame_rate_hz, cfg)
    r = xr.DataArray(_regressors(ds, cfg.retroicor_order, slice(None)), dims=("t", "r"))
    grid = session.grid
    coef = xr.DataArray(state.coef.reshape(-1, *grid).astype(np.float64), dims=("r", *SPACE))
    mean = xr.DataArray(state.mean.reshape(grid), dims=SPACE)
    std = xr.DataArray(state.std.reshape(grid), dims=SPACE)
    feats = ((y - xr.dot(r, coef, dim="r")) - mean) / std
    out: xr.DataArray = feats.astype(np.float32).transpose("t", *SPACE).rename("features")
    return out


def write_features(
    session: SessionStore, state: PreprocessorState, out_dir: Path, chunk_t: int
) -> Path:
    """Materialise ``features.zarr`` for one session under ``out_dir/<key>/``."""
    path = out_dir / session.key / FEATURES
    if path.exists():
        return path
    feats = transform_session(session, state).chunk({"t": chunk_t, "x": -1, "y": -1, "z": -1})
    ds = feats.to_dataset()
    ds.attrs.update({"preprocessor_hash": state.state_hash(), "session": session.key})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", consolidated=False)
    return path


def read_features(path: Path) -> np.ndarray:
    out: np.ndarray = xr.open_zarr(path, consolidated=False)["features"].values
    return out


def read_events(session: SessionStore) -> pd.DataFrame:
    return pd.read_parquet(session.path / EVENTS)


def read_truth(session: SessionStore) -> pd.DataFrame:
    """Latent simulator state. Only evaluation code may call this."""
    return pd.read_parquet(session.path / TRUTH)
