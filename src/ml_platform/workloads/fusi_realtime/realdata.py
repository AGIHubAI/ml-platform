"""Loader for the public Caltech fUS-BMI dataset (Griggs et al., Nat. Neurosci. 2024).

STATUS: written against the *documented* structure and UNTESTED on the real files.
Tests use synthetic ``.mat`` files that mimic it. The 34 GB archive was never opened.

What the structure is based on (all from the authors' MATLAB code, rt_fUS_BMI,
``data loading/load_doppler_data.m`` and ``behavior/extract_training_data_and_labels.m``,
https://github.com/wsgriggs2/rt_fUS_BMI):

* files ``doppler data/rt_fUS_data_S{session}_R{run}.mat``; ``project_record.json``
* variables ``dop``, ``behavior``, ``UF``, ``timestamps``, ``neurovascular_map``,
  ``actual_labels``, ``predicted_labels``, ``coreParams``, ``frame_num``,
  ``memory_length``, ``state_num``, ``time_acq``
* ``dop``: 3-D MATLAB array (depth, lateral, time) -> mapped to (t, x, y=1, z)
* ``timestamps``: (n_frames, >=1); column 1 is the frame time in seconds
* ``behavior``: struct array, one element per trial; state fields ``trialstart``,
  ``initialfixation``, ``fixationhold``, ``cue``, ``memory``, ``target_acquire``,
  ``target_hold``, ``reward``, ``iti`` hold ``'HH:MM:SS,mmm'`` time strings

Not confirmed (guessed): the per-trial direction/target field name (several are tried),
and whether the raw ``timestamps`` are numeric or strings. Both are handled defensively.

Data is imaging-plane (2-D), so the y axis has length 1. The dataset is licensed
CC BY-NC 4.0 (non-commercial); never commit it. Download with
``scripts/download_fusi_caltech.py`` (opt-in).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ml_platform.logging import get_logger

log = get_logger(__name__)

RECORD_ID = "pa710-cdn95"
RECORD_URL = f"https://data.caltech.edu/records/{RECORD_ID}"
FILENAME = "fUS_BMI_Data.zip"
FILE_URL = f"{RECORD_URL}/files/{FILENAME}?download=1"
SIZE_BYTES = 34_253_258_547
MD5 = "6d07baffa0158c9fa9bd57c9c3d5aded"
LICENSE = "CC BY-NC 4.0 (non-commercial)"
DEFAULT_FS_HZ = 2.0

DOPPLER_KEYS = ("dop", "doppler", "Doppler", "pdi", "PDI")
TIMESTAMP_KEYS = ("timestamps", "timestamp", "time_acq")
BEHAVIOR_KEYS = ("behavior", "behaviour")
DIRECTION_KEYS = ("direction", "target", "target_num", "targetnum", "label", "angle")
ONSET_STATES = ("cue", "memory", "target_acquire", "trialstart")
_TIME_RE = re.compile(r"^\s*(-)?(\d+):(\d+):(\d+)[,.](\d+)\s*$")


@dataclass(frozen=True)
class FusiRecording:
    """One fUS run: power Doppler movie plus trial labels.

    ``doppler`` is float32 (t, x, y, z). Caltech data are 2-D imaging planes, so
    ``y == 1``; x is lateral, z is depth.
    """

    name: str
    doppler: np.ndarray
    fs_hz: float
    events: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)
    source: Path = Path()


def iter_caltech_sessions(root: Path) -> Iterator[Path]:
    """Yield ``.mat`` files below ``root`` in sorted order."""
    yield from sorted(Path(root).rglob("*.mat"))


def _first_key(d: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    return next((k for k in keys if k in d), None)


def _read_mat(path: Path) -> dict[str, Any]:
    """Read a MAT file; v7.3 (HDF5) needs the optional ``h5py``."""
    from scipy.io import loadmat

    try:
        return {
            k: v
            for k, v in loadmat(path, squeeze_me=True, struct_as_record=False).items()
            if not k.startswith("__")
        }
    except NotImplementedError:
        return _read_mat_v73(path)


def _read_mat_v73(path: Path) -> dict[str, Any]:
    """Minimal HDF5 reader: numeric datasets only, axes reversed back to MATLAB order.

    The ``behavior`` struct array is stored as object references and is skipped, so
    events are empty for v7.3 files.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ValueError(
            f"{path.name} is MATLAB v7.3 (HDF5); install h5py (uv add h5py) to read it"
        ) from exc
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for k, v in f.items():
            if isinstance(v, h5py.Dataset) and v.dtype.kind in "fiu":
                out[k] = np.asarray(v).T
    out["__v73__"] = True
    return out


def _parse_time(s: Any) -> float:
    """Parse ``'HH:MM:SS,mmm'`` to seconds; NaN for empty/unparseable."""
    if isinstance(s, (int, float, np.number)):
        return float(s)
    if isinstance(s, np.ndarray) and s.dtype.kind in "US":
        s = str(s.item()) if s.size == 1 else "".join(map(str, s.ravel()))
    if not isinstance(s, str):
        return float("nan")
    m = _TIME_RE.match(s)
    if not m:
        return float("nan")
    sign = -1.0 if m.group(1) else 1.0
    h, mi, se = int(m.group(2)), int(m.group(3)), int(m.group(4))
    frac = float("0." + m.group(5))
    return sign * (h * 3600 + mi * 60 + se + frac)


def _frame_times(raw: Any, n_t: int) -> np.ndarray | None:
    """Frame times in seconds from ``timestamps`` (numeric or time strings)."""
    arr = np.asarray(raw)
    if arr.dtype.kind in "fiu":
        col = arr if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)[:, 0]
    else:
        items = arr.ravel().tolist()
        col = np.array([_parse_time(x) for x in items], dtype=float)
    return col.astype(float) if col.shape[0] == n_t else None


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _field_names(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return list(obj)
    return list(getattr(obj, "_fieldnames", []))


def _events_from_behavior(behavior: Any, t0: float) -> pd.DataFrame:
    """Trial table (trial, onset_s, direction) from the per-trial behaviour structs."""
    cols = ["trial", "onset_s", "direction"]
    trials = list(np.atleast_1d(behavior).ravel())
    if not trials or not _field_names(trials[0]):
        return pd.DataFrame(columns=cols)
    names = _field_names(trials[0])
    onset_state = next((s for s in ONSET_STATES if s in names), None)
    dir_key = next((k for k in DIRECTION_KEYS if k in names), None)
    rows = []
    for i, tr in enumerate(trials):
        onset = _parse_time(_field(tr, onset_state)) - t0 if onset_state else float("nan")
        d = _field(tr, dir_key) if dir_key else None
        rows.append({"trial": i, "onset_s": onset, "direction": _scalar(d)})
    df = pd.DataFrame(rows, columns=cols)
    if dir_key is None:
        df = df.drop(columns="direction")
    return df


def _scalar(d: Any) -> Any:
    if d is None:
        return None
    a = np.asarray(d)
    if a.size == 0:
        return None
    return a.item() if a.size == 1 else a.tolist()


def _to_txz(dop: np.ndarray, found: list[str]) -> np.ndarray:
    """MATLAB (depth, lateral, time) -> (t, x, y=1, z); 4-D input is (z, x, y, t)."""
    if dop.ndim == 3:
        return np.transpose(dop, (2, 1, 0))[:, :, None, :].astype(np.float32)
    if dop.ndim == 4:
        return np.transpose(dop, (3, 1, 2, 0)).astype(np.float32)
    raise ValueError(f"doppler array must be 3-D or 4-D, got shape {dop.shape}; keys: {found}")


def load_caltech_session(path: Path) -> FusiRecording:
    """Load one ``rt_fUS_data_S*_R*.mat`` into a :class:`FusiRecording`.

    Raises ``ValueError`` naming the keys found if no Doppler array is recognised.
    """
    path = Path(path)
    data = _read_mat(path)
    keys = sorted(data)
    dk = _first_key(data, DOPPLER_KEYS)
    if dk is None:
        raise ValueError(
            f"{path.name}: no Doppler array (tried {DOPPLER_KEYS}); keys found: {keys}"
        )
    dop = np.asarray(data[dk])
    if dop.dtype.kind not in "fiu":
        raise ValueError(f"{path.name}: '{dk}' is not numeric ({dop.dtype}); keys found: {keys}")
    doppler = _to_txz(dop, keys)
    n_t = doppler.shape[0]

    meta: dict[str, Any] = {"keys": keys, "doppler_key": dk, "license": LICENSE}
    fs = DEFAULT_FS_HZ
    t0 = 0.0
    tk = _first_key(data, TIMESTAMP_KEYS)
    times = _frame_times(data[tk], n_t) if tk else None
    if times is not None and np.isfinite(times).all() and n_t > 1:
        t0 = float(times[0])
        dt = float(np.median(np.diff(times)))
        meta["timestamps_s"] = times - t0
        if dt > 0:
            fs = 1.0 / dt
    else:
        meta["timestamps_note"] = "frame times absent or unparseable; assuming 2 Hz"

    bk = _first_key(data, BEHAVIOR_KEYS)
    events = (
        _events_from_behavior(data[bk], t0)
        if bk
        else pd.DataFrame(columns=["trial", "onset_s", "direction"])
    )
    if data.get("__v73__"):
        meta["note"] = "v7.3 file: behavior struct not read, events empty"
    for k in ("actual_labels", "predicted_labels"):
        if k in data and np.asarray(data[k]).dtype.kind in "fiu":
            meta[k] = np.asarray(data[k]).ravel()
    m = re.search(r"S(\d+)_R(\d+)", path.stem)
    if m:
        meta["session"], meta["run"] = int(m.group(1)), int(m.group(2))
    log.info("fusi.realdata.loaded", name=path.stem, shape=list(doppler.shape), fs_hz=fs)
    return FusiRecording(
        name=path.stem, doppler=doppler, fs_hz=fs, events=events, meta=meta, source=path
    )


def to_xarray(rec: FusiRecording) -> xr.Dataset:
    """Dataset with ``doppler`` (t, x, y, z) and ``t`` in seconds."""
    t = np.arange(rec.doppler.shape[0]) / rec.fs_hz
    return xr.Dataset(
        {"doppler": (("t", "x", "y", "z"), rec.doppler)},
        coords={"t": t},
        attrs={"fs_hz": rec.fs_hz, "source": str(rec.source), "license": LICENSE},
    )
