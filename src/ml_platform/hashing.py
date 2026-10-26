"""Content hashing helpers used for lineage and reproducibility metadata.

All hashes are SHA-256 hex digests of a canonical serialization, so they are
stable across processes, machines and pandas/pyarrow versions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CHUNK = 1 << 20


def _canonical_series_bytes(s: pd.Series) -> bytes:
    """Canonical byte representation of one column (dtype-normalized)."""
    if pd.api.types.is_bool_dtype(s):
        arr = s.to_numpy(dtype=np.uint8)
        return b"b" + arr.tobytes()
    if pd.api.types.is_integer_dtype(s) and not s.isna().any():
        return b"i" + s.to_numpy(dtype="<i8").tobytes()
    if pd.api.types.is_numeric_dtype(s):
        arr = s.to_numpy(dtype="<f8", na_value=np.nan)
        # normalise -0.0 and NaN payloads so equal values hash equally
        arr = np.where(arr == 0.0, 0.0, arr)
        arr = np.where(np.isnan(arr), np.nan, arr)
        return b"f" + arr.astype("<f8").tobytes()
    if pd.api.types.is_datetime64_any_dtype(s):
        vals = [None if pd.isna(v) else pd.Timestamp(v).isoformat() for v in s]
        return b"t" + json.dumps(vals).encode()
    vals = [None if _is_na_scalar(v) else _jsonable(v) for v in s]
    return b"o" + json.dumps(vals, sort_keys=True, default=str).encode()


def _is_na_scalar(v: Any) -> bool:
    if isinstance(v, list | tuple | dict | np.ndarray):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def hash_dataframe(df: pd.DataFrame, *, sort_rows_by: str | None = None) -> str:
    """Content hash of a DataFrame.

    Columns are sorted by name and each column is converted to a canonical dtype
    (int64 / float64 / json) before hashing; the index is ignored. Row order
    matters unless ``sort_rows_by`` names a key column to sort by first.
    """
    if sort_rows_by is not None:
        df = df.sort_values(sort_rows_by, kind="mergesort")
    h = hashlib.sha256()
    h.update(f"rows={len(df)}".encode())
    for col in sorted(map(str, df.columns)):
        h.update(b"\x00col\x00" + col.encode())
        h.update(_canonical_series_bytes(df[col].reset_index(drop=True)))
    return h.hexdigest()


def hash_config(cfg: Any) -> str:
    """Hash a pydantic model / dict via canonical (sorted-key) JSON."""
    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump(mode="json")
    payload = json.dumps(_jsonable(cfg), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, or of a directory (sorted relative paths + bytes)."""
    p = Path(path)
    h = hashlib.sha256()
    files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
    for f in files:
        if p.is_dir():
            h.update(str(f.relative_to(p)).encode() + b"\x00")
        with f.open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
    return h.hexdigest()


def git_sha(short: bool = False) -> str | None:
    """Current git commit of the source tree (``None`` outside a repo); ``+dirty`` suffix if modified."""
    repo = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if short:
        sha = sha[:12]
    return f"{sha}+dirty" if dirty else sha
