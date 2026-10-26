"""Where generated state lives, and a hard stop when that place is an unmounted drive.

Local runs of the fUSI profiles keep every byte of generated state on an external
drive (default ``/Volumes/T7/ml-platform-data``, override with ``$ML_STORAGE_ROOT``):
session volumes, Zarr features, checkpoints, MLflow db + artifacts, Dagster storage,
Ray temp + object spill, Dask temp. Config files point there with
``${VAR:-default}`` interpolation (``config._read_yaml``).

The failure mode this module prevents: on macOS an unmounted ``/Volumes/T7`` is just a
missing directory, and ``mkdir -p /Volumes/T7/...`` silently creates it on the *internal*
disk. `require_mounted` refuses any path under ``/Volumes/<name>`` whose volume is not
an actual mount point.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

STORAGE_ENV = "ML_STORAGE_ROOT"
DEFAULT_STORAGE_ROOT = Path("/Volumes/T7/ml-platform-data")
SPILL_ENV = "ML_RAY_SPILL_DIR"
VOLUMES = Path("/Volumes")


class StorageNotMountedError(RuntimeError):
    """A configured path lives on an external volume that is not mounted."""


def storage_root() -> Path:
    return Path(os.environ.get(STORAGE_ENV) or DEFAULT_STORAGE_ROOT)


def _volume_of(path: Path) -> Path | None:
    p = Path(os.path.abspath(path))
    if len(p.parts) >= 3 and Path(*p.parts[:2]) == VOLUMES:
        return Path(*p.parts[:3])
    return None


def require_mounted(path: str | os.PathLike[str]) -> Path:
    """Return ``path``; raise if it is under ``/Volumes/<name>`` and that volume is not mounted."""
    p = Path(path)
    vol = _volume_of(p)
    if vol is not None and not os.path.ismount(vol):
        raise StorageNotMountedError(
            f"{p} is on {vol}, which is not mounted. Mount the drive, or point "
            f"${STORAGE_ENV} (and the paths in the config) at another location. "
            "Refusing to write to the internal disk instead."
        )
    return p


def apply_runtime_env(
    *,
    dagster_home: Path | None,
    ray_temp_dir: Path | None,
    ray_spill_dir: Path | None,
    dask_temp_dir: Path | None,
) -> dict[str, str]:
    """Export DAGSTER_HOME / RAY_TMPDIR / spill / Dask temp for this process (setdefault only).

    Call before Dagster or Ray start. Existing environment variables win, so an operator
    can still redirect any single store.
    """
    pairs = {
        "DAGSTER_HOME": dagster_home,
        "RAY_TMPDIR": ray_temp_dir,
        SPILL_ENV: ray_spill_dir,
        "DASK_TEMPORARY_DIRECTORY": dask_temp_dir,
    }
    out: dict[str, str] = {}
    for key, path in pairs.items():
        if path is None:
            continue
        value = os.environ.setdefault(key, str(path))
        require_mounted(value).mkdir(parents=True, exist_ok=True)
        out[key] = value
    return out


def ray_spill_system_config() -> dict[str, str]:
    """``_system_config`` for ``ray.init`` that spills objects to ``$ML_RAY_SPILL_DIR``."""
    spill = os.environ.get(SPILL_ENV)
    if not spill:
        return {}
    require_mounted(spill).mkdir(parents=True, exist_ok=True)
    return {
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": spill}}
        )
    }
