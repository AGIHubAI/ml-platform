"""Frame-level data validation (schema, finiteness, ranges, duplicates)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ml_platform.data.schema import STATUSES


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    n_rows: int
    kind: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> ValidationReport:
        if self.errors:
            raise DataValidationError("; ".join(self.errors))
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "n_rows": self.n_rows,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and c[:1] == "f" and c[1:].isdigit()]


def validate_frame(
    df: pd.DataFrame,
    *,
    kind: Literal["auto", "observations", "pool"] = "auto",
    n_features: int | None = None,
    feature_bounds: tuple[float, float] = (-3.0, 3.0),
    bounds_tolerance: float = 1e-9,
) -> ValidationReport:
    """Validate an observations frame (``experiment_id`` ...) or a pool frame (``candidate_id``)."""
    if kind == "auto":
        kind = "pool" if "candidate_id" in df.columns else "observations"
    rep = ValidationReport(n_rows=len(df), kind=kind)
    id_col = "candidate_id" if kind == "pool" else "experiment_id"
    required = [id_col] + (
        ["cost"] if kind == "pool" else ["round_id", "response", "measurement_std", "status"]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.errors.append(f"missing columns: {missing}")

    fcols = _feature_cols(df)
    if not fcols:
        rep.errors.append("no feature columns (f00..)")
    elif n_features is not None and len(fcols) != n_features:
        rep.errors.append(f"expected {n_features} feature columns, found {len(fcols)}")
    if len(df) == 0:
        rep.warnings.append("empty frame")
        return rep

    if fcols:
        X = df[fcols].to_numpy(dtype=np.float64, na_value=np.nan)
        n_bad = int((~np.isfinite(X)).any(axis=1).sum())
        if n_bad:
            rep.errors.append(f"{n_bad} rows with non-finite features")
        lo, hi = feature_bounds
        finite = np.where(np.isfinite(X), X, 0.0)
        n_oob = int(
            ((finite < lo - bounds_tolerance) | (finite > hi + bounds_tolerance)).any(axis=1).sum()
        )
        if n_oob:
            rep.errors.append(f"{n_oob} rows with features outside [{lo}, {hi}]")

    if id_col in df.columns:
        if df[id_col].isna().any():
            rep.errors.append(f"null {id_col}")
        n_dupe = int(df[id_col].duplicated().sum())
        if n_dupe:
            rep.errors.append(f"{n_dupe} duplicate {id_col} values")

    if kind == "pool" and "cost" in df.columns:
        cost = df["cost"].to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.all(np.isfinite(cost) & (cost > 0)):
            rep.errors.append("cost must be finite and > 0")

    if kind == "observations" and not missing:
        bad_status = sorted(set(df["status"].astype(str)) - set(STATUSES))
        if bad_status:
            rep.errors.append(f"invalid status values: {bad_status}")
        if (df["round_id"] < 0).any():
            rep.errors.append("negative round_id")
        measured = df["status"] == "measured"
        resp = df.loc[measured, "response"].to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.all(np.isfinite(resp)):
            rep.errors.append("measured rows must have finite response")
        std = df.loc[measured, "measurement_std"].to_numpy(dtype=np.float64, na_value=np.nan)
        std = std[~np.isnan(std)]
        if np.any(~np.isfinite(std) | (std < 0)):
            rep.errors.append("measurement_std must be finite and >= 0")
        if not measured.any():
            rep.warnings.append("no measured rows")
    return rep
