"""Feature / target standardization fitted on the training split only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_EPS = 1e-8


@dataclass(frozen=True)
class Normalizer:
    x_mean: NDArray[np.float64]
    x_std: NDArray[np.float64]
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X: ArrayLike, y: ArrayLike) -> Normalizer:
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        x_std = Xa.std(axis=0)
        y_std = float(ya.std())
        return cls(
            x_mean=Xa.mean(axis=0),
            x_std=np.where(x_std < _EPS, 1.0, x_std),
            y_mean=float(ya.mean()),
            y_std=y_std if y_std > _EPS else 1.0,
        )

    @classmethod
    def identity(cls, n_features: int) -> Normalizer:
        return cls(np.zeros(n_features), np.ones(n_features), 0.0, 1.0)

    @property
    def n_features(self) -> int:
        return int(self.x_mean.shape[0])

    def transform_x(self, X: ArrayLike) -> NDArray[np.float32]:
        return ((np.asarray(X, dtype=np.float64) - self.x_mean) / self.x_std).astype(np.float32)

    def transform_y(self, y: ArrayLike) -> NDArray[np.float32]:
        return ((np.asarray(y, dtype=np.float64) - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y_std_units: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(y_std_units, dtype=np.float64) * self.y_std + self.y_mean

    def inverse_scale(self, sigma_std_units: ArrayLike) -> NDArray[np.float64]:
        """De-standardize a standard deviation (scale only, no shift)."""
        return np.asarray(sigma_std_units, dtype=np.float64) * self.y_std

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_mean": [float(v) for v in self.x_mean],
            "x_std": [float(v) for v in self.x_std],
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Normalizer:
        return cls(
            x_mean=np.asarray(d["x_mean"], dtype=np.float64),
            x_std=np.asarray(d["x_std"], dtype=np.float64),
            y_mean=float(d["y_mean"]),
            y_std=float(d["y_std"]),
        )
