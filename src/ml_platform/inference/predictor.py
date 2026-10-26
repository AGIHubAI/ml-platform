"""Framework-free model wrapper used by evaluation, batch inference and serving.

`Predictor` loads a checkpoint (model weights + training normalizer) and
returns predictions in *raw response units*.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn

from ml_platform.data.normalization import Normalizer
from ml_platform.data.schema import feature_columns
from ml_platform.models.mlp import ResidualMLP, count_parameters, model_from_spec
from ml_platform.models.uncertainty import mc_dropout_predict
from ml_platform.training.checkpointing import (
    checkpoint_hash,
    load_checkpoint,
    resolve_checkpoint,
)


class Predictor:
    def __init__(
        self,
        model: nn.Module,
        normalizer: Normalizer,
        *,
        device: str | torch.device = "cpu",
        mc_samples: int = 30,
        residual_std: float | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.normalizer = normalizer
        self.mc_samples = mc_samples
        # homoscedastic noise estimate (standardized units) used as aleatoric
        # term for models trained with MSE (they have no variance head)
        self.residual_std = residual_std
        self.feature_columns = feature_columns(normalizer.n_features)
        self._info = dict(info or {})

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_checkpoint(
        cls, path: str | os.PathLike[str], device: str | torch.device = "cpu"
    ) -> Predictor:
        file = resolve_checkpoint(path)
        payload = load_checkpoint(file, map_location=device)
        model = model_from_spec(payload["model_spec"])
        model.load_state_dict(payload["model_state"])
        cfg = payload.get("config", {})
        metrics = payload.get("metrics", {})
        info = {
            "checkpoint_path": str(file.parent),
            "checkpoint_hash": checkpoint_hash(file),
            "epochs_trained": int(payload["epoch"]),
            "n_parameters": count_parameters(model),
            "model_spec": payload["model_spec"],
            "dataset_hash": payload.get("dataset_hash"),
            "seed": payload.get("seed"),
            "trained_world_size": payload.get("world_size"),
            "val_metrics": {k: float(v) for k, v in metrics.items()},
        }
        return cls(
            model,
            Normalizer.from_dict(payload["normalizer"]),
            device=device,
            mc_samples=int(cfg.get("evaluation", {}).get("mc_samples", 30)),
            residual_std=float(metrics["val_rmse"]) if "val_rmse" in metrics else None,
            info=info,
        )

    # ------------------------------------------------------------------ helpers
    def _to_array(self, X: ArrayLike | pd.DataFrame) -> NDArray[np.float64]:
        if isinstance(X, pd.DataFrame):
            return X[self.feature_columns].to_numpy(dtype=np.float64)
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.normalizer.n_features:
            raise ValueError(f"expected {self.normalizer.n_features} features, got {arr.shape[1]}")
        return arr

    @property
    def has_variance_head(self) -> bool:
        return isinstance(self.model, ResidualMLP) and self.model.predict_variance

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict(self, X: ArrayLike | pd.DataFrame, batch_size: int = 8192) -> NDArray[np.float64]:
        """Deterministic prediction (dropout off), raw units, shape ``(n,)``."""
        Xn = torch.from_numpy(self.normalizer.transform_x(self._to_array(X)))
        self.model.eval()
        outs = [
            self.model(Xn[s : s + batch_size].to(self.device)).double().cpu()
            for s in range(0, Xn.shape[0], batch_size)
        ]
        out = torch.cat(outs).numpy() if outs else np.zeros(0)
        return self.normalizer.inverse_y(out)

    def predict_with_uncertainty(
        self,
        X: ArrayLike | pd.DataFrame,
        n_samples: int | None = None,
        *,
        seed: int | None = 0,
        include_aleatoric: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """MC-dropout ``(mean, std)`` in raw units, shape ``(n,)`` each, std > 0.

        By default std is *epistemic* (what active learning should chase). With
        ``include_aleatoric=True`` the observation-noise estimate is added in
        quadrature, giving a predictive std suitable for NLL/interval coverage.
        """
        Xn = self.normalizer.transform_x(self._to_array(X))
        mean, std = mc_dropout_predict(
            self.model,
            Xn,
            n_samples or self.mc_samples,
            seed=seed,
            device=self.device,
            return_aleatoric=include_aleatoric,
        )
        if include_aleatoric and not self.has_variance_head and self.residual_std is not None:
            std = np.sqrt(std**2 + self.residual_std**2)
        return self.normalizer.inverse_y(mean), self.normalizer.inverse_scale(std)

    def model_info(self) -> dict[str, Any]:
        return {**self._info, "device": str(self.device), "n_features": self.normalizer.n_features}


def load_predictor(path: str | Path, device: str = "cpu") -> Predictor:
    return Predictor.from_checkpoint(path, device=device)
