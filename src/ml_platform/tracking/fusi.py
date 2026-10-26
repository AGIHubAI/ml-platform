"""MLflow tracking + registry for the fUSI decoder (own experiment and registered model).

Same lifecycle as the surrogate (`ModelRegistry`: candidate -> validated -> production
only through ``promote_if_passed``), different names (``fusi.tracking``), and a pyfunc
wrapper that carries the decoder *and* the preprocessor it was trained with, because
a decoder is only valid on features produced by its own frozen preprocessing state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

from ml_platform.config import PlatformConfig, TrackingConfig
from ml_platform.tracking.mlflow_client import Tracker
from ml_platform.tracking.registry import ModelRegistry


def fusi_tracking_config(cfg: PlatformConfig) -> TrackingConfig:
    return TrackingConfig(
        tracking_uri=cfg.tracking.tracking_uri,
        experiment=cfg.fusi.tracking.experiment,
        registered_model=cfg.fusi.tracking.registered_model,
    )


def fusi_tracker(cfg: PlatformConfig, tracking_uri: str | None = None) -> Tracker:
    return Tracker(fusi_tracking_config(cfg), tracking_uri=tracking_uri)


def fusi_registry(cfg: PlatformConfig, tracker: Tracker | None = None) -> ModelRegistry:
    return ModelRegistry(tracker=tracker or fusi_tracker(cfg))


class FusiDecoderPyfunc(PythonModel):  # type: ignore[misc]
    """Input: one row per *preprocessed* frame (flattened voxels, time order).

    Output: ``p0..p{C-1}`` class probabilities per frame (causal, as streamed).
    """

    def load_context(self, context: Any) -> None:
        from ml_platform.workloads.fusi_realtime.models import load_any

        self.kind, self.model = load_any(str(Path(context.artifacts["checkpoint"])))

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        from ml_platform.workloads.fusi_realtime.models import PcaLdaDecoder, softmax_np
        from ml_platform.workloads.fusi_realtime.train import predict_session

        x = model_input.to_numpy(dtype=np.float32)
        if isinstance(self.model, PcaLdaDecoder):
            feats = PcaLdaDecoder.frame_features(x, self.model.n_frames)
            probs = softmax_np(self.model.decision(feats))
        else:
            grid = tuple(self.model.spec()["grid"])
            probs = predict_session(self.model, x.reshape(len(x), *grid))
        return pd.DataFrame(probs, columns=[f"p{i}" for i in range(probs.shape[1])])


def decoder_signature(grid: tuple[int, int, int], n_classes: int) -> tuple[Any, pd.DataFrame]:
    n = int(np.prod(grid))
    example = pd.DataFrame(np.zeros((2, n), dtype=np.float32), columns=[f"v{i}" for i in range(n)])
    out = pd.DataFrame(
        np.full((2, n_classes), 1.0 / n_classes), columns=[f"p{i}" for i in range(n_classes)]
    )
    return infer_signature(example, out), example


def register_decoder(
    registry: ModelRegistry,
    run_id: str,
    checkpoint_dir: str | os.PathLike[str],
    *,
    grid: tuple[int, int, int],
    n_classes: int,
    tags: dict[str, Any] | None = None,
) -> str:
    return registry.register(
        run_id,
        checkpoint_dir,
        tags=tags,
        python_model=FusiDecoderPyfunc(),
        signature=decoder_signature(grid, n_classes),
    )
