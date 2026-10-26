from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml_platform.config import PlatformConfig
from ml_platform.data import ArrayDataset
from ml_platform.inference import Predictor
from ml_platform.training import (
    SimulatedWorkerFailure,
    Trainer,
    TrainResult,
    latest_checkpoint,
    load_checkpoint,
    predict_array,
)


def _fresh(ds: tuple[ArrayDataset, ArrayDataset]) -> tuple[ArrayDataset, ArrayDataset]:
    tr, va = ds
    return (
        ArrayDataset(tr.X, tr.y, ids=tr.ids, dataset_hash=tr.dataset_hash),
        ArrayDataset(va.X, va.y, ids=va.ids, dataset_hash=va.dataset_hash),
    )


def test_training_reduces_loss_and_records_metadata(trained) -> None:
    result, tr, va = trained
    assert isinstance(result, TrainResult)
    losses = [h["train_loss"] for h in result.history]
    assert losses[-1] < losses[0] * 0.8
    assert result.epochs_completed == 5
    assert result.world_size == 1 and result.device == "cpu"
    assert result.seed == 0 and result.dataset_hash == "test-dataset"
    assert result.hyperparams["training.lr"] == pytest.approx(3e-3)
    assert result.checkpoint_path.exists() and result.checkpoint_hash
    assert result.n_train == len(tr) and result.n_val == len(va)
    assert {"val_rmse", "val_mae", "val_r2", "train_loss"} <= set(result.metrics)
    assert result.metrics["val_rmse"] < 1.0  # better than predicting the mean
    assert "cpu_threads" in result.device_info
    assert result.to_dict()["checkpoint_path"] == str(result.checkpoint_path)


def test_training_is_deterministic(tmp_path: Path, datasets, session_cfg: PlatformConfig) -> None:
    cfg = session_cfg.with_overrides(**{"training.epochs": 3})
    r1 = Trainer(cfg).train(*_fresh(datasets), checkpoint_dir=tmp_path / "a")
    r2 = Trainer(cfg).train(*_fresh(datasets), checkpoint_dir=tmp_path / "b")
    assert r1.history == r2.history
    s1 = load_checkpoint(r1.checkpoint_path)["model_state"]
    s2 = load_checkpoint(r2.checkpoint_path)["model_state"]
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    r3 = Trainer(cfg.with_overrides(seed=1)).train(*_fresh(datasets), checkpoint_dir=tmp_path / "c")
    assert r3.history != r1.history


def test_checkpoint_roundtrip_predictions_identical(trained) -> None:
    result, _, va = trained
    ckpt = load_checkpoint(result.checkpoint_path)
    assert set(ckpt) >= {
        "model_state",
        "optimizer_state",
        "epoch",
        "metrics",
        "normalizer",
        "config",
        "rng_state",
    }
    p1 = Predictor.from_checkpoint(result.checkpoint_path)
    p2 = Predictor.from_checkpoint(result.checkpoint_path.parent)  # via `latest`
    X = va.filter(regex=r"^f\d+$").to_numpy()
    np.testing.assert_array_equal(p1.predict(X), p2.predict(X))
    np.testing.assert_array_equal(p1.predict(va), p1.predict(X))  # DataFrame input


def test_trainer_predictions_match_predictor(tmp_path: Path, datasets, session_cfg) -> None:
    cfg = session_cfg.with_overrides(**{"training.epochs": 2})
    trainer = Trainer(cfg)
    result = trainer.train(*_fresh(datasets), checkpoint_dir=tmp_path)
    X = datasets[1].X
    np.testing.assert_allclose(
        predict_array(trainer, X),
        Predictor.from_checkpoint(result.checkpoint_path).predict(X),
        rtol=0,
        atol=1e-6,
    )


def test_fail_and_resume_matches_uninterrupted(tmp_path: Path, datasets, session_cfg) -> None:
    cfg = session_cfg.with_overrides(**{"training.epochs": 5})
    ckpt_dir = tmp_path / "ckpt"
    with pytest.raises(SimulatedWorkerFailure) as exc:
        Trainer(cfg).train(*_fresh(datasets), checkpoint_dir=ckpt_dir, fail_at_epoch=3)
    assert exc.value.epoch == 3
    latest = latest_checkpoint(ckpt_dir)
    assert latest is not None and latest.name == "epoch_0003"

    epochs_seen: list[int] = []
    resumed = Trainer(cfg).train(
        *_fresh(datasets),
        checkpoint_dir=ckpt_dir,
        resume_from=ckpt_dir,
        on_epoch_end=lambda e, m, p: epochs_seen.append(e),
    )
    assert epochs_seen == [4, 5]
    assert resumed.epochs_completed == 5 and resumed.resumed_from is not None
    assert len(resumed.history) == 5

    straight = Trainer(cfg).train(*_fresh(datasets), checkpoint_dir=tmp_path / "straight")
    assert resumed.history == straight.history
    assert resumed.metrics == straight.metrics


def test_resume_rejects_different_dataset(tmp_path: Path, datasets, session_cfg) -> None:
    cfg = session_cfg.with_overrides(**{"training.epochs": 2})
    Trainer(cfg).train(*_fresh(datasets), checkpoint_dir=tmp_path)
    tr, va = _fresh(datasets)
    tr.dataset_hash = "other-dataset"
    with pytest.raises(ValueError, match="refusing to resume"):
        Trainer(cfg).train(tr, va, checkpoint_dir=tmp_path / "x", resume_from=tmp_path)


def test_gaussian_nll_training(tmp_path: Path, datasets, session_cfg) -> None:
    cfg = session_cfg.with_overrides(**{"training.epochs": 3, "training.loss": "gaussian_nll"})
    result = Trainer(cfg).train(*_fresh(datasets), checkpoint_dir=tmp_path)
    p = Predictor.from_checkpoint(result.checkpoint_path)
    assert p.has_variance_head
    mu, sd = p.predict_with_uncertainty(datasets[1].X, 5, include_aleatoric=True)
    assert (sd > 0).all() and np.isfinite(mu).all()


def test_resume_with_different_world_size_reseeds_deterministically(
    tmp_path: Path, datasets, session_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint from 2 ranks resumed by 1 process: not bit-exact, but deterministic + warned."""
    cfg = session_cfg.with_overrides(**{"training.epochs": 4})
    ckpt_dir = tmp_path / "ckpt"
    with pytest.raises(SimulatedWorkerFailure):
        Trainer(cfg).train(*_fresh(datasets), checkpoint_dir=ckpt_dir, fail_at_epoch=2)
    model_pt = ckpt_dir / "epoch_0002" / "model.pt"
    payload = torch.load(model_pt, weights_only=True)
    assert len(payload["rng_states"]) == 1  # single process: one state
    payload["rng_states"] = [payload["rng_states"][0]] * 2  # pretend world_size=2
    payload["world_size"] = 2
    torch.save(payload, model_pt)

    warnings: list[dict] = []

    def run() -> TrainResult:
        trainer = Trainer(cfg)
        monkeypatch.setattr(
            trainer.log, "warning", lambda event, **kw: warnings.append({"event": event, **kw})
        )
        return trainer.train(*_fresh(datasets), checkpoint_dir=tmp_path / "r", resume_from=model_pt)

    a, b = run(), run()
    assert a.history == b.history
    assert [w["event"] for w in warnings] == ["training.resume_not_bit_exact"] * 2
    assert warnings[0]["checkpoint_world_size"] == 2 and warnings[0]["world_size"] == 1
