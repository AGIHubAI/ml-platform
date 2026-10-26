"""Candidate-pool generation and (simulated) measurement of experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ml_platform.config import DataConfig, PlatformConfig
from ml_platform.data.schema import ExperimentRecord, feature_columns, utcnow
from ml_platform.data.synthetic_oracle import SyntheticOracle

FEATURE_CLIP = 3.0


def _data_cfg(cfg: PlatformConfig | DataConfig) -> DataConfig:
    return cfg.data if isinstance(cfg, PlatformConfig) else cfg


def make_oracle(cfg: PlatformConfig | DataConfig) -> SyntheticOracle:
    """The oracle for a config. Seeded by ``data.pool_seed``: the pool and the
    hidden system together define one "scientific problem"."""
    d = _data_cfg(cfg)
    return SyntheticOracle(seed=d.pool_seed, n_features=d.n_features)


def candidate_id(i: int) -> str:
    return f"cand_{i:06d}"


def generate_candidate_pool(cfg: PlatformConfig | DataConfig) -> pd.DataFrame:
    """Deterministic candidate pool: columns ``candidate_id, f00..f31, cost``.

    Features are standard normal, clipped to [-3, 3] (a bounded design space).
    """
    d = _data_cfg(cfg)
    rng = np.random.default_rng(np.random.SeedSequence([d.pool_seed, 0x9001]))
    X = np.clip(rng.standard_normal((d.pool_size, d.n_features)), -FEATURE_CLIP, FEATURE_CLIP)
    oracle = make_oracle(d)
    cols = feature_columns(d.n_features)
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "candidate_id", [candidate_id(i) for i in range(d.pool_size)])
    df["cost"] = oracle.cost(X)
    return df


def pool_features(pool: pd.DataFrame, n_features: int | None = None) -> np.ndarray:
    cols = (
        feature_columns(n_features)
        if n_features is not None
        else [c for c in pool.columns if c.startswith("f") and c[1:].isdigit()]
    )
    return pool[cols].to_numpy(dtype=np.float64)


def measure_candidates(
    candidates: pd.DataFrame,
    oracle: SyntheticOracle,
    round_id: int,
    seed: int,
    provenance: dict[str, Any] | None = None,
) -> list[ExperimentRecord]:
    """Run the synthetic experiment on ``candidates`` (rows of the pool).

    The measurement RNG is derived from ``(oracle.seed, round_id, seed)`` so that
    *re-running the same experimental round* reproduces the same noisy values —
    this is what makes round materialization idempotent on retry.
    """
    X = pool_features(candidates, oracle.n_features)
    rng = np.random.default_rng(np.random.SeedSequence([oracle.seed, round_id, seed, 0x3EA5]))
    y, std = oracle.measure(X, rng)
    created = utcnow()
    base = {"oracle_seed": oracle.seed, "measurement_seed": seed, **(provenance or {})}
    return [
        ExperimentRecord(
            experiment_id=str(cid),
            round_id=round_id,
            features=[float(v) for v in x],
            response=float(yi),
            measurement_std=float(si),
            status="measured",
            created_at=created,
            provenance=dict(base),
        )
        for cid, x, yi, si in zip(candidates["candidate_id"], X, y, std, strict=True)
    ]


def initial_observations(
    pool: pd.DataFrame, oracle: SyntheticOracle, n: int, seed: int
) -> list[ExperimentRecord]:
    """Round 0: ``n`` candidates drawn uniformly at random (seeded) and measured."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1417]))
    idx = np.sort(rng.choice(len(pool), size=n, replace=False))
    return measure_candidates(
        pool.iloc[idx],
        oracle,
        round_id=0,
        seed=seed,
        provenance={"source": "initial_random_design", "selection_seed": seed},
    )


def observed_ids(records: Iterable[ExperimentRecord]) -> set[str]:
    return {r.experiment_id for r in records}
