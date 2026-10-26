"""One active-learning decision step: predictions -> constraints -> ranking -> batch.

``select_batch`` is the pure, deterministic core of the ``selected_experiments``
asset. The pipeline is::

    pool ⋈ predictions  --constraints-->  feasible  --acquisition-->  ranked
        --(optional diversity)-->  top-k  =  next experimental batch

Domain constraints (``constraints.py``) are applied *before* ranking and are
derived from ``cfg.active_learning`` — they encode lab knowledge, not model
knowledge, and live apart from the surrogate.

Determinism & versioning: given the same inputs the selection is identical
(ties broken by ``candidate_id``), and ``selection_hash`` commits to the round,
the model version, the acquisition settings, the constraints and the selected
rows. Written with ``write_selection`` it becomes a versioned artifact
(``selected.parquet`` + ``selection.json``) that the oracle / lab consumes.

Selection bias (read before analysing results): the selected batches are, by
construction, *not* a random sample — they concentrate where the model
predicts high values or is uncertain. Consequences: (1) metrics computed on
actively-acquired rounds are not unbiased estimates of performance on the
pool; (2) the training distribution drifts toward promising regions, so the
model may become over-confident elsewhere; (3) greedy top-k batches tend to be
redundant (near-duplicates of one another, all exploiting the same mode). The
optional diversity step (``diversity_radius``) mitigates (3) by greedily
skipping candidates within a feature-space radius of an already-selected one,
at the price of a further, deliberate bias toward spread-out batches. The
random initial round (round 0) remains the unbiased reference sample.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_platform.active_learning.acquisition import Strategy, rank_candidates
from ml_platform.active_learning.constraints import (
    Constraint,
    ExcludeIds,
    FeatureBounds,
    MaxCost,
    apply_constraints,
    describe,
)
from ml_platform.config import ActiveLearningConfig, PlatformConfig
from ml_platform.hashing import hash_config, hash_dataframe

SELECTED_FILE = "selected.parquet"
SELECTION_MANIFEST = "selection.json"
SELECTION_FORMAT_VERSION = 1


@dataclass
class SelectionResult:
    """The next experimental batch plus its audit trail.

    ``selected`` columns: ``rank, candidate_id, pred_mean, pred_std, score,
    round_id, f00..f31[, cost]`` (features included so the oracle/lab can run
    the experiments without re-joining the pool).
    """

    selected: pd.DataFrame
    summary: dict[str, Any]
    selection_hash: str
    round_id: int
    model_version: str | None = None
    ranked: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def candidate_ids(self) -> list[str]:
        return self.selected["candidate_id"].astype(str).tolist()


def default_constraints(
    al_cfg: ActiveLearningConfig, observed_ids: Collection[str]
) -> list[Constraint]:
    """Constraints implied by ``cfg.active_learning`` plus duplicate exclusion."""
    lo, hi = al_cfg.feature_bounds
    cons: list[Constraint] = [ExcludeIds(observed_ids), FeatureBounds(float(lo), float(hi))]
    if al_cfg.max_cost is not None:
        cons.append(MaxCost(float(al_cfg.max_cost)))
    return cons


def _diverse_top_k(
    ranked: pd.DataFrame, X: np.ndarray, k: int, radius: float
) -> tuple[np.ndarray, int]:
    """Greedy batch diversification over rank order.

    Walk the ranking; accept a candidate unless it lies within ``radius``
    (Euclidean, raw feature space) of an already-accepted one. If fewer than
    ``k`` survive, fill the remainder with the best skipped candidates (so the
    batch size is always honoured). Returns (row indices, n_skipped).
    """
    chosen: list[int] = []
    skipped: list[int] = []
    for i in range(len(ranked)):
        if len(chosen) == k:
            break
        if chosen:
            d = np.sqrt(((X[chosen] - X[i]) ** 2).sum(axis=1)).min()
            if d < radius:
                skipped.append(i)
                continue
        chosen.append(i)
    n_skipped = len(skipped)
    if len(chosen) < k:
        chosen.extend(skipped[: k - len(chosen)])
    return np.sort(np.asarray(chosen, dtype=np.int64)), n_skipped


def select_batch(
    pool: pd.DataFrame,
    predictions: pd.DataFrame,
    observed_ids: Collection[str],
    cfg: PlatformConfig,
    *,
    round_id: int,
    model_version: str | None = None,
    batch_size: int | None = None,
    strategy: Strategy = "ucb",
    beta: float | None = None,
    best_observed: float | None = None,
    diversity_radius: float | None = None,
    extra_constraints: Sequence[Constraint] = (),
    seed: int | None = None,
) -> SelectionResult:
    """Choose the next experimental batch for ``round_id``.

    Args:
        pool: candidate pool (``candidate_id, f00.., cost``).
        predictions: ``candidate_id, pred_mean, pred_std`` (e.g. from
            ``inference.batch.predict_pool``); may cover a subset of the pool.
        observed_ids: ids already measured in earlier rounds (excluded).
        cfg: platform config; ``active_learning.{beta, feature_bounds,
            max_cost}`` and ``data.batch_size_per_round`` are used.
        round_id: the round these experiments will form.
        model_version: version of the model that produced ``predictions``.
        batch_size / beta: override the config values.
        strategy: acquisition function (default UCB, handoff §11).
        best_observed: incumbent best response (required for ``"ei"``).
        diversity_radius: if set, greedy feature-space de-duplication.
        extra_constraints: additional domain constraints.
        seed: seed for randomized strategies (default ``cfg.seed + round_id``).
    """
    k = int(batch_size if batch_size is not None else cfg.data.batch_size_per_round)
    beta_v = float(beta if beta is not None else cfg.active_learning.beta)
    seed_v = int(seed if seed is not None else cfg.seed + round_id)
    if k <= 0:
        raise ValueError("batch size must be positive")

    preds = predictions[["candidate_id", "pred_mean", "pred_std"]].copy()
    preds["candidate_id"] = preds["candidate_id"].astype(str)
    if preds["candidate_id"].duplicated().any():
        raise ValueError("predictions contain duplicate candidate ids")
    base = pool.copy()
    base["candidate_id"] = base["candidate_id"].astype(str)
    joined = base.merge(preds, on="candidate_id", how="inner", validate="one_to_one")
    joined = joined.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)

    constraints = [*default_constraints(cfg.active_learning, observed_ids), *extra_constraints]
    report = apply_constraints(joined, constraints)
    feasible = report.kept

    ranked = rank_candidates(
        feasible,
        feasible["pred_mean"].to_numpy(),
        feasible["pred_std"].to_numpy(),
        beta=beta_v,
        strategy=strategy,
        best=best_observed,
        seed=seed_v,
    )
    ranked_full = ranked.merge(
        feasible.drop(columns=["pred_mean", "pred_std"]), on="candidate_id", how="left"
    )

    n_div_skipped = 0
    if diversity_radius is not None and diversity_radius > 0 and len(ranked_full) > 0:
        fcols = [c for c in ranked_full.columns if c[:1] == "f" and c[1:].isdigit()]
        # only the head of the ranking can matter; bound the O(k * n) scan
        head = ranked_full.head(max(50 * k, 5000))
        idx, n_div_skipped = _diverse_top_k(
            head, head[fcols].to_numpy(dtype=np.float64), k, float(diversity_radius)
        )
        selected = head.iloc[idx].copy()
    else:
        selected = ranked_full.head(k).copy()

    selected["round_id"] = np.int64(round_id)
    fcols = [c for c in selected.columns if c[:1] == "f" and c[1:].isdigit()]
    extra = [c for c in ("cost",) if c in selected.columns]
    selected = selected[
        ["rank", "candidate_id", "pred_mean", "pred_std", "score", "round_id", *fcols, *extra]
    ].reset_index(drop=True)

    constraint_desc = [describe(c) for c in constraints]
    selection_hash = hash_config(
        {
            "format_version": SELECTION_FORMAT_VERSION,
            "round_id": round_id,
            "model_version": model_version,
            "strategy": strategy,
            "beta": beta_v,
            "seed": seed_v if strategy == "thompson" else None,
            "batch_size": k,
            "diversity_radius": diversity_radius,
            "constraints": constraint_desc,
            "selected": hash_dataframe(selected),
        }
    )

    def _stats(col: str) -> dict[str, float] | None:
        if selected.empty:
            return None
        v = selected[col].to_numpy(dtype=np.float64)
        return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}

    summary: dict[str, Any] = {
        "round_id": round_id,
        "model_version": model_version,
        "strategy": strategy,
        "beta": beta_v,
        "batch_size": k,
        "n_pool": len(pool),
        "n_predictions": len(preds),
        "n_scored": len(joined),
        "constraints": report.summary(),
        "constraint_specs": constraint_desc,
        "n_feasible": report.n_kept,
        "n_selected": len(selected),
        "diversity_radius": diversity_radius,
        "n_diversity_skipped": n_div_skipped,
        "score": _stats("score"),
        "pred_mean": _stats("pred_mean"),
        "pred_std": _stats("pred_std"),
        "pool_pred_mean": float(joined["pred_mean"].mean()) if len(joined) else None,
        "pool_pred_std": float(joined["pred_std"].mean()) if len(joined) else None,
    }
    if len(selected) < k:
        summary["warning"] = f"only {len(selected)} feasible candidates for batch size {k}"
    return SelectionResult(
        selected=selected,
        summary=summary,
        selection_hash=selection_hash,
        round_id=round_id,
        model_version=model_version,
        ranked=ranked,
    )


def write_selection(
    result: SelectionResult,
    out_dir: str | os.PathLike[str],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a selection as ``selected.parquet`` + ``selection.json``.

    The manifest carries the ``selection_hash`` (the artifact version), round,
    model version, constraint summary and ``selected_ids``. Rewriting the same
    selection is idempotent (same content, same hash). Returns ``out_dir``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{SELECTED_FILE}.tmp"
    result.selected.to_parquet(tmp, index=False)
    tmp.replace(out / SELECTED_FILE)
    manifest = {
        "format_version": SELECTION_FORMAT_VERSION,
        "selection_hash": result.selection_hash,
        "round_id": result.round_id,
        "model_version": result.model_version,
        "n_selected": len(result.selected),
        "selected_ids": result.candidate_ids,
        "summary": result.summary,
        **({"metadata": extra_metadata} if extra_metadata else {}),
    }
    tmp_m = out / f".{SELECTION_MANIFEST}.tmp"
    tmp_m.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    tmp_m.replace(out / SELECTION_MANIFEST)
    return out


def read_selection(out_dir: str | os.PathLike[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load ``(selected_frame, manifest)`` written by `write_selection`."""
    out = Path(out_dir)
    manifest: dict[str, Any] = json.loads((out / SELECTION_MANIFEST).read_text())
    return pd.read_parquet(out / SELECTED_FILE), manifest
