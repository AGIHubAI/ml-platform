"""Domain constraints on which experiments may be run.

Constraints are deliberately **separate from the ML model**. The surrogate
answers "what do we expect to measure?"; constraints answer "is this
experiment allowed / feasible / worth paying for?" — knowledge that belongs to
the lab (safe operating ranges, budget, already-run experiments), changes on a
different cadence than the model, and must never be learned away. Keeping
them as small, named, composable predicates means a scientist can audit
exactly why a candidate was excluded (see the rejection summary), and the
same constraints apply regardless of which model/acquisition produced the
ranking.

A constraint is any object with ``name``, ``reason`` and
``__call__(frame) -> bool mask`` (True = keep). Constraints are applied
*before* ranking so that infeasible candidates never consume batch slots.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]


@runtime_checkable
class Constraint(Protocol):
    """A named feasibility predicate over a candidate frame."""

    @property
    def name(self) -> str:
        """Unique identifier used in rejection summaries."""
        ...

    @property
    def reason(self) -> str:
        """Human-readable explanation of why a candidate is rejected."""
        ...

    def __call__(self, frame: pd.DataFrame) -> BoolArray: ...


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c[:1] == "f" and c[1:].isdigit()]


@dataclass(frozen=True)
class FeatureBounds:
    """Every feature must lie within ``[lo, hi]`` (e.g. a safe operating range)."""

    lo: float
    hi: float
    columns: tuple[str, ...] | None = None
    name: str = "feature_bounds"

    @property
    def reason(self) -> str:
        return f"a feature lies outside [{self.lo}, {self.hi}]"

    def __call__(self, frame: pd.DataFrame) -> BoolArray:
        cols = list(self.columns) if self.columns else _feature_cols(frame)
        X = frame[cols].to_numpy(dtype=np.float64)
        return np.asarray(((self.lo <= X) & (self.hi >= X)).all(axis=1), dtype=bool)


@dataclass(frozen=True)
class MaxCost:
    """Experimental cost (column ``cost``) must not exceed ``max_cost``."""

    max_cost: float
    column: str = "cost"
    name: str = "max_cost"

    @property
    def reason(self) -> str:
        return f"{self.column} > {self.max_cost}"

    def __call__(self, frame: pd.DataFrame) -> BoolArray:
        if self.column not in frame.columns:
            raise KeyError(f"MaxCost needs a {self.column!r} column")
        return np.asarray(frame[self.column].to_numpy(dtype=np.float64) <= self.max_cost)


@dataclass(frozen=True, init=False)
class ExcludeIds:
    """Exclude candidates whose id is in ``ids`` (e.g. already measured)."""

    ids: frozenset[str]
    column: str = "candidate_id"
    name: str = "exclude_observed"
    why: str = "already observed"

    def __init__(
        self,
        ids: Iterable[str],
        column: str = "candidate_id",
        name: str = "exclude_observed",
        why: str = "already observed",
    ) -> None:
        object.__setattr__(self, "ids", frozenset(str(i) for i in ids))
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "why", why)

    @property
    def reason(self) -> str:
        return self.why

    def __call__(self, frame: pd.DataFrame) -> BoolArray:
        return np.asarray(~frame[self.column].astype(str).isin(self.ids).to_numpy(), dtype=bool)


@dataclass(frozen=True)
class AllOf:
    """Conjunction of constraints (itself a `Constraint`)."""

    constraints: tuple[Constraint, ...]
    name: str = "all_of"

    @property
    def reason(self) -> str:
        return "; ".join(c.reason for c in self.constraints)

    def __call__(self, frame: pd.DataFrame) -> BoolArray:
        mask = np.ones(len(frame), dtype=bool)
        for c in self.constraints:
            mask &= c(frame)
        return mask


def all_of(*constraints: Constraint) -> AllOf:
    return AllOf(tuple(constraints))


def describe(constraint: Constraint) -> dict[str, Any]:
    """JSON-able description of a constraint (for manifests / hashes)."""
    d: dict[str, Any] = {"name": constraint.name, "reason": constraint.reason}
    if isinstance(constraint, FeatureBounds):
        d.update(lo=constraint.lo, hi=constraint.hi)
    elif isinstance(constraint, MaxCost):
        d.update(max_cost=constraint.max_cost, column=constraint.column)
    elif isinstance(constraint, ExcludeIds):
        d.update(n_ids=len(constraint.ids))
    elif isinstance(constraint, AllOf):
        d["constraints"] = [describe(c) for c in constraint.constraints]
    return d


@dataclass
class ConstraintReport:
    """Result of applying constraints: kept rows + an auditable summary."""

    kept: pd.DataFrame
    n_input: int
    n_kept: int
    # candidates each constraint rejects, evaluated independently on the input
    rejected_by: dict[str, int] = field(default_factory=dict)
    # candidates rejected *only* by that constraint (its marginal effect)
    rejected_only_by: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "n_input": self.n_input,
            "n_kept": self.n_kept,
            "n_rejected": self.n_input - self.n_kept,
            "rejected_by": dict(self.rejected_by),
            "rejected_only_by": dict(self.rejected_only_by),
            "reasons": dict(self.reasons),
        }


def apply_constraints(frame: pd.DataFrame, constraints: Sequence[Constraint]) -> ConstraintReport:
    """Filter ``frame`` to rows satisfying every constraint.

    Constraint names must be unique. Each constraint is evaluated on the full
    input, so ``rejected_by`` counts are independent of ordering (a candidate
    violating two constraints is counted under both).
    """
    names = [c.name for c in constraints]
    if len(set(names)) != len(names):
        raise ValueError(f"constraint names must be unique: {names}")
    n = len(frame)
    masks = {c.name: np.asarray(c(frame), dtype=bool) for c in constraints}
    for name, m in masks.items():
        if m.shape != (n,):
            raise ValueError(f"constraint {name!r} returned mask of shape {m.shape}, expected {n}")
    keep = np.ones(n, dtype=bool)
    for m in masks.values():
        keep &= m
    n_fail = np.zeros(n, dtype=np.int64)
    for m in masks.values():
        n_fail += ~m
    return ConstraintReport(
        kept=frame.loc[keep].reset_index(drop=True),
        n_input=n,
        n_kept=int(keep.sum()),
        rejected_by={k: int((~m).sum()) for k, m in masks.items()},
        rejected_only_by={k: int(((~m) & (n_fail == 1)).sum()) for k, m in masks.items()},
        reasons={c.name: c.reason for c in constraints},
    )
