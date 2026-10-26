"""Record and manifest schemas for experimental data.

An `ExperimentRecord` is one (planned or measured) experiment. Experiments are
persisted in immutable *rounds*; each round has a `RoundManifest` whose
``parent_hashes`` link it to the previous round's manifest, forming a
hash chain (Merkle-style lineage) over the whole experimental history.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ml_platform.hashing import hash_config

Status = Literal["candidate", "selected", "measured", "failed"]
STATUSES: tuple[str, ...] = ("candidate", "selected", "measured", "failed")

SCHEMA_VERSION = 1


def feature_columns(n_features: int) -> list[str]:
    """Canonical feature column names: ``f00 .. f{n-1}``."""
    width = max(2, len(str(n_features - 1)))
    return [f"f{i:0{width}d}" for i in range(n_features)]


def round_key(round_id: int) -> str:
    """Canonical string key of a round (also the Dagster partition key): ``round_003``."""
    return f"round_{round_id:03d}"


def parse_round_key(key: str) -> int:
    if not key.startswith("round_"):
        raise ValueError(f"not a round key: {key!r}")
    return int(key.removeprefix("round_"))


def utcnow() -> datetime:
    return datetime.now(UTC)


class ExperimentRecord(BaseModel):
    """One experiment. ``experiment_id`` equals the candidate id it was drawn from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    features: list[float] = Field(min_length=1)
    response: float | None = None
    measurement_std: float | None = None
    status: Status
    created_at: datetime = Field(default_factory=utcnow)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("features")
    @classmethod
    def _finite_features(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("features must be finite")
        return v

    @model_validator(mode="after")
    def _status_consistency(self) -> ExperimentRecord:
        if self.status == "measured":
            if self.response is None or not math.isfinite(self.response):
                raise ValueError("measured records need a finite response")
            if self.measurement_std is not None and not (
                math.isfinite(self.measurement_std) and self.measurement_std >= 0
            ):
                raise ValueError("measurement_std must be finite and >= 0")
        elif self.response is not None and not math.isfinite(self.response):
            raise ValueError("response must be finite or None")
        return self


class RoundManifest(BaseModel):
    """Metadata of one immutable experimental round.

    ``dataset_hash`` is the content hash of this round's observations
    (scientific content only: ids, round, features, response, std, status —
    ``created_at``/``provenance`` are metadata and excluded so that a retried
    materialization of the *same* measurements hashes identically).
    ``parent_hashes`` holds the `manifest_hash` of the previous round (empty for
    round 0), chaining rounds together.
    """

    model_config = ConfigDict(extra="forbid")

    round_id: int = Field(ge=0)
    n_records: int = Field(ge=0)
    dataset_hash: str
    parent_hashes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    provenance: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def round_key(self) -> str:
        return round_key(self.round_id)

    @property
    def manifest_hash(self) -> str:
        """Hash of the content-bearing manifest fields (the lineage chain link)."""
        return hash_config(
            {
                "round_id": self.round_id,
                "n_records": self.n_records,
                "dataset_hash": self.dataset_hash,
                "parent_hashes": list(self.parent_hashes),
                "schema_version": self.schema_version,
            }
        )
