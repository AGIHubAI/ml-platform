"""Immutable experimental-round store and training-dataset construction.

Layout under ``root`` (normally ``cfg.paths.data_dir``)::

    candidate_pool/{pool.parquet, manifest.json}
    rounds/round_000/{observations.parquet, manifest.json}
    rounds/round_001/...

Rounds are **write-once**: writing a round that already exists with identical
scientific content is an idempotent no-op (safe for orchestration retries);
writing *different* content raises `ImmutableRoundError`. Files are made
read-only after writing. The effective training set is always the union of
rounds ``0..N``, which gives point-in-time reproducibility: "the data model
v7 was trained on" is exactly ``training_frame(up_to_round=k)``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml_platform.data.schema import (
    ExperimentRecord,
    RoundManifest,
    feature_columns,
    round_key,
    utcnow,
)
from ml_platform.data.validation import validate_frame
from ml_platform.hashing import hash_config, hash_dataframe

OBSERVATIONS_FILE = "observations.parquet"
MANIFEST_FILE = "manifest.json"
POOL_FILE = "pool.parquet"

# Columns that define the scientific content of a round (see RoundManifest docs).
_META_COLUMNS = ("created_at", "provenance")


class ImmutableRoundError(RuntimeError):
    """Raised when attempting to overwrite an existing round with different content."""


class RoundSequenceError(ValueError):
    """Raised when a round would break the contiguous, duplicate-free round chain."""


# --------------------------------------------------------------------------- frames
def records_to_frame(
    records: Sequence[ExperimentRecord], n_features: int | None = None
) -> pd.DataFrame:
    """Records -> canonical flat frame (features expanded to ``f00..``), sorted by id."""
    if not records:
        if n_features is None:
            raise ValueError("n_features required for an empty record list")
        cols = feature_columns(n_features)
        empty = pd.DataFrame(
            {
                "experiment_id": pd.Series(dtype=object),
                "round_id": pd.Series(dtype="int64"),
                **{c: pd.Series(dtype="float64") for c in cols},
                "response": pd.Series(dtype="float64"),
                "measurement_std": pd.Series(dtype="float64"),
                "status": pd.Series(dtype=object),
                "created_at": pd.Series(dtype="datetime64[us, UTC]"),
                "provenance": pd.Series(dtype=object),
            }
        )
        return empty
    n = n_features or len(records[0].features)
    if any(len(r.features) != n for r in records):
        raise ValueError(f"all records must have {n} features")
    cols = feature_columns(n)
    X = np.asarray([r.features for r in records], dtype=np.float64)
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "experiment_id", [r.experiment_id for r in records])
    df.insert(1, "round_id", np.asarray([r.round_id for r in records], dtype=np.int64))
    df["response"] = np.asarray(
        [np.nan if r.response is None else r.response for r in records], dtype=np.float64
    )
    df["measurement_std"] = np.asarray(
        [np.nan if r.measurement_std is None else r.measurement_std for r in records],
        dtype=np.float64,
    )
    df["status"] = [r.status for r in records]
    df["created_at"] = pd.to_datetime([r.created_at for r in records], utc=True)
    df["provenance"] = [json.dumps(r.provenance, sort_keys=True, default=str) for r in records]
    return df.sort_values("experiment_id", kind="mergesort").reset_index(drop=True)


def frame_to_records(df: pd.DataFrame) -> list[ExperimentRecord]:
    fcols = [c for c in df.columns if c[:1] == "f" and c[1:].isdigit()]
    out = []
    for d in df.to_dict(orient="records"):
        out.append(
            ExperimentRecord(
                experiment_id=str(d["experiment_id"]),
                round_id=int(d["round_id"]),
                features=[float(d[c]) for c in fcols],
                response=None if pd.isna(d["response"]) else float(d["response"]),
                measurement_std=None
                if pd.isna(d["measurement_std"])
                else float(d["measurement_std"]),
                status=d["status"],
                created_at=pd.Timestamp(d["created_at"]).to_pydatetime(),
                provenance=json.loads(d["provenance"]) if d["provenance"] else {},
            )
        )
    return out


def content_hash(frame: pd.DataFrame) -> str:
    """Content hash of observations (excludes created_at/provenance, row-order invariant)."""
    content = frame.drop(columns=[c for c in _META_COLUMNS if c in frame.columns])
    return hash_dataframe(content, sort_rows_by="experiment_id")


# --------------------------------------------------------------------------- store
def _make_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


class RoundStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.rounds_dir = self.root / "rounds"
        self.pool_dir = self.root / "candidate_pool"

    def __repr__(self) -> str:
        return f"RoundStore({str(self.root)!r})"

    # ---------------------------------------------------------------- paths
    def round_dir(self, round_id: int) -> Path:
        return self.rounds_dir / round_key(round_id)

    def exists(self, round_id: int) -> bool:
        return (self.round_dir(round_id) / MANIFEST_FILE).exists()

    # ---------------------------------------------------------------- rounds
    def list_rounds(self) -> list[int]:
        if not self.rounds_dir.exists():
            return []
        ids = []
        for p in self.rounds_dir.iterdir():
            if p.is_dir() and p.name.startswith("round_") and (p / MANIFEST_FILE).exists():
                ids.append(int(p.name.removeprefix("round_")))
        return sorted(ids)

    def latest_round_id(self) -> int | None:
        rounds = self.list_rounds()
        return rounds[-1] if rounds else None

    def read_manifest(self, round_id: int) -> RoundManifest:
        path = self.round_dir(round_id) / MANIFEST_FILE
        if not path.exists():
            raise FileNotFoundError(f"round {round_id} does not exist in {self.root}")
        return RoundManifest.model_validate_json(path.read_text())

    def read_round(self, round_id: int) -> pd.DataFrame:
        path = self.round_dir(round_id) / OBSERVATIONS_FILE
        if not path.exists():
            raise FileNotFoundError(f"round {round_id} does not exist in {self.root}")
        return pd.read_parquet(path)

    def read_records(self, round_id: int) -> list[ExperimentRecord]:
        return frame_to_records(self.read_round(round_id))

    def write_round(
        self,
        round_id: int,
        records: Sequence[ExperimentRecord] | pd.DataFrame,
        provenance: dict[str, Any] | None = None,
    ) -> RoundManifest:
        """Persist a round (write-once, idempotent for identical content)."""
        if isinstance(records, pd.DataFrame):
            frame = records.sort_values("experiment_id", kind="mergesort").reset_index(drop=True)
        else:
            frame = records_to_frame(list(records))
        if len(frame) and not (frame["round_id"] == round_id).all():
            raise ValueError(f"all records must have round_id={round_id}")
        validate_frame(frame, kind="observations").raise_if_failed()
        digest = content_hash(frame)

        if self.exists(round_id):
            existing = self.read_manifest(round_id)
            if existing.dataset_hash == digest:
                return existing
            raise ImmutableRoundError(
                f"{round_key(round_id)} already exists with different content "
                f"(existing {existing.dataset_hash[:12]}, new {digest[:12]}); "
                "rounds are immutable — write a new round instead"
            )

        latest = self.latest_round_id()
        expected = 0 if latest is None else latest + 1
        if round_id != expected:
            raise RoundSequenceError(f"next round must be {expected}, got {round_id}")
        parents: list[str] = []
        if round_id > 0:
            parents = [self.read_manifest(round_id - 1).manifest_hash]
            overlap = set(frame["experiment_id"]) & self.observed_ids()
            if overlap:
                raise RoundSequenceError(
                    f"{len(overlap)} experiment ids already present in earlier rounds, "
                    f"e.g. {sorted(overlap)[:3]}"
                )

        manifest = RoundManifest(
            round_id=round_id,
            n_records=len(frame),
            dataset_hash=digest,
            parent_hashes=parents,
            created_at=utcnow(),
            provenance=provenance or {},
        )
        self._atomic_write_dir(
            self.round_dir(round_id),
            {
                OBSERVATIONS_FILE: lambda p: frame.to_parquet(p, index=False),
                MANIFEST_FILE: lambda p: p.write_text(manifest.model_dump_json(indent=2)),
            },
        )
        # If a concurrent writer won the race, fall back to the idempotency check.
        on_disk = self.read_manifest(round_id)
        if on_disk.dataset_hash != digest:
            raise ImmutableRoundError(f"{round_key(round_id)} was written concurrently")
        return on_disk

    def _atomic_write_dir(self, final: Path, writers: dict[str, Any]) -> None:
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.parent / f".tmp_{final.name}_{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            for name, write in writers.items():
                write(tmp / name)
                _make_read_only(tmp / name)
            try:
                tmp.rename(final)
            except OSError:
                if not (final / MANIFEST_FILE).exists():
                    raise
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    # ---------------------------------------------------------------- unions
    def _rounds_up_to(self, up_to_round: int | None) -> list[int]:
        rounds = self.list_rounds()
        if up_to_round is None:
            return rounds
        if up_to_round not in rounds:
            raise FileNotFoundError(f"round {up_to_round} does not exist")
        return [r for r in rounds if r <= up_to_round]

    def all_observations(self, up_to_round: int | None = None) -> pd.DataFrame:
        frames = [self.read_round(r) for r in self._rounds_up_to(up_to_round)]
        if not frames:
            raise FileNotFoundError(f"no rounds in {self.root}")
        return pd.concat(frames, ignore_index=True)

    def training_frame(self, up_to_round: int | None = None) -> pd.DataFrame:
        """Union of rounds ``0..up_to_round`` (default: latest), measured rows only."""
        df = self.all_observations(up_to_round)
        df = df[df["status"] == "measured"]
        return df.sort_values(["round_id", "experiment_id"], kind="mergesort").reset_index(
            drop=True
        )

    def observed_ids(self, up_to_round: int | None = None) -> set[str]:
        if not self.list_rounds():
            return set()
        return set(self.all_observations(up_to_round)["experiment_id"].astype(str))

    def dataset_hash(self, up_to_round: int | None = None) -> str:
        """Identifier of the training dataset ``union(round_000..round_N)``.

        It is the manifest hash of round N, which (via ``parent_hashes``)
        transitively commits to the content of every earlier round.
        """
        rounds = self._rounds_up_to(up_to_round)
        if not rounds:
            raise FileNotFoundError(f"no rounds in {self.root}")
        return self.read_manifest(rounds[-1]).manifest_hash

    def verify_chain(self) -> bool:
        """Re-hash every round and check the parent links. Raises on corruption."""
        prev: RoundManifest | None = None
        for r in self.list_rounds():
            m = self.read_manifest(r)
            if content_hash(self.read_round(r)) != m.dataset_hash:
                raise ImmutableRoundError(f"{round_key(r)} content does not match its manifest")
            expected = [prev.manifest_hash] if prev is not None else []
            if m.parent_hashes != expected:
                raise ImmutableRoundError(f"{round_key(r)} has a broken parent link")
            prev = m
        return True

    # ---------------------------------------------------------------- candidate pool
    def write_pool(self, pool: pd.DataFrame, provenance: dict[str, Any] | None = None) -> dict:
        """Persist the candidate pool (write-once, idempotent for identical content)."""
        validate_frame(pool, kind="pool").raise_if_failed()
        digest = hash_dataframe(pool.reset_index(drop=True))
        manifest_path = self.pool_dir / MANIFEST_FILE
        if manifest_path.exists():
            existing: dict[str, Any] = json.loads(manifest_path.read_text())
            if existing["pool_hash"] == digest:
                return existing
            raise ImmutableRoundError("candidate pool already exists with different content")
        manifest = {
            "pool_hash": digest,
            "n_candidates": len(pool),
            "n_features": sum(1 for c in pool.columns if c[:1] == "f" and c[1:].isdigit()),
            "created_at": utcnow().isoformat(),
            "provenance": provenance or {},
        }
        manifest["manifest_hash"] = hash_config(
            {k: manifest[k] for k in ("pool_hash", "n_candidates", "n_features")}
        )
        self._atomic_write_dir(
            self.pool_dir,
            {
                POOL_FILE: lambda p: pool.to_parquet(p, index=False),
                MANIFEST_FILE: lambda p: p.write_text(json.dumps(manifest, indent=2)),
            },
        )
        return json.loads(manifest_path.read_text())

    def has_pool(self) -> bool:
        return (self.pool_dir / MANIFEST_FILE).exists()

    def read_pool(self) -> pd.DataFrame:
        return pd.read_parquet(self.pool_dir / POOL_FILE)

    def pool_manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads((self.pool_dir / MANIFEST_FILE).read_text())
        return result


# --------------------------------------------------------------------------- torch datasets
class ArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory (X, y) dataset in *raw* units; the trainer standardizes it.

    Carries ids/round ids and the dataset hash for lineage.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ids: Sequence[str] | None = None,
        round_ids: np.ndarray | None = None,
        dataset_hash: str | None = None,
    ) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
        if len(self.X) != len(self.y):
            raise ValueError("X and y length mismatch")
        self.ids = list(ids) if ids is not None else [str(i) for i in range(len(self.y))]
        self.round_ids = (
            np.asarray(round_ids, dtype=np.int64)
            if round_ids is not None
            else np.zeros(len(self.y), dtype=np.int64)
        )
        self.dataset_hash = dataset_hash
        # standardized tensors are attached by the trainer (see `set_normalized`)
        self._Xt = torch.from_numpy(self.X.astype(np.float32))
        self._yt = torch.from_numpy(self.y.astype(np.float32))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, dataset_hash: str | None = None) -> ArrayDataset:
        fcols = [c for c in frame.columns if c[:1] == "f" and c[1:].isdigit()]
        return cls(
            frame[fcols].to_numpy(dtype=np.float64),
            frame["response"].to_numpy(dtype=np.float64),
            ids=frame["experiment_id"].astype(str).tolist()
            if "experiment_id" in frame.columns
            else None,
            round_ids=frame["round_id"].to_numpy() if "round_id" in frame.columns else None,
            dataset_hash=dataset_hash,
        )

    def set_normalized(self, Xn: np.ndarray, yn: np.ndarray) -> None:
        self._Xt = torch.from_numpy(np.ascontiguousarray(Xn, dtype=np.float32))
        self._yt = torch.from_numpy(np.ascontiguousarray(yn, dtype=np.float32))

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._Xt[idx], self._yt[idx]


ValStrategy = Literal["hash", "random", "newest_round"]


def _hash_unit_interval(ids: Sequence[str], seed: int) -> np.ndarray:
    """Map each id to a stable pseudo-uniform number in [0, 1) (seeded)."""
    out = np.empty(len(ids), dtype=np.float64)
    for i, eid in enumerate(ids):
        digest = hashlib.sha256(f"{seed}:{eid}".encode()).digest()
        out[i] = int.from_bytes(digest[:8], "big") / 2**64
    return out


def train_val_split(
    frame: pd.DataFrame,
    val_fraction: float,
    seed: int,
    strategy: ValStrategy = "hash",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train/validation split.

    * ``hash`` (default): an experiment is in validation iff
      ``sha256(seed:experiment_id)`` maps below ``val_fraction``. The
      assignment of an experiment **never changes as rounds are added**, so a
      model trained on rounds 0..k never saw any validation row used to
      evaluate the round k+1 model — required for a fair paired comparison
      against the incumbent model. (A per-round random re-split would leak the
      incumbent's training rows into the new validation set and make the
      incumbent look far better than it is.) The validation fraction is
      approximate (binomial).
    * ``random``: seeded permutation of rows ordered by ``experiment_id``;
      exact fraction, but the assignment changes when rows are added (do not
      use it to compare models across rounds).
    * ``newest_round``: hold out the newest round (a *prospective* check: "does
      the model trained on the past predict the next batch?"). Falls back to
      ``hash`` when only one round exists. Active-learning rounds are
      deliberately off-distribution, so this metric is harsher.

    Caveat — batch-effect leakage: in a real lab, measurements taken in the same
    round share batch effects (reagent lot, instrument calibration, day). A
    row-level split puts same-batch rows on both sides and overstates accuracy;
    grouping by round (``newest_round``) is the honest choice there. The
    synthetic oracle has no batch effects, so row-level splits are unbiased here.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    ordered = frame.sort_values("experiment_id", kind="mergesort").reset_index(drop=True)
    if strategy == "newest_round" and ordered["round_id"].nunique() > 1:
        newest = ordered["round_id"].max()
        mask = (ordered["round_id"] == newest).to_numpy()
    elif strategy == "random":
        n_val = max(1, round(len(ordered) * val_fraction))
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5B17]))
        perm = rng.permutation(len(ordered))
        mask = np.zeros(len(ordered), dtype=bool)
        mask[perm[:n_val]] = True
    else:
        u = _hash_unit_interval(ordered["experiment_id"].astype(str).tolist(), seed)
        mask = u < val_fraction
        if not mask.any() or mask.all():  # degenerate tiny frames
            mask = np.zeros(len(ordered), dtype=bool)
            mask[int(np.argmin(u))] = True
    return ordered[~mask].reset_index(drop=True), ordered[mask].reset_index(drop=True)
