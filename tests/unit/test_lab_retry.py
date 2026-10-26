"""Caller-side retry of the (idempotent, journaled) lab measurement after an actor crash.

No Ray cluster: the actor handle and ``ray.get``/``ray.kill`` are faked so the
retry policy of ``RayCompute.run_experiments`` is exercised deterministically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError

from ml_platform.config import PlatformConfig
from ml_platform.orchestration.pipeline import RayCompute
from ml_platform.ray_runtime import tasks


class _Method:
    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def remote(self, *args: Any) -> Any:
        return lambda: self._fn(*args)  # a "ref" = deferred call, resolved by fake ray.get


class _FakeLab:
    """Stand-in actor handle; the journal (shared dict) survives 'restarts'."""

    def __init__(self, journal: dict[str, float], failures: list[type[Exception]]) -> None:
        self.journal, self.failures, self.physical = journal, failures, 0
        self.measure = _Method(self._measure)
        self.stats = _Method(lambda: {"n_physical_measurements": self.physical})

    def _measure(self, round_id: int, ids: list[str], seed: int) -> pd.DataFrame:
        if self.failures:
            exc = self.failures.pop(0)
            raise exc("lab restarting", None) if exc is ActorUnavailableError else exc()
        for c in ids:
            if c not in self.journal:
                self.journal[c] = float(len(self.journal))
                self.physical += 1
        return pd.DataFrame({"experiment_id": ids, "response": [self.journal[c] for c in ids]})


@pytest.mark.parametrize(
    ("failures", "n_actors"),
    [([ActorUnavailableError], 1), ([ActorDiedError], 2), ([], 1)],
)
def test_measure_is_retried_after_lab_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failures: list[type[Exception]],
    n_actors: int,
) -> None:
    journal: dict[str, float] = {"c0": 42.0}  # measured before the crash
    pending = list(failures)
    created: list[_FakeLab] = []

    def start(*_a: Any, **_k: Any) -> _FakeLab:
        lab = _FakeLab(journal, pending)
        created.append(lab)
        return lab

    monkeypatch.setattr(tasks, "start_experiment_simulator", start)
    monkeypatch.setattr(ray, "get", lambda ref: ref())
    monkeypatch.setattr(ray, "kill", lambda *a, **k: None)
    monkeypatch.setattr(RayCompute, "ensure", lambda self: {})

    compute = RayCompute(measure_retry_backoff_s=0.0)
    frame, stats = compute.run_experiments(
        PlatformConfig.for_tests(),
        tmp_path / "pool.parquet",
        round_id=1,
        candidate_ids=["c0", "c1", "c2"],
        journal_path=tmp_path / "journal.jsonl",
    )
    assert frame["experiment_id"].tolist() == ["c0", "c1", "c2"]
    assert frame["response"].iloc[0] == 42.0  # replayed from the journal, not re-measured
    assert len(created) == n_actors  # ActorDiedError -> a fresh simulator (same journal)
    assert stats["lab_retries"] == [e.__name__ for e in failures]
    assert sum(lab.physical for lab in created) == 2  # c1, c2 measured exactly once


def test_measure_gives_up_after_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failures: list[type[Exception]] = [ActorUnavailableError] * 5
    monkeypatch.setattr(tasks, "start_experiment_simulator", lambda *a, **k: _FakeLab({}, failures))
    monkeypatch.setattr(ray, "get", lambda ref: ref())
    monkeypatch.setattr(ray, "kill", lambda *a, **k: None)
    monkeypatch.setattr(RayCompute, "ensure", lambda self: {})
    with pytest.raises(ActorUnavailableError):
        RayCompute(measure_attempts=3, measure_retry_backoff_s=0.0).run_experiments(
            PlatformConfig.for_tests(),
            tmp_path / "pool.parquet",
            round_id=1,
            candidate_ids=["c1"],
            journal_path=tmp_path / "j.jsonl",
        )
    assert len(failures) == 2  # three attempts consumed three failures
