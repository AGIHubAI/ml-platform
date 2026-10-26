"""Latency statistics and the fUSI promotion gate."""

from __future__ import annotations

from ml_platform.workloads.fusi_realtime.config import GateConfig
from ml_platform.workloads.fusi_realtime.gates import (
    decide_promotion,
    latency_report_from_records,
)
from ml_platform.workloads.fusi_realtime.splits import CheckResult

MS = 1_000_000


def _record(i: int, stage_ms: float) -> dict:  # type: ignore[type-arg]
    t = i * 100 * MS
    ts, cur = {}, t + 2 * MS  # 2 ms transfer
    for s in ("ingest", "preprocess", "decode", "action"):
        ts[s] = (cur, cur + int(stage_ms * MS))
        cur += int(stage_ms * MS) + MS  # 1 ms queue wait between stages
    return {"index": i, "t_acq_ns": t, "ts": ts}


def _report(stage_ms: float, *, dropped: int = 0, backlog: int = 0, n: int = 100):  # type: ignore[no-untyped-def]
    recs = [_record(i, stage_ms) for i in range(n - dropped)]
    return latency_report_from_records(
        recs,
        frame_rate_hz=10.0,
        grid=(32, 32, 32),
        n_frames=n,
        n_dropped_backpressure=dropped,
        max_in_flight=2,
        final_in_flight=backlog,
        duration_s=n / 10,
    )


def test_latency_report_percentiles() -> None:
    r = _report(5.0)
    assert r.stages["decode"].p50 == 5.0 and r.stages["transfer"].p99 == 2.0
    assert abs(r.end_to_end.p99 - (2 + 4 * 5 + 3)) < 1e-9
    assert r.queue_wait.p50 == 1.0 and r.n_completed == 100 and r.drop_fraction == 0
    assert "end_to_end" in r.table()


def _decide(report, acc=0.6, ref=0.5, checks=()):  # type: ignore[no-untyped-def]
    return decide_promotion(
        candidate_acc=acc,
        reference_acc=ref,
        reference="baseline",
        latency=report,
        checks=list(checks),
        cfg=GateConfig(),
    )


def test_gate_passes_fast_accurate_clean_candidate() -> None:
    g = _decide(_report(5.0))
    assert g.passed and g.reasons == []


def test_gate_failure_reasons() -> None:
    assert any("p99" in r for r in _decide(_report(40.0)).reasons)  # 2+160+3 ms e2e
    assert any("stage budget" in r for r in _decide(_report(25.0)).reasons)  # 100 ms of stages
    assert any("dropped" in r for r in _decide(_report(5.0, dropped=3)).reasons)
    assert any("backlog" in r for r in _decide(_report(5.0, backlog=1)).reasons)
    assert any("does not beat" in r for r in _decide(_report(5.0), acc=0.5, ref=0.5).reasons)
    assert any("minimum" in r for r in _decide(_report(5.0), acc=0.15, ref=0.1).reasons)
    leak = CheckResult("temporal_gap", False, "min gap 0 frames")
    g = _decide(_report(5.0), checks=[leak])
    assert not g.passed and g.reasons == ["check temporal_gap failed: min gap 0 frames"]


def test_ci_budget_is_configurable() -> None:
    slow = _report(18.0)  # e2e ~ 77 ms, stages 72 ms
    assert decide_promotion(
        candidate_acc=0.6,
        reference_acc=0.5,
        reference="b",
        latency=slow,
        checks=[],
        cfg=GateConfig(p99_budget_ms=60.0),
    ).reasons
    assert decide_promotion(
        candidate_acc=0.6,
        reference_acc=0.5,
        reference="b",
        latency=slow,
        checks=[],
        cfg=GateConfig(p99_budget_ms=400.0),
    ).passed
