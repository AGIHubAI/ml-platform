"""Latency statistics and the promotion gate of the fUSI decoder.

Gate (all must hold, every failure is a reason string):

1. every leakage / overfitting check passed (`splits.CheckResult`);
2. test accuracy >= ``min_accuracy`` and beats the current production decoder (or, with
   no production decoder yet, the PCA + LDA baseline) by ``min_accuracy_improvement``;
3. end-to-end p99 <= ``p99_budget_ms`` at the source frame rate, and the per-stage
   p99s sum to <= ``stage_budget_ms`` (the working budget inside the ceiling);
4. zero backlog: dropped-frame fraction <= ``max_drop_fraction`` and nothing left in
   flight when the stream stops.

Why 130 ms: in a cursor BCI the brain's own visual-feedback delay was estimated at ~130 ms
in one nonhuman primate (Golub, Yu, Chase, IEEE EMBC 2012, doi 10.1109/EMBC.2012.6346182);
the system's added delay is capped there. One animal, one study: a reference point, not a law.
80 ms is the working stage budget, 50 ms is jitter headroom. The published closed-loop
fUS-BMI ran at 2 Hz, one update per 500 ms (Griggs et al., Nat. Neurosci. 2024). The CI config loosens the
budget because shared CPU runners are not the reference machine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ml_platform.workloads.fusi_realtime.config import GateConfig
from ml_platform.workloads.fusi_realtime.splits import CheckResult

STAGES: tuple[str, ...] = ("transfer", "ingest", "preprocess", "decode", "action")


@dataclass(frozen=True)
class StageStats:
    n: int
    p50: float
    p95: float
    p99: float
    mean: float
    max: float

    @classmethod
    def of(cls, values_ms: Sequence[float]) -> StageStats:
        v = np.asarray(values_ms, dtype=np.float64)
        if v.size == 0:
            return cls(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        p50, p95, p99 = np.percentile(v, [50, 95, 99])
        return cls(int(v.size), float(p50), float(p95), float(p99), float(v.mean()), float(v.max()))


@dataclass(frozen=True)
class LatencyReport:
    """Result of one real-time run (``realtime.run_realtime``)."""

    frame_rate_hz: float
    grid: tuple[int, int, int]
    n_frames: int  # produced by the source after warm-up
    n_completed: int
    n_dropped_backpressure: int  # refused at the source: pipeline already max_in_flight deep
    n_dropped_late: int  # discarded by a stage because the frame was already too old
    max_in_flight: int
    final_in_flight: int
    achieved_hz: float
    stages: dict[str, StageStats]
    end_to_end: StageStats
    queue_wait: StageStats
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return self.n_dropped_backpressure + self.n_dropped_late

    @property
    def drop_fraction(self) -> float:
        return self.n_dropped / max(self.n_frames, 1)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_dropped"] = self.n_dropped
        d["drop_fraction"] = self.drop_fraction
        return d

    def flat_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {
            "rt_e2e_p50_ms": self.end_to_end.p50,
            "rt_e2e_p95_ms": self.end_to_end.p95,
            "rt_e2e_p99_ms": self.end_to_end.p99,
            "rt_dropped": float(self.n_dropped),
            "rt_drop_fraction": self.drop_fraction,
            "rt_max_in_flight": float(self.max_in_flight),
            "rt_achieved_hz": self.achieved_hz,
        }
        for name, s in self.stages.items():
            out[f"rt_{name}_p50_ms"] = s.p50
            out[f"rt_{name}_p99_ms"] = s.p99
        return out

    def table(self) -> str:
        """Fixed-width text table (what the benchmark script prints)."""
        lines = [f"{'stage':<12}{'n':>6}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}  (ms)"]
        rows = [
            *self.stages.items(),
            ("queue_wait", self.queue_wait),
            ("end_to_end", self.end_to_end),
        ]
        for name, s in rows:
            lines.append(f"{name:<12}{s.n:>6}{s.p50:>9.2f}{s.p95:>9.2f}{s.p99:>9.2f}{s.max:>9.2f}")
        lines.append(
            f"frames {self.n_frames}  completed {self.n_completed}  "
            f"dropped {self.n_dropped} (backpressure {self.n_dropped_backpressure}, "
            f"late {self.n_dropped_late})  max in flight {self.max_in_flight}  "
            f"final backlog {self.final_in_flight}  achieved {self.achieved_hz:.2f} Hz"
        )
        return "\n".join(lines)


def latency_report_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    frame_rate_hz: float,
    grid: tuple[int, int, int],
    n_frames: int,
    n_dropped_backpressure: int,
    max_in_flight: int,
    final_in_flight: int,
    duration_s: float,
    meta: Mapping[str, Any] | None = None,
) -> LatencyReport:
    """Aggregate per-frame timing records (ns timestamps, see ``realtime``) into a report."""
    done = [r for r in records if not r.get("dropped")]
    late = [r for r in records if r.get("dropped")]
    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    waits: list[float] = []
    e2e: list[float] = []
    for r in done:
        ts = r["ts"]
        per_stage["transfer"].append((ts["ingest"][0] - r["t_acq_ns"]) / 1e6)
        prev_end = ts["ingest"][1]
        for s in ("ingest", "preprocess", "decode", "action"):
            a, b = ts[s]
            per_stage[s].append((b - a) / 1e6)
            if s != "ingest":
                waits.append((a - prev_end) / 1e6)
            prev_end = b
        e2e.append((ts["action"][1] - r["t_acq_ns"]) / 1e6)
    return LatencyReport(
        frame_rate_hz=frame_rate_hz,
        grid=grid,
        n_frames=n_frames,
        n_completed=len(done),
        n_dropped_backpressure=n_dropped_backpressure,
        n_dropped_late=len(late),
        max_in_flight=max_in_flight,
        final_in_flight=final_in_flight,
        achieved_hz=len(done) / duration_s if duration_s > 0 else 0.0,
        stages={s: StageStats.of(v) for s, v in per_stage.items()},
        end_to_end=StageStats.of(e2e),
        queue_wait=StageStats.of(waits),
        meta=dict(meta or {}),
    )


@dataclass(frozen=True)
class GateDecision:
    """Duck-compatible with the registry's gate (``passed`` + ``reasons``)."""

    passed: bool
    reasons: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)


def latency_gate(report: LatencyReport, cfg: GateConfig) -> list[str]:
    reasons = []
    p99 = report.end_to_end.p99
    if not np.isfinite(p99):
        reasons.append("no completed frames: end-to-end latency unknown")
    elif p99 > cfg.p99_budget_ms:
        reasons.append(f"end-to-end p99 {p99:.1f} ms > budget {cfg.p99_budget_ms:.0f} ms")
    stage_sum = sum(s.p99 for s in report.stages.values() if np.isfinite(s.p99))
    if stage_sum > cfg.stage_budget_ms:
        reasons.append(
            f"sum of stage p99s {stage_sum:.1f} ms > stage budget {cfg.stage_budget_ms:.0f} ms"
        )
    if report.drop_fraction > cfg.max_drop_fraction:
        reasons.append(
            f"dropped {report.n_dropped}/{report.n_frames} frames "
            f"({report.drop_fraction:.1%} > {cfg.max_drop_fraction:.1%})"
        )
    if report.final_in_flight > 0:
        reasons.append(f"backlog: {report.final_in_flight} frames still in flight at stop")
    return reasons


def decide_promotion(
    *,
    candidate_acc: float,
    reference_acc: float,
    reference: str,
    latency: LatencyReport,
    checks: Sequence[CheckResult],
    cfg: GateConfig,
) -> GateDecision:
    """``reference`` is "production" or "baseline" (what ``reference_acc`` was measured on)."""
    reasons = [f"check {c.name} failed: {c.detail}" for c in checks if not c.passed]
    if candidate_acc < cfg.min_accuracy:
        reasons.append(f"test accuracy {candidate_acc:.3f} < minimum {cfg.min_accuracy:.3f}")
    if not candidate_acc > reference_acc + cfg.min_accuracy_improvement:
        reasons.append(
            f"test accuracy {candidate_acc:.3f} does not beat {reference} {reference_acc:.3f}"
        )
    reasons += latency_gate(latency, cfg)
    return GateDecision(
        passed=not reasons,
        reasons=reasons,
        metrics={
            "candidate_acc": candidate_acc,
            "reference_acc": reference_acc,
            "reference": reference,
            "e2e_p99_ms": latency.end_to_end.p99,
            "p99_budget_ms": cfg.p99_budget_ms,
            "drop_fraction": latency.drop_fraction,
            "checks_passed": all(c.passed for c in checks),
        },
    )
