"""Real-time hot path: Ray actors ingest -> preprocess -> decode -> action, one frame at a time.

::

    source (driver, 10 Hz clock)          one actor per stage, calls queued per actor
      frame N  ─► Ingest.push ─► Preprocess.process ─► Decoder.step ─► Action.apply ─► feedback
      frame N+1     ─► Ingest.push ─► Preprocess.process ─► ...        (stages overlap)

* Each stage is a Ray actor; the driver chains the four calls with ObjectRefs and does
  not block, so preprocessing of frame N+1 overlaps decoding of frame N.
* **Drop, never queue**: a new frame is refused at the source when ``max_in_flight``
  frames are already in the pipeline (backpressure drop), and a stage discards a frame
  that is already older than ``late_ms`` when the stage starts (late drop). Both are
  counted; a dropped frame's absolute index still advances the decoder's RoPE positions.
* The ingest actor keeps a preallocated ring buffer of the last ``ring_size`` IQ
  volumes (the in-memory tap a record path would drain asynchronously).
* Timing: every stage records wall-clock ``time.time_ns()`` start/end (comparable across
  processes on one host) and emits an OpenTelemetry span (``telemetry.py``).
* The decoder actor supports an atomic weight hot-swap in a separate concurrency group,
  so a swap never waits behind, or blocks, the frame stream.
* Closed loop: the action's cursor velocity is fed back to the simulated user.

Only the stage logic here is Ray-specific; the arithmetic lives in ``preprocess`` and
``models`` and is identical to the batch path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch

from ml_platform.config import PlatformConfig
from ml_platform.logging import get_logger
from ml_platform.ray_runtime.cluster import ensure_ray
from ml_platform.workloads.fusi_realtime import telemetry
from ml_platform.workloads.fusi_realtime.config import TelemetryConfig
from ml_platform.workloads.fusi_realtime.gates import LatencyReport, latency_report_from_records
from ml_platform.workloads.fusi_realtime.models import load_streaming
from ml_platform.workloads.fusi_realtime.preprocess import PreprocessorState, StreamingPreprocessor
from ml_platform.workloads.fusi_realtime.sim import FusiSimulator, direction_vectors

log = get_logger(__name__)


def _late(payload: dict[str, Any], late_ms: float, now_ns: int) -> bool:
    return (now_ns - payload["t_acq_ns"]) / 1e6 > late_ms


def _drop(payload: dict[str, Any], stage: str) -> dict[str, Any]:
    payload["dropped"] = stage
    payload.pop("iq", None)
    payload.pop("volume", None)
    return payload


@ray.remote
class IngestActor:
    def __init__(
        self, ring_size: int, shape: tuple[int, ...], late_ms: float, exporter: TelemetryConfig
    ) -> None:
        self.ring = np.zeros((ring_size, *shape), dtype=np.complex64)  # preallocated slots
        self.slots = ring_size
        self.late_ms = late_ms
        self.tracer = telemetry.configure(exporter)

    def push(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.time_ns()
        if _late(payload, self.late_ms, t0):
            return _drop(payload, "ingest")
        slot = payload["index"] % self.slots
        np.copyto(self.ring[slot], payload["iq"])
        payload["iq"] = self.ring[slot]  # serialised on return; the ring keeps its copy
        t1 = time.time_ns()
        payload["ts"] = {"ingest": (t0, t1)}
        telemetry.record_stage(self.tracer, payload["carrier"], "ingest", t0, t1, slot=slot)
        return payload

    def recent(self, n: int) -> np.ndarray:
        return self.ring[:n].copy()

    def flush(self) -> None:
        telemetry.flush()


@ray.remote
class PreprocessActor:
    def __init__(self, state_dir: str, late_ms: float, exporter: TelemetryConfig) -> None:
        self.pre = StreamingPreprocessor(PreprocessorState.load(state_dir))
        self.late_ms = late_ms
        self.tracer = telemetry.configure(exporter)

    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.time_ns()
        if payload.get("dropped"):
            return payload
        if _late(payload, self.late_ms, t0):
            return _drop(payload, "preprocess")
        iq = payload.pop("iq")
        payload["volume"] = self.pre.step(
            iq, payload["cardiac_phase"], payload["respiratory_phase"]
        )
        t1 = time.time_ns()
        payload["ts"]["preprocess"] = (t0, t1)
        telemetry.record_stage(self.tracer, payload["carrier"], "preprocess", t0, t1)
        return payload

    def flush(self) -> None:
        telemetry.flush()


@ray.remote(concurrency_groups={"control": 1})  # type: ignore[call-overload]
class DecoderActor:
    def __init__(
        self, checkpoint: str, late_ms: float, threads: int, exporter: TelemetryConfig
    ) -> None:
        torch.set_num_threads(threads)
        self.decoder = load_streaming(checkpoint)
        self.version = checkpoint
        self.late_ms = late_ms
        self.tracer = telemetry.configure(exporter)

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.time_ns()
        if payload.get("dropped"):
            return payload
        if _late(payload, self.late_ms, t0):
            return _drop(payload, "decode")
        decoder = self.decoder  # one read: a concurrent swap cannot tear this frame
        payload["probs"] = decoder.step(payload.pop("volume"), int(payload["index"]))
        payload["model_version"] = self.version
        t1 = time.time_ns()
        payload["ts"]["decode"] = (t0, t1)
        telemetry.record_stage(self.tracer, payload["carrier"], "decode", t0, t1)
        return payload

    @ray.method(concurrency_group="control")
    def flush(self) -> None:
        telemetry.flush()

    @ray.method(concurrency_group="control")
    def swap(self, checkpoint: str) -> str:
        """Load new weights off the frame path, then swap the reference atomically."""
        fresh = load_streaming(checkpoint)
        self.decoder, self.version = fresh, checkpoint
        return checkpoint


@ray.remote
class ActionActor:
    """Decoded intent -> bounded cursor velocity (expected direction under the posterior)."""

    def __init__(
        self,
        n_directions: int,
        gain: float,
        max_speed: float,
        frame_rate_hz: float,
        exporter: TelemetryConfig,
    ) -> None:
        self.dirs = direction_vectors(n_directions)
        self.gain, self.max_speed = gain, max_speed
        self.dt = 1.0 / frame_rate_hz
        self.cursor = np.zeros(2)
        self.tracer = telemetry.configure(exporter)

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.time_ns()
        if payload.get("dropped"):
            return _record(payload)
        p = payload["probs"]
        v = self.gain * (p @ self.dirs)
        speed = float(np.linalg.norm(v))
        if speed > self.max_speed:  # safety bound on speed
            v = v * (self.max_speed / speed)
        self.cursor = np.clip(self.cursor + v * self.dt, -1.0, 1.0)  # workspace bound
        payload["velocity"] = v
        payload["decoded"] = int(np.argmax(p))
        t1 = time.time_ns()
        payload["ts"]["action"] = (t0, t1)
        telemetry.record_stage(self.tracer, payload["carrier"], "action", t0, t1)
        return _record(payload)

    def flush(self) -> None:
        telemetry.flush()


def _record(payload: dict[str, Any]) -> dict[str, Any]:
    """Small per-frame record returned to the driver (no volumes)."""
    keep = ("index", "t_acq_ns", "ts", "dropped", "velocity", "decoded", "cue", "model_version")
    return {k: payload[k] for k in keep if k in payload}


@dataclass
class RealtimePipeline:
    ingest: Any
    preprocess: Any
    decoder: Any
    action: Any

    def submit(self, payload: dict[str, Any]) -> Any:
        return self.action.apply.remote(
            self.decoder.step.remote(
                self.preprocess.process.remote(self.ingest.push.remote(payload))
            )
        )

    def kill(self) -> None:
        actors = (self.ingest, self.preprocess, self.decoder, self.action)
        ray.get([a.flush.remote() for a in actors])  # batched span exporters
        for a in actors:
            ray.kill(a)


def build_pipeline(
    cfg: PlatformConfig, *, checkpoint: Path, preprocessor_dir: Path, grid: tuple[int, int, int]
) -> RealtimePipeline:
    """``checkpoint``: a decoder checkpoint *directory* (``model.pt`` or ``baseline.npz``)."""
    rt, sim = cfg.fusi.realtime, cfg.fusi.sim
    exp = cfg.fusi.telemetry
    return RealtimePipeline(
        ingest=IngestActor.options(  # type: ignore[attr-defined]
            num_cpus=0.5
        ).remote(32, (sim.n_ensemble, *grid), rt.late_ms, exp),
        preprocess=PreprocessActor.options(  # type: ignore[attr-defined]
            num_cpus=1
        ).remote(str(preprocessor_dir), rt.late_ms, exp),
        decoder=DecoderActor.options(  # type: ignore[attr-defined]
            num_cpus=1
        ).remote(str(checkpoint), rt.late_ms, rt.torch_threads, exp),
        action=ActionActor.options(  # type: ignore[attr-defined]
            num_cpus=0.5
        ).remote(sim.n_directions, 2.0, 2.0, sim.frame_rate_hz, exp),
    )


def run_realtime(
    cfg: PlatformConfig,
    *,
    checkpoint: Path,
    preprocessor_dir: Path,
    seconds: float | None = None,
    grid: tuple[int, int, int] | None = None,
    session: int = 900,
) -> tuple[LatencyReport, list[dict[str, Any]]]:
    """Stream the closed-loop simulator at ``frame_rate_hz`` through the actor pipeline."""
    ensure_ray()
    fc = cfg.fusi
    rt = fc.realtime
    grid = tuple(grid or fc.sim.grid)  # type: ignore[assignment]
    assert grid is not None
    seconds = rt.seconds if seconds is None else seconds
    period = 1.0 / fc.sim.frame_rate_hz
    n_total = rt.warmup_frames + round(seconds * fc.sim.frame_rate_hz)
    sim = FusiSimulator(fc.sim, session=session, grid=grid, emit="iq", closed_loop=True)
    tracer = telemetry.configure(fc.telemetry)
    pipe = build_pipeline(cfg, checkpoint=checkpoint, preprocessor_dir=preprocessor_dir, grid=grid)
    # make sure every actor finished __init__ (model load) before the clock starts
    ray.get([pipe.ingest.recent.remote(0)])
    pending: dict[Any, int] = {}
    records: list[dict[str, Any]] = []
    dropped_bp = max_in_flight = 0
    spans: dict[int, Any] = {}
    # one trace per trial: a root span per task segment (a trial's plan phase, or the
    # rest that follows it); every frame span is its child
    segment: tuple[int, str] | None = None
    segment_span: Any = None
    last_trial = -1

    def collect(timeout: float) -> None:
        if not pending:
            return
        ready, _ = ray.wait(list(pending), num_returns=len(pending), timeout=timeout)
        for ref in ready:
            idx = pending.pop(ref)
            rec = ray.get(ref)
            if "velocity" in rec:
                sim.feedback(np.asarray(rec["velocity"]))
            span = spans.pop(idx, None)
            if span is not None:
                span.end(
                    end_time=rec["ts"]["action"][1]
                    if "ts" in rec and "action" in rec["ts"]
                    else None
                )
            if idx >= rt.warmup_frames:
                records.append(rec)

    t_start = time.perf_counter()
    next_t = t_start
    for i in range(n_total):
        frame = sim.next_frame()
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        next_t += period
        t_acq = time.time_ns()
        collect(0)
        if len(pending) >= rt.max_in_flight:
            if i >= rt.warmup_frames:
                dropped_bp += 1
            continue
        last_trial = frame.trial if frame.trial >= 0 else last_trial
        seg = (last_trial, frame.phase)
        if seg != segment:
            if segment_span is not None:
                segment_span.end(end_time=t_acq)
            segment = seg
            segment_span = telemetry.start_span(
                tracer,
                "fusi.trial",
                start_ns=t_acq,
                trial=seg[0],
                phase=seg[1],
                cue=frame.cue_direction,
            )
        span = telemetry.start_span(
            tracer,
            "fusi.frame",
            parent=telemetry.carrier_for(segment_span),
            start_ns=t_acq,
            frame=i,
        )
        spans[i] = span
        carrier = telemetry.carrier_for(span)
        payload = {
            "index": i,
            "t_acq_ns": t_acq,
            "iq": frame.iq,
            "cardiac_phase": frame.cardiac_phase,
            "respiratory_phase": frame.respiratory_phase,
            "cue": frame.cue_direction,
            "carrier": carrier,
        }
        pending[pipe.submit(payload)] = i
        max_in_flight = max(max_in_flight, len(pending))
    t_stop = time.perf_counter()
    # zero-backlog check: one frame period + the budget after the last acquisition
    collect(period + fc.gates.p99_budget_ms / 1e3)
    final_in_flight = len(pending)
    collect(30.0)
    if segment_span is not None:
        segment_span.end()
    telemetry.flush()
    duration = t_stop - t_start - rt.warmup_frames * period
    pipe.kill()
    report = latency_report_from_records(
        records,
        frame_rate_hz=fc.sim.frame_rate_hz,
        grid=grid,
        n_frames=n_total - rt.warmup_frames,
        n_dropped_backpressure=dropped_bp,
        max_in_flight=max_in_flight,
        final_in_flight=final_in_flight,
        duration_s=duration,
        meta={
            "seconds": seconds,
            "late_ms": rt.late_ms,
            "max_in_flight_limit": rt.max_in_flight,
            "torch_threads": rt.torch_threads,
            "otel_exporter": fc.telemetry.exporter,
            "trials_succeeded": int(sum(sim.outcomes)),
            "trials_finished": len(sim.outcomes),
            "final_engagement": sim.user.engagement,
        },
    )
    log.info(
        "fusi.realtime.done",
        e2e_p50_ms=round(report.end_to_end.p50, 2),
        e2e_p99_ms=round(report.end_to_end.p99, 2),
        dropped=report.n_dropped,
        final_in_flight=final_in_flight,
    )
    return report, records
