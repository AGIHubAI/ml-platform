"""OpenTelemetry for the fUSI workload, with no vendor lock-in.

What gets a span:

* hot path: one trace per **trial**; a ``fusi.frame`` span per frame under the trial
  span, and one child span per stage (ingest, preprocess, decode, action) created in
  the actor process that ran it, with the stage's measured start/end. Context travels
  with the frame as a W3C ``traceparent`` carrier;
* training: ``fusi.train.epoch`` and ``fusi.train.step`` spans in every Ray Train worker;
* Dagster: one ``dagster.asset.<name>`` span per fUSI asset materialisation;
* Ray Serve: ``fusi.serve.request`` per request.

Exporters (``fusi.telemetry.exporter``):

* ``none`` (default) -- the API's no-op tracer; latency reports come from frame records;
* ``console`` -- SDK ``ConsoleSpanExporter``;
* ``file`` -- JSON lines, one file per process (``spans-<pid>.jsonl``) under
  ``fusi.telemetry.file_dir``: the local exporter for the Mac (no collector needed);
* ``otlp`` -- OTLP/HTTP to ``otlp_endpoint`` (or ``$OTEL_EXPORTER_OTLP_ENDPOINT``) if
  ``opentelemetry-exporter-otlp`` is installed; otherwise a warning and no-op.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ml_platform.logging import get_logger
from ml_platform.workloads.fusi_realtime.config import TelemetryConfig

log = get_logger(__name__)

TRACER_NAME = "ml_platform.fusi_realtime"
_PROPAGATOR = TraceContextTextMapPropagator()
_configured: str | None = None


def _file_exporter(directory: str) -> Any:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    from ml_platform.storage import require_mounted

    class JsonLinesSpanExporter(SpanExporter):  # type: ignore[misc]
        """Append each finished span as one JSON line (per-process file: no lock needed)."""

        def __init__(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.path = path

        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            with self.path.open("a") as f:
                for s in spans:
                    f.write(s.to_json(indent=None) + "\n")
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    return JsonLinesSpanExporter(require_mounted(directory) / f"spans-{os.getpid()}.jsonl")


def configure(cfg: TelemetryConfig | str, service: str = "fusi-realtime") -> Tracer:
    """Install a tracer provider for this process once (actors/workers call this on start)."""
    global _configured
    tcfg = TelemetryConfig(exporter=cfg) if isinstance(cfg, str) else cfg  # type: ignore[arg-type]
    if tcfg.exporter == "none":
        return trace.get_tracer(TRACER_NAME)
    if _configured is None:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        if tcfg.exporter == "console":
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif tcfg.exporter == "file":
            directory = tcfg.file_dir or "otel"
            # batched: file writes stay off the frame path; actors flush before exit
            provider.add_span_processor(BatchSpanProcessor(_file_exporter(directory)))
        elif tcfg.exporter == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                    OTLPSpanExporter,
                )

                kwargs = {"endpoint": tcfg.otlp_endpoint} if tcfg.otlp_endpoint else {}
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(**kwargs)))
            except ImportError:
                log.warning(
                    "telemetry.otlp_unavailable", hint="install opentelemetry-exporter-otlp"
                )
        trace.set_tracer_provider(provider)
        _configured = tcfg.exporter
    return trace.get_tracer(TRACER_NAME)


def flush() -> None:
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


def carrier_for(span: Any) -> dict[str, str]:
    """W3C trace-context carrier that makes ``span`` the parent downstream."""
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier, context=trace.set_span_in_context(span))
    return carrier


def start_span(
    tracer: Tracer,
    name: str,
    *,
    parent: dict[str, str] | None = None,
    start_ns: int | None = None,
    **attributes: Any,
) -> Any:
    """Start a span, optionally under a propagated ``parent`` carrier."""
    ctx = _PROPAGATOR.extract(parent) if parent else None
    return tracer.start_span(name, context=ctx, start_time=start_ns, attributes=attributes)


def record_stage(
    tracer: Tracer,
    carrier: dict[str, str],
    stage: str,
    start_ns: int,
    end_ns: int,
    **attributes: Any,
) -> None:
    """Emit a finished child span for ``stage`` with explicit timestamps."""
    s = start_span(tracer, f"fusi.{stage}", parent=carrier, start_ns=start_ns, **attributes)
    s.end(end_time=end_ns)


@contextmanager
def span(tracer: Tracer, name: str, **attributes: Any) -> Iterator[Any]:
    """Context-managed span in the current context (training steps, assets, requests)."""
    with tracer.start_as_current_span(name, attributes=attributes) as s:
        yield s
