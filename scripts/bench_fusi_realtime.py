"""Real-time latency benchmark: stream the closed-loop fUSI simulator through the Ray actors.

Usage::

    uv run python scripts/bench_fusi_realtime.py [--seconds 30] [--grid 32] [--checkpoint DIR]

Without ``--checkpoint`` it builds a decoder of the configured architecture with random
weights and fits the preprocessor on a short simulated calibration session: latency does
not depend on the weights, and this keeps the benchmark independent of a training run.
Reports per-stage and end-to-end p50/p95/p99, drops, queue depth and the latency gate
(``fusi.gates``: p99 <= budget at the source rate, zero backlog). Writes JSON to
``reports/fusi/bench/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

import numpy as np

from ml_platform.config import load_config
from ml_platform.workloads.fusi_realtime import preprocess as PP
from ml_platform.workloads.fusi_realtime.gates import latency_gate
from ml_platform.workloads.fusi_realtime.models import build_decoder, save_decoder
from ml_platform.workloads.fusi_realtime.sim import FusiSimulator


def calibration_assets(cfg, grid: tuple[int, int, int], out: Path, n_frames: int = 100) -> Path:  # type: ignore[no-untyped-def]
    """Preprocessor fitted on a separate calibration session + random-weight decoder."""
    fc = cfg.fusi
    sim = FusiSimulator(fc.sim, session=800, grid=grid)
    frames = sim.run(n_frames)
    power = np.stack([PP.clutter_filter(f.iq, fc.preprocess.clutter_rank) for f in frames])
    y = PP.detrended_log(power, fc.sim.frame_rate_hz, fc.preprocess)
    card = np.array([f.cardiac_phase for f in frames])
    resp = np.array([f.respiratory_phase for f in frames])
    state = PP.fit_preprocessor(
        [("calibration", 0, n_frames, y, card, resp)],
        grid=grid,
        frame_rate_hz=fc.sim.frame_rate_hz,
        cfg=fc.preprocess,
    )
    ckpt = out / "decoder"
    ckpt.mkdir(parents=True, exist_ok=True)
    state.save(ckpt / "preprocessor")
    save_decoder(build_decoder(fc.decoder, grid, fc.sim.n_directions), str(ckpt / "model.pt"))
    return ckpt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None)
    p.add_argument("--seconds", type=float, default=None, help="stream length (default: config)")
    p.add_argument(
        "--grid", type=int, default=None, help="cubic grid edge (default: fusi.sim.grid)"
    )
    p.add_argument("--checkpoint", type=Path, default=None, help="trained decoder dir")
    p.add_argument("--otel", choices=["none", "console", "file", "otlp"], default=None)
    p.add_argument(
        "--budget-ms", type=float, default=None, help="override fusi.gates.p99_budget_ms"
    )
    args = p.parse_args(argv)

    from ml_platform.workloads.fusi_realtime.realtime import run_realtime

    cfg = load_config(args.config or os.environ.get("ML_CONFIG") or "fusi-micro")
    cfg.paths.apply_runtime_env()  # Ray temp/spill etc. from the config; fails if unmounted
    fc = cfg.fusi
    if args.otel:
        fc.telemetry.exporter = args.otel
    if args.budget_ms is not None:
        fc.gates.p99_budget_ms = args.budget_ms
    grid = (args.grid,) * 3 if args.grid else tuple(fc.sim.grid)
    scratch = cfg.paths.resolved().artifacts_dir / "fusi" / "bench"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fusi_bench_", dir=scratch) as tmp:
        ckpt = args.checkpoint or calibration_assets(cfg, grid, Path(tmp))  # type: ignore[arg-type]
        report, _ = run_realtime(
            cfg,
            checkpoint=ckpt,
            preprocessor_dir=ckpt / "preprocessor",
            seconds=args.seconds,
            grid=grid,  # type: ignore[arg-type]
        )
    from ml_platform.ray_runtime.cluster import shutdown_ray

    shutdown_ray()
    reasons = latency_gate(report, fc.gates)
    print(
        f"\nfUSI real-time benchmark: grid {grid}, {fc.sim.frame_rate_hz:g} Hz, "
        f"{report.n_frames} frames, ensemble {fc.sim.n_ensemble}"
    )
    print(report.table())
    print(
        f"latency gate (p99 <= {fc.gates.p99_budget_ms:g} ms, drops <= "
        f"{fc.gates.max_drop_fraction:.0%}, zero backlog): {'PASS' if not reasons else 'FAIL'}"
    )
    for r in reasons:
        print(f"  - {r}")
    out = cfg.paths.resolved().reports_dir / "fusi" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"latency_{grid[0]}cube_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(
        json.dumps({**report.to_dict(), "gate_reasons": reasons}, indent=2, default=str)
    )
    print(f"wrote {path}")
    return 0 if not reasons else 1


if __name__ == "__main__":
    sys.exit(main())
