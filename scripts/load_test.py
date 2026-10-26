"""Small async load test for the surrogate-model server.

Usage::

    uv run python scripts/load_test.py [--url http://127.0.0.1:8000] \
        [--concurrency 32] [--requests 2000] [--endpoint predict|predict_batch]

Keeps ``--concurrency`` requests in flight until ``--requests`` have completed
and reports throughput, latency percentiles and the batch sizes the server
formed (from /model-info). Try ``--concurrency 1`` vs ``64`` to see dynamic
batching trade latency for throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import httpx
import numpy as np


async def _run(a: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(a.seed)
    X = np.clip(rng.standard_normal((min(a.requests, 4096), a.n_features)), -3, 3)
    latencies: list[float] = []
    errors: dict[str, int] = {}
    counter = iter(range(a.requests))
    limits = httpx.Limits(max_connections=a.concurrency, max_keepalive_connections=a.concurrency)

    async with httpx.AsyncClient(base_url=a.url, timeout=a.timeout, limits=limits) as client:
        before = (await client.get("/model-info")).json().get("metrics", {})

        async def worker() -> None:
            for i in counter:
                x = X[i % len(X)]
                if a.endpoint == "predict":
                    payload: dict[str, Any] = {"features": x.tolist()}
                else:
                    rows = X[np.arange(i, i + a.batch_rows) % len(X)]
                    payload = {"features": rows.tolist()}
                t0 = time.perf_counter()
                try:
                    r = await client.post(f"/{a.endpoint}", json=payload)
                    key = str(r.status_code)
                except httpx.HTTPError as exc:
                    key = type(exc).__name__
                latencies.append((time.perf_counter() - t0) * 1e3)
                if key != "200":
                    errors[key] = errors.get(key, 0) + 1

        # warm-up (not measured)
        await client.post("/predict", json={"features": X[0].tolist()})
        t_start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(a.concurrency)))
        wall = time.perf_counter() - t_start
        after = (await client.get("/model-info")).json().get("metrics", {})

    lat = np.asarray(latencies)
    ok = len(lat) - sum(errors.values())
    rows_per_req = 1 if a.endpoint == "predict" else a.batch_rows
    b0, b1 = before.get("batches_total", 0), after.get("batches_total", 0)
    r0, r1 = before.get("rows_predicted", 0), after.get("rows_predicted", 0)
    return {
        "endpoint": f"/{a.endpoint}",
        "concurrency": a.concurrency,
        "requests": len(lat),
        "ok": ok,
        "errors": errors,
        "wall_s": round(wall, 3),
        "throughput_rps": round(len(lat) / wall, 1),
        "rows_per_s": round(ok * rows_per_req / wall, 1),
        "latency_ms": {
            "mean": round(float(lat.mean()), 2),
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
            "max": round(float(lat.max()), 2),
        },
        "server_batches": {
            "n_batches_during_test": b1 - b0,
            "mean_batch_size_during_test": round((r1 - r0) / max(1, b1 - b0), 2),
            "recent_batch_sizes": after.get("batch_size", {}).get("last"),
            "max_recent": after.get("batch_size", {}).get("max_recent"),
            "histogram_le": after.get("batch_size", {}).get("histogram_le"),
        },
        "server_latency_ms": after.get("latency_ms"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--requests", type=int, default=2000)
    p.add_argument("--endpoint", choices=["predict", "predict_batch"], default="predict")
    p.add_argument("--batch-rows", type=int, default=32, help="rows per /predict_batch request")
    p.add_argument("--n-features", type=int, default=32)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="print raw JSON only")
    a = p.parse_args(argv)

    res = asyncio.run(_run(a))
    if a.json:
        print(json.dumps(res, indent=2))
        return 0 if not res["errors"] else 1
    lat = res["latency_ms"]
    sb = res["server_batches"]
    print(f"endpoint        {res['endpoint']}  concurrency={res['concurrency']}")
    print(f"requests        {res['requests']} ok={res['ok']} errors={res['errors'] or 0}")
    print(f"wall time       {res['wall_s']} s")
    print(f"throughput      {res['throughput_rps']} req/s   ({res['rows_per_s']} rows/s)")
    print(
        f"latency ms      p50={lat['p50']}  p95={lat['p95']}  p99={lat['p99']}  "
        f"mean={lat['mean']}  max={lat['max']}"
    )
    print(
        f"server batches  {sb['n_batches_during_test']} batches, "
        f"mean size {sb['mean_batch_size_during_test']}, max recent {sb['max_recent']}"
    )
    print(f"batch hist (<=) {sb['histogram_le']}")
    print(f"recent sizes    {sb['recent_batch_sizes']}")
    return 0 if not res["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
