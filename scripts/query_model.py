"""Query a running surrogate-model server (Ray Serve).

Usage::

    uv run python scripts/query_model.py [--url http://127.0.0.1:8000] [--n 5]

Calls /health, /model-info, /predict for a few candidates and /predict_batch.
Candidates come from the persisted candidate pool (``<data_dir>/candidate_pool``)
when it exists, otherwise random feature vectors are used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd


def _load_candidates(pool_path: Path | None, n: int, n_features: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if pool_path is not None and pool_path.exists():
        pool = pd.read_parquet(pool_path)
        rows = pool.iloc[np.sort(rng.choice(len(pool), size=min(n, len(pool)), replace=False))]
        fcols = [c for c in pool.columns if c[:1] == "f" and c[1:].isdigit()]
        return pd.DataFrame(
            {
                "candidate_id": rows["candidate_id"].astype(str),
                "features": rows[fcols].values.tolist(),
            }
        )
    X = np.clip(rng.standard_normal((n, n_features)), -3, 3)
    return pd.DataFrame({"candidate_id": [f"random_{i}" for i in range(n)], "features": X.tolist()})


def _default_pool_path() -> Path | None:
    try:
        from ml_platform.config import load_config

        cfg = load_config()
        return cfg.paths.resolved().data_dir / "candidate_pool" / "pool.parquet"
    except Exception:
        return None


def _section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=5, help="number of candidates to score")
    p.add_argument("--pool", type=Path, default=None, help="pool parquet (default: data dir)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=30.0)
    a = p.parse_args(argv)

    with httpx.Client(base_url=a.url, timeout=a.timeout) as client:
        try:
            health = client.get("/health")
        except httpx.HTTPError as exc:
            print(
                f"cannot reach {a.url}: {exc}\nstart the server first: make serve", file=sys.stderr
            )
            return 1
        _section("GET /health")
        print(json.dumps(health.json(), indent=2))

        _section("GET /model-info")
        info: dict[str, Any] = client.get("/model-info").json()
        metrics = info.pop("metrics", {})
        print(json.dumps(info, indent=2))

        n_features = int(info.get("n_features") or 32)
        cands = _load_candidates(a.pool or _default_pool_path(), a.n, n_features, a.seed)
        _section(f"POST /predict  ({len(cands)} candidates, one request each)")
        print(f"{'candidate':<16}{'mean':>10}{'std':>10}{'latency_ms':>12}  version")
        for cid, feats in zip(cands["candidate_id"], cands["features"], strict=True):
            t0 = time.perf_counter()
            r = client.post("/predict", json={"features": list(feats)})
            dt = (time.perf_counter() - t0) * 1e3
            r.raise_for_status()
            d = r.json()
            print(f"{cid:<16}{d['mean']:>10.4f}{d['std']:>10.4f}{dt:>12.1f}  {d['model_version']}")

        _section(f"POST /predict_batch  ({len(cands)} vectors in one request)")
        t0 = time.perf_counter()
        r = client.post("/predict_batch", json={"features": [list(f) for f in cands["features"]]})
        dt = (time.perf_counter() - t0) * 1e3
        r.raise_for_status()
        batch = r.json()
        for cid, d in zip(cands["candidate_id"], batch["predictions"], strict=True):
            ucb = d["mean"] + d["std"]
            print(f"{cid:<16} mean={d['mean']:+.4f} std={d['std']:.4f} ucb(beta=1)={ucb:+.4f}")
        print(f"batch latency: {dt:.1f} ms, model_version={batch['model_version']}")

        _section("serving metrics (this replica)")
        after = client.get("/model-info").json().get("metrics", metrics)
        print(json.dumps(after, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
