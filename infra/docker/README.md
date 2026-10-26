# Docker image and Compose stack

The platform runs without Docker (`make ray`, `make dagster`, `make mlflow`).
Compose reproduces the same process topology in containers so the demo can
show a real multi-node Ray cluster driven by a separate Dagster control plane.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | One image for every service: `python:3.12-slim` + uv, `uv sync --frozen --no-default-groups` (no dev/notebook deps), copies `src/`, `configs/`, `scripts/`. |
| `Dockerfile.dockerignore` | BuildKit build-context filter (keeps `.venv`, `data/`, `mlruns/` ... out of the context). |
| `dagster.yaml` | Dagster instance: SQLite storage on the `dagster_storage` volume, `QueuedRunCoordinator`, local compute logs. |
| `workspace.yaml` | Loads the code location `ml_platform.orchestration.definitions`. |
| `healthcheck.py` | Stdlib-only HTTP/process healthcheck (slim image has no curl/procps). |
| `../../docker-compose.yml` | `ray-head`, `ray-worker` (scalable), `ray-worker-gpu` (profile `gpu`), `dagster-webserver`, `dagster-daemon`, `mlflow`. |

Why a single image: the Ray client inside the Dagster containers and the Ray
cluster must run the same Python and Ray versions, and Ray workers must be able
to import `ml_platform`. One image built from `uv.lock` guarantees both.

## Usage

```bash
cp .env.example .env                                  # optional; defaults match
docker compose up --build -d                          # head + 1 worker + Dagster + MLflow
docker compose up -d --scale ray-worker=3             # add CPU workers
docker compose --profile gpu up -d                    # + GPU worker (NVIDIA toolkit)
docker compose ps                                     # healthchecks
docker compose exec dagster-webserver python scripts/run_closed_loop.py
docker compose down            # keep volumes;  `down -v` wipes data/mlflow/dagster state
```

| UI / endpoint | URL |
|---------------|-----|
| Dagster | http://localhost:3000 |
| MLflow | http://localhost:5000 |
| Ray dashboard / Job API | http://localhost:8265 |
| Ray Serve (`ml_platform.inference.serve:app`) | http://localhost:8000 |
| Ray metrics (Prometheus format) | http://localhost:8080/metrics |
| Ray client from the host | `RAY_ADDRESS=ray://localhost:10001` |

## How the services connect

```text
dagster-webserver ─┐   RAY_ADDRESS=ray://ray-head:10001 (Ray client)
dagster-daemon ────┼─────────────────────────► ray-head ◄── ray-worker x N (ray-head:6379)
                   │                              │ Serve proxy :8000
                   └──► mlflow:5000 ◄─────────────┘ MLFLOW_TRACKING_URI=http://mlflow:5000
shared volumes: data → /app/data, reports → /app/reports, artifacts → /app/artifacts
```

* Dagster runs are thin: assets call `RayComputeResource`, which connects to the
  cluster through `RAY_ADDRESS`; DDP training, evaluation tasks and Serve
  deployments execute on `ray-head` / `ray-worker`.
* All containers that touch rounds, reports or checkpoints mount the same
  volumes at the same paths, standing in for a shared filesystem / bucket.
* MLflow runs with `--serve-artifacts`: clients upload artifacts over HTTP to
  the tracking server, which writes them to `/mlflow/artifacts`. Clients never
  need direct access to the artifact store.
* `ray-head` sets `RAY_SERVE_DEFAULT_HTTP_HOST=0.0.0.0` so the Serve proxy is
  reachable on the published port.

## Local vs production storage

| Concern | Here (demo) | Production replacement |
|---------|-------------|------------------------|
| Dagster run/event/schedule storage | SQLite on `dagster_storage` volume | Postgres (`storage.postgres` in `dagster.yaml`; Helm `postgresql.*`) |
| Dagster compute logs | local dir | S3/GCS compute log manager |
| MLflow backend store | `sqlite:////mlflow/mlflow.db` | `postgresql://…/mlflow` (managed Cloud SQL / RDS) |
| MLflow artifacts | `/mlflow/artifacts` volume | `--artifacts-destination s3://…` / `gs://…` / MinIO (`MLFLOW_S3_ENDPOINT_URL`) |
| Rounds, reports, checkpoints | named volumes | object storage (Ray Train `RunConfig(storage_path="s3://…")`) or an RWX PVC |

Postgres/MinIO services are intentionally not part of this compose file; the
single-machine demo does not need them. See `infra/k8s/README.md`.

## Image size

`uv.lock` resolves `torch` from PyPI; on Linux that wheel depends on the
`nvidia-*` CUDA runtime wheels, so the image is several GB even though
CPU-only execution works. The lock is not changed for Docker. A slimmer
CPU-only image would route torch to `https://download.pytorch.org/whl/cpu`
via `[tool.uv.sources]` (a lock change, so a deliberate decision).
