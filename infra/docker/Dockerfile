# syntax=docker/dockerfile:1.7
#
# One image for every service in docker-compose.yml:
#   ray-head / ray-worker   -> `ray start ...`
#   dagster-webserver       -> `dagster-webserver ...`
#   dagster-daemon          -> `dagster-daemon run ...`
#   mlflow                  -> `mlflow server ...`
#
# A single image guarantees that the Ray client inside the Dagster containers
# and the Ray cluster run byte-identical Python, Ray, torch and project code
# (Ray client requires matching Python + Ray versions on both sides).
#
# Build (from the repo root):
#   docker build -f infra/docker/Dockerfile -t ml-platform:dev .
#
# NOTE on image size: uv.lock resolves torch from PyPI. On linux/amd64 the PyPI
# torch wheel depends on the nvidia-* CUDA runtime wheels, so the image is
# several GB even though CPU-only execution works fine. The lock is the source
# of truth and is not changed here; a CPU-only production image would pin the
# torch CPU index (https://download.pytorch.org/whl/cpu) via [tool.uv.sources].

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS runtime

# uv is copied from the official distroless image (pinned major/minor).
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    DAGSTER_HOME=/opt/dagster/dagster_home \
    ML_CONFIG=configs/local.yaml

# Minimal OS deps: git (git_sha() provenance), tini (PID 1 signal handling for
# `ray start --block` and the Dagster daemon).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencies only (cached layer; invalidated only when the lock changes).
#    --no-default-groups drops the dev + notebooks groups (jupyter, jax, ...).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-default-groups --no-install-project

# 2) Project source + runtime inputs.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

# 3) Install the project itself into the venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups

# Dagster instance + workspace config. Durable Dagster state lives on a volume
# at /opt/dagster/storage (see dagster.yaml), NOT in DAGSTER_HOME, so the config
# baked into the image is never shadowed by an old volume.
COPY infra/docker/dagster.yaml infra/docker/workspace.yaml ${DAGSTER_HOME}/
COPY infra/docker/healthcheck.py /usr/local/bin/healthcheck.py

RUN mkdir -p /app/data /app/reports /app/artifacts /opt/dagster/storage /mlflow

# Ports: 3000 dagster-webserver, 5000 mlflow, 6379 ray GCS, 8000 ray serve,
#        8080 ray metrics export (Prometheus), 8265 ray dashboard, 10001 ray client.
EXPOSE 3000 5000 6379 8000 8080 8265 10001

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-c", "import ml_platform; print('ml_platform image OK')"]
