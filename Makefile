# ML Platform demo — every target runs through `uv run` (see `make help`).
#
#   make install test          # environment + fast test suite
#   make mlflow   (terminal 1) # MLflow UI      http://localhost:5000
#   make dagster  (terminal 2) # Dagster UI     http://localhost:3000
#   make closed-loop           # 3 closed-loop rounds (Dagster → Ray → PyTorch DDP → MLflow)
#   make serve    (terminal 3) # Ray Serve of the production model on :8000
#   make query                 # query the served model
#
# Variables: CONFIG=configs/distributed.yaml (default: $ML_CONFIG or configs/local.yaml),
#            ROUNDS=3, FRESH=1 (closed-loop from scratch), RAY_ADDRESS=auto (use `make ray`).

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run
CONFIG ?=
ROUNDS ?= 3
FRESH ?=
MLFLOW_PORT ?= 5000
DAGSTER_PORT ?= 3000
RAY_DASHBOARD_PORT ?= 8265
SERVE_URL ?= http://127.0.0.1:8000
NB_OUT ?= artifacts/notebooks

export MLFLOW_DISABLE_AGENT_HINT := 1
export DAGSTER_HOME ?= $(CURDIR)/dagster_home
export RAY_USAGE_STATS_ENABLED ?= 0

CONFIG_ARG := $(if $(CONFIG),--config $(CONFIG),)

.PHONY: help install test test-integration test-smoke ray ray-stop dagster mlflow \
        generate-data train evaluate serve query load-test closed-loop failure-demo \
        lint format typecheck validate notebooks clean fusi-pipeline fusi-bench

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------------- setup & quality
install: ## Create/refresh the virtualenv from uv.lock
	$(UV) sync

test: ## Unit + smoke tests (CPU only; integration/gpu excluded)
	$(RUN) pytest

test-integration: ## Expensive distributed / multi-service tests
	$(RUN) pytest -m integration

test-smoke: ## Quick end-to-end smoke tests (Ray, batch inference, 2-round closed loop)
	$(RUN) pytest -m smoke

lint: ## ruff check + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## ruff format + autofix
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck: ## mypy
	$(RUN) mypy

validate: ## Validate the Dagster code location
	@mkdir -p "$(DAGSTER_HOME)"
	$(RUN) dagster definitions validate -m ml_platform.orchestration.definitions

# ----------------------------------------------------------------------------- services
ray: ## Start a local Ray head node with dashboard (then use RAY_ADDRESS=auto)
	$(RUN) ray start --head --include-dashboard=true --dashboard-host=127.0.0.1 \
		--dashboard-port=$(RAY_DASHBOARD_PORT) --disable-usage-stats
	@echo "Ray dashboard: http://127.0.0.1:$(RAY_DASHBOARD_PORT)  — export RAY_ADDRESS=auto to use it"

ray-stop: ## Stop the local Ray node started by `make ray`
	$(RUN) ray stop

dagster: ## Dagster UI + daemon (http://localhost:3000); DAGSTER_HOME=./dagster_home
	@mkdir -p "$(DAGSTER_HOME)"
	$(RUN) dagster dev -m ml_platform.orchestration.definitions -p $(DAGSTER_PORT)

mlflow: ## MLflow UI/server on the same SQLite db + artifact dir the pipeline writes
	$(RUN) mlflow server --backend-store-uri sqlite:///$(CURDIR)/mlflow.db \
		--artifacts-destination $(CURDIR)/mlartifacts --serve-artifacts \
		--host 127.0.0.1 --port $(MLFLOW_PORT)

# ----------------------------------------------------------------------------- fUSI workload
FUSI_CONFIG ?= configs/fusi-micro.yaml
# the repo-default DAGSTER_HOME exported above must not override the profile's (T7) one
FUSI_ENV := $(if $(filter $(CURDIR)/dagster_home,$(DAGSTER_HOME)),env -u DAGSTER_HOME,)

fusi-pipeline: ## fUSI Dagster pipeline (FUSI_CONFIG=configs/fusi-micro.yaml; state on T7)
	$(FUSI_ENV) $(RUN) python scripts/run_fusi_pipeline.py --config $(FUSI_CONFIG)

fusi-bench: ## fUSI real-time latency benchmark (SECONDS=60, GRID=32)
	$(FUSI_ENV) $(RUN) python scripts/bench_fusi_realtime.py --config $(FUSI_CONFIG) \
		--seconds $(or $(SECONDS),60) --grid $(or $(GRID),32)

# ----------------------------------------------------------------------------- workflow
generate-data: ## Candidate pool + round_000 (Dagster bootstrap_job)
	$(RUN) python scripts/bootstrap.py $(CONFIG_ARG)

train: ## Ray Train DDP on the latest round union; evaluate, register, gate-promote
	$(RUN) python scripts/run_training.py $(CONFIG_ARG) --init-data --register

evaluate: ## Evaluate the production (else newest) model on the latest data; write reports/
	$(RUN) python scripts/run_evaluation.py $(CONFIG_ARG)

closed-loop: ## ROUNDS closed-loop rounds through Dagster (FRESH=1 wipes generated state first)
	$(RUN) python scripts/run_closed_loop.py --rounds $(ROUNDS) $(CONFIG_ARG) $(if $(FRESH),--fresh,)

failure-demo: ## Worker failure -> checkpoint recovery; experiments are recorded, not re-run
	$(RUN) python scripts/demo_failure_recovery.py $(CONFIG_ARG)

serve: ## Serve the *production* model from the registry with Ray Serve (Ctrl-C to stop)
	@resolved=$$($(RUN) python -c "import sys; \
	from ml_platform.config import load_config; \
	from ml_platform.tracking import ModelRegistry; \
	p = ModelRegistry(load_config($(if $(CONFIG),'$(CONFIG)',None))).production_version(); \
	print(p.checkpoint_path, p.version) if p else sys.exit('error: no production model in the registry. ' \
	'Models are promoted only when they pass the evaluation gates: run more rounds (make closed-loop) ' \
	'and check the gate reasons in reports/ or MLflow.')" | tail -n 1); \
	set -- $$resolved; \
	echo "serving production model v$$2 ($$1)"; \
	$(RUN) python -m ml_platform.inference.serve --checkpoint "$$1" --model-version "$$2" $(CONFIG_ARG)

query: ## Query the running model server (/health, /model-info, /predict, /predict_batch)
	$(RUN) python scripts/query_model.py --url $(SERVE_URL)

load-test: ## Concurrent load test against the running model server
	$(RUN) python scripts/load_test.py --url $(SERVE_URL)

notebooks: ## Execute every notebook (outputs to artifacts/notebooks)
	@shopt -s nullglob; nbs=(notebooks/*.ipynb); \
	if [ $${#nbs[@]} -eq 0 ]; then echo "no notebooks yet"; exit 0; fi; \
	mkdir -p $(NB_OUT); \
	$(RUN) jupyter nbconvert --to notebook --execute --output-dir $(NB_OUT) \
		--ExecutePreprocessor.timeout=900 "$${nbs[@]}"

clean: ## Remove caches and all generated state (data/, reports/, artifacts/, MLflow, Dagster runs)
	$(RUN) python scripts/run_closed_loop.py --fresh --rounds 0 $(CONFIG_ARG)
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
