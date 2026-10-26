# CLAUDE.md — ML Platform

Local-first research ML platform: Dagster (asset lineage) → Ray / Ray Train (placement) →
PyTorch DDP (gradient sync) → MLflow (tracking + gated registry) → Ray Serve, closing the loop
through active learning and an immutable, hash-chained round store. Spec: `docs/HANDOFF.md`;
module contracts: `docs/PLAN.md`; user-facing overview: `README.md`.

## Architecture rules

Do not place business/scientific logic inside Dagster definitions.
Do not place cluster-management logic inside model code.
Keep orchestration adapters thin.

## Package boundaries (`src/ml_platform/`)

| Package | May import | Must NOT import |
|---|---|---|
| `data/`, `models/`, `evaluation/`, `active_learning/`, `training/trainer.py`, `training/checkpointing.py`, `inference/predictor.py` | torch, numpy, pandas, pydantic, `config`, `hashing`, `logging` | ray, dagster, mlflow |
| `training/distributed.py`, `inference/batch.py`, `inference/serve.py`, `ray_runtime/` | ray + the pure packages | dagster, mlflow |
| `tracking/` (only place that imports mlflow) | mlflow + pure packages | dagster |
| `orchestration/pipeline.py` | everything above; Ray only via lazy imports inside `RayCompute` | dagster |
| `workloads/fusi_realtime/{sim,preprocess,batch,splits,models,dataset,train,gates,realdata}.py` | torch, numpy, scipy, dask, xarray, zarr, `config`, `hashing`, `logging` | ray, dagster, mlflow |
| `workloads/fusi_realtime/{realtime,train_ray,serve_decoder}.py` | ray (Train / Serve) + the pure fUSI modules | dagster, mlflow |
| `workloads/fusi_realtime/{caltech_like,replay,bandit}.py` | numpy, pandas, the pure fUSI modules | torch, ray, dagster, mlflow |
| `workloads/fusi_realtime/telemetry.py` | opentelemetry api + sdk | ray, dagster, mlflow |
| `storage.py` | stdlib | everything else |
| `orchestration/fusi_pipeline.py` / `fusi_assets.py` | like `pipeline.py` / `assets.py` (Ray lazily inside steps) | dagster / — |
| `orchestration/{assets,resources,jobs,sensors,partitions,definitions}.py` | dagster + `pipeline` | — |

The first row and the pure fUSI modules are enforced by `tests/unit/test_logging_boundaries.py`. `orchestration/pipeline.py` holds one plain
step function per asset; assets only resolve resources, bind log ids, call the step, and emit
metadata. Scripts call the same step functions.

## Commands

```bash
uv sync                      # env from uv.lock (never pip install)
make test                    # unit + smoke (CPU); integration/gpu deselected by pyproject addopts
make test-smoke | make test-integration
make lint | make format | make typecheck | make validate   # ruff / mypy / dagster definitions validate
make closed-loop [ROUNDS=3] [FRESH=1] [CONFIG=configs/distributed.yaml]
make generate-data | make train | make evaluate | make failure-demo
make mlflow | make dagster | make ray | make serve          # long-running; separate terminals
make query | make load-test | make notebooks | make clean
uv run pytest tests/unit/test_evaluation.py -k gates        # single test
```

## Testing policy

- `tests/unit/`: fast, no Ray cluster; every new pure function gets a unit test.
- `tests/smoke/` (`pytestmark = pytest.mark.smoke`): short real Ray / closed-loop runs; in CI.
- `tests/integration/` (`pytest.mark.integration`): multi-worker DDP, Ray Train recovery, Serve,
  Dagster loop; opt-in in CI. `gpu` marker = needs CUDA.
- CI (`.github/workflows/ci.yml`): ruff, mypy, unit, `pytest -m smoke`, `dagster definitions validate`.
- Before handing off: `make lint typecheck test` (and `make validate` if orchestration changed).

## Conventions

- Python 3.12, full type hints, ruff (line length 100), mypy (`make typecheck`), docstrings that
  explain *why*.
- Config: typed pydantic `PlatformConfig` in `config.py`, loaded by `load_config(path|name|None)`
  (`$ML_CONFIG` or `configs/local.yaml`; YAML supports `extends:`). Add a field to the pydantic
  section *and* `configs/local.yaml`; never read raw YAML/dicts elsewhere.
- Logging: `from ml_platform.logging import get_logger, bound_ids`;
  `log = get_logger(__name__)`; event names are dotted (`"training.epoch_end"`) with kwargs.
  Bind correlation ids `run_id, round_id, dataset_id, model_version, ray_job_id` via
  `bound_ids(...)` / `get_logger(name, **ids)`. No `print` in library code (scripts may print).
- Ray: always `ray_runtime.cluster.ensure_ray(address)` — never call `ray.init` directly (it sets
  the runtime env, disables the uv-run worker hook and honours `RAY_ADDRESS`). Pick resources with
  `ray_runtime.resources.select_resources(cfg)`; never hard-code `num_gpus`.
- Dagster gotcha: **no `from __future__ import annotations`** in `orchestration/assets.py`,
  `resources.py`, `sensors.py` (Dagster inspects real annotations of asset/resource/sensor
  signatures and `ConfigurableResource` fields). Other modules use it freely.
- Data is write-once: never modify files under `data/rounds/`; new data = new round via
  `RoundStore.write_round`. Physical-experiment steps get no retry policy.
- Hash everything that defines a result (`hashing.hash_dataframe/hash_config/hash_file`, `git_sha`).
- Promotion only through `ModelRegistry.promote_if_passed(version, gate)`.

## How to add…

**A model**
1. Implement an `nn.Module` in `models/` with a `.spec()` dict like `ResidualMLP.spec()` (`models/mlp.py`).
2. Teach `models.mlp.model_from_spec` to rebuild it (checkpoints store `model_spec`; `Predictor`
   uses it), and select it in `build_model(cfg)` (add a `ModelConfig` field if configurable).
3. `Trainer(cfg, model_factory=...)` also accepts a factory for experiments.
4. Unit tests in `tests/unit/test_models.py` (forward shape, checkpoint round-trip).

**A metric**
1. Pure function `(y, yhat) -> float` in `evaluation/metrics.py` (add to `regression_metrics` if general).
2. Use it in `evaluation/evaluator.py::evaluate`; if it gates, extend `check_gates` + `EvaluationConfig`
   + `configs/local.yaml`. It then flows to `metrics.json`, MLflow `eval_*` and Dagster metadata.
3. For Ray fan-out, register it in `ray_runtime/tasks.py::METRICS`.
4. Test in `tests/unit/test_evaluation.py`.

**A Dagster asset**
1. Write the logic as a plain function in `orchestration/pipeline.py` returning a small frozen
   `*Info` dataclass (paths + hashes, not data); unit-test it without Dagster.
2. Add a thin `@asset` in `orchestration/assets.py` (`partitions_def=rounds_partitions`,
   `group_name=GROUP`, a retry policy only if the step is idempotent) that uses `_ids(context)`,
   calls the step, and returns `Output(info, metadata=...)`.
3. Append it to `ROUND_ASSETS`/`ALL_ASSETS`; add to a job selection in `orchestration/jobs.py`.
4. `make validate`; extend `tests/unit/test_dagster_definitions.py`.

**A Ray workload**
1. Put the remote function/actor in `ray_runtime/tasks.py` (or `inference/batch.py` for inference);
   pure compute stays in the scientific package and is only wrapped here.
2. Resources from `select_resources(cfg).remote_options()`; bound in-flight work with
   `bounded_map`; set `max_retries` only for idempotent work.
3. Expose it as a method on `orchestration/pipeline.py::RayCompute` (+ `ComputeBackend` protocol,
   + pass-through on `RayComputeResource` if assets/notebooks call it). Assets never import ray.
4. Smoke test in `tests/smoke/` using a small local cluster.
