# ML Platform — Handoff Specification

## Objective

Create a self-contained repository demonstrating an ML platform suitable for heterogeneous scientific workloads.

The system should demonstrate the complete lifecycle:

```text
scientific data
    ↓
data validation / feature preparation
    ↓
distributed PyTorch training
    ↓
experiment tracking
    ↓
evaluation + comparison
    ↓
model registration
    ↓
distributed inference
    ↓
uncertainty / candidate selection
    ↓
next experimental round
```

The demo must run on public or synthetic data, without cloud credentials or a Kubernetes cluster.

Use synthetic scientific data representing an expensive, noisy experimental system. The abstraction should be intentionally generic so that the same architecture could support molecular design, biophysical modeling, signal processing, imaging, or other computational-science workloads.

The primary goal is not a sophisticated model. The primary goal is to demonstrate good **research infrastructure architecture**.

---

# 1. Design principles

**Reproducibility** — Every dataset version, training configuration, checkpoint, model artifact, evaluation result, and selected experimental batch should be traceable.

**Separation of orchestration concerns** — Dagster owns durable pipeline and data-asset orchestration. Ray owns distributed computation. PyTorch owns model and training semantics. MLflow owns experiment/model metadata. Ray Serve owns inference execution.

**Scientific rather than conventional product ML** — Assume observations are expensive, sparse, noisy, heterogeneous, generated incrementally. The system therefore needs to support iterative experimental rounds and explicit uncertainty.

**Local-first** — Everything should be runnable on one development machine. Distributed execution should still use multiple Ray workers/processes so that execution semantics are realistic. GPU support should be automatic when GPUs exist but not required.

**Production-shaped, demo-sized** — Avoid toy scripts where orchestration is simulated. The implementation should use real Dagster jobs/assets, Ray workers, PyTorch distributed execution, tracking, checkpointing, and Ray Serve.

---

# 2. Demo scientific problem

Create a synthetic closed-loop scientific optimization task. Each candidate experiment has approximately 32 continuous features `x ∈ R^32`. A hidden nonlinear oracle produces a response `y = f(x) + heteroscedastic_noise`. The model never has direct access to `f`.

Initially only a small subset of the candidate population has labels:

```text
candidate pool:       100,000
initial observations:     500
new experiments/round:    100
```

The workflow trains a neural network surrogate model on existing observations. The model predicts experimental response. The active-learning stage ranks unlabeled candidates using uncertainty and predicted value and selects the next experimental batch. The synthetic oracle then produces their noisy measurements. These observations are appended as a new immutable experimental round (`round_000`, `round_001`, `round_002`, ...). The pipeline then retrains.

The purpose is to demonstrate infrastructure supporting `data → model → decision → new data → model`, not to demonstrate a particular biological model.

---

# 3. Repository layout

```text
ml-platform/
├── README.md  CLAUDE.md  pyproject.toml  uv.lock  Makefile  docker-compose.yml  .env.example  .gitignore
├── configs/ {local,distributed,gpu}.yaml
├── src/ml_platform/
│   ├── data/            schema.py synthetic_oracle.py generation.py validation.py datasets.py
│   ├── models/          mlp.py losses.py uncertainty.py
│   ├── training/        trainer.py distributed.py checkpointing.py config.py
│   ├── evaluation/      evaluator.py metrics.py comparison.py reporting.py
│   ├── inference/       predictor.py batch.py serve.py
│   ├── active_learning/ acquisition.py constraints.py loop.py
│   ├── tracking/        mlflow_client.py registry.py
│   ├── ray_runtime/     cluster.py tasks.py resources.py
│   └── orchestration/   definitions.py assets.py jobs.py resources.py sensors.py partitions.py
├── notebooks/ 00_system_map … 13_jax_and_pytorch_execution_models (.ipynb)
├── scripts/ bootstrap.py run_training.py run_evaluation.py run_closed_loop.py query_model.py
├── infra/docker/  infra/k8s/{README.md, raycluster.yaml, rayjob.yaml}
├── reports/.gitkeep  data/.gitkeep
└── tests/{unit,integration,smoke}
```

Use `uv` for environment and lock-file management. Resolve mutually compatible current stable versions and commit the lock file.

---

# 4. Data model

```python
ExperimentRecord(
    experiment_id: str,
    round_id: int,
    features: list[float],
    response: float | None,
    measurement_std: float | None,
    status: Literal["candidate", "selected", "measured", "failed"],
    created_at: datetime,
    provenance: dict,
)
```

Persist experimental rounds independently. Never mutate old rounds. The effective training dataset is `union(round_000 ... round_N)`. This demonstrates scientific lineage and point-in-time reproducibility.

Generate hashes for dataset, training configuration, source revision if available, model artifact. Store hashes as run metadata.

---

# 5. PyTorch model

Modest residual MLP (32 → 256 → 256 → 128 → prediction) with dropout so MC dropout can provide a simple uncertainty estimate. Keep modeling deliberately straightforward; the infrastructure should be more sophisticated than the network.

Implement `train()`, `validation()`, `checkpoint()`, `resume()`, `predict()`, `predict_with_uncertainty()`.

Training must be deterministic given the same seed where PyTorch permits. Record seed, dataset hash, git SHA, hyperparameters, world size, device information, training duration.

---

# 6. Distributed PyTorch training

Real distributed training using PyTorch DistributedDataParallel + Ray Train. Do not merely launch independent Ray tasks.

Demonstrate: DistributedSampler, one model replica per worker, gradient synchronization, checkpoint creation, checkpoint restoration, rank-aware logging, metric aggregation.

Support 1 worker, 2 CPU workers, multiple GPUs when available.

Explain in comments and documentation: *Ray determines where workers execute. PyTorch DDP determines how model replicas synchronize gradients.* The training code should remain usable outside Ray where practical.

---

# 7. Ray architecture

Demonstrate tasks, actors, object references/object store. Use Ray Train for distributed training; Ray tasks for parallel evaluation or data processing; an actor where persistent state makes sense (inference/model-cache worker or experimental simulator).

Include an example showing resource declaration `@ray.remote(num_cpus=2, num_gpus=1)` but dynamically select resources so CPU-only systems keep working.

Document task scheduling, actors, object store, backpressure, resource requests, placement, failure/retry behavior.

---

# 8. Dagster architecture

Use Dagster's asset model rather than one giant imperative job:

```text
candidate_pool → observed_experiments → training_dataset → trained_model → evaluation_report
→ registered_model → candidate_predictions → selected_experiments → new_experimental_results
```

Model experimental rounds using partitions where appropriate. Dagster is responsible for dependencies, lineage, materialization, partitioning, retry policies, scheduling, observability. Dagster calls Ray-based services/functions for expensive distributed operations. Dagster is not responsible for individual GPU worker scheduling. Create a resource abstraction such as `class RayComputeResource(...)` so assets call a stable API.

---

# 9. MLflow

Run MLflow locally. Track parameters, metrics, dataset identifier, artifacts, evaluation report, checkpoints, environment metadata. Register models after evaluation. Lifecycle states conceptually: candidate, validated, production, archived. Do not automatically promote a model merely because training completed; promotion depends on evaluation gates.

---

# 10. Evaluation framework

A first-class package. Generate `metrics.json`, `report.md`, `predictions.parquet`, `comparison.json`. Regression metrics MAE, RMSE, R²; uncertainty diagnostics if practical. Bootstrap confidence intervals for at least one principal metric. Model comparison paired on the same evaluation examples. Configurable gates:

```yaml
evaluation:
  max_rmse: 0.35
  min_improvement_vs_baseline: 0.02
```

A model that fails evaluation remains available for research but does not become the serving model.

---

# 11. Active learning

Acquisition: `score = predicted_response + beta * predictive_uncertainty`. Constraints filter candidates before ranking (feature ranges, maximum experimental cost, duplicate exclusion). Produce a versioned `selected_experiments` artifact. Domain constraints are encoded separately from the ML model.

---

# 12. Ray Serve inference

Deploy the validated model using Ray Serve with `POST /predict`, `POST /predict_batch`, `GET /health`, `GET /model-info`. Support batching. Track request count, batch size, latency, model version. Create a small load-test script. README discusses batching, replicas, CPU/GPU resource assignment, autoscaling, backpressure, model loading, rolling model replacement.

---

# 13. Closed-loop demonstration

`make closed-loop` executes: generate initial observations → materialize training dataset → distributed training → evaluate → register validated model → predict candidate pool → rank → select next batch → query synthetic oracle → create next immutable round. Run at least two rounds. Print a summary: round, observations, model version, validation RMSE, candidate score, selected experiments, elapsed time.

---

# 14. Failure and recovery demonstration

Reproducible scenario: checkpoint after epoch 3 → simulated worker/process failure → resume from checkpoint → finish. (Or a controlled Ray task failure.) Make clear the distinction between *retrying computation* and *re-running a scientific experiment*; experimental materializations must be idempotent and explicitly versioned.

---

# 15. Tests

Deterministic dataset generation; schema validation; immutable rounds; model forward pass; checkpoint save/restore; single-process training; distributed two-worker training; metric calculation; model comparison; evaluation gating; acquisition ranking; Dagster asset graph loading; Ray task execution; Ray Serve health/prediction; one minimal end-to-end closed-loop smoke test. Unit suite fast; expensive tests marked separately.

---

# 16. Makefile interface

`install test test-integration ray dagster mlflow generate-data train evaluate serve query closed-loop lint format clean`. Commands should also work through `uv run`.

---

# 17. Docker Compose

Ray head, Ray worker, Dagster webserver, Dagster daemon, MLflow. Prefer SQLite/local artifacts. Describe Postgres/object storage replacements for production. App also runs without Docker.

---

# 18. Kubernetes / KubeRay

Example manifests (not required for the demo): Dagster control plane → Ray cluster/KubeRay → CPU workers, GPU workers, Ray Serve. Resource requests/limits. Document node selectors, GPU pools, autoscaling, spot/preemptible nodes, persistent artifact storage.

---

# 19. Observability

Structured logging carrying run_id, round_id, dataset_id, model_version, ray_job_id where relevant. Describe production mapping to Prometheus, Grafana, OpenTelemetry, central log storage.

---

# 20. Foundations notebooks

Each notebook: conceptual model; small derivation or diagram; minimal executable example; connection to repository implementation; failure modes; practical exercise.

* **00 System map** — lifecycle; data/compute/workflow-control/metadata/serving planes; what Ray, Dagster, PyTorch, MLflow, Kubernetes, Ray Serve each own; architecture diagram.
* **01 PyTorch training foundations** — tensors, autograd, modules, optimizers, train/eval mode, DataLoader, batching, checkpoint state, mixed precision, GPU execution; manual training loop first.
* **02 Distributed training and DDP** — processes vs threads, rank, world size, process groups, DistributedSampler, gradient all-reduce, synchronous SGD, effective batch size, comm/compute ratio, stragglers; tiny runnable DDP example; why gradients are averaged and effect of world size.
* **03 Ray distributed execution** — tasks, actors, ObjectRefs, object store, scheduling, resources, serialization, backpressure, fault handling; one task-based and one actor-based implementation.
* **04 Ray Train and resource management** — Ray Train ↔ DDP; worker groups, placement, GPU assignment, checkpointing, retries, data movement, elasticity; how a worker observes rank/world size.
* **05 Dagster scientific workflows** — assets, ops, jobs, resources, partitions, sensors, schedules, materializations, lineage, retries; why an experimental dataset is an asset not a job step; Dagster vs Ray scheduling.
* **06 Experiment tracking and reproducibility** — run identity, code version, config, dataset version, environment, randomness, artifacts, registry; reproduce an earlier model from stored metadata; reproducibility under nondeterministic kernels/data.
* **07 Evaluation and statistical comparison** — splits, leakage, baselines, paired evaluation, bootstrap CIs, calibration, distribution shift, slice metrics, thresholds; one aggregate metric can conceal regressions.
* **08 Distributed inference** — online vs batch, replicas, dynamic batching, throughput vs latency, P50/P95/P99, queueing, backpressure, model loading, GPU memory, autoscaling; Ray Serve with varying batch size and concurrency.
* **09 Active learning and closed-loop optimization** — exploration/exploitation, uncertainty sampling, acquisition functions, relation to Bayesian optimization, constraints, batch acquisition, selection bias; multiple rounds against the oracle; plot learning efficiency.
* **10 Sparse, noisy, high-cost scientific data** — measurement noise, aleatoric vs epistemic, replicates, missing data, heteroscedasticity, small-N, batch effects; random splits leaking experimental conditions.
* **11 Distributed systems and fault tolerance** — at-most/at-least/exactly-once, idempotency, checkpointing, retries, partial failure, stragglers, durable vs ephemeral state; retrying computation vs retrying a physical experiment.
* **12 Kubernetes and accelerator scheduling** — pod, node, deployment, job, requests/limits, GPU resources, node pools, affinity, taints/tolerations, autoscaling, persistent storage, KubeRay; map local Ray onto k8s.
* **13 PyTorch and JAX execution models** — eager, autograd, JIT, functional transforms, vmap, sharded execution, state management; same small model in both frameworks; no second training stack.

---

# 21–22. Architecture documentation / Why Dagster + Ray?

README contains a Mermaid diagram (Dagster → data assets / Ray Train / evaluation; Ray Train → PyTorch DDP → MLflow → registry → Ray Serve → candidate scoring → active learning → new round → Dagster) and documents why each framework is present.

Section **"Why Dagster + Ray?"**: Dagster understands *dataset A produces model B evaluated by report C* and stores durable lineage. Ray understands *training needs 4 GPUs, evaluation needs 32 CPU tasks, inference needs 2 persistent replicas*. PyTorch DDP understands *how gradients from process 0 synchronize with 1…N*. Separate layers — visible in source code.

---

# 23. Engineering quality

Type hints, meaningful docstrings, ruff, pytest, mypy/pyright, pre-commit, structured logging, configuration objects, DI at orchestration boundaries. Avoid inheritance overuse; small composable interfaces. Scientific model code imports neither Dagster nor MLflow. Narrow adapters around infrastructure.

# 24. CI

GitHub Actions on PRs: lint, type checks, unit tests, small Ray smoke test, Dagster definitions validation. No GPUs; GPU tests opt-in.

# 25. CLAUDE.md

Concise: architecture, package boundaries, commands, testing policy, conventions, how to add a model / metric / Dagster asset / Ray workload. Explicitly: *Do not place business/scientific logic inside Dagster definitions. Do not place cluster-management logic inside model code. Keep orchestration adapters thin.*

# 26. Implementation order

1 package + synthetic dataset · 2 single-process training · 3 evaluation · 4 MLflow · 5 Ray distributed training · 6 Dagster · 7 active learning · 8 Ray Serve · 9 failure/recovery · 10 Docker Compose · 11 Kubernetes · 12 notebooks · 13 docs. Run relevant tests at the end of each stage.

# 27. Acceptance criteria

`uv sync` works · `make test` passes CPU-only · distributed training launches ≥2 worker processes · DDP used · Dagster shows lineage · MLflow has params, metrics, dataset ID, checkpoints, eval artifacts · failed evaluation does not promote · Ray Serve endpoints work · `make closed-loop` runs ≥2 rounds · rounds immutable and traceable · checkpoint restorable · clear Dagster/Ray separation · all notebooks execute from a fresh env · README maps local design to Kubernetes/KubeRay.

# 28. Final demonstration

README ends with a 5–10 minute sequence: `uv sync; make test; make mlflow; make dagster; make closed-loop; make serve; uv run python scripts/query_model.py`, and where to inspect Dagster lineage, Ray worker execution, MLflow history, evaluation reports, registry, selected candidates, Ray Serve endpoint.
