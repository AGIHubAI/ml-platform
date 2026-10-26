# Implementation Plan & Module Contracts

This is the working plan the implementation team follows. It fixes the
cross-module contracts up front so stages can be built by different agents
without drifting apart. The product spec lives in `docs/HANDOFF.md`.

## Pinned toolchain (resolved, see `uv.lock`)

Python 3.12 · torch 2.14 · ray 2.58 (train, serve) · dagster 1.13 ·
mlflow 3.16 · jax 0.11 (notebooks only) · pydantic 2 · structlog.

## Stages, owners, and ordering

| Wave | Stage (handoff §26) | Owner agent | Depends on |
|------|---------------------|-------------|------------|
| 0 | repo, lock, contracts, CI skeleton | lead | – |
| 1 | 1–3: data, model, single-process training, evaluation | `core-ml` | 0 |
| 1 | 10–11, 24: Docker Compose, KubeRay manifests, CI, pre-commit | `infra` | 0 (parallel) |
| 2 | 4, 5, 9: MLflow tracking/registry, Ray runtime, Ray Train DDP, failure/recovery | `distributed` | 1 |
| 2 | 7, 8: active learning, inference (batch + Ray Serve) | `decision-serving` | 1 (parallel with `distributed`) |
| 3 | 6, 13-loop: Dagster assets/jobs/sensors, scripts, Makefile, `make closed-loop` | `orchestration` | 2 |
| 4 | 12: notebooks 00–06 / 07–13 | `notebooks-a`, `notebooks-b` | 3 |
| 4 | 13: README, CLAUDE.md, architecture docs | `docs` | 3 |
| 5 | acceptance verification | lead | all |

### Status

| Wave | Status |
|------|--------|
| 0–3 | done — package, data/model/training/evaluation, MLflow + Ray + DDP, active learning + serving, Dagster assets, scripts, Makefile, infra, CI |
| 4 | done — notebooks 00–13 (executed outputs committed), README.md, CLAUDE.md |
| 5 | done — hardening fixes; acceptance verified from a fresh clone (see below) |

Rules for every agent:

* Only edit files in your ownership list. Need a change elsewhere? Report it.
* Run what you write. Every stage ends with `uv run pytest` (+ its integration
  tests) and `uv run ruff check`. Do not leave generated code unexecuted.
* Do not commit; the lead commits and pushes after each wave.
* Scientific code (`data`, `models`, `training/trainer.py`, `evaluation`,
  `active_learning`) must not import dagster, mlflow, or ray.

## Package boundaries

```
data/            schemas, oracle, generation, immutable round store     (pure)
models/          ResidualMLP, losses, MC-dropout uncertainty           (pure torch)
training/        config, Trainer (single-process, DDP-aware), checkpoints (pure torch)
training/distributed.py   Ray Train wrapper around Trainer             (ray)
evaluation/      metrics, bootstrap, paired comparison, gates, reports (pure)
active_learning/ acquisition, constraints, loop step                   (pure)
inference/       predictor (pure), batch (ray tasks/actors), serve (ray serve)
tracking/        narrow MLflow adapter + registry lifecycle            (mlflow)
ray_runtime/     cluster init, resource selection, generic tasks/actors (ray)
orchestration/   thin Dagster assets/resources calling the above       (dagster)
```

## Shared contracts

### Config — `ml_platform.config` (owned by `core-ml`)

`PlatformConfig` (pydantic) loaded by `load_config(path | None)` from
`configs/{local,distributed,gpu}.yaml`, env var `ML_CONFIG` selects the file.
Sections:

```yaml
seed: 0
paths:   {data_dir: data, reports_dir: reports, artifacts_dir: artifacts}
data:    {n_features: 32, pool_size: 100000, initial_observations: 500,
          batch_size_per_round: 100, pool_seed: 0}
model:   {hidden_dims: [256, 256, 128], dropout: 0.1}
training:{epochs: 30, batch_size: 64, lr: 1.0e-3, weight_decay: 1.0e-4,
          val_fraction: 0.2, deterministic: true, checkpoint_every: 1}
distributed: {num_workers: 2, use_gpu: auto, cpus_per_worker: 1}
evaluation: {max_rmse: 0.35, min_improvement_vs_baseline: 0.02,
             bootstrap_samples: 1000, mc_samples: 30}
active_learning: {beta: 1.0, max_cost: null, feature_bounds: [-3.0, 3.0]}
tracking: {tracking_uri: "sqlite:///mlflow.db", experiment: ml-closed-loop,
           registered_model: ml-surrogate}
serve: {num_replicas: 1, max_batch_size: 64, batch_wait_timeout_s: 0.01, port: 8000}
```

Note: `max_rmse` is on standardized responses (targets are z-scored with
training statistics stored in the checkpoint).

### Data — `ml_platform.data`

* `schema.ExperimentRecord` (pydantic, fields per handoff §4) +
  `schema.RoundManifest{round_id, n_records, dataset_hash, parent_hashes,
  created_at, provenance}`.
* `synthetic_oracle.SyntheticOracle(seed, n_features)` with
  `.mean(X) -> ndarray`, `.noise_std(X) -> ndarray` (heteroscedastic),
  `.measure(X, rng) -> (y, std)`, `.cost(X) -> ndarray`.
* `generation.generate_candidate_pool(cfg) -> DataFrame` columns
  `candidate_id, f00..f31, cost`; deterministic in `pool_seed`.
* `generation.initial_observations(pool, oracle, n, seed) -> list[ExperimentRecord]`.
* `datasets.RoundStore(root)`: `write_round(round_id, records, provenance) -> RoundManifest`
  (write-once: identical content → idempotent no-op returning the existing
  manifest; different content → `ImmutableRoundError`), `read_round(id)`,
  `list_rounds()`, `latest_round_id()`, `training_frame(up_to_round) -> DataFrame`
  (union of rounds, measured only), `dataset_hash(up_to_round) -> str`.
  Layout: `data/rounds/round_000/{observations.parquet, manifest.json}`, pool at
  `data/candidate_pool/{pool.parquet, manifest.json}`.
* `validation.validate_frame(df) -> ValidationReport` (finite, ranges, dupes, schema).
* Hashing helpers in `ml_platform.hashing`: `hash_dataframe`, `hash_config`,
  `hash_file`, `git_sha()`.

### Model / training

* `models.mlp.ResidualMLP(in_dim, hidden_dims, dropout)`; `forward(x) -> (B,)`.
* `models.uncertainty.mc_dropout_predict(model, X, n_samples) -> (mean, std)`.
* `training.trainer.Trainer(cfg, model_factory)`: `train(train_ds, val_ds, *,
  checkpoint_dir, resume_from=None, fail_at_epoch=None, on_epoch_end=None) -> TrainResult`,
  `validate()`, `save_checkpoint()`, `load_checkpoint()`. Works single-process and
  inside a torch.distributed process group (wraps in DDP, uses DistributedSampler,
  only rank 0 writes checkpoints, metrics all-reduced).
* `training.checkpointing`: checkpoint = dir with `model.pt` holding
  `{model_state, optimizer_state, epoch, metrics, normalizer, config, rng_state}`.
* `TrainResult{checkpoint_path, metrics, epochs_completed, duration_s, world_size,
  device, seed, dataset_hash}`.
* `inference.predictor.Predictor.from_checkpoint(path)`: `predict(X)`,
  `predict_with_uncertainty(X, n_samples)` — returns de-standardized values.

### Evaluation

* `evaluation.metrics`: `mae, rmse, r2, nll_gaussian, coverage(y, mu, sigma, z)`,
  `bootstrap_ci(metric_fn, y, yhat, n, seed) -> (point, lo, hi)`.
* `evaluation.comparison.paired_compare(y, pred_a, pred_b, metric, n_boot) -> ComparisonResult`.
* `evaluation.evaluator.evaluate(predictor, eval_frame, baseline_predictions, cfg) -> EvaluationResult`
  with `.metrics`, `.gate: GateDecision{passed, reasons}`, `.write(out_dir)` producing
  `metrics.json, report.md, predictions.parquet, comparison.json`.
* Baseline: previous production model if any, else a mean/ridge baseline.

### Tracking (adapter; nothing else imports mlflow)

`tracking.mlflow_client.Tracker(cfg.tracking)`: `start_run(name, tags) ctx`,
`log_params`, `log_metrics`, `log_artifacts(dir, path)`, `set_tags`.
`tracking.registry.ModelRegistry`: `register(run_id, artifact_path) -> version`,
`set_stage(version, "candidate"|"validated"|"production"|"archived")` (implemented
via MLflow aliases + a `lifecycle` version tag), `production_version()`,
`promote_if_passed(version, gate)`.

### Ray runtime

`ray_runtime.cluster.ensure_ray(cfg)` (idempotent init; local or `RAY_ADDRESS`),
`ray_runtime.resources.select_resources(cfg) -> WorkerResources` (GPU auto-detect),
`ray_runtime.tasks`: parallel bootstrap/eval tasks and a stateful
`ExperimentSimulator` actor that owns the oracle.
`training.distributed.train_distributed(cfg, round_id, ...) -> TrainResult` via
`ray.train.torch.TorchTrainer`.

### Active learning

`active_learning.acquisition.ucb(mu, sigma, beta)`, `rank_candidates(...)`;
`constraints`: composable `Constraint` callables (`FeatureBounds`, `MaxCost`,
`ExcludeIds`); `loop.select_batch(pool, predictions, observed_ids, cfg) -> DataFrame`.

### Orchestration

Dagster assets (dynamic partitions by round, e.g. `round_001`): `candidate_pool`,
`observed_experiments`, `training_dataset`, `trained_model`, `evaluation_report`,
`registered_model`, `candidate_predictions`, `selected_experiments`,
`new_experimental_results`. Resources: `RayComputeResource`, `TrackingResource`,
`RoundStoreResource`. Assets contain only glue.

## Acceptance checklist (handoff §27)

Verified by the lead from a fresh clone of the public repo (2026-09-27):

- `uv sync` ✓ · `make test` ✓ (105 passed, CPU-only) · `make test-integration` ✓ (11 passed)
- `make closed-loop ROUNDS=2` ✓ — 2 DDP workers per round (distinct pids), rounds
  `round_001`/`round_002` written read-only with parent-hash chain and selection provenance
- MLflow: params, `dataset_id`, metrics, checkpoint, evaluation artifacts per run ✓
- Round 0 fails its gate and stays `candidate`; round 1 is promoted to `production` ✓
- `make serve` + `scripts/query_model.py`: `/health`, `/model-info`, `/predict`, `/predict_batch` ✓
- Bit-exact checkpoint resume (single-process and 2-worker DDP) ✓ — `make failure-demo`
- Dagster lineage via `dagster definitions validate` + GraphQL asset graph ✓
- `make notebooks`: all 14 notebooks execute ✓ · GitHub Actions CI green ✓
