"""Generate notebooks/06_experiment_tracking_and_reproducibility.ipynb."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbhelpers_a import code, md, setup_cell, write  # noqa: E402

cells = [
    md(r"""
    # 06 · Experiment tracking and reproducibility

    **Goal.** Make "which model is this, and can we get it back?" a question with a
    mechanical answer. We record every input that determines a model — code, config,
    data, environment, randomness — train → log → register a model with the platform's
    MLflow adapter, and then **reproduce it from the stored metadata alone** and check the
    result bit-for-bit. Finally we look at what breaks reproducibility (nondeterministic
    kernels, mutable data, world size) and what to do about it.
    """),
    md(r"""
    ## 1. Conceptual model

    A trained model is (approximately) a deterministic function of its inputs:

    $$
    \theta^\ast = \mathcal{T}\big(\underbrace{\text{code}}_{\texttt{git\_sha}},\;
    \underbrace{\text{config}}_{\texttt{config\_hash}},\;
    \underbrace{\text{data}}_{\texttt{dataset\_hash}},\;
    \underbrace{\text{seed}}_{\text{RNG}},\;
    \underbrace{\text{environment}}_{\text{versions, device, world size}}\big)
    $$

    Reproducibility = recording every argument of $\mathcal{T}$ *by content*, not by name.
    "the latest data" or "config v2" are names that can silently change meaning;
    a SHA-256 of the canonical bytes cannot. If any argument is unrecorded, the model is
    not reproducible — merely re-trainable.

    ```text
      git_sha ─────────┐
      config ─ hash ───┤                 ┌── params  (hyper-parameters, seed, world size, sizes)
      round store ─ dataset_hash ─┤      ├── metrics (per-epoch history)
      seed / RNG state ┤  Trainer ──► run├── tags    (hashes, env versions, device)
      env (torch, ray, ─┘    │           ├── inputs  (dataset digest + schema)
        python, device)      ▼           └── artifacts (checkpoint dir, config.json, train_result.json)
                         checkpoint ── checkpoint_hash ──► registry version (candidate → validated → production)
    ```

    | Concept | Identity in this platform |
    |---|---|
    | run identity | MLflow `run_id` (+ correlation ids in logs: `run_id`, `round_id`, `dataset_id`, `ray_job_id`) |
    | code version | `hashing.git_sha()` (`+dirty` suffix if the tree has uncommitted changes) |
    | config | `cfg.config_hash()` = `hashing.hash_config` (canonical sorted-key JSON) of the *scientific* sections only |
    | dataset version | `RoundStore.dataset_hash(round)` — manifest hash chaining all earlier rounds |
    | randomness | `cfg.seed` + every rank's RNG state stored in every checkpoint |
    | artifact | `training.checkpointing.checkpoint_hash` (SHA-256 of `model.pt`) |
    | model lineage | registry version tags `run_id`, `checkpoint_hash`, `lifecycle` |
    """),
    setup_cell("nb06"),
    code(r"""
    import numpy as np
    import pandas as pd
    import torch

    # This notebook must never touch the repo's mlflow.db: an MLFLOW_TRACKING_URI in the environment
    # would win over the config (see tracking.mlflow_client.resolve_tracking_uri), so drop it here.
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    import logging
    logging.getLogger("mlflow").setLevel(logging.ERROR)       # keep MLflow's INFO chatter out of the notebook

    from ml_platform.config import PlatformConfig
    cfg = PlatformConfig.for_tests(WORK).with_overrides(**{"training.epochs": 6})
    print("tracking URI:", cfg.tracking.tracking_uri)
    print("data dir    :", cfg.paths.data_dir)
    """),
    md(r"""
    ## 2. Recording each input

    ### Data version — immutable rounds + content hashes
    Observations are written as **write-once rounds**. Each round's manifest stores the
    content hash of its rows and the hash of the previous manifest, so the manifest hash of
    round $N$ commits to the content of rounds $0..N$ (a hash chain, like git commits).
    """),
    code(r"""
    from ml_platform.data import RoundStore, generate_candidate_pool, initial_observations, make_oracle
    from ml_platform.data import ImmutableRoundError, measure_candidates

    store = RoundStore(cfg.paths.data_dir)
    pool = generate_candidate_pool(cfg)
    oracle = make_oracle(cfg)
    store.write_pool(pool)
    m0 = store.write_round(0, initial_observations(pool, oracle, cfg.data.initial_observations, cfg.seed))
    print("round_000 content hash :", m0.dataset_hash[:16], "| manifest hash:", m0.manifest_hash[:16])

    # rewriting identical content is an idempotent no-op; different content is refused
    same = store.write_round(0, initial_observations(pool, oracle, cfg.data.initial_observations, cfg.seed))
    print("rewrite identical round -> no-op:", same.manifest_hash == m0.manifest_hash)
    try:
        store.write_round(0, initial_observations(pool, oracle, cfg.data.initial_observations, seed=123))
    except ImmutableRoundError as e:
        print("rewrite with different data -> ImmutableRoundError:", str(e)[:90], "…")

    observed = store.observed_ids()
    new_ids = pool[~pool["candidate_id"].isin(observed)].head(50)
    m1 = store.write_round(1, measure_candidates(new_ids, oracle, round_id=1, seed=cfg.seed))
    print("round_001 parent_hashes:", [h[:16] for h in m1.parent_hashes], "(= manifest hash of round_000)")
    print("dataset id for training on rounds 0..0:", store.dataset_hash(0)[:16])
    print("dataset id for training on rounds 0..1:", store.dataset_hash(1)[:16])
    print("chain verified:", store.verify_chain())
    """),
    md(r"""
    ### Config, code and environment
    """),
    code(r"""
    from ml_platform.hashing import git_sha, hash_config
    from ml_platform.tracking import environment_metadata

    print("config hash         :", cfg.config_hash()[:16])
    print("  + epochs changed  :", cfg.with_overrides(**{"training.epochs": 7}).config_hash()[:16])
    print("  + data dir moved  :", cfg.with_overrides(**{"paths.data_dir": "/elsewhere"}).config_hash()[:16])
    print("  hashed sections   :", sorted(cfg.scientific_dump()))
    print("  key order ignored :", hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1}))
    print("git sha             :", git_sha())
    for k, v in environment_metadata().items():
        print(f"  {k:<16}: {v}")
    """),
    md(r"""
    Two caveats visible above:

    * If `git_sha` ends in `+dirty`, the working tree had uncommitted changes: the SHA does
      **not** identify the code that ran. Production training should refuse dirty trees or
      log the diff as an artifact.
    * `config_hash` covers only what determines the science (`seed`, `data`, `model`,
      `training`, `evaluation`, `active_learning` and the number of DDP workers, which sets
      the global batch). `paths`, `tracking`, `serve` and placement details are excluded, so
      moving the data directory (here a temp dir) or pointing at another MLflow server does
      not change it; the full config is still logged as the `config.json` artifact.

    ### Randomness
    The `Trainer` seeds Python/NumPy/PyTorch (`training.config.seed_everything`), enables
    deterministic kernels (`configure_determinism`), derives each epoch's shuffle from
    `(seed, epoch)` and stores the full RNG state in every checkpoint. Same inputs ⇒ same
    bytes; a different seed ⇒ a different model:
    """),
    code(r"""
    from ml_platform.data import ArrayDataset, train_val_split
    from ml_platform.training import Trainer, checkpoint_hash, load_checkpoint

    def train_once(cfg: PlatformConfig, round_id: int, out: Path):
        frame = store.training_frame(round_id)
        ds_hash = store.dataset_hash(round_id)
        tr, va = train_val_split(frame, cfg.training.val_fraction, cfg.seed, cfg.training.val_strategy)
        result = Trainer(cfg).train(ArrayDataset.from_frame(tr, dataset_hash=ds_hash),
                                    ArrayDataset.from_frame(va, dataset_hash=ds_hash), checkpoint_dir=out)
        return result, frame

    def weights_digest(ckpt_dir) -> str:
        import hashlib
        h = hashlib.sha256()
        for k, v in load_checkpoint(ckpt_dir)["model_state"].items():
            h.update(k.encode() + v.numpy().tobytes())
        return h.hexdigest()[:16]

    ra, _ = train_once(cfg, 1, WORK / "seed0_a")
    rb, _ = train_once(cfg, 1, WORK / "seed0_b")
    rc, _ = train_once(cfg.with_overrides(seed=1), 1, WORK / "seed1")
    for name, r in [("seed 0, run a", ra), ("seed 0, run b", rb), ("seed 1       ", rc)]:
        print(f"{name}: checkpoint_hash={r.checkpoint_hash[:16]}  weights={weights_digest(r.checkpoint_path)}  "
              f"val_rmse={r.metrics['val_rmse']:.6f}")
    """),
    md(r"""
    ## 3. Train → log → register

    `tracking.Tracker` is the only place that imports MLflow. `log_train_result` records
    params, per-epoch metrics, the hash tags, environment metadata, the dataset as a run
    *input* (digest + schema, not rows) and artifacts (checkpoint, `config.json`,
    `train_result.json`). `ModelRegistry.register` logs the checkpoint as an
    `mlflow.pyfunc` model and creates a registry version in lifecycle stage `candidate`.
    """),
    code(r"""
    from ml_platform.tracking import ModelRegistry, PromotionError, Tracker
    warnings.filterwarnings("ignore")                          # mlflow pyfunc type-hint advisory

    ROUND = 1
    result, train_frame = train_once(cfg, ROUND, WORK / "ckpt_original")
    tracker = Tracker(cfg.tracking)
    t0 = time.perf_counter()
    with tracker.start_run(f"train_round_{ROUND:03d}", tags={"round_id": ROUND}) as run_id:
        tracker.log_train_result(result, cfg=cfg, dataset_frame=train_frame, dataset_name=f"rounds_0_{ROUND}")
    registry = ModelRegistry(tracker=tracker)
    version = registry.register(run_id, result.checkpoint_path, tags={"round_id": ROUND})
    print(f"run {run_id} logged + registered as {registry.name} v{version} in {time.perf_counter() - t0:.1f}s")
    print("lifecycle stage:", registry.stage(version))
    try:
        registry.set_stage(version, "production")
    except PromotionError as e:
        print("promotion without an evaluation gate refused:", e)
    """),
    code(r"""
    run = tracker.get_run(run_id)
    keys = ["git_sha", "config_hash", "dataset_hash", "checkpoint_hash", "torch_version", "python_version",
            "device.device", "world_size", "round_id"]
    print("tags:");   [print(f"  {k:<16} {run.data.tags.get(k)}") for k in keys]
    print("params (subset):", {k: run.data.params[k] for k in ["seed", "training.epochs", "training.lr", "world_size", "n_train"]})
    print("dataset inputs :", [(d.dataset.name, d.dataset.digest) for d in run.inputs.dataset_inputs])
    print("artifacts      :", sorted(p.name for p in tracker.run_artifact_dir(run_id).iterdir()))
    print("val_rmse history:", [round(m.value, 4) for m in tracker.client.get_metric_history(run_id, "val_rmse")])
    print("registry versions:", registry.list_versions())
    """),
    md(r"""
    ## 4. Reproduce the model from stored metadata

    Pretend the training process, its variables and its checkpoint directory are gone. All
    we have is the **registry version** and the tracking store. Reproduction procedure:

    1. registry version → `run_id` (version tag);
    2. run → `config.json` artifact → `PlatformConfig`; verify `cfg.config_hash()` == `config_hash`
       tag (the hash covers only the *scientific* sections — `PlatformConfig.scientific_dump()` —
       so moving the data dir or switching the MLflow URI does not count as drift);
    3. run → `round_id` + `dataset_hash` tags → rebuild the training frame from the immutable
       round store and verify its hash;
    4. compare `git_sha` and library versions with the current environment (warn on mismatch);
    5. retrain with the recovered config and compare **checkpoint hash** and **predictions**.
    """),
    code(r"""
    del result, train_frame                        # "forget" everything from the original training
    shutil.rmtree(WORK / "ckpt_original")

    mv_tags = registry.version_tags(version)
    src_run = tracker.get_run(mv_tags["run_id"])
    tags, params = src_run.data.tags, src_run.data.params

    # (2) config
    cfg_repro = PlatformConfig.model_validate(json.loads((tracker.run_artifact_dir(src_run.info.run_id) / "config.json").read_text()))
    assert cfg_repro.config_hash() == tags["config_hash"], "config drift"
    # (3) data
    round_id = int(tags["round_id"])
    assert store.dataset_hash(round_id) == tags["dataset_hash"], "dataset drift"
    # (4) code + environment
    env_now = environment_metadata()
    drift = {k: (tags.get(k), v) for k, v in env_now.items() if tags.get(k) != v}
    print("config hash verified, dataset hash verified")
    print("code:", "same git sha" if git_sha() == tags["git_sha"] else f"DIFFERENT ({tags['git_sha']} -> {git_sha()})",
          "| environment drift:", drift or "none")

    # (5) retrain
    repro, _ = train_once(cfg_repro, round_id, WORK / "ckpt_reproduced")
    print("\ncheckpoint_hash original  :", tags["checkpoint_hash"][:32])
    print("checkpoint_hash reproduced:", repro.checkpoint_hash[:32])
    print("bit-identical checkpoint  :", repro.checkpoint_hash == tags["checkpoint_hash"])
    """),
    code(r"""
    from ml_platform.inference import Predictor

    registered = Predictor.from_checkpoint(registry.checkpoint_path(version))   # what serving would load
    reproduced = Predictor.from_checkpoint(repro.checkpoint_path)
    Xq = pool.sample(1000, random_state=0)
    diff = np.abs(registered.predict(Xq) - reproduced.predict(Xq))
    print(f"predictions on 1000 pool candidates: max |registered - reproduced| = {diff.max():.3g}")
    print("registered model info:", {k: registered.model_info()[k] for k in ("epochs_trained", "seed", "trained_world_size")},
          "| dataset:", registered.model_info()["dataset_hash"][:16])
    """),
    md(r"""
    Same config, same data, same seed, same code, same library versions, CPU,
    world size 1 ⇒ the same `model.pt` bytes. That is the strongest form of
    reproducibility; §5 is about when it is not achievable and what to settle for.

    ## 5. Where bit-reproducibility breaks

    **Floating point is not associative.** $(a+b)+c \ne a+(b+c)$ in finite precision, so
    anything that changes the *order* of a reduction changes the last bits — and training
    amplifies last-bit differences over thousands of steps:
    """),
    code(r"""
    x = torch.randn(1_000_000, dtype=torch.float32, generator=torch.Generator().manual_seed(0))
    s1 = x.sum()
    s2 = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(1))].sum()
    s3 = x.view(1000, 1000).sum(0).sum()
    print(f"same numbers, different summation order: {s1.item():.6f} {s2.item():.6f} {s3.item():.6f}")
    print("float64 reference                      :", f"{x.double().sum().item():.6f}")
    """),
    md(r"""
    | Source of nondeterminism | Example | Mitigation |
    |---|---|---|
    | GPU atomics / reduction order | `index_add_`, `scatter_add_`, cuBLAS split-K, some cuDNN convolution algorithms | `torch.use_deterministic_algorithms(True)` (+ `CUBLAS_WORKSPACE_CONFIG=:4096:8`), `cudnn.benchmark=False` — done in `configure_determinism`; accept a speed cost |
    | cuDNN autotuner | picks the fastest kernel per run | `torch.backends.cudnn.benchmark = False` |
    | TF32 / mixed precision | different mantissa widths across GPUs/settings | pin and **record** precision settings |
    | Apple MPS | not bit-reproducible | never chosen automatically (`resolve_device`) |
    | thread count / BLAS build | different blocking → different sums | record versions, set threads; compare with tolerances across machines |
    | world size | effective batch size, sampler shards, all-reduce order (nb 02, 04) | record `world_size`; reproduce with the same one |
    | DDP resume | per-rank RNG streams (nb 04) | checkpoint every rank's RNG state |
    | data loader workers | worker seeding, completion order | seed workers (`worker_init_fn`), deterministic sampler |
    | **data sources** | "SELECT * FROM results" today ≠ yesterday; files overwritten in place | **immutable, content-hashed rounds** (`RoundStore`) |
    | external services | the oracle / lab itself is noisy | record measurements once; never re-measure to "reproduce" (nb 03 `ExperimentSimulator`) |
    | library upgrades | new kernels, changed defaults | lock file (`uv.lock`), record versions as tags |

    When bit-reproducibility is impossible (multi-GPU training with nondeterministic
    kernels, different hardware), redefine the target: **statistical reproducibility** —
    retrain with the recorded inputs over several seeds and check that the evaluation
    metric falls inside the original's confidence interval (notebook 07's bootstrap CIs).
    Seed variation is itself informative: it is a floor on how small a "real" improvement
    between two models can be.
    """),
    code(r"""
    seeds = range(4)
    rmses = [train_once(cfg.with_overrides(seed=s), 1, WORK / f"seedscan_{s}")[0].metrics["val_rmse"] for s in seeds]
    print("val_rmse across seeds:", np.round(rmses, 4), f"-> mean {np.mean(rmses):.4f} ± {np.std(rmses, ddof=1):.4f} (sd)")
    print("an 'improvement' smaller than ~2 sd of seed noise is not distinguishable from luck")
    """),
    md(r"""
    ## 6. Connection to this repository

    | Concept | Where |
    |---|---|
    | tracking adapter (only module importing mlflow), URI resolution, artifact location | `src/ml_platform/tracking/mlflow_client.py` — `Tracker`, `Tracker.log_train_result`, `Tracker.log_dataset`, `resolve_tracking_uri`, `environment_metadata` |
    | registry + gate-driven lifecycle | `src/ml_platform/tracking/registry.py` — `ModelRegistry.register`, `set_stage`, `promote_if_passed`, `checkpoint_path`, `SurrogatePyfunc` |
    | hashes | `src/ml_platform/hashing.py` — `hash_dataframe`, `hash_config`, `hash_file`, `git_sha` |
    | immutable data versions | `src/ml_platform/data/datasets.py` — `RoundStore.write_round`, `dataset_hash`, `verify_chain`, `content_hash`; `data/schema.py::RoundManifest.manifest_hash` |
    | seeds, determinism, RNG snapshots | `src/ml_platform/training/config.py` — `seed_everything`, `configure_determinism`, `get_rng_state`, `set_rng_state` |
    | checkpoint contents + hash | `src/ml_platform/training/checkpointing.py` — `save_checkpoint`, `checkpoint_hash` |
    | what a run records (TrainResult) | `src/ml_platform/training/trainer.py::TrainResult` |
    | tests | `tests/unit/test_tracking.py`, `tests/unit/test_config_hashing.py`, `tests/unit/test_round_store.py` |

    ## 7. Failure modes

    * **Tracking by name, not content** ("latest.parquet", "config_v2") — silently changes meaning.
    * **Mutable data sources** — the training set cannot be rebuilt; the dataset hash in the run no longer matches anything.
    * **Dirty git trees** — `git_sha` points at code that did not run.
    * **Logging to the wrong store** — a stray `MLFLOW_TRACKING_URI` (it wins over config) or a relative `sqlite:///mlflow.db` resolved against the CWD (the adapter anchors it at the repo root).
    * **Unrecorded environment** — a torch/CUDA upgrade changes kernels; without version tags you cannot tell why numbers moved.
    * **Unrecorded world size / precision / device** — the "same" config produces a different model.
    * **Promoting because training finished** — the registry refuses `production` without a passing evaluation gate.
    * **Re-running a physical experiment to reproduce a dataset** — you get *new* noise; reproduce from recorded measurements.
    * **Expecting bit-identical GPU results** without deterministic algorithms — compare with tolerances / statistically.
    * **Config hashes that include machine-specific paths** — equal science, unequal hashes (see §2).

    ## 8. Exercise

    1. Retrain with `training.deterministic=False` and `torch.set_num_threads(1)` vs `(8)`. Is the checkpoint hash still
       stable on CPU? Which of the table's sources applies?
    2. Write `reproduce(version) -> dict` that performs §4 end-to-end and returns a report (config/data/code/env checks,
       checkpoint-hash match, max prediction diff). Make it *fail* when the round store's hash chain is broken
       (`RoundStore.verify_chain`).
    3. `PlatformConfig.scientific_dump()` keeps `distributed.num_workers` but drops `distributed.use_gpu`. Argue both
       choices. Is `training.device` scientific (CPU vs CUDA kernels give different bits) or operational?
    """),
    code(r"""
    shutil.rmtree(WORK, ignore_errors=True)
    print("cleaned up", WORK)
    """),
]

if __name__ == "__main__":
    write("06_experiment_tracking_and_reproducibility", cells)
