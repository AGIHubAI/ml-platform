"""The fUSI real-time workload as plain step functions (no Dagster import).

::

    fusi_session[sub00_ses00..]  (partitioned: one per recording session)
      -> fusi_split -> fusi_preprocessor -> fusi_dataset (+ leakage checks)
      -> fusi_decoder (Ray Train DDP) -> fusi_evaluation (+ test-access / gap checks,
         real-time latency run, gate) -> fusi_registered_decoder

Same contract as ``orchestration/pipeline.py``: each step returns a small frozen
``*Info`` record of paths + hashes; the Dagster assets in ``fusi_assets.py`` only bind
ids, call the step and emit metadata. Ray is imported lazily (training, real-time run).

Storage (under ``paths.data_dir/fusi`` unless noted; everything write-once / content
addressed)::

    sessions/<key>/                       raw.zarr, events.parquet, truth.parquet, manifest
    splits/<split_hash>.parquet           block plan
    preprocessors/<state_hash>/           fitted RETROICOR + z-score state (train fold only)
    features/<state_hash>/<key>/          features.zarr
    datasets/<dataset_hash>/              windows.parquet, checks.json, dataset.json
    test_access.json                      test-fold access counter per split hash
    artifacts/fusi/decoders/<run_id>/     model.pt | baseline.npz, preprocessor/
    reports/fusi/<run_id>/                evaluation.json, latency.json
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_platform.config import PlatformConfig
from ml_platform.hashing import hash_config, hash_file
from ml_platform.logging import get_logger
from ml_platform.workloads.fusi_realtime import batch as B
from ml_platform.workloads.fusi_realtime import splits as S
from ml_platform.workloads.fusi_realtime.dataset import (
    DECODER_INPUTS,
    SessionArrays,
    frame_accuracy,
    trial_accuracy,
)
from ml_platform.workloads.fusi_realtime.gates import (
    GateDecision,
    LatencyReport,
    decide_promotion,
)
from ml_platform.workloads.fusi_realtime.models import (
    BASELINE_FILE,
    TRANSFORMER_FILE,
    FusiDecoder,
    PcaLdaDecoder,
    load_any,
    write_baseline_checkpoint,
)
from ml_platform.workloads.fusi_realtime.preprocess import PreprocessorState
from ml_platform.workloads.fusi_realtime.sim import FusiSimulator, frame_labels
from ml_platform.workloads.fusi_realtime.train import predict_session

log = get_logger(__name__)

PREPROCESSOR_SUBDIR = "preprocessor"


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class FusiSessionInfo:
    key: str
    subject: str
    path: Path
    n_frames: int
    events_hash: str
    variables: tuple[str, ...]


@dataclass(frozen=True)
class FusiSplitInfo:
    split_hash: str
    path: Path
    hold_out: str
    gap_frames: int
    sessions: dict[str, list[str]]


@dataclass(frozen=True)
class FusiPreprocessorInfo:
    path: Path
    state_hash: str
    fit_sessions: tuple[str, ...]
    fit_n_frames: int


@dataclass(frozen=True)
class FusiDatasetInfo:
    dataset_hash: str
    path: Path
    split_hash: str
    preprocessor_hash: str
    windows_hash: str
    n_windows: dict[str, int]
    checks: list[dict[str, Any]]

    def check(self, name: str) -> dict[str, Any]:
        return next(c for c in self.checks if c["name"] == name)


@dataclass(frozen=True)
class FusiDecoderInfo:
    run_id: str
    experiment_id: str
    kind: str
    checkpoint_dir: Path
    checkpoint_hash: str
    dataset_hash: str
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusiEvaluationInfo:
    run_id: str
    accuracy: dict[str, float]
    reference: str
    reference_acc: float
    test_accesses: int
    latency: dict[str, Any]
    gate_passed: bool
    gate_reasons: list[str]
    checks: list[dict[str, Any]]
    report_path: Path

    def check(self, name: str) -> dict[str, Any]:
        return next(c for c in self.checks if c["name"] == name)


@dataclass(frozen=True)
class FusiRegistrationInfo:
    version: str
    stage: str
    gate_passed: bool
    reasons: list[str]


# --------------------------------------------------------------------------- paths
def fusi_root(cfg: PlatformConfig) -> Path:
    return cfg.paths.resolved().data_dir / "fusi"


def session_keys(cfg: PlatformConfig) -> list[str]:
    s = cfg.fusi.sessions
    return [
        f"sub{sub:02d}_ses{ses:02d}"
        for sub in range(s.n_subjects)
        for ses in range(s.sessions_per_subject)
    ]


def _parse_key(key: str) -> tuple[int, int]:
    sub, ses = key.split("_")
    return int(sub[3:]), int(ses[3:])


def _check_dicts(checks: list[S.CheckResult]) -> list[dict[str, Any]]:
    return [json.loads(json.dumps(asdict(c), default=str)) for c in checks]


# --------------------------------------------------------------------------- steps
def simulate_session(cfg: PlatformConfig, key: str) -> FusiSessionInfo:
    """Simulate one recording session into the write-once session store."""
    subject, session = _parse_key(key)
    sc = cfg.fusi.sessions
    sim = FusiSimulator(cfg.fusi.sim, subject=subject, session=session, grid=sc.grid, emit="iq")
    n_frames = round(sc.session_s * cfg.fusi.sim.frame_rate_hz)
    store = B.write_simulated_session(
        fusi_root(cfg) / "sessions",
        sim,
        key=key,
        subject=f"sub{subject:02d}",
        n_frames=n_frames,
        chunk_t=cfg.fusi.preprocess.chunk_t,
        store_iq=sc.store == "iq",
        clutter_rank=cfg.fusi.preprocess.clutter_rank,
    )
    return FusiSessionInfo(
        store.key, store.subject, store.path, store.n_frames, store.events_hash, store.variables
    )


def _stores(sessions: list[FusiSessionInfo]) -> dict[str, B.SessionStore]:
    return {s.key: B.SessionStore.load(s.path) for s in sessions}


def plan_split(cfg: PlatformConfig, sessions: list[FusiSessionInfo]) -> FusiSplitInfo:
    metas = [S.SessionMeta(s.key, s.subject, s.n_frames) for s in sessions]
    plan = S.make_split(metas, cfg.fusi.split, cfg.fusi.sim.frame_rate_hz)
    h = plan.split_hash()
    path = fusi_root(cfg) / "splits" / f"{h[:16]}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = plan.frame()
    df["hold_out"] = plan.hold_out
    df["gap_frames"] = plan.gap_frames
    df.to_parquet(path, index=False)
    return FusiSplitInfo(
        h, path, plan.hold_out, plan.gap_frames, {k: plan.sessions(k) for k in S.SPLITS}
    )


def load_plan(info: FusiSplitInfo) -> S.SplitPlan:
    df = pd.read_parquet(info.path)
    blocks = tuple(
        S.Block(str(r["session"]), str(r["subject"]), int(r["start"]), int(r["end"]), r["split"])
        for r in df.to_dict("records")
    )
    return S.SplitPlan(hold_out=info.hold_out, gap_frames=info.gap_frames, blocks=blocks)


def fit_preprocessing(
    cfg: PlatformConfig, sessions: list[FusiSessionInfo], split: FusiSplitInfo
) -> FusiPreprocessorInfo:
    """Fit RETROICOR + normalisation on the train blocks only; save the versioned state."""
    plan = load_plan(split)
    train = [(b.session, b.start, b.end) for b in plan.blocks_of("train")]
    with B.dask_threads():
        state = B.fit_preprocessor_dask(_stores(sessions), train, cfg.fusi.preprocess)
    state = PreprocessorState(**{**state.__dict__, "meta": {"split_hash": split.split_hash}})
    h = state.state_hash()
    path = fusi_root(cfg) / "preprocessors" / h[:16]
    if not path.exists():
        state.save(path)
    return FusiPreprocessorInfo(path, h, state.fit_sessions, state.fit_n_frames)


def build_dataset(
    cfg: PlatformConfig,
    sessions: list[FusiSessionInfo],
    split: FusiSplitInfo,
    preprocessor: FusiPreprocessorInfo,
) -> FusiDatasetInfo:
    """Frozen features for every session + leakage-checked windows + the check results."""
    fc = cfg.fusi
    plan = load_plan(split)
    state = PreprocessorState.load(preprocessor.path)
    feat_root = fusi_root(cfg) / "features" / preprocessor.state_hash[:16]
    with B.dask_threads():
        for s in _stores(sessions).values():
            B.write_features(s, state, feat_root, fc.preprocess.chunk_t)
    windows = S.make_windows(plan, fc.split.window_frames, fc.split.window_stride)
    inputs = sorted(set(DECODER_INPUTS) | {v for s in sessions for v in s.variables})
    checks = S.split_checks(plan, windows, fit_blocks=state.fit_blocks, inputs=inputs, cfg=fc.split)
    w_hash = S.windows_hash(windows)
    d_hash = hash_config(
        {
            "split": split.split_hash,
            "preprocessor": preprocessor.state_hash,
            "windows": w_hash,
            "events": sorted(s.events_hash for s in sessions),
            "sim": fc.sim.model_dump(mode="json"),
        }
    )
    out = fusi_root(cfg) / "datasets" / d_hash[:16]
    out.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out / "windows.parquet", index=False)
    check_rows = _check_dicts(checks)
    (out / "checks.json").write_text(json.dumps(check_rows, indent=2))
    meta = {
        "dataset_hash": d_hash,
        "split_hash": split.split_hash,
        "preprocessor_hash": preprocessor.state_hash,
        "features_dir": str(feat_root),
        "sessions": {s.key: str(s.path) for s in sessions},
        "inputs": inputs,
    }
    (out / "dataset.json").write_text(json.dumps(meta, indent=2))
    n_windows = {k: int((windows.split == k).sum()) for k in S.SPLITS}
    log.info(
        "fusi.dataset",
        dataset_hash=d_hash[:12],
        n_windows=n_windows,
        checks_passed=all(c.passed for c in checks),
    )
    return FusiDatasetInfo(
        d_hash, out, split.split_hash, preprocessor.state_hash, w_hash, n_windows, check_rows
    )


def load_arrays(
    cfg: PlatformConfig, dataset: FusiDatasetInfo, splits: tuple[str, ...]
) -> tuple[dict[str, SessionArrays], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Feature arrays + labels of the sessions used by ``splits`` only (test is loaded only
    by evaluation, which counts the access)."""
    meta = json.loads((dataset.path / "dataset.json").read_text())
    windows = pd.read_parquet(dataset.path / "windows.parquet")
    windows = windows[windows.split.isin(splits)]
    arrays: dict[str, SessionArrays] = {}
    events: dict[str, pd.DataFrame] = {}
    for key in sorted(windows.session.unique()):
        store = B.SessionStore.load(Path(meta["sessions"][key]))
        ev = B.read_events(store)
        feats = B.read_features(Path(meta["features_dir"]) / key / B.FEATURES)
        arrays[key] = SessionArrays(feats, frame_labels(ev, cfg.fusi.sim))
        events[key] = ev
    return arrays, windows, events


def _grid(arrays: dict[str, SessionArrays]) -> tuple[int, int, int]:
    return tuple(next(iter(arrays.values())).features.shape[1:])  # type: ignore[return-value]


def _fit_baseline(
    cfg: PlatformConfig, arrays: dict[str, SessionArrays], windows: pd.DataFrame
) -> PcaLdaDecoder:
    dc = cfg.fusi.decoder
    train = windows[windows.split == "train"]
    xs, ys = [], []
    for key, w in train.groupby("session"):
        a = arrays[str(key)]
        mask = np.zeros(len(a.labels), dtype=bool)
        for s, e in w[["start", "end"]].to_numpy():
            mask[s:e] = True
        xs.append(PcaLdaDecoder.frame_features(a.features, dc.baseline_frames)[mask])
        ys.append(a.labels[mask])
    model = PcaLdaDecoder(
        dc.pca_components, dc.lda_shrinkage, dc.baseline_frames, cfg.fusi.sim.n_directions
    )
    return model.fit(np.vstack(xs), np.concatenate(ys))


def train_decoder(cfg: PlatformConfig, dataset: FusiDatasetInfo) -> FusiDecoderInfo:
    """Train the candidate (Ray Train DDP transformer, or the PCA+LDA baseline) + log to MLflow."""
    from ml_platform.tracking.fusi import fusi_tracker
    from ml_platform.tracking.wandb_backend import WandbMirror

    fc = cfg.fusi
    arrays, windows, _ = load_arrays(cfg, dataset, ("train", "val"))
    grid = _grid(arrays)
    tracker = fusi_tracker(cfg)
    tags = {
        "workload": "fusi_realtime",
        "dataset_hash": dataset.dataset_hash,
        "split_hash": dataset.split_hash,
        "preprocessor_hash": dataset.preprocessor_hash,
        "windows_hash": dataset.windows_hash,
        "fusi_config_hash": fc.config_hash(),
        "decoder_kind": fc.decoder.kind,
    }
    with tracker.start_run(f"fusi_train_{fc.decoder.kind}", tags=tags) as run_id:
        out = cfg.paths.resolved().artifacts_dir / "fusi" / "decoders" / run_id
        out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            fusi_root(cfg) / "preprocessors" / dataset.preprocessor_hash[:16],
            out / PREPROCESSOR_SUBDIR,
            dirs_exist_ok=True,
        )
        baseline = _fit_baseline(cfg, arrays, windows)
        baseline.save(str(out / BASELINE_FILE))
        if fc.decoder.kind == "pca_lda":
            write_baseline_checkpoint(baseline, str(out))
        summary: dict[str, Any] = {"kind": fc.decoder.kind, "grid": list(grid)}
        if fc.decoder.kind == "transformer":
            from ml_platform.workloads.fusi_realtime.train_ray import train_decoder_distributed

            summary.update(
                train_decoder_distributed(
                    cfg,
                    arrays=arrays,
                    windows=windows,
                    grid=grid,
                    n_classes=fc.sim.n_directions,
                    out_dir=out,
                    preprocessor_hash=dataset.preprocessor_hash,
                    dataset_hash=dataset.dataset_hash,
                )
            )
            mirror = WandbMirror(
                fc.tracking.wandb,
                project=fc.tracking.wandb_project,
                run_name=run_id,
                config=fc.scientific_dump(),
            )
            for row in summary["history"]:
                step = int(row["epoch"])
                metrics = {k: v for k, v in row.items() if k != "epoch"}
                tracker.log_metrics(metrics, step=step)
                mirror.log(metrics, step=step)
            mirror.finish()
            tracker.log_metrics(
                {"duration_s": summary["duration_s"], "best_epoch": summary["best_epoch"]}
            )
            tracker.set_tags(
                {
                    "world_size": summary["world_size"],
                    "strategy": summary["strategy"],
                    "params_in_sync": summary["params_in_sync"],
                }
            )
        ckpt_file = out / TRANSFORMER_FILE
        tracker.log_params(
            {
                **{f"decoder.{k}": v for k, v in fc.decoder.model_dump().items()},
                **{f"training.{k}": v for k, v in fc.training.model_dump().items()},
                "grid": list(grid),
                "n_windows_train": dataset.n_windows["train"],
                "n_windows_val": dataset.n_windows["val"],
            }
        )
        tracker.log_dict({"checks": dataset.checks}, "leakage_checks.json")
        tracker.log_artifacts(out, artifact_path="decoder")
        info = FusiDecoderInfo(
            run_id=run_id,
            experiment_id=tracker.experiment_id,
            kind=fc.decoder.kind,
            checkpoint_dir=out,
            checkpoint_hash=hash_file(ckpt_file),
            dataset_hash=dataset.dataset_hash,
            summary={k: v for k, v in summary.items() if k != "workers"},
        )
    return info


def _probs(checkpoint_dir: Path, kind: str, features: np.ndarray) -> np.ndarray:
    """Per-frame class scores; ``kind="pca_lda"`` always uses the dir's ``baseline.npz``."""
    model: FusiDecoder | PcaLdaDecoder
    if kind == "pca_lda":
        model = PcaLdaDecoder.load(str(checkpoint_dir / BASELINE_FILE))
    else:
        _, model = load_any(str(checkpoint_dir))
    if isinstance(model, FusiDecoder):
        return predict_session(model, features)
    return model.decision(PcaLdaDecoder.frame_features(features, model.n_frames))


def _accuracy(
    checkpoint_dir: Path,
    kind: str,
    arrays: dict[str, SessionArrays],
    events: dict[str, pd.DataFrame],
    keys: list[str],
) -> tuple[float, float]:
    """(frame accuracy, trial accuracy) pooled over ``keys`` sessions."""
    preds, labels, trial = [], [], []
    for k in keys:
        p = _probs(checkpoint_dir, kind, arrays[k].features).argmax(1)
        preds.append(p)
        labels.append(arrays[k].labels)
        trial.append(trial_accuracy(p, arrays[k].labels, events[k]))
    return frame_accuracy(np.concatenate(preds), np.concatenate(labels)), float(np.nanmean(trial))


def _bump_test_access(cfg: PlatformConfig, split_hash: str) -> int:
    """Count test-fold accesses per split (persisted; never reset by the pipeline)."""
    path = fusi_root(cfg) / "test_access.json"
    counts = json.loads(path.read_text()) if path.exists() else {}
    counts[split_hash] = int(counts.get(split_hash, 0)) + 1
    path.write_text(json.dumps(counts, indent=2))
    return counts[split_hash]


def _production_reference(
    cfg: PlatformConfig,
    dataset: FusiDatasetInfo,
    test_keys: list[str],
    events: dict[str, pd.DataFrame],
) -> tuple[str, float] | None:
    """Accuracy of the production decoder on *this* test fold, through its own preprocessor.

    Skipped (None) when the production decoder saw any current test session in its
    training fold -- comparing on its training data would flatter it.
    """
    from ml_platform.tracking.fusi import fusi_registry

    prod = fusi_registry(cfg).production_version()
    if prod is None:
        return None
    ckpt = prod.checkpoint_path
    state = PreprocessorState.load(ckpt / PREPROCESSOR_SUBDIR)
    if set(state.fit_sessions) & set(test_keys):
        log.warning(
            "fusi.production_reference_skipped", reason="production trained on test sessions"
        )
        return None
    meta = json.loads((dataset.path / "dataset.json").read_text())
    kind, _ = load_any(str(ckpt))
    arrays = {}
    with B.dask_threads():
        for k in test_keys:
            store = B.SessionStore.load(Path(meta["sessions"][k]))
            feats = B.transform_session(store, state).values
            arrays[k] = SessionArrays(feats, frame_labels(events[k], cfg.fusi.sim))
    acc, _ = _accuracy(ckpt, kind, arrays, events, test_keys)
    return f"production v{prod.version}", acc


def evaluate_decoder(
    cfg: PlatformConfig, dataset: FusiDatasetInfo, decoder: FusiDecoderInfo
) -> FusiEvaluationInfo:
    """Accuracy (train/val/test, candidate vs reference), real-time latency run, gate."""
    from ml_platform.tracking.fusi import fusi_tracker
    from ml_platform.workloads.fusi_realtime.realtime import run_realtime

    fc = cfg.fusi
    arrays, windows, events = load_arrays(cfg, dataset, ("train", "val", "test"))
    keys = {k: sorted(windows.loc[windows.split == k, "session"].unique()) for k in S.SPLITS}
    n_access = _bump_test_access(cfg, dataset.split_hash)
    ckpt = decoder.checkpoint_dir
    acc: dict[str, float] = {}
    for split in S.SPLITS:
        acc[f"{split}_frame_acc"], acc[f"{split}_trial_acc"] = _accuracy(
            ckpt, decoder.kind, arrays, events, keys[split]
        )
        acc[f"baseline_{split}_frame_acc"], _ = _accuracy(
            ckpt, "pca_lda", arrays, events, keys[split]
        )
    prod = _production_reference(cfg, dataset, keys["test"], events)
    if prod is not None:
        reference, reference_acc = prod
    elif decoder.kind == "transformer":
        reference, reference_acc = "baseline (pca_lda)", acc["baseline_test_frame_acc"]
    else:
        reference, reference_acc = "chance", 1.0 / fc.sim.n_directions
    grid = tuple(np.asarray(arrays[keys["test"][0]].features.shape[1:]).tolist())
    report: LatencyReport
    report, _ = run_realtime(
        cfg,
        checkpoint=ckpt,
        preprocessor_dir=ckpt / PREPROCESSOR_SUBDIR,
        grid=grid,  # type: ignore[arg-type]
    )
    eval_checks = [
        S.check_test_access(n_access, fc.split.max_test_accesses),
        S.check_train_test_gap(
            acc["train_frame_acc"], acc["test_frame_acc"], fc.split.max_train_test_gap
        ),
    ]
    dataset_checks = [S.CheckResult(**c) for c in dataset.checks]
    gate: GateDecision = decide_promotion(
        candidate_acc=acc["test_frame_acc"],
        reference_acc=reference_acc,
        reference=reference,
        latency=report,
        checks=[*dataset_checks, *eval_checks],
        cfg=fc.gates,
    )
    out = cfg.paths.resolved().reports_dir / "fusi" / decoder.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))
    (out / "latency.txt").write_text(report.table() + "\n")
    evaluation = {
        "accuracy": acc,
        "reference": reference,
        "reference_acc": reference_acc,
        "test_accesses": n_access,
        "gate": {"passed": gate.passed, "reasons": gate.reasons, **gate.metrics},
        "checks": _check_dicts([*dataset_checks, *eval_checks]),
        "split_hash": dataset.split_hash,
        "dataset_hash": dataset.dataset_hash,
    }
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2, default=str))
    tracker = fusi_tracker(cfg)
    with tracker.start_run(run_id=decoder.run_id):
        tracker.log_metrics({**{f"eval_{k}": v for k, v in acc.items()}, **report.flat_metrics()})
        tracker.log_metrics({"test_set_accesses": n_access, "reference_acc": reference_acc})
        tracker.set_tags(
            {
                "gate_passed": str(gate.passed).lower(),
                "gate_reasons": " | ".join(gate.reasons)[:5000] or "-",
                "eval_reference": reference,
            }
        )
        tracker.log_artifacts(out, artifact_path="evaluation")
    log.info(
        "fusi.evaluation",
        test_acc=round(acc["test_frame_acc"], 3),
        reference=reference,
        reference_acc=round(reference_acc, 3),
        e2e_p99_ms=round(report.end_to_end.p99, 2),
        gate_passed=gate.passed,
    )
    return FusiEvaluationInfo(
        run_id=decoder.run_id,
        accuracy=acc,
        reference=reference,
        reference_acc=reference_acc,
        test_accesses=n_access,
        latency=report.to_dict(),
        gate_passed=gate.passed,
        gate_reasons=gate.reasons,
        checks=_check_dicts(eval_checks),
        report_path=out,
    )


def register_decoder(
    cfg: PlatformConfig, decoder: FusiDecoderInfo, evaluation: FusiEvaluationInfo
) -> FusiRegistrationInfo:
    """Register as candidate; promote only through the gate (``promote_if_passed``)."""
    from ml_platform.tracking.fusi import fusi_registry
    from ml_platform.tracking.fusi import register_decoder as _register

    registry = fusi_registry(cfg)
    grid = tuple(decoder.summary.get("grid", cfg.fusi.sessions.grid))
    version = _register(
        registry,
        decoder.run_id,
        decoder.checkpoint_dir,
        grid=grid,  # type: ignore[arg-type]
        n_classes=cfg.fusi.sim.n_directions,
        tags={
            "dataset_hash": decoder.dataset_hash,
            "decoder_kind": decoder.kind,
            "test_frame_acc": round(evaluation.accuracy["test_frame_acc"], 4),
            "e2e_p99_ms": round(float(evaluation.latency["end_to_end"]["p99"]), 3),
        },
    )
    gate = GateDecision(evaluation.gate_passed, evaluation.gate_reasons)
    stage = registry.promote_if_passed(version, gate)
    log.info("fusi.registered", model_version=version, stage=stage, gate_passed=gate.passed)
    return FusiRegistrationInfo(version, stage, gate.passed, list(gate.reasons))
