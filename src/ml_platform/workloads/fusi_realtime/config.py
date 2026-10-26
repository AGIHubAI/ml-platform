"""Typed configuration of the ``fusi_realtime`` workload (the ``fusi:`` section of the YAML).

Pure pydantic, no framework imports, so `ml_platform.config.PlatformConfig` can embed
it as ``cfg.fusi`` without pulling Dask/Ray/Torch into config loading.

The section is deliberately *not* part of ``PlatformConfig.scientific_dump``: the
closed-loop surrogate's ``config_hash`` must not change when a fUSI knob changes.
The workload hashes its own section instead (`FusiConfig.config_hash`).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class SimConfig(_Section):
    """Synthetic volumetric functional-ultrasound source (see ``sim.py``)."""

    grid: tuple[int, int, int] = (32, 32, 32)
    frame_rate_hz: float = 10.0
    # Compound frames per volume used by the SVD clutter filter. Real scanners use
    # ~100-300; 8 keeps the ensemble cheap while leaving tissue clutter low-rank.
    n_ensemble: int = 8
    n_directions: int = 8
    signal_mode: Literal["hemodynamic", "reporter"] = "hemodynamic"
    # Hemodynamic response: gamma-shaped HRF peaking ~1.5 s after neural onset
    # (fUSI blood-volume lag of 1-2 s, Norman et al. 2021).
    hrf_peak_s: float = 1.5
    hrf_width_s: float = 0.9
    # ASSUMPTION: first-order kinetics of an acoustic calcium reporter. The sources
    # read report no in-vivo time constant; 0.2 s sits in the 100-300 ms range used here.
    reporter_tau_s: float = 0.2
    # Relative blood-volume change at full tuning (power Doppler changes of a few % in NHP PPC).
    signal_gain: float = 0.2
    planning_fraction: float = 0.18  # share of voxels in the tuned "planning region"
    cardiac_hz: float = 1.2
    respiratory_hz: float = 0.25
    cardiac_amp: float = 0.06
    respiratory_amp: float = 0.04
    # Power-Doppler speckle ~ Gamma(looks)/looks: the number of decorrelated blood samples
    # per volume (real ensembles ~100-300 frames; 64 is a conservative middle).
    speckle_looks: float = 64.0
    thermal_noise: float = 0.02
    drift_per_min: float = 0.05
    clutter_db: float = 30.0  # tissue clutter power above blood signal
    # Task timing (memory-guided planning, Griggs et al. 2024): cue -> plan/hold -> rest.
    plan_s: float = 3.0
    rest_s: float = 1.5
    # Simulated user: probability of attending the cue, plus learning from feedback.
    lapse_rate: float = 0.05
    feedback_learning_rate: float = 0.15
    max_engagement: float = 1.5
    seed: int = 0


class SessionsConfig(_Section):
    """How many sessions/subjects the batch pipeline simulates (Dagster partitions)."""

    n_subjects: int = 1
    sessions_per_subject: int = 12
    session_s: float = 90.0
    # Batch volumes default smaller than the real-time 32^3 so the full
    # simulate -> Zarr -> Dask pipeline stays a few hundred MB on a laptop.
    grid: tuple[int, int, int] = (16, 16, 16)
    # "doppler": the SVD clutter filter runs at acquisition (as on a scanner) and Zarr
    # holds power Doppler (~15 MB/session at 16^3). "iq": Zarr holds the IQ ensembles
    # (~16x larger) and the Dask batch job runs the clutter filter.
    store: Literal["doppler", "iq"] = "doppler"


class PreprocessConfig(_Section):
    clutter_rank: int = 1  # singular components removed (tissue) per ensemble
    drift_tau_s: float = 20.0  # causal EMA baseline removed from log power Doppler
    retroicor_order: int = 2  # Fourier order of the cardiac/respiratory regressors (Glover 2000)
    eps: float = 1.0e-6
    chunk_t: int = 32  # Dask chunk length along time


class SplitConfig(_Section):
    """Leakage-safe split policy; every field is enforced by an asset check."""

    hold_out: Literal["session", "subject", "time_block"] = "session"
    n_val_sessions: int = 1
    n_test_sessions: int = 1
    gap_s: float = 10.0  # minimum temporal gap between train and test blocks in one session
    window_frames: int = 24  # decoder input length (frames) per training sequence
    window_stride: int = 12
    # Decoder input variables; anything else in a session store is refused.
    allowed_features: tuple[str, ...] = ("doppler", "iq", "cardiac_phase", "respiratory_phase")
    max_test_accesses: int = 5
    max_train_test_gap: float = 0.35  # accuracy points; larger -> overfit, promotion blocked


class DecoderConfig(_Section):
    kind: Literal["transformer", "pca_lda"] = "transformer"
    # patch edge in voxels (int = cube, or per axis for anisotropic grids such as a 2-D
    # imaging plane with y = 1) -> prod(grid / patch) spatial tokens per volume
    patch: int | tuple[int, int, int] = 8
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    attn_window: int = 24  # causal sliding window in frames; KV cache size per layer
    dropout: float = 0.3
    pca_components: int = 32  # baseline
    lda_shrinkage: float = 0.2
    baseline_frames: int = 6  # baseline averages the last N frames


class FusiTrainingConfig(_Section):
    epochs: int = 8
    # GLOBAL batch: each of num_workers ranks gets batch_size / num_workers windows per
    # step, so changing the worker count does not change what one step sees.
    batch_size: int = 16
    lr: float = 2.0e-3
    weight_decay: float = 0.1
    input_noise: float = 1.0  # Gaussian augmentation on z-scored features (std units)
    num_workers: int = 2  # Ray Train DDP world size
    # "fsdp2" shards parameters with torch.distributed.fsdp.fully_shard (see train.py).
    strategy: Literal["ddp", "fsdp2"] = "ddp"


class RealtimeConfig(_Section):
    seconds: float = 20.0
    warmup_frames: int = 10
    # A frame older than this when a stage starts is dropped and counted, never queued.
    late_ms: float = 100.0
    max_in_flight: int = 3  # pipeline depth; a new frame beyond it is dropped
    torch_threads: int = 2


class TelemetryConfig(_Section):
    """OpenTelemetry for hot path, training, Dagster assets and Serve (``telemetry.py``)."""

    exporter: Literal["none", "console", "file", "otlp"] = "none"
    file_dir: str | None = None  # "file": one JSON-lines file per process in this dir
    otlp_endpoint: str | None = (
        None  # "otlp": OTLP/HTTP endpoint (else $OTEL_EXPORTER_OTLP_ENDPOINT)
    )


class FusiServeConfig(_Section):
    """Ray Serve decoder deployment (stateless window decode; ``serve_decoder.py``)."""

    num_replicas: int = 2
    max_batch_size: int = 16
    batch_wait_timeout_s: float = 0.005


class GateConfig(_Section):
    """Promotion gate. 130 ms: feedback delay estimated in one monkey (Golub et al., EMBC 2012)."""

    p99_budget_ms: float = 130.0
    stage_budget_ms: float = 80.0
    max_drop_fraction: float = 0.0
    min_accuracy_improvement: float = 0.0
    min_accuracy: float = 0.2  # chance is 1/8 = 0.125


class FusiTrackingConfig(_Section):
    experiment: str = "ml-fusi-realtime"
    registered_model: str = "ml-fusi-decoder"
    wandb: bool = False  # optional second backend; no-op unless installed and configured
    wandb_project: str = "ml-fusi-realtime"


class CaltechLikeConfig(_Section):
    """Simulator calibrated to the public Caltech fUS-BMI structure (``caltech_like.py``).

    Every value is either confirmed from the dataset description / authors' code or an
    explicit assumption; the table lives in the ``caltech_like`` module docstring.
    """

    frame_rate_hz: float = 2.0  # confirmed: dataset description (real-time 2 Hz power Doppler)
    n_lateral: int = 128  # ASSUMPTION: plane width in voxels (divisible by patch 8)
    n_depth: int = 96  # ASSUMPTION: plane depth in voxels (divisible by patch 8)
    n_directions: int = 8  # paper: 8 targets; 2 gives the binary variant
    # Trial structure (seconds). State names from rt_fUS_BMI; durations are ASSUMPTIONS.
    fixation_s: float = 1.0
    cue_s: float = 1.0
    memory_s: float = 2.0
    target_acquire_s: float = 1.0
    target_hold_s: float = 1.0
    reward_s: float = 0.5
    iti_s: float = 2.0
    hrf_peak_s: float = 1.5  # hemodynamic lag 1-2 s (Norman et al. 2021)
    hrf_width_s: float = 0.9
    signal_gain: float = 0.2  # ASSUMPTION
    speckle_looks: float = 16.0  # ASSUMPTION: noisier than the default simulator
    cardiac_amp: float = 0.02  # ASSUMPTION: pulsatility averaged within a 0.5 s frame
    respiratory_amp: float = 0.02  # ASSUMPTION
    lapse_rate: float = 0.05  # ASSUMPTION
    seed: int = 0


class DecoderArmConfig(_Section):
    """One PCA+LDA decoder arm of the bandit (see ``bandit.py``)."""

    name: str
    n_frames: int = 3  # causal mean over the last N frames
    n_components: int = 32
    shrinkage: float = 0.2
    validated: bool = True  # unvalidated arms are masked out (hard safety constraint)


def _default_arms() -> list[DecoderArmConfig]:
    return [
        DecoderArmConfig(name="lda_w1_k32", n_frames=1, n_components=32),
        DecoderArmConfig(name="lda_w3_k32", n_frames=3, n_components=32),
        DecoderArmConfig(name="lda_w5_k16", n_frames=5, n_components=16, shrinkage=0.5),
        DecoderArmConfig(name="weak_k2", n_frames=1, n_components=2, shrinkage=0.9),
        DecoderArmConfig(
            name="candidate_unvalidated", n_frames=4, n_components=24, validated=False
        ),
    ]


class BanditConfig(_Section):
    """Closed-loop contextual bandit over (decoder arm, feedback gain) actions."""

    policy: Literal["linucb", "thompson", "uniform"] = "linucb"
    alpha: float = 1.0  # LinUCB exploration width
    thompson_v: float = 0.5  # LinearThompson posterior scale
    n_trials: int = 200
    seed: int = 0
    arms: list[DecoderArmConfig] = Field(default_factory=_default_arms)
    gains: tuple[float, ...] = (1.0, 1.5, 2.0)
    max_gain: float = 1.5  # HARD limit: actions with a larger gain are masked, not penalised
    context_window: int = 10  # trials in the recent-success-rate feature
    trial_window_s: float = 4.0  # cue -> end of target acquire
    # FittedSimEnv ASSUMPTION: gain g scales the evoked amplitude by
    # 1 + gain_sensitivity*(g-1) - gain_overload*(g-1)^2 (inverted U, no source).
    gain_sensitivity: float = 0.5
    gain_overload: float = 0.6
    lapse_rate: float = 0.05


class FusiConfig(_Section):
    sim: SimConfig = Field(default_factory=SimConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    training: FusiTrainingConfig = Field(default_factory=FusiTrainingConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    serve: FusiServeConfig = Field(default_factory=FusiServeConfig)
    gates: GateConfig = Field(default_factory=GateConfig)
    tracking: FusiTrackingConfig = Field(default_factory=FusiTrackingConfig)
    caltech_like: CaltechLikeConfig = Field(default_factory=CaltechLikeConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)

    def scientific_dump(self) -> dict[str, Any]:
        """What defines a fUSI result (placement/budget sections dropped)."""
        full = self.model_dump(mode="json")
        return {
            k: full[k] for k in ("sim", "sessions", "preprocess", "split", "decoder", "training")
        }

    def config_hash(self) -> str:
        from ml_platform.hashing import hash_config

        return hash_config(self.scientific_dump())

    @classmethod
    def for_tests(cls) -> FusiConfig:
        """Tiny sizes for unit/smoke tests (seconds on a laptop CPU)."""
        return cls(
            sim=SimConfig(grid=(8, 8, 8), n_ensemble=4, plan_s=2.0, rest_s=1.0),
            sessions=SessionsConfig(sessions_per_subject=4, session_s=30.0, grid=(8, 8, 8)),
            split=SplitConfig(window_frames=12, window_stride=6, gap_s=5.0),
            decoder=DecoderConfig(patch=4, d_model=32, n_heads=2, n_layers=2, attn_window=12),
            training=FusiTrainingConfig(epochs=3, batch_size=8, num_workers=2),
            realtime=RealtimeConfig(seconds=3.0, warmup_frames=3),
            gates=GateConfig(p99_budget_ms=1000.0, stage_budget_ms=1000.0, max_drop_fraction=0.5),
        )
