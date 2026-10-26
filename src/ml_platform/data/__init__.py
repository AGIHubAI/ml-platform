"""Scientific data layer: schemas, synthetic oracle, generation, validation, round store."""

from ml_platform.data.datasets import (
    ArrayDataset,
    ImmutableRoundError,
    RoundSequenceError,
    RoundStore,
    content_hash,
    frame_to_records,
    records_to_frame,
    train_val_split,
)
from ml_platform.data.generation import (
    generate_candidate_pool,
    initial_observations,
    make_oracle,
    measure_candidates,
)
from ml_platform.data.normalization import Normalizer
from ml_platform.data.schema import (
    ExperimentRecord,
    RoundManifest,
    feature_columns,
    parse_round_key,
    round_key,
)
from ml_platform.data.synthetic_oracle import SyntheticOracle
from ml_platform.data.validation import DataValidationError, ValidationReport, validate_frame

__all__ = [
    "ArrayDataset",
    "DataValidationError",
    "ExperimentRecord",
    "ImmutableRoundError",
    "Normalizer",
    "RoundManifest",
    "RoundSequenceError",
    "RoundStore",
    "SyntheticOracle",
    "ValidationReport",
    "content_hash",
    "feature_columns",
    "frame_to_records",
    "generate_candidate_pool",
    "initial_observations",
    "make_oracle",
    "measure_candidates",
    "parse_round_key",
    "records_to_frame",
    "round_key",
    "train_val_split",
    "validate_frame",
]
