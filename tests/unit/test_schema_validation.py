from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ml_platform.data import (
    DataValidationError,
    ExperimentRecord,
    RoundManifest,
    feature_columns,
    frame_to_records,
    parse_round_key,
    records_to_frame,
    round_key,
    validate_frame,
)


def _rec(**kw: object) -> ExperimentRecord:
    base: dict[str, object] = {
        "experiment_id": "cand_000001",
        "round_id": 0,
        "features": [0.1] * 32,
        "response": 1.0,
        "measurement_std": 0.2,
        "status": "measured",
    }
    base.update(kw)
    return ExperimentRecord(**base)  # type: ignore[arg-type]


def test_valid_record_roundtrip() -> None:
    r = _rec()
    df = records_to_frame([r])
    assert list(df.columns[:3]) == ["experiment_id", "round_id", "f00"]
    back = frame_to_records(df)[0]
    assert back.features == r.features and back.response == r.response


@pytest.mark.parametrize(
    "bad",
    [
        {"features": [math.nan] + [0.0] * 31},
        {"features": [math.inf] * 32},
        {"features": []},
        {"status": "done"},
        {"status": "measured", "response": None},
        {"status": "measured", "response": math.nan},
        {"measurement_std": -1.0},
        {"round_id": -1},
        {"experiment_id": ""},
        {"unexpected": 1},
    ],
)
def test_bad_records_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _rec(**bad)


def test_candidate_without_response_ok() -> None:
    r = _rec(status="candidate", response=None, measurement_std=None)
    assert r.response is None


def test_feature_columns_and_round_keys() -> None:
    assert feature_columns(32)[0] == "f00" and feature_columns(32)[-1] == "f31"
    assert round_key(3) == "round_003" and parse_round_key("round_003") == 3


def test_manifest_hash_ignores_timestamps() -> None:
    a = RoundManifest(round_id=0, n_records=1, dataset_hash="abc", provenance={"x": 1})
    b = RoundManifest(round_id=0, n_records=1, dataset_hash="abc", provenance={"x": 2})
    assert a.manifest_hash == b.manifest_hash
    c = RoundManifest(round_id=0, n_records=1, dataset_hash="abd")
    assert a.manifest_hash != c.manifest_hash


def _good_frame(n: int = 5) -> pd.DataFrame:
    return records_to_frame([_rec(experiment_id=f"cand_{i:06d}") for i in range(n)])


def test_validate_frame_ok_and_errors() -> None:
    assert validate_frame(_good_frame()).ok

    df = _good_frame()
    df.loc[0, "f03"] = np.nan
    df.loc[1, "f04"] = 10.0
    rep = validate_frame(df)
    assert not rep.ok
    assert any("non-finite" in e for e in rep.errors)
    assert any("outside" in e for e in rep.errors)
    with pytest.raises(DataValidationError):
        rep.raise_if_failed()

    dupes = pd.concat([_good_frame(2), _good_frame(2)])
    assert any("duplicate" in e for e in validate_frame(dupes).errors)

    bad_status = _good_frame().assign(status="bogus")
    assert any("status" in e for e in validate_frame(bad_status).errors)

    no_resp = _good_frame()
    no_resp.loc[0, "response"] = np.nan
    assert any("finite response" in e for e in validate_frame(no_resp).errors)

    missing = _good_frame().drop(columns=["response"])
    assert any("missing" in e for e in validate_frame(missing).errors)


def test_validate_pool(pool: pd.DataFrame) -> None:
    assert validate_frame(pool, n_features=32).kind == "pool"
    assert validate_frame(pool, n_features=32).ok
    bad = pool.head(10).copy()
    bad.loc[0, "cost"] = -1.0
    assert not validate_frame(bad).ok
