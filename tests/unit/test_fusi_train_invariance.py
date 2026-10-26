"""Pure helpers behind worker-count invariance: global batches, sampler, per-sample noise."""

from __future__ import annotations

from itertools import chain

import numpy as np
import pandas as pd
import pytest

from ml_platform.workloads.fusi_realtime.dataset import SessionArrays, WindowDataset
from ml_platform.workloads.fusi_realtime.train import GlobalBatchSampler, global_batches


def _step_batches(n: int, gb: int, world: int, epoch: int) -> list[list[int]]:
    samplers = [GlobalBatchSampler(n, gb, world=world, rank=r, seed=3) for r in range(world)]
    for s in samplers:
        s.set_epoch(epoch)
    per_rank = [list(s) for s in samplers]
    return [
        list(chain.from_iterable(per_rank[r][k] for r in range(world)))
        for k in range(len(samplers[0]))
    ]


@pytest.mark.parametrize("epoch", [0, 1])
def test_global_batch_identical_for_any_world(epoch: int) -> None:
    ref = _step_batches(50, 8, 1, epoch)
    assert ref == global_batches(50, 8, 3, epoch)
    assert len(ref) == 6  # drop_last
    for world in (2, 4):
        assert _step_batches(50, 8, world, epoch) == ref


def test_epochs_differ_and_cover_without_repeats() -> None:
    a, b = global_batches(40, 8, 0, 0), global_batches(40, 8, 0, 1)
    assert a != b
    assert sorted(i for batch in a for i in batch) == list(range(40))


def test_indivisible_or_too_small_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GlobalBatchSampler(50, 6, world=4, rank=0, seed=0)
    with pytest.raises(ValueError, match="fewer"):
        GlobalBatchSampler(4, 8, world=2, rank=0, seed=0)


def test_noise_is_a_function_of_seed_epoch_index() -> None:
    rng = np.random.default_rng(0)
    arrays = {
        "s": SessionArrays(
            rng.standard_normal((20, 2, 2, 2)).astype(np.float32), np.zeros(20, dtype=np.int64)
        )
    }
    win = pd.DataFrame({"session": "s", "start": [0, 5], "end": [5, 10]})
    ds = WindowDataset(arrays, win, noise=1.0, seed=7)
    x0, _ = ds[1]
    x0b, _ = ds[1]
    assert (x0 == x0b).all()
    ds.epoch = 1
    assert not (ds[1][0] == x0).all()
    assert not (ds[0][0] == ds[1][0]).all()
    clean = WindowDataset(arrays, win)
    assert (clean[1][0] == arrays["s"].features[5:10]).all()
