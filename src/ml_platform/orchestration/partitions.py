"""Experimental rounds as Dagster dynamic partitions (``round_000``, ``round_001``, ...).

Rounds are *dynamic*: nobody knows in advance how many experimental rounds a
campaign will run, and a new round comes into existence only when the lab
(``new_experimental_results``) or an external process writes it to the
RoundStore. Partition keys are identical to RoundStore directory names, so a
Dagster partition and ``data/rounds/<key>/`` always refer to the same
immutable dataset.
"""

from __future__ import annotations

from dagster import DagsterInstance, DynamicPartitionsDefinition

from ml_platform.data import parse_round_key, round_key

ROUNDS_PARTITION_NAME = "rounds"

rounds_partitions = DynamicPartitionsDefinition(name=ROUNDS_PARTITION_NAME)


def partition_round_id(partition_key: str) -> int:
    """``"round_003"`` -> 3."""
    return parse_round_key(partition_key)


def ensure_round_partition(instance: DagsterInstance, round_id: int) -> str:
    """Register ``round_XXX`` as a dynamic partition (idempotent); returns the key."""
    key = round_key(round_id)
    if not instance.has_dynamic_partition(ROUNDS_PARTITION_NAME, key):
        instance.add_dynamic_partitions(ROUNDS_PARTITION_NAME, [key])
    return key


__all__ = [
    "ROUNDS_PARTITION_NAME",
    "ensure_round_partition",
    "partition_round_id",
    "round_key",
    "rounds_partitions",
]
