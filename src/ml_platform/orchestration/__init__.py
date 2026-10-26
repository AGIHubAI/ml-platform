"""Orchestration layer: thin Dagster adapters over the platform (handoff §8).

* ``pipeline``    — the closed-loop step functions (plain Python, no Dagster);
* ``resources``   — Dagster resources (config, RoundStore, MLflow, Ray compute);
* ``partitions``  — dynamic ``rounds`` partitions;
* ``assets``      — the asset graph (glue only);
* ``jobs`` / ``sensors`` — jobs, the new-round sensor, the nightly schedule;
* ``definitions`` — the code location (``defs``).

Importing this package does not import Dagster; import the submodules.
"""
