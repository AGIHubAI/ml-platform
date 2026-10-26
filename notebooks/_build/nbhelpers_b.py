"""Shared helpers for the notebook generator scripts ``build_07.py`` … ``build_13.py``.

Each ``build_XX.py`` describes one notebook as a list of markdown/code cells so
that the notebooks stay reviewable (plain Python diff) and regenerable::

    uv run python notebooks/_build/build_07.py            # write the .ipynb (no outputs)
    uv run python notebooks/_build/build_07.py --execute  # write + execute in place

Execution uses the repository venv's ``python3`` kernel. ``RAY_TMPDIR`` is
pointed at a private temp root for the run and removed afterwards, so executing
a notebook never leaves Ray session directories behind in ``/tmp/ray``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.notebooknode import NotebookNode

BUILD_DIR = Path(__file__).resolve().parent
NB_DIR = BUILD_DIR.parent
REPO_ROOT = NB_DIR.parent


def md(text: str) -> NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> NotebookNode:
    return nbformat.v4.new_code_cell(dedent(text).strip("\n"))


def preamble(tag: str, extra: str = "") -> NotebookNode:
    """Standard first code cell: quiet logs, a private temp workspace, plotting defaults."""
    return code(
        f'''
        import os, sys, tempfile, time, warnings
        from pathlib import Path

        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        os.environ.setdefault("ML_LOG_LEVEL", "WARNING")   # keep structlog output out of the notebook
        os.environ.setdefault("RAY_DEDUP_LOGS", "1")
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        {extra}
        import numpy as np
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt

        np.set_printoptions(precision=4, suppress=True)
        pd.set_option("display.precision", 4)
        pd.set_option("display.width", 120)
        plt.rcParams.update({{"figure.dpi": 80, "figure.figsize": (7.5, 3.6), "axes.grid": True,
                             "grid.alpha": 0.3}})

        # Every artifact this notebook creates lives in a fresh temp dir -- never in the repo's
        # data/, artifacts/ or mlflow.db.
        WORK = Path(tempfile.mkdtemp(prefix="{tag}_"))
        T0 = time.perf_counter()
        print("workspace:", WORK)
        '''
    )


def write_notebook(name: str, cells: list[NotebookNode]) -> Path:
    nb = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    path = NB_DIR / f"{name}.ipynb"
    nbformat.write(nb, path)
    return path


def execute(path: Path, timeout: int = 600) -> float:
    """Execute ``path`` in place with nbconvert; returns wall seconds."""
    ray_tmp = Path(tempfile.mkdtemp(prefix="nbb_ray_", dir="/tmp"))
    env = {**os.environ, "RAY_TMPDIR": str(ray_tmp), "MLFLOW_DISABLE_AGENT_HINT": "1"}
    t0 = time.perf_counter()
    try:
        subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--inplace", f"--ExecutePreprocessor.timeout={timeout}",
                "--ExecutePreprocessor.kernel_name=python3", str(path),
            ],
            check=True, env=env, cwd=str(NB_DIR),
        )
    finally:
        shutil.rmtree(ray_tmp, ignore_errors=True)
    return time.perf_counter() - t0


def main(name: str, cells: list[NotebookNode]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="execute the notebook in place")
    args = ap.parse_args()
    path = write_notebook(name, cells)
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    if args.execute:
        dt = execute(path)
        size_kb = path.stat().st_size / 1024
        print(f"executed {path.name} in {dt:.1f}s ({size_kb:.0f} KB)")
