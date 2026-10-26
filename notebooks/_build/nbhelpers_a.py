"""Shared helpers for the notebook generator scripts build_01..build_06 (notebooks-a).

Each ``build_XX.py`` describes a notebook as a list of markdown/code cells and
writes ``notebooks/XX_*.ipynb`` (unexecuted). Execute + embed outputs with::

    uv run python notebooks/_build/build_01.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_*.ipynb
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path(__file__).resolve().parents[1]


def md(src: str) -> nbformat.NotebookNode:
    return new_markdown_cell(textwrap.dedent(src).strip("\n"))


def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(textwrap.dedent(src).strip("\n"))


def write(name: str, cells: list[nbformat.NotebookNode]) -> Path:
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    path = NB_DIR / f"{name}.ipynb"
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print(f"wrote {path}")
    return path


SETUP = '''
import os, sys, time, json, shutil, tempfile, warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("ML_LOG_LEVEL", "WARNING")   # keep structlog output out of the notebook
warnings.filterwarnings("ignore", category=UserWarning)

WORK = Path(tempfile.mkdtemp(prefix="{prefix}_"))       # every artifact of this notebook lives here
print("scratch dir:", WORK)
'''


def setup_cell(prefix: str) -> nbformat.NotebookNode:
    return code(SETUP.replace("{prefix}", prefix))
