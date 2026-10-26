"""Opt-in downloader for the Caltech fUS-BMI dataset (34.3 GB).

Dataset: Griggs et al., "Decoding Motor Plans Using a Closed-Loop Ultrasonic
Brain-Machine Interface", https://data.caltech.edu/records/pa710-cdn95 .
License CC BY-NC 4.0 (non-commercial). Nothing is downloaded unless ``--yes``.

Usage::

    uv run python scripts/download_fusi_caltech.py            # info only
    uv run python scripts/download_fusi_caltech.py --yes [--extract]
    uv run python scripts/download_fusi_caltech.py --verify-only

Target dir: ``$ML_FUSI_DATA_DIR`` or ``~/data/ml-fusi/caltech-pa710-cdn95``. It must
be outside the repository so the data can never be committed by accident. Downloads
resume from ``<file>.part`` via HTTP Range; the md5 is re-computed over the whole file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

import httpx

RECORD_ID = "pa710-cdn95"
FILENAME = "fUS_BMI_Data.zip"
URL = f"https://data.caltech.edu/records/{RECORD_ID}/files/{FILENAME}?download=1"
SIZE_BYTES = 34_253_258_547
MD5 = "6d07baffa0158c9fa9bd57c9c3d5aded"
LICENSE = "CC BY-NC 4.0 (non-commercial use only)"
DEFAULT_DIR = "~/data/ml-fusi/caltech-pa710-cdn95"
ENV_DIR = "ML_FUSI_DATA_DIR"
REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNK = 1 << 20


def target_dir() -> Path:
    """Resolve the data directory from the environment (default outside the repo)."""
    return Path(os.environ.get(ENV_DIR, DEFAULT_DIR)).expanduser().resolve()


def is_inside_repo(path: Path, repo: Path = REPO_ROOT) -> bool:
    """True if ``path`` is the repository or lies below it."""
    p = path.resolve()
    return p == repo or repo in p.parents


def md5_file(path: Path) -> str:
    """Streaming md5 so a 34 GB file never has to fit in memory."""
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check_disk_space(directory: Path, needed: int) -> None:
    """Raise ``OSError`` if ``directory`` (or its nearest parent) lacks ``needed`` bytes."""
    probe = directory
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed:
        raise OSError(f"need {needed / 1e9:.1f} GB free, only {free / 1e9:.1f} GB at {probe}")


def download(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` resuming from ``dest + '.part'`` with a Range request.

    Returns the ``.part`` path; the caller verifies the checksum and renames it.
    """
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with httpx.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as resp:
        if resp.status_code == 416:  # already complete
            return part
        resp.raise_for_status()
        if offset and resp.status_code == 200:  # server ignored Range: restart
            offset = 0
        mode = "ab" if offset else "wb"
        with part.open(mode) as f:
            for chunk in resp.iter_bytes(CHUNK):
                f.write(chunk)
    return part


def fetch_and_verify(url: str, dest: Path, md5: str) -> Path:
    """Download (resumable), verify md5 over the full file, then move into place."""
    part = download(url, dest)
    got = md5_file(part)
    if got != md5:
        raise ValueError(f"md5 mismatch for {part}: expected {md5}, got {got}")
    part.replace(dest)
    return dest


def extract(zip_path: Path, out_dir: Path) -> None:
    """Unzip next to the archive; refuses members escaping ``out_dir``."""
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if root not in (root / name).resolve().parents:
                raise ValueError(f"unsafe path in archive: {name}")
        zf.extractall(root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--yes", action="store_true", help="actually download")
    ap.add_argument("--verify-only", action="store_true", help="only re-hash existing file")
    ap.add_argument("--extract", action="store_true", help="unzip after verification")
    ap.add_argument("--url", default=URL, help="override URL (testing)")
    a = ap.parse_args(argv)

    d = target_dir()
    dest = d / FILENAME
    print(f"file:    {FILENAME}\nsize:    {SIZE_BYTES / 1e9:.1f} GB\nmd5:     {MD5}")
    print(f"license: {LICENSE}\nsource:  https://data.caltech.edu/records/{RECORD_ID}")
    print(f"target:  {dest}")
    if is_inside_repo(d):
        print(f"error: target {d} is inside the repository; set {ENV_DIR}", file=sys.stderr)
        return 2
    if a.verify_only:
        if not dest.exists():
            print(f"error: {dest} not found", file=sys.stderr)
            return 1
        ok = md5_file(dest) == MD5
        print("md5 ok" if ok else "md5 MISMATCH")
        return 0 if ok else 1
    if not a.yes:
        print("dry run: pass --yes to download (non-commercial license applies)")
        return 0
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    try:
        check_disk_space(d, SIZE_BYTES - have)
        fetch_and_verify(a.url, dest, MD5)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"verified {dest}")
    if a.extract:
        extract(dest, d)
        print(f"extracted to {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
