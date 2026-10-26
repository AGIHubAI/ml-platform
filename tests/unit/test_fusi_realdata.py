"""Tests for the Caltech fUS loader (synthetic .mat) and the opt-in downloader."""

from __future__ import annotations

import hashlib
import importlib.util
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import savemat

from ml_platform.workloads.fusi_realtime import realdata as rd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_fusi_caltech.py"
spec = importlib.util.spec_from_file_location("download_fusi_caltech", SCRIPT)
assert spec and spec.loader
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def _write_session(path: Path, n_t: int = 6, with_dir: bool = True) -> None:
    dtype = [("cue", "O"), ("memory", "O"), ("target_acquire", "O")]
    if with_dir:
        dtype.append(("direction", "O"))
    beh = np.empty((2,), dtype=dtype)
    for i in range(2):
        s = 1 + 2 * i
        row = [f"10:00:0{s},000", f"10:00:0{s + 1},000", f"10:00:0{s + 1},500"]
        if with_dir:
            row.append(i + 1)
        beh[i] = tuple(row)
    secs = 36000.0 + 0.5 * np.arange(n_t)
    savemat(
        path,
        {
            "dop": np.random.default_rng(0).random((5, 4, n_t)),  # (depth, lateral, t)
            "timestamps": np.stack([secs, np.ones(n_t), np.ones(n_t)], axis=1),
            "behavior": beh,
        },
    )


def test_load_shapes_fs_events(tmp_path: Path) -> None:
    p = tmp_path / "rt_fUS_data_S3_R2.mat"
    _write_session(p)
    rec = rd.load_caltech_session(p)
    assert rec.doppler.shape == (6, 4, 1, 5)
    assert rec.doppler.dtype == np.float32
    assert rec.fs_hz == pytest.approx(2.0)
    assert list(rec.events.columns) == ["trial", "onset_s", "direction"]
    assert len(rec.events) == 2
    assert rec.meta["session"] == 3 and rec.meta["run"] == 2
    assert [q.name for q in rd.iter_caltech_sessions(tmp_path)] == [p.name]


def test_missing_direction_drops_column(tmp_path: Path) -> None:
    p = tmp_path / "a.mat"
    _write_session(p, with_dir=False)
    assert "direction" not in rd.load_caltech_session(p).events.columns


def test_unrecognised_structure(tmp_path: Path) -> None:
    p = tmp_path / "bad.mat"
    savemat(p, {"foo": np.zeros(3), "bar": np.zeros(2)})
    with pytest.raises(ValueError, match="foo"):
        rd.load_caltech_session(p)


def test_to_xarray(tmp_path: Path) -> None:
    p = tmp_path / "a.mat"
    _write_session(p)
    ds = rd.to_xarray(rd.load_caltech_session(p))
    assert ds["doppler"].dims == ("t", "x", "y", "z")
    assert ds["t"].values[1] == pytest.approx(0.5)
    assert ds.attrs["fs_hz"] == pytest.approx(2.0)
    assert "CC BY-NC" in ds.attrs["license"]


@pytest.fixture
def server(tmp_path: Path) -> Any:
    root = tmp_path / "srv"
    root.mkdir()
    payload = bytes(range(256)) * 40
    (root / "f.bin").write_bytes(payload)

    class H(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            rng = self.headers.get("Range")
            if not rng:
                return super().do_GET()
            start = int(rng.split("=")[1].split("-")[0])
            body = payload[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: Any) -> None:
            pass

    httpd = HTTPServer(("127.0.0.1", 0), partial(H, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/f.bin", payload
    httpd.shutdown()


def test_download_resume_and_md5(tmp_path: Path, server: Any) -> None:
    url, payload = server
    dest = tmp_path / "out" / "f.bin"
    dest.parent.mkdir()
    (tmp_path / "out" / "f.bin.part").write_bytes(payload[:1000])
    dl.fetch_and_verify(url, dest, hashlib.md5(payload).hexdigest())
    assert dest.read_bytes() == payload
    assert not dest.with_name("f.bin.part").exists()


def test_download_md5_mismatch(tmp_path: Path, server: Any) -> None:
    url, _ = server
    with pytest.raises(ValueError, match="md5 mismatch"):
        dl.fetch_and_verify(url, tmp_path / "f.bin", "0" * 32)
    assert not (tmp_path / "f.bin").exists()


def test_refuses_target_inside_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dl.ENV_DIR, str(dl.REPO_ROOT / "data"))
    assert dl.main(["--yes"]) == 2


def test_no_network_without_yes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(dl.ENV_DIR, str(tmp_path / "d"))

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("network call")

    monkeypatch.setattr(dl, "download", boom)
    assert dl.main([]) == 0
    assert not (tmp_path / "d").exists()
