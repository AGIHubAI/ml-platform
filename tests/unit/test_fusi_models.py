"""fUSI decoders: attention vs SDPA, KV-cache streaming == full recompute, baseline, checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml_platform.workloads.fusi_realtime.config import DecoderConfig
from ml_platform.workloads.fusi_realtime.models import (
    PcaLdaDecoder,
    StreamingPcaLda,
    attention,
    band_causal_mask,
    build_decoder,
    load_any,
    load_streaming,
    save_decoder,
    write_baseline_checkpoint,
)
from ml_platform.workloads.fusi_realtime.train import predict_session

GRID = (8, 8, 8)
DCFG = DecoderConfig(patch=4, d_model=32, n_heads=4, n_layers=2, attn_window=5, dropout=0.0)


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_decoder(DCFG, GRID, 8).eval()


def test_attention_matches_sdpa_prefill_and_single_token() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 9, 16) for _ in range(3))
    pos = torch.arange(9)
    mask = band_causal_mask(pos, pos, window=4)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(attention(q, k, v, mask), ref, atol=1e-6, rtol=1e-5)
    # one new query against a cache of keys (decode step), no mask
    q1 = torch.randn(2, 4, 1, 16)
    torch.testing.assert_close(
        attention(q1, k, v), F.scaled_dot_product_attention(q1, k, v), atol=1e-6, rtol=1e-5
    )
    # causal special case: a window >= T is plain causal attention
    full = band_causal_mask(pos, pos, window=100)
    torch.testing.assert_close(
        attention(q, k, v, full),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        atol=1e-6,
        rtol=1e-5,
    )


@torch.no_grad()
def test_kv_cache_streaming_equals_full_recompute() -> None:
    model = _model()
    x = torch.randn(2, 17, *GRID)
    full = model(x)  # (B, T, C), band-masked over the whole sequence
    cache = None
    steps = []
    for t in range(x.shape[1]):
        logits, cache = model.step(x[:, t], t, cache)
        steps.append(logits)
    torch.testing.assert_close(torch.stack(steps, 1), full, atol=1e-5, rtol=1e-4)
    assert cache is not None
    assert all(c.k.shape[2] == DCFG.attn_window for c in cache)  # bounded cache


@torch.no_grad()
def test_dropped_frame_keeps_absolute_positions() -> None:
    model = _model()
    x = torch.randn(1, 10, *GRID)
    keep = [t for t in range(10) if t != 4]  # frame 4 dropped on the hot path
    full = model(x[:, keep], torch.tensor(keep))
    cache, outs = None, []
    for t in keep:
        logits, cache = model.step(x[:, t], t, cache)
        outs.append(logits)
    torch.testing.assert_close(torch.stack(outs, 1), full, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_chunked_session_prediction_is_exact() -> None:
    model = _model()
    x = np.random.default_rng(0).standard_normal((40, *GRID)).astype(np.float32)
    chunked = predict_session(model, x, chunk=7)
    full = torch.softmax(model(torch.from_numpy(x)[None]), -1)[0].numpy()
    np.testing.assert_allclose(chunked, full, atol=1e-5)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _model()
    save_decoder(model, str(tmp_path / "model.pt"))
    kind, loaded = load_any(str(tmp_path))
    assert kind == "transformer"
    x = torch.randn(1, 6, *GRID)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))
    stream = load_streaming(str(tmp_path))
    p = stream.step(x[0, 0].numpy(), 0)
    assert p.shape == (8,) and abs(p.sum() - 1) < 1e-5


def test_pca_lda_baseline_learns_and_streams(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    patterns = rng.standard_normal((4, 64))
    labels = rng.integers(0, 4, 400)
    vols = (patterns[labels] + 0.5 * rng.standard_normal((400, 64))).reshape(400, 4, 4, 4)
    feats = PcaLdaDecoder.frame_features(vols, 1)
    model = PcaLdaDecoder(8, 0.2, 1, 4).fit(feats[:300], labels[:300])
    assert (model.predict(feats[300:]) == labels[300:]).mean() > 0.9
    write_baseline_checkpoint(model, str(tmp_path))
    kind, _ = load_any(str(tmp_path))
    assert kind == "pca_lda"
    stream = load_streaming(str(tmp_path))
    assert isinstance(stream, StreamingPcaLda)
    online = np.stack([stream.step(v, i) for i, v in enumerate(vols[300:])]).argmax(1)
    np.testing.assert_array_equal(online, model.predict(feats[300:]))


def test_fit_survives_batches_without_labels() -> None:
    import pandas as pd

    from ml_platform.workloads.fusi_realtime.config import FusiTrainingConfig
    from ml_platform.workloads.fusi_realtime.dataset import SessionArrays, WindowDataset
    from ml_platform.workloads.fusi_realtime.train import fit_decoder

    rng = np.random.default_rng(0)
    labels = np.full(60, -100)
    labels[40:] = rng.integers(0, 8, 20)  # first windows carry no label at all
    arrays = {"s": SessionArrays(rng.standard_normal((60, *GRID)).astype(np.float32), labels)}
    windows = pd.DataFrame({"session": "s", "start": [0, 10, 20, 40], "end": [10, 20, 30, 50]})
    model = build_decoder(DCFG, GRID, 8)
    cfg = FusiTrainingConfig(epochs=2, batch_size=1, input_noise=0.0)
    res = fit_decoder(model, WindowDataset(arrays, windows), None, cfg, seed=0)
    assert all(np.isfinite(h["train_loss"]) for h in res.history)
    assert all(torch.isfinite(p).all() for p in model.parameters())
