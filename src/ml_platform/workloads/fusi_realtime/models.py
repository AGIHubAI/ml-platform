"""Decoders: a streaming spatiotemporal transformer and a PCA + LDA baseline.

**Transformer** (`FusiDecoder`)::

    volume (x,y,z) -> cubic patches -> linear patch embedding + learned patch position
      -> 1 bidirectional attention block over the patches of that volume -> flatten + linear
      -> one token per volume
    tokens over time -> L causal blocks (RoPE, sliding window of ``attn_window`` frames)
      -> LayerNorm -> linear head (n_directions logits per frame)

Why these choices:

* RoPE makes attention scores depend on *relative* frame offsets only, and the causal
  window bounds what each layer sees. Together they make streaming exact: decoding one
  volume at a time with a per-layer KV cache of the last ``attn_window - 1`` frames
  gives the same logits as a full band-masked forward pass over the whole sequence
  (``tests/unit/test_fusi_models.py``), for arbitrarily long sessions, with O(window)
  memory. Positions are absolute frame indices, so a dropped frame leaves a gap
  instead of shifting every later position.
* Attention is written out (`attention`) rather than calling
  ``F.scaled_dot_product_attention``, so the masking/caching logic is explicit; the
  tests check it against SDPA.

**Baseline** (`PcaLdaDecoder`): mean of the last ``baseline_frames`` volumes ->
PCA -> shrinkage LDA, the classical fUS-BMI decoder family (Norman 2021, Griggs 2024
used class-wise PCA + LDA). Pure numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ml_platform.workloads.fusi_realtime.config import DecoderConfig


# --------------------------------------------------------------------------- attention
def band_causal_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, window: int) -> torch.Tensor:
    """Boolean ``(Tq, Tk)`` mask: key visible iff ``0 <= q - k < window``."""
    diff = q_pos[:, None] - k_pos[None, :]
    return (diff >= 0) & (diff < window)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Scaled dot-product attention ``softmax(q k^T / sqrt(d) + mask) v`` from scratch.

    ``q`` (..., Tq, d), ``k``/``v`` (..., Tk, d); ``mask`` broadcastable bool, True = keep.
    """
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    """Rotary position embedding on the last dim of ``x`` (..., T, d) at positions ``pos`` (T,)."""
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.to(torch.float32)[:, None] * inv[None, :]  # (T, d/2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


@dataclass
class LayerCache:
    """Per-layer KV cache: keys/values ``(B, H, T, dh)`` and their absolute positions ``(T,)``."""

    k: torch.Tensor
    v: torch.Tensor
    pos: torch.Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh, self.window = n_heads, d_model // n_heads, window
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return q, k, v

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Full sequence: ``x`` (B, T, d), ``pos`` (T,) absolute frame indices."""
        b, t, d = x.shape
        q, k, v = self._split(x)
        q, k = rope(q, pos), rope(k, pos)
        y = attention(q, k, v, band_causal_mask(pos, pos, self.window))
        return self.drop(self.out(y.transpose(1, 2).reshape(b, t, d)))

    def step(
        self, x: torch.Tensor, pos: torch.Tensor, cache: LayerCache | None
    ) -> tuple[torch.Tensor, LayerCache]:
        """One new token ``x`` (B, 1, d) at ``pos`` (1,); returns output and updated cache."""
        b, _, d = x.shape
        q, k, v = self._split(x)
        q, k = rope(q, pos), rope(k, pos)
        if cache is not None:
            keep = band_causal_mask(pos, cache.pos, self.window)[0]  # drop keys out of window
            k = torch.cat([cache.k[:, :, keep], k], dim=2)
            v = torch.cat([cache.v[:, :, keep], v], dim=2)
            kpos = torch.cat([cache.pos[keep], pos])
        else:
            kpos = pos
        y = attention(q, k, v)  # every cached key is inside the window by construction
        new_cache = LayerCache(k=k, v=v, pos=kpos)
        return self.drop(self.out(y.transpose(1, 2).reshape(b, 1, d))), new_cache


class TemporalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, window, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pos)
        return x + self.mlp(self.ln2(x))

    def step(
        self, x: torch.Tensor, pos: torch.Tensor, cache: LayerCache | None
    ) -> tuple[torch.Tensor, LayerCache]:
        a, cache = self.attn.step(self.ln1(x), pos, cache)
        x = x + a
        return x + self.mlp(self.ln2(x)), cache


def patch_shape(patch: int | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if isinstance(patch, int):
        return (patch, patch, patch)
    px, py, pz = (int(p) for p in patch)
    return (px, py, pz)


class SpatialEncoder(nn.Module):
    """Volume ``(N, x, y, z)`` -> one token ``(N, d)`` via patch tokens + one attention block."""

    def __init__(
        self,
        grid: tuple[int, int, int],
        patch: int | tuple[int, int, int],
        d_model: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        p3 = patch_shape(patch)
        if any(g % p for g, p in zip(grid, p3, strict=True)):
            raise ValueError(f"grid {grid} must be divisible by patch {p3}")
        self.grid, self.patch = grid, p3
        self.n_patches = math.prod(g // p for g, p in zip(grid, p3, strict=True))
        self.embed = nn.Linear(math.prod(p3), d_model)
        self.pos = nn.Parameter(torch.zeros(self.n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.ln = nn.LayerNorm(d_model)
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        # Flatten (not mean-pool) the patch tokens: direction lives in the spatial
        # pattern across patches, which a mean would average away.
        self.pool = nn.Linear(self.n_patches * d_model, d_model)

    def patches(self, vol: torch.Tensor) -> torch.Tensor:
        n = vol.shape[0]
        px, py, pz = self.patch
        gx, gy, gz = (g // p for g, p in zip(self.grid, self.patch, strict=True))
        x = vol.reshape(n, gx, px, gy, py, gz, pz).permute(0, 1, 3, 5, 2, 4, 6)
        return x.reshape(n, gx * gy * gz, px * py * pz)

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        tok = self.embed(self.patches(vol)) + self.pos  # (N, P, d)
        n, p, d = tok.shape
        dh = d // self.h
        q, k, v = self.qkv(self.ln(tok)).view(n, p, 3, self.h, dh).permute(2, 0, 3, 1, 4)
        y = attention(q, k, v).transpose(1, 2).reshape(n, p, d)
        tok = tok + self.drop(self.out(y))
        out: torch.Tensor = self.pool(tok.reshape(n, p * d))
        return out


class FusiDecoder(nn.Module):
    """Spatiotemporal transformer; ``forward`` for training, ``step`` for streaming."""

    def __init__(
        self,
        grid: tuple[int, int, int],
        n_classes: int,
        *,
        patch: int | tuple[int, int, int],
        d_model: int,
        n_heads: int,
        n_layers: int,
        attn_window: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._spec: dict[str, Any] = {
            "kind": "transformer",
            "grid": list(grid),
            "n_classes": n_classes,
            "patch": list(patch_shape(patch)),
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "attn_window": attn_window,
            "dropout": dropout,
        }
        self.spatial = SpatialEncoder(grid, patch, d_model, n_heads, dropout)
        self.blocks = nn.ModuleList(
            TemporalBlock(d_model, n_heads, attn_window, dropout) for _ in range(n_layers)
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def spec(self) -> dict[str, Any]:
        return dict(self._spec)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` (B, T, x, y, z) -> logits (B, T, C). ``pos`` (T,) defaults to ``0..T-1``."""
        b, t = x.shape[:2]
        if pos is None:
            pos = torch.arange(t, device=x.device)
        h = self.spatial(x.reshape(b * t, *x.shape[2:])).view(b, t, -1)
        for blk in self.blocks:
            h = blk(h, pos)
        out: torch.Tensor = self.head(self.ln(h))
        return out

    def step(
        self, vol: torch.Tensor, pos: int, cache: list[LayerCache] | None = None
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """One new volume ``vol`` (B, x, y, z) at absolute frame ``pos`` -> logits (B, C)."""
        p = torch.tensor([pos], device=vol.device)
        h = self.spatial(vol).unsqueeze(1)
        new: list[LayerCache] = []
        for i, blk in enumerate(self.blocks):
            assert isinstance(blk, TemporalBlock)
            h, c = blk.step(h, p, cache[i] if cache is not None else None)
            new.append(c)
        return self.head(self.ln(h))[:, 0], new


def build_decoder(cfg: DecoderConfig, grid: tuple[int, int, int], n_classes: int) -> FusiDecoder:
    return FusiDecoder(
        grid,
        n_classes,
        patch=cfg.patch,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        attn_window=cfg.attn_window,
        dropout=cfg.dropout,
    )


def decoder_from_spec(spec: dict[str, Any]) -> FusiDecoder:
    s = dict(spec)
    if s.pop("kind", "transformer") != "transformer":
        raise ValueError(f"not a transformer spec: {spec}")
    grid = tuple(s.pop("grid"))
    n_classes = int(s.pop("n_classes"))
    return FusiDecoder(grid, n_classes, **s)  # type: ignore[arg-type]


def save_decoder(model: FusiDecoder, path: str, **extra: Any) -> None:
    torch.save({"model_spec": model.spec(), "model_state": model.state_dict(), **extra}, path)


def load_decoder(path: str, device: str = "cpu") -> tuple[FusiDecoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = decoder_from_spec(payload["model_spec"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


class StreamingDecoder:
    """Hot-path wrapper: one volume in, class probabilities out, KV cache kept inside."""

    def __init__(self, model: FusiDecoder) -> None:
        self.model = model.eval()
        self.cache: list[LayerCache] | None = None

    def reset(self) -> None:
        self.cache = None

    @torch.inference_mode()
    def step(self, volume: np.ndarray, pos: int) -> np.ndarray:
        x = torch.from_numpy(np.array(volume, dtype=np.float32, copy=True))[
            None
        ]  # Ray arrays are read-only
        logits, self.cache = self.model.step(x, pos, self.cache)
        out: np.ndarray = torch.softmax(logits, dim=-1)[0].numpy()
        return out


# --------------------------------------------------------------------------- baseline
@dataclass
class PcaLdaDecoder:
    """PCA + shrinkage LDA on the mean of the last ``n_frames`` volumes (pure numpy)."""

    n_components: int
    shrinkage: float
    n_frames: int
    n_classes: int
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    intercept_: np.ndarray | None = None

    @staticmethod
    def frame_features(volumes: np.ndarray, n_frames: int) -> np.ndarray:
        """Causal running mean over ``n_frames``: ``(T, *space)`` -> ``(T, V)``."""
        x = volumes.reshape(volumes.shape[0], -1).astype(np.float64)
        c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
        idx = np.arange(1, x.shape[0] + 1)
        lo = np.maximum(idx - n_frames, 0)
        out: np.ndarray = (c[idx] - c[lo]) / (idx - lo)[:, None]
        return out

    def fit(self, feats: np.ndarray, labels: np.ndarray) -> PcaLdaDecoder:
        keep = labels >= 0
        x, y = feats[keep], labels[keep]
        self.mean_ = x.mean(0)
        xc = x - self.mean_
        k = min(self.n_components, xc.shape[0] - 1, xc.shape[1])
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        self.components_ = vt[:k]
        z = xc @ self.components_.T
        means = np.stack(
            [
                z[y == c].mean(0) if np.any(y == c) else np.zeros(z.shape[1])
                for c in range(self.n_classes)
            ]
        )
        resid = z - means[y]
        cov = np.cov(resid, rowvar=False).reshape(k, k)
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.trace(cov) / k * np.eye(k)
        prec = np.linalg.inv(cov)
        priors = np.array([max(np.mean(y == c), 1e-6) for c in range(self.n_classes)])
        self.coef_ = means @ prec
        self.intercept_ = -0.5 * np.einsum("ck,ck->c", self.coef_, means) + np.log(priors)
        return self

    def decision(self, feats: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        assert self.coef_ is not None and self.intercept_ is not None
        z = (feats - self.mean_) @ self.components_.T
        out: np.ndarray = z @ self.coef_.T + self.intercept_
        return out

    def predict(self, feats: np.ndarray) -> np.ndarray:
        out: np.ndarray = self.decision(feats).argmax(1)
        return out

    def save(self, path: str) -> None:
        assert self.mean_ is not None and self.components_ is not None
        assert self.coef_ is not None and self.intercept_ is not None
        np.savez(
            path,
            kind=np.array("pca_lda"),
            hyper=np.array([self.n_components, self.shrinkage, self.n_frames, self.n_classes]),
            mean=self.mean_,
            components=self.components_,
            coef=self.coef_,
            intercept=self.intercept_,
        )

    @classmethod
    def load(cls, path: str) -> PcaLdaDecoder:
        z = np.load(path)
        k, shrink, n_frames, n_classes = z["hyper"]
        return cls(
            n_components=int(k),
            shrinkage=float(shrink),
            n_frames=int(n_frames),
            n_classes=int(n_classes),
            mean_=z["mean"],
            components_=z["components"],
            coef_=z["coef"],
            intercept_=z["intercept"],
        )


class StreamingPcaLda:
    """Hot-path wrapper for the baseline: causal running mean of the last ``n_frames``."""

    def __init__(self, model: PcaLdaDecoder) -> None:
        self.model = model
        self.buf: list[np.ndarray] = []

    def reset(self) -> None:
        self.buf = []

    def step(self, volume: np.ndarray, pos: int) -> np.ndarray:
        self.buf.append(volume.reshape(-1).astype(np.float64))
        self.buf = self.buf[-self.model.n_frames :]
        feats = np.mean(self.buf, axis=0)[None]
        out: np.ndarray = softmax_np(self.model.decision(feats))[0]
        return out


# Every decoder checkpoint dir has ``model.pt`` (what the registry resolves): the torch
# checkpoint for the transformer, or a small marker pointing at ``baseline.npz``.
TRANSFORMER_FILE = "model.pt"
BASELINE_FILE = "baseline.npz"


def write_baseline_checkpoint(model: PcaLdaDecoder, checkpoint_dir: str) -> None:
    from pathlib import Path

    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    model.save(str(d / BASELINE_FILE))
    torch.save({"kind": "pca_lda", "file": BASELINE_FILE}, str(d / TRANSFORMER_FILE))


def load_any(checkpoint_dir: str) -> tuple[str, FusiDecoder | PcaLdaDecoder]:
    """``("transformer", FusiDecoder)`` or ``("pca_lda", PcaLdaDecoder)`` from a checkpoint dir."""
    from pathlib import Path

    d = Path(checkpoint_dir)
    payload = torch.load(str(d / TRANSFORMER_FILE), map_location="cpu", weights_only=False)
    if payload.get("kind") == "pca_lda":
        return "pca_lda", PcaLdaDecoder.load(str(d / payload["file"]))
    return "transformer", load_decoder(str(d / TRANSFORMER_FILE))[0]


def load_streaming(checkpoint_dir: str) -> StreamingDecoder | StreamingPcaLda:
    """Streaming wrapper for whatever decoder ``checkpoint_dir`` holds."""
    _kind, model = load_any(checkpoint_dir)
    if isinstance(model, FusiDecoder):
        return StreamingDecoder(model)
    return StreamingPcaLda(model)


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    out: np.ndarray = e / e.sum(-1, keepdims=True)
    return out


__all__ = [
    "FusiDecoder",
    "LayerCache",
    "PcaLdaDecoder",
    "StreamingDecoder",
    "attention",
    "band_causal_mask",
    "build_decoder",
    "decoder_from_spec",
    "load_decoder",
    "rope",
    "save_decoder",
]
