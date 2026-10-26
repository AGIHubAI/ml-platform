"""Hidden synthetic "experimental system".

The oracle plays the role of nature / the wet lab: the model never sees
`mean` directly, only noisy `measure` outcomes.

Structure of the hidden response f: R^d -> R (d = 32 by default):

* the input is projected onto a k = 6 dimensional *active subspace*
  ``z = W x`` (W has orthonormal rows, random but fixed by the seed). Real
  scientific responses are often driven by a few latent factors;
* in that subspace f is a sum of
  - a linear trend along latent z0 (the dominant, easy-to-learn signal),
  - three Gaussian bumps of different heights/widths (the tallest one, on the
    rising side of the trend, is the "optimum region" active learning should
    find; it sits off-centre so it is under-represented in a random sample),
  - sinusoidal terms and a multiplicative interaction (genuine
    non-linearity/non-additivity),
  - a weak linear term over *all* d raw features (so every feature matters a
    little);

  Signal/noise is tuned so the default ResidualMLP reaches a standardized
  validation RMSE of roughly 0.40-0.45 on 500 random observations and
  ~0.30-0.35 on 800-1000, i.e. the evaluation gate (max_rmse 0.35) tends to
  fail on the initial round and pass after a few rounds of new data;
* observation noise is heteroscedastic: its std grows smoothly along a
  different latent direction (some regions of design space are noisier to
  measure);
* ``cost(x)`` is positive and larger for extreme designs.

Everything is a deterministic function of ``seed``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

N_LATENT = 6


class SyntheticOracle:
    def __init__(self, seed: int = 0, n_features: int = 32) -> None:
        if n_features < N_LATENT:
            raise ValueError(f"n_features must be >= {N_LATENT}")
        self.seed = int(seed)
        self.n_features = int(n_features)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0x0AC1E]))

        # orthonormal projection onto the latent active subspace (k x d)
        q, _ = np.linalg.qr(rng.standard_normal((n_features, N_LATENT)))
        self._W: FloatArray = q.T.copy()

        # bumps in latent dims 0..3: (centre, width, height). The tallest bump
        # sits at positive z0 (the direction of the linear trend) in the tail of
        # the design distribution, so it is rare in a random initial design.
        signs = rng.choice([-1.0, 1.0], size=(3, 4))
        signs[0, 0] = 1.0
        self._bump_centres: FloatArray = signs * np.array(
            [[1.6, 1.2, 0.9, 0.6], [0.8, 0.9, 1.1, 0.5], [0.3, 0.4, 0.2, 1.0]]
        )
        self._bump_widths: FloatArray = np.array([1.4, 1.6, 1.8])
        self._bump_heights: FloatArray = np.array([3.0, 1.2, 0.8])

        self._freq: FloatArray = rng.uniform(1.2, 1.8, size=2)
        self._phase: FloatArray = rng.uniform(0, 2 * np.pi, size=2)
        self._linear: FloatArray = rng.normal(0.0, 0.03, size=n_features)
        self._cost_dir: FloatArray = rng.standard_normal(n_features) / np.sqrt(n_features)

    # ------------------------------------------------------------------ helpers
    def _as_2d(self, X: ArrayLike) -> FloatArray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {arr.shape[1]}")
        return arr

    def latent(self, X: ArrayLike) -> FloatArray:
        """Latent coordinates z = W x, shape (n, 6)."""
        return self._as_2d(X) @ self._W.T

    # ------------------------------------------------------------------ public API
    def mean(self, X: ArrayLike) -> FloatArray:
        """Noise-free response f(x), shape (n,)."""
        X2 = self._as_2d(X)
        z = X2 @ self._W.T
        zb = z[:, None, :4] - self._bump_centres[None, :, :]  # (n, 3, 4)
        bumps = self._bump_heights * np.exp(
            -0.5 * np.sum(zb**2, axis=-1) / self._bump_widths**2
        )  # (n, 3)
        f = bumps.sum(axis=1)
        f += 1.4 * z[:, 0]
        f += 0.4 * np.sin(self._freq[0] * z[:, 4] + self._phase[0])
        f += 0.25 * np.cos(self._freq[1] * z[:, 5] + self._phase[1]) * np.tanh(z[:, 0])
        f += 0.25 * z[:, 1] * z[:, 2]
        f += X2 @ self._linear
        return f

    def noise_std(self, X: ArrayLike) -> FloatArray:
        """Heteroscedastic measurement noise std, in (0.1, 0.7)."""
        z = self.latent(X)
        return 0.1 + 0.6 / (1.0 + np.exp(-1.5 * (z[:, 3] - 0.5)))

    def measure(
        self, X: ArrayLike, rng: np.random.Generator | int | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Run the (simulated) experiment: returns ``(y, std)`` with y = f(x) + eps."""
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        mu = self.mean(X)
        std = self.noise_std(X)
        return mu + std * gen.standard_normal(mu.shape[0]), std

    def cost(self, X: ArrayLike) -> FloatArray:
        """Positive experimental cost (arbitrary units, ~1 at the origin)."""
        X2 = self._as_2d(X)
        radial = np.mean(X2**2, axis=1)
        return np.exp(0.3 * (X2 @ self._cost_dir)) * (0.5 + 0.5 * radial)

    def optimum_hint(self) -> FloatArray:
        """Raw-space point mapping onto the tallest bump centre (for diagnostics only)."""
        z = np.zeros(N_LATENT)
        z[:4] = self._bump_centres[0]
        return z @ self._W
