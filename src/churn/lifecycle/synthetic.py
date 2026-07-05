"""Synthetic gaussian features + a linear-boundary label -- shared retrain/drift demo data."""

from __future__ import annotations

import numpy as np


def linear_boundary_xy(
    rng: np.random.Generator, n: int, dims: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` gaussian rows in ``dims`` dims; binary label from a fixed boundary ``x0 + 0.5*x1``.

    Only the first two columns carry signal (extra columns are noise features). Draws exactly two
    numpy normals -- the ``(n, dims)`` design then the ``n`` label noise -- so callers that keep
    using the same ``rng`` afterwards see the identical stream. The orchestration retrain branch's
    reference/current windows and the retrain test's holdout share this one generator.
    """
    x = rng.normal(0.0, 1.0, (n, dims))
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0.0, 1.0, n) > 0).astype(int)
    return x, y
