"""Latent-health-path generator (D5.1) -- the common cause of events and the label.

A per-customer scalar health state drifts via the frozen weekly AR(1)/OU recursion toward a
static-seeded baseline ``mu(x) = s_mu * q(x)``. This module *draws* the stochastic path; it composes
the frozen, golden-vector-pinned kernels (``latent_baseline``, ``latent_step``) and adds no new
functional form. It takes **no label input** (D6 invariant i): the label is a downstream Bernoulli
of ``h(t0)`` (``hazard_prob``) and the events are downstream draws off ``h_w`` -- health is their
common cause, never the reverse.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from churn.simulator import kernels as K


def simulate_health_paths(
    q: NDArray[np.float64], params: dict, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Weekly health paths for ``n`` customers with quality index ``q``.

    Returns an ``(n, W + 1)`` array of ``h_0 .. h_W`` (``W = latent.weeks``); column ``w`` is the
    health entering week ``w`` and the final column is ``h(t0) = h_W``. Vectorised over customers;
    ``eps`` is drawn one week at a time so the stream order is stable and hand-reproducible.
    """
    q = np.asarray(q, dtype=np.float64)
    n = q.shape[0]
    lat = params["latent"]
    weeks = int(lat["weeks"])
    kappa, sigma, sigma0 = float(lat["kappa"]), float(lat["sigma"]), float(lat["sigma0"])
    s_mu = float(params["quality_index"]["s_mu"])

    mu = K.latent_baseline(q, s_mu)
    h = np.empty((n, weeks + 1), dtype=np.float64)
    h[:, 0] = mu + sigma0 * rng.standard_normal(n)
    for w in range(weeks):
        eps = rng.standard_normal(n)
        h[:, w + 1] = K.latent_step(h[:, w], mu, kappa, eps, sigma)
    return h


def health_at_t0(paths: NDArray[np.float64]) -> NDArray[np.float64]:
    """h(t0) = h_W, the last column of a path array (the state driving the hazard/label)."""
    return np.asarray(paths, dtype=np.float64)[:, -1]
