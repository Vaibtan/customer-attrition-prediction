"""Deterministic, domain-separated RNG streams for the Phase-1 generator.

The whole generator is a pure function of ``(frozen params, seed)`` (D6 invariant iv). Each
concern -- population sampling, the latent path, and every event family -- draws from its own
independent ``numpy.random.Generator`` spawned from a single root ``SeedSequence``, so streams do
not interfere and any one can be regenerated in isolation (e.g. the null-stream control re-draws
only the event families off the *same* latent + label streams). This mirrors the beacon's
domain-separated-tag philosophy (D3/D4) for the exploratory seeds Phase 1 develops on.
"""

from __future__ import annotations

import numpy as np

# Canonical, ordered stream names. Order is load-bearing: SeedSequence.spawn() is positional, so
# appending a NEW name is safe (existing streams are unchanged) but reordering/removing is not.
STREAM_NAMES: tuple[str, ...] = (
    "population",  # static-attribute sampling (D6.4)
    "latent",  # AR(1)/OU health paths (D5.1)
    "login",  # login/order/support arrival counts (Poisson) (D6.2)
    "order",
    "support",
    "payment",  # per-cycle Bernoulli families (D6.2)
    "downgrade",
    "session_pages",  # per-login session depth (Normal) (D5.5)
    "sentiment",  # per-ticket sentiment (Normal) (D5.5)
    "label",  # the single hazard Bernoulli at t0 (D5.2)
)


def make_streams(
    seed: int, names: tuple[str, ...] = STREAM_NAMES
) -> dict[str, np.random.Generator]:
    """Spawn one independent, reproducible ``Generator`` per named stream from ``seed``."""
    root = np.random.SeedSequence(int(seed))
    children = root.spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children, strict=True)}
