"""Frozen simulator kernels -- the declared functional forms (PHASE0_LOCK_DECISIONS.md D5).

Every function here is a *pure* function of ``(inputs, frozen coefficients)``. The coefficients
live in ``simulator.params.json`` (single source of truth, like ``config.py``) and are bound to
these kernels by ``params.py``; the kernels themselves physically cannot use an unfrozen value.

Golden-vector conformance tests (``tests/test_simulator_kernels.py``) pin each declared form
against hand-computed outputs -- this is guard (2) of the three the lock enforces (D1): it catches
a *wrong-but-consistent* formula (e.g. a logistic link silently a step function) that the controls
and leakage sentinels would miss.

Only the deterministic *links* live here (rate/prob/mean as a function of latent health ``h``). The
stochastic *draws* (Exponential/Bernoulli/Normal given a rate/prob/mean) are Phase-1 generation.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z):
    """Logistic sigmoid, elementwise. Clipped at +/-60 to saturate without overflow.

    At |z| <= 60 the result is already 0.0/1.0 to float64 precision, so the clip only tames
    the far tail (real churn/event logits never reach it) and never evaluates an overflowing exp.
    """
    z = np.clip(np.asarray(z, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


# --- Static customer-quality index q(x) (D5.4) -----------------------------------------------


def standardize(x, mean, std):
    """Standardize a raw numeric feature against frozen anchor statistics."""
    return (np.asarray(x, dtype=float) - mean) / std


def quality_index_raw(plan_score, z_age, z_spend, z_aov, weights):
    """Raw linear quality index over encoded static features (before final standardization).

    ``weights`` carries the frozen *direction* weights; ``plan`` dominates (anchor-faithful).
    Region/device carry zero weight in the anchor (they add ~nothing) and are omitted here.
    """
    return (
        weights["plan"] * np.asarray(plan_score, dtype=float)
        + weights["age"] * np.asarray(z_age, dtype=float)
        + weights["spend"] * np.asarray(z_spend, dtype=float)
        + weights["aov"] * np.asarray(z_aov, dtype=float)
    )


# --- Latent health state h (D5.1) ------------------------------------------------------------


def latent_baseline(q, s_mu):
    """Static-seeded health mean the AR(1) process reverts toward: mu(x) = s_mu * q(x)."""
    return s_mu * np.asarray(q, dtype=float)


def latent_step(h, mu, kappa, eps, sigma):
    """One weekly AR(1)/OU step: h_{w+1} = h_w + kappa*(mu - h_w) + sigma*eps_w."""
    return h + kappa * (mu - h) + sigma * eps


# --- Hazard / label rule (D5.2) --------------------------------------------------------------


def hazard_prob(h, q, alpha0, alpha_h, alpha_stat):
    """Churn probability for the window (t0, t0+90d], evaluated at h(t0).

    logit = alpha0 + alpha_h*h + alpha_stat*q, with frozen alpha_h < 0 and alpha_stat < 0
    (healthier / higher-quality customers churn less). This *is* the analytic Bayes-optimal
    oracle score (D4): P(y=1 | h(t0), q(x)) with no fitted estimator.
    """
    return sigmoid(
        alpha0 + alpha_h * np.asarray(h, dtype=float) + alpha_stat * np.asarray(q, dtype=float)
    )


# --- Event-emission links (D5.5) -------------------------------------------------------------
# Three canonical links; each event family is one of these bound to its frozen (a, b). The sign
# of b encodes direction (frozen in params): b > 0 for login/order/session-depth/sentiment
# (healthier -> more), b < 0 for payment-failure/support/downgrade (unhealthy -> more).


def exp_link(h, a, b):
    """Log-linear Poisson *rate*: exp(a + b*h). Used by login, order, and support arrivals."""
    return np.exp(a + b * np.asarray(h, dtype=float))


def logistic_link(h, a, b):
    """Logit Bernoulli *probability*: sigmoid(a + b*h). Used by payment-failure and downgrade."""
    return sigmoid(a + b * np.asarray(h, dtype=float))


def linear_link(h, a, b):
    """Identity-link Gaussian *mean*: a + b*h. Used by session depth (pages) and sentiment."""
    return a + b * np.asarray(h, dtype=float)
