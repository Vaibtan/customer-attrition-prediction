"""Latent-health-path generator tests (D5.1) -- built on the frozen kernels.

The generator is Phase-1 code (spec section 5 boundary); it must implement the *declared* AR(1)/OU
recursion exactly (pinned against ``kernels.latent_step`` here), stay a pure function of
``(frozen params, seed)`` (determinism), take **no label input** (invariant i), and reproduce the
frozen stationary regime (h(t0) ~ N(mu(x), sigma0^2)).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from churn.simulator import kernels as K
from churn.simulator import latent as L
from churn.simulator import params as P
from churn.simulator import rng as R


@pytest.fixture(scope="module")
def params() -> dict:
    return P.load_params()


def test_stream_discipline_is_deterministic_and_independent():
    a = R.make_streams(123)
    b = R.make_streams(123)
    c = R.make_streams(124)
    # Same seed -> identical draws on the same named stream.
    assert a["latent"].standard_normal(5).tolist() == b["latent"].standard_normal(5).tolist()
    # Different seed -> different draws.
    assert a["latent"].standard_normal(5).tolist() != c["latent"].standard_normal(5).tolist()
    # Distinct named streams are independent (not the same sub-stream).
    s = R.make_streams(7)
    assert s["latent"].standard_normal(5).tolist() != s["login"].standard_normal(5).tolist()


def test_make_streams_exposes_every_canonical_stream():
    s = R.make_streams(0)
    assert set(s) == set(R.STREAM_NAMES)


def test_health_path_shape_and_endpoint(params):
    q = np.array([-1.0, 0.0, 1.0])
    streams = R.make_streams(42)
    h = L.simulate_health_paths(q, params, streams["latent"])
    weeks = params["latent"]["weeks"]
    assert h.shape == (3, weeks + 1)  # h_0 .. h_W inclusive
    # h(t0) is the final column.
    assert np.allclose(L.health_at_t0(h), h[:, -1])


def test_health_recursion_matches_the_frozen_kernel_exactly(params):
    # Re-run the declared recursion by hand off the same seed and assert bit-equality.
    q = np.array([-0.5, 0.3, 1.2, -2.0])
    streams = R.make_streams(2024)
    h = L.simulate_health_paths(q, params, streams["latent"])

    lat = params["latent"]
    s_mu = params["quality_index"]["s_mu"]
    mu = K.latent_baseline(q, s_mu)
    rng = R.make_streams(2024)["latent"]  # identical stream
    z = rng.standard_normal(len(q))
    expected = np.empty_like(h)
    expected[:, 0] = mu + lat["sigma0"] * z
    for w in range(lat["weeks"]):
        eps = rng.standard_normal(len(q))
        expected[:, w + 1] = K.latent_step(expected[:, w], mu, lat["kappa"], eps, lat["sigma"])
    assert np.allclose(h, expected)


def test_health_generation_takes_no_label_input():
    # Structural guard for invariant (i): the health generator cannot be a function of y.
    sig = set(inspect.signature(L.simulate_health_paths).parameters)
    assert "y" not in sig and "label" not in sig and "churn" not in sig


def test_health_is_deterministic_given_seed(params):
    q = np.linspace(-2, 2, 50)
    h1 = L.simulate_health_paths(q, params, R.make_streams(99)["latent"])
    h2 = L.simulate_health_paths(q, params, R.make_streams(99)["latent"])
    assert np.array_equal(h1, h2)


def test_stationary_regime_of_h_at_t0(params):
    # h(t0) ~ N(mu(x), sigma0^2): the mean tracks mu(x) and the marginal std ~ sigma0.
    rng = R.make_streams(5)
    n = 40_000
    q = rng["population"].standard_normal(n)  # unit-variance q, like the frozen anchor index
    h = L.simulate_health_paths(q, params, rng["latent"])
    ht0 = L.health_at_t0(h)
    s_mu = params["quality_index"]["s_mu"]
    # Regress-free checks: conditional mean ~ mu(x); residual std ~ sigma0.
    resid = ht0 - s_mu * q
    assert abs(float(resid.mean())) < 0.05
    assert float(resid.std()) == pytest.approx(params["latent"]["sigma0"], abs=0.05)


def test_health_plus_hazard_reproduces_frozen_regime_on_anchor(params):
    # End-to-end: real health paths (52 AR(1) steps) + frozen hazard on the anchor reproduce the
    # locked regime bands -- the D5.9 recovery ceiling that the simulated h(t0) must realise.
    df = pd.read_csv(P.config.DATA_PATH)
    static = df[["region", "device_type", "subscription_plan", *P.STATIC_NUMERICS]].copy()
    q_anchor = P.quality_index(static, params)
    rng = R.make_streams(11)
    idx = rng["population"].integers(0, len(q_anchor), size=40_000)
    q = q_anchor[idx]
    h = L.simulate_health_paths(q, params, rng["latent"])
    ht0 = L.health_at_t0(h)
    haz = params["hazard"]
    p = K.hazard_prob(ht0, q, haz["alpha0"], haz["alpha_h"], haz["alpha_stat"])
    y = (rng["label"].random(len(q)) < p).astype(int)

    base = float(y.mean())
    static_auc = roc_auc_score(y, -q)
    oracle_auc = roc_auc_score(y, p)
    assert 0.45 <= base <= 0.51, base
    assert 0.61 <= static_auc <= 0.67, static_auc
    assert 0.77 <= oracle_auc <= 0.83, oracle_auc
    assert oracle_auc - static_auc >= 0.12
