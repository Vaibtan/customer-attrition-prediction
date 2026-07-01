"""Dataset orchestrator tests -- population + latent + events + label as one frozen Dataset.

Ties the tested components together: the simulated label is a single Bernoulli of ``h(t0)`` and
``q`` (the analytic oracle score), events never depend on the label (invariant i), the anchor keeps
its real label alongside the resimulated one, and the whole draw is seed-deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.simulator import generate as G
from churn.simulator import kernels as K
from churn.simulator import params as P


@pytest.fixture(scope="module")
def params() -> dict:
    return P.load_params()


@pytest.fixture(scope="module")
def ds(params) -> G.Dataset:
    return G.build_population_dataset(params, seed=2026, n_synthetic=4000)


def test_shapes_and_keys(ds, params):
    n = params["population"]["anchor_n"] + 4000
    assert len(ds.customers) == n
    assert len(ds.labels) == n
    assert len(ds.cohort) == n
    assert ds.customers["customer_id"].is_unique
    assert (ds.cohort["t0"] == G.events.ANCHOR_T0).all()
    assert not ds.events.empty


def test_label_is_the_oracle_bernoulli(ds, params):
    # oracle_score == hazard_prob(h_t0, q); the label was drawn as Bernoulli(oracle_score).
    haz = params["hazard"]
    expected = K.hazard_prob(
        ds.customers["h_t0"].to_numpy(),
        ds.customers["q"].to_numpy(),
        haz["alpha0"],
        haz["alpha_h"],
        haz["alpha_stat"],
    )
    assert np.allclose(ds.customers["oracle_score"].to_numpy(), expected)


def test_anchor_base_rate_matches_frozen_regime(ds):
    anchor = ds.labels[ds.labels["is_anchor"]]
    assert 0.43 <= float(anchor["y"].mean()) <= 0.53


def test_real_label_only_on_anchor(ds):
    assert ds.labels[ds.labels["is_anchor"]]["real_churned"].notna().all()
    assert ds.labels[~ds.labels["is_anchor"]]["real_churned"].isna().all()


def test_determinism(params):
    a = G.build_population_dataset(params, seed=5, n_synthetic=1000)
    b = G.build_population_dataset(params, seed=5, n_synthetic=1000)
    assert np.array_equal(a.labels["y"].to_numpy(), b.labels["y"].to_numpy())
    assert a.events.equals(b.events)


def test_events_independent_of_label_stream(params):
    # Invariant (i): the event log is byte-identical whether or not the label is later shuffled --
    # events are a function of health only, never y. (Shuffle is an analysis-time control.)
    ds = G.build_population_dataset(params, seed=7, n_synthetic=800)
    rng = np.random.default_rng(0)
    shuffled = ds.shuffle_labels(rng)
    assert ds.events.equals(shuffled.events)
    assert not np.array_equal(ds.labels["y"].to_numpy(), shuffled.labels["y"].to_numpy())


def test_null_stream_holds_label_fixed_but_changes_events(params):
    informative = G.build_population_dataset(params, seed=9, n_synthetic=800)
    null = G.build_population_dataset(params, seed=9, n_synthetic=800, null_stream=True)
    # Control validity (D6.8): same seed -> identical health + label, only events differ.
    assert np.array_equal(informative.labels["y"].to_numpy(), null.labels["y"].to_numpy())
    assert len(informative.events) != len(null.events) or not informative.events.equals(null.events)
