"""Experiment-layer tests -- the three named baselines + positive/negative controls (§4.4/§4.5).

Validates the instrument on an exploratory seed: the three distinctly-named baselines are emitted
and ordered sensibly, the positive control's paired ROC/PR lower bounds are positive, and BOTH
negative controls (label-shuffle, null-stream) show no lift (necessary sanity checks, §4.7).
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.featurestore import offline as OFF
from churn.instrument import experiment as EXP
from churn.simulator import generate as G
from churn.simulator import params as P


@pytest.fixture(scope="module")
def params() -> dict:
    return P.load_params()


@pytest.fixture(scope="module")
def anchor(params):
    ds = G.build_population_dataset(params, seed=4242, n_synthetic=0)
    pit = OFF.compute_pit_features(ds.events, ds.cohort)
    return ds, pit


def test_paired_diff_ci_customer_level_matches_row_level_for_singletons():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    a = rng.random(400)
    b = rng.random(400)
    groups = np.arange(400)  # one row per customer -> customer resample == row resample
    from sklearn.metrics import roc_auc_score

    ci_g = EXP.paired_diff_ci(y, a, b, roc_auc_score, groups=groups, n_rounds=200, seed=1)
    ci_r = EXP.paired_diff_ci(y, a, b, roc_auc_score, groups=None, n_rounds=200, seed=1)
    assert ci_g["diff_lo"] == pytest.approx(ci_r["diff_lo"], abs=1e-9)


def test_three_named_baselines_are_emitted_and_ordered(anchor):
    ds, pit = anchor
    rep = EXP.compute_baselines(ds, pit, seed=7)
    for key in (
        "real_static_reference_auc",
        "synthetic_static_auc",
        "synthetic_static_plus_event_auc",
    ):
        assert 0.5 <= rep[key] <= 0.95, (key, rep[key])
    # Event features recover latent-health signal static cannot see (the positive-control headline).
    assert rep["synthetic_static_plus_event_auc"] > rep["synthetic_static_auc"] + 0.03
    # The real-label reference sits in the weak-signal anchor band and is reported on its own.
    assert 0.55 <= rep["real_static_reference_auc"] <= 0.68


def test_positive_control_lower_bounds_are_positive(anchor):
    ds, pit = anchor
    ctrl = EXP.positive_control(ds, pit, seed=7, bootstrap_seed=123, n_rounds=400)
    assert ctrl["delta_roc"]["diff_lo"] > 0
    assert ctrl["delta_pr"]["diff_lo"] > 0


def test_label_shuffle_control_shows_no_lift(anchor):
    ds, pit = anchor
    shuffled = ds.shuffle_labels(np.random.default_rng(0))
    rep = EXP.compute_baselines(shuffled, pit, seed=7)
    # A leak-free pipeline cannot predict a permuted label: no lift, no signal above chance
    # (OOF AUC may sit slightly below 0.5 from fitting fold noise -- the guard is "not above").
    assert rep["synthetic_static_plus_event_auc"] < 0.55
    assert rep["synthetic_static_auc"] < 0.55


def test_null_stream_control_shows_no_event_lift(params):
    null = G.build_population_dataset(params, seed=4242, n_synthetic=0, null_stream=True)
    pit = OFF.compute_pit_features(null.events, null.cohort)
    rep = EXP.compute_baselines(null, pit, seed=7)
    # Static AUC preserved; event lift ~ 0 (events carry no health signal when slopes are zeroed).
    assert rep["synthetic_static_auc"] > 0.58
    assert abs(rep["synthetic_static_plus_event_auc"] - rep["synthetic_static_auc"]) < 0.03
