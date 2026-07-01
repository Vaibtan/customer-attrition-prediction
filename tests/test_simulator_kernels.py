"""Golden-vector conformance tests for the frozen simulator kernels (D5.10).

Guard (2) of the lock's three (D1 code-fidelity): fixed (inputs, params) -> hand-computed
expected outputs, pinning that the code computes the *declared* functional forms. A logistic
link silently implemented as a step function -- which controls + leakage sentinels would NOT
catch -- fails here. Expected values are computed by hand in the comments so the fixtures are
auditable without running the code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from churn.simulator import kernels as K

# --- sigmoid ---------------------------------------------------------------------------------


def test_sigmoid_reference_points():
    assert K.sigmoid(np.array(0.0)) == pytest.approx(0.5)
    assert K.sigmoid(np.array(1.0)) == pytest.approx(0.7310585786300049)
    assert K.sigmoid(np.array(-1.0)) == pytest.approx(0.2689414213699951)


def test_sigmoid_is_numerically_stable_at_extremes():
    # No overflow warning / inf; saturates to the asymptotes.
    assert K.sigmoid(np.array(1000.0)) == pytest.approx(1.0)
    assert K.sigmoid(np.array(-1000.0)) == pytest.approx(0.0)


def test_sigmoid_vectorizes():
    out = K.sigmoid(np.array([-1.0, 0.0, 1.0]))
    assert out == pytest.approx([0.2689414213699951, 0.5, 0.7310585786300049])


# --- quality index q(x) (D5.4) ---------------------------------------------------------------

WEIGHTS = {"plan": 1.0, "age": 0.25, "spend": 0.20, "aov": 0.15}


def test_quality_index_raw_is_the_declared_weighted_sum():
    # 1.0*2 + 0.25*1 + 0.20*(-1) + 0.15*0 = 2 + 0.25 - 0.20 = 2.05
    out = K.quality_index_raw(plan_score=2, z_age=1.0, z_spend=-1.0, z_aov=0.0, weights=WEIGHTS)
    assert out == pytest.approx(2.05)


def test_standardize_matches_z_score():
    # (10 - 5) / 2.5 = 2.0
    assert K.standardize(10.0, mean=5.0, std=2.5) == pytest.approx(2.0)


# --- latent health h (D5.1) ------------------------------------------------------------------


def test_latent_baseline_is_s_mu_times_q():
    assert K.latent_baseline(q=1.5, s_mu=0.4) == pytest.approx(0.6)


def test_latent_step_is_the_declared_ar1_update():
    # h + kappa*(mu - h) + sigma*eps
    # 1 + 0.2*(0 - 1) + 0.6*0 = 0.8
    assert K.latent_step(h=1.0, mu=0.0, kappa=0.2, eps=0.0, sigma=0.6) == pytest.approx(0.8)
    # 0 + 0.25*(2 - 0) + 0.5*1 = 0.5 + 0.5 = 1.0
    assert K.latent_step(h=0.0, mu=2.0, kappa=0.25, eps=1.0, sigma=0.5) == pytest.approx(1.0)


# --- hazard / label (D5.2) -------------------------------------------------------------------


def test_hazard_prob_is_logistic_in_h_and_q():
    # logit = 0 + (-1)*0 + (-1)*0 = 0 -> 0.5
    assert K.hazard_prob(h=0.0, q=0.0, alpha0=0.0, alpha_h=-1.0, alpha_stat=-1.0) == pytest.approx(
        0.5
    )
    # logit = 0.5 + (-1)*1 + (-0.5)*1 = -1.0 -> sigmoid(-1)
    assert K.hazard_prob(h=1.0, q=1.0, alpha0=0.5, alpha_h=-1.0, alpha_stat=-0.5) == pytest.approx(
        0.2689414213699951
    )
    # logit = 0.2 + (-0.8)*(-1) + (-0.4)*0.5 = 0.2 + 0.8 - 0.2 = 0.8 -> sigmoid(0.8)
    assert K.hazard_prob(h=-1.0, q=0.5, alpha0=0.2, alpha_h=-0.8, alpha_stat=-0.4) == pytest.approx(
        0.6899744811276125
    )


def test_hazard_prob_decreases_in_health_and_quality():
    # Frozen alpha_h, alpha_stat < 0: healthier / higher-quality -> lower churn.
    base = K.hazard_prob(0.0, 0.0, 0.0, -1.0, -1.0)
    assert K.hazard_prob(1.0, 0.0, 0.0, -1.0, -1.0) < base
    assert K.hazard_prob(0.0, 1.0, 0.0, -1.0, -1.0) < base


# --- event-emission links (D5.5) -------------------------------------------------------------


def test_exp_link_is_log_linear_rate():
    # exp(0.5 + 0.25*2) = exp(1.0)
    assert K.exp_link(h=2.0, a=0.5, b=0.25) == pytest.approx(math.e)
    # exp(ln(4) + 0.4*0) = 4.0  (generic exp-link check: exp_link(0, ln(r), b) == r)
    assert K.exp_link(h=0.0, a=math.log(4.0), b=0.4) == pytest.approx(4.0)


def test_logistic_link_is_logit_probability():
    # sigmoid(0 + (-0.5)*1) = sigmoid(-0.5)
    assert K.logistic_link(h=1.0, a=0.0, b=-0.5) == pytest.approx(0.3775406687981454)
    # sigmoid(ln(0.05/0.95)) = 0.05  (generic logit check: logistic_link(0, ln(p/(1-p)), b) == p)
    assert K.logistic_link(h=0.0, a=math.log(0.05 / 0.95), b=-0.7) == pytest.approx(0.05)


def test_linear_link_is_identity_mean():
    # 1 + 0.5*3 = 2.5
    assert K.linear_link(h=3.0, a=1.0, b=0.5) == pytest.approx(2.5)
    # 8 + 1.0*0 = 8.0  (session-depth baseline at h=0)
    assert K.linear_link(h=0.0, a=8.0, b=1.0) == pytest.approx(8.0)


def test_links_vectorize_over_health_paths():
    h = np.array([-1.0, 0.0, 1.0])
    assert K.exp_link(h, a=0.0, b=1.0) == pytest.approx([math.exp(-1), 1.0, math.e])
    assert K.linear_link(h, a=0.0, b=2.0) == pytest.approx([-2.0, 0.0, 2.0])


def test_event_link_directions_match_the_frozen_sign_convention():
    # Healthier (h up) -> more logins, deeper sessions (b > 0).
    assert K.exp_link(1.0, a=0.0, b=0.4) > K.exp_link(-1.0, a=0.0, b=0.4)
    assert K.linear_link(1.0, a=0.0, b=1.0) > K.linear_link(-1.0, a=0.0, b=1.0)
    # Unhealthier (h down) -> more payment failures, more support tickets (b < 0).
    assert K.logistic_link(-1.0, a=0.0, b=-0.7) > K.logistic_link(1.0, a=0.0, b=-0.7)
    assert K.exp_link(-1.0, a=0.0, b=-0.4) > K.exp_link(1.0, a=0.0, b=-0.4)
