"""Promotion gate (ENHANCEMENT_PLAN.md Sec 4.8) -- statistical teeth, no false precision.

Pins the five behaviours that make promotion trustworthy: a better-by-noise challenger is NOT
promoted; a genuine improvement is; the incumbent wins exact ties; a calibration (Brier) regression
blocks; and a per-segment degradation blocks -- even when the overall metric improved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.lifecycle import promotion as PROMO

N = 4000
N_ROUNDS = 200  # small bootstrap keeps the test fast; the gate uses config.BOOTSTRAP_ROUNDS


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, N)
    plan = rng.choice(["Free", "Pro", "Premium"], size=N, p=[0.6, 0.3, 0.1])
    region = rng.choice(["North", "South", "East", "West"], size=N)
    segments = pd.DataFrame({"subscription_plan": plan, "region": region})
    return rng, y, segments


def _signal(y, rng, strength, noise):
    return np.clip(0.5 + strength * (y - 0.5) + rng.normal(0, noise, len(y)), 0.001, 0.999)


def test_better_by_noise_is_not_promoted(data):
    rng, y, segments = data
    champion = _signal(y, rng, strength=0.7, noise=0.15)  # already strong
    challenger = np.clip(champion + rng.normal(0, 0.02, N), 0.001, 0.999)  # same signal + jitter
    decision = PROMO.evaluate_promotion(
        y, champion, challenger, segment_frame=segments, n_rounds=N_ROUNDS, seed=1
    )
    assert decision.primary_passed is False
    assert decision.promote is False


def test_genuine_improvement_is_promoted(data):
    rng, y, segments = data
    champion = _signal(y, rng, strength=0.25, noise=0.30)  # weak-ish
    challenger = _signal(y, rng, strength=0.75, noise=0.12)  # clearly better + calibrated
    decision = PROMO.evaluate_promotion(
        y, champion, challenger, segment_frame=segments, n_rounds=N_ROUNDS, seed=2
    )
    assert decision.delta_roc["diff_lo"] > 0
    assert decision.primary_passed is True
    assert all(g.passed for g in decision.guardrails)
    assert decision.promote is True


def test_incumbent_wins_exact_tie(data):
    rng, y, segments = data
    champion = _signal(y, rng, strength=0.6, noise=0.15)
    decision = PROMO.evaluate_promotion(
        y, champion, champion.copy(), segment_frame=segments, n_rounds=N_ROUNDS, seed=3
    )
    assert decision.promote is False


def test_calibration_regression_blocks_promotion(data):
    rng, y, segments = data
    champion = _signal(y, rng, strength=0.35, noise=0.25)  # weaker but calibrated
    strong = _signal(y, rng, strength=0.8, noise=0.10)  # better ranking...
    # ...but compressed toward 0 (ranking kept -> high AUC; under-confident -> bad Brier).
    challenger = np.clip(strong * 0.3, 0.001, 0.999)
    decision = PROMO.evaluate_promotion(
        y, champion, challenger, segment_frame=segments, n_rounds=N_ROUNDS, seed=4
    )
    brier_guard = next(g for g in decision.guardrails if "brier" in g.name)
    assert brier_guard.passed is False
    assert decision.promote is False


def test_segment_degradation_blocks_promotion(data):
    rng, y, segments = data
    champion = _signal(y, rng, strength=0.45, noise=0.22)
    challenger = _signal(y, rng, strength=0.8, noise=0.10)  # better overall...
    # ...but random (useless) on the Premium segment -> that segment's AUC collapses.
    premium = (segments["subscription_plan"] == "Premium").to_numpy()
    challenger = challenger.copy()
    challenger[premium] = rng.uniform(0.001, 0.999, premium.sum())
    decision = PROMO.evaluate_promotion(
        y, champion, challenger, segment_frame=segments, n_rounds=N_ROUNDS, seed=5
    )
    seg_guard = next(g for g in decision.guardrails if "segment" in g.name)
    assert seg_guard.passed is False
    assert decision.promote is False
