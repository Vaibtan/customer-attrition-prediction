"""Retrain-and-gate loop: a better retrain is promoted; a noise retrain keeps the incumbent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.lifecycle import retrain as R
from churn.lifecycle.adapters import ColumnSubsetModel
from churn.lifecycle.synthetic import linear_boundary_xy

N = 4000
N_ROUNDS = 200


@pytest.fixture(scope="module")
def holdout():
    rng = np.random.default_rng(11)
    x, y = linear_boundary_xy(rng, N)
    segments = pd.DataFrame(
        {
            "subscription_plan": rng.choice(["Free", "Pro", "Premium"], N),
            "region": rng.choice(["North", "South"], N),
        }
    )
    return rng, x, y, segments


def test_better_retrain_is_promoted(holdout):
    rng, x, y, segments = holdout
    weak = LogisticRegression().fit(x[:, :1], np.roll(y, 1))  # a poor champion (mislabeled fit)
    champion = ColumnSubsetModel(weak, cols=[0])
    strong = LogisticRegression(max_iter=1000).fit(x, y)  # uses all features, correct labels
    outcome = R.retrain_and_gate(
        lambda: ColumnSubsetModel(strong, cols=[0, 1, 2]),
        champion,
        x,
        y,
        segment_frame=segments,
        n_rounds=N_ROUNDS,
        seed=1,
    )
    assert outcome.promoted is True
    assert outcome.champion is outcome.challenger


def test_noise_retrain_keeps_incumbent(holdout):
    rng, x, y, segments = holdout
    strong = LogisticRegression(max_iter=1000).fit(x, y)
    champion = ColumnSubsetModel(strong, cols=[0, 1, 2])
    outcome = R.retrain_and_gate(
        lambda: ColumnSubsetModel(
            LogisticRegression(max_iter=1000).fit(x, y), cols=[0, 1, 2]
        ),  # same fit
        champion,
        x,
        y,
        segment_frame=segments,
        n_rounds=N_ROUNDS,
        seed=2,
    )
    assert outcome.promoted is False
    assert outcome.champion is champion
