"""Interpretability: odds ratios + per-customer reason codes from the linear model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from churn import interpret
from churn.data import split_features_target
from churn.pipeline import build_pipeline


def fit_pipeline(sample):
    X, y, ids = split_features_target(sample)
    return build_pipeline(LogisticRegression(max_iter=500)).fit(X, y), X


def test_odds_ratios_columns_and_ordering(sample):
    pipe, X = fit_pipeline(sample)
    df = interpret.odds_ratios(pipe)
    assert {"feature", "coef", "odds_ratio"} <= set(df.columns)
    assert len(df) > 0
    abs_coef = df["coef"].abs().to_numpy()
    assert np.all(np.diff(abs_coef) <= 1e-9)
    assert np.allclose(df["odds_ratio"], np.exp(df["coef"]))


def test_reason_codes_shape(sample):
    pipe, X = fit_pipeline(sample)
    codes = interpret.reason_codes_for_frame(pipe, X.head(5), k=3)
    assert len(codes) == 5
    assert all(c.count(";") == 2 for c in codes)
    assert all("num__" not in c and "cat__" not in c for c in codes)
