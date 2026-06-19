"""Interpretability: odds ratios + per-customer reason codes from the linear model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from churn import interpret
from churn.data import split_features_target
from churn.pipeline import build_pipeline


def _fit(sample):
    X, y, _ = split_features_target(sample)
    return build_pipeline(LogisticRegression(max_iter=500)).fit(X, y), X


def test_odds_ratios_columns_and_ordering(sample):
    pipe, _ = _fit(sample)
    df = interpret.odds_ratios(pipe)
    assert {"feature", "coef", "odds_ratio"} <= set(df.columns)
    assert len(df) > 0
    # Sorted by absolute coefficient, descending.
    abs_coef = df["coef"].abs().to_numpy()
    assert np.all(np.diff(abs_coef) <= 1e-9)
    # odds_ratio == exp(coef).
    assert np.allclose(df["odds_ratio"], np.exp(df["coef"]))


def test_reason_codes_shape(sample):
    pipe, X = _fit(sample)
    codes = interpret.reason_codes_for_frame(pipe, X.head(5), k=3)
    assert len(codes) == 5
    # k=3 contributions → 2 separators each.
    assert all(c.count(";") == 2 for c in codes)
    # Prefixes are stripped for readability.
    assert all("num__" not in c and "cat__" not in c for c in codes)
