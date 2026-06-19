"""Interpretability: odds ratios + per-customer reason codes from the linear model."""

from __future__ import annotations

import numpy as np
import pytest
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


def test_contributions_columns_are_named_features(sample):
    pipe, X = fit_pipeline(sample)
    contrib = interpret.contributions(pipe, X.head(8))
    assert list(contrib.columns) == interpret.feature_names(pipe)
    assert len(contrib) == 8


def test_contributions_reconstruct_the_log_odds(sample):
    """Sum of signed contributions (+ intercept) must equal the model's log-odds.

    This only holds when every transformed column is paired with *its own*
    coefficient, so it catches z<->coef order misalignment, not just length drift.
    """
    pipe, X = fit_pipeline(sample)
    rows = X.head(25)
    contrib = interpret.contributions(pipe, rows)
    intercept = float(pipe.named_steps["clf"].intercept_[0])
    logits = contrib.to_numpy().sum(axis=1) + intercept
    proba = 1.0 / (1.0 + np.exp(-logits))
    assert np.allclose(proba, pipe.predict_proba(rows)[:, 1], atol=1e-8)


def test_signed_coefficient_attaches_to_correct_feature(raw_full):
    """days_since_last_login is the dominant churn signal (+r); its *label* must
    carry a positive coefficient -> guards the name<->coef pairing semantically."""
    pipe, _ = fit_pipeline(raw_full)
    coef = interpret.signed_coefficients(pipe)
    assert coef["days_since_last_login"] > 0


def test_contributions_rejects_coef_width_mismatch(sample):
    pipe, X = fit_pipeline(sample)
    clf = pipe.named_steps["clf"]
    clf.coef_ = clf.coef_[:, :-1]
    with pytest.raises(ValueError, match="align"):
        interpret.contributions(pipe, X.head(3))
