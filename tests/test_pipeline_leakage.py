"""Leakage guard: fitted preprocessing statistics depend only on the fit data.

One assertion per fitted statistic in the numeric branch (REV-24): a leak regression in the
winsor caps or the imputer medians must fail this file on its own, not hide behind the scaler
mean staying coincidentally different.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from churn import config
from churn.data import split_features_target
from churn.pipeline import build_pipeline


def numeric_steps(fitted_pipe):
    return fitted_pipe.named_steps["pre"].named_transformers_["num"].named_steps


@pytest.fixture(scope="module")
def fitted_pair(raw_full):
    """One pipeline fitted on the train split, one on the full frame (the leak stand-in)."""
    X, y, ids = split_features_target(raw_full)
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )
    pipe_train = build_pipeline(LogisticRegression(max_iter=500)).fit(X_train, y_train)
    pipe_full = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    return pipe_train, pipe_full


def test_scaler_stats_are_subset_specific(fitted_pair):
    pipe_train, pipe_full = fitted_pair
    mean_train = numeric_steps(pipe_train)["scale"].mean_
    mean_full = numeric_steps(pipe_full)["scale"].mean_
    assert not np.allclose(mean_train, mean_full)


def test_winsor_caps_are_subset_specific(fitted_pair):
    pipe_train, pipe_full = fitted_pair
    win_train = numeric_steps(pipe_train)["winsorize"]
    win_full = numeric_steps(pipe_full)["winsorize"]
    caps_train = np.concatenate([win_train.lower_, win_train.upper_])
    caps_full = np.concatenate([win_full.lower_, win_full.upper_])
    assert not np.allclose(caps_train, caps_full)


def test_imputer_medians_are_subset_specific(fitted_pair):
    pipe_train, pipe_full = fitted_pair
    med_train = numeric_steps(pipe_train)["impute"].statistics_
    med_full = numeric_steps(pipe_full)["impute"].statistics_
    assert not np.allclose(med_train, med_full)


def test_preprocessing_lives_inside_estimator():
    pipe = build_pipeline(LogisticRegression())
    steps = dict(pipe.named_steps)
    assert {"repair", "features", "pre", "clf"} <= set(steps)
