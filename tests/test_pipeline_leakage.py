"""Leakage guard: fitted preprocessing statistics depend only on the fit data."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from churn import config
from churn.data import split_features_target
from churn.pipeline import build_pipeline


def scaler_step(fitted_pipe):
    return fitted_pipe.named_steps["pre"].named_transformers_["num"].named_steps["scale"]


def test_scaler_stats_are_subset_specific(raw_full):
    X, y, ids = split_features_target(raw_full)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )

    pipe_train = build_pipeline(LogisticRegression(max_iter=500)).fit(X_train, y_train)
    pipe_full = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)

    mean_train = scaler_step(pipe_train).mean_
    mean_full = scaler_step(pipe_full).mean_

    assert not np.allclose(mean_train, mean_full)


def test_preprocessing_lives_inside_estimator():
    pipe = build_pipeline(LogisticRegression())
    steps = dict(pipe.named_steps)
    assert {"repair", "features", "pre", "clf"} <= set(steps)
